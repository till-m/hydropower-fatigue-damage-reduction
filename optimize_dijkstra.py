import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rainflow
from icecream import ic
import einops
import pytorch_lightning as pl
import heapq
import pickle
import itertools
from multiprocess import Pool

import logging # disable the log from seed_everything
logging.getLogger("lightning_fabric.utilities.seed").propagate = False

import logging # disable the log from seed_everything
logging.getLogger("lightning_fabric.utilities.seed").propagate = False

from config import CONFIG

artifact_dir = './artifacts/model-qid4ycyv:v0'

stress_model = CONFIG["model"](**CONFIG["model_kwargs"])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    stress_model.load_state_dict(torch.load(artifact_dir + '/model.ckpt', map_location=torch.device(device))['state_dict'])
except FileNotFoundError:
    import wandb
    run = wandb.init()
    model_name = artifact_dir.split('/')[-1]
    artifact = run.use_artifact('sdsc-paired-hydro/paired-hydro-transient-selection/' + model_name, type='model')
    artifact_dir = artifact.download()
    print(artifact_dir)
    run.finish()
    run = wandb.Api().run(run.entity + '/' + run.project + '/' + run.id)
    run.delete()
    stress_model.load_state_dict(torch.load(artifact_dir + '/model.ckpt', map_location=torch.device(device))['state_dict'])

CONTROL_COLUMNS = CONFIG['data_module_kwargs']['setup_kwargs']['control_columns']
TARGET_COLUMNS = CONFIG['data_module_kwargs']['setup_kwargs']['target_columns']

params = CONTROL_COLUMNS
var = pd.Series((75034.15757525747, 62.90646327009763, 0.6769653823074555), index = params)
var2 = pd.Series({
    "Manner_1": 446.27317513825193,
    "Manner_4": 544.8019217354232,
})

start_canonical = pd.Series((0, 0, 2.543133), index=params)
end_canonical = pd.Series((735.840516, 16.955307963782552, 2.543133), index=params)

start = start_canonical / np.sqrt(var)
end =  end_canonical / np.sqrt(var)

t_start = 0
t_end = 100000


def print_statusline(msg: str): # https://stackoverflow.com/a/43952192
    last_msg_length = len(getattr(print_statusline, 'last_msg', ''))
    print(' ' * last_msg_length, end='\r')
    print(msg, end='\r')
    sys.stdout.flush()  # Some say they needed this, I didn't.
    setattr(print_statusline, 'last_msg', msg)

def damage_index(signal):
    # Parameters for damage computation

    # Slope of the curve
    m=8

    # number of cycle for knee point
    Nk=2e6

    # Ultimate stress
    Rm = 865

    # [MPa] Endurance Limit
    sigaf=Rm/2  

    # Computation : Rainflow counting algorithm
    #ext, exttime = turningpoints(signal, dt)
    rf = rainflow.extract_cycles(signal) # range, mean, count, i_start, i_end
    rf = np.array([i for i in rf]).T
    rf[1] = 0.5* rf[1] # Matlab function is half as big for some reason

    siga = (rf[0]*Rm)/(Rm-rf[1])  #rf[2] # Mean stress correction (Goodman)
    rf[0] = siga
    damage_full = (rf[2]/Nk) * (siga/sigaf)**m
    damage = np.sum(damage_full)

    return damage#, rf, damage_full

def instant_cost(trajectories):
    cost = np.sum(trajectories.reshape((trajectories.shape[0], -1))**2, axis=1)
    return cost

def parallel_damage_cost(trajectories: torch.Tensor):
    trajectories = trajectories.detach().numpy()
    trajectories = trajectories[..., 0] # Manner1 only
    p = Pool(len(trajectories))
    result = p.map_async(damage_index, trajectories).get()
    return np.array(result) ** (1/8.)

def damage_cost(trajectory):
    return [damage_index(signal_[..., 0])**(1/8.) for signal_ in trajectory]

class DijkstraStartupOptimizer():
    def __init__(self, start, end, grid_size=256, interpol_length=None, mode='instant', diagonal_neighbors=False, max_hop_distance=1, allow_decrease=True, forbidden_region=True) -> None:
        self.start = start
        self.end = end
        self.interpol_length = interpol_length if interpol_length is not None else 83_200 // grid_size
        self.grid_size = grid_size
        self.mode = mode
        self.diagonal_neighbors = diagonal_neighbors
        self.max_hop_distance = max_hop_distance
        self.allow_decrease = allow_decrease

        turbine_speed = np.linspace(start[0], end[0], grid_size) #
        gvo = np.linspace(start[1], end[1], grid_size)
        self.grid = np.array([[x_i, y_j] for y_j in gvo for x_i in turbine_speed]).reshape((grid_size, grid_size, 2))
        self.forbidden_region = forbidden_region
        
    
    def repeat_last_point(self, u, n_times=4096):
        assert len(u.shape) == 3
        last_point = einops.rearrange(u[:, -1], 'b c -> b 1 c')
        last_point = einops.repeat(last_point,  'b t1 c -> b (t t1) c', t=n_times)
        return torch.cat((u, last_point), dim=1)

    def parallel_construct_trajectories(self, paths):
        p = Pool(len(paths))
        return p.map_async(self.construct_trajectory, paths).get()

    def construct_trajectory(self, nodes, n_intial_segments=None):
        if n_intial_segments is None:
            n_intial_segments = 2048 // self.interpol_length
        prev = tuple(nodes[0])

        # repeat initial node
        trajectory = [*einops.repeat(self.grid[tuple(nodes[0])], 'c -> reps t c', t=self.interpol_length, reps=n_intial_segments)]

        for node in nodes[1:]:
            node = tuple(node)
            trajectory.append(self.interpolate_trajectory(self.grid[prev], self.grid[node]))
            prev = node
        return einops.rearrange(trajectory, 'seg t c -> 1 (seg t) c')
    
    def interpolate_trajectory(self, node1, node2) -> np.ndarray:
        return np.array([np.linspace(node1[i], node2[i], num=self.interpol_length) for i, _ in enumerate(node1)]).T

    def calculate_stress(self, u, return_offset=False, seed_everything=True) -> torch.Tensor:
        head = end[-1] * torch.ones(u.shape[:-1] + (1,))
        u = torch.cat((torch.from_numpy(u), head), dim=-1)
        match len(u.shape):
            case 2:
                u = einops.rearrange(u, 't c -> 1 t c')
            case 3:
                pass
            case _:
                raise ValueError

        u = self.repeat_last_point(u)

        if seed_everything:
            pl.seed_everything(42)
        stress, offset = stress_model(u.to(stress_model.device, dtype=stress_model.dtype), return_offset=True)
        stress = torch.einsum('i,...i-> ...i', torch.sqrt(torch.from_numpy(var2.to_numpy())), stress)
        offset = torch.einsum('i,...i-> ...i', torch.sqrt(torch.from_numpy(var2.to_numpy())), offset)

        if return_offset:
            return stress[:, :-1024], offset[:, :-1024]
        return stress[:, :-1024]

    def get_path(self, goal, parent) -> np.ndarray:
        nodes = []
        node = goal
        while node is not None:
            if node in nodes:
                ic(node)
                ic(nodes)
                raise RuntimeError
            nodes.append(node)
            node = parent[node]
        nodes.reverse()
        return nodes

    def print_state(self, path, node, goal, heapsize, visited):
        print(f"Path length: {len(path)}\t\tCurrent node: {node}\t\tGoal node: {goal}\t\tHeapsize: {heapsize}\t\t Visited: {visited}", end='\r')

    def optimize(self, interactive=True):
        with torch.no_grad():
            if interactive:
                plt.ion()
                fig, axs = plt.subplots(2 + 2, 1, figsize=(6, 12))
                fig.tight_layout()
                for ax in axs:
                    ax.grid(True)
                    axs[0].set_title(CONTROL_COLUMNS[0])
                    axs[0].plot([0])
                    axs[1].set_title(CONTROL_COLUMNS[1])
                    axs[1].plot([0])
                    axs[2].set_title(TARGET_COLUMNS[0])
                    axs[2].plot([0])
                    axs[3].set_title(TARGET_COLUMNS[1])
                    axs[3].plot([0])

            dims = self.grid.shape[:-1]
            n_dim = len(dims)

            
            start_node = tuple(0 for _ in range(n_dim))
            
            goal = tuple(i-1 for i in dims)

            # create a priority queue to hold the unvisited cells
            heap = [(0, start_node)]
            # create a dictionary to hold the cost of visiting each cell
            cost = {start_node: 0}
            # create a dictionary to hold the parent cell of each cell
            parent = {start_node: None}
            # create a set to hold the visited cells
            visited = set()


            if isinstance(self.max_hop_distance, int):
                max_hop_distance = [self.max_hop_distance] * n_dim
            else:
                max_hop_distance = self.max_hop_distance

            if self.diagonal_neighbors:
                if self.allow_decrease:
                    sample_from = [1, -1]
                else:
                    sample_from = [1]

                sample_sets = [
                    set([i * q for i in sample_from for q in range(max_hop_distance[d] + 1)])
                    for d in range(n_dim)
                ]
                sample_from = list(itertools.product(*sample_sets))
                sample_from = [np.array(item) for item in sample_from if any(item)]

                base_neighbors = np.array(sample_from).astype(int)
            else:
                sample_from = []
                for d in range(n_dim):
                    for i in range(1, max_hop_distance[d] + 1):
                        step = np.zeros(n_dim)
                        step[d] = i
                        sample_from.append(step)
                        if self.allow_decrease:
                            sample_from.append(-step)

                base_neighbors = np.vstack(sample_from).astype(int)
            
            ic(base_neighbors)

            try:
                with open('ckpt.pickle', 'rb') as handle:
                    parent, cost, heap, visited = pickle.load(handle)
                print("Continuing from Checkpoint...")
            except FileNotFoundError:
                pass

            while heap:
                with open('ckpt.pickle', 'wb') as handle:
                    pickle.dump((parent, cost, heap, visited), handle, protocol=pickle.HIGHEST_PROTOCOL)
                # get the cell with the lowest cost
                current_cost, current_node = heapq.heappop(heap)
                # if the cell has already been visited, skip it
                if current_node in visited:
                    assert current_cost > cost[current_node]
                    continue
                if current_node == goal:
                    ic()
                    break

                path = self.get_path(current_node, parent)
                self.print_state(path, current_node, goal, len(heap), len(visited))
                if interactive:
                    trajectory = self.construct_trajectory(path)
                    stress = self.calculate_stress(trajectory).squeeze()
                    axs[0].lines[0].set_data(np.arange(trajectory.shape[-2]), trajectory.squeeze(0)[:, 0] * np.sqrt(var[0]))
                    axs[1].lines[0].set_data(np.arange(trajectory.shape[-2]), trajectory.squeeze(0)[:, 1]* np.sqrt(var[1]))
                    axs[2].lines[0].set_data(np.arange(stress.shape[-2]), stress[:, 0].detach())
                    axs[3].lines[0].set_data(np.arange(stress.shape[-2]), stress[:, 1].detach())
                    for ax in axs:
                        ax.relim()
                        ax.autoscale_view()
                    plt.pause(0.0001)

                # mark the cell as visited
                visited.add(current_node)
                neighbors = base_neighbors + current_node

                # Eliminate oob (idx<0)
                neighbors = neighbors[(neighbors>=0).all(axis=1)]
                # Eliminate oob (idx>dim)
                neighbors = neighbors[(neighbors<dims).all(axis=1)]
                # Eliminate blocked squares
                neighbors = [tuple(n) for n in neighbors if n is not parent[current_node]]
                if self.forbidden_region:
                    neighbors = [n for n in neighbors if (n[0]< 75 or n[1] > 123)]

                trajectories = einops.rearrange(self.parallel_construct_trajectories([path + [node] for node in neighbors]), 'b1 b2 t c -> (b1 b2) t c', b2=1)
                # calculate the cost of visiting the neighbor
                if self.mode=='damage':
                    new_costs =  parallel_damage_cost(self.calculate_stress(trajectories))
                elif self.mode=='instant':
                    new_costs = instant_cost(trajectories)
                else:
                    raise ValueError
                # if the neighbor has not been visited or the new cost is less than the current cost
                for neighbor, new_cost in zip(neighbors, new_costs):
                    if isinstance(new_cost, torch.Tensor):
                        new_cost = new_cost.item()
                    new_cost = max([new_cost, current_cost])
                    if neighbor not in cost or new_cost < cost[neighbor]:
                        # update the cost and parent cell of the neighbor
                        cost[neighbor] = new_cost
                        parent[neighbor] = current_node
                        heapq.heappush(heap, (new_cost, neighbor))
            ic()

            final_path = self.get_path(goal, parent)
            final_trajectory = self.construct_trajectory(final_path)
            final_stress = self.calculate_stress(final_trajectory).squeeze()
            return parent, cost, final_trajectory, final_stress

if __name__ == '__main__':
    optimizer = DijkstraStartupOptimizer(start, end, grid_size=256, interpol_length=128, diagonal_neighbors=True, mode='damage', allow_decrease=False, max_hop_distance=(2, 6), forbidden_region=True)

    res = optimizer.optimize(interactive=False)

    with open('final.pickle', 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("FINISHED")
