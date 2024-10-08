import rainflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

import cyl1tf

from config import CONFIG
from pathlib import Path
import wandb

CONTROL_COLUMNS = CONFIG["data_module_kwargs"]["setup_kwargs"]["control_columns"]
TARGET_COLUMNS = CONFIG["data_module_kwargs"]["setup_kwargs"]["target_columns"]

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

    return damage, rf, damage_full

def find_type(in_str):
    file = in_str.split('/')[-1]
    type_ = file.split('_')[0]
    return type_

def convert_numeric_string(num):
    if num[-1].isnumeric():
        if num[0].isnumeric():
            return int(num[0]) + int(num[2]) / 10
        else:
            return int(num[-1])
    else:
        return 1.0

def find_fsim(in_str):
    folder = in_str.split('/')[-2]
    num = folder[-3:]
    return convert_numeric_string(num)

def find_type_and_fsim(in_str):
    # String magic to find the type (BEP, 2Slopes, Linear, Classic)
    # and Froude similarity from the path of the file
    if '/' in in_str:
        return find_type(in_str), find_fsim(in_str)
    else:
        return in_str.split('_')[1], convert_numeric_string(in_str[-3:])

def create_damage_plot(model, datamodule):
    run_id = wandb.run.id
    dir_ = Path(f'./paired-hydro-transient-selection/{run_id}/checkpoints')
    files = [dir_ / x.name for x in dir_.iterdir() if x.is_file()]
    assert len(files) == 1

    model.load_state_dict(torch.load(files[0], map_location=model.device)['state_dict'])

    var = datamodule.var
    columns = (
        ['type_', 'fsim', 'title']
        + TARGET_COLUMNS
        + [col + "_l1" for col in TARGET_COLUMNS]
        + [col + "_pred" for col in TARGET_COLUMNS]
        + [col + "_pred_l1" for col in TARGET_COLUMNS]
    )

    damage_ = pd.DataFrame(columns=columns, index=np.arange(len(datamodule.train_dset) + len(datamodule.train_dset)))
    damage_.head()

    sets = [datamodule.train_dset, datamodule.val_dset]
    # Manual selection
    global_idx = 0
    for i, set_ in enumerate(sets):
        n_runs = len(set_)
        for idx in tqdm(range(n_runs)):
            run = set_.get_item_for_plot(idx)
            (title, control_columns, target_columns), (X_, y_) = run
            X_ = X_.reshape((1,) + X_.shape)
            y_pred = model(X_).detach().numpy().squeeze() *  np.sqrt(var[TARGET_COLUMNS]).to_numpy()
            y_ = y_.detach().numpy().squeeze() *  np.sqrt(var[TARGET_COLUMNS]).to_numpy()

            #y_pred = y_pred.copy(order='C') 

            damage  = []
            damage_l1 = []
            damage_pred = []
            damage_pred_l1 = []
            for j, c in enumerate(target_columns):
                dmg_, _, _ = damage_index(y_[..., j])
                damage.append(dmg_)
                dmg_l1, _, _ = damage_index(cyl1tf.calc_fit(y_[..., j], rel_scale=0.0001))
                damage_l1.append(dmg_l1)
                dmg_pred, _, _ = damage_index(y_pred[..., j])
                damage_pred.append(dmg_pred)
                # .copy Fixes ValueError: ndarray is not C-contiguous
                dmg_pred_l1, _, _ = damage_index(cyl1tf.calc_fit(y_pred[..., j].copy(order='C'), rel_scale=0.0001))
                damage_pred_l1.append(dmg_pred_l1)
            res = (*find_type_and_fsim(title), title, *damage, *damage_l1, *damage_pred, *damage_pred_l1)
            damage_.iloc[global_idx] = res
            global_idx += 1

    colours = ['red', 'green', 'orange', 'purple']

    plots = {}
    for type_ in ['BEP', 'Linear', 'Classic', '2Slopes']:
        fig, axs = plt.subplots(1, len(TARGET_COLUMNS), figsize=(12,6), sharey=True)
        fig.suptitle(type_)
        for i in range(len(TARGET_COLUMNS)):
            col = TARGET_COLUMNS[i]
            subdf = damage_[damage_['type_'] == type_]
            axs[i].set_title(col)
            axs[i].set_xlabel('Froude Similarity')
            axs[i].set_ylabel('Damage')
            axs[i].semilogy(subdf['fsim'], subdf[col], marker='.', linestyle='none', label='true', color=colours[0], alpha=0.3)
            axs[i].semilogy(subdf['fsim'], subdf[col + "_pred"], marker='x', linestyle='none', label="predicted", color=colours[1], alpha=0.3)
            axs[i].semilogy(subdf['fsim'], subdf[col + "_l1"], marker='+', linestyle='none', label="L1", color=colours[2], alpha=0.3)
            axs[i].semilogy(subdf['fsim'], subdf[col + "_pred_l1"], marker='d', linestyle='none', label="L1 predicted", color=colours[3], alpha=0.3)

            axs[i].legend(loc='lower right')
            axs[i].grid()
        plots["damage/" + type_] = fig
    model.trainer.logger.experiment.log(plots)
    return plots