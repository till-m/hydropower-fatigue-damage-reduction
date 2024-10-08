import torch
import torch.nn as nn
from tqdm import tqdm
import os

def square(in_):
    return in_**2


def weighted_mse_loss(in_, target, weight):
    """
    Calculates the mean squared error between the input and target tensors,
    weighting the loss by the given weight tensor.
    """
    return torch.sum(weight * (in_ - target) ** 2)/torch.sum(weight)


def weighted_smoothness_loss(in_, weight, axis=-1):
    in_ = in_.swapaxes(0, axis)
    weight = weight.swapaxes(0, axis)
    return torch.sum(weight[2:]*(in_[2:] - 2* in_[1:-1] + in_[:-2])**2)/torch.sum(weight[2:])


def initialize(config):
    """
    Initialize model, optimizer, and dataloaders based on the given config.
    
    Parameters:
    config (dict): A dictionary containing the following keys:
        - "model" (callable): A function that returns a new model instance.
        - "model_init_kwargs" (dict): Keyword arguments to pass to the model function when instantiating the model.
        - "optimizer" (callable): A function that returns a new optimizer instance.
        - "optimizer_init_kwargs" (dict): Keyword arguments to pass to the optimizer function when instantiating the optimizer.
        - "assign_test" (float): The percentage of data to assign to the test set.
        - "data_directory" (str): The path to the directory containing the data.
        - "time_series_length" (int): The length of the time series data.
        - "control_columns" (list of str): The names of the control columns in the data.
        - "target_columns" (list of str): The names of the target columns in the data.
    
    Returns:
    tuple: A tuple containing the following elements:
        - model (nn.Module): The initialized model.
        - optimizer (optim.Optimizer): The initialized optimizer.
        - train_dloader (DataLoader): The dataloader for the training set.
        - test_dloader (DataLoader): The dataloader for the test set.
    """
    train_dloader, test_dloader = make_dataloaders(
        batch_size=config["batch_size"],
        assign_test=config["assign_test"],
        data_directory=config["data_directory"],
        time_series_length=config["time_series_length"],
        control_columns=config["control_columns"],
        target_columns=config["target_columns"],
        resample_factor=config["resample_factor"],
        shuffle_train=config["shuffle_train"],
        device=config["device"]
    )
    return train_dloader, test_dloader


def teardown(model, path):
    """
    Save the model to a file with the name specified in the config.
    
    Parameters:
    model (nn.Module): The model to save.
    path (str): The name to use for the saved model file.
    """
    out_folder = "artifacts/" + path 
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    torch.save(model, out_folder + "/model.pt")


def train_step(model: nn.Module, optimizer, scheduler, loss, train_dloader, device):
    """
    Trains the model for one epoch using the given data loader, loss function, and optimizer.
    Returns the average loss and the output of the model on the last batch of data.
    """
    # Set model to training mode
    model = model.train()

    epoch_loss = []
    for X_, y_, weights in tqdm(train_dloader):
        losses = []
        # Transfer data to device and zero the gradients
        X_, y_, weights = X_.to(device), y_.to(device), weights.to(device)


        def closure():
            optimizer.zero_grad()
            out = model(X_)
            loss_ = loss(out, y_, weights)
            loss_.backward()
            return loss_

        loss_ = optimizer.step(closure)
        
        # Record average loss for the epoch
        epoch_loss.append(loss_.item())
    epoch_loss = sum(epoch_loss)/len(epoch_loss)
    
    # Set model to evaluation mode
    model = model.eval()
    
    if scheduler is not None:
        scheduler.step()
    return epoch_loss


def test_step(model: nn.Module, loss, test_dloader, device):
    """
    Evaluates the model using the given data loader.
    Returns the average loss and the output of the model on the last batch of data.
    """
    # Set model to evaluation mode
    model = model.eval()
    
    epoch_loss = []
    with torch.no_grad():
        all_pred = []
        for X_, y_, weights in tqdm(test_dloader):
            losses = []
            # Transfer data to device
            X_, y_, weights = X_.to(device), y_.to(device), weights.to(device)

            # Make predictions and compute loss
            out = model(X_)
            losses.append(loss(out, y_, weights))
            
            # Record average loss for the epoch
            epoch_loss.append(sum(losses).item()/len(losses))
        epoch_loss = sum(epoch_loss)/len(epoch_loss)
    return epoch_loss
