#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import os
import torch
from scipy.signal import decimate

#import cyl1tf
import torch.nn.functional as F

from .damage import calculate_damage_history


def get_filepaths(folder):
    """
    Returns a list of file paths for all files in the specified folder.

    Parameters:
    folder (str): The path to the folder to search for files.

    Returns:
    list: A list of file paths.
    """
    for dirpath, dirnames, filenames in os.walk(folder):
        return [dirpath + '/' + file for file in filenames]


def preprocess(assign_val,
              in_folder="data/c1sel",
              out_folder="data/normalized",
              add_diff=[],
              damage=[],
              damage_cumsum=False):
    """
    Normalizes the data in the input folder by subtracting the mean and dividing by the standard deviation
    of the training data. The normalized data is saved to the output folder. The training data is defined as
    all files that do not contain the string "assign_val" in their names, and the test data is defined as
    all files that contain the string "assign_val" in their names.

    Parameters:
    assign_val (str): The string used to distinguish between training and test data.
    in_folder (str): The path to the folder containing the input data.
    out_folder (str): The path to the folder where the normalized data will be saved.
    
    Returns:
    None
    """
    print("processing files...")

    all_files = get_filepaths(in_folder)
    train_files = [file for file in all_files if assign_val not in file]
    test_files = [file for file in all_files if assign_val in file]

    assert len(train_files) + len(test_files) == len(all_files)

    means = []
    means2 = []
    weights = []

    for file_path in train_files:
        if not file_path.endswith('.parquet'):
            continue
        df = pd.read_parquet(file_path)
        for col in damage:
            dh = calculate_damage_history(df[col])
            pad_ = len(df.index) - dh.size
            dh = np.pad(dh, (0, pad_))
            if damage_cumsum:
                dh = np.cumsum(dh)
            df[col] = dh
        means.append(df.mean())
        means2.append((df**2).mean())
        weights.append(len(df.index))

    means = np.sum(pd.concat(means, axis=1) * weights, axis=1)/np.sum(weights)
    means2 = np.sum(pd.concat(means2, axis=1) * weights, axis=1)/np.sum(weights)
    var = means2 - means**2

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for file_path in all_files:
        if not file_path.endswith('.parquet'):
            continue
        df = pd.read_parquet(file_path)
        for col in damage:
            dh = calculate_damage_history(df[col])
            pad_ = len(df.index) - dh.size
            dh = np.pad(dh, (0, pad_))
            if damage_cumsum:
                dh = np.cumsum(dh)
            df[col] = dh
        df = df/np.sqrt(var)
        filename = file_path.split('/')[-1]
        if add_diff:
            for col in add_diff:
                df[col+'_diff'] = df[col].diff()
            df = df.dropna() # drop 1st row, which has NaN for df
        df.to_parquet(out_folder + '/' + filename)
    return var
    


def rpad_or_trim_axis_0(tensor: torch.Tensor, pad_size: int):
    """
    Right-side pads or trims the input tensor along the first dimension so that it has a size of pad_size.
    If the input tensor has size larger than pad_size, it is truncated.
    If the input tensor has size smaller than pad_size, it is padded with zeros.

    Parameters:
    tensor (torch.Tensor): The input tensor.
    pad_size (int): The desired size of the output tensor along the first dimension.

    Returns:
    torch.Tensor: The padded or trimmed tensor.
    """
    if tensor.shape[0] > pad_size:
        tensor = tensor[:pad_size]
    elif tensor.shape[0] < pad_size:
        pad = tuple([0 for _ in tensor.shape[1:]]) * 2
        pad = pad + (0, int(pad_size - tensor.shape[0]))
        tensor = F.pad(tensor, pad=pad)

    return tensor


def lpad_or_trim_axis_last(tensor: torch.Tensor, pad_size: int):
    """
    Left-side pads or trims the input tensor along the last dimension so that it has a size of pad_size.
    If the input tensor has size larger than pad_size, it is truncated.
    If the input tensor has size smaller than pad_size, it is padded with zeros.

    Parameters:
    tensor (torch.Tensor): The input tensor.
    pad_size (int): The desired size of the output tensor along the last dimension.

    Returns:
    torch.Tensor: The padded or trimmed tensor.
    """
    if tensor.shape[-1] > pad_size:
        tensor = tensor[...,-pad_size:]
    elif tensor.shape[-1] < pad_size:
        pad = tuple([0 for _ in tensor.shape[1:]]) * 2
        pad = (pad_size - tensor.shape[-1], 0)+ pad 
        tensor = F.pad(tensor, pad=pad)

    return tensor


def load_run(path, control_columns, target_columns, resample_factor):
    """
    Load run data from parquet file and decimate control and target columns.
    
    Parameters
    ----------
    path : str
        The file path of the parquet file.
    control_columns : list
        List of control column names.
    target_columns : list
        List of target column names.
    resample_factor : int
        The decimation factor to apply to control and target columns.
    
    Returns
    -------
    tuple
        Tuple containing the decimated control and target data as PyTorch tensors.
    """
    df_ = pd.read_parquet(path)
    df = pd.DataFrame()
    for col in (control_columns + target_columns):
        df[col] = decimate(df_[col], resample_factor)
    X_ = torch.Tensor(df[control_columns].values)
    y_ = torch.Tensor(df[target_columns].values)
    return X_, y_
