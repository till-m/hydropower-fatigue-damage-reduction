import hashlib
import os
import torch
import matplotlib.pyplot as plt

import hashlib
from _hashlib import HASH as Hash
from pathlib import Path
from typing import Union


def md5_update_from_file(filename: Union[str, Path], hash: Hash) -> Hash:
    assert Path(filename).is_file()
    with open(str(filename), "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash.update(chunk)
    return hash


def md5_file(filename: Union[str, Path]) -> str:
    return str(md5_update_from_file(filename, hashlib.md5()).hexdigest())


def md5_update_from_dir(directory: Union[str, Path], hash: Hash) -> Hash:
    if not Path(directory).is_dir():
        raise FileNotFoundError
    for path in sorted(Path(directory).iterdir(), key=lambda p: str(p).lower()):
        hash.update(path.name.encode())
        if path.is_file():
            hash = md5_update_from_file(path, hash)
        elif path.is_dir():
            hash = md5_update_from_dir(path, hash)
    return hash


def md5_dir(directory: Union[str, Path]) -> str:
    try:
        return str(md5_update_from_dir(directory, hashlib.md5()).hexdigest())
    except FileNotFoundError:
        return 'empty-dir'
    


def plot_approximation(y_true, y_pred, title, columns):
    """
    Plots the true and predicted values for multiple columns.

    Parameters:
    - y_true (np.ndarray): An array of shape (timesteps, n_columns) containing the true values.
    - y_pred (np.ndarray): An array of shape (timesteps, n_columns) containing the predicted values.
    - title (str): The title to use for the plots.
    - columns (List[str]): A list of strings with the names of the columns.

    Returns:
    - List[matplotlib.figure.Figure]: A list of matplotlib figure objects, one for each column.
    """
    
    figs = []
    for j in range(y_pred.shape[-1]):
        fig = plt.figure()
        ax = fig.gca()
        ax.set_title(title + ' ' + columns[j])
        ax.plot(y_true[:, j], label='true')
        ax.plot(y_pred[:, j], label='pred')
        ax.legend(loc='best')
        ax.grid(True)
        figs.append(fig)
    return figs
