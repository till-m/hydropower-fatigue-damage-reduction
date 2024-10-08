import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from .processing import preprocess, get_filepaths, rpad_or_trim_axis_0, load_run


class TransientDataModule(pl.LightningDataModule):
    def __init__(self, preprocess_kwargs, setup_kwargs, batch_size):
        super().__init__()
        self.var = preprocess(**preprocess_kwargs)
        self.preprocess_kwargs = preprocess_kwargs
        self.setup_kwargs = setup_kwargs
        self.batch_size = batch_size
        #self.prepare_data_per_node = False
    
    def setup(self, stage: str) -> None:
        all_files = get_filepaths(self.setup_kwargs["data_directory"])
        train_files = [file for file in all_files if self.preprocess_kwargs["assign_val"] not in file]
        val_files = [file for file in all_files if self.preprocess_kwargs["assign_val"] in file]

        if stage == 'fit':
            self.train_dset = TransientDataset(
                    train_files,
                    self.setup_kwargs["time_series_length"],
                    self.setup_kwargs["control_columns"],
                    self.setup_kwargs["target_columns"],
                    self.setup_kwargs["resample_factor"]
                )

            self.val_dset = TransientDataset(
                val_files,
                self.setup_kwargs["time_series_length"],
                self.setup_kwargs["control_columns"],
                self.setup_kwargs["target_columns"],
                self.setup_kwargs["resample_factor"]
            )

            self.plot_dset = TransientDataset(
                    self.setup_kwargs["plot_files"],
                    None,
                    self.setup_kwargs["control_columns"],
                    self.setup_kwargs["target_columns"],
                    self.setup_kwargs["resample_factor"]
                )

    def train_dataloader(self):
        return DataLoader(self.train_dset, batch_size=self.batch_size, shuffle=self.setup_kwargs["shuffle_train"])

    def val_dataloader(self):
        return DataLoader(self.val_dset, batch_size=self.batch_size)

    def plot_dataloader(self):
        return DataLoader(self.plot_dset, batch_size=1)


class TransientDataset(Dataset):
    def __init__(self, file_list, ts_length, control_columns, target_columns, resample_factor):
        """
        Initializes a TransientDataset object.
        
        Parameters:
        file_list (list): A list of file paths for the data files to be included in the dataset.
        ts_length (int): The length of the time series in the dataset.
        control_columns (list): A list of strings specifying the control columns in the data.
        target_columns (list): A list of strings specifying the target columns in the data.
        resample_factor (int): The factor to resample by
        
        Returns:
        None
        """
        self.file_list = file_list
        self.ts_length = ts_length
        self.control_columns, self.target_columns = control_columns, target_columns
        self.resample_factor = resample_factor
    
    def __len__(self):
        """
        Returns the number of time series in the dataset.
        
        Returns:
        int: The number of time series in the dataset.
        """
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        Returns the idx-th time series in the dataset.
        
        Parameters:
        idx (int): The index of the time series to be returned.
        
        Returns:
        tuple: A tuple containing the control values (torch.Tensor), target values (torch.Tensor),
               and weight values (torch.Tensor) for the idx-th time series.
        """
        X_, y_ = load_run(
            self.file_list[idx],
            self.control_columns,
            self.target_columns,
            self.resample_factor
        )
        if self.ts_length is not None:
            weight = torch.ones_like(y_)

            X_ = rpad_or_trim_axis_0(X_, self.ts_length)
            y_ = rpad_or_trim_axis_0(y_, self.ts_length)
            weight = rpad_or_trim_axis_0(weight, self.ts_length)

            return X_, y_, weight
        return X_, y_
    
    def make_title(self, file_name: str):
        return file_name.split('/')[-1].split('.')[0]
    
    def get_item_for_plot(self, idx):
        title = self.make_title(self.file_list[idx])
        if self.ts_length is not None:
            X_, y_, weight = self[idx]
            return (title, self.control_columns, self.target_columns), (X_, y_, weight)
        X_, y_ = self[idx]
        return (title, self.control_columns, self.target_columns), (X_, y_)

