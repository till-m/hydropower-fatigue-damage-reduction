import torch
from torch import optim

import src.models as models
import src.data_models as data_models
from src.utils import md5_dir

TARGET_COLUMNS = ["Manner_1", "Manner_4"]
CONTROL_COLUMNS = ["Turbine_speed_(rpm)", "Additionnal_GVO_(XFLEX)", "High_head_(A)"]
DATA_DIRECTORY = "data/normalized"

iTFDNN_CONFIG = {
    "model": models.iTFDNN,
    "model_kwargs": {
        "warm_start_mean_model": 50,
        "window_size": 2048,
        "hop_length" : 256,
        "amp_model": models.ResDNN,
        "amp_model_kwargs": {
            "n_hidden": 128,
            "num_layers": 8
        },
        "phase_model": models.ResDNN,
        "phase_model_kwargs": {
            "n_hidden": 128,
            "num_layers": 6
        },
        "mean_model": models.ResDNN,
        "mean_model_kwargs": {
            "n_hidden": 64,
            "num_layers": 6
        },
        "n_in": len(CONTROL_COLUMNS),
        "n_out": len(TARGET_COLUMNS),
        "optimizer_config": {
            "optimizer": torch.optim.Adam,
            "optimizer_init_kwargs": {
                "lr": 1e-4,
            },
            "scheduler": optim.lr_scheduler.ExponentialLR,
            "scheduler_init_kwargs": {
                "gamma": 0.9998
            },
        }
    },
}

RESAMPLE_FACTOR = 1

DATA_CONFIG = {
    "data_module": data_models.TransientDataModule,
    "data_module_kwargs": {
        "batch_size": 1,
        "preprocess_kwargs": {
            "assign_val": 'BEP',
            "in_folder": "data/c1sel"
        },
        "setup_kwargs" : {
            "data_directory": DATA_DIRECTORY,
            "control_columns": CONTROL_COLUMNS,
            "target_columns": TARGET_COLUMNS,
            "resample_factor": RESAMPLE_FACTOR,
            "time_series_length": None,
            "shuffle_train": False,
            "plot_files": [
                DATA_DIRECTORY + "/140_2Slopes_0_9.parquet",
                DATA_DIRECTORY + "/100_BEP_4_8.parquet",
                DATA_DIRECTORY + "/000_Classic_2_4.parquet",
                DATA_DIRECTORY + "/231_Linear_3_6.parquet",
            ]
        }
    }
}

TRAINER_CONFIG = {
    "accelerator": "gpu",
    "max_epochs": 10000,
    "log_every_n_steps": 1,
    "gradient_clip_val": 1.,
}


CONFIG = {
    **iTFDNN_CONFIG,
    **DATA_CONFIG,
    "trainer_config": TRAINER_CONFIG,
    "dataset_md5": md5_dir("data/c1sel")
}
