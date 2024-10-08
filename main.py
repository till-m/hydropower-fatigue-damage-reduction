import wandb

from config import CONFIG
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
LOG_RUN = True # Turn of for debugging, turn on to record to wandb

if LOG_RUN:
    wandb.init(
        project="paired-hydro-transient-selection",
        entity="sdsc-paired-hydro",
        config=CONFIG
    )
    logger = pl.loggers.wandb.WandbLogger(
        log_model=True)
else:
    logger = True # use default


callbacks=[
    EarlyStopping(monitor="loss/validation", mode="min", check_finite=True, patience=1000),
    ModelCheckpoint(monitor="loss/validation", filename='model-{epoch}', auto_insert_metric_name=False)
]

trainer = pl.Trainer(logger=logger, **CONFIG["trainer_config"], callbacks=callbacks)
datamodule = CONFIG["data_module"](**CONFIG["data_module_kwargs"])
wandb.run.summary['preprocess_var'] = datamodule.var.to_dict()

print(isinstance(datamodule, pl.LightningDataModule))

model = CONFIG["model"](**CONFIG["model_kwargs"])

trainer.fit(model=model, datamodule=datamodule)
