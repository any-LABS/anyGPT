import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    StochasticWeightAveraging,
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from anyGPT.config.settings import AnyGPTSettings
from anyGPT.data.next_token_dataset import NextTokenDataset
from anyGPT.models.lightning import AnyGPTLit


class AnyGPTTrainer:
    def __init__(self, settings: AnyGPTSettings):
        super().__init__()
        self.settings = settings
        pl.seed_everything(self.settings.training_config.seed)
        if self.settings.torch_config.compile:
            self.model = torch.compile(AnyGPTLit(self.settings))
        else:
            self.model = AnyGPTLit(self.settings)

        if self.settings.training_config.init_from is not "scratch":
            self.model.from_pretrained(self.settings.training_config.init_from)

        self.train_set = NextTokenDataset(
            self.settings.io_config.dataset,
            "train",
            self.settings.model_config.block_size,
        )
        self.val_set = NextTokenDataset(
            self.settings.io_config.dataset,
            "val",
            self.settings.model_config.block_size,
        )
        self.train_dataloader = DataLoader(
            self.train_set,
            batch_size=self.settings.training_config.batch_size,
            num_workers=12,
            shuffle=True,
        )
        self.val_dataloader = DataLoader(
            self.val_set,
            batch_size=self.settings.training_config.batch_size,
            num_workers=12,
            shuffle=True,
        )
        self.logger = TensorBoardLogger(
            self.settings.io_config.out_dir, self.settings.io_config.experiment_name
        )
        self.trainer = pl.Trainer(
            max_steps=self.settings.training_config.max_steps,
            gradient_clip_val=self.settings.training_config.grad_clip,
            accumulate_grad_batches=self.settings.training_config.accumulate_gradients,
            callbacks=[
                StochasticWeightAveraging(
                    swa_lrs=self.settings.training_config.swa_lrs
                ),
                EarlyStopping("val/loss", mode="min"),
                ModelCheckpoint(
                    monitor="val/loss",
                    save_last=True,
                    save_top_k=2,
                    mode="min",
                    save_on_train_epoch_end=False,
                    auto_insert_metric_name=True,
                ),
            ],
            val_check_interval=self.settings.training_config.val_check_interval,
            logger=self.logger,
            enable_checkpointing=self.settings.io_config.enable_checkpointing,
            num_sanity_val_steps=0,
            log_every_n_steps=self.settings.io_config.log_every_n_steps,
            limit_train_batches=self.settings.training_config.limit_train_batches,
            limit_val_batches=self.settings.training_config.limit_val_batches,
            limit_test_batches=self.settings.training_config.limit_test_batches,
            precision=self.settings.torch_config.precision,
        )

    def fit(self):
        print(
            f"Starting to train {self.settings.model_config.name} with {self.settings.io_config.dataset} dataset."
        )
        self.trainer.fit(
            self.model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
        )
