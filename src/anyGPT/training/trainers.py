import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    StochasticWeightAveraging,
    EarlyStopping,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from anyGPT.config.settings import AnyGPTSettings
from anyGPT.data.next_token_dataset import NextTokenDataset
from anyGPT.models.anygpt_lit import AnyGPTLit
from anyGPT.models.anygpt_ppo_lit import AnyGPTPPOLit
from anyGPT.training.model_checkpoint import AnyGPTModelCheckpoint


class AnyGPTPreTrainer:
    def __init__(self, settings: AnyGPTSettings):
        self.settings = settings
        pl.seed_everything(self.settings.training_config.seed, workers=True)
        if self.settings.torch_config.compile:
            self.model = torch.compile(AnyGPTLit(self.settings))
        else:
            self.model = AnyGPTLit(self.settings)

        if self.settings.training_config.init_from != "scratch":
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
            self.settings.io_config.out_dir,
            self.settings.io_config.experiment_name + "-pretrain",
        )
        self.trainer = pl.Trainer(
            max_steps=self.settings.training_config.max_steps,
            gradient_clip_val=self.settings.training_config.grad_clip,
            accumulate_grad_batches=self.settings.training_config.accumulate_gradients,
            callbacks=[
                StochasticWeightAveraging(
                    swa_lrs=self.settings.training_config.swa_lrs
                ),
                EarlyStopping("val_loss", mode="min"),
                AnyGPTModelCheckpoint(
                    self.model.model,
                    monitor="val_loss",
                    save_last=True,
                    save_top_k=2,
                    mode="min",
                    save_on_train_epoch_end=False,
                    auto_insert_metric_name=True,
                    filename="anygpt-pretrained-{epoch:02d}--{step:02d}-{val_loss:.2f}",
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
            accelerator=self.settings.torch_config.accelerator,
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


class AnyGPTPPOTrainer:
    def __init__(self, settings: AnyGPTSettings):
        self.settings = settings
        pl.seed_everything(self.settings.training_config.seed, workers=True)
        if self.settings.torch_config.compile:
            self.model = torch.compile(AnyGPTPPOLit(self.settings))
        else:
            self.model = AnyGPTPPOLit(self.settings)

        self.logger = TensorBoardLogger(
            self.settings.io_config.out_dir,
            self.settings.io_config.experiment_name + "-rlhf",
        )

        self.trainer = pl.Trainer(
            max_epochs=self.settings.ppo_config.epochs,
            accumulate_grad_batches=self.settings.training_config.accumulate_gradients,
            callbacks=[
                StochasticWeightAveraging(
                    swa_lrs=self.settings.training_config.swa_lrs
                ),
                EarlyStopping("avg_ep_reward", mode="max", patience=100),
                AnyGPTModelCheckpoint(
                    self.model.policy.actor,
                    monitor="avg_ep_reward",
                    save_last=True,
                    save_top_k=2,
                    mode="max",
                    save_on_train_epoch_end=False,
                    auto_insert_metric_name=True,
                    filename="anygpt-rl-training-{epoch:02d}--{step:02d}-{avg_ep_reward:.2f}",
                    every_n_train_steps=self.settings.io_config.log_every_n_steps * 10,
                ),
            ],
            logger=self.logger,
            enable_checkpointing=self.settings.io_config.enable_checkpointing,
            log_every_n_steps=self.settings.io_config.log_every_n_steps,
            precision=self.settings.torch_config.precision,
            accelerator=self.settings.torch_config.accelerator,
        )

    def fit(self):
        print(
            f"Starting to RL train {self.settings.model_config.name} with "
            f"{self.settings.ppo_config.env_kwargs['dataset']} dataset \n"
            f"and environment {self.settings.ppo_config.env}."
        )
        self.trainer.fit(
            self.model,
        )
