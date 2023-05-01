import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import StochasticWeightAveraging
from pytorch_lightning.utilities.model_summary import ModelSummary
from torch.utils.data import DataLoader, random_split
from anyGPT.config.settings import AnyGPTSettings
from anyGPT.data.next_token_dataset import NextTokenDataset
from anyGPT.models.lightning import AnyGPTLit


class AnyGPTTrainer:
    def __init__(self, settings: AnyGPTSettings):
        super().__init__()
        self.settings = settings
        if self.settings.torch_config.compile:
            self.model = torch.compile(AnyGPTLit(self.settings))
        else:
            self.model = AnyGPTLit(self.settings)
        self.dataset = NextTokenDataset(self.settings.io_config.dataset, self.settings.model_config.block_size)
        self.train_set_size = int(len(self.dataset) * 0.8)
        self.val_set_size = len(self.dataset) - self.train_set_size
        seed = torch.Generator().manual_seed(42)
        self.train_set, self.val_set = random_split(self.dataset, [self.train_set_size, self.val_set_size],
                                                    generator=seed)
        self.train_dataloader = DataLoader(self.train_set, batch_size=self.settings.training_config.batch_size,
                                           num_workers=12, shuffle=True)
        self.val_dataloader = DataLoader(self.val_set, batch_size=self.settings.training_config.batch_size,
                                         num_workers=12)
        self.trainer = pl.Trainer(max_epochs=self.settings.training_config.max_epochs,
                                  gradient_clip_val=self.settings.training_config.grad_clip,
                                  accumulate_grad_batches=self.settings.training_config.accumulate_gradients,
                                  callbacks=[StochasticWeightAveraging(swa_lrs=self.settings.training_config.swa_lrs)],
                                  val_check_interval=self.settings.training_config.val_check_interval)

    def fit(self):
        print(f"Starting to train {self.settings.model_config.name} with {self.settings.io_config.dataset} dataset.")
        print("Model Summary")
        summary = ModelSummary(self.model)
        print(summary)
        self.trainer.fit(self.model, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader)
