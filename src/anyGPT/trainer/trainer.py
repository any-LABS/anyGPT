import lightning.pytorch as pl
from pytorch_lightning.utilities.model_summary import ModelSummary
from torch.utils.data import DataLoader
from anyGPT.config.settings import AnyGPTSettings
from anyGPT.data.next_token_dataset import NextTokenDataset
from anyGPT.models.lightning import AnyGPTLit


class AnyGPTTrainer:
    def __init__(self, settings: AnyGPTSettings):
        super().__init__()
        self.settings = settings
        self.model = AnyGPTLit(self.settings)
        self.dataset = NextTokenDataset(self.settings.io_config.dataset, self.settings.model_config.block_size)
        self.dataloader = DataLoader(self.dataset)
        self.trainer = pl.Trainer(limit_train_batches=0.25, max_epochs=1)

    def fit(self):
        print(f"Starting to train {self.settings.model_config.name} with {self.settings.io_config.dataset} dataset.")
        print("Model Summary")
        summary = ModelSummary(self.model)
        print(summary)
        self.trainer.fit(self.model, train_dataloaders=self.dataloader)
