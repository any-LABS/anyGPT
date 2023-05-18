from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl
from anyGPT.models.anygpt import AnyGPT
import torch


class AnyGPTModelCheckpoint(ModelCheckpoint):
    PT_FILE_EXTENSION = ".pt"

    def __init__(self, model: AnyGPT, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        torch_filepath = filepath.replace(self.FILE_EXTENSION, self.PT_FILE_EXTENSION)
        torch.save(self.model.state_dict(), torch_filepath)

    def _remove_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        super()._remove_checkpoint(trainer, filepath)
        torch_filepath = filepath.replace(self.FILE_EXTENSION, self.PT_FILE_EXTENSION)
        trainer.strategy.remove_checkpoint(torch_filepath)
