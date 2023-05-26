from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from anyGPT.data.util import load_metadata
from anyGPT.models.anygpt import AnyGPT


class AnyGPTModelCheckpoint(ModelCheckpoint):
    PT_FILE_EXTENSION = ".pt"

    def __init__(self, model: AnyGPT, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.model = model

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        torch_filepath = filepath.replace(self.FILE_EXTENSION, self.PT_FILE_EXTENSION)
        metadata = load_metadata(trainer.model.settings.io_config.dataset)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": self.model.config,
                "settings": trainer.model.settings,
                "metadata": metadata,
            },
            torch_filepath,
        )

    def _remove_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        super()._remove_checkpoint(trainer, filepath)
        torch_filepath = filepath.replace(self.FILE_EXTENSION, self.PT_FILE_EXTENSION)
        trainer.strategy.remove_checkpoint(torch_filepath)
