from anyGPT.models.anygpt import AnyGPT
from torch import nn


class AnyGPTCritic(AnyGPT):
    def __init__(self, model: AnyGPT, shared_backbone: bool = False):
        super().__init__(model.config)
        self._update_weights(model, shared_backbone)
        self.lm_head = nn.Linear(self.config.embedding_size, 1, bias=False)

    def _update_weights(self, model: AnyGPT, shared_backbone: bool = False):
        if shared_backbone:
            self.transformer = model.transformer
        self.load_state_dict(model.state_dict(), strict=True)
