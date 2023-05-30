from torch import nn

from anyGPT.models.anygpt import AnyGPT, _init_weights


class AnyGPTCritic(AnyGPT):
    def __init__(self, model: AnyGPT, shared_backbone: bool = False):
        super().__init__(model.config)
        self._update_weights(model, shared_backbone)
        self.freeze_params(["adapter"])
        self.lm_head = nn.Linear(self.config.embedding_size, 1, bias=False)
        self.lm_head.apply(_init_weights)

    def _update_weights(self, model: AnyGPT, shared_backbone: bool = False) -> None:
        if shared_backbone:
            self.transformer = model.transformer
        else:
            self.load_state_dict(model.state_dict(), strict=True)
