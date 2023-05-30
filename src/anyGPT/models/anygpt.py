import math
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from anyGPT.config.settings import ModelConfig, AnyGPTSettings
from anyGPT.models.modules import TxBlock, LayerNorm


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


class AnyGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.embedding_size),
                "wpe": nn.Embedding(config.block_size, config.embedding_size),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList([TxBlock(config) for _ in range(config.num_layers)]),
            }
        )
        if config.move_layer_norm:
            self.transformer.update({"ln_f": LayerNorm(config)})

        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(_init_weights)

        if config.move_layer_norm:
            for pn, p in self.named_parameters():
                if pn.endswith("c_proj.weight"):
                    torch.nn.init.normal_(
                        p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.num_layers)
                    )

    def forward(self, idx, targets=None):
        device = idx.device
        # b: batch size, t: sequence length
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only "
            f"{self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        token_embedding = self.transformer.wte(idx)
        position_embedding = self.transformer.wpe(pos)
        x = self.transformer.drop(token_embedding + position_embedding)

        for block in self.transformer.h:
            x = block(x)

        if self.config.move_layer_norm:
            x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def generate(
        self,
        start_ids: List,
        max_new_tokens: int = 500,
        temperature: float = 0.8,
        top_k: int = 200,
    ) -> torch.Tensor:
        y = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]
        block_size = self.config.block_size
        for _ in range(max_new_tokens):
            y_cond = y if y.size(1) <= block_size else y[:, -block_size:]
            logits, _ = self(y_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            y_next = torch.multinomial(probs, num_samples=1)
            y = torch.cat((y, y_next), dim=1)

        return y

    def freeze_params(self, skip_layers: List[str]) -> None:
        for name, param in self.named_parameters():
            for layer in skip_layers:
                if layer not in name:
                    param.requires_grad = False
                    break

    def unfreeze_params(self):
        for param in self.parameters():
            param.requires_grad = True

    @staticmethod
    def load_from_pretrained(
        checkpoint_path: str, fine_tune: bool = False
    ) -> Tuple[nn.Module, AnyGPTSettings, dict]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint["model_config"]
        settings = checkpoint["settings"]
        metatdata = checkpoint["metadata"]
        config.fine_tune = fine_tune
        model = AnyGPT(config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.freeze_params(["adapter"])
        model.to(settings.torch_config.device)

        return model, settings, metatdata

    @property
    def device(self):
        return next(self.parameters()).device
