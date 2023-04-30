import torch
import torch.nn as nn
from torch.nn import functional as F

from anyGPT.config.settings import ModelConfig
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

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.embedding_size),
            wpe=nn.Embedding(config.block_size, config.embedding_size),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([TxBlock(config) for _ in range(config.num_layers)]),
            ln_f=LayerNorm(config)
        ))

        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(_init_weights)

    def forward(self, idx, targets=None):
        device = idx.device
        # b: batch size, t: sequence length
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only " \
                                            f"{self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        token_embedding = self.transformer.wte(idx)
        position_embedding = self.transformer.wpe(pos)
        x = self.transformer.drop(token_embedding + position_embedding)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
