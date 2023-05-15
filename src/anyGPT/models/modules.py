import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from anyGPT.config.settings import ModelConfig
from anyGPT.models.operators import new_gelu


class LayerNorm(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.embedding_size))
        self.bias = (
            nn.Parameter(torch.zeros(config.embedding_size)) if config.bias else None
        )

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.embedding_size % config.num_heads == 0

        # query, key, value projections for all heads, in a batch
        self.c_attn = nn.Linear(
            config.embedding_size, 3 * config.embedding_size, bias=config.bias
        )

        # output projection
        self.c_proj = nn.Linear(
            config.embedding_size, config.embedding_size, bias=config.bias
        )

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.num_heads = config.num_heads
        self.embedding_size = config.embedding_size
        self.dropout = config.dropout

        # use flash attention!!! yay
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "Yo yo! anyGPT will use slow attention. Flash attention requires PyTorch >= 2.0"
            )
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        # B: batch size, T: sequence length, C: embedding size
        B, T, C = x.size()

        # compute query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.embedding_size, dim=2)

        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))

        return y


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(
            config.embedding_size, 4 * config.embedding_size, bias=config.bias
        )
        self.c_proj = nn.Linear(
            4 * config.embedding_size, config.embedding_size, bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        s = self.dropout(x)
        return s


class TxBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config)
        self.attention = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config)
        self.mlp = MLP(config)
        if config.move_layer_norm:
            self._forward = self._pre_layer_norm
        else:
            self._forward = self._post_layer_norm

    def forward(self, x):
        x = self._forward(x)
        return x

    def _pre_layer_norm(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def _post_layer_norm(self, x):
        x = self.ln_1(x + self.attention(x))
        x = self.ln_2(x + self.mlp(x))
        return x
