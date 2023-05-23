import inspect

import lightning.pytorch as pl
import torch.optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import GPT2LMHeadModel

from anyGPT.config.settings import AnyGPTSettings
from anyGPT.models.anygpt import AnyGPT
from anyGPT.models.modules import LayerNorm


class AnyGPTLit(pl.LightningModule):
    def __init__(self, settings: AnyGPTSettings):
        super().__init__()
        self.settings = settings
        self.model = AnyGPT(self.settings.model_config)
        self.save_hyperparameters()

    def from_pretrained(self, name):
        assert name in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        self._update_settings("gpt2")
        self.model = AnyGPT(self.settings.model_config)
        state_dict = self.model.state_dict()
        state_dict_keys = state_dict.keys()
        state_dict_keys = [k for k in state_dict_keys if not k.endswith(".attn.bias")]
        model_hf = GPT2LMHeadModel.from_pretrained(name)
        state_dict_hf = model_hf.state_dict()
        state_dict_hf_keys = state_dict_hf.keys()
        state_dict_hf_keys = [
            k for k in state_dict_hf_keys if not k.endswith(".attn.masked_bias")
        ]
        state_dict_hf_keys = [
            k for k in state_dict_hf_keys if not k.endswith(".attn.bias")
        ]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        assert len(state_dict_hf_keys) == len(
            state_dict_keys
        ), f"mismatched keys: {len(state_dict_hf_keys)} != {len(state_dict_keys)}"
        for k in state_dict_hf_keys:
            if any(k.endswith(w) for w in transposed):
                assert state_dict_hf[k].shape[::-1] == state_dict[k].shape
                with torch.no_grad():
                    state_dict[k].copy_(state_dict_hf[k].t())
            else:
                assert state_dict_hf[k].shape == state_dict[k].shape
                with torch.no_grad():
                    state_dict[k].copy_(state_dict_hf[k])

    def _update_settings(self, name):
        model_specs = {
            "gpt2": {
                "num_layers": 12,
                "num_heads": 12,
                "embedding_size": 768,
            },  # 124M params
            "gpt2-medium": {
                "num_layers": 24,
                "num_heads": 16,
                "embedding_size": 1024,
            },  # 350M params
            "gpt2-large": {
                "num_layers": 36,
                "num_heads": 20,
                "embedding_size": 1280,
            },  # 774M params
            "gpt2-xl": {
                "num_layers": 48,
                "num_heads": 25,
                "embedding_size": 1600,
            },  # 1558M params
        }[name]
        model_specs["vocab_size"] = 50257
        model_specs["block_size"] = 1024
        model_specs["bias"] = True
        model_specs["name"] = name
        self.settings.model_config.update(model_specs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_index):
        x, y = batch
        logits, loss = self.model(x, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_index):
        x, y = batch
        logits, loss = self.model(x, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        modules_to_decay = torch.nn.Linear
        modules_not_to_decay = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        learning_rate = self.settings.training_config.learning_rate
        betas = (
            self.settings.training_config.beta1,
            self.settings.training_config.beta2,
        )
        if self.settings.training_config.decay_lr:
            for mn, m in self.named_modules():
                for pn, _ in m.named_parameters():
                    fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                    # random note: because named_modules and named_parameters are recursive
                    # we will see the same tensors p many many times. but doing it this way
                    # allows us to know which parent module any tensor p belongs to...
                    if pn.endswith("bias"):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, modules_to_decay):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, modules_not_to_decay):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
            decay.remove("model.lm_head.weight")
            param_dict = dict(self.named_parameters())
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert (
                len(inter_params) == 0
            ), "parameters %s made it into both decay/no_decay sets!" % (
                str(inter_params),
            )
            assert (
                len(param_dict.keys() - union_params) == 0
            ), "parameters %s were not separated into either decay/no_decay set!" % (
                str(param_dict.keys() - union_params),
            )
            optim_groups = [
                {
                    "params": [param_dict[pn] for pn in sorted(decay)],
                    "initial_lr": self.settings.training_config.learning_rate,
                    "weight_decay": self.settings.training_config.weight_decay,
                },
                {
                    "params": [param_dict[pn] for pn in sorted(no_decay)],
                    "initial_lr": self.settings.training_config.learning_rate,
                    "weight_decay": 0.0,
                },
            ]
        else:
            optim_groups = self.parameters()

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self.device == "cuda"
        extra_args = {"fused": True} if use_fused else {}
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        if self.settings.training_config.decay_lr:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.settings.training_config.warmup_iters,
                eta_min=self.settings.training_config.min_lr,
                last_epoch=self.settings.training_config.max_steps,
            )
            return [optimizer], [scheduler]
        else:
            return optimizer
