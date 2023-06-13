from contextlib import nullcontext
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from anyGPT.config.settings import AnyGPTSettings
from anyGPT.models.anygpt import AnyGPT
from anyGPT.models.anygpt_critic import AnyGPTCritic


class PPOPolicy(nn.Module):
    def __init__(self, settings: AnyGPTSettings):
        super().__init__()
        self.settings = settings
        self.actor = self._init_actor()
        self.actor_ref = self._init_actor().eval()
        self.critic = self._init_critic()
        self.action_size = self.settings.ppo_config.action_size

    def _init_actor(self):
        checkpoint = self.settings.ppo_config.checkpoint
        actor, _, _ = AnyGPT.load_from_pretrained(checkpoint, fine_tune=True)
        return actor

    def _init_critic(self):
        critic = AnyGPTCritic(self.actor, self.settings.ppo_config.shared_actor_critic)
        return critic

    def forward(self, *args, **kwargs):
        action = self.actor(*args, **kwargs)
        value = self.critic(*args, **kwargs)
        return action, value

    def params(self):
        params = list(self.actor.parameters())
        params += list(self.critic.parameters())
        return params

    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        block_size: int,
        device: torch.device,
        use_reference: bool = True,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        probs = torch.tensor([]).to(device)
        log_probs = torch.tensor([]).to(device)
        log_probs_ref = torch.tensor([]).to(device)
        values = torch.tensor([]).to(device)

        for i in range(max_new_tokens):
            with torch.no_grad() if i < max_new_tokens - 1 else nullcontext():
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
                logits, _ = self.actor(x_cond)
                value_next, _ = self.critic(x_cond)
                values = torch.cat((values, value_next), dim=1)
                logits = logits[:, -1, :] / temperature
                probs_next = F.softmax(logits, dim=-1)
                x_next = torch.multinomial(probs_next, num_samples=1)
                probs_x_next = torch.gather(probs_next, 1, x_next)
                log_probs_x_next = torch.log(probs_x_next)
                log_probs = torch.cat((log_probs, log_probs_x_next), dim=1)
                probs = torch.cat((probs, probs_x_next), dim=1)

                if use_reference:
                    logits_ref, _ = self.actor_ref(x_cond)
                    logits_ref = logits_ref[:, -1, :]
                    probs_ref_next = F.softmax(logits_ref, dim=-1)
                    probs_ref_x_next = torch.gather(probs_ref_next, 1, x_next)
                    log_probs_ref_x_next = torch.log(probs_ref_x_next)
                    log_probs_ref = torch.cat(
                        (log_probs_ref, log_probs_ref_x_next), dim=1
                    )

                x = torch.cat((x, x_next), dim=1)

        return (
            x[:, -max_new_tokens:],
            probs[:, -max_new_tokens:],
            log_probs[:, -max_new_tokens:],
            log_probs_ref[:, -max_new_tokens:],
            values.squeeze(dim=2),
        )
