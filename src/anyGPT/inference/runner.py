import tiktoken
import torch
import torch.nn.functional as F

from anyGPT.data.util import create_enc_dec
from anyGPT.models.lightning import AnyGPTLit


class AnyGPTRunner:
    def __init__(self, checkpoint_path):
        self.model = AnyGPTLit.load_from_checkpoint(checkpoint_path).eval()
        self.encode, self.decode = create_enc_dec(self.model.settings.io_config.dataset)

    def sample(self, x, max_new_tokens: int = 500, temperature: float = 0.8, top_k=200):
        start_ids = self.encode(x)
        y = torch.tensor(start_ids, dtype=torch.long, device=self.model.device)[
            None, ...
        ]
        block_size = self.model.settings.model_config.block_size
        for _ in range(max_new_tokens):
            y_cond = y if y.size(1) <= block_size else y[:, -block_size:]
            logits, _ = self.model(y_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            y_next = torch.multinomial(probs, num_samples=1)
            y = torch.cat((y, y_next), dim=1)

        output = self.decode(y[0].tolist())

        return output
