from typing import Any

from anyGPT.data.util import create_enc_dec
from anyGPT.models.anygpt import AnyGPT


class AnyGPTRunner:
    def __init__(self, checkpoint_path):
        self.model, settings = AnyGPT.load_from_pretrained(checkpoint_path)
        self.settings = settings
        self.encode, self.decode = create_enc_dec(self.settings.io_config.dataset)

    def sample(
        self,
        x: str,
        max_new_tokens: int = 500,
        temperature: float = 0.8,
        top_k: int = 200,
        **kwargs: Any
    ) -> str:
        start_ids = self.encode(x)
        y = self.model.generate(start_ids, max_new_tokens, temperature, top_k)
        output = self.decode(y[0].tolist())

        return output
