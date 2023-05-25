import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from anyGPT.data.next_token_dataset import NextTokenDataset
from anyGPT.data.util import create_enc_dec


class SequenceClassificationEnv(gym.Env):
    def __init__(
        self,
        dataset: str = None,
        render_mode: str = None,
        block_size: int = 2,
        label: str = "clean",
        model_name: str = "madhurjindal/autonlp-Gibberish-Detector-492513457",
        encoded: bool = True,
        device: str = "cpu",
    ):
        """

        :param dataset: The dataset to sample observations from.
        :param render_mode:
        :param block_size: The size of the observation string.
        :param label: The label to optimize for. Will depend on the HF model loaded.
        :param model_name: The name of the HF sequence classifier.
        :param encoded: If set to true, observations and actions should be encoded with the proper gpt style tokenizer.
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        assert block_size >= 2
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.block_size = block_size
        self.reward_model = pipeline(
            "sentiment-analysis", model=model, tokenizer=tokenizer, device=device
        )
        if encoded:
            self.observation_space = spaces.Discrete(self.block_size)
            self.action_space = spaces.Discrete(self.block_size)
        else:
            self.observation_space = spaces.Text(block_size)
            self.action_space = spaces.Text(block_size)
        self.dataset = NextTokenDataset(dataset, "train", block_size)
        self.render_mode = render_mode
        self.encode, self.decode = create_enc_dec(dataset)
        self.label = label
        self.encoded = encoded

    def _get_obs(self):
        rand_idx = np.random.random_integers(0, self.block_size)
        output = self.dataset[rand_idx][0]
        if self.encoded:
            output = output.astype(np.int64)
        else:
            output = self.decode(output)
        return output

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        observation = self._get_obs()
        info = {}

        return observation, info

    def step(self, action):
        if self.encoded:
            action = self.decode(action.squeeze())
        output = self.reward_model(action, top_k=4)
        reward = 0.0
        for result in output:
            if result["label"] == self.label:
                reward = result["score"]
        terminated = True
        truncated = False
        if self.encoded:
            observation = -1
        else:
            observation = "[EPISODE_END]"
        info = {}

        return observation, reward, terminated, truncated, info
