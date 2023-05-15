import gymnasium as gym
from gymnasium import spaces
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import numpy as np

from anyGPT.data.next_token_dataset import NextTokenDataset
from anyGPT.data.util import create_enc_dec


class SequenceClassificationEnv(gym.Env):
    def __init__(
        self,
        dataset=None,
        render_mode=None,
        block_size=1024,
        label="clean",
        model_name="madhurjindal/autonlp-Gibberish-Detector-492513457",
    ):
        """

        :param dataset:
        :param render_mode:
        :param block_size:
        :param label: The label to optimize for. Will depend on the HF model loaded.
        """
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.block_size = block_size
        self.detector = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        self.observation_space = spaces.Text(block_size)
        self.action_space = spaces.Text(block_size)
        self.dataset = NextTokenDataset(dataset, "train", block_size)
        self.render_mode = render_mode
        self.encode, self.decode = create_enc_dec(dataset)
        self.label = label

    def _get_obs(self):
        rand_idx = np.random.random_integers(0, self.block_size)
        output = self.decode(self.dataset[rand_idx][0])
        return output

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        observation = self._get_obs()
        info = {}

        return observation, info

    def step(self, action):
        output = self.detector(action, top_k=4)
        reward = 0.0
        for result in output:
            if result["label"] == self.label:
                reward = result["score"]
        terminated = True
        observation = None
        info = {}

        return observation, reward, terminated, False, info
