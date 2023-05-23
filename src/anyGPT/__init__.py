import os.path

from gymnasium.envs.registration import register

DEFAULT_DIR = os.path.join(os.path.expanduser("~"), ".cache", "anygpt")
DEFAULT_DATADIR = os.path.join(DEFAULT_DIR, "data")
RAW_DATADIR = os.path.join(DEFAULT_DATADIR, "raw_data")

register(
    id="anyGPT/SequenceClassificationEnv-v0",
    entry_point="anyGPT.environments.sequence_classification_env:SequenceClassificationEnv",
    max_episode_steps=1,
)
