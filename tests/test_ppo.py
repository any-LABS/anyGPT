import gymnasium
import pytest
import lightning.pytorch as pl
import torch

from anyGPT.config.util import parse_config, config_to_settings
from anyGPT.models.anygpt import AnyGPT
from anyGPT.models.anygpt_ppo_lit import AnyGPTPPOLit
from anyGPT.models.ppo_policy import AnyGPTCritic, PPOPolicy

config_file = """
model_config:
  name: 'gpt-2-10M-char'
  block_size: 256
  dropout: 0.2
  embedding_size: 384
  num_heads: 6
  num_layers: 6
  vocab_size: 65
  bias: false

io_config:
  experiment_name: 'gpt-2-char-rl'
  dataset: 'shakespeare_karpathy_char'

ppo_config:
  checkpoint: 'results/gpt-2-char/version_0/checkpoints/last.ckpt'
  shared_actor_critic: true
  env_kwargs:
    label: "neutral"
    model_name: "j-hartmann/emotion-english-distilroberta-base"
    dataset: "shakespeare_karpathy_char"
"""


@pytest.fixture(scope="session")
def settings():
    config = parse_config(config_file)
    settings = config_to_settings(config)
    return settings


def test_init_from_checkpoint(settings):
    ppo = AnyGPTPPOLit(settings)
    actor_critic = ppo.policy
    assert actor_critic is not None
    actor = ppo.policy.actor
    assert actor is not None
    assert isinstance(actor, AnyGPT)
    critic = ppo.policy.critic
    assert critic is not None
    assert isinstance(critic, AnyGPTCritic)


def test_init_env(settings):
    ppo = AnyGPTPPOLit(settings)
    env = ppo.env
    assert isinstance(env, gymnasium.Env)


def test_training(settings):
    # policy = PPOPolicy(settings).to("cpu")
    # y = torch.tensor([2, 3], dtype=torch.long, device="cpu")[None, ...]
    # output = policy.generate(y, 10, 256, "cpu")
    # print(output)
    ppo = AnyGPTPPOLit(settings).to("cuda")
    # ppo.sample_trajectories()
    trainer = pl.Trainer(limit_train_batches=10, max_epochs=1)
    trainer.fit(ppo)
