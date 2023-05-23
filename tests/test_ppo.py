import gymnasium
import pytest

from anyGPT.config.util import parse_config, config_to_settings
from anyGPT.models.anygpt import AnyGPT
from anyGPT.models.anygpt_ppo_lit import AnyGPTPPOLit
from anyGPT.models.ppo_policy import AnyGPTCritic

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
  checkpoint: 'tests/models/pre-trained-10M-char.pt'
  shared_actor_critic: true
  action_size: 8
  observation_size: 8
  batch_size: 2
  buffer_size: 4
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


def test_sample_trajectories(settings):
    ppo = AnyGPTPPOLit(settings)
    trajectories = ppo.sample_trajectories()
    assert trajectories is not None
    for trajectory in trajectories:
        assert trajectory is not None
        state, action, logp, lopg_ref, qval, adv = trajectory
        assert state.shape == (1, 8)
        assert action.shape == (1, 8)
        assert logp.shape == (1, 8)
        assert lopg_ref.shape == (1, 8)
        assert qval.shape == (1, 8)
        assert adv.shape == (1, 8)
