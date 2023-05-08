from anyGPT.config.settings import ModelConfig
from anyGPT.config.util import parse_config, config_to_settings

model_config_file = """
model_config:
    name: 'hello-gpt'
    num_layers: 3
    num_heads: 3
"""


def test_model_config():
    config = parse_config(model_config_file)
    settings = config_to_settings(config)
    assert settings.model_config.name == "hello-gpt"
    assert settings.model_config.num_layers == 3
    assert settings.model_config.num_heads == 3


training_config_file = """
training_config:
    learning_rate: 1.0e-3
    max_steps: 100
"""


def test_training_config():
    config = parse_config(training_config_file)
    settings = config_to_settings(config)
    assert settings.training_config.learning_rate == 1e-3
    assert settings.training_config.max_steps == 100


io_config_file = """
io_config:
    dataset: 'the_meaning_of_life'
    experiment_name: 'is 42'
"""


def test_io_config():
    config = parse_config(io_config_file)
    settings = config_to_settings(config)
    assert settings.io_config.dataset == "the_meaning_of_life"
    assert settings.io_config.experiment_name == "is 42"


torch_config_file = """
torch_config:
    backend: 'cpu'
"""


def test_torch_config():
    config = parse_config(torch_config_file)
    settings = config_to_settings(config)
    assert settings.torch_config.backend == "cpu"


full_config_file = """
model_config:
  name: 'gpt-2-30M'
  block_size: 256
  dropout: 0.2
  embedding_size: 384
  num_heads: 6
  num_layers: 6

training_config:
  learning_rate: 1.0e-3
  batch_size: 8
  accumulate_gradients: 8
  beta2: 0.99
  min_lr: 1.0e-4
  max_steps: 5000
  val_check_interval: 200
  limit_val_batches: 100

io_config:
  experiment_name: 'gpt-2'
  dataset: 'shakespeare_complete'
"""


def test_full_config():
    config = parse_config(full_config_file)
    settings = config_to_settings(config)
    assert settings.model_config.block_size == 256
    assert settings.training_config.batch_size == 8
    assert settings.io_config.experiment_name == "gpt-2"
