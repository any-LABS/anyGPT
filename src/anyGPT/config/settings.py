from dataclasses import dataclass
from typing import Union


class SimpleConfig:
    def __init__(self, **kwargs):
        kwarg_keys = kwargs.keys()
        for key in self.__annotations__.keys():
            if key in kwarg_keys:
                self.__dict__[key] = kwargs[key]

    def update(self, kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class ModelConfig(SimpleConfig):
    name: str = "gpt-2-124M"
    block_size: int = 1024
    vocab_size: int = 50257
    num_layers: int = 12
    num_heads: int = 12
    embedding_size: int = 768
    dropout: float = 0.2
    bias: bool = True
    move_layer_norm: bool = True
    adapter_bottleneck_factor: int = 48
    fine_tune: bool = False


@dataclass
class TrainingConfig(SimpleConfig):
    learning_rate: float = 6e-4
    batch_size: int = 8
    accumulate_gradients: int = 8
    swa_lrs: float = 6e-4
    max_steps: int = 5000
    limit_train_batches: Union[float, int] = 1.0
    limit_val_batches: Union[float, int] = 1.0
    limit_test_batches: Union[float, int] = 1.0
    val_check_interval: Union[float, int] = 100
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 100
    min_lr: float = 6e-5
    init_from: str = "scratch"
    seed: int = 42


@dataclass
class IOConfig(SimpleConfig):
    dataset: str = "shakespeare_karpathy"
    out_dir: str = "results"
    experiment_name: str = "anygpt"
    log_every_n_steps: int = 10
    enable_checkpointing: bool = True


@dataclass
class TorchConfig(SimpleConfig):
    backend: str = "nccl"
    device: str = "cuda"
    precision: str = "16-mixed"  # 32, 16-mixed, bf16-mixed, 64
    compile: bool = True
    accelerator: str = "auto"


@dataclass
class PPOConfig(SimpleConfig):
    learning_rate: float = 1e-3
    lamda: float = 0.95
    gamma: float = 1.0
    max_episode_len: int = 1
    batch_size: int = 16
    buffer_size: int = 128
    clip_ratio: float = 0.2
    beta: float = 5e-3
    beta_kl: float = 1e-2
    epochs: int = 4
    num_optim_iters: int = 4
    env: str = "anyGPT/SequenceClassificationEnv-v0"
    env_kwargs: dict = None
    checkpoint: str = None
    shared_actor_critic: bool = False
    action_size: int = 512
    observation_size: int = 512
    scale_critic_loss: bool = True


@dataclass
class AnyGPTSettings:
    model_config: ModelConfig
    training_config: TrainingConfig
    io_config: IOConfig
    torch_config: TorchConfig
    ppo_config: PPOConfig

    def __init__(self, **kwargs):
        kwarg_keys = kwargs.keys()
        for key in self.__annotations__.keys():
            if key not in kwarg_keys:
                self.__dict__[key] = self.__annotations__[key]()
            else:
                self.__dict__[key] = self.__annotations__[key](**kwargs[key])

    def __post_init__(self):
        # post load validation should go here
        if isinstance(self.model_config, dict):
            self.model_config = ModelConfig(**self.model_config)
        if isinstance(self.training_config, dict):
            self.training_config = TrainingConfig(**self.training_config)
        if isinstance(self.io_config, dict):
            self.io_config = IOConfig(**self.io_config)
        if isinstance(self.torch_config, dict):
            self.torch_config = TorchConfig(**self.torch_config)
        if isinstance(self.ppo_config, dict):
            self.ppo_config = PPOConfig(**self.ppo_config)
