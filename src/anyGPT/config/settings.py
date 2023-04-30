from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str = 'gpt-2'
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    num_layers: int = 12
    num_heads: int = 12
    embedding_size: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    def __post_init__(self):
        # post load validation should go here
        pass


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5

    def __post_init__(self):
        # post load validation should go here
        pass


@dataclass
class IOConfig:
    dataset: str = 'princess_of_mars'
    out_dir: str = 'results'
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = 'scratch'

    def __post_init__(self):
        # post load validation should go here
        pass


@dataclass
class TorchConfig:
    backend: str = 'nccl'
    device: str = 'cuda'
    dtype: str = 'bfloat16'
    compile: bool = True

    def __post_init__(self):
        # post load validation should go here
        pass


@dataclass
class AnyGPTSettings:
    model_config: ModelConfig
    training_config: TrainingConfig
    io_config: IOConfig
    torch_config: TorchConfig

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
