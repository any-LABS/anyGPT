from dataclasses import dataclass


class SimpleConfig:
    def __init__(self, **kwargs):
        kwarg_keys = kwargs.keys()
        for key in self.__annotations__.keys():
            if key in kwarg_keys:
                self.__dict__[key] = kwargs[key]


@dataclass
class ModelConfig(SimpleConfig):
    name: str = 'gpt-2-124M'
    block_size: int = 1024
    vocab_size: int = 50304
    num_layers: int = 12
    num_heads: int = 12
    embedding_size: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    move_layer_norm: bool = True


@dataclass
class TrainingConfig(SimpleConfig):
    learning_rate: float = 1e-4
    batch_size: int = 8
    accumulate_gradients: int = 4
    swa_lrs: float = 1.0e-2
    max_epochs: int = 100
    val_check_interval: float = 0.25
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5


@dataclass
class IOConfig(SimpleConfig):
    dataset: str = 'princess_of_mars'
    out_dir: str = 'results'
    experiment_name: str = 'gpt-2-124M'
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = 'scratch'


@dataclass
class TorchConfig(SimpleConfig):
    backend: str = 'nccl'
    device: str = 'cuda'
    dtype: str = 'bfloat16'
    compile: bool = True


@dataclass
class AnyGPTSettings:
    model_config: ModelConfig
    training_config: TrainingConfig
    io_config: IOConfig
    torch_config: TorchConfig

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
