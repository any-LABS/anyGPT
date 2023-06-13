# Table of Contents

* [anyGPT.config.settings](#anyGPT.config.settings)
  * [SimpleConfig](#anyGPT.config.settings.SimpleConfig)
    * [\_\_init\_\_](#anyGPT.config.settings.SimpleConfig.__init__)
    * [update](#anyGPT.config.settings.SimpleConfig.update)
  * [TorchConfig](#anyGPT.config.settings.TorchConfig)
    * [backend](#anyGPT.config.settings.TorchConfig.backend)
    * [device](#anyGPT.config.settings.TorchConfig.device)
    * [precision](#anyGPT.config.settings.TorchConfig.precision)
    * [compile](#anyGPT.config.settings.TorchConfig.compile)
    * [accelerator](#anyGPT.config.settings.TorchConfig.accelerator)
    * [devices](#anyGPT.config.settings.TorchConfig.devices)

<a id="anyGPT.config.settings"></a>

# anyGPT.config.settings

<a id="anyGPT.config.settings.SimpleConfig"></a>

## SimpleConfig Objects

```python
class SimpleConfig()
```

SimpleConfig - base class for config dataclass objects

<a id="anyGPT.config.settings.SimpleConfig.__init__"></a>

#### \_\_init\_\_

```python
def __init__(**kwargs)
```

Initializes dataclass based on keys in the yaml dict.

**Arguments**:

- `kwargs`: a dictionary sourced from a yaml config file.

<a id="anyGPT.config.settings.SimpleConfig.update"></a>

#### update

```python
def update(kwargs)
```

Updates the values in the dataclass.

**Arguments**:

- `kwargs`: dictionary of key/value pairs to update.

<a id="anyGPT.config.settings.TorchConfig"></a>

## TorchConfig Objects

```python
@dataclass
class TorchConfig(SimpleConfig)
```

Torch configuration.

<a id="anyGPT.config.settings.TorchConfig.backend"></a>

#### backend

Specifies which backend to use. Default='nccl'. Currently disabled.

<a id="anyGPT.config.settings.TorchConfig.device"></a>

#### device

Specifies which device to use. Default=cuda. Options are 'cpu', 'cuda'

<a id="anyGPT.config.settings.TorchConfig.precision"></a>

#### precision

Specifies the precision used during training. Acceptable values are 32, 16-mixed,
bf16-mixed, or 64.

<a id="anyGPT.config.settings.TorchConfig.compile"></a>

#### compile

Specifies whether to compile the model for training. Default=true

<a id="anyGPT.config.settings.TorchConfig.accelerator"></a>

#### accelerator

Specifies which accelerator to use. Default=auto.

<a id="anyGPT.config.settings.TorchConfig.devices"></a>

#### devices

Specifies how many or which GPUs to use. If a string, specifies which devices.
If an int, specifies how many to use. Default=-1 (use all available devices). Only used for RLHF
