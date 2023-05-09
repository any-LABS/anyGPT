# anyGPT

[![Build and Test](https://github.com/miguelalonsojr/anyGPT/actions/workflows/test.yaml/badge.svg)](https://github.com/miguelalonsojr/anyGPT/actions/workflows/test.yaml)

anyGPT is a general purpose library for training any type of GPT model. Support for gpt-1, gpt-2, and gpt-3 models.
Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) by [Andrej Karpathy](https://github.com/karpathy), the goal
of this project is to provide tools for the training and usage of GPT style large language models. The aim is to provide
a tool that is

* production ready
* easily configurable
* scalable
* free and open-source
* accessible by general software engineers and enthusiasts
* easily reproducible and deployable

You don't need a Ph.D. in Machine Learning or Natural Language Processing to use anyGPT.

## Installation

>**_NOTE_**: It is recommended that you set up of a python virtual environment
using [mamba](https://mamba.readthedocs.io/en/latest/), [conda](https://docs.conda.io/en/latest/),
or [poetry](https://python-poetry.org/).
To install anyGPT:

```shell
$ pip install anyGPT
```

### Using Docker

The Docker image supports GPU passthrough for training and inference. In order to enable GPU
passthrough please follow the guide for installing the [NVidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for your OS.

>**_NOTE_** On Windows you need to follow the guide to get [NVidia Container Toolkit](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) setup on WSL2. [Docker WSL2 Backend](https://docs.docker.com/desktop/windows/wsl/) is required.

Once NVidia Container Toolkit and Docker is setup correctly, build the Docker image

```shell
$ docker build -t anygpt .
```

Use the following command to login to the container interactively, and use
anygpt as if it was on your local host

```shell
$ docker run --gpus all -it anygpt
```

#### Mounting Volumes

It is recommended to mount a local directory into your container in order to 
share data between your local host and the container. This will allow you to 
save trained checkpoints, reuse datasets between runs and more.

```shell
$ docker run --gpus all -v /path/to/local/dir:/data -it anygpt
```

The above example mounts `/path/to/local/dir` to the `/data` directory in the container, and all data and changes are shared between them dynamically.

### Dependencies

* torch >= 2.0.0
* numpy
* transformers
* datasets
* tiktoken
* wandb
* tqdm
* PyYAML
* lightning
* tensorboard

## Features

### Current

* CLI and config file driven GPT training
* Supports CPU, GPU, TPU, IPU, and HPU
* Distributed training strategies for training at scale

### Roadmap

* Documentation
* HuggingFace integration
    * Load pre-trained gpt models
    * push to hub
* Easy spin VM spinup/getting started with
    * Downloading of pre-trained models
    * FastAPI end-points for containerized microservice deployment
    * Gradio ChatGPT style interface for testing and experimentation
* Fine-tuning of pre-trained models
* Reinforcement Learning from Human Feedback and Rules Base Reward Modeling for LLM alignment
* More dataformat support beyond hosted text files
*

## Usage

### Data Preparation

```shell
$ anygpt-prepare-data -n shakespeare_complete -u https://www.gutenberg.org/cache/epub/100/pg100.txt
```

### Training

Create a config file. In this example, I'll call it `gpt-2-30M.yaml`. You can also check out the [example configuration files][example-configs].

```yaml title="gpt-2-30M.yaml"
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
```

```shell
$ anygpt-train gpt-2-30M.yaml
```

### Inference

```shell
$ anygpt-run results/gpt-1/version_0/checkpoints/epoch=0-step=5000.ckpt \
"JAQUES.
All the world’s a stage,
And all the men and women merely players;
They have their exits and their entrances,
And one man in his time plays many parts,"
```

## Documentation

Coming soon!

## Limitations

TBD

## License

The goal of this project is to enable organizations, both large and small, to train and use GPT style
Large Language Models. I believe the future is open-source, with people and organizations being able to
train from scratch or fine-tune models and deploy to production without relying on gatekeepers. So I'm releasing this
under an [MIT license](../LICENSE) for the benefit of all and in the hope that the community will find it useful.

[github_url]: https://github.com/miguelalonsojr/anyGPT/tree/main
[example-configs]: https://github.com/miguelalonsojr/anyGPT/tree/main/examples/config "Example configuration files."
