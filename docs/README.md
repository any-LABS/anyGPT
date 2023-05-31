# anyGPT

[![any-LABS - anyGPT](https://img.shields.io/static/v1?label=any-LABS&message=anyGPT&color=blue&logo=github)](https://github.com/any-LABS/anyGPT "Go to GitHub repo")
[![stars - anyGPT](https://img.shields.io/github/stars/any-LABS/anyGPT?style=social)](https://github.com/any-LABS/anyGPT)
[![forks - anyGPT](https://img.shields.io/github/forks/any-LABS/anyGPT?style=social)](https://github.com/any-LABS/anyGPT)
[![CI](https://github.com/any-LABS/anyGPT/workflows/CI/badge.svg)](https://github.com/any-LABS/anyGPT/actions?query=workflow:"CI")
[![GitHub tag](https://img.shields.io/github/tag/any-LABS/anyGPT?include_prereleases=&sort=semver&color=blue)](https://github.com/any-LABS/anyGPT/releases/)
[![License](https://img.shields.io/badge/License-MIT-blue)](#license)
[![issues - anyGPT](https://img.shields.io/github/issues/any-LABS/anyGPT)](https://github.com/any-LABS/anyGPT/issues)

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

#### Non interactive Docker

The above documentation explains how to run a Docker container with an interactive session of anyGPT. You can
also run anyGPT commands to completion using Docker by overriding the entrypoint

```shell
$ docker run --gpus=all -v /path/to/your/data:/data --entrypoint anygpt-run -it anygpt /data/test.pt "hello world"
```

The above command runs `anygpt-run` with the parameters `/data/test.pt "hello world"`

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
* Easy spin up using Docker
* FastAPI end-points for containerized microservice deployment
* HuggingFace integration
    * Load pre-trained gpt models

### Roadmap

* Documentation
* HuggingFace integration
    * push to hub
* Easy spin VM spinup/getting started with
    * Downloading of pre-trained models
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
$ anygpt-run results/gpt-2-pretrain/version_0/checkpoints/last.pt \
"JAQUES.
All the world’s a stage,
And all the men and women merely players;
They have their exits and their entrances,
And one man in his time plays many parts,"
```

### Inference Microservice

anyGPT supports running models as a hosted microservice with a singular endpoint for inference.
To launch the microservice, use the `anygpt-serve` entrypoint.

#### Commandline Options

```shell
$ anygpt-serve -h
usage: anyGPT inference service [-h] [--port PORT] [--log-level LOG_LEVEL] model

Loads an anyGPT model and hosts it on a simple microservice that can run inference over the network.

positional arguments:
  model                 Path t0 the trained model checkpoint to load

options:
  -h, --help            show this help message and exit
  --port PORT           Port to start the microservice on (default: 5000)
  --host HOST           Host to bind microservice to (default: 127.0.0.1)
  --log-level LOG_LEVEL
                        uvicorn log level (default: info)
```

#### Example

```shell
$ anygpt-serve results/gpt-2-pretrain/version_0/checkpoints/last.pt --port 5000 --log-level info
```

#### Sending Requests

`anygpt-serve` uses [FastAPI](https://fastapi.tiangolo.com/lo/#interactive-api-docs) to serve the microservice.
To see the available microservice api go to the  `/docs` endpoint in your browser once the microservice is started

#### Spinning up Microservice in Docker

```shell
$ docker run --gpus=all -v /path/to/your/data:/data -p 5000:5000 --entrypoint anygpt-serve -it anygpt /data/test.pt --port 5000 --host 0.0.0.0 --log-level info
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:5000 (Press CTRL+C to quit)
```


## Documentation

[![view - Documentation](https://img.shields.io/badge/view-Documentation-blue?style=for-the-badge)](https://any-labs.github.io/anyGPT/ "Go to project documentation")

## Limitations

TBD

## License

The goal of this project is to enable organizations, both large and small, to train and use GPT style
Large Language Models. I believe the future is open-source, with people and organizations being able to
train from scratch or fine-tune models and deploy to production without relying on gatekeepers. So I'm releasing this
under an [MIT license](../LICENSE) for the benefit of all and in the hope that the community will find it useful.

Released under [MIT](/LICENSE) by [@any-LABS](https://github.com/any-LABS).

[github_url]: https://github.com/miguelalonsojr/anyGPT/tree/main
[example-configs]: https://github.com/miguelalonsojr/anyGPT/tree/main/examples/config "Example configuration files."
