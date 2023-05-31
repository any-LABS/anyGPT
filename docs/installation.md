# Installation

>**_NOTE_**: It is recommended that you set up of a python virtual environment
using [mamba](https://mamba.readthedocs.io/en/latest/), [conda](https://docs.conda.io/en/latest/),
or [poetry](https://python-poetry.org/).
To install anyGPT:

```shell
$ pip install anyGPT
```

## Using Docker

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

### Mounting Volumes

It is recommended to mount a local directory into your container in order to 
share data between your local host and the container. This will allow you to 
save trained checkpoints, reuse datasets between runs and more.

```shell
$ docker run --gpus all -v /path/to/local/dir:/data -it anygpt
```

The above example mounts `/path/to/local/dir` to the `/data` directory in the container, and all data and changes are shared between them dynamically.

### Non interactive Docker

The above documentation explains how to run a Docker container with an interactive session of anyGPT. You can
also run anyGPT commands to completion using Docker by overriding the entrypoint

```shell
$ docker run --gpus=all -v /path/to/your/data:/data --entrypoint anygpt-run -it anygpt /data/test.ckpt "hello world"
```

The above command runs `anygpt-run` with the parameters `/data/test.ckpt "hello world"`

## Dependencies

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