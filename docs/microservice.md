# Inference Microservice

anyGPT supports running models as a hosted microservice with a singular endpoint for inference.
To launch the microservice, use the `anygpt-serve` entrypoint.

## Commandline Options

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

## Example

```shell
$ anygpt-serve results/gpt-2-pretrain/version_0/checkpoints/last.pt --port 5000 --log-level info
```

## Sending Requests

`anygpt-serve` uses [FastAPI](https://fastapi.tiangolo.com/lo/#interactive-api-docs) to serve the microservice.
To see the available microservice api go to the  `/docs` endpoint in your browser once the microservice is started

## Spinning up Microservice in Docker

```shell
$ docker run --gpus=all -v /path/to/your/data:/data -p 5000:5000 --entrypoint anygpt-serve -it anygpt /data/test.pt --port 5000 --host 0.0.0.0 --log-level info
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:5000 (Press CTRL+C to quit)
```