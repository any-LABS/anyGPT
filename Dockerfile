FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install everything needed for python3.10 and pip
RUN apt update && \
    apt install --no-install-recommends -y build-essential software-properties-common curl && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.10 python3.10-distutils python3-pip && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy over necessary files to install deps
COPY setup.cfg ./util/gen_req.py ./anygpt-utils/

# Install deps
RUN python3.10 ./anygpt-utils/gen_req.py ./anygpt-utils/ > ./anygpt-utils/requirements.txt && \
    pip3 install -r ./anygpt-utils/requirements.txt

# Install package
COPY pyproject.toml setup.cfg ./anygpt/
COPY ./tests ./anygpt/tests
COPY ./examples ./anygpt/examples
COPY ./src ./anygpt/src

RUN pip3 install ./anygpt

ENTRYPOINT ["/bin/bash"]