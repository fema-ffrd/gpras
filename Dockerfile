### Install common dependencies ###
FROM tensorflow/tensorflow:latest-jupyter AS base

WORKDIR /workspace

ADD . .

# hecdss is not on PyPI
RUN pip install -i https://test.pypi.org/simple/ hecdss
RUN python -c "from hecdss import download_hecdss"
RUN apt-get update
RUN apt-get install libgfortran5

### Make development container ###
FROM base AS dev

RUN apt update

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install

RUN aws --version

# Install git and python dev dependencies
RUN apt install -y git

RUN pip install -e .[dev]

### Make production container ###
FROM base AS prod

RUN pip install .

