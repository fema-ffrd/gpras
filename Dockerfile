FROM tensorflow/tensorflow:latest-jupyter AS base

WORKDIR /workspace

ADD . .

# Make development container
FROM base AS dev

RUN apt update

RUN apt install -y git

RUN pip install -e .[dev]

# Make production container
FROM base AS prod

RUN pip install .

