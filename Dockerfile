FROM pytorch/pytorch AS base

WORKDIR /workspace

COPY pyproject.toml ./

# Make development container
FROM base AS dev

RUN apt update

RUN apt install -y git

COPY gpras/__init__.py gpras/__init__.py

RUN pip install -e .[dev]

# Make production container
FROM base AS prod

COPY ./gpras ./gpras

RUN pip install .

