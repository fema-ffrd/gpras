FROM tensorflow/tensorflow:latest-jupyter AS base

WORKDIR /workspace

ADD . .

# hecdss is not on PyPI
RUN pip install -i https://test.pypi.org/simple/ hecdss
RUN python -c "from hecdss import download_hecdss"

# Make development container
FROM base AS dev

RUN apt update

RUN apt install -y git

RUN pip install -e .[dev]

# Make production container
FROM base AS prod

RUN pip install .

