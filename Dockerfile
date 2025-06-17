FROM tensorflow/tensorflow:latest-jupyter AS base

WORKDIR /workspace

ADD . .

RUN git clone https://github.com/HydrologicEngineeringCenter/hec-dss-python.git

RUN pip install hec-dss-python/

RUN python -c "from hecdss import download_hecdss"


# Make development container
FROM base AS dev

RUN apt update

RUN apt install -y git libgfortran5

RUN pip install -e .[dev]

# Make production container
FROM base AS prod

RUN pip install .

