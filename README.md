# Surrogate Modeling of HEC-RAS using Gaussian Process Regression
[![Build](https://img.shields.io/github/actions/workflow/status/fema-ffrd/gpras/ci.yaml?branch=main)](.github/workflows/ci.yml)
[![License](https://img.shields.io/github/license/fema-ffrd/gpras)](LICENSE)
[![Release](https://img.shields.io/github/v/release/fema-ffrd/gpras)](https://github.com/fema-ffrd/gpras/releases)
[![Issues](https://img.shields.io/github/issues/fema-ffrd/gpras)](https://github.com/fema-ffrd/gpras/issues)
![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Linter: Ruff](https://img.shields.io/badge/linter-ruff-orange)

<div align="center">
    <img src="./images/logo.png" alt="gpras Logo" width="250"/>
</div>

## Overview

HEC-RAS is a widely used tool for modeling river hydraulics and flood events; however, its detailed computational simulations can be time-consuming and resource-intensive, especially when performing stochastic simulation with many thousands of storm events. This software provides research- and production-level tooling to emulates HEC-RAS outputs with significantly reduced computation time via Gaussian Process Regression (GPR).  The GPR surrogate models may be used to predict flood depths from a variety of input configurations, including

1. Lower-fidelity HEC-RAS models (models with coarse grid resolution),
2. HEC-HMS reach-level stage or discharge hydrographs, and
3. Reach inflow hydrograph features.

The surrogate models are trained to reproduce the outputs of a high-resolution "benchmark" model that must be developed by the user beforehand. Parameters of the GPR surrogate are optimized to predict the benchmark flooding given a set of input features relating to either the benchmark model forcing or outputs from a lower-fidelity HEC-RAS model. Graphical summaries of the training and predicting process is shown in the images below.


## Installation

> [!NOTE]
> The most current code is on the 'dev' branch

To use this software, please use either the [devcontainer](./.devcontainer/devcontainer.json) or clone this repo and install with pip.


```bash
git clone https://github.com/fema-ffrd/gpras.git
```

```bash
pip install .
```

## Overview of gpras Formulations

### Upskill Low-Fidelity HEC-RAS
<div align='center'>
    <img src="./images/process_1.jpg" alt="training_testing" width="600"/>
</div>

<div align='center'>
    <img src="./images/process_1_fit.jpg" alt="training_testing" width="600"/>
</div>

<div align='center'>
    <img src="./images/process_1_pred.jpg" alt="training_testing" width="600"/>
</div>

### Upskill HEC-HMS Indirect
<div align='center'>
    <img src="./images/process_2.jpg" alt="training_testing" width="600"/>
</div>

### Upskill HEC-HMS Direct
<div align='center'>
    <img src="./images/process_3.jpg" alt="training_testing" width="600"/>
</div>

## Overview of Production Workflows

### Pipeline
<div align='center'>
    <img src="./images/workflow.jpg" alt="training_testing" width="600"/>
</div>

### Pre-processing
<div align='center'>
    <img src="./images/pre_process.jpg" alt="training_testing" width="600"/>
</div>
