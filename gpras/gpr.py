"""Methods for GPR training, tuning, and prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class GPRAS:
    """A Gaussian Process Regression emulator for HEC-RAS."""

    __model_type__: str = "BaseGPRAS"

    def __init__(self, hf_model: str):
        """Construct class."""
        # Load the high-fidelity model metadata
        self.hf_model_path = hf_model
        with open(self.hf_model_path) as f:
            self.hf_model = json.load(f)

        # GPR model definition
        self.scaling_method: dict[Any, Any] | None = None  # parameters of scaling function
        self.eof_mode_count: int | None = None  # spatial modes used from EOF analysis
        self.inducing_fraction: float | None = None  # percent of data for sparse GPR
        self.kernel: dict[Any, Any] | None = None  # parameters of the GPR kernel

        # Training/testing information
        self._runs: list[str] = []  # List of HEC-RAS run ids loaded as data
        self.training_runs: list[str] = []  # Subset of self.runs used for training the model
        self.gpflow_parameters: dict[Any, Any] | None = None  # parameters of a fitted GPFlow model
        self.testing_runs: list[str] = []  # Subset of self.runs used for testing the model
        self.testing_performance: dict[Any, Any] | None = None  # Summary of performance on testing runs
        self.validation_runs: list[str] = []  # Subset of self.runs used for validation the model
        self.validation_performance: dict[Any, Any] | None = None  # Summary of performance on testing runs

        # For the following data, there will be a multi index on ras run and timestep
        self.data_x: pd.DataFrame | None = None  # Unscaled input features. Columns=depends on subclass; rows=timesteps.
        self.data_y: pd.DataFrame | None = None  # Unscaled targets. Columns=cell WSE; rows=timesteps.
        self.scaled_data_x: pd.DataFrame | None = None  # Scaled input features.
        self.scaled_data_y: pd.DataFrame | None = None  # Scaled targets.
        self.reduced_data_x: pd.DataFrame | None = None  # EOF analysis decomposed scaled inputs.
        self.reduced_data_y: pd.DataFrame | None = None  # EOF analysis decomposed scaled targets.

    @classmethod
    def from_file(cls: type[GPRAS], json_path: str) -> type[GPRAS]:
        """Load model from json."""
        with open(json_path) as f:
            j = json.load(f)

        inst = cls(j["properties"]["hf_model"])
        for i in j["properties"]:
            setattr(inst, i, j["properties"][i])

        return cls

    def to_file(self, out_path: str) -> None:
        """Save model representation to a STAC item json."""
        idx = Path(out_path).stem
        properties = {
            "model_type": self.__model_type__,
            "hf_model": self.hf_model_path,
            "scaling_method": self.scaling_method,
            "eof_mode_count": self.eof_mode_count,
            "inducing_fraction": self.inducing_fraction,
            "kernel": self.kernel,
            "gpflow_parameters": self.gpflow_parameters,
            "runs": self.runs,
            "training_runs": self.training_runs,
            "testing_runs": self.testing_runs,
            "validation_runs": self.validation_runs,
            "testing_performance": self.testing_performance,
            "validation_performance": self.validation_performance,
        }
        out_json = {"id": idx, "properties": properties}

        with open(out_path, mode="w") as f:
            json.dump(out_json, f, indent=4)

    def build_model(self, kernel: dict[Any, Any], inducing_fraction: float) -> None:
        """Define the GPR model."""
        # TODO: write this section.
        self.kernel = kernel
        self.inducing_fraction = inducing_fraction
        # self.model = gpflow.models.GPR()

    @property
    def runs(self) -> list[str]:
        """List of HEC-RAS runs used in training, testing, and validation."""
        return self._runs

    @runs.setter
    def runs(self, val: list[str]) -> None:
        """Set the runs for this model, and load their data."""
        # Placeholder for subclass-specific methods.
        self._runs = val
        return

    def scale_data(self) -> None:
        """Apply the specified transformation to the data to generate self.scaled_data."""
        assert self.scaling_method is not None, "self.scaling_method must be defined prior to data scaling."
        # TODO: write scaling code

    def reduce_data(self) -> None:
        """Apply an EOF analysis to scaled data to reduce dimensionality."""
        assert self.eof_mode_count is not None, "self.eof_mode_count must be defined prior to data reduction."
        # TODO: write EOF code

    def split_training_testing_validation(self) -> None:
        """Split runs into training, testing and validation."""
        # Skeleton code.  Probably better to use a canned algorithm.
        # Ex. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        # Also, train-test split will be dynamic for k-fold CV

        # runs = [i["href"] for i in self.hf_model["assets"]]
        # self.testing_runs = sample(runs, 1)
        # runs = set(runs).difference(self.testing_runs)
        # self.testing_runs = sample(runs, 1)
        # self.validation_runs = list(set(runs).difference(self.testing_runs))

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Use GPR model to predict targets from inputs."""
        # TODO: write this code.
        return np.array([0])


class RasEmulator(GPRAS):
    """Emulate high-fidelity HEC-RAS from low-fidelity HEC-RAS."""

    __model_type__: str = "RasEmulator"

    def __init__(self, lf_model: str, hf_model: str):
        """Construct class."""
        super().__init__(hf_model)
        self.lf_model_path = lf_model
        with open(self.lf_model_path) as f:
            self.lf_model = json.load(f)

    @property
    def runs(self) -> list[str]:
        """Pass-through to access parent getter."""
        return super().runs

    @runs.setter
    def runs(self, val: list[str]) -> None:
        """Load runs from HEC-RAS."""
        pass


class InterpolationEmulator(GPRAS):
    """Emulate high-fidelity HEC-RAS from estimated WSEL grid."""

    __model_type__: str = "InterpolationEmulator"

    # TODO: Plan and implement approach #2


class HydrographEmulator(GPRAS):
    """Emulate high-fidelity HEC-RAS from a hydrograph."""

    __model_type__: str = "HydrographEmulator"

    # TODO: Plan and implement approach #3
