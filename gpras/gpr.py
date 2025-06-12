"""Methods for GPR training, tuning, and prediction."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any, TypeVar

import geopandas as gpd
import numpy as np
import pandas as pd
from hecstac.ras.assets import GeometryHdfAsset, PlanHdfAsset
from hecstac.ras.item import RASModelItem
from numpy.typing import NDArray

T = TypeVar("T", bound="GPRAS")


class GPRAS:
    """A Gaussian Process Regression emulator for HEC-RAS."""

    __model_type__: str = "BaseGPRAS"

    def __init__(self, hf_model: str, hf_runs: list[str], hf_mesh_id: str):
        """Construct class."""
        # Load the high-fidelity model metadata
        self.hf_model_path = hf_model
        with open(self.hf_model_path) as f:
            self.hf_model = RASModelItem.from_dict(json.load(f))
        self.hf_runs = hf_runs
        self.hf_mesh_id = hf_mesh_id

        # GPR model definition
        self.mesh_id: str | None = None  # The mesh id that will be emulated
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
        # self.data_x: pd.DataFrame | None = None  # Unscaled input features. Columns=depends on subclass; rows=timesteps.
        # self.data_y: pd.DataFrame | None = None  # Unscaled targets. Columns=cell WSE; rows=timesteps.
        # self.scaled_data_x: pd.DataFrame | None = None  # Scaled input features.
        # self.scaled_data_y: pd.DataFrame | None = None  # Scaled targets.
        # self.reduced_data_x: pd.DataFrame | None = None  # EOF analysis decomposed scaled inputs.
        # self.reduced_data_y: pd.DataFrame | None = None  # EOF analysis decomposed scaled targets.

    @classmethod
    def from_file(cls: type[T], json_path: str) -> T:
        """Load model from json."""
        with open(json_path) as f:
            j = json.load(f)

        return cls.from_dict(j)

    @classmethod
    def from_dict(cls: type[T], dict_rep: dict[Any, Any]) -> T:
        """Load model from dictionary."""
        inst = cls(**dict_rep)
        return inst

    @cached_property
    def plans(self) -> dict[str, PlanHdfAsset]:
        """Get a dictionary mapping plan name to PlanHdfAsset."""
        return {
            i.extra_fields["HEC-RAS:plan_information_plan_name"]: i.file.hdf_object
            for i in self.hf_model.assets.values()
            if isinstance(i, PlanHdfAsset)
        }

    @cached_property
    def geometries(self) -> dict[str, GeometryHdfAsset]:
        """Get a dictionary mapping geometry name to GeometryHdfAsset."""
        return {
            i.file.hdf_object.get_geom_attrs()["Title"]: i.file.hdf_object
            for i in self.hf_model.assets.values()
            if isinstance(i, GeometryHdfAsset)
        }

    @cached_property
    def geometry_hf(self) -> gpd.GeoDataFrame:
        """Get polygons from the high-fidelity geometry."""
        meshes = self._get_mesh_geometry_from_plans(self.hf_runs)
        return meshes[meshes["mesh_name"] == self.hf_mesh_id]

    def _get_mesh_geometry_from_plans(self, runs: list[str]) -> gpd.GeoDataFrame:
        """Get mesh polygons used to create a set of plans."""
        geoms = [j.get_attrs(j.PLAN_INFO_PATH)["Geometry Title"] for i, j in self.plans.items() if i in runs]
        assert all(i == geoms[0] for i in geoms), "Multiple geometries found in the hf model runs."
        geom = self.geometries[geoms[0]]
        return geom.mesh_cell_polygons()

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
            "hf_runs": self.hf_runs,  # TODO: find way to include lf_runs here
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

    @cached_property
    def data_y(self) -> pd.DataFrame:
        """WSE information from the HF model."""
        return self._extract_plan_wsels(self.hf_runs)

    def _extract_plan_wsels(self, runs: list[str]) -> pd.DataFrame:
        """Extract water surface elevations from a HEC-RAS plan at each computational cell."""
        run_store = []
        for ind, r in enumerate(runs):
            f = self.plans[r]
            ts = f.mesh_timeseries_output(self.hf_mesh_id, "Water Surface").values
            df = pd.DataFrame(ts)
            df["run"] = ind
            df["t"] = df.index.to_list()
            run_store.append(df)
        all_runs = pd.concat(run_store)
        all_runs = all_runs.set_index(["run", "t"])
        return all_runs

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

    def fit(self) -> None:
        """Fit the GPR model."""
        # TODO: Implement this.


class RasEmulator(GPRAS):
    """Emulate high-fidelity HEC-RAS from low-fidelity HEC-RAS."""

    __model_type__: str = "RasEmulator"

    def __init__(self, ras_model_path: str, hf_runs: list[str], lf_runs: list[str], hf_mesh_id: str, lf_mesh_id: str):
        """Construct class."""
        super().__init__(ras_model_path, hf_runs, hf_mesh_id)
        self.hf_runs = hf_runs
        self.lf_runs = lf_runs
        self.hf_mesh_id = hf_mesh_id
        self.lf_mesh_id = lf_mesh_id

    @cached_property
    def geometry_lf(self) -> gpd.GeoDataFrame:
        """Get polygons from the low-fidelity geometry."""
        meshes = self._get_mesh_geometry_from_plans(self.lf_runs)
        return meshes[meshes["mesh_name"] == self.lf_mesh_id]

    @cached_property
    def cell_resampler(self) -> dict[str, str]:
        """A dictionary that spatially joins lf and hf cells."""
        mesh_resampled = gpd.overlay(self.geometry_hf, self.geometry_lf[["cell_id", "geometry"]], how="intersection")
        mesh_resampled["area"] = mesh_resampled.geometry.area
        mesh_resampled = mesh_resampled.sort_values(by="area")
        mesh_resampled = mesh_resampled.drop_duplicates(subset="cell_id_1", keep="last")
        mesh_resampled = mesh_resampled[["cell_id_1", "cell_id_2"]]
        return dict(zip(mesh_resampled["cell_id_1"], mesh_resampled["cell_id_2"], strict=True))

    @cached_property
    def data_x(self) -> pd.DataFrame:
        """WSE information from the LF model resampled to hf dimensions."""
        d = self._extract_plan_wsels(self.lf_runs)
        d_resample = pd.DataFrame({col: d[self.cell_resampler[col]] for col in self.data_y.columns})
        return d_resample


class InterpolationEmulator(GPRAS):
    """Emulate high-fidelity HEC-RAS from estimated WSEL grid."""

    __model_type__: str = "InterpolationEmulator"

    # TODO: Plan and implement approach #2


class HydrographEmulator(GPRAS):
    """Emulate high-fidelity HEC-RAS from a hydrograph."""

    __model_type__: str = "HydrographEmulator"

    # TODO: Plan and implement approach #3
