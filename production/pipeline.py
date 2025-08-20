"""Example script for running a GPR model training workflow."""

import json
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Self, TypedDict, cast

import geopandas as gpd
import numpy as np
from shapely import Polygon

from gpras.gpr import GPRAS, KernelType
from gpras.metrics import export_metric_summary
from gpras.preprocess import PreProcessor, RasExtracter, RasReader
from gpras.ras.model import RasModel
from gpras.utils.plotting import ec_pairplot, ec_timeseries, performance_cdf, performance_scatterplot


class EventPlan(TypedDict):
    """Metadata for a HEC-RAS plan."""

    plan_title: str
    event_number: int
    type: str


@dataclass
class Config:
    """Settings to control where and how the GPR model is fit."""

    hf_ras_stac_path: str
    lf_ras_stac_path: str
    mesh_id: str
    area_of_interest_path: str
    event_plan_path: str
    wet_threshold_depth: float
    kernel: KernelType
    inducing_fraction: float
    working_directory: str
    cell_id_field: str
    depth: bool
    spatial_mode_count: int

    def __post_init__(self) -> None:
        """Create directories."""
        self.working_directory_path = Path(self.working_directory)
        self.plot_dir = self.working_directory_path / "plots"
        self.model_dir = self.working_directory_path / "model"
        self.metric_dir = self.working_directory_path / "metrics"

        Path(self.plot_dir).mkdir(exist_ok=True, parents=True)
        (Path(self.plot_dir) / "ec_timeseries").mkdir(exist_ok=True, parents=True)
        Path(self.model_dir).mkdir(exist_ok=True, parents=True)
        Path(self.metric_dir).mkdir(exist_ok=True, parents=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Initialize this class from a python dictionary."""
        return cls(**d)

    @classmethod
    def from_file(cls, fpath: str) -> Self:
        """Initiate this class from a json file."""
        with open(fpath) as f:
            d = json.load(f)
        return cls.from_dict(d)

    @cached_property
    def hf_ras(self) -> RasModel:
        """Representation of the high-fidelity HEC-RAS model."""
        with open(self.hf_ras_stac_path) as f:
            d = json.load(f)
        return cast(RasModel, RasModel.from_dict(d))

    @cached_property
    def lf_ras(self) -> RasModel:
        """Representation of the low-fidelity HEC-RAS model."""
        with open(self.lf_ras_stac_path) as f:
            d = json.load(f)
        return cast(RasModel, RasModel.from_dict(d))

    @cached_property
    def event_plan_json(self) -> list[EventPlan]:
        """Dictionary with details on the modeled event plans."""
        with open(self.event_plan_path) as f:
            event_plan_json = json.load(f)
        return cast(list[EventPlan], event_plan_json)

    @cached_property
    def train_plans(self) -> list[str]:
        """A list of HEC-RAS plans that should be used for training."""
        return [i["plan_title"] for i in self.event_plan_json if i["type"] == "Train"]

    @cached_property
    def test_plans(self) -> list[str]:
        """A List of HEC-RAS plans that should be used for testing."""
        return [i["plan_title"] for i in self.event_plan_json if i["type"] == "Test"]

    @cached_property
    def area_of_interest(self) -> Polygon:
        """A subset of the model domain that the GPR should be fit for."""
        return gpd.read_file(self.area_of_interest_path).to_crs(self.hf_ras.crs).iloc[0].geometry

    @cached_property
    def training_data_db(self) -> str:
        """Path to folder containing training data parquets."""
        return str(Path(self.working_directory) / "data" / "training.db")

    @cached_property
    def testing_data_db(self) -> str:
        """Path to folder containing testing data parquets."""
        return str(Path(self.working_directory) / "data" / "testing.db")


def main(config_path: str, testing: bool) -> None:
    """Automate the process of loading HEC-RAS model data, fitting the GPR, and making predictions."""
    # Load data
    t1 = time.perf_counter()
    print("Loading data")
    config = Config.from_file(config_path)
    if testing:  # Optional subsetting (faster)
        config.train_plans = config.train_plans[:2]
        config.test_plans = config.test_plans[:2]
    if not Path(config.training_data_db).exists():
        _extracter = RasExtracter(
            hf_ras=config.hf_ras,
            lf_ras=config.lf_ras,
            mesh_id=config.mesh_id,
            plans=config.train_plans,
            area_of_interest=config.area_of_interest,
            cell_id_field=config.cell_id_field,
        )
        _extracter.export_db(config.training_data_db)
    extracter = RasReader(config.training_data_db)
    hf_data_df, lf_data_df = extracter.aligned_datasets
    hf_data = hf_data_df.values
    lf_data = lf_data_df.values

    # Preprocess data
    t2 = time.perf_counter()
    print("Preprocessing data")
    reducer = PreProcessor(wet_threshold=config.wet_threshold_depth, depth=config.depth)
    reducer.fit(hf_data, extracter.cell_elevations, extracter.cell_areas, config.spatial_mode_count)
    reducer.plot_pca_summary(config.plot_dir / "pca_summary.png")
    y = reducer.transform(hf_data)
    x = reducer.transform(lf_data)

    ec_pairplot(x, y, 5, config.plot_dir / "pairplot.png")
    ec_timeseries(x, y, 5, hf_data_df.index, config.plot_dir / "ec_timeseries")

    # Fit GPR
    t3 = time.perf_counter()
    print("Fitting GPR")
    gpr = GPRAS(config.kernel)
    gpr.fit(x, y, config.inducing_fraction)
    gpr.to_file(config.model_dir / "gpr.json")

    # Load test data
    t4 = time.perf_counter()
    print("Loading and pre-processing test data")
    if not Path(config.testing_data_db).exists():
        _extracter = RasExtracter(
            config.hf_ras,
            config.lf_ras,
            config.mesh_id,
            config.test_plans,
            config.area_of_interest,
            config.cell_id_field,
            hf_resampler=extracter.hf_resampler,
            lf_resampler=extracter.lf_resampler,
        )
        _extracter.export_db(config.testing_data_db)
    test_extracter = RasReader(config.testing_data_db)
    hf_test_data_df, lf_test_data_df = test_extracter.aligned_datasets
    hf_test_data = hf_test_data_df.values
    lf_test_data = lf_test_data_df.values
    x_test = reducer.transform(lf_test_data)
    y_test = reducer.transform(hf_test_data)

    ec_pairplot(x_test, y_test, 5, config.plot_dir / "pairplot_test.png")

    # Predict test data
    t5 = time.perf_counter()
    print("Making predictions")
    mean_pred, _ = gpr.predict(x_test)
    y_test_pred = reducer.reverse_transform(mean_pred)
    if config.depth:
        y_test_pred += reducer.elevations

    # Assess performance
    t6 = time.perf_counter()
    print("Calculating metrics and making performance plots")
    export_metric_summary(hf_test_data, y_test_pred, config.metric_dir / "performance_metrics.db")
    performance_scatterplot(
        lf_test_data,
        hf_test_data,
        y_test_pred,
        config.plot_dir / "performance_scatterplot.png",
    )
    performance_cdf(
        lf_test_data,
        hf_test_data,
        y_test_pred,
        config.plot_dir / "performance_cdf.png",
    )
    ec_pairplot(mean_pred, y_test, 5, config.plot_dir / "pairplot_test_predicted.png")

    lf_test_data_depth = reducer.wse_2_depth(lf_test_data)
    hf_test_data_depth = reducer.wse_2_depth(hf_test_data)
    y_test_pred_depth = reducer.wse_2_depth(y_test_pred)
    performance_scatterplot(
        lf_test_data_depth,
        hf_test_data_depth,
        y_test_pred_depth,
        config.plot_dir / "performance_scatterplot_depth.png",
        depth=True,
    )

    # Print timing stats
    t7 = time.perf_counter()
    print(f"Finished whole process in {round(t6 - t1, 1)} seconds")
    print(f" - {round(t2 - t1, 1)} seconds to load data from {len(config.train_plans)} plans")
    print(f" - {round(t3 - t2, 1)} seconds to pre-process data")
    print(f" - {round(t4 - t2, 1)} seconds to fit the GPR")
    print(f" - {round(t5 - t4, 1)} seconds to load and pre-process data from {len(config.test_plans)} plans")
    print(f" - {round(t6 - t5, 1)} seconds to make predictions")
    print(f" - {round(t7 - t6, 1)} seconds to measure accuracy and plot results")

    # Make GIS layers
    print("Exporting GIS")
    if False:  # TODO: debug this
        hf_geom = extracter.hf_geometry_aoi
        lf_geom = extracter.lf_geometry_aoi
        hf_geom["lf_cell_id"] = extracter.lf_resampler
        hf_geom["wetness_class"] = reducer.wetness_classes
        hf_geom["lf_hf_rmse"] = np.mean((lf_test_data - hf_test_data) ** 2, axis=0) ** 0.5
        hf_geom["upskill_hf_rmse"] = np.mean((y_test_pred - hf_test_data) ** 2, axis=0) ** 0.5
        for i in range(reducer.eofs.shape[0]):
            x_full = np.empty(reducer.dry_indices.shape[0])
            x_full[reducer.dry_indices] = np.nan
            x_full[~reducer.dry_indices] = reducer.eofs[i, :]
            hf_geom[f"EOF_{i}"] = x_full
        hf_geom.to_file("qc.gpkg", layer="HF_cells")
        lf_geom.to_file("qc.gpkg", layer="LF_cells")


if __name__ == "__main__":
    main("production/configurations/0.1.0/gpr_training.config.json", False)
