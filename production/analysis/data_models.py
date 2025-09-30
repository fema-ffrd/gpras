"""Object-oriented representations of configuration and data objects."""

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Self, TypedDict, cast

import geopandas as gpd
from shapely import Polygon

from gpras.gpr import InductionInitializerType, KernelType, OptimizerType
from gpras.preprocess import (
    DataBuilder,
    HmsPreProcessor,
    HmsUpskillDataBuilder,
    HydraulicParameterType,
    PreProcessor,
    PseudoSurfaceDataBuilder,
    RasInterpolaterBuilder,
    RasReader,
    RasUpskillDataBuilder,
)
from gpras.ras.model import RasModel

LFModelType = Literal["ras_upskill", "pseudo_surface", "hms_upskill"]


class EventPlan(TypedDict):
    """Metadata for a HEC-RAS plan."""

    plan_title: str
    event_number: int
    type: str
    set: str


@dataclass
class Config:
    """Settings to control where and how the GPR model is fit."""

    working_directory: str

    hf_ras_stac_path: str
    mesh_id: str
    cell_id_field: str
    area_of_interest_path: str
    event_plan_path: str

    hydraulic_parameter: HydraulicParameterType
    wet_threshold_depth: float
    spatial_mode_count: int

    kernel: KernelType
    inducing_pt_count: int
    optimizer: OptimizerType
    induction_pt_initializer: InductionInitializerType
    optimizer_kwargs: dict[Any, Any]

    generate_plots: bool
    save_dbs: bool
    save_preprocessor: bool

    lf_model_type: LFModelType
    lf_ras_stac_path: str | None = None
    inflow_dss_dir: str | None = None
    inflow_hms_elements: list[str] | None = None
    precip_dss_dir: str | None = None
    precip_spatial_mode_count: int | None = None
    fluvial_lf_preprocessor_path: str | None = None
    fluvial_hf_preprocessor_path: str | None = None
    fluvial_gpr_path: str | None = None
    us_bc_id_ras: str | None = None
    ds_bc_id_ras: str | None = None
    us_bc_id_hms: str | None = None
    ds_bc_id_hms: str | None = None
    centerline_path: str | None = None

    def __post_init__(self) -> None:
        """Create directories."""
        self.working_directory_path = Path(self.working_directory)
        self.plot_dir = self.working_directory_path / "plots"
        self.model_dir = self.working_directory_path / "model"
        self.metric_dir = self.working_directory_path / "metrics"
        self.testing_data_db = str(Path(self.working_directory) / "data" / "testing.db")
        self.training_data_db = str(Path(self.working_directory) / "data" / "training.db")
        self.model_path = self.model_dir / "gpr.pkl"
        self.hf_preprocessor_path = self.model_dir / "hf_preprocessor.pkl"
        if self.lf_model_type == "ras_upskill":
            self.lf_preprocessor_path = self.hf_preprocessor_path
        else:
            self.lf_preprocessor_path = self.model_dir / "lf_preprocessor.pkl"
        self.preprocessor_path = self.model_dir / "preprocessor.pkl"  # TODO: remove this
        self.timer_path = self.model_dir / "timers.json"
        self.metric_db_path = self.metric_dir / "performance_metrics.db"

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
        if self.lf_ras_stac_path is None:
            raise RuntimeError("Tried to call lf_ras property, but no lf_ras value was provided during initialization")
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
    def data_builder(self) -> type[DataBuilder]:
        """The specific data builder/extracter for the LF model type."""
        if self.lf_model_type == "ras_upskill":
            return RasUpskillDataBuilder
        elif self.lf_model_type == "pseudo_surface":
            return PseudoSurfaceDataBuilder
        elif self.lf_model_type == "hms_upskill":
            return HmsUpskillDataBuilder
        elif self.lf_model_type == "ras_interpolate":
            return RasInterpolaterBuilder

    @cached_property
    def data_reader(self) -> type[RasReader]:
        """The specific data reader for the LF model type."""
        if self.lf_model_type == "ras_upskill":
            return RasReader
        elif self.lf_model_type == "pseudo_surface" or self.lf_model_type == "hms_upskill":
            return RasReader  # TODO: Implement this
        else:
            return RasReader
            raise RuntimeError(f"No data reader available for LF model type '{self.lf_model_type}'")

    @cached_property
    def preprocessor(self) -> type[PreProcessor | HmsPreProcessor]:
        """The specific data builder/extracter for the LF model type."""
        if self.lf_model_type == "ras_upskill":
            return PreProcessor
        elif self.lf_model_type == "hms_upskill":
            return HmsPreProcessor
        else:
            raise RuntimeError(f"No preprocessor available for LF model type '{self.lf_model_type}'")
