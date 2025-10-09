"""Object-oriented representations of configuration and data objects."""

import json
from dataclasses import dataclass, field
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
    """Settings to control where and how the Gaussian Process Regression (GPR) model is fit.

    Each field defines a configurable parameter for training or evaluating
    the surrogate model. This class can be serialized to and from JSON.
    """

    # === Core Paths/IDs ===
    working_directory: str = field(
        metadata={
            "help": "Root directory for data, metrics, model files, and plots.",
            "example": "/workspaces/gpras/data/ras_upskill",
        }
    )
    hf_ras_stac_path: str = field(
        metadata={
            "help": "Path to the STAC item for the high-fidelity (HF) HEC-RAS model.",
            "example": "/workspaces/gpras/data/bridgeport_HF/bridgeport.stac.json",
        }
    )
    area_of_interest_path: str = field(
        metadata={
            "help": "Path to the area-of-interest (AOI) polygon geopackage.",
            "example": "/workspaces/gpras/data/shared/project_area.gpkg",
        }
    )
    event_plan_path: str = field(
        metadata={
            "help": "Path to the event plan json created during automated run creation (make_ras_runs.py).",
            "example": "/workspaces/gpras/data/shared/event_plans.json",
        }
    )
    mesh_id: str = field(
        metadata={
            "help": "Unique mesh identifier within the HF RAS model.",
            "example": "bridgeport_1",
        }
    )

    # === Model Setup ===
    hydraulic_parameter: HydraulicParameterType = field(
        metadata={
            "help": "Primary hydraulic variable being modeled.",
            "choices": ["wse", "depth", "velocity"],
            "example": "depth",
        }
    )
    wet_threshold_depth: float = field(
        metadata={
            "help": "Minimum depth threshold (m) used to classify always wet vs. always dry conditions.",
            "example": 0.5,
        }
    )
    spatial_mode_count: int = field(
        metadata={
            "help": "Number of spatial modes retained from dimensionality reduction.  If 0, North's rule is used",
            "example": 10,
        },
    )
    kernel: KernelType = field(
        metadata={
            "help": "Kernel used in the GPR model.",
            "choices": [
                "Matern12",
                "Matern32",
                "Matern52",
                "RBF",
                "Linear",
                "Polynomial",
                "Periodic",
                "Exponential",
            ],
            "example": "Matern32",
        }
    )
    inducing_pt_count: int = field(
        metadata={
            "help": "Number of inducing points used for sparse GPR approximation.",
            "example": 50,
        }
    )
    optimizer: OptimizerType = field(
        metadata={
            "help": "Optimization algorithm used for hyperparameter tuning.",
            "choices": ["two-stage", "adam", "L-BFGS-B", "stochastic", "diffential_evolution"],
            "example": "L-BFGS-B",
        }
    )
    induction_pt_initializer: InductionInitializerType = field(
        metadata={
            "help": "Strategy for initializing inducing points in the training domain.",
            "choices": ["kmeans", "grid"],
            "example": "kmeans",
        }
    )
    optimizer_kwargs: dict[str, Any] = field(
        metadata={
            "help": "Dictionary of keyword arguments passed to the optimizer.",
            "example": {"max_iter": 1000},
        }
    )

    # === Outputs ===
    generate_plots: bool = field(
        metadata={
            "help": "Whether to generate diagnostic plots during training and evaluation.",
            "example": True,
        }
    )
    save_dbs: bool = field(
        metadata={
            "help": "Whether to save model input/output databases (for faster reanalysis).",
            "example": True,
        }
    )

    # === Low-Fidelity (LF) Model ===
    lf_model_type: LFModelType = field(
        metadata={
            "help": "Type of low-fidelity model used.",
            "choices": ["ras_upskill", "pseudo_surface", "hms_upskill"],
            "example": "ras_upskill",
        }
    )
    lf_ras_stac_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to the STAC item for the LF RAS model (required for ras_upskill).",
            "example": "/workspaces/gpras/data/bridgeport_LF/bridgeport.stac.json",
        },
    )
    inflow_dss_dir: str | None = field(
        default=None,
        metadata={
            "help": "Directory containing HMS DSS files for each plan (required for hms_upskill and pseudo_surface).",
            "example": "data/bridgeport_HF/gpr_dss_files/flow_boundaries",
        },
    )
    inflow_hms_elements: list[list[str]] | None = field(
        default=None,
        metadata={
            "help": "List of HMS element names and data type to extract as features (required for hms_upskill and pseudo_surface).",
            "example": [["west-fork_s340", "FLOW"], ["west-fork_s340", "FLOW-BASE"], ["west-fork_s330", "FLOW"]],
        },
    )
    precip_dss_dir: str | None = field(
        default=None,
        metadata={
            "help": "Directory containing excess precipitation dss files for each plan (required for hms_upskill and pseudo_surface).",
            "example": "data/precip/",
        },
    )
    precip_spatial_mode_count: int = field(
        default=0,
        metadata={
            "help": "Number of spatial modes for the precipitation variable. (required for hms_upskill and pseudo_surface)",
            "example": 5,
        },
    )
    fluvial_lf_preprocessor_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to LF pre-processor from an HMS upskill model. (required for pseudo_surface)",
            "example": "/workspaces/gpras/data/hms_upskill/model/lf_preprocessor.pkl",
        },
    )
    fluvial_hf_preprocessor_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to HF pre-processor from an HMS upskill model. (required for pseudo_surface)",
            "example": "/workspaces/gpras/data/hms_upskill/model/hf_preprocessor.pkl",
        },
    )
    fluvial_gpr_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to a pre-trained fluvial GPR model file. (required for pseudo_surface)",
            "example": "/workspaces/gpras/data/hms_upskill/model/gpr.pkl",
        },
    )
    us_bc_id_ras: str | None = field(
        default=None,
        metadata={
            "help": "Upstream boundary condition ID in HEC-RAS model. Used for rating curve development. (required for pseudo_surface)",
            "example": "Inflow1",
        },
    )
    ds_bc_id_ras: str | None = field(
        default=None,
        metadata={
            "help": "Downstream boundary condition ID in RAS model. Used for rating curve development. (required for pseudo_surface)",
            "example": "West_Fork_S020_Inlet|bridgeport_1",
        },
    )
    us_bc_id_hms: str | None = field(
        default=None,
        metadata={
            "help": "Upstream boundary condition ID in RAS model. (required for pseudo_surface)",
            "example": "West_Fork_S020_Outlet|bridgeport_1",
        },
    )
    ds_bc_id_hms: str | None = field(
        default=None,
        metadata={
            "help": "Downstream boundary condition ID in HMS model. (required for pseudo_surface)",
            "example": "Outflow_HMS1",
        },
    )
    centerline_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to the river centerline file. (required for pseudo_surface)",
            "example": "data/centerline.shp",
        },
    )

    # === Miscellaneous ===
    cell_id_field: str = field(
        default="cell_id",
        metadata={
            "help": "Field name containing the unique cell IDs in the RAS mesh geodataframe.",
            "example": "cell_id",
        },
    )

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
