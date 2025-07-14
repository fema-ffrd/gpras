"""Functions to automate the creation of HEC-RAS plans with FFRD data."""

import json
import os
import shutil
from dataclasses import MISSING, dataclass, fields
from datetime import datetime
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Self, cast

import h5py
import numpy as np
from hecdss import HecDss
from numpy.typing import NDArray

from gpras.ras.flow import FlowBoundaryCondition, UnsteadyFlowFile
from gpras.ras.model import RasModel
from gpras.ras.plan import PlanFile
from gpras.utils.s3_utils import s3_2_file


@dataclass
class Settings:
    """Configuration settings for generating HEC-RAS plans with FFRD data.

    Attributes:
        ras_model_stac (str): Path to the model STAC item JSON.
        precip_hdf_path (str): Path to the excess precipitation HDF file (local or S3).
        hdf_data_path (str): Internal path within the HDF file to the data.
        precip_dss_template_path (str): Path to a template DSS file for precipitation.
        flow_dss_path_src (str): Source path for HMS flow DSS data (local or S3).
        flow_template_suffix (str): Suffix of the unsteady flow file template.
        dss_dir (str): Directory name for storing DSS files. This will be place in the RAS model directory.
        flow_title (str): Title for the flow data.
        plan_title (str): Title for the simulation plan.
        plan_short_id (str): Short identifier for the plan.
        geom_file_suffix (str): Suffix for the geometry file used in the plan.
        computation_interval (str): Time interval for computations.
        output_interval (str): Time interval for output recording.
        instantaneous_interval (str): Interval for instantaneous results.
        mapping_interval (str): Interval for mapping outputs.
        flow_bc_dir (str, optional): Directory name for flow boundary conditions. Defaults to "flow_boundaries".
        precip_bc_dir (str, optional): Directory name for precipitation boundary conditions. Defaults to "precipitation_boundaries".
        start_time (datetime, optional): Start time of the simulation. This is filled in dynamically by the script after reading the dss files.
        end_time (datetime, optional): End time of the simulation. This is filled in dynamically by the script after reading the dss files.
        precip_dss_data_path (str, optional): Internal path where the precip data will be stored in the generated precip dss. Defaults to "//gpr/PRECIPITATION/{}/{}/RUN:SST/".
        precip_dss_start_path (str, optional): Initial DSS path for precipitation. This is filled in dynamically by the script after reading the dss files.
        flow_file_path (str, optional): Path to the generated flow file. This is filled in dynamically by the script depending on the existing flow files.
    """

    ras_model_stac: str
    precip_hdf_path: str
    hdf_data_path: str
    precip_dss_template_path: str
    flow_dss_path_src: str
    flow_template_path: str
    dss_dir: str
    flow_title: str
    plan_title: str
    plan_short_id: str
    geom_file_suffix: str
    computation_interval: str
    output_interval: str
    instantaneous_interval: str
    mapping_interval: str
    flow_bc_dir: str = "flow_boundaries"
    precip_bc_dir: str = "precipitation_boundaries"
    start_time: datetime | None = None
    end_time: datetime | None = None
    precip_dss_data_path: str = "//gpr/PRECIPITATION/{}/{}/RUN:SST/"
    precip_dss_start_path: str | None = None
    flow_file_path: str | None = None

    def __post_init__(self) -> None:
        """Load the ras model after initializing the dataclass."""
        with open("data/bridgeport/bridgeport.stac.json") as f:
            self.ras_model: RasModel = RasModel.from_dict(json.load(f))
        Path(self.flow_dss_path_absolute).parent.mkdir(exist_ok=True, parents=True)
        Path(self.precip_dss_path_absolute).parent.mkdir(exist_ok=True, parents=True)

    @classmethod
    def from_file(cls, path: str) -> Self:
        """Load configuration from a file."""
        with open(path) as f:
            data = json.load(f)

        # Light validation.  TODO: consider migrating to pydantic
        required_fields = {f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING}

        missing = required_fields - data.keys()
        if missing:
            raise KeyError(f"Missing required config key(s): {', '.join(missing)}")

        return cls(**data)

    @cached_property
    def ras_model_root(self) -> str:
        """Path to the directory containing the HEC-RAS model."""
        return str(Path(self.ras_model.pm.model_root_dir).resolve())

    @cached_property
    def flow_dss_path_absolute(self) -> str:
        """Absolute path where the flow dss will be copied to."""
        return str(Path(self.ras_model_root) / self.dss_dir / self.flow_bc_dir / f"{self.flow_title}.dss")

    @cached_property
    def precip_dss_path_absolute(self) -> str:
        """Absolute path where the precipitation dss will be copied to."""
        return str(Path(self.ras_model_root) / self.dss_dir / self.precip_bc_dir / f"{self.flow_title}.dss")

    @cached_property
    def flow_dss_path_relative(self) -> str:
        """Relative (to the RAS model) path where the flow dss will be copied to."""
        return f"./{self.dss_dir}/{self.flow_bc_dir}/{self.flow_title}.dss"

    @cached_property
    def precip_dss_path_relative(self) -> str:
        """Relative (to the RAS model) path where the precipitation dss will be copied to."""
        return f"./{self.dss_dir}/{self.precip_bc_dir}/{self.flow_title}.dss"


def add_run(settings: Settings) -> None:
    """Create and add flow and plan files to the RAS model."""
    flow = make_unsteady_flow_file(settings)
    settings.flow_file_path = settings.ras_model.add_text_file(flow)
    plan = make_plan_file(settings)
    settings.ras_model.add_text_file(plan)


def make_unsteady_flow_file(settings: Settings) -> UnsteadyFlowFile:
    """Generate an unsteady flow file class with flow boundary and precipitation conditions."""
    # Initialize flow file from template
    flow = UnsteadyFlowFile.from_file(settings.flow_template_path)
    flow.flow_title = settings.flow_title
    flow.file_description = ""

    # Add flow boundary conditions
    if os.path.exists(settings.flow_dss_path_absolute):
        os.remove(settings.flow_dss_path_absolute)
    copy_file_s3_or_local(settings.flow_dss_path_src, settings.flow_dss_path_absolute)
    flow = add_boundary_conditions_to_unsteady_flow(flow, settings)

    # Add precipitation boundary conditions
    hdf_2_dss(settings)
    flow = add_precipitation_to_unsteady_flow(flow, settings)

    return flow


def copy_file_s3_or_local(from_path: str, to_path: str) -> None:
    """Copy a file to a local destination from S3 or a local path."""
    if from_path.startswith("s3://"):
        s3_2_file(from_path, to_path)
    else:
        shutil.copy(from_path, to_path)


def clean_ffrd_bc(idx: str) -> str:
    """Make RAS boundary condition ID match SST.dss elemnt id format."""
    idx = idx.strip()
    if idx.startswith("bc_"):
        idx = idx[3:]
    if idx.endswith("_base"):
        idx = idx[:-5]
    return idx


def add_boundary_conditions_to_unsteady_flow(flow: UnsteadyFlowFile, settings: Settings) -> UnsteadyFlowFile:
    """Write flow boundary condition paths to an unsteady flow file."""
    # Open the DSS file from HMS
    dss = HecDss(settings.flow_dss_path_absolute)
    cat = dss.get_catalog()
    elements = [i.B for i in cat]

    # Iterate over boundary conditions and link data
    for i in flow.boundary_conditions.bcs:
        # Get element id
        if i.bc_line_id.strip() != "":
            ele_id = clean_ffrd_bc(i.bc_line_id)
            param = "FLOW-BASE"
        elif i.sa_2d_id.strip() != "":
            ele_id = clean_ffrd_bc(i.sa_2d_id)
            param = "FLOW"
        else:
            continue

        # Skip elements not in sst file
        if ele_id not in elements or not isinstance(i, FlowBoundaryCondition):
            continue

        # Find appropriate DSS record
        path = [j for j in cat if ele_id == j.B and param == j.C][0]

        # Set model start and end for later use
        if settings.start_time is None or settings.end_time is None:
            record = dss.get(path)
            dts = record.times
            settings.start_time = min(dts)
            settings.end_time = max(dts)
            if settings.start_time is None or settings.end_time is None:
                raise RuntimeError("Unable to determine start and end times from SST dss flow file records.")

        # Reformat path to handle USACE's dss formatting errors
        split = str(path).split("/")
        split[4] = f"{settings.start_time.strftime('%d%b%Y')}-{settings.end_time.strftime('%d%b%Y')}"
        path = "/".join(split)

        # Log DSS data in flow file
        i.dss_file = settings.flow_dss_path_relative
        i.dss_path = path

    return flow


def add_precipitation_to_unsteady_flow(flow: UnsteadyFlowFile, settings: Settings) -> UnsteadyFlowFile:
    """Write precipitation boundary condition path to an unsteady flow file."""
    flow.precipitation.dss_filename = settings.precip_dss_path_relative
    flow.precipitation.dss_filepath = settings.precip_dss_start_path
    return flow


def hdf_2_dss(settings: Settings) -> None:
    """Convert USACE HDF precipitation data to DSS format for HEC-RAS."""
    # Load excess precipitation data
    data = load_hdf_data_s3_or_local(settings.precip_hdf_path, settings.hdf_data_path)

    # Determine temporal resolution
    if settings.start_time is None or settings.end_time is None:
        raise ValueError("Both start_time and end_time must be set before calling hdf_2_dss.")
    td = settings.end_time - settings.start_time
    interval = td / data.shape[0]
    t_i = settings.start_time
    t_j = t_i + interval

    # Initialize the dss file to write to
    settings.precip_dss_start_path = settings.precip_dss_data_path.format(
        t_i.strftime("%d%b%Y:%H%M"), t_j.strftime("%d%b%Y:%H%M")
    )
    if os.path.exists(settings.precip_dss_path_absolute):
        os.remove(settings.precip_dss_path_absolute)
    shutil.copy(settings.precip_dss_template_path, settings.precip_dss_path_absolute)

    # Write data to the DSS file
    with HecDss(settings.precip_dss_path_absolute) as dss:
        # Identify grid shape
        catalog = dss.get_catalog()
        record_template = dss.get(next(iter(catalog)))
        shape = np.array((record_template.numberOfCellsY, record_template.numberOfCellsX))

        # Clean out old records
        for i in catalog:
            dss.delete(str(i))

        # Write new records
        for i in range(data.shape[0]):
            tmp_data = np.flipud(np.reshape(data[i, :], shape))
            record_template.data = tmp_data
            record_template.id = settings.precip_dss_data_path.format(
                t_i.strftime("%d%b%Y:%H%M"), t_j.strftime("%d%b%Y:%H%M")
            )
            dss.put(record_template)
            t_i += interval
            t_j += interval


def load_hdf_data_s3_or_local(hdf_path: str, hdf_data_path: str) -> NDArray[Any]:
    """Load HDF dataset from either an S3 path or a local file."""
    if hdf_path.startswith("s3://"):
        with TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "tmp.hdf")
            s3_2_file(hdf_path, tmp_path)
            with h5py.File(tmp_path, "r") as f:
                data = f[hdf_data_path][:]
    else:
        with h5py.File(hdf_path, "r") as f:
            data = f[hdf_data_path][:]
    return cast(NDArray[Any], data)


def make_plan_file(settings: Settings) -> PlanFile:
    """Generate a plan file that contains the generated flow file."""
    if settings.start_time is None or settings.end_time is None:
        raise ValueError("Both start_time and end_time must be set before calling make_plan_file.")
    if settings.flow_file_path is None:
        raise ValueError("Flow file must be generated before calling make_plan_file.")
    plan = PlanFile()
    plan.settings["Plan Title"] = settings.plan_title
    plan.settings["Short Identifier"] = settings.plan_short_id
    plan.settings["Simulation Date"] = (
        f"{settings.start_time.strftime('%d%b%Y,%H%M')},{settings.end_time.strftime('%d%b%Y,%H%M')}"
    )
    plan.settings["Geom File"] = settings.geom_file_suffix
    plan.settings["Flow File"] = settings.flow_file_path.split(".")[-1]
    plan.settings["Computation Interval"] = settings.computation_interval
    plan.settings["Output Interval"] = settings.output_interval
    plan.settings["Instantaneous Interval"] = settings.instantaneous_interval
    plan.settings["Mapping Interval"] = settings.mapping_interval
    return plan


def make_runs_from_selected_events(settings_path: str) -> None:
    """Create multiple simulation runs based on a base configuration and STAC item from event selection step."""
    with open(settings_path) as f:
        base_settings = json.load(f)

    with open(base_settings["events_stac_path"]) as f:
        events_stac = json.load(f)
    del base_settings["events_stac_path"]

    for ind, i in enumerate(events_stac["assets"]):
        asset = events_stac["assets"][i]
        base_settings["flow_dss_path_src"] = asset["href"]
        base_settings["precip_hdf_path"] = base_settings["flow_dss_path_src"].replace(
            "SST.dss", "exported-precip_trinity.p01.tmp.hdf"
        )
        base_settings["flow_title"] = f"gpr{ind}"
        base_settings["plan_title"] = f"gpr{ind}"
        base_settings["plan_short_id"] = f"gpr{ind}"
        settings = Settings(**base_settings)
        add_run(settings)


if __name__ == "__main__":
    settings = Settings(
        ras_model_stac="data/bridgeport/bridgeport.stac.json",
        precip_hdf_path="s3://trinity-pilot/conformance/simulations/event-data/9373/hydrology/exported-precip_trinity.p01.tmp.hdf",
        hdf_data_path="/Event Conditions/Meteorology/Precipitation/Values",
        precip_dss_template_path="exported-precip_trinity.dss",
        flow_dss_path_src="s3://trinity-pilot/conformance/simulations/event-data/9373/hydrology/SST.dss",
        flow_template_path="/workspaces/gpras/data/bridgeport/bridgeport.u01",
        dss_dir="gpr_dss_files",
        flow_title="gpr000",
        plan_title="gpr000",
        plan_short_id="gpr",
        geom_file_suffix="g03",
        computation_interval="10SEC",
        output_interval="1HOUR",
        instantaneous_interval="1HOUR",
        mapping_interval="20MIN",
    )
    add_run(settings)
