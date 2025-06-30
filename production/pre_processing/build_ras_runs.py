"""Workflow script to generate ras runs from a template model and list of events."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import h5py
import numpy as np
from hecdss import HecDss
from numpy.typing import NDArray

from gpras.ras.model import RasModel
from gpras.ras.plan import EventCondition, FlowHydrographBC, PrecipitationBC
from gpras.utils.file_utils import hdf_2_dss_grid
from gpras.utils.s3_utils import s3_2_file

PRECIP_HDF_PATH = "/Event Conditions/Meteorology/Precipitation/Values"
FLOW_HYDROGRAPH_BCS_PATH_BASE = "2D: {} BCLine: {}"


def get_dss_data(dss_path: str, parameter: str, element_ids: list[str]) -> dict[str, dict[str, NDArray[Any]]]:
    """Load HEC-DSS data from s3."""
    with NamedTemporaryFile(suffix=".dss") as tmp_dss:
        local_dss = tmp_dss.name
        s3_2_file(dss_path, local_dss)
        tmp_dss.seek(0)
        summed_data = {}
        with HecDss(local_dss) as dss:
            catalog = dss.get_catalog()
            for e in element_ids:
                # TODO: Determine if USACE uses the three baseflow timeseries or just the first for FFRD
                path = [i for i in catalog if e == i.B and parameter == i.C][0]
                data = dss.get(str(path))
                summed_data[e] = {"q": data.values, "t": data.times}
                # paths = [i for i in catalog if e == i.B and i.C.startswith(parameter)]
                # data = [dss.get(i).values for i in paths]
                # min_shape = min([len(i) for i in data])
                # data = [i[-min_shape:] for i in data]
                # summed_data[e] = np.sum(data, axis=0)
    return summed_data


def get_hdf_precipitation(hdf_path: str) -> NDArray[np.float32]:
    """Load hdf data for precipitation."""
    with NamedTemporaryFile(suffix=".hdf") as tmp_dss:
        local_dss = tmp_dss.name
        s3_2_file(hdf_path, local_dss)
        tmp_dss.seek(0)
        with h5py.File(local_dss) as f:
            precipitation: NDArray[np.float32] = f[PRECIP_HDF_PATH][()]
    return precipitation


def build_boundary_conditions(
    ras_model: RasModel, template_run: str, event_dss: str, event_hdf: str
) -> list[EventCondition]:
    """Build boundary conditions for a set of model runs."""
    # Define precipitation and temperature forcing
    precipitation = get_hdf_precipitation(event_hdf)
    p_bc = PrecipitationBC(precipitation)

    # Define baseflows
    bcs = ras_model.assets[template_run].file.bc_lines
    subbasins = []
    paths = []
    for _, row in bcs.iterrows():
        if not row["name"].endswith("_base"):
            continue
        subbasins.append(row["name"].replace("_base", "").replace("bc_", ""))
        paths.append(FLOW_HYDROGRAPH_BCS_PATH_BASE.format(row["mesh_name"], row["name"]))
    dss_data = get_dss_data(event_dss, "FLOW-BASE", subbasins)
    baseflows = []
    for key, p in zip(dss_data, paths, strict=False):
        baseflows.append(FlowHydrographBC(dss_data[key]["q"], p, dss_data[key]["t"]))

    # base_flow_lines = [i.replace("_base", "").replace("bc_", "") for i in bcs["name"].values if i.endswith("_base")]
    # baseflows = get_dss_data(event_dss, "FLOW-BASE", base_flow_lines)
    # flows = [FlowHydrographBC(v["q"], f"bc_{k}_base", v["t"]) for k, v in baseflows.items()]

    # Define exterior inflows
    # TODO: Implement this.  Didn't have example model at the time of coding

    return [p_bc, *baseflows]


def make_attribute_dict(plan_id: str, boundary_conditions: list[EventCondition]) -> dict[str, Any]:
    """Build a dictionary to update plan attributes."""
    min_time = max([min(i.timesteps) for i in boundary_conditions if hasattr(i, "timesteps")])
    max_time = min([max(i.timesteps) for i in boundary_conditions if hasattr(i, "timesteps")])
    txt_sim_time = f"{min_time.strftime('%d%b%Y,%H%M').upper()},{max_time.strftime('%d%b%Y,%H%M').upper()}"
    hdf_start = min_time.strftime("%d%b%Y %H:%M:%S")
    hdf_end = max_time.strftime("%d%b%Y %H:%M:%S")
    hdf_sim_time = f"{hdf_start} to {hdf_end}"
    plan_dict = {
        "hdf": {
            "Plan Name": plan_id,
            "Plan ShortID": plan_id,
            "Plan Title": plan_id,
            "Simulation End Time": hdf_end,
            "Simulation Start Time": hdf_start,
            "Time Window": hdf_sim_time,
        },
        "txt": {"Short Identifier": plan_id, "Plan Title": plan_id, "Simulation Date": txt_sim_time},
    }
    return plan_dict


def copy_hms_data(event: dict[str, str], ras_prj_path: str, template_precip_dss: str) -> dict[str, str]:
    """Copy hms data from s3 to local."""
    # Define directory
    dss_dir = Path(ras_prj_path).parent / "dss"
    dss_dir.mkdir(exist_ok=True)
    baseflows_path = dss_dir / "flow_base" / f"flow_base_{event['id']}.dss"
    precip_path = dss_dir / "precip" / f"precip_{event['id']}.dss"
    # Move files
    s3_2_file(event["dss"], str(baseflows_path))
    with NamedTemporaryFile(suffix=".hdf") as tmp_hdf:
        local_hdf = tmp_hdf.name
        s3_2_file(event["hdf"], local_hdf)
        tmp_hdf.seek(0)
        hdf_2_dss_grid(local_hdf, template_precip_dss, PRECIP_HDF_PATH, str(precip_path))
    # Update event metadata
    event["baseflow_dss"] = str(baseflows_path)
    event["precip_dss"] = str(precip_path)
    return event


def build_runs(event_list: list[dict[str, str]], ras_model: RasModel, geometry_suffix: str) -> None:
    """Build new ras runs from a list of events."""
    for ind, event in enumerate(event_list):
        try:
            print(f"Building event for {event['id']}")
            # Make a new flow file
            # flow_suffix = generate_flow_file(ras_model, geometry_suffix, event["id"])

            # Make a new plan file
            # generate_plan_file(ras_model, geometry_suffix, flow_suffix, event["id"])

            # Update flow file

            bcs = build_boundary_conditions(ras_model, template_run, event["dss"], event["hdf"])
            attrs = make_attribute_dict(f"gpr{ind}", bcs)
            ras_model.make_new_plan(template_run, attrs, bcs)
        except Exception as e:
            print(f"Error on {event}")
            print(e)

    with open(ras_model.self_href, mode="w") as f:
        json.dump(ras_model.to_dict(), f, indent=4)


if __name__ == "__main__":
    event_path = "data/events.json"
    ras_path = "data/bridgeport/bridgeport.stac.json"
    template_run = "bridgeport.p01.hdf"
    geom_file = "g01"
    template_precip_dss = ""
    if not os.path.exists(ras_path):
        rm = RasModel.from_prj(ras_path.replace("stac.json", ".prj"))
        with open(ras_path, mode="w") as f:
            json.dump(rm.to_dict(), f, indent=4)
    with open(event_path) as f:
        events = json.load(f)
    with open(ras_path) as f:
        ras_model = RasModel.from_dict(json.load(f))
    for ind, event in enumerate(events):
        events[ind] = copy_hms_data(event, ras_model.self_href, template_precip_dss)
    build_runs(events, ras_model, geom_file)
