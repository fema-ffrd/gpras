"""Tools for building, running, and extracting data from HEC-RAS models."""

import shutil
from dataclasses import dataclass
from typing import Any, TypeVar

import h5py
import numpy as np
from hecstac.ras.assets import GenericAsset
from hecstac.ras.item import RASModelItem
from numpy.typing import NDArray

from gpras.utils.file_utils import detect_file_properties, get_filename

FloatType = TypeVar("FloatType", bound=np.floating[Any])
GenAsset = TypeVar("GenAsset", bound=GenericAsset)


@dataclass
class EventCondition:
    """Generic event condition class."""

    data: NDArray[np.float32]

    @property
    def path(self) -> str:
        """The type-dependent path of the bc data."""
        return "Event Conditions"


@dataclass
class FlowHydrographBC(EventCondition):
    """An unsteady flow hydrograph boundary condition."""

    idx: str

    @property
    def path(self) -> str:
        """The type-dependent path of the bc data."""
        return f"/Event Conditions/Unsteady/Boundary Conditions/Flow Hydrographs/{self.idx}"


@dataclass
class PrecipitationBC(EventCondition):
    """A meteorology precipitation boundary condition."""

    @property
    def path(self) -> str:
        """The type-dependent path of the bc data."""
        return "/Event Conditions/Meteorology/Precipitation/Values"


@dataclass
class TemperatureBC(EventCondition):
    """A meteorology temperature boundary condition."""

    @property
    def path(self) -> str:
        """The type-dependent path of the bc data."""
        return "/Event Conditions/Meteorology/Temperature/Values"


BoundaryType = TypeVar("BoundaryType", bound=EventCondition)


class RasModel(RASModelItem):  # type: ignore[misc]
    """Subclass of hecstac with enhanced read/write/execute."""

    def make_new_plan(
        self, template_run: str, plan_attrs: dict[str, str], boundary_conditions: list[BoundaryType]
    ) -> str:
        """Append a new plan to the model."""
        # Define paths
        src_path: str = self.assets[template_run].href
        new_run = self.increment_suffix(self.plan_assets)
        dst_path: str = src_path.replace(template_run, new_run)
        src_txt_path: str = src_path.replace(".hdf", "")
        dst_txt_path: str = dst_path.replace(".hdf", "")

        # Copy file
        shutil.copy(src_path, dst_path)
        shutil.copy(src_txt_path, dst_txt_path)

        # Update metadata
        update_hdf_attributes(dst_path, "Plan Data/Plan Information", plan_attrs)
        update_text_attributes(dst_txt_path, {"Short Identifier": plan_attrs.get("Plan ShortID", "")})

        # Update project file
        add_plan_to_text_file(self.pf.fpath, dst_txt_path.split(".")[-1])

        # Write data
        for i in boundary_conditions:
            update_hdf_data(dst_path, i.path, i.data)

        # Add assets
        self.add_asset(get_filename(dst_path), dst_path)
        self.add_asset(get_filename(dst_txt_path), dst_txt_path)
        return dst_path

    def increment_suffix(self, paths: list[GenAsset]) -> str:
        """Take a list of paths, find the highest-numbered, and return that file incremented by 1."""
        suffixes = {int(get_filename(i.href).split(".")[-1].lstrip("p")): get_filename(i) for i in paths}
        max_plan_ind = max(suffixes.keys())
        new_plan_ind = max_plan_ind + 1
        max_suffix = f"p{str(max_plan_ind).zfill(2)}"
        new_suffix = f"p{str(new_plan_ind).zfill(2)}"
        return suffixes[max_plan_ind].replace(max_suffix, new_suffix)


def update_hdf_attributes(hdf_path: str, attr_path: str, attrs: dict[str, str]) -> None:
    """Update the attributes of an hdf path."""
    with h5py.File(hdf_path, "r+") as f:
        hdf_attrs = f[attr_path].attrs
        for k, v in attrs.items():
            hdf_attrs[k] = v.encode()


def update_hdf_data(hdf_path: str, data_path: str, data: NDArray[np.float32]) -> None:
    """Overwrite data in an hdf file."""
    with h5py.File(hdf_path, "r+") as f:
        f[data_path] = data


def update_text_attributes(txt_path: str, attrs: dict[str, str]) -> None:
    """Update an attribute in a ras text file."""
    encoding, newline = detect_file_properties(txt_path)
    with open(txt_path, encoding=encoding) as f:
        lines = f.readlines()
    for i in lines:
        splitted = i.split("=")
        if splitted[0] in attrs:
            splitted[-1] = attrs[i]
            i = "=".join(splitted)
    with open(txt_path, mode="w", encoding=encoding, newline=newline) as f:
        f.writelines(lines)


def add_plan_to_text_file(txt_path: str, plan_suffix: str) -> None:
    """Add a plan suffix to a project file."""
    encoding, newline = detect_file_properties(txt_path)
    with open(txt_path, encoding=encoding) as f:
        lines = f.readlines()
    last_index = max((ind for ind, line in enumerate(lines) if line.startswith("Plan File")), default=len(lines))
    lines.insert(last_index + 1, f"Plan File={plan_suffix}\n")
    with open(txt_path, mode="w", encoding=encoding, newline=newline) as f:
        f.writelines(lines)
