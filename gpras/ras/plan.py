"""Tools for building, running, and extracting data from HEC-RAS models."""

from dataclasses import dataclass
from typing import Any, TypeVar

import h5py
import numpy as np
from hecstac.ras.assets import GenericAsset
from numpy.typing import NDArray

from gpras.utils.file_utils import detect_file_properties

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
    timesteps: NDArray[Any]

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


def update_hdf_attributes(hdf_path: str, attr_path: str, attrs: dict[str, str]) -> None:
    """Update the attributes of an hdf path."""
    with h5py.File(hdf_path, "r+") as f:
        hdf_attrs = f[attr_path].attrs
        for k, v in attrs.items():
            if isinstance(v, str):
                # Note that h5py has very weird behavior when writing bytes to an attribute.
                # np.string_ was the only way I could find to write information as bytes type
                hdf_attrs[k] = np.string_(v.encode())
            else:
                hdf_attrs[k] = v


def update_hdf_data(hdf_path: str, data_path: str, data: NDArray[np.float32]) -> None:
    """Overwrite data in an hdf file."""
    with h5py.File(hdf_path, "a") as f:
        # d = f[data_path]
        # d[:] = data
        print(f"deleting {data_path}")
        del f[data_path]
        f.create_dataset(data_path, data=data)


def update_text_attributes(txt_path: str, attrs: dict[str, str]) -> None:
    """Update an attribute in a ras text file."""
    encoding, newline = detect_file_properties(txt_path)
    with open(txt_path, encoding=encoding) as f:
        lines = f.readlines()
    for ind, i in enumerate(lines):
        splitted = i.split("=")
        if splitted[0] in attrs:
            splitted[-1] = attrs[splitted[0]] + "\n"
            lines[ind] = "=".join(splitted)
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
