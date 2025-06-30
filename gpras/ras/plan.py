"""Tools for building, running, and extracting data from HEC-RAS models."""

from dataclasses import dataclass
from typing import Any, TypeVar

import h5py
import numpy as np
from numpy.typing import NDArray

FloatType = TypeVar("FloatType", bound=np.floating[Any])


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

        print(f"deleting {data_path}")
        del f[data_path]
        f.create_dataset(data_path, data=data)
