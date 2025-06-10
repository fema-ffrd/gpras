"""Tools for running and extracting data from HEC-HMS models."""

import numpy as np
from hecstac.hms.item import HMSModelItem
from numpy.typing import NDArray


class HMSModel(HMSModelItem):  # type: ignore[misc]
    """OOP representation of a HEC-HMS model."""

    def execute(self, plan: str) -> None:
        """Run a plan."""
        # TODO: write this code

    def get_excess_precipitation_grid(self, plan: str) -> NDArray[np.float64]:
        """Extract excess precipitation raster from an HMS run."""
        # TODO: write this code
        return np.array([0])
