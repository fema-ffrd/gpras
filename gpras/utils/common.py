"""Utilities shared across tasks."""

from typing import cast

import numpy as np
import rasterio  # type: ignore
from numpy.typing import NDArray


def load_raster(raster_path: str) -> NDArray[np.number]:
    """Load a raster into memory."""
    with rasterio.open(raster_path) as src:
        data = src.read()
    return cast(NDArray[np.number], data)
