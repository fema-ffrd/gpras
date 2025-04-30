"""Utilities shared across tasks."""

import numpy as np
import rasterio


def load_raster(raster_path: str) -> np.ndarray:
    """Load a raster into memory."""
    with rasterio.open(raster_path) as src:
        data = src.read()
    return data
