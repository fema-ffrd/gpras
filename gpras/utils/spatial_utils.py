"""Utilities for geospatial data."""

from typing import Any

from affine import Affine
from rasterio.transform import from_origin


def ras_hdf_precip_transform(p_attrs: dict[Any, Any]) -> Affine:
    """Convert the information from a HEC-RAS plan hdf file to a python transformation."""
    pixel_size = p_attrs["Raster Cellsize"]
    upper_left_x = p_attrs["Raster Left"]
    upper_left_y = p_attrs["Raster Top"]

    return from_origin(upper_left_x, upper_left_y, pixel_size, pixel_size)
