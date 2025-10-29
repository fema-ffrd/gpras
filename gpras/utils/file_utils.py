"""Convenient functions for working with files."""

import os
import shutil
from datetime import datetime
from pathlib import PurePath, PurePosixPath
from urllib.parse import unquote, urlparse

import chardet
import h5py
import numpy as np
from hecdss import HecDss


def get_filename(path_str: str) -> str:
    """Get the trailing name of a file."""
    parsed = urlparse(path_str)
    if parsed.scheme == "s3":
        return PurePosixPath(unquote(parsed.path)).name
    return PurePath(path_str).name


def detect_file_properties(path: str, sample_size: int = 4096) -> tuple[str, str]:
    """Checks encoding and newline for files."""
    with open(path, "rb") as f:
        raw = f.read(sample_size)
    detected = chardet.detect(raw)
    encoding = detected["encoding"] or "utf-8"
    newline = None
    if b"\r\n" in raw:
        newline = "\r\n"
    elif b"\r" in raw:
        newline = "\r"
    else:
        newline = "\n"
    return encoding, newline


def hdf_2_dss_grid(hdf_path: str, hdf_data_path: str, dss_temp_path: str, out_path: str) -> None:
    """Copy a dss file and overwrite data with data from an hdf (only supports gridded data)."""
    # Get new data
    with h5py.File(hdf_path, "r") as f:
        data = f[hdf_data_path][:]

    # Copy template dss
    os.remove(out_path)
    shutil.copy(dss_temp_path, out_path)

    # Overwrite data
    with HecDss(out_path) as dss:
        catalog = dss.get_catalog()
        sorted_items = [str(i) for i in sorted(catalog, key=lambda x: datetime.strptime(x.D, "%d%b%Y:%H%M"))]
        record_1 = dss.get(sorted_items[0])
        shape = np.array((record_1.numberOfCellsY, record_1.numberOfCellsX))
        # TODO: Validate dimensions
        for i in range(len(sorted_items)):
            tmp_data = np.flipud(np.reshape(data[i, :], shape))
            tmp_dss = dss.get(sorted_items[i])
            tmp_dss.data = tmp_data
            dss.put(tmp_dss)
