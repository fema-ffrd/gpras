"""Tools to load data and preprocess for GPR."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from gpras.consts import DEPTH_DIR_SUBPATH
from gpras.utils.common import load_raster


@dataclass
class TrainingData:
    """Training data for the GPR model."""

    data_dir: str

    @property
    def _depth_grid_dir(self) -> str:
        """Path to directory where depth grids are stored."""
        return str(Path(self.data_dir) / DEPTH_DIR_SUBPATH)

    @property
    def _depth_grid_paths(self) -> list[str]:
        """Scan directory for depth grids."""
        paths = []
        for i in Path(self._depth_grid_dir).iterdir():
            # TODO: add validation
            paths.append(str(i))
        return paths

    @property
    def _dem_path(self) -> str:
        """Path to the DEM."""
        return str(Path(self.data_dir) / DEPTH_DIR_SUBPATH)

    @cached_property
    def depth_grids(self) -> dict[str, np.ndarray]:
        """Numpy arrays for each depth grid."""
        return {i: load_raster(i) for i in self._depth_grid_paths}

    @cached_property
    def elevation_grid(self) -> np.ndarray:
        """Numpy array for the DEM."""
        return load_raster(self._dem_path)

    @property
    def inputs(self) -> np.ndarray:
        """Input features for GPR training."""
        all_data = []
        x_space = np.linspace(
            0, 1, self.elevation_grid.shape[0], endpoint=True
        )  # Normalized x and y
        y_space = np.linspace(0, 1, self.elevation_grid.shape[1], endpoint=True)
        x, y = np.meshgrid(x_space, y_space)
        for d in self.depth_grids:
            data = np.array(
                [
                    x.flatten(),
                    y.flatten(),
                    self.elevation_grid.flatten(),
                    self.depth_grids[d].flatten(),
                ]
            ).T
            all_data.append(data)
        return np.concatenate(all_data, axis=0)

    @property
    def outputs(self) -> np.ndarray:
        """Output targets for GPR training."""
        # TODO: Implement this


@dataclass
class ToyTrainingData(TrainingData):
    """Fake training data for the GPR model."""

    data_dir: str = ""
    resolution: int = 10000

    @cached_property
    def depth_grids(self) -> dict[str, np.ndarray]:
        """Numpy arrays for each depth grid."""
        out_dict = {}
        for i in range(2):
            out_dict[f"{i}.tif"] = self.elevation_grid + (
                i * 10
            )  # Constant depth for each grid
        return out_dict

    @cached_property
    def elevation_grid(self) -> np.ndarray:
        """Numpy array for the DEM."""
        x_space = np.linspace(0, 1, self.resolution, endpoint=True)
        y_space = np.linspace(0, 1, self.resolution, endpoint=True)
        x, y = np.meshgrid(x_space, y_space)
        d = np.max(np.stack([x, y]), axis=0)
        noise = (np.sin(x * 3 * np.pi) + np.sin(y * 5 * np.pi)) / 10
        return d + noise
