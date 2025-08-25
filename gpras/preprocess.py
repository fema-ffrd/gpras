"""Tools to wrangle HEC-RAS data into a format usable by the gaussian process regression model."""

from functools import cached_property
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from shapely import Polygon
from sklearn.decomposition import PCA, IncrementalPCA

from gpras.ras.model import RasModel


class RasExtracter:
    """Convenience class for extracting data from RAS models and aligning high and low fidelity model data."""

    def __init__(
        self,
        hf_ras: RasModel,
        lf_ras: RasModel,
        mesh_id: str,
        plans: list[str],
        area_of_interest: Polygon,
        cell_id_field: str = "cell_id",
        flow_convergence_threshold: float = 0.95,
        cutoffs: dict[str, int] | None = None,
        hf_resampler: NDArray[Any] | None = None,
        lf_resampler: NDArray[Any] | None = None,
    ):
        """Construct class."""
        self.hf_ras = hf_ras
        self.lf_ras = lf_ras
        self.mesh_id = mesh_id
        self.plans = plans
        self.area_of_interest = area_of_interest
        self.cell_id_field = cell_id_field
        self.flow_convergence_threshold = flow_convergence_threshold

        if hf_resampler is None or lf_resampler is None:
            self.set_spatial_resamplers()
        else:
            self.hf_resampler = hf_resampler
            self.lf_resampler = lf_resampler
        self.cutoffs = cutoffs or {}

    @cached_property
    def aligned_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Trim WSE timeseries spatially and temporally."""
        hf_store = []
        lf_store = []
        # Need to do this incrementally to save RAM.
        for p in self.plans:
            # Load
            hf_data = self.get_plan_data(True, p)
            lf_data = self.get_plan_data(False, p)

            # Spatially subset
            hf_data = hf_data[:, self.hf_resampler]
            lf_data = lf_data[:, self.lf_resampler]
            lf_mask = lf_data < self.cell_elevations
            lf_data[lf_mask] = np.repeat(self.cell_elevations[:, np.newaxis], lf_data.shape[0], axis=1).T[lf_mask]

            # Temporally subset
            if p not in self.cutoffs:
                self.cutoffs[p] = self.get_cutoff(hf_data, lf_data)
            cutoff = self.cutoffs[p]

            # Format
            index = pd.MultiIndex.from_arrays([[p] * cutoff, range(cutoff)], names=["run", "t"])
            hf_df = pd.DataFrame(hf_data[:cutoff, :], columns=self.hf_resampler, index=index)
            hf_store.append(hf_df)
            lf_df = pd.DataFrame(lf_data[:cutoff, :], columns=self.lf_resampler, index=index)
            lf_store.append(lf_df)

        hf_merged = pd.concat(hf_store)
        lf_merged = pd.concat(lf_store)
        return hf_merged, lf_merged

    def get_cutoff(self, hf_data: NDArray[Any], lf_data: NDArray[Any]) -> int:
        """Determine when the model is 95% done changing."""
        min_length = min([len(hf_data), len(lf_data)])
        combo = np.concatenate([hf_data[:min_length, :], lf_data[:min_length, :]], axis=1)
        dx_dt = np.abs(np.diff(combo, axis=0))
        normalizer = np.sum(dx_dt, axis=0)
        normalizer[normalizer == 0] = 1
        dx_dt /= normalizer
        dx_dt = np.sum(dx_dt, axis=1) / np.sum(dx_dt)
        cum_dx_dt = np.cumsum(dx_dt)
        return cast(int, np.argmax(cum_dx_dt > self.flow_convergence_threshold))

    def get_plan_data(self, hf: bool, plan: str) -> NDArray[Any]:
        """Get water surface elevation timeseries from a HEC-RAS model."""
        model = self.hf_ras if hf else self.lf_ras
        asset = model.plan_hdfs[plan]
        ts: NDArray[Any] = asset.mesh_timeseries_output(self.mesh_id, "Water Surface").values
        return ts

    def set_spatial_resamplers(self) -> None:
        """Set the index arrays that are used to resample LF to HF."""
        hf_geom = self.hf_geometry_aoi
        lf_geom = self.lf_geometry_aoi

        mesh_resampled = gpd.overlay(
            hf_geom, lf_geom[[self.cell_id_field, "geometry"]], how="intersection", keep_geom_type=True
        )
        mesh_resampled["area"] = mesh_resampled.geometry.area
        mesh_resampled = mesh_resampled.sort_values(by="area")
        mesh_resampled = mesh_resampled.drop_duplicates(subset=f"{self.cell_id_field}_1", keep="last")
        mesh_resampled = mesh_resampled[[f"{self.cell_id_field}_1", f"{self.cell_id_field}_2"]]

        self.hf_mesh = hf_geom
        self.hf_resampler = mesh_resampled[f"{self.cell_id_field}_1"].values
        self.lf_resampler = mesh_resampled[f"{self.cell_id_field}_2"].values

    @cached_property
    def _hf_geometry_full(self) -> gpd.GeoDataFrame:
        return self._get_geometry_full(self.hf_ras)

    @cached_property
    def _lf_geometry_full(self) -> gpd.GeoDataFrame:
        return self._get_geometry_full(self.lf_ras)

    def _get_geometry_full(self, model: RasModel) -> gpd.GeoDataFrame:
        return model.get_plan_geometry(self.plans, self.mesh_id)

    @cached_property
    def hf_geometry_aoi(self) -> gpd.GeoDataFrame:
        """Geometry for the high-fidelity model within the area of interest."""
        return self._hf_geometry_full[self._hf_mask].copy()

    @cached_property
    def lf_geometry_aoi(self) -> gpd.GeoDataFrame:
        """Geometry for the low-fidelity model within the area of interest."""
        return self._lf_geometry_full[self._lf_mask].copy()

    @cached_property
    def _hf_mask(self) -> NDArray[Any]:
        return self._get_spatial_mask(self._hf_geometry_full)

    @cached_property
    def _lf_mask(self) -> NDArray[Any]:
        return self._get_spatial_mask(self._lf_geometry_full)

    def _get_spatial_mask(self, geom: gpd.GeoDataFrame) -> NDArray[Any]:
        return cast(NDArray[Any], geom.intersects(self.area_of_interest).values)

    @cached_property
    def cell_areas(self) -> NDArray[Any]:
        """Area of cells within the area of interest."""
        return cast(NDArray[Any], self.hf_ras.get_cell_areas(self.plans[0], self.mesh_id)[self.hf_resampler])

    @cached_property
    def cell_elevations(self) -> NDArray[Any]:
        """Elevation of cells within the area of interest."""
        return cast(
            NDArray[Any], self.hf_ras.get_cell_minimum_elevation(self.plans[0], self.mesh_id)[self.hf_resampler]
        )

    @cached_property
    def hf_mesh(self) -> gpd.GeoDataFrame:
        """Get the high-fidelity mesh geometry."""
        return self.hf_mesh

class PreProcessor:
    """Class to transform HEC-RAS data for use in upskilling low-fidelity (lf) models to high-fidelity (hf)."""

    def __init__(
        self,
        spatial_mode_count: int = 0,
        input_mean: NDArray[Any] | None = None,
        wet_threshold: float = 0.03,
        elevations: NDArray[Any] | None = None,
        depth: bool = False,
        wetness_classes: NDArray[Any] | None = None,
        weights: NDArray[Any] | None = None,
        eofs: NDArray[Any] | None = None,
        eigenvalues: NDArray[Any] | None = None,
        n_samples_fit: float = 0,
        x_mean: NDArray[Any] | None = None,
        x_std: NDArray[Any] | None = None,
    ):
        """Preprocessor class constructor.

        When default values are used (None, for most arguments), their values will be set during the fit() method.

        Args:
            spatial_mode_count (int): Number of spatial modes for PCA. Defaults to 0.
            input_mean (NDArray[Any] | None, optional): Mean values for centering input data. Defaults to None.
            wet_threshold (float, optional): Threshold for determining whether a cell gets wet. Defaults to 0.03.
            elevations (NDArray[Any] | None, optional): Elevation values for the cells. Defaults to None.
            depth (bool, optional): Whether output data should be depths or WSE. Defaults to False.
            wetness_classes (NDArray[Any] | None, optional): Wetness classification of cells. Defaults to None.
            weights (NDArray[Any] | None, optional): Weighting factors for the cells (typically cell area). Defaults to None.
            eofs (NDArray[Any] | None, optional): Empirical Orthogonal Functions (EOFs) from PCA. Defaults to None.
            eigenvalues (NDArray[Any] | None, optional): Eigenvalues from PCA. Defaults to None.
            n_samples_fit (float, optional): Number of samples used during PCA fitting. Defaults to 0.
            x_mean (NDArray[Any] | None, optional): Mean of spatial modes. Defaults to None.
            x_std (NDArray[Any] | None, optional): Standard deviation of spatial modes. Defaults to None.

        Returns:
            None
        """
        self.spatial_mode_count: int = spatial_mode_count

        self.input_mean: NDArray[Any] = input_mean or np.empty(0, dtype=float)

        self.wet_threshold = wet_threshold
        self.elevations: NDArray[Any] = elevations or np.empty(0, dtype=float)
        self.depth = depth
        self.wetness_classes: NDArray[np.str_] = wetness_classes or np.empty(0, dtype=np.str_)

        self.weights: NDArray[Any] = weights or np.empty(0, dtype=float)

        self.eofs: NDArray[Any] = eofs or np.empty(0, dtype=float)
        self.eigenvalues: NDArray[Any] = eigenvalues or np.empty(0, dtype=float)
        self.n_samples_fit = n_samples_fit

        self.x_mean: NDArray[Any] = x_mean or np.empty(0, dtype=float)
        self.x_std: NDArray[Any] = x_std or np.empty(0, dtype=float)

    @property
    def dry_indices(self) -> NDArray[np.bool_]:
        """Identify cells that are always dry.

        Returns:
            NDArray[np.bool_]: Boolean array indicating dry cells.
        """
        if self.wetness_classes is None:
            raise ValueError("wetness_classes must be numpy array to access dry_indices")
        return np.equal(self.wetness_classes, "AD")

    def fit(
        self,
        x: NDArray[Any],
        elevations: NDArray[Any],
        weights: NDArray[Any] | None = None,
        spatial_mode_count: int | None = None,
    ) -> None:
        """Fit the preprocessor to the input data using PCA.

        Filters out always-dry cells, centers the data, optionally applies weights,
        and fits a PCA model. Determines number of modes using North's Rule, if not set.

        Args:
            x (NDArray[Any]): Array of shape (samples, cells) representing water surface elevations.
            elevations (NDArray[Any]): Elevation values for each cell.
            weights (NDArray[Any]): Optional weighting array for cells (e.g., cell area).
            spatial_mode_count (int): Optional number of spatial modes to use.  Otherwise uses North's rule.

        Returns:
            None
        """
        # Filter cells that are always dry or always wet
        self.elevations = elevations
        if self.depth:
            x = self.wse_2_depth(x)
            self.wetness_classes = self.classify_wetness_depth(x)
        else:
            self.wetness_classes = self.classify_wetness_wse(x, elevations)
        x = x[:, ~self.dry_indices]

        # Apply first round of scaling
        self.input_mean = x.mean(axis=0)
        x = x - self.input_mean

        # Weight by cell area (or other)
        if weights is not None:
            self.weights = weights[~self.dry_indices]
            x *= self.weights

        # Fit PCA
        pca = IncrementalPCA()  # Documentation says that the function can batch itself
        pca.fit(x)

        # Reduce modes
        if spatial_mode_count is None:
            self.spatial_mode_count = self._compute_norths_rule(pca)
            # TODO: Consider these methods https://stats.stackexchange.com/questions/33917/how-to-determine-significant-principal-components-using-bootstrapping-or-monte-c
        else:
            self.spatial_mode_count = spatial_mode_count

        # Set results
        self.eofs = pca.components_[: self.spatial_mode_count]
        self.eigenvalues = pca.explained_variance_
        self.n_samples_fit = pca.n_samples_seen_

        # Set second round of standardization
        x = np.dot(x, self.eofs.T)
        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0)

    def transform(self, x: NDArray[Any]) -> NDArray[Any]:
        """Transform new input data using the fitted PCA model.

        Applies centering, weighting, and projects onto retained EOFs.

        Args:
            x (NDArray[Any]): Array of shape (samples, cells) to be transformed.

        Returns:
            NDArray[Any]: Array of transformed data in EOF space (samples, spatial_mode_count).
        """
        # Filter cells that are always dry or always wet
        if self.depth:
            x = self.wse_2_depth(x)
        x = x[:, ~self.dry_indices].copy()

        # Apply first round of scaling
        x = x - self.input_mean

        # Weight by cell area (or other)
        if self.weights is not None:
            x *= self.weights

        # Apply PCA
        x = np.dot(x, self.eofs.T)

        # Standardize
        x = (x - self.x_mean) / self.x_std

        return x

    def wse_2_depth(self, x: NDArray[Any]) -> NDArray[Any]:
        """Convert water surface elevation data to depths."""
        d: NDArray[Any] = x - self.elevations
        d[d < 0] = 0
        return d

    def reverse_transform(self, x: NDArray[Any]) -> NDArray[Any]:
        """Reverse the PCA transformation back to the original space.

        Reconstructs the full water surface elevation field, filling in
        always-dry cells with their original elevation values.

        Args:
            x (NDArray[Any]): Array of shape (samples, spatial_mode_count) in EOF space.

        Returns:
            NDArray[Any]: Array of shape (samples, cells) in original space.
        """
        x = (x * self.x_std) + self.x_mean
        x = np.dot(x, self.eofs)
        if self.weights is not None:
            x /= self.weights
        x += self.input_mean
        x_full = np.empty((x.shape[0], self.dry_indices.shape[0]))
        if self.depth:
            x_full[:, self.dry_indices] = 0
        else:
            x_full[:, self.dry_indices] = self.elevations[self.dry_indices]
        x_full[:, ~self.dry_indices] = x
        return x_full

    def _compute_norths_rule(self, pca: PCA | IncrementalPCA) -> int:
        """Determine the optimal number of PCA modes using North's Rule.

        North's Rule compares the drop-off between successive eigenvalues
        to an estimate of sampling uncertainty. Components with eigenvalues less
        than one are automatically dropped.

        Args:
            pca (PCAType): A fitted PCA or IncrementalPCA object.

        Returns:
            int: Number of significant EOF modes to retain.
        """
        if isinstance(pca, PCA):
            n = pca.n_samples_
        elif isinstance(pca, IncrementalPCA):
            n = pca.n_samples_seen_
        else:
            return 0
        eigenvalues = pca.explained_variance_
        eigenvalues = eigenvalues[eigenvalues > 1]  # Kaiser rule. Filter out eigenvalues <= 1
        if len(eigenvalues) == 0:
            return 0

        d_eigen = np.abs(np.diff(eigenvalues))
        d_error = np.sqrt(2 / n) * eigenvalues[:-1]
        ind = np.argmax(d_eigen <= d_error)
        if ind == 0:
            return int(len(eigenvalues))
        else:
            return int(ind)

    def classify_wetness_wse(self, x: NDArray[Any], elevations: NDArray[Any]) -> NDArray[np.str_]:
        """Classify each cell as always dry (AD), always flooded (AF), or transitionally flooded (TF).

        Classification is based on how the depth varies across samples relative to a wetness threshold.

        Args:
            x (NDArray[Any]): Array of shape (samples, cells) representing water surface elevations.
            elevations (NDArray[Any]): Elevation values for each cell.

        Returns:
            NDArray[Any]: Array of strings ("AD", "AF", or "TF") indicating wetness class per cell.
        """
        max_depth = x.max(axis=0) - elevations
        min_depth = x.min(axis=0) - elevations
        return self._classify_depths(max_depth, min_depth)

    def classify_wetness_depth(self, x: NDArray[Any]) -> NDArray[np.str_]:
        """Classify each cell as always dry (AD), always flooded (AF), or transitionally flooded (TF).

        Classification is based on how the depth varies across samples relative to a wetness threshold.

        Args:
            x (NDArray[Any]): Array of shape (samples, cells) representing water surface elevations.
            elevations (NDArray[Any]): Elevation values for each cell.

        Returns:
            NDArray[Any]: Array of strings ("AD", "AF", or "TF") indicating wetness class per cell.
        """
        max_depth = x.max(axis=0)
        min_depth = x.min(axis=0)
        return self._classify_depths(max_depth, min_depth)

    def _classify_depths(self, max_depth: NDArray[Any], min_depth: NDArray[Any]) -> NDArray[np.str_]:
        classes = np.empty(max_depth.shape, dtype="<U2")
        classes[max_depth < self.wet_threshold] = "AD"  # Always Dry
        classes[max_depth > self.wet_threshold] = "TF"  # Transitionally Flooded
        classes[min_depth > self.wet_threshold] = "AF"  # Always Flooded
        return classes

    def plot_pca_summary(self, out_path: str | Path) -> None:
        """Plot a summary of PCA eigenvalues with uncertainty and highlight number of selected modes.

        Args:
            out_path (str): Path to save the generated plot to.

        Returns:
            None
        """
        stderr = np.sqrt(2 / self.n_samples_fit) * self.eigenvalues
        inds = np.arange(self.eigenvalues.shape[0])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(inds, self.eigenvalues, c="k", label="Eigenvalues")
        ax.fill_between(
            inds, self.eigenvalues - stderr, self.eigenvalues + stderr, color="gray", alpha=0.3, label="± Error"
        )
        ax.axvline(
            x=self.spatial_mode_count - 1, color="red", ls="--", label=f"Selected Modes ({self.spatial_mode_count})"
        )

        ax.set_xlabel("EOF Mode Index")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.set_title("Variance Explained by EOF Modes")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
