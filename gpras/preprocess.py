"""Tools to wrangle HEC-RAS data into a format usable by the gaussian process regression model."""

from functools import cached_property
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.decomposition import PCA, IncrementalPCA


class CellResampler:
    """Spatially resamples coarse grids to match dimensions of finer grids by repeating elements."""

    def __init__(
        self,
        geometry_hf: gpd.GeoDataFrame,
        geometry_lf: gpd.GeoDataFrame,
        cell_id_field: str,
        area_of_interest: gpd.GeoDataFrame | None = None,
    ):
        """Initialize the CellResampler class.

        Args:
            geometry_hf (gpd.GeoDataFrame): High-fidelity geometry containing cell data.
            geometry_lf (gpd.GeoDataFrame): Low-fidelity geometry containing cell data.
            cell_id_field (str): The column name for the cell ID field in both geometries.
            area_of_interest (gpd.GeoDataFrame | None, optional): Geometry defining the area of interest for spatial filtering.
        """
        self.geometry_hf = geometry_hf  # TODO: add spatial filtering to area of interest
        self.geometry_lf = geometry_lf
        self.cell_id_field = cell_id_field
        self.area_of_interest = area_of_interest

    @cached_property
    def cell_resampler(self) -> dict[str, str]:
        """Create a mapping dictionary from high-fidelity (hf) cell id to low-fidelity (lf) cell id.

        Returns:
            dict[str, str]: A dictionary where keys are hf cell IDs and values are lf cell IDs.
        """
        mesh_resampled = gpd.overlay(
            self.geometry_hf, self.geometry_lf[[self.cell_id_field, "geometry"]], how="intersection"
        )
        mesh_resampled["area"] = mesh_resampled.geometry.area
        mesh_resampled = mesh_resampled.sort_values(by="area")
        mesh_resampled = mesh_resampled.drop_duplicates(subset=f"{self.cell_id_field}_1", keep="last")
        mesh_resampled = mesh_resampled[[f"{self.cell_id_field}_1", f"{self.cell_id_field}_2"]]
        return dict(zip(mesh_resampled["cell_id_1"], mesh_resampled["cell_id_2"], strict=True))

    def resample_lf_to_hf(self, lf_df: pd.DataFrame) -> pd.DataFrame:
        """Resample data from low-fidelity (lf) cells to high-fidelity (hf) cells.

        Args:
            lf_df (pd.DataFrame): A DataFrame containing data indexed by low-fidelity cell IDs.

        Returns:
            pd.DataFrame: A DataFrame containing resampled data with columns ordered to match high-fidelity cell ids.
        """
        return pd.DataFrame(
            {col: lf_df[self.cell_resampler[col]] for col in self.geometry_hf[self.cell_id_field].values}
        )


class PreProcessor:
    """Class to transform HEC-RAS data for use in upskilling low-fidelity (lf) models to high-fidelity (hf)."""

    def __init__(
        self,
        spatial_mode_count: int = 0,
        mean_array: NDArray[Any] | None = None,
        wet_threshold: float = 0.03,
        elevations: NDArray[Any] | None = None,
        wetness_classes: NDArray[Any] | None = None,
        weights: NDArray[Any] | None = None,
        eofs: NDArray[Any] | None = None,
        eigenvalues: NDArray[Any] | None = None,
        n_samples_fit: float = 0,
    ):
        """Preprocessor class constructor.

        Args:
            spatial_mode_count (int): Number of spatial modes for PCA. Defaults to 0.
            mean_array (NDArray[Any] | None, optional): Mean values for centering input data. Defaults to None.
            wet_threshold (float, optional): Threshold for determining whether a cell gets wet. Defaults to 0.03.
            elevations (NDArray[Any] | None, optional): Elevation values for the cells. Defaults to None.
            wetness_classes (NDArray[Any] | None, optional): Wetness classification of cells. Defaults to None.
            weights (NDArray[Any] | None, optional): Weighting factors for the cells (typically cell area). Defaults to None.
            eofs (NDArray[Any] | None, optional): Empirical Orthogonal Functions (EOFs) from PCA. Defaults to None.
            eigenvalues (NDArray[Any] | None, optional): Eigenvalues from PCA. Defaults to None.
            n_samples_fit (float, optional): Number of samples used during PCA fitting. Defaults to 0.

        Returns:
            None
        """
        self.spatial_mode_count: int = spatial_mode_count

        self.mean_array: NDArray[Any] = mean_array or np.empty(0, dtype=float)

        self.wet_threshold = wet_threshold
        self.elevations: NDArray[Any] = elevations or np.empty(0, dtype=float)
        self.wetness_classes: NDArray[np.str_] = wetness_classes or np.empty(0, dtype=np.str_)

        self.weights: NDArray[Any] = weights or np.empty(0, dtype=float)

        self.eofs: NDArray[Any] = eofs or np.empty(0, dtype=float)
        self.eigenvalues: NDArray[Any] = eigenvalues or np.empty(0, dtype=float)
        self.n_samples_fit = n_samples_fit

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
    ) -> None:
        """Fit the preprocessor to the input data using PCA.

        Filters out always-dry cells, centers the data, optionally applies weights,
        and fits a PCA model. Determines number of modes using North's Rule, if not set.

        Args:
            x (NDArray[Any]): Array of shape (samples, cells) representing water surface elevations.
            elevations (NDArray[Any]): Elevation values for each cell.
            weights (NDArray[Any]): Optional weighting array for cells (e.g., cell area).

        Returns:
            None
        """
        # Filter cells that are always dry or always wet
        self.wetness_classes = self.classify_wetness(x, elevations)
        self.elevations = elevations
        x = x[:, not self.dry_indices]
        if weights is not None:
            weights = weights[not self.dry_indices]
            self.weights = weights

        # Apply first round of scaling
        self.mean_array = x.mean(axis=0)
        x = x - self.mean_array

        # Weight by cell area (or other)
        if len(self.weights) > 0:
            x *= weights

        # Fit PCA
        pca = IncrementalPCA()  # Documentation says that the function can batch itself
        pca.fit(x)

        # Reduce modes
        if self.spatial_mode_count == 0:
            self.spatial_mode_count = self._compute_norths_rule(pca)
            # TODO: Consider these methods https://stats.stackexchange.com/questions/33917/how-to-determine-significant-principal-components-using-bootstrapping-or-monte-c

        # Set results
        self.eofs = pca.components_[: self.spatial_mode_count]
        self.eigenvalues = pca.explained_variance_
        self.n_samples_fit = pca.n_samples_seen_

    def transform(self, x: NDArray[Any]) -> NDArray[Any]:
        """Transform new input data using the fitted PCA model.

        Applies centering, weighting, and projects onto retained EOFs.

        Args:
            x (NDArray[Any]): Array of shape (samples, cells) to be transformed.

        Returns:
            NDArray[Any]: Array of transformed data in EOF space (samples, spatial_mode_count).
        """
        # Filter cells that are always dry or always wet
        x = x[:, not self.dry_indices].copy()

        # Apply first round of scaling
        x = x - self.mean_array

        # Weight by cell area (or other)
        if self.weights is not None:
            x *= self.weights

        # Apply PCA
        x = np.dot(x, self.eofs.T)

        return x

    def reverse_transform(self, x: NDArray[Any]) -> NDArray[Any]:
        """Reverse the PCA transformation back to the original space.

        Reconstructs the full water surface elevation field, filling in
        always-dry cells with their original elevation values.

        Args:
            x (NDArray[Any]): Array of shape (samples, spatial_mode_count) in EOF space.

        Returns:
            NDArray[Any]: Array of shape (samples, cells) in original space.
        """
        x = np.dot(x, self.eofs)
        if self.weights is not None:
            x /= self.weights
        x = x + self.mean_array
        x_full = np.empty((x.shape[0], self.dry_indices.shape[0]))
        x_full[:, self.dry_indices] = self.elevations[self.dry_indices]
        x_full[:, not self.dry_indices] = x
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

    def classify_wetness(self, x: NDArray[Any], elevations: NDArray[Any]) -> NDArray[np.str_]:
        """Classify each cell as always dry (AD), always flooded (AF), or transitionally flooded (TF).

        Classification is based on how the depth varies across samples relative to a wetness threshold.

        Args:
            x (NDArray[Any]): Array of shape (samples, cells) representing water surface elevations.
            elevations (NDArray[Any]): Elevation values for each cell.

        Returns:
            NDArray[Any]: Array of strings ("AD", "AF", or "TF") indicating wetness class per cell.
        """
        classes = np.empty(elevations.shape, dtype="<U2")
        max_depth = x.max(axis=0) - elevations
        min_depth = x.min(axis=0) - elevations
        classes[max_depth < self.wet_threshold] = "AD"  # Always Dry
        classes[max_depth > self.wet_threshold] = "TF"  # Transitionally Flooded
        classes[min_depth > self.wet_threshold] = "AF"  # Always Flooded
        return classes

    def plot_pca_summary(self, out_path: str) -> None:
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
