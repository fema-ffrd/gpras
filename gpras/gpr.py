"""Methods for GPR training, tuning, and prediction."""

from __future__ import annotations

import json
import logging
import os
from functools import cached_property
from pathlib import Path
from typing import Any, TypeVar

import geopandas as gpd
import gpflow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from hecstac.ras.assets import GeometryHdfAsset, PlanHdfAsset
from hecstac.ras.item import RASModelItem
from numpy.typing import NDArray
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import KFold

T = TypeVar("T", bound="GPRAS")


class GPRAS:
    """A Gaussian Process Regression emulator for HEC-RAS."""

    __model_type__: str = "BaseGPRAS"

    def __init__(
        self, hf_model: str, hf_runs: list[str], testing_runs: list[str], hf_mesh_id: str, enable_cell_weighting: bool
    ):
        """Construct class."""
        # Load the high-fidelity model metadata
        self.hf_model_path = hf_model
        with open(self.hf_model_path) as f:
            self.hf_model = RASModelItem.from_dict(json.load(f))
        self.hf_runs = hf_runs
        self.hf_mesh_id = hf_mesh_id
        self.testing_runs = testing_runs
        self.training_runs = list(set(self.hf_runs) - set(self.testing_runs))
        self.enable_cell_weighting = enable_cell_weighting

        # GPR model definition
        self.mesh_id: str | None = None  # The mesh id that will be emulated
        self.scaling_method: dict[Any, Any] | None = None  # parameters of scaling function
        self.eof_mode_count: int | None = None  # spatial modes used from EOF analysis
        self.inducing_fraction: float | None = None  # percent of data for sparse GPR
        self.kernel: dict[Any, Any] | None = None  # parameters of the GPR kernel

        # Training/testing information

        self._runs: list[str] = []  # List of HEC-RAS run ids loaded as data
        # self.training_runs: list[str] = []  # Subset of self.runs used for training the model
        self.gpflow_parameters: dict[Any, Any] | None = None  # parameters of a fitted GPFlow model
        # self.testing_runs: list[str] = []  # Subset of self.runs used for testing the model
        self.testing_performance: dict[Any, Any] | None = None  # Summary of performance on testing runs
        self.validation_runs: list[str] = []  # Subset of self.runs used for validation the model
        self.validation_performance: dict[Any, Any] | None = None  # Summary of performance on testing runs

        # For the following data, there will be a multi index on ras run and timestep
        # self.data_x: pd.DataFrame | None = None  # Unscaled input features. Columns=depends on subclass; rows=timesteps.
        # self.data_y: pd.DataFrame | None = None  # Unscaled targets. Columns=cell WSE; rows=timesteps.
        # self.scaled_data_x: pd.DataFrame | None = None  # Scaled input features.
        # self.scaled_data_y: pd.DataFrame | None = None  # Scaled targets.
        # self.reduced_data_x: pd.DataFrame | None = None  # EOF analysis decomposed scaled inputs.
        # self.reduced_data_y: pd.DataFrame | None = None  # EOF analysis decomposed scaled targets.

    @classmethod
    def from_file(cls: type[T], json_path: str) -> T:
        """Load model from json."""
        with open(json_path) as f:
            j = json.load(f)

        return cls.from_dict(j)

    @classmethod
    def from_dict(cls: type[T], dict_rep: dict[Any, Any]) -> T:
        """Load model from dictionary."""
        inst = cls(**dict_rep)
        return inst

    @cached_property
    def plans(self) -> dict[str, PlanHdfAsset]:
        """Get a dictionary mapping plan name to PlanHdfAsset."""
        return {
            i.extra_fields["HEC-RAS:plan_information_plan_name"]: i.file.hdf_object
            for i in self.hf_model.assets.values()
            if isinstance(i, PlanHdfAsset)
        }

    @cached_property
    def geometries(self) -> dict[str, GeometryHdfAsset]:
        """Get a dictionary mapping geometry name to GeometryHdfAsset."""
        return {
            i.file.hdf_object.get_geom_attrs()["Title"]: i.file.hdf_object
            for i in self.hf_model.assets.values()
            if isinstance(i, GeometryHdfAsset)
        }

    @cached_property
    def geometry_hf(self) -> gpd.GeoDataFrame:
        """Get polygons from the high-fidelity geometry."""
        meshes = self._get_mesh_geometry_from_plans(self.hf_runs)
        return meshes[meshes["mesh_name"] == self.hf_mesh_id]

    def _get_mesh_geometry_from_plans(self, runs: list[str]) -> gpd.GeoDataFrame:
        """Get mesh polygons used to create a set of plans."""
        geoms = [j.get_attrs(j.PLAN_INFO_PATH)["Geometry Title"] for i, j in self.plans.items() if i in runs]
        assert all(i == geoms[0] for i in geoms), "Multiple geometries found in the hf model runs."
        geom = self.geometries[geoms[0]]
        return geom.mesh_cell_polygons()

    @cached_property
    def hf_geometry_title(self) -> str:
        """Return geometry title used by the first high-fidelity run."""
        plan = self.plans[self.hf_runs[0]]
        return str(plan.get_attrs(plan.PLAN_INFO_PATH)["Geometry Title"])

    @cached_property
    def cell_minimum_elevation(self) -> dict[int, float]:
        """Get minimum elevation values for HF cell IDs.

        Returns a dictionary mapping HF cell IDs (from geometry) to minimum elevation values,
        filtered to include only cell IDs used in the simulation output (data_y).

        Returns:
        dict[int, float]: A dictionary where keys are HF cell IDs and values are minimum elevation values.
        """
        geometry_title = self.hf_geometry_title

        # Locate the correct GeometryHdfAsset
        geometry_asset = next(
            asset
            for asset in self.hf_model.assets.values()
            if isinstance(asset, GeometryHdfAsset) and asset.file.hdf_object.get_geom_attrs()["Title"] == geometry_title
        )

        mesh_path = f"Geometry/2D Flow Areas/{self.hf_mesh_id}"
        mesh_hdf = geometry_asset.file.hdf_object.file[mesh_path]

        # All minimum elevations
        elevations = mesh_hdf["Cells Minimum Elevation"][()]

        # Map elevations to cell IDs using self.geometry_hf
        df = self.geometry_hf.set_index("cell_id").copy()
        df["elevation"] = elevations[df.index]  # index is cell ID, values are elevations

        # Retain only cell IDs used in data_y
        valid_ids = set(self.data_y.columns)
        filtered = df.loc[df.index.intersection(valid_ids)]

        return dict(filtered["elevation"].to_dict())

    @cached_property
    def hf_classify_wse_cells(self) -> dict[str, dict[str, NDArray[np.int_]]]:
        """Classify HF cells into flood categories for each run.

        Classifies HF cells into four flood categories: Always Dry (AD), Ever Wet (wet), Always Flooded (AF), and Temporarily Flooded (TF) for each run.

        Returns:
            dict[str, dict[str, NDArray[np.int_]]]: A nested dictionary where keys are run IDs and values
        """
        dry_threshold = 0.1
        classifications_by_run = {}

        for run_id in self.training_runs:
            wse = self.data_y.xs(run_id, level="run").values  # shape: (n_timesteps, n_cells)

            cell_ids = self.data_y.columns
            elevations = np.array([self.cell_minimum_elevation[i] for i in cell_ids])
            mesh_elevations_w_threshold = dry_threshold + elevations

            ad_idx = np.where(wse.max(axis=0) < mesh_elevations_w_threshold)[0]
            wet_idx = np.where(wse.max(axis=0) > mesh_elevations_w_threshold)[0]
            af_idx = np.where(wse.min(axis=0) > mesh_elevations_w_threshold)[0]
            tf_idx = np.where(
                (wse.min(axis=0) < mesh_elevations_w_threshold) & (wse.max(axis=0) > mesh_elevations_w_threshold)
            )[0]

            classifications_by_run[run_id] = {"AD": ad_idx, "wet": wet_idx, "AF": af_idx, "TF": tf_idx}

        return classifications_by_run

    def flood_classification_shp(self, run_ids: list[str], output_path: str) -> None:
        """Export a combined shapefile of flood classifications (AD, AF, TF, wet) for given HF runs.

        Parameters:
            run_ids: List of high-fidelity HEC-RAS plan IDs
            output_path: Full path to output shapefile (e.g., 'output/flood_categories.shp')
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Cached property: cell polygons (indexed by cell_id)
        mesh_polygons = self.geometry_hf.set_index("cell_id")

        # Cached property: classification results per run
        classification = self.hf_classify_wse_cells

        # Collect all records
        all_records = []

        for run_id in run_ids:
            run_cats = classification[run_id]
            for label in ["AD", "wet", "AF", "TF"]:
                idx = run_cats[label]
                gdf = mesh_polygons.loc[idx].copy()
                gdf["category"] = label
                gdf["run_id"] = run_id
                all_records.append(gdf)

        # Combine into a single GeoDataFrame
        combined_gdf = GeoDataFrame(pd.concat(all_records), crs=mesh_polygons.crs)

        # Export to shapefile
        combined_gdf.to_file(output_path)

    def to_file(self, out_path: str) -> None:
        """Save model representation to a STAC item json."""
        idx = Path(out_path).stem
        properties = {
            "model_type": self.__model_type__,
            "hf_model": self.hf_model_path,
            "scaling_method": self.scaling_method,
            "eof_mode_count": self.eof_mode_count,
            "inducing_fraction": self.inducing_fraction,
            "kernel": self.kernel,
            "gpflow_parameters": self.gpflow_parameters,
            "hf_runs": self.hf_runs,  # TODO: find way to include lf_runs here
            "training_runs": self.training_runs,
            "testing_runs": self.testing_runs,
            "validation_runs": self.validation_runs,
            "testing_performance": self.testing_performance,
            "validation_performance": self.validation_performance,
        }
        out_json = {"id": idx, "properties": properties}

        with open(out_path, mode="w") as f:
            json.dump(out_json, f, indent=4)

    def build_model(self, kernel_type: str, kernel_params: dict[Any, Any], inducing_fraction: float) -> None:
        """Define the GPR model."""
        # TODO: write this section.

        # Store kernel type, parameters, and inducing fraction
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params
        self.inducing_fraction = inducing_fraction

        kernel_options = {
            "Matern12": gpflow.kernels.Matern12,
            "Matern32": gpflow.kernels.Matern32,
            "Matern52": gpflow.kernels.Matern52,
            "RBF": gpflow.kernels.SquaredExponential,
            "Linear": gpflow.kernels.Linear,
            "Polynomial": gpflow.kernels.Polynomial,
            "Periodic": gpflow.kernels.Periodic,
            "Exponential": gpflow.kernels.Exponential,
            # Add more kernels as needed
            # Combination kernels
            # gpflow_kernel = gpflow.kernels.Sum([gpflow.kernels.RBF(), gpflow.kernels.Linear()])
            # OR
            # gpflow_kernel = gpflow.kernels.Product([gpflow.kernels.RBF(), gpflow.kernels.Periodic(base_kernel=gpflow.kernels.RBF(), period=1.0)])
            # Custom kernels can also be defined using gpflow.kernels.Kernel
            # Consider Change-points kernel
        }

        # Define valid parameters for each kernel
        valid_params = {
            "Matern12": ["lengthscales", "variance"],
            "Matern32": ["lengthscales", "variance"],
            "Matern52": ["lengthscales", "variance"],
            "RBF": ["lengthscales", "variance"],
            "Linear": ["bias_variance", "variance"],
            "Polynomial": ["degree", "variance"],
            "Periodic": ["lengthscales", "variance", "period"],
            "Exponential": ["lengthscales", "variance"],
        }

        assert all(
            param in valid_params[kernel_type] for param in self.kernel_params
        ), f"Invalid parameters for {kernel_type}: {kernel_params.keys()}"

        assert self.kernel_type in kernel_options, f"Invalid kernel type: {kernel_type}"

        # Initialize the kernel
        # kernel = kernel_options[self.kernel_type](**self.kernel_params)

        # # Define the model based on inducing fraction
        # if self.inducing_fraction < 1.0:
        #     num_inducing = int(len(X_train) * self.inducing_fraction)
        #     inducing_points = X_train[:num_inducing]
        #     self.model = gpflow.models.SGPR(
        #         data=(X_train, Y_train),
        #         kernel=kernel,
        #         inducing_variable=inducing_points,
        #     )
        # else:
        #     self.model = gpflow.models.GPR(
        #         data=(X_train, Y_train),
        #         kernel=kernel,
        #     )

    @cached_property
    def data_y(self) -> pd.DataFrame:
        """WSE information from the HF model."""
        return self._extract_plan_wsels(self.hf_runs)

    @cached_property
    def mean_across_timesteps(self) -> pd.DataFrame:
        """Compute Mean WSE values."""
        mean_y = self.data_y.groupby(level=0).mean()
        return mean_y

    @cached_property
    def n_timesteps_per_run(self) -> dict[str, int]:
        """Returns the number of unique timesteps for each high-fidelity run.

        Returns:
            Dictionary mapping run ID (str) to number of timesteps (int)
        """
        result = self.data_y.reset_index().groupby("run")["t"].nunique().to_dict()
        return dict(result)

    def _extract_plan_wsels(self, runs: list[str]) -> pd.DataFrame:
        """Extract water surface elevations from a HEC-RAS plan at each computational cell."""
        run_store = []
        for _ind, r in enumerate(runs):
            f = self.plans[r]

            ts = f.mesh_timeseries_output(self.hf_mesh_id, "Water Surface").values
            df = pd.DataFrame(ts)
            # df["run"] = ind
            df["run"] = r
            df["t"] = df.index.to_list()
            run_store.append(df)
        all_runs = pd.concat(run_store)
        all_runs = all_runs.set_index(["run", "t"])
        return all_runs

    @cached_property
    def cell_area_weights(self) -> dict[int, float]:
        """Cached property that computes cell areas as weights.

        Returns:
            - A dictionary mapping cell IDs to surface area (used as weights).

        Only computed if `enable_cell_weighting` is enabled.
        """
        if not getattr(self, "enable_cell_weighting", False):
            return {}

        geometry_title = self.hf_geometry_title

        # Locate geometry HDF asset
        geometry_asset = next(
            asset
            for asset in self.hf_model.assets.values()
            if isinstance(asset, GeometryHdfAsset) and asset.file.hdf_object.get_geom_attrs()["Title"] == geometry_title
        )

        ras_geom = geometry_asset.file.hdf_object
        mesh_path = f"Geometry/2D Flow Areas/{self.hf_mesh_id}"
        mesh_hdf = ras_geom[mesh_path]

        # Extract cell surface areas
        areas = mesh_hdf["Cells Surface Area"][()]
        area_dict = dict(enumerate(areas))

        return area_dict

    def generate_kfold_splits(
        self, run_ids: list[str], n_splits: int, shuffle: bool, random_state: int
    ) -> dict[str, dict[str, list[str]]]:
        """Generate K-Fold splits for the given run IDs and return as a dictionary.

        Parameters:
            run_ids (list[str]): List of HF run IDs (e.g., ["hf1", "hf2", ...])
            n_splits (int): Number of K-fold splits
            shuffle (bool): Whether to shuffle before splitting
            random_state (int): Seed for reproducibility

        Returns:
            dict: {
                "fold_0": {"train": [...], "test": [...]},
                "fold_1": {...},
                ...
            }
        """
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        splits_dict = {}

        for i, (train_idx, test_idx) in enumerate(kf.split(run_ids)):
            train_runs = [run_ids[j] for j in train_idx]
            test_runs = [run_ids[j] for j in test_idx]
            splits_dict[f"fold_{i}"] = {"train": train_runs, "cv": test_runs}

        return splits_dict

    def eof_center(self, data: np.ndarray, data_mean: np.ndarray) -> np.ndarray:
        """Center the data by subtracting the temporal mean."""
        return data - data_mean

    def eof_weight(self, data_centered: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Apply cell area weighting and convert to float32."""
        weighted = data_centered * weights
        return np.round(weighted, 3).astype(np.float32)

    def norths_rule(self, pca_obj: IncrementalPCA, n_samples: int) -> int:
        """Determine the number of significant EOFs using North's rule."""
        eigenvalues = pca_obj.explained_variance_
        eigenvalues = eigenvalues[eigenvalues > 1]  # Filter out eigenvalues <= 1
        if len(eigenvalues) == 0:
            return 0

        d_eigen = np.abs(np.diff(eigenvalues))
        d_error = np.sqrt(2 / n_samples) * eigenvalues[:-1]
        check_error_boundary = np.where(d_eigen <= d_error)[0]

        if check_error_boundary.size == 0:
            return int(len(eigenvalues))
        else:
            return int(check_error_boundary[0])

    def visualize_all_norths_rule(self, output_dir: str | None = None) -> None:
        """Loop through all K-fold results and plot North's Rule analysis for each.

        Parameters:
            output_dir: Directory to save plots. If None, displays interactively.

        Returns:
            None
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        for fold_name, result in self.kfold_eof_results.items():
            eigvals = result["explained_variance_ratio"]
            n_samples = sum(self.n_timesteps_per_run[run] for run in result["train_runs"])
            stderr = np.sqrt(2 / n_samples) * eigvals
            diff = np.abs(np.diff(eigvals))
            boundary = np.where(diff <= stderr[:-1])[0]

            # Determine cutoff
            cutoff = len(eigvals) if boundary.size == 0 else boundary[0]

            # Plot
            x = np.arange(1, len(eigvals) + 1)
            plt.figure(figsize=(8, 5))
            plt.plot(x, eigvals, label="Eigenvalues (Variance)", marker="o")
            plt.fill_between(x, eigvals - stderr, eigvals + stderr, color="gray", alpha=0.3, label="± Error")
            plt.axvline(cutoff + 1, color="red", linestyle="--", label=f"Significant Modes = {cutoff}")
            plt.xlabel("EOF Mode")
            plt.ylabel("Explained Variance Ratio")
            plt.title(f"North’s Rule – {fold_name}")
            plt.legend()
            plt.grid(True)

            if output_dir:
                filename = os.path.join(output_dir, f"norths_rule_{fold_name}.png")
                plt.savefig(filename, dpi=150, bbox_inches="tight")
                plt.close()
            else:
                plt.show()

    def perform_eof_kfold_analysis(self, n_splits: int = 5, n_modes: int = 100) -> None:
        """Perform EOF analysis using IncrementalPCA over K-fold splits.

        Stores EOF components, variance, and weighted temporal mean for each fold in self.kfold_eof_results.

        Parameters:
            n_splits: Number of K-fold splits
            n_modes: Number of EOF modes to retain

        Returns:
            None (results saved to self.kfold_eof_results)
        """
        logging.info("Starting EOF analysis across %d folds with %d EOF modes...", n_splits, n_modes)
        self.kfold_eof_results = {}

        # Generate K-fold splits using string-based run IDs
        k_fold_splits = self.generate_kfold_splits(
            run_ids=self.hf_runs, n_splits=n_splits, shuffle=True, random_state=42
        )

        for fold_name, split in k_fold_splits.items():
            logging.info(f"\n--- Fold {fold_name} ---")
            train_runs = split["train"]
            logging.info("Training runs: %s", train_runs)

            # Union wet indices across training runs
            all_wet_indices = set()
            for run in train_runs:
                wet_cells = self.hf_classify_wse_cells[run]["wet"]
                all_wet_indices.update(wet_cells)
            wet_indices = sorted(all_wet_indices)
            logging.info("Unioned wet cells: %d", len(wet_indices))

            # Compute weighted mean across wet cells using mean_across_timesteps
            means_per_run = np.array([self.mean_across_timesteps.loc[run, wet_indices] for run in train_runs])
            weights_for_mean = np.array([self.n_timesteps_per_run[run] for run in train_runs])
            combined_mean = np.average(means_per_run, axis=0, weights=weights_for_mean)
            logging.info("Computed weighted temporal mean over wet cells")

            # Get area-based weights if enabled
            if getattr(self, "use_cell_weights", False):
                weights = np.array([self.cell_area_weights[idx] for idx in wet_indices])
                logging.info("Using cell area weights")
            else:
                weights = np.ones_like(combined_mean)

            # Initialize PCA and total time step counter
            pca = IncrementalPCA(n_components=n_modes)
            total_timesteps = 0

            # Fit PCA incrementally over each training run
            for run in train_runs:
                wse = self.data_y.xs(run, level="run").values[:, wet_indices]
                wse_centered = wse - combined_mean
                wse_weighted = np.round(wse_centered * weights, 3).astype(np.float32)
                pca.partial_fit(wse_weighted)
                total_timesteps += wse.shape[0]
                logging.info("Partial fit on run %s (%d timesteps)", run, wse.shape[0])

            n_selected = self.norths_rule(pca, total_timesteps)

            logging.info("Selected %d significant EOFs (North's Rule)", n_selected)

            # Save results for this fold
            self.kfold_eof_results[fold_name] = {
                "EOFs": pca.components_[:n_selected],
                "explained_variance_ratio": pca.explained_variance_ratio_[:n_selected],
                "n_modes_selected": n_selected,
                "data_mean": combined_mean,
                "wet_indices": wet_indices,
                "train_runs": train_runs,
            }
            logging.info("EOFs and statistics stored for fold %s\n", fold_name)

    def plot_variance_explained(
        self,
        explained_variance_ratio: np.ndarray,
        n_selected: int,
        plot_dir: str | None = None,
        fold: str | None = None,
    ) -> None:
        """Plot cumulative variance explained by EOF modes and highlight selected number of modes.

        Parameters:
            explained_variance_ratio: Array of explained variance ratios.
            n_selected: Number of significant modes selected by North’s Rule.
        """
        cumulative_variance = np.cumsum(explained_variance_ratio)
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_variance, marker="o", label="Cumulative Variance")
        plt.axvline(x=n_selected - 1, color="red", linestyle="--", label=f"Selected Modes ({n_selected})")
        plt.xlabel("EOF Mode Index")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Variance Explained by EOF Modes")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if plot_dir:
            plt.savefig(os.path.join(plot_dir, f"{fold}_variance_explained.png"))
            plt.close()
        else:
            plt.show()

        plt.show()

    def visualize_all_kfold_eof_results_separate(self, output_dir: str | None = None, n_modes_to_plot: int = 3) -> None:
        """Visualize EOF analysis results for all folds.

        Plots cumulative variance explained and saves each spatial EOF mode as a separate figure.

        Parameters:
        output_dir (str): Directory to save figures. If None, plots are shown but not saved.
        n_modes_to_plot (int): Number of spatial EOF modes to plot for each fold.
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        for fold_name, result in self.kfold_eof_results.items():
            print(f"\nVisualizing results for {fold_name}...")

            # --- Variance explained plot ---
            self.plot_variance_explained(
                explained_variance_ratio=result["explained_variance_ratio"],
                n_selected=result["n_modes_selected"],
                plot_dir=output_dir,
                fold=fold_name,
            )

            # --- Spatial EOFs: separate plots per mode ---
            eof_modes = result["EOFs"]
            wet_indices = result["wet_indices"]
            # Compute centroids from geometry
            centroids = self.geometry_hf.geometry.centroid
            x_coords = centroids.x.values
            y_coords = centroids.y.values

            for i in range(min(n_modes_to_plot, eof_modes.shape[0])):
                eof = eof_modes[i]
                fig, ax = plt.subplots(figsize=(6, 5))
                sc = ax.scatter(
                    x_coords[wet_indices],
                    y_coords[wet_indices],
                    c=eof,
                    cmap="coolwarm",
                    s=8,
                    edgecolor="k",
                    linewidth=0.2,
                )
                ax.set_title(f"Fold {fold_name} - Spatial EOF Mode {i + 1}")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_aspect("equal")
                fig.colorbar(sc, ax=ax, label="EOF amplitude")
                plt.xticks(rotation=90)
                if output_dir:
                    plt.savefig(os.path.join(output_dir, f"{fold_name}_EOF_mode_{i + 1}.png"))
                    plt.close()
                else:
                    plt.show()

    def create_pseudo_ecs(
        self, data: np.ndarray, eofs: np.ndarray, weights: np.ndarray, data_mean: np.ndarray = None
    ) -> np.ndarray:
        """Generate pseudo expansion coefficients (ECs) from data using provided EOFs and weights.

        Parameters:
            data: Time series data (timesteps x spatial points)
            EOFs: Empirical Orthogonal Functions (modes x spatial points)
            weights: Weights for each spatial point (usually cell areas)
            data_mean: Mean to use for centering; if None, computed from `data`

        Returns:
            Pseudo ECs (timesteps x modes)
        """
        if data_mean is None:
            data_mean = np.mean(data, axis=0)

        logging.info("Centering and weighting WSE data for EC computation")
        data_centered = data - data_mean
        data_centered_weighted = data_centered * weights
        pseudo_ecs = np.dot(data_centered_weighted, eofs.T)

        return pseudo_ecs

    def derive_ecs_from_eofs(self, fold_name: str) -> np.ndarray:
        """Compute expansion coefficients (ECs) for all training runs of a given fold.

        Parameters:
            fold_name: Name of the fold (e.g., 'fold_0')

        Returns:
            Combined ECs (timesteps x modes)
        """
        logging.info(f"Deriving ECs for fold: {fold_name}")

        fold_result = self.kfold_eof_results[fold_name]
        eofs = fold_result["EOFs"]
        data_mean = fold_result["data_mean"]
        wet_indices = fold_result["wet_indices"]
        train_runs = fold_result["train_runs"]

        if getattr(self, "use_cell_weights", False):
            weights = np.array([self.cell_area_weights[idx] for idx in wet_indices])
        else:
            weights = np.ones_like(data_mean)

        all_ecs = []
        for run in train_runs:
            wse = self.data_y.xs(run, level="run").values[:, wet_indices]
            ecs = self.create_pseudo_ecs(wse, eofs, weights, data_mean)
            all_ecs.append(ecs)
            logging.info(f"Derived ECs for run: {run}, shape: {ecs.shape}")

        return np.vstack(all_ecs)

    def compute_and_store_all_hf_ecs(self) -> None:
        """Compute and store HF ECs from all K-folds into self.Y_train.

        This will populate self.Y_train as a dictionary with fold names as keys
        and EC arrays (timesteps × modes) as values.
        """
        self.Y_train = {}
        for fold_name in self.kfold_eof_results:
            ecs = self.derive_ecs_from_eofs(fold_name)
            self.Y_train[fold_name] = ecs
            logging.info("Stored HF ECs for %s into self.Y_train", fold_name)

    def visualize_all_kfold_ecs(
        self,
        n_modes_to_plot: int = 3,
        output_dir: str | None = None,
        show_run_boundaries: bool = True,
    ) -> None:
        """Visualize expansion coefficients (ECs) over time for the top modes across all K-folds.

        For each fold in `self.kfold_eof_results`, this method plots the time series of expansion
        coefficients for the top `n_modes_to_plot` EOF modes. If `show_run_boundaries` is True, it
        overlays vertical dashed lines and labels to indicate transitions between training runs.

        Parameters:
            n_modes_to_plot (int): Number of leading EC modes to plot. Defaults to 3.
            output_dir [str]: Directory to save plots. If None, plots are shown interactively.
            show_run_boundaries (bool): Whether to annotate run boundaries. Defaults to True.

        Returns:
            None
        """
        for fold_name in self.kfold_eof_results:
            logging.info("Visualizing ECs for %s", fold_name)
            fold_result = self.kfold_eof_results[fold_name]
            train_runs = fold_result["train_runs"]

            # Prepare ECs and record run transition points
            ecs_list = []
            run_boundaries = []
            total_timesteps = 0

            for run in train_runs:
                wse = self.data_y.xs(run, level="run").values[:, fold_result["wet_indices"]]
                weights = (
                    np.array([self.cell_area_weights[idx] for idx in fold_result["wet_indices"]])
                    if getattr(self, "use_cell_weights", False)
                    else np.ones_like(fold_result["data_mean"])
                )
                ecs = self.create_pseudo_ecs(
                    data=wse,
                    eofs=fold_result["EOFs"],
                    weights=weights,
                    data_mean=fold_result["data_mean"],
                )
                ecs_list.append(ecs)
                run_boundaries.append((total_timesteps, run))
                total_timesteps += ecs.shape[0]

            all_ecs = np.vstack(ecs_list)
            n_timesteps, n_modes = all_ecs.shape

            fig, axs = plt.subplots(n_modes_to_plot, 1, figsize=(14, 3 * n_modes_to_plot), sharex=True)
            fig.suptitle(f"Expansion Coefficients for {fold_name}", fontsize=16)

            for i in range(min(n_modes_to_plot, n_modes)):
                ax = axs[i] if n_modes_to_plot > 1 else axs
                ax.plot(all_ecs[:, i], label=f"EC Mode {i + 1}")
                ax.set_ylabel(f"Mode {i + 1}", fontsize=12)
                ax.grid(True)

                if show_run_boundaries:
                    for t, run_id in run_boundaries:
                        ax.axvline(x=t, color="gray", linestyle="--", linewidth=0.8)
                        ax.text(t + 2, ax.get_ylim()[1] * 0.9, run_id, fontsize=9, rotation=90, verticalalignment="top")

            axs[-1].set_xlabel("Time Step", fontsize=12)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                file_path = os.path.join(output_dir, f"ECs_{fold_name}.png")
                plt.savefig(file_path)
                plt.close(fig)
                logging.info("Saved EC plot for %s to %s", fold_name, file_path)
            else:
                plt.show()

    @cached_property
    def data_y_train(self) -> dict[str, np.ndarray]:
        """Cached property to compute and store high-fidelity ECs (targets) for all folds.

        Returns:
            dict[str, np.ndarray]: Mapping from fold names to HF EC arrays (timesteps × modes).
        """
        logging.info("Computing and caching HF ECs as training targets (data_y_train)")
        self.compute_and_store_all_hf_ecs()
        return self.Y_train

    def scale_data(self) -> None:
        """Apply the specified transformation to the data to generate self.scaled_data."""
        assert self.scaling_method is not None, "self.scaling_method must be defined prior to data scaling."
        # TODO: write scaling code

    def reduce_data(self) -> None:
        """Apply an EOF analysis to scaled data to reduce dimensionality."""
        assert self.eof_mode_count is not None, "self.eof_mode_count must be defined prior to data reduction."
        # TODO: write EOF code

    def split_training_testing_validation(self) -> None:
        """Split runs into training, testing and validation."""
        # Skeleton code.  Probably better to use a canned algorithm.
        # Ex. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        # Also, train-test split will be dynamic for k-fold CV

        # runs = [i["href"] for i in self.hf_model["assets"]]
        # self.testing_runs = sample(runs, 1)
        # runs = set(runs).difference(self.testing_runs)
        # self.testing_runs = sample(runs, 1)
        # self.validation_runs = list(set(runs).difference(self.testing_runs))

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Use GPR model to predict targets from inputs."""
        # TODO: write this code.
        return np.array([0])

    def fit(self) -> None:
        """Fit the GPR model."""
        # TODO: Implement this.


class RasEmulator(GPRAS):
    """Emulate high-fidelity HEC-RAS from low-fidelity HEC-RAS."""

    __model_type__: str = "RasEmulator"

    def __init__(
        self,
        ras_model_path: str,
        hf_runs: list[str],
        testing_runs: list[str],
        lf_runs: list[str],
        hf_mesh_id: str,
        lf_mesh_id: str,
        enable_cell_weighting: bool,
    ):
        """Construct class."""
        super().__init__(ras_model_path, hf_runs, testing_runs, hf_mesh_id, enable_cell_weighting)
        self.hf_runs = hf_runs
        self.testing_runs = testing_runs
        self.training_runs = list(set(self.hf_runs) - set(self.testing_runs))
        self.lf_runs = lf_runs
        self.hf_mesh_id = hf_mesh_id
        self.lf_mesh_id = lf_mesh_id
        self.enable_cell_weighting = enable_cell_weighting

    @cached_property
    def geometry_lf(self) -> gpd.GeoDataFrame:
        """Get polygons from the low-fidelity geometry."""
        meshes = self._get_mesh_geometry_from_plans(self.lf_runs)
        return meshes[meshes["mesh_name"] == self.lf_mesh_id]

    @cached_property
    def cell_resampler(self) -> dict[str, str]:
        """A dictionary that spatially joins lf and hf cells."""
        mesh_resampled = gpd.overlay(self.geometry_hf, self.geometry_lf[["cell_id", "geometry"]], how="intersection")
        mesh_resampled["area"] = mesh_resampled.geometry.area
        mesh_resampled = mesh_resampled.sort_values(by="area")
        mesh_resampled = mesh_resampled.drop_duplicates(subset="cell_id_1", keep="last")
        mesh_resampled = mesh_resampled[["cell_id_1", "cell_id_2"]]
        return dict(zip(mesh_resampled["cell_id_1"], mesh_resampled["cell_id_2"], strict=True))

    @cached_property
    def run_mapping_hf_to_lf(self) -> dict[str, str]:
        """Maps high-fidelity (HF) run IDs to corresponding low-fidelity (LF) run IDs."""
        return dict(zip(self.hf_runs, self.lf_runs, strict=False))

    @cached_property
    def data_x(self) -> pd.DataFrame:
        """WSE information from the LF model resampled to hf dimensions."""
        d = self._extract_plan_wsels(self.lf_runs)
        d_resample = pd.DataFrame({col: d[self.cell_resampler[col]] for col in self.data_y.columns})
        return d_resample

    def filter_dry_areas(
        self,
        data: np.ndarray,
        elevation: np.ndarray,
        dry_threshold: float = 0.1,
        lower_dry_ele: float = 0.0,
    ) -> np.ndarray:
        """Filter dry areas by forcing WSE below elevation threshold to elevation - offset.

        Args:
            data (np.ndarray): Water surface elevation values (shape: timesteps x cells).
            elevation (np.ndarray): Ground elevation values per cell (shape: cells,).
            dry_threshold (float): Minimum height above elevation to consider a cell wet. Defaults to 0.03.
            lower_dry_ele (float): Value to subtract from elevation for dry cells. Defaults to 0.0.

        Returns:
            np.ndarray: Filtered WSE array with dry values replaced.
        """
        if data.shape[1] != elevation.shape[0]:
            raise ValueError(
                f"Mismatched dimensions: data has {data.shape[1]} cells, but elevation has {elevation.shape[0]}"
            )

        elevation = elevation.reshape(1, -1)  # Ensure broadcastable shape (1 x cells)
        threshold = elevation + dry_threshold
        replacement = elevation - lower_dry_ele

        return np.where(data > threshold, data, replacement)

    def derive_lf_ecs_from_hf_eof(self, fold_name: str) -> np.ndarray:
        """Derive low-fidelity expansion coefficients (ECs) using HF EOFs for a given fold.

        This routine filters dry areas, centers and weights LF data using HF EOF statistics,
        and projects the LF data onto the HF EOF spatial modes.

        Args:
            fold_name (str): Fold name (e.g., 'fold_0').

        Returns:
            np.ndarray: Expansion coefficients (timesteps x modes) for LF data in this fold.
        """
        logging.info("Deriving LF ECs using HF EOFs for %s", fold_name)

        result = self.kfold_eof_results[fold_name]
        wet_indices = result["wet_indices"]
        eofs = result["EOFs"]
        # hf_mean = result["data_mean"]
        train_runs = result["train_runs"]

        # Retrieve elevation in HF cell order for wet cells
        elevation_hf_order = np.array([self.cell_minimum_elevation[idx] for idx in wet_indices])

        if getattr(self, "use_cell_weights", False):
            weights = np.array([self.cell_area_weights[idx] for idx in wet_indices])
        else:
            weights = np.ones(len(wet_indices), dtype=np.float32)

        all_ecs = []
        lf_train_runs = []
        for hf_run in train_runs:
            lf_train_runs.append(self.run_mapping_hf_to_lf.get(hf_run))
        lf_data_runs = [self.data_x.xs(run, level="run").values[:, wet_indices] for run in lf_train_runs]
        lf_data_stacked = np.vstack(lf_data_runs)
        lf_filtered = self.filter_dry_areas(lf_data_stacked, elevation_hf_order)
        all_ecs = self.create_pseudo_ecs(lf_filtered, eofs, weights, data_mean=None)
        return all_ecs

    def compute_and_store_all_lf_ecs(self) -> dict[str, np.ndarray]:
        """Compute and store LF ECs from all K-folds into self.X_train.

        This will populate lf_ecs as a dictionary with fold names as keys
        and EC arrays (timesteps x modes) as values.
        """
        # self.X_train = {}
        lf_ecs = {}
        for fold_name in self.kfold_eof_results:
            ecs = self.derive_lf_ecs_from_hf_eof(fold_name)
            lf_ecs[fold_name] = ecs
        return lf_ecs

    @cached_property
    def data_x_train(self) -> dict[str, np.ndarray]:
        """Cached property to compute and store low-fidelity ECs (inputs) for all folds.

        Returns:
            dict[str, np.ndarray]: Mapping from fold names to LF EC arrays (timesteps x modes).
        """
        logging.info("Computing and caching LF ECs as training inputs (data_x_train)")
        lf_ecs_dict = self.compute_and_store_all_lf_ecs()
        return lf_ecs_dict

    def visualize_all_lf_ecs(self, n_modes_to_plot: int = 3, output_dir: str | None = None) -> None:
        """Visualize LF ECs over time for the top `n_modes_to_plot` modes across all K-folds.

        Parameters:
            n_modes_to_plot (int): Number of leading ECs to visualize.
            output_dir [str]: Directory to save plots. If None, shows plots interactively.
        """
        for fold_name in self.kfold_eof_results:
            logging.info("Visualizing LF ECs for %s", fold_name)

            ecs = self.data_x_train[fold_name]
            train_runs = self.kfold_eof_results[fold_name]["train_runs"]

            # Determine where each run starts and ends
            run_lengths = [self.data_y.xs(run, level="run").shape[0] for run in train_runs]
            run_boundaries = np.cumsum(run_lengths[:-1])
            # run_start_positions = np.insert(run_boundaries, 0, 0)

            fig, axs = plt.subplots(n_modes_to_plot, 1, figsize=(12, 3 * n_modes_to_plot), sharex=True)
            fig.suptitle(f"LF Expansion Coefficients for {fold_name}", fontsize=14)

            for i in range(min(n_modes_to_plot, ecs.shape[1])):
                ax = axs[i] if n_modes_to_plot > 1 else axs
                ax.plot(ecs[:, i], label=f"EC Mode {i + 1}")

                y_min, y_max = ax.get_ylim()

                # Add vertical lines with labels
                run_ends = np.cumsum(run_lengths)
                for _j, boundary in enumerate(run_boundaries):
                    ax.axvline(x=boundary, color="gray", linestyle="--", linewidth=0.8)
                for j, run in enumerate(train_runs):
                    # midpoint = run_start_positions[j] + run_lengths[j] // 2
                    xpos = run_ends[j] - 1
                    # ax.text(midpoint, ax.get_ylim()[1] * 0.95, run, rotation=90, fontsize=8, ha="center", va="top")
                    ax.text(
                        xpos,
                        y_max - 0.05 * (y_max - y_min),
                        run,
                        rotation=90,
                        fontsize=8,
                        ha="left",
                        va="top",
                        color="gray",
                    )
                ax.set_ylabel(f"EC {i + 1}")
                ax.grid(True)

            axs[-1].set_xlabel("Time Step")
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filepath = os.path.join(output_dir, f"LF_ECs_{fold_name}.png")
                plt.savefig(filepath)
                plt.close(fig)
                logging.info("Saved LF EC plot to %s", filepath)
            else:
                plt.show()

    def compare_lf_hf_ecs(
        self,
        output_dir: str,
        n_modes_to_plot: int = 3,
    ) -> None:
        """Plot HF vs LF ECs per fold for the top `n_modes_to_plot` modes.

        For each fold:
        - Creates a time series plot of HF and LF ECs over time.
        - Creates a scatter plot of LF ECs vs HF ECs for each mode.

        Args:
            output_dir (str): Directory to save plots.
            n_modes_to_plot (int): Number of leading EC modes to compare.
        """
        os.makedirs(output_dir, exist_ok=True)

        for fold_name in self.kfold_eof_results:
            hf_ecs = self.data_y_train[fold_name]
            lf_ecs = self.data_x_train[fold_name]

            assert hf_ecs.shape == lf_ecs.shape, f"Mismatch in shape for fold {fold_name}"

            time = np.arange(hf_ecs.shape[0])

            for mode in range(min(n_modes_to_plot, hf_ecs.shape[1])):
                fig, axs = plt.subplots(2, 1, figsize=(12, 8))

                # Time series plot
                axs[0].plot(time, hf_ecs[:, mode], label="HF", color="blue")
                axs[0].plot(time, lf_ecs[:, mode], label="LF", color="red", linestyle="--")
                axs[0].set_title(f"EC Time Series for Fold {fold_name} - Mode {mode + 1}")
                axs[0].set_xlabel("Time Step")
                axs[0].set_ylabel("EC Value")
                axs[0].legend()
                axs[0].grid(True)

                # Scatter plot
                axs[1].scatter(hf_ecs[:, mode], lf_ecs[:, mode], color="purple", alpha=0.6)
                axs[1].set_title(f"LF vs HF EC Scatter for Fold {fold_name} - Mode {mode + 1}")
                axs[1].set_xlabel("HF EC")
                axs[1].set_ylabel("LF EC")
                axs[1].grid(True)

                plt.tight_layout()
                filename = os.path.join(output_dir, f"Compare_ECs_{fold_name}_Mode{mode + 1}.png")
                plt.savefig(filename)
                plt.close(fig)


class InterpolationEmulator(GPRAS):
    """Emulate high-fidelity HEC-RAS from estimated WSEL grid."""

    __model_type__: str = "InterpolationEmulator"

    # TODO: Plan and implement approach #2


class HydrographEmulator(GPRAS):
    """Emulate high-fidelity HEC-RAS from a hydrograph."""

    __model_type__: str = "HydrographEmulator"

    # TODO: Plan and implement approach #3
