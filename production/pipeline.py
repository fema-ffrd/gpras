"""Example script for running a GPR model training workflow."""

import json
import os
from typing import TypedDict

import numpy as np

from gpras.gpr import GPRAS, KernelType
from gpras.preprocess import CellResampler, PreProcessor
from gpras.ras.model import RasModel
from gpras.utils.plotting import ec_pairplot, ec_timeseries, performance_cdf, performance_scatterplot


class Config(TypedDict):
    """Type hinting for config dictionary."""

    ras_stac_path: str
    mesh_id: str
    hf_plans: list[str]
    lf_plans: list[str]
    wet_threshold_depth: float
    kernel: KernelType
    inducing_fraction: float
    plot_dir: str


config: Config = {
    "ras_stac_path": "/workspaces/gpras/data/Muncie/Muncie.stac.json",
    "mesh_id": "2D Interior Area",
    "hf_plans": ["hf1", "hf2", "hf3"],
    "lf_plans": ["lf1", "lf2", "lf3"],
    "wet_threshold_depth": 0.03,
    "kernel": "Exponential",
    "inducing_fraction": 0.2,
    "plot_dir": "plots",
}

# Load data
with open(config["ras_stac_path"]) as f:
    d = json.load(f)
    ras: RasModel = RasModel.from_dict(d)
hf_geom = ras.get_plan_geometry(config["hf_plans"], config["mesh_id"])
lf_geom = ras.get_plan_geometry(config["lf_plans"], config["mesh_id"])
hf_data = ras.get_plan_wsels(config["hf_plans"], config["mesh_id"])
lf_data = ras.get_plan_wsels(config["lf_plans"], config["mesh_id"])
weights = ras.get_cell_areas(config["hf_plans"][0], config["mesh_id"])
elevations = ras.get_cell_minimum_elevation(config["hf_plans"][0], config["mesh_id"])

# Preprocess
resampler = CellResampler(hf_geom, lf_geom, "cell_id")
lf_data_resampled = resampler.resample_lf_to_hf(lf_data)
reducer = PreProcessor(wet_threshold=config["wet_threshold_depth"])
reducer.fit(hf_data.values, elevations, weights)
reducer.plot_pca_summary(os.path.join(config["plot_dir"], "pca_summary.png"))
y = reducer.transform(hf_data.values)
x = reducer.transform(lf_data_resampled.values)

# Plot
ec_pairplot(x, y, 5, os.path.join(config["plot_dir"], "pairplot.png"))
ec_timeseries(x, y, 5, hf_data.index, os.path.join(config["plot_dir"], "ec_timeseries"))

# Fit GPR
gpr = GPRAS(config["kernel"])
gpr.fit(x, y, config["inducing_fraction"])
gpr.to_file("gpr.json")

# Predict
mean_pred, var_pred = gpr.predict(x)
y_pred = reducer.reverse_transform(mean_pred)
performance_scatterplot(
    lf_data_resampled.values, hf_data.values, y_pred, os.path.join(config["plot_dir"], "performance_scatterplot.png")
)
performance_cdf(
    lf_data_resampled.values, hf_data.values, y_pred, os.path.join(config["plot_dir"], "performance_cdf.png")
)

# Make GIS layers
hf_geom["lf_cell_id"] = hf_geom["cell_id"].map(resampler.cell_resampler)
hf_geom["wetness_class"] = reducer.wetness_classes
hf_geom["lf_hf_rmse"] = np.mean((lf_data_resampled.values - hf_data.values) ** 2, axis=0) ** 0.5
hf_geom["upskill_hf_rmse"] = np.mean((y_pred - hf_data.values) ** 2, axis=0) ** 0.5
for i in range(reducer.eofs.shape[0]):
    x_full = np.empty(reducer.dry_indices.shape[0])
    x_full[reducer.dry_indices] = np.nan
    x_full[~reducer.dry_indices] = reducer.eofs[i, :]
    hf_geom[f"EOF_{i}"] = x_full
hf_geom.to_file("qc.gpkg", layer="HF_cells")
lf_geom.to_file("qc.gpkg", layer="LF_cells")
