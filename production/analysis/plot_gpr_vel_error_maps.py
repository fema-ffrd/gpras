"""Plot velocity results."""

# %% import packages

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# %% define paths

# inputs
path_ras_results = r"/workspaces/gpras/data/ras_upskill_velocity/metrics/ras_upskill_cell_metrics_20251021.xlsx"
path_hms_results = r"/workspaces/gpras/data/hms_upskill_velocity/metrics/hms_upskill_cell_metrics_20251021.xlsx"
path_mesh_gpkg = r"/workspaces/gpras/data/shared/qc.gpkg"

# outputs

# %% open mesh and load data
gdf_hf_mesh = gpd.read_file(path_mesh_gpkg, layer="HF_cells")
df_ras_results = pd.read_excel(path_ras_results)
df_hms_results = pd.read_excel(path_hms_results)

# %% calcualte merge results to mesh for _ras and _hms
# mean_mts_v  mean max velocity at each cell averaged over test plans
# mean_rmse_v  RMSE of errors across all timestep for each testing plan, averaged over all testing plans.
# rmse_mts_v Error at the maximum timestep for each testing plan, with RMSE taken over all testing plans.


def calculate_rmse(series: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate rmse."""
    return np.sqrt(np.mean(np.square(series)))


# df_mean_mts_v_ras = pd.pivot_table(df_ras_results, values='hf_val_cell_mts', index=['cell_id', ])
df_ras_agg = df_ras_results.groupby("cell_id").agg(
    {"hf_val_cell_mts": ["mean"], "rmse_cell_toi": ["mean"], "err_cell_mts": [calculate_rmse]}
)
df_ras_agg.columns = ["mean_mts_v", "mean_rmse_v", "rmse_err_mts"]

df_ras_agg["rel_mean_mts_v"] = df_ras_agg["rmse_err_mts"] / df_ras_agg["mean_mts_v"]

gdf_hf_mesh_ras = gdf_hf_mesh.merge(df_ras_agg, left_on="cell_id", right_index=True, how="left")


# %% calcualte merge results to mesh for _ras and _hms
# mean_mts_v  mean max velocity at each cell averaged over test plans
# mean_rmse_v  RMSE of errors across all timestep for each testing plan, averaged over all testing plans.
# rmse_mts_v Error at the maximum timestep for each testing plan, with RMSE taken over all testing plans.


# df_mean_mts_v_ras = pd.pivot_table(df_ras_results, values='hf_val_cell_mts', index=['cell_id', ])
df_hms_agg = df_hms_results.groupby("cell_id").agg(
    {"hf_val_cell_mts": ["mean"], "rmse_cell_toi": ["mean"], "err_cell_mts": [calculate_rmse]}
)
df_hms_agg.columns = ["mean_mts_v", "mean_rmse_v", "rmse_err_mts"]

df_hms_agg["rel_mean_mts_v"] = df_hms_agg["rmse_err_mts"] / df_hms_agg["mean_mts_v"]

gdf_hf_mesh_hms = gdf_hf_mesh.merge(df_hms_agg, left_on="cell_id", right_index=True, how="left")

# %% plot max velocity


fig, ax = plt.subplots(figsize=(12, 8))
gdf_hf_mesh_ras.plot(
    column="mean_mts_v",
    ax=ax,
    cmap="jet",
    vmin=0,
    vmax=0.9,
    edgecolor="none",
    legend=True,
    legend_kwds={"label": "Velocity (fps)"},
)

# Set equal aspect ratio and adjust limits
ax.set_aspect("equal")
ax.autoscale_view()

plt.title("RMSE of Maximum Timestep Residual", fontsize=16, fontweight="bold")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
# plt.savefig(Path(output_plot_path))
# plt.close(fig)

# %% plot max velocity - HMS

fig, ax = plt.subplots(figsize=(12, 8))
gdf_hf_mesh_hms.plot(
    column="rmse_err_mts",
    ax=ax,
    cmap="jet",
    vmin=0,
    vmax=0.9,
    edgecolor="none",
    legend=True,
    legend_kwds={"label": "Velocity (fps)"},
)

# Set equal aspect ratio and adjust limits
ax.set_aspect("equal")
ax.autoscale_view()

plt.title("RMSE of Maximum Timestep Residual", fontsize=16, fontweight="bold")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
# plt.savefig(Path(output_plot_path))
# plt.close(fig)

# %%
