"""Plot flood maps and video for selected TS for LF, HF, and test results."""

# %% import packages
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams["animation.ffmpeg_path"] = "/usr/local/lib/python3.11/dist-packages/ffmpeg"

# %% define paths
# inputs
path_hf_results = r"/workspaces/gpras/production/post_processing/data/df_hf_test_data_depth.csv"
path_lf_results = r"/workspaces/gpras/production/post_processing/data/df_lf_test_data_depth.csv"
path_y_test_results = r"/workspaces/gpras/production/post_processing/data/y_test_pred_depth.csv"
path_mesh_gpkg = r"/workspaces/gpras/qc.gpkg"
path_fluvial_bounds = r"/workspaces/gpras/production/post_processing/GIS/masks/fluvial_mask_gpr40_t55_0ft.shp"

# oututs
path_hf_mesh_gpr40_results_shp = r"/workspaces/gpras/production/post_processing/data/gdf_hf_mesh_gpr40_results.shp"
path_test_mesh_gpr40_results_mp4 = (
    r"/workspaces/gpras/production/post_processing/data/gdf_te_mesh_gpr40_results_ani.mp4"
)
path_test_mesh_gpr40_results_gif = (
    r"/workspaces/gpras/production/post_processing/data/gdf_te_mesh_gpr40_results_ani.gif"
)

path_test_mesh_gpr40_results_gif_fluv = (
    r"/workspaces/gpras/production/post_processing/data/gdf_te_mesh_gpr40_results_fluv_ani.gif"
)

# %% load data
df_hf_results = pd.read_csv(path_hf_results)
df_lf_results = pd.read_csv(path_lf_results)
df_test_results = pd.read_csv(path_y_test_results)
gdf_lf_mesh = gpd.read_file(path_mesh_gpkg, driver="GPKG", layer="LF_cells")
gdf_hf_mesh = gpd.read_file(path_mesh_gpkg, driver="GPKG", layer="HF_cells")
gdf_fluvial_bounds = gpd.read_file(path_fluvial_bounds)

# %% redefine mesh cell id as strings
gdf_hf_mesh["cell_id"] = gdf_hf_mesh["cell_id"].astype(str)
gdf_lf_mesh["cell_id"] = gdf_lf_mesh["cell_id"].astype(str)

# %% define run parameters
gpr = "gpr40"
show_plots = 1

# %% get results for this gpr only
df_hf_results_gpr = df_hf_results.loc[df_hf_results.run == gpr, :].copy()
df_lf_results_gpr = df_lf_results.loc[df_lf_results.run == gpr, :].copy()
df_test_results_gpr = df_test_results.loc[df_test_results.run == gpr, :].copy()

# %% find the hf result ts with the maximum flood depth
df_hf_results_gpr_max = df_hf_results_gpr.iloc[:, 2:].aggregate("mean", axis=1)
if show_plots == 1:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(df_hf_results_gpr_max)), df_hf_results_gpr_max.values)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Max depth across AOI (ft)")
max_ts_gpr = np.argmax(df_hf_results_gpr_max.values)

# %% join all results from max ts to gdf_hf_mesh
df_hf_results_gpr_long = df_hf_results_gpr.transpose().iloc[1:, :].astype("float64").copy()
df_hf_results_gpr_long.columns = df_hf_results_gpr_long.loc["t", :].astype("int").astype("string")
df_hf_results_gpr_long.drop(["t"], inplace=True)
df_hf_results_gpr_long_suf = df_hf_results_gpr_long.add_suffix("_hf_40").add_prefix("t")

df_lf_results_gpr_long = df_lf_results_gpr.transpose().iloc[1:, :].astype("float64").copy()
df_lf_results_gpr_long.columns = df_lf_results_gpr_long.loc["t", :].astype("int").astype("string")
df_lf_results_gpr_long.drop(["t"], inplace=True)
df_lf_results_gpr_long_suf = df_lf_results_gpr_long.add_suffix("_lf_40").add_prefix("t")

df_test_results_gpr_long = df_test_results_gpr.transpose().iloc[1:, :].astype("float64").copy()
df_test_results_gpr_long.columns = df_test_results_gpr_long.loc["t", :].astype("int").astype("string")
df_test_results_gpr_long.drop(["t"], inplace=True)
df_test_results_gpr_long_suf = df_test_results_gpr_long.add_suffix("_te_40").add_prefix("t")

gdf_hf_mesh = gdf_hf_mesh.merge(
    df_hf_results_gpr_long_suf.loc[:, :],  # [str(max_ts_gpr)+'_hf_40']],
    left_on="cell_id",
    right_index=True,
    how="left",
)

gdf_hf_mesh = gdf_hf_mesh.merge(
    df_lf_results_gpr_long_suf.loc[:, :],  # [str(max_ts_gpr)+'_lf_'+gpr]],
    left_on="cell_id",
    right_index=True,
    how="left",
)

gdf_hf_mesh = gdf_hf_mesh.merge(
    df_test_results_gpr_long_suf.loc[:, :],  # [str(max_ts_gpr)+'_te_'+gpr]],
    left_on="cell_id",
    right_index=True,
    how="left",
)


# %% save to geodatabase
gdf_hf_mesh.to_file(path_hf_mesh_gpr40_results_shp)

# %% create animation

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(131)
ax1.set_axis_off()
norm = plt.Normalize(vmin=0, vmax=20, clip=True)
sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
sm.set_array([])  # Only needed for adding the colorbar
colorbar = fig.colorbar(sm, ax=ax1, orientation="horizontal", shrink=0.5, format="%.2f")

ax2 = fig.add_subplot(132)
ax2.set_axis_off()
# Initialize the colorbar variable with a fixed normalization
norm = plt.Normalize(vmin=0, vmax=20, clip=True)
sm = plt.cm.ScalarMappable(
    cmap="Blues",
    norm=norm,
)
sm.set_array([])  # Only needed for adding the colorbar
colorbar = fig.colorbar(sm, ax=ax2, orientation="horizontal", shrink=0.5, format="%.2f")

ax3 = fig.add_subplot(133)
ax3.set_axis_off()
norm = plt.Normalize(vmin=0, vmax=20, clip=True)
sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
sm.set_array([])  # Only needed for adding the colorbar
colorbar = fig.colorbar(sm, ax=ax3, orientation="horizontal", shrink=0.5, format="%.2f")

# Set fixed axis limits
xlim = (gdf_hf_mesh.total_bounds[0], gdf_hf_mesh.total_bounds[2])
ylim = (gdf_hf_mesh.total_bounds[1], gdf_hf_mesh.total_bounds[3])
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax3.set_xlim(xlim)
ax3.set_ylim(ylim)


# Plot initial conditions
boundary1 = gdf_hf_mesh.boundary.plot(ax=ax1, edgecolor="black", linewidth=0.1)
boundary2 = gdf_hf_mesh.boundary.plot(ax=ax2, edgecolor="black", linewidth=0.1)
boundary3 = gdf_hf_mesh.boundary.plot(ax=ax3, edgecolor="black", linewidth=0.1)
boundary_fluvial1 = gdf_fluvial_bounds.boundary.plot(ax=ax1, edgecolor="black", linewidth=0.5)
boundary_fluvial2 = gdf_fluvial_bounds.boundary.plot(ax=ax2, edgecolor="black", linewidth=0.5)
boundary_fluvial3 = gdf_fluvial_bounds.boundary.plot(ax=ax3, edgecolor="black", linewidth=0.5)

# Initialize the colorbar variable with a fixed normalization
# norm = plt.Normalize(vmin=0, vmax=20)
# sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
# sm.set_array([])  # Only needed for adding the colorbar
# colorbar = fig.colorbar(sm, ax=ax1, orientation='horizontal', shrink=0.5, format='%.2f')
plt.tight_layout()


def animate(timestep: int) -> None:
    """Animate flood maps."""
    ax1.clear()
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_axis_off()
    ax2.clear()
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_axis_off()
    ax3.clear()
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.set_axis_off()

    # Plot initial boundaries
    boundary1 = gdf_hf_mesh.boundary.plot(ax=ax1, edgecolor="black", linewidth=0.1)
    boundary2 = gdf_hf_mesh.boundary.plot(ax=ax2, edgecolor="black", linewidth=0.1)
    boundary3 = gdf_hf_mesh.boundary.plot(ax=ax3, edgecolor="black", linewidth=0.1)
    boundary_fluvial1 = gdf_fluvial_bounds.boundary.plot(ax=ax1, edgecolor="black", linewidth=0.5)
    boundary_fluvial2 = gdf_fluvial_bounds.boundary.plot(ax=ax2, edgecolor="black", linewidth=0.5)
    boundary_fluvial3 = gdf_fluvial_bounds.boundary.plot(ax=ax3, edgecolor="black", linewidth=0.5)

    # Plot the data for the current year
    gdf_hf_mesh.plot(ax=ax1, column="t" + str(timestep) + "_lf_40", legend=False, cmap="Blues", norm=norm)
    gdf_hf_mesh.plot(ax=ax2, column="t" + str(timestep) + "_te_40", legend=False, cmap="Blues", norm=norm)
    gdf_hf_mesh.plot(ax=ax3, column="t" + str(timestep) + "_hf_40", legend=False, cmap="Blues", norm=norm)

    # Add year annotation at the top
    ax1.annotate(f"gpr40 Low-Fidelity: {timestep}", xy=(0.5, 1.05), xycoords="axes fraction", fontsize=12, ha="center")
    ax2.annotate(f"gpr40 Upskilled: {timestep}", xy=(0.5, 1.05), xycoords="axes fraction", fontsize=12, ha="center")
    ax3.annotate(f"gpr40 High-fidelity: {timestep}", xy=(0.5, 1.05), xycoords="axes fraction", fontsize=12, ha="center")

    plt.tight_layout()

    # unused script to pass Ruff review
    boundary1 = boundary1
    boundary2 = boundary2
    boundary3 = boundary3
    boundary_fluvial1 = boundary_fluvial1
    boundary_fluvial2 = boundary_fluvial2
    boundary_fluvial3 = boundary_fluvial3


# Create the animation
list_timestep = np.arange(0, 31)
# list_frame = ['t'+str(x)+'_te_40' for x in np.arange(1,20)]
animation = FuncAnimation(fig, animate, frames=list_timestep, repeat=False, interval=1000)


# Save the animation as a GIF
# writer = FFMpegWriter(fps=0.5, metadata=dict(artist='Me'), bitrate=1800)
writer = PillowWriter(fps=1)
animation.save(path_test_mesh_gpr40_results_gif_fluv, dpi=300, writer=writer)

plt.show()


# %%


# %%
