"""Plot flood maps and video for selected TS for LF, HF, and test results."""

# %% import packages
import os

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
path_y_high_test_results = r"/workspaces/gpras/production/post_processing/data/y_high_test_pred_depth.csv"
path_y_low_test_results = r"/workspaces/gpras/production/post_processing/data/y_low_test_pred_depth.csv"
path_mesh_gpkg = r"/workspaces/gpras/qc.gpkg"
path_fluvial_bounds = r"/workspaces/gpras/production/post_processing/GIS/masks/fluvial_mask_gpr40_t55_0ft.shp"

# oututs
filename_hf_mesh_gpr40_results_shp = "gdf_hf_mesh_gpr_results.shp"
path_test_mesh_gpr40_results_mp4 = (
    r"/workspaces/gpras/production/post_processing/data/gdf_te_mesh_gpr40_results_ani.mp4"
)
path_test_mesh_gpr40_results_gif = (
    r"/workspaces/gpras/production/post_processing/data/gdf_te_mesh_gpr40_results_ani.gif"
)

path_test_mesh_gpr40_results_gif_fluv = (
    r"/workspaces/gpras/production/post_processing/data/gdf_te_mesh_gpr40_results_fluv_ani.gif"
)

# %% define run parameters
event_id_list = np.arange(36, 50)
for event_id in event_id_list:
    print(f"Plotting event id {event_id}")
    gpr = "gpr" + str(event_id)
    show_plots = 1
    plot_animation = True
    save_event_gdf = False

    # define paths
    path_plots = rf"/workspaces/gpras/production/post_processing/plots/{gpr}//"
    filename_hf_mesh_gpr_results_shp = f"gdf_hf_mesh_gpr{event_id}_results.shp"
    filename_gif = f"gdf_te_mesh_gpr{event_id}_results_ani.gif"
    filename_ts_pts = f"timeseries_mean_select_cells_gpr{event_id}.png"
    filename_ts_pts_map = f"map_mean_select_cells_gpr{event_id}.png"

    # make directory for results
    folder_plots = rf"/workspaces/gpras/production/post_processing/plots/{gpr}//"
    os.makedirs(folder_plots, exist_ok=True)

    #  load data
    df_hf_results = pd.read_csv(path_hf_results)
    df_lf_results = pd.read_csv(path_lf_results)
    df_test_results = pd.read_csv(path_y_test_results)
    df_test_high_results = pd.read_csv(path_y_high_test_results)
    df_test_low_results = pd.read_csv(path_y_low_test_results)
    gdf_lf_mesh = gpd.read_file(path_mesh_gpkg, driver="GPKG", layer="LF_cells")
    gdf_hf_mesh = gpd.read_file(path_mesh_gpkg, driver="GPKG", layer="HF_cells")
    gdf_fluvial_bounds = gpd.read_file(path_fluvial_bounds)

    #  redefine mesh cell id as strings
    gdf_hf_mesh["cell_id"] = gdf_hf_mesh["cell_id"].astype(str)
    gdf_lf_mesh["cell_id"] = gdf_lf_mesh["cell_id"].astype(str)

    #  get results for this gpr only
    df_hf_results_gpr = df_hf_results.loc[df_hf_results.run == gpr, :].copy()
    df_lf_results_gpr = df_lf_results.loc[df_lf_results.run == gpr, :].copy()
    df_test_results_gpr = df_test_results.loc[df_test_results.run == gpr, :].copy()
    df_test_high_results_gpr = df_test_high_results.loc[df_test_high_results.run == gpr, :].copy()
    df_test_low_results_gpr = df_test_low_results.loc[df_test_low_results.run == gpr, :].copy()
    N_ts = df_hf_results_gpr.loc[df_hf_results_gpr.run == gpr, "t"].values.max()

    #  find the hf result ts with the maximum flood depth
    df_hf_results_gpr_max = df_hf_results_gpr.iloc[:, 2:].aggregate("mean", axis=1)
    if show_plots == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.arange(len(df_hf_results_gpr_max)), df_hf_results_gpr_max.values)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Max depth across AOI (ft)")
    max_ts_gpr = np.argmax(df_hf_results_gpr_max.values)

    #  join all results from max ts to gdf_hf_mesh
    df_hf_results_gpr_long = df_hf_results_gpr.transpose().iloc[1:, :].astype("float64").copy()
    df_hf_results_gpr_long.columns = df_hf_results_gpr_long.loc["t", :].astype("int").astype("string")
    df_hf_results_gpr_long.drop(["t"], inplace=True)
    df_hf_results_gpr_long_suf = df_hf_results_gpr_long.add_suffix(f"_hf_{event_id}").add_prefix("t")

    df_lf_results_gpr_long = df_lf_results_gpr.transpose().iloc[1:, :].astype("float64").copy()
    df_lf_results_gpr_long.columns = df_lf_results_gpr_long.loc["t", :].astype("int").astype("string")
    df_lf_results_gpr_long.drop(["t"], inplace=True)
    df_lf_results_gpr_long_suf = df_lf_results_gpr_long.add_suffix(f"_lf_{event_id}").add_prefix("t")

    df_test_results_gpr_long = df_test_results_gpr.transpose().iloc[1:, :].astype("float64").copy()
    df_test_results_gpr_long.columns = df_test_results_gpr_long.loc["t", :].astype("int").astype("string")
    df_test_results_gpr_long.drop(["t"], inplace=True)
    df_test_results_gpr_long_suf = df_test_results_gpr_long.add_suffix(f"_te_{event_id}").add_prefix("t")

    df_test_high_results_gpr_long = df_test_high_results_gpr.transpose().iloc[1:, :].astype("float64").copy()
    df_test_high_results_gpr_long.columns = df_test_high_results_gpr_long.loc["t", :].astype("int").astype("string")
    df_test_high_results_gpr_long.drop(["t"], inplace=True)
    df_test_high_results_gpr_long_suf = df_test_high_results_gpr_long.add_suffix(f"_th_{event_id}").add_prefix("t")

    df_test_low_results_gpr_long = df_test_low_results_gpr.transpose().iloc[1:, :].astype("float64").copy()
    df_test_low_results_gpr_long.columns = df_test_low_results_gpr_long.loc["t", :].astype("int").astype("string")
    df_test_low_results_gpr_long.drop(["t"], inplace=True)
    df_test_low_results_gpr_long_suf = df_test_low_results_gpr_long.add_suffix(f"_tl_{event_id}").add_prefix("t")

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

    gdf_hf_mesh = gdf_hf_mesh.merge(
        df_test_high_results_gpr_long_suf.loc[:, :],  # [str(max_ts_gpr)+'_te_'+gpr]],
        left_on="cell_id",
        right_index=True,
        how="left",
    )

    gdf_hf_mesh = gdf_hf_mesh.merge(
        df_test_low_results_gpr_long_suf.loc[:, :],  # [str(max_ts_gpr)+'_te_'+gpr]],
        left_on="cell_id",
        right_index=True,
        how="left",
    )

    #  save to geodatabase
    if save_event_gdf:
        gdf_hf_mesh.to_file(os.path.join(path_plots, filename_hf_mesh_gpr40_results_shp))

    #  create animation

    if plot_animation:

        fig = plt.figure(figsize=(12, 5))

        ax1 = fig.add_subplot(131)
        ax1.set_axis_off()
        norm = plt.Normalize(vmin=0, vmax=10, clip=True)
        sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
        sm.set_array([])  # Only needed for adding the colorbar
        colorbar = fig.colorbar(sm, ax=ax1, orientation="horizontal", shrink=0.5, format="%.2f")

        ax2 = fig.add_subplot(132)
        ax2.set_axis_off()
        # Initialize the colorbar variable with a fixed normalization
        norm = plt.Normalize(vmin=0, vmax=10, clip=True)
        sm = plt.cm.ScalarMappable(
            cmap="Blues",
            norm=norm,
        )
        sm.set_array([])  # Only needed for adding the colorbar
        colorbar = fig.colorbar(sm, ax=ax2, orientation="horizontal", shrink=0.5, format="%.2f")

        ax3 = fig.add_subplot(133)
        ax3.set_axis_off()
        norm = plt.Normalize(vmin=0, vmax=10, clip=True)
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
            ax1.clear()  # noqa: B023
            ax1.set_xlim(xlim)  # noqa: B023
            ax1.set_ylim(ylim)  # noqa: B023
            ax1.set_axis_off()  # noqa: B023
            ax2.clear()  # noqa: B023
            ax2.set_xlim(xlim)  # noqa: B023
            ax2.set_ylim(ylim)  # noqa: B023
            ax2.set_axis_off()  # noqa: B023
            ax3.clear()  # noqa: B023
            ax3.set_xlim(xlim)  # noqa: B023
            ax3.set_ylim(ylim)  # noqa: B023
            ax3.set_axis_off()  # noqa: B023

            # Plot initial boundaries
            boundary1 = gdf_hf_mesh.boundary.plot(ax=ax1, edgecolor="black", linewidth=0.1)  # noqa: B023
            boundary2 = gdf_hf_mesh.boundary.plot(ax=ax2, edgecolor="black", linewidth=0.1)  # noqa: B023
            boundary3 = gdf_hf_mesh.boundary.plot(ax=ax3, edgecolor="black", linewidth=0.1)  # noqa: B023
            boundary_fluvial1 = gdf_fluvial_bounds.boundary.plot(ax=ax1, edgecolor="black", linewidth=0.5)  # noqa: B023
            boundary_fluvial2 = gdf_fluvial_bounds.boundary.plot(ax=ax2, edgecolor="black", linewidth=0.5)  # noqa: B023
            boundary_fluvial3 = gdf_fluvial_bounds.boundary.plot(ax=ax3, edgecolor="black", linewidth=0.5)  # noqa: B023

            # Plot the data for the current year
            gdf_hf_mesh.plot(  # noqa: B023
                ax=ax1,  # noqa: B023
                column="t" + str(timestep) + f"_lf_{event_id}",  # noqa: B023
                legend=False,
                cmap="Blues",
                norm=norm,  # noqa: B023
            )
            gdf_hf_mesh.plot(  # noqa: B023
                ax=ax2,  # noqa: B023
                column="t" + str(timestep) + f"_te_{event_id}",  # noqa: B023
                legend=False,
                cmap="Blues",
                norm=norm,  # noqa: B023
            )
            gdf_hf_mesh.plot(  # noqa: B023
                ax=ax3,  # noqa: B023
                column="t" + str(timestep) + f"_hf_{event_id}",  # noqa: B023
                legend=False,
                cmap="Blues",
                norm=norm,  # noqa: B023
            )

            # Add year annotation at the top
            ax1.annotate(  # noqa: B023
                f"gpr{event_id} Low-Fidelity: {timestep}",  # noqa: B023
                xy=(0.5, 1.05),
                xycoords="axes fraction",
                fontsize=12,
                ha="center",
            )
            ax2.annotate(  # noqa: B023
                f"gpr{event_id} Upskilled: {timestep}",  # noqa: B023
                xy=(0.5, 1.05),
                xycoords="axes fraction",
                fontsize=12,
                ha="center",
            )
            ax3.annotate(  # noqa: B023
                f"gpr{event_id} High-fidelity: {timestep}",  # noqa: B023
                xy=(0.5, 1.05),
                xycoords="axes fraction",
                fontsize=12,
                ha="center",
            )

            plt.tight_layout()

            # unused script to pass Ruff review
            boundary1 = boundary1
            boundary2 = boundary2
            boundary3 = boundary3
            boundary_fluvial1 = boundary_fluvial1
            boundary_fluvial2 = boundary_fluvial2
            boundary_fluvial3 = boundary_fluvial3

        # Create the animation
        list_timestep = np.arange(0, N_ts)
        # list_frame = ['t'+str(x)+'_te_40' for x in np.arange(1,20)]
        animation = FuncAnimation(fig, animate, frames=list_timestep, repeat=False, interval=500)

        # Save the animation as a GIF
        # writer = FFMpegWriter(fps=0.5, metadata=dict(artist='Me'), bitrate=1800)
        writer = PillowWriter(fps=1)
        animation.save(
            os.path.join(path_plots, filename_gif),
            dpi=250,
            writer=writer,
        )

        plt.show()

    #
    list_cell_ids = ["38372", "37899", "38614", "38064", "53880", "51292", "2305", "87788"]
    list_cell_labels = [
        " A. trinity_river_inlet",
        " B. trinity_river_outlet",
        " C. trinity_river_mid_W",
        " D. trinity_river_mid_E",
        " E. trinity_branch_N",
        " F. trinity_branch_S",
        " G. upslope_NW",
        " H. upslope_NE",
    ]
    #  create plot
    fig = plt.figure(figsize=(6.5, 4))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    gdf_hf_pts = gdf_hf_mesh[["cell_id", "geometry"]].copy()
    gdf_hf_pts.index = gdf_hf_pts["cell_id"].copy()
    gdf_hf_pts["is_pt"] = gdf_hf_pts["cell_id"].isin(list_cell_ids)
    gdf_hf_pts.plot(ax=ax, column="is_pt", cmap="binary", edgecolor="black", linewidth=0.2)
    for i, idx in enumerate(list_cell_ids):
        coord_x = gdf_hf_pts.centroid.x[idx]
        coord_y = gdf_hf_pts.centroid.y[idx]
        ax.text(coord_x, coord_y, list_cell_labels[i])
        ax.plot(coord_x, coord_y, "o", color="r")
    plt.suptitle("Location of Select Cells for Plotting")
    plt.savefig(os.path.join(path_plots, filename_ts_pts_map), dpi=300)

    #
    list_ts = np.arange(N_ts)
    fig = plt.figure(figsize=(6.5, 8))
    plt.suptitle(f"Timeseries of Depth at Select Cells in gpr{event_id}\n")
    for i in np.arange(8):
        ax = fig.add_subplot(4, 2, i + 1)
        this_cell_id = list_cell_ids[i]
        this_label = list_cell_labels[i]
        hf_vals = [
            gdf_hf_mesh.loc[gdf_hf_mesh.cell_id == this_cell_id, f"t{x}_hf_{event_id}"].values[0] for x in list_ts
        ]
        lf_vals = [
            gdf_hf_mesh.loc[gdf_hf_mesh.cell_id == this_cell_id, f"t{x}_lf_{event_id}"].values[0] for x in list_ts
        ]
        pred_vals = [
            gdf_hf_mesh.loc[gdf_hf_mesh.cell_id == this_cell_id, f"t{x}_te_{event_id}"].values[0] for x in list_ts
        ]
        pred_vals_high = [
            gdf_hf_mesh.loc[gdf_hf_mesh.cell_id == this_cell_id, f"t{x}_th_{event_id}"].values[0] for x in list_ts
        ]
        pred_vals_low = [
            gdf_hf_mesh.loc[gdf_hf_mesh.cell_id == this_cell_id, f"t{x}_tl_{event_id}"].values[0] for x in list_ts
        ]
        ax.plot(list_ts, hf_vals, "k-", label="HF Model")
        ax.plot(list_ts, pred_vals, "k--", label="GPR Model")
        ax.plot(list_ts, lf_vals, color="0.4", ls="-", alpha=0.5, label="LF Model")
        plt.fill_between(list_ts, pred_vals_low, pred_vals_high, color="skyblue", alpha=0.5, ec=None)
        if i == 1:
            ax.legend(loc="upper right", frameon=False)
        if i in [6, 7]:
            ax.set_xlabel("Time step")
        if i in [0, 2, 4, 6]:
            ax.set_ylabel("Cell Depth (ft)")
        ax.set_title(f"{list_cell_labels[i]}: #{list_cell_ids[i]}")
        plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(os.path.join(path_plots, filename_ts_pts), dpi=300)


# %%


# %%
