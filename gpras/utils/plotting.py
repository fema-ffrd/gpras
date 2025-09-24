"""Utilities for generating diagnostic and QC plots."""

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as MplPolygon
from numpy.typing import NDArray

COMMON_COLORS = ["#0084FF", "#FF7F0E", "#009E73", "#E69F00", "#CC79A7"]


def apply_formatting(fig: Figure, ax: Axes | Sequence[Axes]) -> None:
    """Apply consistent formatting to plots."""
    plt.rcParams["font.family"] = "DejaVu Sans"
    axs = [ax] if isinstance(ax, Axes) else list(ax)

    fig.patch.set_facecolor("white")

    for a in axs:
        a.set_facecolor("#f7f7f7")
        for spine in a.spines.values():
            spine.set_color("#666666")
            spine.set_linewidth(0.8)
        a.grid(True, color="#cccccc", linestyle="--", linewidth=0.7, alpha=0.7)
        a.set_axisbelow(True)
        a.tick_params(axis="both", which="major", color="#666666", labelsize=10, length=5)
        a.tick_params(axis="both", which="minor", color="#999999", labelsize=8, length=3)
        a.xaxis.label.set_size(10)
        a.yaxis.label.set_size(10)
        a.xaxis.label.set_weight("medium")
        a.yaxis.label.set_weight("medium")
        a.title.set_size(14)
        a.title.set_weight("heavy")
        legend = a.get_legend()
        if legend:
            legend.set_frame_on(False)
            legend.get_frame().set_alpha(0.0)

    fig.tight_layout()


def ec_pairplot(
    x: NDArray[Any],
    y: NDArray[Any],
    modes_to_plot: int,
    out_path: str | Path,
    inducing_points: NDArray[Any] | None = None,
) -> None:
    """Generate a Seaborn pairplot comparing low-fidelity and high-fidelity EOF mode values.

    This plot is useful for assesing how closely LF modes are to HF modes (diagonal plots).
    It is also useful for exploring whether distinct trends exist within each row that the GPR could leverage for
    prediction.

    Args:
        x (NDArray[Any]): Low-fidelity EOF coefficients with shape (n_samples, n_modes).
        y (NDArray[Any]): High-fidelity EOF coefficients with shape (n_samples, n_modes).
        modes_to_plot (int): Number of EOF modes to include in the plot.
        out_path (str): Location to save the plot.
        inducing_points (NDArray[Any], optional): Fitted inducing point for the sparse GPR (n_pts, n_modes).

    Returns:
        None
    """
    x_cols = [f"EOF_{i}_LF" for i in range(modes_to_plot)]
    y_cols = [f"EOF_{i}_HF" for i in range(modes_to_plot)]
    df_x = pd.DataFrame(x[:, :modes_to_plot], columns=x_cols)
    df_y = pd.DataFrame(y[:, :modes_to_plot], columns=y_cols)
    df = pd.concat([df_x, df_y], axis=1)
    g = sns.pairplot(
        df, x_vars=x_cols, y_vars=y_cols, plot_kws={"marker": "+", "linewidth": 1, "size": 1, "color": COMMON_COLORS[0]}
    )
    for i, (x_col, y_col) in enumerate(zip(x_cols, y_cols, strict=False)):
        ax = g.axes[i, i]
        x_min, x_max = df[x_col].min(), df[x_col].max()
        y_min, y_max = df[y_col].min(), df[y_col].max()
        min_val = min(x_min, y_min)
        max_val = max(x_max, y_max)
        ax.plot([min_val, max_val], [min_val, max_val], color="k", linestyle="--", linewidth=2.5)

    if inducing_points is not None:
        for i in range(len(x_cols)):
            for j in range(len(y_cols)):
                ax = g.axes[i, j]
                ax.scatter(inducing_points[:, j], inducing_points[:, i], alpha=0.6, color="red", marker="x", s=100)
    apply_formatting(g.figure, g.axes.flatten())
    g.savefig(Path(out_path))


def ec_timeseries(
    x: NDArray[Any],
    y: NDArray[Any],
    modes_to_plot: int,
    ind: pd.Index,
    out_dir: str | Path,
    low_est: NDArray[Any] | None = None,
    est: NDArray[Any] | None = None,
    high_est: NDArray[Any] | None = None,
) -> None:
    """Plot EOF time series for low- and high-fidelity models by event plan.

    Args:
        x (NDArray[Any]): Low-fidelity EOF coefficients with shape (n_samples, n_modes).
        y (NDArray[Any]): High-fidelity EOF coefficients with shape (n_samples, n_modes).
        modes_to_plot (int): Number of EOF modes to plot.
        out_dir (str): Dirctory to save the plots.
        ind (pd.Index): Index object (e.g., MultiIndex) used to group samples by plan.
        low_est (NDArray[Any], optional): Low-fidelity EOF coefficients upper CI.
        est (NDArray[Any], optional): Low-fidelity EOF coefficients mean estimate.
        high_est (NDArray[Any], optional): Low-fidelity EOF coefficients upper CI.

    Returns:
        None
    """
    events = np.unique(ind.get_level_values(0), return_counts=True)
    cum_index = 0
    for event_label, count in zip(*events, strict=False):
        fig, axs = plt.subplots(nrows=modes_to_plot, figsize=(6.5, 2 * modes_to_plot), sharex=True)
        for i, ax in enumerate(axs):
            ax.plot(y[cum_index : cum_index + count, i], label="HF model", c=COMMON_COLORS[0])
            ax.plot(x[cum_index : cum_index + count, i], label="LF model", c=COMMON_COLORS[1])
            if low_est is not None and est is not None and high_est is not None:
                ax.plot(est[cum_index : cum_index + count, i], label="GPR", c="k")
                ax.fill_between(
                    np.arange(count),
                    low_est[cum_index : cum_index + count, i],
                    high_est[cum_index : cum_index + count, i],
                    label="CI",
                    fc="k",
                    alpha=0.1,
                )
            ax.set_ylabel(f"EOF_{i}")
            ax.set_yticks([], labels=[])
        cum_index += count
        axs[0].legend()
        axs[-1].set_xlabel("Timestep")
        fig.suptitle(f"Plan {event_label}")
        apply_formatting(fig, axs)
        fig.savefig(Path(out_dir) / f"Plan_{event_label}.png")
        plt.close(fig)


def performance_scatterplot(
    lf: NDArray[Any], hf: NDArray[Any], lf_upskill: NDArray[Any], out_path: str | Path, depth: bool = False
) -> None:
    """Plot scatterplots comparing low-fidelity vs high-fidelity and upskilled vs high-fidelity models depth estimates.

    Args:
        lf (NDArray[Any]): Low-fidelity model output (e.g., water surface elevations).
        hf (NDArray[Any]): High-fidelity model output.
        lf_upskill (NDArray[Any]): Output of the upskilled low-fidelity model.
        out_path (str): Location to save the plot.
        depth (bool): Whether the data is depth (true) or WSE (false)

    Returns:
        None
    """
    lf, hf, lf_upskill = lf.flatten(), hf.flatten(), lf_upskill.flatten()

    fig, axs = plt.subplots(ncols=2, figsize=(6.5, 4), sharey=True)
    metric = "Depth" if depth else "WSE"

    axs[0].scatter(lf, hf, s=1, c=COMMON_COLORS[0], alpha=0.8)
    ll, ur = min([lf.min(), hf.min()]), max([lf.max(), hf.max()])
    axs[0].plot((ll, ur), (ll, ur), ls="dashed", c="k")
    rmse = np.mean((lf - hf) ** 2) ** 0.5
    axs[0].text(0.95, 0.05, f"rmse: {round(rmse, 2)}", ha="right", va="bottom", transform=axs[0].transAxes)
    axs[0].set_ylabel(f"High-fidelity Model {metric} (ft)")
    axs[0].set_xlabel(f"Low-fidelity Model {metric} (ft)")

    axs[1].scatter(lf_upskill, hf, s=1, c=COMMON_COLORS[0], alpha=0.8)
    ll, ur = min([lf_upskill.min(), hf.min()]), max([lf_upskill.max(), hf.max()])
    axs[1].plot((ll, ur), (ll, ur), ls="dashed", c="k")
    rmse = np.mean((lf_upskill - hf) ** 2) ** 0.5
    axs[1].text(0.95, 0.05, f"rmse: {round(rmse, 2)}", ha="right", va="bottom", transform=axs[1].transAxes)
    axs[1].set_xlabel(f"Upskilled Model {metric} (ft)")
    apply_formatting(fig, axs)
    fig.savefig(Path(out_path))
    plt.close(fig)


def performance_cdf(lf: NDArray[Any], hf: NDArray[Any], lf_upskill: NDArray[Any], out_path: str | Path) -> None:
    """Plot cumulative distribution of absolute error for low-fidelity and upskilled models.

    Args:
        lf (NDArray[Any]): Low-fidelity model output.
        hf (NDArray[Any]): High-fidelity model output.
        lf_upskill (NDArray[Any]): Output of the upskilled low-fidelity model.
        out_path (str): Location to save the plot.

    Returns:
        None
    """
    lf_residual = np.sort(np.abs(lf - hf).flatten())
    upskill_residual = np.sort(np.abs(lf_upskill - hf).flatten())
    pcts = np.linspace(0, 100, len(lf_residual))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(lf_residual, pcts, label="Low-Fidelity Model", c=COMMON_COLORS[0])
    ax.plot(upskill_residual, pcts, label="Upskilled Model", c=COMMON_COLORS[1])
    ax.set_ylabel("Percent of Cells")
    ax.set_xlabel("Absolute Error Less Than (ft)")
    ax.legend()
    apply_formatting(fig, ax)
    fig.savefig(Path(out_path))
    plt.close(fig)


def plot_pca_summary(
    eigenvalues: NDArray[Any], n_samples_fit: int, spatial_mode_count: int, out_path: str | Path
) -> None:
    """Plot a summary of PCA eigenvalues with uncertainty and highlight number of selected modes."""
    stderr = np.sqrt(2 / n_samples_fit) * eigenvalues
    inds = np.arange(eigenvalues.shape[0])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(inds, eigenvalues, c="k", label="Eigenvalues")
    ax.fill_between(inds, eigenvalues - stderr, eigenvalues + stderr, color="gray", alpha=0.3, label="± Error")
    ax.axvline(x=spatial_mode_count - 1, color="red", ls="--", label=f"Selected Modes ({spatial_mode_count})")

    ax.set_xlabel("EOF Mode Index")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("Variance Explained by EOF Modes")
    ax.legend()
    ax.grid(True)
    apply_formatting(fig, ax)
    fig.savefig(out_path)
    plt.close(fig)


def ec_timeseries_alt(x: pd.DataFrame, y: pd.DataFrame, out_dir: str | Path) -> None:
    """Plot EOF time series for low- and high-fidelity models by event plan (alt: all LF ECs on every plot)."""
    modes_to_plot = len(y.columns)
    for event_label in x.index.get_level_values(0).unique():
        fig, axs = plt.subplots(nrows=modes_to_plot, figsize=(6.5, 4 * modes_to_plot), sharex=True)
        sub_x = x.loc[event_label]
        sub_y = y.loc[event_label]
        for i, ax in enumerate(axs):
            ax.plot(sub_y[sub_y.columns[i]], label="HF model", c="k", lw=2)
            for j in range(len(sub_x.columns)):
                ax.plot(sub_x[sub_x.columns[j]], label=sub_x.columns[j], alpha=0.6, lw=1)
            ax.set_ylabel(sub_y.columns[i])
            ax.set_yticks([], labels=[])
        axs[0].legend()
        axs[-1].set_xlabel("Timestep")
        fig.suptitle(f"Plan {event_label}")
        apply_formatting(fig, axs)
        fig.savefig(Path(out_dir) / f"Plan_{event_label}.png", dpi=600)
        plt.close(fig)


def appr_3_pairplot(x: pd.DataFrame, y: pd.DataFrame, out_dir: str | Path) -> None:
    """Plot EOF time series for low- and high-fidelity models by event plan (alt: all LF ECs on every plot)."""
    rows = len(y.columns)
    cols = len(x.columns)
    for event_label in x.index.get_level_values(0).unique():
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(4 * cols, 4 * rows))
        sub_x = x.loc[event_label]
        sub_y = y.loc[event_label]
        for ind_x, _x in enumerate(x.columns):
            for ind_y, _y in enumerate(y.columns):
                if ind_y > ind_x:
                    continue
                axs[ind_y, ind_x].scatter(sub_x[_x], sub_y[_y], c="k", alpha=0.5)
        for ind, _y in enumerate(y.columns):
            axs[ind, 0].set_ylabel(_y)
        for ind, _x in enumerate(x.columns):
            axs[ind, 0].set_xlabel(_x)
        fig.suptitle(f"Plan {event_label}")
        # apply_formatting(fig, axs)
        fig.savefig(Path(out_dir) / f"Plan_{event_label}.png", dpi=600)
        plt.close(fig)


def ts_clipping(arr: NDArray[Any], cutoffs: tuple[int, int], out_path: str) -> None:
    """Plot changes in WSE/feature value across timesteps and visualize what's clipped out."""
    arr = arr[:, np.any(arr > 0, axis=0)]
    dx_dt_max = np.quantile(arr, 0.99, axis=1)
    dx_dt_ave = np.sum(arr, axis=1) / np.sum(arr)
    cum_dx_dt = np.cumsum(arr, axis=0)
    cum_dx_dt_max = np.quantile(cum_dx_dt, 0.01, axis=1)
    cum_dx_dt_ave = np.sum(cum_dx_dt, axis=1) / cum_dx_dt.shape[1]

    fig, axs = plt.subplots(nrows=2, figsize=(6.5, 4), sharex=True)

    axs[0].plot(dx_dt_max, c="k", alpha=0.5, lw=1, label="99th percentile")
    axs[0].plot(dx_dt_ave, c="k", label="average")
    axs[1].plot(cum_dx_dt_max, c="k", alpha=0.5, lw=1, label="99th percentile")
    axs[1].plot(cum_dx_dt_ave, c="k", label="average")
    axs[0].axvline(cutoffs[0], ls="dashed", c="r")
    axs[1].axvline(cutoffs[0], ls="dashed", c="r")
    axs[0].axvline(cutoffs[1], ls="dashed", c="r")
    axs[1].axvline(cutoffs[1], ls="dashed", c="r")

    axs[1].set_xlabel("Timestep Index")
    axs[0].set_ylabel("dx/dt")
    axs[1].set_ylabel("CDF of dx/dt")
    fig.suptitle("Changes in Cell/Feature Values")

    axs[1].legend()
    apply_formatting(fig, axs)
    fig.savefig(out_path)
    plt.close(fig)


def map_mesh_errors(
    mesh_df: gpd.GeoDataFrame,
    error_db_path: str | Path,
    output_plot_path: str | Path,
    suffix: str,
    error_field: str = "rmse_cell_toi",
    error_metric: str = "RMSE",
) -> gpd.GeoDataFrame:
    """Map error values onto mesh polygons and create a visualization.

    Parameters:
        mesh_df: GeoDataFrame with at least ['cell_id', 'geometry'].
        error_db_path: Path to SQLite database containing a 'cell_metrics' table.
        output_plot_path: Path to write the output plot.
        error_type: Column in 'cell_metrics' representing the error metric to map.

    Returns:
        GeoDataFrame merged with error values (column 'error_value').
    """
    if mesh_df.empty:
        raise ValueError("mesh_df is empty; nothing to plot.")

    cell_ids = mesh_df["cell_id"].tolist()
    placeholders = ",".join(["?" for _ in cell_ids])
    query = f"SELECT event, cell_id, {error_field} FROM cell_metrics WHERE cell_id IN ({placeholders})"

    with sqlite3.connect(error_db_path) as conn:
        # Validate column exists
        cols_df = pd.read_sql_query("PRAGMA table_info(cell_metrics)", conn)
        if error_field not in cols_df["name"].values:
            raise ValueError(
                f"Requested error_field '{error_field}' not found in cell_metrics columns: {list(cols_df['name'])}"
            )
        error_df = pd.read_sql_query(query, conn, params=cell_ids)

    error_df["error_value"] = error_df[error_field].fillna(0)

    # map for each event
    Path(output_plot_path).mkdir(exist_ok=True, parents=True)
    colormap_limits = (-3, 3)  # (error_df["error_value"].min(), error_df["error_value"].max())
    events = error_df["event"].unique()
    for event in events:
        sub = error_df[error_df["event"] == event]
        merged_df = mesh_df.merge(sub, on="cell_id", how="left").to_crs("epsg:4326")
        map_errors(
            merged_df[merged_df["event"] == event],
            Path(output_plot_path) / f"{suffix}_{event}.png",
            error_metric,
            event,
            colormap_limits,
        )
    return merged_df


def map_errors(
    merged_df: gpd.GeoDataFrame,
    output_plot_path: str | Path,
    error_metric: str,
    event: str,
    colormap_limits: tuple[float, float],
) -> None:
    """Create error map using matplotlib.

    Parameters:
    merged_df: GeoDataFrame containing merged mesh and error data
    Assumes geometry column contains polygon coordinates or shapely geometries
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    merged_df.plot(
        column="error_value",
        ax=ax,
        vmin=colormap_limits[0],
        vmax=colormap_limits[1],
        edgecolor="none",
        legend=True,
        legend_kwds={"label": error_metric},
    )

    # Set equal aspect ratio and adjust limits
    ax.set_aspect("equal")
    ax.autoscale_view()

    plt.title(f"{error_metric} Map - {event}", fontsize=16, fontweight="bold")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(Path(output_plot_path))
    plt.close(fig)


def plot_timeseries_metrics(
    db_path: str | Path,
    out_path: str | Path,
    metrics_field: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    overlay: bool = False,
) -> None:
    """Plot timeseries error metrics stored in the performance metrics database.

    Reads the 'timeseries_metrics' table created by `export_metric_summary` and plots
    each selected metric over timestep index.

    Parameters:
        db_path: Path to `performance_metrics.db` (SQLite database).
        out_path: Path to save the output plot (e.g., PNG).
        metrics_field: Optional iterable of column names to plot. If None, all numeric
                 columns in the table are plotted.
        metrics: Optional label of plots.
        overlay: If True, plot all metrics on a single set of axes with a legend.
                 If False (default), create one subplot per metric (stacked rows).

    Returns:
        None
    """
    db_path = Path(db_path)
    if not db_path.exists():  # pragma: no cover - guard clause
        raise FileNotFoundError(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        # Verify table exists
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        if "timeseries_metrics" not in tables["name"].tolist():
            raise ValueError(
                "Table 'timeseries_metrics' not found in database. Available tables: " f"{tables['name'].tolist()}"
            )
        ts_df = pd.read_sql_query("SELECT * FROM timeseries_metrics", conn)

    # Determine which columns to plot
    if metrics_field is None:
        # Select numeric columns only
        numeric_cols = ts_df.select_dtypes(include=["number"]).columns.tolist()
        plot_cols = numeric_cols
    else:
        plot_cols = list(metrics_field)
        missing = [c for c in plot_cols if c not in ts_df.columns]
        if missing:
            raise ValueError(f"Requested metrics_field not present: {missing}. Available: {ts_df.columns.tolist()}")

    if not plot_cols:
        raise ValueError("No metrics_field to plot after filtering.")

    # list of events
    events = ts_df["event"].unique().tolist()

    # plot limits
    y_limits = (np.floor(ts_df[plot_cols].min().min()), np.ceil(ts_df[plot_cols].max().max()))
    for event in events:
        event_mask = ts_df["event"] == event
        event_df = ts_df[event_mask]
        # Build figure / axes
        if overlay:
            fig, ax = plt.subplots(figsize=(6.5, 4))
            axs: list[Axes] = [ax]
            for i, col in enumerate(plot_cols):
                if metrics is not None:
                    ax.plot(event_df.index, event_df[col], label=metrics[i])
                else:
                    ax.plot(event_df.index, event_df[col], label=col)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Metric Value")
            ax.legend()
        else:
            fig, axs_arr = plt.subplots(nrows=len(plot_cols), figsize=(6.5, 2.2 * len(plot_cols)), sharex=True)
            axs = list(axs_arr.ravel()) if isinstance(axs_arr, np.ndarray) else [axs_arr]
            for ax, col in zip(axs, plot_cols, strict=False):
                ax.plot(event_df.index, event_df[col], c=COMMON_COLORS[0])
                ax.set_ylabel(col)
            axs[-1].set_xlabel("Timestep")

        fig.suptitle(f"Timeseries Error Metrics - {event}")
        ax.set_ylim(y_limits)
        apply_formatting(fig, axs)
        Path(out_path).mkdir(exist_ok=True, parents=True)
        fig.savefig(Path(out_path) / f"error_ts_{event}.png")
        plt.close(fig)


def summary_plots(
    db_path: str | Path,
    out_path: str | Path,
    metrics: dict[str, dict[str, str]],
) -> None:
    """Generates and saves boxplot summary plots for specified metrics from a SQLite database.

    This function reads a table named 'scalar_metrics' from the provided SQLite database,
    extracts the specified metric fields for each unique event, and creates boxplots
    summarizing the distribution of each metric across events. Each plot is saved as a PNG
    file in the specified output directory.

    Parameters:
        db_path: Path to `performance_metrics.db` (SQLite database).
        out_path: Path to save the output plot (e.g., PNG).
        metrics (dict[str, dict[str, str]]): Dictionary mapping metric fields to their labels.

    Notes:
        - Each plot is saved as 'summary_<column>.png' in the output directory.
        - The function assumes that the 'event' column exists in the metrics tables.

    Returns:
        None
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        # Verify table exists
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        if "cell_metrics" not in tables["name"].tolist():
            raise ValueError(
                "Table 'cell_metrics' not found in database. Available tables: " f"{tables['name'].tolist()}"
            )
        cell_df = pd.read_sql_query("SELECT * FROM cell_metrics", conn)

        # list of events
        events = cell_df["event"].unique().tolist()

        # Error Plots
        for table in metrics:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            for metrics_field, metrics_label in metrics[table].items():
                fig, ax = plt.subplots(figsize=(6.5, 4))
                if len(df) == len(events):
                    sorted_events = events.copy()
                    sorted_events.sort()
                    ax.scatter(sorted_events, df.sort_values(by="event")[metrics_field])
                    ax.grid()
                else:
                    ax.boxplot([df[df["event"] == event][metrics_field] for event in events], labels=events)
                plt.xticks(rotation=45)
                ax.set_ylabel(metrics_label)
                ax.set_title(f"{metrics_label} for Testing Dataset")
                fig.tight_layout()
                fig.savefig(Path(out_path) / f"summary_{table}_{metrics_field}.png")
                plt.close(fig)

        # Number of Time Steps
        timeseries_df = pd.read_sql_query("SELECT * FROM timeseries_metrics", conn)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.bar(sorted_events, [len(timeseries_df[timeseries_df["event"] == event]) for event in sorted_events])
        plt.xticks(rotation=45)
        ax.set_ylabel("Number of Time Steps")
        ax.set_title("Number of Time Steps for Testing Dataset")
        fig.tight_layout()
        fig.savefig(Path(out_path) / "summary_timeseries.png")
        plt.close(fig)


def plot_spatial_eof(
    plot_dir: str | Path,
    eof_vector: NDArray[Any],
    mode: int,
    wet_cell_ids: NDArray[Any],
    mesh_df: gpd.GeoDataFrame,
    cell_id_field: str = "cell_id",
    title: str = "Spatial EOF Pattern",
    cmap: str = "seismic",
    shared_vmax: float | None = None,
) -> None:
    """Plot a single spatial EOF pattern using cell polygons from mesh_df.

    Parameters:
    - plot_dir: Directory to save the plot
    - eof_vector: 1D numpy array of size (n_wet_cells,) for one EOF mode
    - wet_cell_ids: 1D numpy array of cell ids where EOF values go in the mesh_df
    - mesh_df: GeoDataFrame containing mesh information (e.g., cell geometry)
    - title: plot title
    - cmap: colormap to use
    - shared_vmax: if provided, uses same scale for all plots
    """
    # Map EOF_vector values to the corresponding cell polygons in mesh_df
    mesh_df = mesh_df.copy()
    mesh_df["EOF_value"] = 0  # Initialize with zeros
    mesh_df["EOF_value"] = mesh_df["EOF_value"].astype(float)
    mesh_df.set_index(cell_id_field, inplace=True)
    mesh_df.loc[wet_cell_ids, "EOF_value"] = eof_vector

    vmax = shared_vmax if shared_vmax is not None else np.max(np.abs(eof_vector))  # symmetric color scale

    # Plot using polygons
    fig, ax = plt.subplots(figsize=(10, 8))
    mesh_df.plot(
        column="EOF_value", cmap=cmap, vmin=-vmax, vmax=vmax, legend=True, ax=ax, legend_kwds={"label": "EOF Amplitude"}
    )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    plt.tight_layout()
    # plt.show()
    plt.savefig(Path(plot_dir) / f"eof_{mode}.png")
    plt.close(fig)


def plot_eof_maps(
    eofs: NDArray[Any],
    wet_cell_ids: NDArray[Any],
    mesh_df: gpd.GeoDataFrame,
    plot_dir: str | Path,
    n_modes: int = 3,
    cell_id_field: str = "cell_id",
    cmap: str = "seismic",
) -> None:
    """Plot the first n EOF modes using a consistent color scale.

    Parameters:
    - eofs: 2D array (n_modes, n_wet_cells)
    - wet_cell_ids: 1D array of wet cell ids
    - mesh_df: GeoDataFrame containing mesh information (e.g., cell geometry)
    - n_modes: how many EOF modes to plot
    - cmap: colormap
    """
    shared_vmax = np.max(np.abs(eofs[:n_modes, :]))
    for i in range(n_modes):
        plot_spatial_eof(
            plot_dir=Path(plot_dir),
            eof_vector=eofs[i, :],
            mode=i + 1,
            wet_cell_ids=wet_cell_ids,
            mesh_df=mesh_df,
            cell_id_field=cell_id_field,
            title=f"Spatial EOF Mode {i+1}",
            cmap=cmap,
            shared_vmax=shared_vmax,
        )


def map_detection_categories(
    mesh_df: gpd.GeoDataFrame,
    y_true: NDArray[Any],
    y_pred: NDArray[Any],
    index: NDArray[Any],
    columns: NDArray[Any],
    output_plot_path: str | Path,
    include_correct_negative: bool = False,
    wet_threshold_depth: float = 0.0,
) -> None:
    """Create a map showing Detected, Miss, and False Alarm categories per cell for the maximum depth (peak).

    A cell is classified using (optionally) the maximum value over time for true and predicted arrays:
      - Detected: y_true > 0 and y_pred > 0
      - Miss:     y_true > 0 and y_pred == 0
      - False Alarm: y_true == 0 and y_pred > 0
      - Correct Negative (optional): y_true == 0 and y_pred == 0

    All input values must be non-negative. Negative values raise a ValueError.

    Parameters:
        mesh_df: GeoDataFrame containing at least ['cell_id','geometry'] rows (one per cell) in same order or with
                 matching cell_id values corresponding to columns in y arrays.
        y_true: 2D array (timesteps, cells) OR 1D array (cells). Values must be >= 0.
        y_pred: 2D array (timesteps, cells) OR 1D array (cells). Values must be >= 0.
        index: NDArray[Any] representing the index (event).
        columns: NDArray[Any] representing the column names (cell IDs).
        output_plot_path: Path to save output PNG.
        include_correct_negative: Whether to display cells where both are zero as a separate category.
        title: Title for the plot.

    Returns:
        None
    """
    # convert the true and predicted data to dataframe
    df_true = pd.DataFrame(y_true, index=pd.MultiIndex.from_tuples(index), columns=columns)
    df_pred = pd.DataFrame(y_pred, index=pd.MultiIndex.from_tuples(index), columns=columns)

    # create the list of events
    # events = set(t[0] for t in index)
    events = {t[0] for t in index}
    events = set(events)
    for event in events:

        # Example: filter rows where first element of index tuple == 'some_value'
        mask = df_true.index.get_level_values(0) == event
        filtered_true = df_true[mask]
        filtered_pred = df_pred[mask]

        true_cell_vals: pd.DataFrame = filtered_true.max(axis=0) if y_true.ndim == 2 else y_true
        pred_cell_vals: pd.DataFrame = filtered_pred.max(axis=0) if y_pred.ndim == 2 else y_pred

        true_cell_vals = true_cell_vals.sort_index()
        pred_cell_vals = pred_cell_vals.sort_index()
        mesh_df = mesh_df.sort_values(by="cell_id")
        # cells < wet_threshold_depth are considered "inactive"

        if true_cell_vals.shape[0] != pred_cell_vals.shape[0]:
            raise ValueError("y_true and y_pred must have the same number of cells.")

        if (true_cell_vals < 0).any() or (pred_cell_vals < 0).any():
            raise ValueError("y_true and y_pred must be non-negative.")

        # Apply wet threshold
        true_cell_vals[true_cell_vals < wet_threshold_depth] = 0
        pred_cell_vals[pred_cell_vals < wet_threshold_depth] = 0

        # Attempt to align by cell_id if present and lengths differ
        if true_cell_vals.shape[0] != len(mesh_df):
            # If mesh_df has a 'cell_id' and values correspond to ordering elsewhere, we assume ordering mismatch.
            # For now enforce same length.
            raise ValueError(
                "Number of cells in mesh_df does not match length of y arrays: "
                f"{len(mesh_df)} vs {true_cell_vals.shape[0]}"
            )

        detected_mask = (true_cell_vals > 0) & (pred_cell_vals > 0)
        miss_mask = (true_cell_vals > 0) & (pred_cell_vals == 0)
        false_alarm_mask = (true_cell_vals == 0) & (pred_cell_vals > 0)
        correct_neg_mask = (true_cell_vals == 0) & (pred_cell_vals == 0)

        categories = np.full(true_cell_vals.shape[0], "", dtype=object)
        categories[detected_mask] = "Detected"
        categories[miss_mask] = "Miss"
        categories[false_alarm_mask] = "False Alarm"
        if include_correct_negative:
            categories[correct_neg_mask] = "Correct Negative"

        mesh_df = mesh_df.copy()
        mesh_df["detection_category"] = categories

        # Define colors
        color_map = {
            "Detected": "#009E73",  # green
            "Miss": "#D55E00",  # orange/red
            "False Alarm": "#E69F00",  # gold
            "Correct Negative": "#999999",  # gray
            "": "#FFFFFF",  # empty (should not normally appear)
        }
        # Colors per cell used implicitly when building facecolors list

        # Build plot
        fig, ax = plt.subplots(figsize=(12, 8))
        patches = []
        facecolors = []
        for _, row in mesh_df.iterrows():
            geom = row["geometry"]
            if geom is None:
                continue
            if geom.geom_type == "Polygon":
                polys = [geom]
            elif geom.geom_type == "MultiPolygon":
                polys = list(geom.geoms)
            else:
                continue
            for poly in polys:
                exterior = np.asarray(poly.exterior.coords)
                patches.append(MplPolygon(exterior, closed=True))
                facecolors.append(color_map[row["detection_category"]])

        if not patches:
            raise ValueError("No polygon patches could be constructed from 'mesh_df'.")

        collection = PatchCollection(patches, facecolor=facecolors, edgecolor="#333333", linewidths=0.3, alpha=0.9)
        ax.add_collection(collection)
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.set_title(f"Detection Outcomes - {event}")

        # Legend (only for categories present)
        legend_handles = []
        for label in ["Detected", "Miss", "False Alarm", "Correct Negative"]:
            if label == "Correct Negative" and not include_correct_negative:
                continue
            if label in mesh_df["detection_category"].values:
                legend_handles.append(Patch(facecolor=color_map[label], edgecolor="#333333", label=label))
        if legend_handles:
            ax.legend(handles=legend_handles, frameon=False, loc="upper right")

        apply_formatting(fig, ax)
        fig.savefig(Path(output_plot_path) / f"detection_{event}.png")
        plt.close(fig)
