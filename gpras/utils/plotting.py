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


def ec_pairplot(x: NDArray[Any], y: NDArray[Any], modes_to_plot: int, out_path: str | Path) -> None:
    """Generate a Seaborn pairplot comparing low-fidelity and high-fidelity EOF mode values.

    This plot is useful for assesing how closely LF modes are to HF modes (diagonal plots).
    It is also useful for exploring whether distinct trends exist within each row that the GPR could leverage for
    prediction.

    Args:
        x (NDArray[Any]): Low-fidelity EOF coefficients with shape (n_samples, n_modes).
        y (NDArray[Any]): High-fidelity EOF coefficients with shape (n_samples, n_modes).
        modes_to_plot (int): Number of EOF modes to include in the plot.
        out_path (str): Location to save the plot.

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
    apply_formatting(g.figure, g.axes.flatten())
    g.savefig(out_path)


def ec_timeseries(x: NDArray[Any], y: NDArray[Any], modes_to_plot: int, ind: pd.Index, out_dir: str | Path) -> None:
    """Plot EOF time series for low- and high-fidelity models by event plan.

    Args:
        x (NDArray[Any]): Low-fidelity EOF coefficients with shape (n_samples, n_modes).
        y (NDArray[Any]): High-fidelity EOF coefficients with shape (n_samples, n_modes).
        modes_to_plot (int): Number of EOF modes to plot.
        out_dir (str): Dirctory to save the plots.
        ind (pd.Index): Index object (e.g., MultiIndex) used to group samples by plan.

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
    fig.savefig(out_path)
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
    fig.savefig(out_path)
    plt.close(fig)


def map_mesh_errors(
    mesh_df: gpd.GeoDataFrame,
    error_db_path: str | Path,
    output_plot_path: str | Path,
    error_field: str = "rmse_cell_toi",
    error_metric: str = "RMSE"
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
    query = f"SELECT cell_id, {error_field} FROM cell_metrics WHERE cell_id IN ({placeholders})"

    with sqlite3.connect(error_db_path) as conn:
        # Validate column exists
        cols_df = pd.read_sql_query("PRAGMA table_info(cell_metrics)", conn)
        if error_field not in cols_df["name"].values:
            raise ValueError(
                f"Requested error_field '{error_field}' not found in cell_metrics columns: {list(cols_df['name'])}"
            )
        error_df = pd.read_sql_query(query, conn, params=cell_ids)

    merged_df = mesh_df.merge(error_df, on="cell_id", how="left")
    merged_df["error_value"] = merged_df[error_field].fillna(0)

    map_errors(merged_df, output_plot_path, error_metric)
    return merged_df

def map_errors(merged_df: gpd.GeoDataFrame, output_plot_path: str | Path, error_metric: str) -> None:
    """Create error map using matplotlib.

    Parameters:
    merged_df: GeoDataFrame containing merged mesh and error data
    Assumes geometry column contains polygon coordinates or shapely geometries
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    patches = []
    colors = []
    
    for _, row in merged_df.iterrows():
        geom = row["geometry"]
        if geom is None:
            continue
        # Handle Polygon and MultiPolygon
        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            polys = list(geom.geoms)
        else:
            continue  # skip non-polygonal geometries
        for poly in polys:
            exterior = np.asarray(poly.exterior.coords)
            patches.append(MplPolygon(exterior, closed=True))
            colors.append(row["error_value"])
    
    # Create patch collection
    if not patches:
        raise ValueError("No polygon patches could be constructed from 'merged_df'.")
    p = PatchCollection(patches, alpha=0.8)
    p.set_array(np.array(colors, dtype=float))
    
    # Add patches to plot
    ax.add_collection(p)
    
    # Set colorbar
    cbar = plt.colorbar(p, ax=ax)
    cbar.set_label(error_metric, rotation=270, labelpad=15, fontweight='bold')
    
    # Set equal aspect ratio and adjust limits
    ax.set_aspect('equal')
    ax.autoscale_view()

    plt.title(f'{error_metric} Map', fontsize=16, fontweight='bold')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.tight_layout()
    # plt.show()
    plt.savefig(output_plot_path)
    # plt.close(fig)


def plot_timeseries_metrics(
    db_path: str | Path,
    out_path: str | Path,
    metrics_field: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    overlay: bool = False,
) -> pd.DataFrame:
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
        The DataFrame of timeseries metrics that were plotted.
    """
    db_path = Path(db_path)
    if not db_path.exists():  # pragma: no cover - guard clause
        raise FileNotFoundError(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        # Verify table exists
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        if "timeseries_metrics" not in tables["name"].tolist():
            raise ValueError(
                "Table 'timeseries_metrics' not found in database. Available tables: "
                f"{tables['name'].tolist()}"
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

    # Build figure / axes
    if overlay:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        axs: list[Axes] = [ax]
        for i, col in enumerate(plot_cols):
            if metrics is not None:
                ax.plot(ts_df.index, ts_df[col], label=metrics[i])
            else:
                ax.plot(ts_df.index, ts_df[col], label=col)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Metric Value")
        ax.legend()
    else:
        fig, axs_arr = plt.subplots(nrows=len(plot_cols), figsize=(6.5, 2.2 * len(plot_cols)), sharex=True)
        # Ensure iterable of axes
        axs = list(axs_arr.ravel()) if isinstance(axs_arr, np.ndarray) else [axs_arr]
        for ax, col in zip(axs, plot_cols, strict=False):
            ax.plot(ts_df.index, ts_df[col], c=COMMON_COLORS[0])
            ax.set_ylabel(col)
        axs[-1].set_xlabel("Timestep")

    fig.suptitle("Timeseries Error Metrics")
    apply_formatting(fig, axs)
    fig.savefig(out_path)
    plt.close(fig)
    return ts_df[plot_cols]


def map_detection_categories(
    mesh_df: gpd.GeoDataFrame,
    y_true: NDArray[Any],
    y_pred: NDArray[Any],
    output_plot_path: str | Path,
    include_correct_negative: bool = False,
    title: str = "Detection Outcomes",
    wet_threshold_depth: float = 0.0
) -> gpd.GeoDataFrame:
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
        output_plot_path: Path to save output PNG.
        include_correct_negative: Whether to display cells where both are zero as a separate category.
        title: Title for the plot.

    Returns:
        GeoDataFrame with an added 'detection_category' column.
    """
    true_cell_vals = y_true.max(axis=0) if y_true.ndim == 2 else y_true
    pred_cell_vals = y_pred.max(axis=0) if y_pred.ndim == 2 else y_pred

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
        "Detected": "#009E73",       # green
        "Miss": "#D55E00",           # orange/red
        "False Alarm": "#E69F00",    # gold
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
    ax.set_title(title)

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
    fig.savefig(output_plot_path)
    plt.close(fig)
    return mesh_df
