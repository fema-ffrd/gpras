"""Utilities for generating diagnostic and QC plots."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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


def ec_timeseries_alt(
    x: NDArray[Any], y: NDArray[Any], modes_to_plot: int, ind: pd.Index, x_labels: list[str], out_dir: str | Path
) -> None:
    """Plot EOF time series for low- and high-fidelity models by event plan (alt: all LF ECs on every plot)."""
    events = np.unique(ind.get_level_values(0), return_counts=True)
    cum_index = 0
    for event_label, count in zip(*events, strict=False):
        fig, axs = plt.subplots(nrows=modes_to_plot, figsize=(6.5, 2 * modes_to_plot), sharex=True)
        for i, ax in enumerate(axs):
            ax.plot(y[cum_index : cum_index + count, i], label="HF model", c="k", lw=2)
            for j in range(x.shape[1]):
                ax.plot(x[cum_index : cum_index + count, j], label=x_labels[j], alpha=0.6, lw=1)
            ax.set_ylabel(f"EOF_{i}")
            ax.set_yticks([], labels=[])
        cum_index += count
        axs[0].legend()
        axs[-1].set_xlabel("Timestep")
        fig.suptitle(f"Plan {event_label}")
        apply_formatting(fig, axs)
        fig.savefig(Path(out_dir) / f"Plan_{event_label}.png")
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
