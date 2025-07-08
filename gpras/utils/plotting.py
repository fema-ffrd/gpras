"""Utilities for generating diagnostic and QC plots."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy.typing import NDArray


def ec_pairplot(x: NDArray[Any], y: NDArray[Any], modes_to_plot: int, out_path: str) -> None:
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
    g = sns.pairplot(df, x_vars=x_cols, y_vars=y_cols, plot_kws={"marker": "+", "linewidth": 1, "size": 1})
    for i, (x_col, y_col) in enumerate(zip(x_cols, y_cols, strict=False)):
        ax = g.axes[i, i]
        x_min, x_max = df[x_col].min(), df[x_col].max()
        y_min, y_max = df[y_col].min(), df[y_col].max()
        min_val = min(x_min, y_min)
        max_val = max(x_max, y_max)
        ax.plot([min_val, max_val], [min_val, max_val], color="k", linestyle="--", linewidth=1, alpha=0.8)
    g.savefig(out_path)


def ec_timeseries(x: NDArray[Any], y: NDArray[Any], modes_to_plot: int, out_dir: str, ind: pd.Index) -> None:
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
            ax.plot(y[cum_index : cum_index + count, i], label="HF model")
            ax.plot(x[cum_index : cum_index + count, i], label="LF model")
            ax.set_ylabel(f"EOF_{i}")
            ax.set_yticks([], labels=[])
        cum_index += count
        axs[0].legend()
        axs[-1].set_xlabel("Timestep")
        fig.suptitle(f"Plan {event_label}")
        fig.tight_layout()
        fig.savefig(Path(out_dir) / f"Plan_{event_label}.png")
        plt.close(fig)


def performance_scatterplot(lf: NDArray[Any], hf: NDArray[Any], lf_upskill: NDArray[Any], out_path: str) -> None:
    """Plot scatterplots comparing low-fidelity vs high-fidelity and upskilled vs high-fidelity models depth estimates.

    Args:
        lf (NDArray[Any]): Low-fidelity model output (e.g., water surface elevations).
        hf (NDArray[Any]): High-fidelity model output.
        lf_upskill (NDArray[Any]): Output of the upskilled low-fidelity model.
        out_path (str): Location to save the plot.

    Returns:
        None
    """
    lf, hf, lf_upskill = lf.flatten(), hf.flatten(), lf_upskill.flatten()

    fig, axs = plt.subplots(ncols=2, figsize=(6.5, 4), sharey=True)

    axs[0].scatter(lf, hf, s=1, c="r", alpha=0.8)
    ll, ur = min([lf.min(), hf.min()]), max([lf.max(), hf.max()])
    axs[0].plot((ll, ur), (ll, ur), ls="dashed", c="k")
    rmse = np.mean((lf - hf) ** 2) ** 0.5
    axs[0].text(0.95, 0.05, f"rmse: {round(rmse, 2)}", ha="right", va="bottom", transform=axs[0].transAxes)
    axs[0].set_ylabel("High-fidelity Model WSE (ft)")
    axs[0].set_xlabel("Low-fidelity Model WSE (ft)")

    axs[1].scatter(lf_upskill, hf, s=1, c="r", alpha=0.8)
    ll, ur = min([lf_upskill.min(), hf.min()]), max([lf_upskill.max(), hf.max()])
    axs[1].plot((ll, ur), (ll, ur), ls="dashed", c="k")
    rmse = np.mean((lf_upskill - hf) ** 2) ** 0.5
    axs[1].text(0.95, 0.05, f"rmse: {round(rmse, 2)}", ha="right", va="bottom", transform=axs[1].transAxes)
    axs[1].set_xlabel("Upskilled Model WSE (ft)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def performance_cdf(lf: NDArray[Any], hf: NDArray[Any], lf_upskill: NDArray[Any], out_path: str) -> None:
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
    ax.plot(lf_residual, pcts, label="Low-Fidelity Model")
    ax.plot(upskill_residual, pcts, label="Upskilled Model")
    ax.set_ylabel("Percent of Cells Less Than")
    ax.set_xlabel("Absolute Error (ft)")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)
