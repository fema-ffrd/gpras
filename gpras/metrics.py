"""Metrics to assess the performance of a GPR model."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def export_metric_summary(
    x_all: pd.DataFrame,
    y_all: pd.DataFrame,
    out_path: str | Path,
    depth_threshold: float = 0.5,
    t_tol: int = 0,
    v_tol: float = 0,
) -> None:
    """Export all metrics to a sqlite database."""
    # Initialize lists for per-event dataframes.
    all_scalar = []
    all_timeseries = []
    all_cells = []

    for event in x_all.index.unique(level=0):
        # Subset to event
        x = x_all.loc[event].values
        y = y_all.loc[event].values
        tsteps = x_all.loc[event].index.values

        # Cache maximum timestep for efficiency
        x_mts = np.argmax(x, axis=0)
        y_mts = np.argmax(y, axis=0)

        # Scalar metrics
        scalar_dict = {
            "event": event,
            "rmse_aoi_toi": [rmse_aoi_toi(x, y)],
            "rmse_aoi_mts": [rmse_aoi_mts(x, y, x_mts, y_mts)],
            "nse_aoi_mts": [nse_aoi_mts(x, y, x_mts, y_mts)],
            "err_aoi_toi": [err_aoi_toi(x, y)],
            "err_aoi_mts": [err_aoi_mts(x, y, x_mts, y_mts)],
            "fi_aoi_toi": [fi_aoi_toi(x, y, t_tol, v_tol)],
            "pod_mts": [pod_mts(x, y, depth_threshold, x_mts, y_mts)],
            "rfa_mts": [rfa_mts(x, y, depth_threshold, x_mts, y_mts)],
            "csi_mts": [csi_mts(x, y, depth_threshold, x_mts, y_mts)],
            "f2_mts": [f2_mts(x, y, x_mts, y_mts)],
            "f3_mts": [f3_mts(x, y, x_mts, y_mts)],
        }
        all_scalar.append(pd.DataFrame.from_dict(scalar_dict))

        # Timeseries metrics
        timeseries_dict = {
            "event": np.repeat(event, x.shape[0]),
            "timestep": tsteps,
            "rmse_aoi_ts": rmse_aoi_ts(x, y),
            "err_aoi_ts": err_aoi_ts(x, y),
        }
        all_timeseries.append(pd.DataFrame.from_dict(timeseries_dict))

        # Cell metrics
        cell_dict = {
            "event": np.repeat(event, x.shape[1]),
            "cell_id": x_all.columns,
            "rmse_cell_toi": rmse_cell_toi(x, y),
            "err_cell_mts": err_cell_mts(x, y, x_mts, y_mts),
            "err_cell_toi": err_cell_toi(x, y),
        }
        all_cells.append(pd.DataFrame.from_dict(cell_dict))

    # Export to sqlite
    with sqlite3.connect(out_path) as con:
        pd.concat(all_scalar).to_sql("scalar_metrics", con, index=False, if_exists="replace")
        pd.concat(all_timeseries).to_sql("timeseries_metrics", con, index=False, if_exists="replace")
        pd.concat(all_cells).to_sql("cell_metrics", con, index=False, if_exists="replace")


def rmse_aoi_toi(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Calculate the root-mean-squared-error across all cells and timesteps."""
    return float((((x - y) ** 2).mean()) ** 0.5)


def rmse_aoi_ts(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate the root-mean-squared-error across all cells for each timestep."""
    return np.asarray((((x - y) ** 2).mean(axis=1)) ** 0.5, dtype=np.float64)


def rmse_cell_toi(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate the root-mean-squared-error across all timesteps for each cell."""
    return np.asarray((((x - y) ** 2).mean(axis=0)) ** 0.5, dtype=np.float64)


def rmse_aoi_mts(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    x_mts: NDArray[np.float64] | None = None,
    y_mts: NDArray[np.float64] | None = None,
) -> float:
    """Calculate the root-mean-squared-error across all cells at the maximum timestep of each cell."""
    if x_mts is None:
        x_mts = np.argmax(x, axis=0)
    if y_mts is None:
        y_mts = np.argmax(y, axis=0)
    return float((((x[x_mts, np.arange(x.shape[1])] - y[y_mts, np.arange(y.shape[1])]) ** 2).mean()) ** 0.5)


def err_cell_mts(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    x_mts: NDArray[np.float64] | None = None,
    y_mts: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Calculate the difference at each cell at the maximum timestep of each cell."""
    if x_mts is None:
        x_mts = np.argmax(x, axis=0)
    if y_mts is None:
        y_mts = np.argmax(y, axis=0)
    return np.asarray(x[x_mts, np.arange(x.shape[1])] - y[y_mts, np.arange(y.shape[1])], dtype=np.float64)


def nse_aoi_mts(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    x_mts: NDArray[np.float64] | None = None,
    y_mts: NDArray[np.float64] | None = None,
) -> float:
    """Calculate the Nash-Sutcliffe Efficiency between models at their peaks."""
    if x_mts is None:
        x_mts = np.argmax(x, axis=0)
    if y_mts is None:
        y_mts = np.argmax(y, axis=0)
    num = np.sum((x[x_mts, np.arange(x.shape[1])] - y[y_mts, np.arange(y.shape[1])]) ** 2)
    denom = np.sum((x[x_mts, np.arange(x.shape[1])] - x[x_mts, np.arange(x.shape[1])].mean()) ** 2)
    return float(1 - (num / denom))


def err_aoi_toi(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Calculate the mean difference across all timesteps and cells."""
    return float((x - y).mean())


def err_aoi_mts(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    x_mts: NDArray[np.float64] | None = None,
    y_mts: NDArray[np.float64] | None = None,
) -> float:
    """Calculate the mean difference across cell peaks."""
    if x_mts is None:
        x_mts = np.argmax(x, axis=0)
    if y_mts is None:
        y_mts = np.argmax(y, axis=0)

    return float((x[x_mts, np.arange(x.shape[1])] - y[y_mts, np.arange(y.shape[1])]).mean())


def err_aoi_ts(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate the mean difference across all cells at each timestep."""
    return np.asarray((x - y).mean(axis=1), dtype=np.float64)


def err_cell_toi(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate the mean difference across all timesteps for each cell."""
    return np.asarray((x - y).mean(axis=0), dtype=np.float64)


def fi_aoi_toi(x: NDArray[np.float64], y: NDArray[np.float64], t_tol: int, v_tol: float) -> float:
    """Calculate the fidelity index for model predictions."""
    matching = np.abs(y - x) <= v_tol
    for i in range(1, t_tol + 1):
        tmp = np.abs(y[:-i, :] - x[i:, :]) <= v_tol
        matching[:-i] = tmp | matching[:-i]
    for i in range(1, t_tol + 1):
        tmp = np.abs(x[:-i, :] - y[i:, :]) <= v_tol
        matching[:-i] = tmp | matching[:-i]
    return float(np.sum(matching) / (matching.shape[0] * matching.shape[1]))


def pod_mts(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    depth_threshold: float = 0,
    x_mts: NDArray[np.float64] | None = None,
    y_mts: NDArray[np.float64] | None = None,
) -> float:
    """Calculate the probability of detection at the maximum timestep for each cell."""
    if x_mts is None:
        x_mts = np.argmax(x, axis=0)
    if y_mts is None:
        y_mts = np.argmax(y, axis=0)
    a_detected = np.sum(
        (x[x_mts, np.arange(x.shape[1])] >= depth_threshold) * (y[y_mts, np.arange(y.shape[1])] >= depth_threshold)
    )
    a_missed = np.sum(
        (x[x_mts, np.arange(x.shape[1])] >= depth_threshold) * (y[y_mts, np.arange(y.shape[1])] < depth_threshold)
    )
    return float(a_detected / (a_detected + a_missed))


def rfa_mts(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    depth_threshold: float = 0,
    x_mts: NDArray[np.float64] | None = None,
    y_mts: NDArray[np.float64] | None = None,
) -> float:
    """Calculate the rate of false alarm at the maximum timestep for each cell."""
    if x_mts is None:
        x_mts = np.argmax(x, axis=0)
    if y_mts is None:
        y_mts = np.argmax(y, axis=0)
    a_detected = np.sum(
        (x[x_mts, np.arange(x.shape[1])] >= depth_threshold) * (y[y_mts, np.arange(y.shape[1])] >= depth_threshold)
    )
    a_false_alarm = np.sum(
        (x[x_mts, np.arange(x.shape[1])] < depth_threshold) * (y[y_mts, np.arange(y.shape[1])] >= depth_threshold)
    )
    return float(a_false_alarm / (a_detected + a_false_alarm))


def csi_mts(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    depth_threshold: float = 0,
    x_mts: NDArray[np.float64] | None = None,
    y_mts: NDArray[np.float64] | None = None,
) -> float:
    """Calculate the rate of critical success index at the maximum timestep for each cell."""
    if x_mts is None:
        x_mts = np.argmax(x, axis=0)
    if y_mts is None:
        y_mts = np.argmax(y, axis=0)
    pod = pod_mts(x, y, depth_threshold, x_mts, y_mts)
    rfa = rfa_mts(x, y, depth_threshold, x_mts, y_mts)
    return float(1 / ((1 / pod) + (1 / (1 - rfa)) - 1))


def f2_mts(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    depth_threshold: float = 0,
    x_mts: NDArray[np.float64] | None = None,
    y_mts: NDArray[np.float64] | None = None,
) -> float:
    """Calculate the F2 score at the maximum timestep for each cell."""
    if x_mts is None:
        x_mts = np.argmax(x, axis=0)
    if y_mts is None:
        y_mts = np.argmax(y, axis=0)
    a = np.sum(
        (x[x_mts, np.arange(x.shape[1])] >= depth_threshold) * (y[y_mts, np.arange(y.shape[1])] >= depth_threshold)
    )
    b = np.sum(
        (x[x_mts, np.arange(x.shape[1])] < depth_threshold) * (y[y_mts, np.arange(y.shape[1])] >= depth_threshold)
    )
    c = np.sum(
        (x[x_mts, np.arange(x.shape[1])] >= depth_threshold) * (y[y_mts, np.arange(y.shape[1])] < depth_threshold)
    )
    return float((a - c) / (a + b + c))


def f3_mts(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    depth_threshold: float = 0,
    x_mts: NDArray[np.float64] | None = None,
    y_mts: NDArray[np.float64] | None = None,
) -> float:
    """Calculate the F3 score at the maximum timestep for each cell."""
    if x_mts is None:
        x_mts = np.argmax(x, axis=0)
    if y_mts is None:
        y_mts = np.argmax(y, axis=0)
    a = np.sum(
        (x[x_mts, np.arange(x.shape[1])] >= depth_threshold) * (y[y_mts, np.arange(y.shape[1])] >= depth_threshold)
    )
    b = np.sum(
        (x[x_mts, np.arange(x.shape[1])] < depth_threshold) * (y[y_mts, np.arange(y.shape[1])] >= depth_threshold)
    )
    c = np.sum(
        (x[x_mts, np.arange(x.shape[1])] >= depth_threshold) * (y[y_mts, np.arange(y.shape[1])] < depth_threshold)
    )
    return float((a - b) / (a + b + c))
