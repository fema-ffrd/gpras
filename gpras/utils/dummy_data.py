"""Tools for generating fake RAS model forcing data."""

import random
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

FloatType = TypeVar("FloatType", bound=np.floating[Any])


def log_pdf(x: NDArray[FloatType], mu: float, sigma: float) -> NDArray[FloatType]:
    """Get lognormal pdf values of an array."""
    mask = x != 0
    vals = np.zeros_like(x)
    vals[mask] = 1 / (x[mask] * sigma * np.sqrt(2 * np.pi)) * np.exp(-((np.log(x[mask]) - mu) ** 2) / (2 * sigma**2))
    return vals


def gen_hydrograph(
    peak: float, peak_time: float, flashiness: float, timesteps: NDArray[FloatType]
) -> NDArray[FloatType]:
    """Create a hydrograph based on the lognormal pdf."""
    sigma = flashiness
    mu = (sigma**2) + np.log(peak_time)
    vals: NDArray[FloatType] = log_pdf(timesteps, mu, sigma)
    vals = vals * (peak / vals.max())
    return vals


def gen_random_hydrograph(timesteps: NDArray[FloatType], scale: float) -> NDArray[FloatType]:
    """Generate a complex, multi-peaked hydrograph of an approximate scale."""
    peak_choices = [1, 2, 3, 4]
    peak_range = (0.8, 1.2)
    flash_range = (0.1, 0.75)
    t_range = (0.1, 0.5)

    num_peaks = random.choice(peak_choices)
    ts_vals = np.zeros_like(timesteps)
    for _ in range(num_peaks):
        peak = (random.uniform(*peak_range) / num_peaks) * scale
        flashiness = random.uniform(*flash_range)
        peak_time = random.uniform(*t_range)
        ts_vals += gen_hydrograph(peak, peak_time, flashiness, timesteps)
    return ts_vals


def weather_maker(t: int, m: int, n: int, num_circles: int = 15) -> NDArray[np.float32]:
    """Create precipitation data with several circles moving around the map."""
    np.random.seed(42)
    radii = (np.random.rand(num_circles) * 25) + 10
    intensity = np.random.rand(num_circles)

    out = np.zeros((t, n, m), dtype=np.float32)

    # Initialize random positions and velocities
    positions = np.random.rand(num_circles, 2) * [m, n]  # (x, y)
    velocities = (np.random.rand(num_circles, 2) - 0.5) * 10  # random dx, dy

    yy, xx = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")

    for frame in range(t):
        frame_array = np.zeros((n, m), dtype=np.float32)

        for ind, (int, r) in enumerate(zip(intensity, radii, strict=False)):
            cx, cy = positions[ind]
            for i in range(3):
                tmp_r = r * (0.5**i)
                tmp_int = int * (2**i)
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= tmp_r**2
                frame_array[mask] += tmp_int

            # Update position
            positions[ind] += velocities[ind]

            # Bounce off edges
            for dim in range(2):
                if positions[ind][dim] < r or positions[ind][dim] > [m, n][dim] - r:
                    velocities[ind][dim] *= -1
                    positions[ind][dim] = np.clip(positions[ind][dim], r, [m, n][dim] - r)

        out[frame] = frame_array

    return out.reshape(t, -1)  # unravel to match hdf
