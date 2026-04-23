"""Basic tensor-grid diagnostics."""

from __future__ import annotations

import numpy as np

from .grid import TensorGrid


def cell_volume(grid: TensorGrid) -> float:
    return float(np.prod(grid.spacing))


def mass(grid: TensorGrid, values: np.ndarray) -> float:
    return float(np.sum(np.asarray(values).reshape(-1)) * cell_volume(grid))


def mean(grid: TensorGrid, values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    coords = grid.flat_coordinates()
    m = mass(grid, flat)
    if m == 0.0:
        return np.full(grid.dimension, np.nan)
    return (coords @ flat) * cell_volume(grid) / m


def covariance(grid: TensorGrid, values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    coords = grid.flat_coordinates()
    mu = mean(grid, flat)
    centered = coords - mu[:, None]
    m = mass(grid, flat)
    if m == 0.0:
        return np.full((grid.dimension, grid.dimension), np.nan)
    weighted = centered * flat[None, :]
    return (weighted @ centered.T) * cell_volume(grid) / m

