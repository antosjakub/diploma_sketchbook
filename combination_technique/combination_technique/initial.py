"""Common initial densities for examples and tests."""

from __future__ import annotations

from math import gamma, pi

import numpy as np


def gaussian_density(coords: np.ndarray, covariance: np.ndarray | None = None) -> np.ndarray:
    d, n = coords.shape
    if covariance is None:
        norm = (2.0 * pi) ** (-0.5 * d)
        exponent = -0.5 * np.sum(coords * coords, axis=0)
        return norm * np.exp(exponent)

    covariance = np.asarray(covariance, dtype=float)
    inv = np.linalg.inv(covariance)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("covariance must be positive definite")
    exponent = -0.5 * np.sum(coords * (inv @ coords), axis=0)
    norm = np.exp(-0.5 * (d * np.log(2.0 * pi) + logdet))
    return norm * np.exp(exponent)


def cauchy_density(coords: np.ndarray) -> np.ndarray:
    d = coords.shape[0]
    radius2 = np.sum(coords * coords, axis=0)
    norm = gamma((d + 1.0) / 2.0) / (pi ** ((d + 1.0) / 2.0))
    return norm / (1.0 + radius2) ** ((d + 1.0) / 2.0)


def product_laplace_density(coords: np.ndarray) -> np.ndarray:
    d = coords.shape[0]
    return 2.0 ** (-d) * np.exp(-np.sum(np.abs(coords), axis=0))

