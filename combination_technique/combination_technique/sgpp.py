"""SG++ sparse-grid integration helpers."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _to_data_vector(values: np.ndarray):
    pysgpp = import_pysgpp()
    flat = np.asarray(values, dtype=float).reshape(-1)
    vector = pysgpp.DataVector(flat.size)
    for i, value in enumerate(flat):
        vector[i] = float(value)
    return vector


def _from_data_vector(vector) -> np.ndarray:
    return np.array([float(vector[i]) for i in range(vector.getSize())], dtype=float)


def _to_data_matrix(points: np.ndarray):
    pysgpp = import_pysgpp()
    pts = np.asarray(points, dtype=float)
    matrix = pysgpp.DataMatrix(pts.shape[0], pts.shape[1])
    for i in range(pts.shape[0]):
        for j in range(pts.shape[1]):
            matrix.set(i, j, float(pts[i, j]))
    return matrix


def _candidate_roots() -> list[Path]:
    env_vars = [
        os.environ.get("SGPP_BUILD_LIB"),
        os.environ.get("SGPP_PYTHON_ROOT"),
    ]
    roots = [Path(value).expanduser().resolve() for value in env_vars if value]

    here = Path(__file__).resolve()
    roots.extend(
        [
            here.parents[2] / "SG" / "external" / "SGpp" / "build" / "lib",
            here.parents[2] / "SG" / "external" / "SGpp" / "lib",
        ]
    )
    return roots


def import_pysgpp():
    """Import the local SG++ Python bindings.

    The local environment currently exposes a broken ``PYTHONPATH`` entry that
    points at ``.../pysgpp`` instead of its parent directory. This loader fixes
    that before importing.
    """

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    module = sys.modules.get("pysgpp")
    if module is not None and hasattr(module, "Grid"):
        return module

    bad_suffixes = (
        os.path.join("SGpp", "lib", "pysgpp"),
        os.path.join("SGpp", "build", "lib", "pysgpp"),
    )
    for entry in list(sys.path):
        if entry.endswith(bad_suffixes):
            sys.path.remove(entry)

    for root in _candidate_roots():
        package_init = root / "pysgpp" / "__init__.py"
        compiled = root / "pysgpp" / "_pysgpp_swig.so"
        if not package_init.exists() or not compiled.exists():
            continue
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        sys.modules.pop("pysgpp", None)
        module = importlib.import_module("pysgpp")
        if hasattr(module, "Grid"):
            return module

    raise ImportError(
        "Unable to import pysgpp. Set SGPP_BUILD_LIB or SGPP_PYTHON_ROOT to "
        "the directory that contains the pysgpp package."
    )


def sgpp_available() -> bool:
    try:
        import_pysgpp()
    except ImportError:
        return False
    return True


def _normalize_bounds(
    bounds: tuple[tuple[float, float], ...] | None,
    *,
    dimension: int,
    domain_radius: float | tuple[float, ...] | None,
) -> tuple[tuple[float, float], ...]:
    if bounds is not None:
        if len(bounds) != dimension:
            raise ValueError("bounds must match dimension")
        return tuple((float(left), float(right)) for left, right in bounds)

    if domain_radius is None:
        domain_radius = 1.0

    if isinstance(domain_radius, tuple):
        if len(domain_radius) != dimension:
            raise ValueError("domain_radius tuple must match dimension")
        return tuple((-float(radius), float(radius)) for radius in domain_radius)

    radius = float(domain_radius)
    return tuple((-radius, radius) for _ in range(dimension))


def _coerce_points(points: np.ndarray, dimension: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a two-dimensional array")
    if pts.shape[1] != dimension and pts.shape[0] == dimension:
        pts = pts.T
    if pts.shape[1] != dimension:
        raise ValueError("points must have dimension columns")
    return pts


@dataclass(frozen=True)
class SGppInterpolant:
    """Hierarchical sparse-grid interpolant backed by SG++."""

    grid: object
    alpha: object
    bounds: tuple[tuple[float, float], ...]

    @property
    def dimension(self) -> int:
        return len(self.bounds)

    @property
    def size(self) -> int:
        return int(self.grid.getStorage().getSize())

    def _unit_to_physical(self, points: np.ndarray) -> np.ndarray:
        mapped = np.empty_like(points, dtype=float)
        for axis, (left, right) in enumerate(self.bounds):
            mapped[:, axis] = left + (right - left) * points[:, axis]
        return mapped

    def _physical_to_unit(self, points: np.ndarray) -> np.ndarray:
        mapped = np.empty_like(points, dtype=float)
        for axis, (left, right) in enumerate(self.bounds):
            mapped[:, axis] = (points[:, axis] - left) / (right - left)
        return mapped

    @property
    def volume_scale(self) -> float:
        scale = 1.0
        for left, right in self.bounds:
            scale *= right - left
        return float(scale)

    def grid_points(self) -> np.ndarray:
        storage = self.grid.getStorage()
        unit = np.empty((self.size, self.dimension), dtype=float)
        for i in range(self.size):
            point = storage.getPoint(i)
            for axis in range(self.dimension):
                unit[i, axis] = point.getStandardCoordinate(axis)
        return self._unit_to_physical(unit)

    def nodal_values(self) -> np.ndarray:
        pysgpp = import_pysgpp()
        nodal = pysgpp.DataVector(self.alpha)
        pysgpp.createOperationHierarchisation(self.grid).doDehierarchisation(nodal)
        return _from_data_vector(nodal)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        pysgpp = import_pysgpp()
        pts = _coerce_points(points, self.dimension)
        unit = self._physical_to_unit(pts)
        inside = np.all((unit >= 0.0) & (unit <= 1.0), axis=1)

        values = np.zeros(pts.shape[0], dtype=float)
        if np.any(inside):
            matrix = _to_data_matrix(unit[inside])
            result = pysgpp.DataVector(matrix.getNrows())
            pysgpp.createOperationMultipleEval(self.grid, matrix).mult(self.alpha, result)
            values[inside] = _from_data_vector(result)
        return values

    def interpolate_function(self, fn) -> "SGppInterpolant":
        values = np.asarray(fn(self.grid_points().T), dtype=float).reshape(-1)
        if values.shape != (self.size,):
            raise ValueError(f"callable must return shape ({self.size},), got {values.shape}")
        return self.from_nodal_values(self.grid, values, bounds=self.bounds)

    def integral(self) -> float:
        pysgpp = import_pysgpp()
        return self.volume_scale * float(pysgpp.createOperationQuadrature(self.grid).doQuadrature(self.alpha))

    def mean(self) -> np.ndarray:
        mass = self.integral()
        if mass == 0.0:
            return np.full(self.dimension, np.nan)
        numerator = np.empty(self.dimension, dtype=float)
        for axis in range(self.dimension):
            weighted = self.interpolate_function(
                lambda coords, axis=axis: coords[axis] * self.evaluate(coords.T)
            )
            numerator[axis] = weighted.integral()
        return numerator / mass

    def covariance(self) -> np.ndarray:
        mass = self.integral()
        if mass == 0.0:
            return np.full((self.dimension, self.dimension), np.nan)

        mu = self.mean()
        cov = np.empty((self.dimension, self.dimension), dtype=float)
        for i in range(self.dimension):
            for j in range(self.dimension):
                weighted = self.interpolate_function(
                    lambda coords, i=i, j=j, mu=mu: (coords[i] - mu[i])
                    * (coords[j] - mu[j])
                    * self.evaluate(coords.T)
                )
                cov[i, j] = weighted.integral() / mass
        return cov

    def slice_data(
        self,
        *,
        axes: tuple[int, int] = (0, 1),
        fixed: dict[int, float] | None = None,
        resolution: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.dimension < 2:
            raise ValueError("slice plotting requires dimension at least 2")

        fixed = dict(fixed or {})
        ax0, ax1 = axes
        if ax0 == ax1:
            raise ValueError("axes must be distinct")

        x = np.linspace(*self.bounds[ax0], resolution)
        y = np.linspace(*self.bounds[ax1], resolution)
        X, Y = np.meshgrid(x, y, indexing="xy")

        points = np.zeros((resolution * resolution, self.dimension), dtype=float)
        for axis, (left, right) in enumerate(self.bounds):
            points[:, axis] = fixed.get(axis, 0.5 * (left + right))
        points[:, ax0] = X.reshape(-1)
        points[:, ax1] = Y.reshape(-1)

        Z = self.evaluate(points).reshape(resolution, resolution)
        return X, Y, Z

    def save_slice_plot(
        self,
        path: str,
        *,
        axes: tuple[int, int] = (0, 1),
        fixed: dict[int, float] | None = None,
        resolution: int = 100,
        cmap: str = "viridis",
    ) -> None:
        import matplotlib.pyplot as plt

        X, Y, Z = self.slice_data(axes=axes, fixed=fixed, resolution=resolution)
        fig, plot_axes = plt.subplots(1, 2, figsize=(12, 5))

        heatmap = plot_axes[0].pcolormesh(X, Y, Z, cmap=cmap, shading="auto")
        fig.colorbar(heatmap, ax=plot_axes[0], label="value")
        plot_axes[0].set_xlabel(f"x[{axes[0]}]")
        plot_axes[0].set_ylabel(f"x[{axes[1]}]")
        plot_axes[0].set_title("Sparse-grid interpolation")

        contour = plot_axes[1].contourf(X, Y, Z, levels=20, cmap=cmap)
        fig.colorbar(contour, ax=plot_axes[1], label="value")
        plot_axes[1].set_xlabel(f"x[{axes[0]}]")
        plot_axes[1].set_ylabel(f"x[{axes[1]}]")
        plot_axes[1].set_title("Sparse-grid contour")

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)

    @classmethod
    def from_nodal_values(
        cls,
        grid,
        values: np.ndarray,
        *,
        bounds: tuple[tuple[float, float], ...],
    ) -> "SGppInterpolant":
        pysgpp = import_pysgpp()
        storage = grid.getStorage()
        flat = np.asarray(values, dtype=float).reshape(-1)
        if flat.shape != (storage.getSize(),):
            raise ValueError(f"values must have shape ({storage.getSize()},), got {flat.shape}")
        alpha = _to_data_vector(flat)
        pysgpp.createOperationHierarchisation(grid).doHierarchisation(alpha)
        return cls(grid=grid, alpha=alpha, bounds=bounds)

    @classmethod
    def regular(
        cls,
        *,
        dimension: int,
        level: int,
        bounds: tuple[tuple[float, float], ...] | None = None,
        domain_radius: float | tuple[float, ...] | None = None,
        boundary: bool = False,
    ) -> "SGppInterpolant":
        pysgpp = import_pysgpp()
        grid = (
            pysgpp.Grid.createLinearBoundaryGrid(dimension)
            if boundary
            else pysgpp.Grid.createLinearGrid(dimension)
        )
        grid.getGenerator().regular(level)
        alpha = pysgpp.DataVector(grid.getStorage().getSize())
        return cls(
            grid=grid,
            alpha=alpha,
            bounds=_normalize_bounds(bounds, dimension=dimension, domain_radius=domain_radius),
        )

    @classmethod
    def from_function(
        cls,
        fn,
        *,
        dimension: int,
        level: int,
        bounds: tuple[tuple[float, float], ...] | None = None,
        domain_radius: float | tuple[float, ...] | None = None,
        boundary: bool = False,
    ) -> "SGppInterpolant":
        pysgpp = import_pysgpp()
        interpolant = cls.regular(
            dimension=dimension,
            level=level,
            bounds=bounds,
            domain_radius=domain_radius,
            boundary=boundary,
        )
        values = np.asarray(fn(interpolant.grid_points().T), dtype=float).reshape(-1)
        if values.shape != (interpolant.size,):
            raise ValueError(
                f"callable must return shape ({interpolant.size},), got {values.shape}"
            )
        return cls.from_nodal_values(interpolant.grid, values, bounds=interpolant.bounds)
