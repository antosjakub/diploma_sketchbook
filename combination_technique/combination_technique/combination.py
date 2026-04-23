"""Sparse-grid combination driver."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .fd import BoundaryCondition
from .grid import TensorGrid
from .indices import combination_indices, combination_weight
from .models import OperatorModel
from .solver import TimeStepper, solve_on_grid


@dataclass(frozen=True)
class ComponentSolution:
    levels: tuple[int, ...]
    weight: int
    grid: TensorGrid
    values: np.ndarray

    @property
    def shaped_values(self) -> np.ndarray:
        return self.values.reshape(self.grid.shape)

    def interpolate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate this component solution at points with shape ``(m, d)``."""

        interpolator = RegularGridInterpolator(
            self.grid.axes,
            self.shaped_values,
            bounds_error=False,
            fill_value=0.0,
        )
        return np.asarray(interpolator(points), dtype=float)


@dataclass(frozen=True)
class CombinationResult:
    level: int
    dimension: int
    components: tuple[ComponentSolution, ...]

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the combined solution at points.

        ``points`` may have shape ``(m, d)`` or ``(d, m)``.
        """

        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2:
            raise ValueError("points must be a two-dimensional array")
        if pts.shape[1] != self.dimension and pts.shape[0] == self.dimension:
            pts = pts.T
        if pts.shape[1] != self.dimension:
            raise ValueError("points must have dimension columns")

        total = np.zeros(pts.shape[0], dtype=float)
        for component in self.components:
            total += component.weight * component.interpolate(pts)
        return total

    def combine_on_grid(self, target_grid: TensorGrid) -> np.ndarray:
        coords = target_grid.flat_coordinates().T
        return self.evaluate(coords).reshape(target_grid.shape)


def _solve_component(args) -> ComponentSolution:
    (
        model,
        levels,
        weight,
        domain_radius,
        initial_condition,
        final_time,
        stepper,
        bc,
    ) = args
    grid = TensorGrid.from_level(levels, domain_radius=domain_radius)
    values = solve_on_grid(
        model,
        grid,
        initial_condition,
        final_time=final_time,
        stepper=stepper,
        bc=bc,
    )
    return ComponentSolution(levels=levels, weight=weight, grid=grid, values=values)


def solve_combination(
    model: OperatorModel,
    *,
    level: int,
    initial_condition,
    final_time: float,
    stepper: TimeStepper,
    domain_radius: float | tuple[float, ...],
    bc: BoundaryCondition = "dirichlet",
    max_workers: int | None = None,
    min_level: int = 1,
    max_component_size: int | None = None,
) -> CombinationResult:
    """Solve all component grids and return an implicit combination result."""

    dimension = model.dimension
    levels = combination_indices(level, dimension, min_level=min_level)
    tasks = []
    for ell in levels:
        grid = TensorGrid.from_level(ell, domain_radius=domain_radius)
        if max_component_size is not None and grid.size > max_component_size:
            raise ValueError(
                f"component grid {ell} has size {grid.size}, "
                f"exceeding max_component_size={max_component_size}"
            )
        tasks.append(
            (
                model,
                ell,
                combination_weight(level, dimension, ell),
                domain_radius,
                initial_condition,
                final_time,
                stepper,
                bc,
            )
        )

    if max_workers == 1 or len(tasks) == 1:
        components: Iterable[ComponentSolution] = map(_solve_component, tasks)
        return CombinationResult(level, dimension, tuple(components))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        components = tuple(executor.map(_solve_component, tasks))
    return CombinationResult(level, dimension, components)
