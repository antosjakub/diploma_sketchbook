"""Benchmark helpers for backend and preconditioner comparisons."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from .combination import CombinationResult, solve_combination
from .initial import gaussian_density
from .models import OrnsteinUhlenbeck
from .solver import LinearSolveConfig, TimeStepper


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    operator_backend: str
    linear_solve: LinearSolveConfig


DEFAULT_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        name="matrix_direct",
        operator_backend="matrix",
        linear_solve=LinearSolveConfig(preconditioner="none"),
    ),
    BenchmarkCase(
        name="matrix_ilu",
        operator_backend="matrix",
        linear_solve=LinearSolveConfig(
            method="gmres",
            preconditioner="ilu",
            ilu_drop_tol=1e-4,
            ilu_fill_factor=10.0,
            maxiter=200,
        ),
    ),
    BenchmarkCase(
        name="linear_operator_jacobi",
        operator_backend="linear_operator",
        linear_solve=LinearSolveConfig(
            method="gmres",
            preconditioner="jacobi",
            maxiter=200,
        ),
    ),
)


def equicorrelated_covariance(dimension: int, rho: float, variance: float = 1.0) -> np.ndarray:
    """Return a symmetric positive definite equicorrelated covariance."""

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    lower_bound = -1.0 / (dimension - 1) if dimension > 1 else -np.inf
    if not (lower_bound < rho < 1.0):
        raise ValueError(
            f"rho must lie in ({lower_bound}, 1.0) for dimension={dimension}"
        )
    if variance <= 0.0:
        raise ValueError("variance must be positive")

    covariance = np.full((dimension, dimension), rho, dtype=float)
    np.fill_diagonal(covariance, 1.0)
    return variance * covariance


def _summary_row(
    *,
    repeat: int,
    case: BenchmarkCase,
    result: CombinationResult,
    wall_seconds: float,
    dimension: int,
    level: int,
) -> dict[str, object]:
    return {
        "row_type": "summary",
        "repeat": repeat,
        "case": case.name,
        "operator_backend": case.operator_backend,
        "preconditioner": case.linear_solve.preconditioner,
        "method": case.linear_solve.method,
        "dimension": dimension,
        "level": level,
        "num_components": len(result.components),
        "wall_seconds": wall_seconds,
        "total_component_time": result.total_component_time,
        "total_krylov_iterations": result.total_krylov_iterations,
        "component_levels": "",
        "component_weight": "",
        "component_grid_size": "",
        "component_total_seconds": "",
        "component_setup_seconds": "",
        "component_solve_seconds": "",
        "component_steps": "",
        "component_krylov_iterations": "",
        "component_krylov_iterations_per_step": "",
    }


def _component_rows(
    *,
    repeat: int,
    case: BenchmarkCase,
    result: CombinationResult,
    dimension: int,
    level: int,
) -> Iterable[dict[str, object]]:
    for component in result.components:
        yield {
            "row_type": "component",
            "repeat": repeat,
            "case": case.name,
            "operator_backend": case.operator_backend,
            "preconditioner": component.stats.preconditioner,
            "method": component.stats.method,
            "dimension": dimension,
            "level": level,
            "num_components": len(result.components),
            "wall_seconds": "",
            "total_component_time": "",
            "total_krylov_iterations": "",
            "component_levels": ",".join(str(entry) for entry in component.levels),
            "component_weight": component.weight,
            "component_grid_size": component.grid.size,
            "component_total_seconds": component.stats.total_seconds,
            "component_setup_seconds": component.stats.operator_setup_seconds,
            "component_solve_seconds": component.stats.solve_seconds,
            "component_steps": component.stats.steps,
            "component_krylov_iterations": component.stats.krylov_iterations,
            "component_krylov_iterations_per_step": ",".join(
                str(entry) for entry in component.stats.krylov_iterations_per_step
            ),
        }


def run_backend_benchmark(
    *,
    output_path: str | Path,
    dimension: int,
    level: int,
    final_time: float,
    dt: float,
    domain_radius: float,
    rho: float,
    max_workers: int | None,
    repeats: int = 1,
    min_level: int = 1,
    max_component_size: int | None = None,
    cases: Iterable[BenchmarkCase] = DEFAULT_CASES,
) -> list[dict[str, object]]:
    """Run backend comparisons and write a CSV with summary and component rows."""

    covariance = equicorrelated_covariance(dimension, rho)
    model = OrnsteinUhlenbeck(covariance)
    stepper = TimeStepper(dt=dt, theta=1.0)

    rows: list[dict[str, object]] = []
    for repeat in range(1, repeats + 1):
        for case in cases:
            start = perf_counter()
            result = solve_combination(
                model,
                level=level,
                initial_condition=gaussian_density,
                final_time=final_time,
                stepper=stepper,
                domain_radius=domain_radius,
                bc="dirichlet",
                max_workers=max_workers,
                min_level=min_level,
                max_component_size=max_component_size,
                operator_backend=case.operator_backend,
                linear_solve=case.linear_solve,
            )
            wall_seconds = perf_counter() - start
            rows.append(
                _summary_row(
                    repeat=repeat,
                    case=case,
                    result=result,
                    wall_seconds=wall_seconds,
                    dimension=dimension,
                    level=level,
                )
            )
            rows.extend(
                _component_rows(
                    repeat=repeat,
                    case=case,
                    result=result,
                    dimension=dimension,
                    level=level,
                )
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows
