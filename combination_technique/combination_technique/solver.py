"""Time stepping on one tensor component grid."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .fd import BoundaryCondition
from .grid import TensorGrid
from .models import OperatorModel

OperatorBackend = Literal["matrix", "linear_operator"]
IterativeMethod = Literal["gmres", "bicgstab"]


@dataclass(frozen=True)
class TimeStepper:
    """Theta-method time stepper.

    ``theta=1`` is backward Euler, ``theta=0.5`` is Crank-Nicolson, and
    ``theta=0`` is explicit Euler.
    """

    dt: float
    theta: float = 1.0

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if not 0.0 <= self.theta <= 1.0:
            raise ValueError("theta must lie in [0, 1]")


@dataclass(frozen=True)
class LinearSolveConfig:
    """Configuration for implicit iterative solves."""

    method: IterativeMethod = "gmres"
    rtol: float = 1e-8
    atol: float = 0.0
    maxiter: int | None = None

    def __post_init__(self) -> None:
        if self.rtol < 0.0:
            raise ValueError("rtol must be nonnegative")
        if self.atol < 0.0:
            raise ValueError("atol must be nonnegative")
        if self.maxiter is not None and self.maxiter <= 0:
            raise ValueError("maxiter must be positive when provided")


def _solve_iterative(system, rhs: np.ndarray, config: LinearSolveConfig) -> np.ndarray:
    if config.method == "gmres":
        solution, info = spla.gmres(
            system,
            rhs,
            rtol=config.rtol,
            atol=config.atol,
            maxiter=config.maxiter,
        )
    elif config.method == "bicgstab":
        solution, info = spla.bicgstab(
            system,
            rhs,
            rtol=config.rtol,
            atol=config.atol,
            maxiter=config.maxiter,
        )
    else:
        raise ValueError(f"unsupported iterative method: {config.method}")

    if info != 0:
        raise RuntimeError(f"iterative solve failed with info={info}")
    return np.asarray(solution, dtype=float)


def solve_on_grid(
    model: OperatorModel,
    grid: TensorGrid,
    initial_condition,
    *,
    final_time: float,
    stepper: TimeStepper,
    bc: BoundaryCondition = "dirichlet",
    operator_backend: OperatorBackend = "matrix",
    linear_solve: LinearSolveConfig | None = None,
) -> np.ndarray:
    """Solve one component-grid PDE problem and return a flat solution.

    ``operator_backend="matrix"`` keeps the original assembled sparse-matrix
    path. ``operator_backend="linear_operator"`` uses a matrix-free
    ``scipy.sparse.linalg.LinearOperator`` and iterative solves for implicit
    steps.
    """

    if final_time < 0.0:
        raise ValueError("final_time must be nonnegative")

    u = grid.values_from_callable(initial_condition).astype(float, copy=True)
    if bc == "dirichlet":
        u[grid.boundary_mask()] = 0.0
    if final_time == 0.0:
        return u

    steps = max(1, ceil(final_time / stepper.dt))
    dt = final_time / steps
    linear_solve = linear_solve or LinearSolveConfig()

    if operator_backend == "matrix":
        operator = model.build_operator(grid, bc=bc)

        if stepper.theta == 0.0:
            advance = sparse.eye(grid.size, format="csr") + dt * operator
            for _ in range(steps):
                u = advance @ u
                if bc == "dirichlet":
                    u[grid.boundary_mask()] = 0.0
            return np.asarray(u)

        identity = sparse.eye(grid.size, format="csc")
        lhs = identity - stepper.theta * dt * operator.tocsc()
        rhs = sparse.eye(grid.size, format="csr") + (1.0 - stepper.theta) * dt * operator
        solve = spla.factorized(lhs)

        for _ in range(steps):
            u = solve(rhs @ u)
            if bc == "dirichlet":
                u[grid.boundary_mask()] = 0.0
        return np.asarray(u)

    if operator_backend != "linear_operator":
        raise ValueError("operator_backend must be 'matrix' or 'linear_operator'")

    operator = model.build_linear_operator(grid, bc=bc)

    def apply_rhs(vector: np.ndarray) -> np.ndarray:
        return vector + (1.0 - stepper.theta) * dt * (operator @ vector)

    if stepper.theta == 0.0:
        for _ in range(steps):
            u = apply_rhs(u)
            if bc == "dirichlet":
                u[grid.boundary_mask()] = 0.0
        return np.asarray(u)

    lhs = spla.LinearOperator(
        (grid.size, grid.size),
        matvec=lambda x: np.asarray(x, dtype=float).reshape(-1) - stepper.theta * dt * (operator @ x),
        dtype=float,
    )

    for _ in range(steps):
        u = _solve_iterative(lhs, apply_rhs(u), linear_solve)
        if bc == "dirichlet":
            u[grid.boundary_mask()] = 0.0

    return np.asarray(u)
