"""Time stepping on one tensor component grid."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from time import perf_counter
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .fd import BoundaryCondition
from .grid import TensorGrid
from .models import OperatorModel

OperatorBackend = Literal["matrix", "linear_operator"]
IterativeMethod = Literal["gmres", "bicgstab"]
PreconditionerKind = Literal["none", "jacobi", "ilu"]


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
    preconditioner: PreconditionerKind = "none"
    ilu_drop_tol: float | None = None
    ilu_fill_factor: float | None = None
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
        if self.preconditioner not in {"none", "jacobi", "ilu"}:
            raise ValueError("preconditioner must be 'none', 'jacobi', or 'ilu'")
        if self.ilu_drop_tol is not None and self.ilu_drop_tol < 0.0:
            raise ValueError("ilu_drop_tol must be nonnegative when provided")
        if self.ilu_fill_factor is not None and self.ilu_fill_factor <= 0.0:
            raise ValueError("ilu_fill_factor must be positive when provided")


@dataclass(frozen=True)
class GridSolveStats:
    operator_backend: OperatorBackend
    preconditioner: PreconditionerKind
    method: str
    steps: int
    grid_size: int
    operator_setup_seconds: float
    solve_seconds: float
    total_seconds: float
    krylov_iterations: int = 0
    krylov_iterations_per_step: tuple[int, ...] = ()


@dataclass(frozen=True)
class GridSolveResult:
    values: np.ndarray
    stats: GridSolveStats


def _solve_iterative(system, rhs: np.ndarray, config: LinearSolveConfig, preconditioner=None) -> tuple[np.ndarray, int]:
    iterations = 0

    def callback(_residual) -> None:
        nonlocal iterations
        iterations += 1

    if config.method == "gmres":
        solution, info = spla.gmres(
            system,
            rhs,
            M=preconditioner,
            callback=callback,
            callback_type="pr_norm",
            rtol=config.rtol,
            atol=config.atol,
            maxiter=config.maxiter,
        )
    elif config.method == "bicgstab":
        solution, info = spla.bicgstab(
            system,
            rhs,
            M=preconditioner,
            callback=callback,
            rtol=config.rtol,
            atol=config.atol,
            maxiter=config.maxiter,
        )
    else:
        raise ValueError(f"unsupported iterative method: {config.method}")

    if info != 0:
        raise RuntimeError(f"iterative solve failed with info={info}")
    return np.asarray(solution, dtype=float), iterations


def _build_ilu_preconditioner(system: sparse.spmatrix, config: LinearSolveConfig) -> spla.LinearOperator:
    ilu = spla.spilu(
        system.tocsc(),
        drop_tol=config.ilu_drop_tol,
        fill_factor=config.ilu_fill_factor,
    )
    return spla.LinearOperator(system.shape, matvec=ilu.solve, dtype=float)


def _stats_method_name(operator_backend: OperatorBackend, linear_solve: LinearSolveConfig, theta: float) -> str:
    if theta == 0.0:
        return "explicit_euler"
    if operator_backend == "matrix" and linear_solve.preconditioner == "none":
        return "direct"
    return linear_solve.method


def solve_on_grid_with_stats(
    model: OperatorModel,
    grid: TensorGrid,
    initial_condition,
    *,
    final_time: float,
    stepper: TimeStepper,
    bc: BoundaryCondition = "dirichlet",
    operator_backend: OperatorBackend = "matrix",
    linear_solve: LinearSolveConfig | None = None,
) -> GridSolveResult:
    """Solve one component-grid PDE problem and return values with diagnostics."""

    start_time = perf_counter()

    if final_time < 0.0:
        raise ValueError("final_time must be nonnegative")

    u = grid.values_from_callable(initial_condition).astype(float, copy=True)
    if bc == "dirichlet":
        u[grid.boundary_mask()] = 0.0
    linear_solve = linear_solve or LinearSolveConfig()
    if final_time == 0.0:
        total_seconds = perf_counter() - start_time
        return GridSolveResult(
            values=np.asarray(u),
            stats=GridSolveStats(
                operator_backend=operator_backend,
                preconditioner=linear_solve.preconditioner,
                method="sampling",
                steps=0,
                grid_size=grid.size,
                operator_setup_seconds=0.0,
                solve_seconds=0.0,
                total_seconds=total_seconds,
            ),
        )

    steps = max(1, ceil(final_time / stepper.dt))
    dt = final_time / steps
    setup_start = perf_counter()
    solve_seconds = 0.0
    krylov_iterations = 0
    krylov_iterations_per_step: list[int] = []

    if operator_backend == "matrix":
        operator = model.build_operator(grid, bc=bc)
        operator_setup_seconds = perf_counter() - setup_start

        if stepper.theta == 0.0:
            advance = sparse.eye(grid.size, format="csr") + dt * operator
            solve_start = perf_counter()
            for _ in range(steps):
                u = advance @ u
                if bc == "dirichlet":
                    u[grid.boundary_mask()] = 0.0
            solve_seconds = perf_counter() - solve_start
            total_seconds = perf_counter() - start_time
            return GridSolveResult(
                values=np.asarray(u),
                stats=GridSolveStats(
                    operator_backend=operator_backend,
                    preconditioner=linear_solve.preconditioner,
                    method=_stats_method_name(operator_backend, linear_solve, stepper.theta),
                    steps=steps,
                    grid_size=grid.size,
                    operator_setup_seconds=operator_setup_seconds,
                    solve_seconds=solve_seconds,
                    total_seconds=total_seconds,
                ),
            )

        identity = sparse.eye(grid.size, format="csc")
        lhs = identity - stepper.theta * dt * operator.tocsc()
        rhs = sparse.eye(grid.size, format="csr") + (1.0 - stepper.theta) * dt * operator

        if linear_solve.preconditioner == "jacobi":
            raise ValueError("jacobi preconditioning is only supported with operator_backend='linear_operator'")

        if linear_solve.preconditioner == "none":
            solve_start = perf_counter()
            solve = spla.factorized(lhs)
            for _ in range(steps):
                u = solve(rhs @ u)
                if bc == "dirichlet":
                    u[grid.boundary_mask()] = 0.0
            solve_seconds = perf_counter() - solve_start
            total_seconds = perf_counter() - start_time
            return GridSolveResult(
                values=np.asarray(u),
                stats=GridSolveStats(
                    operator_backend=operator_backend,
                    preconditioner=linear_solve.preconditioner,
                    method=_stats_method_name(operator_backend, linear_solve, stepper.theta),
                    steps=steps,
                    grid_size=grid.size,
                    operator_setup_seconds=operator_setup_seconds,
                    solve_seconds=solve_seconds,
                    total_seconds=total_seconds,
                ),
            )

        if linear_solve.preconditioner != "ilu":
            raise ValueError("matrix backend supports preconditioner='none' or 'ilu'")

        preconditioner = _build_ilu_preconditioner(lhs, linear_solve)
        solve_start = perf_counter()
        for _ in range(steps):
            u, iterations = _solve_iterative(lhs, rhs @ u, linear_solve, preconditioner=preconditioner)
            krylov_iterations += iterations
            krylov_iterations_per_step.append(iterations)
            if bc == "dirichlet":
                u[grid.boundary_mask()] = 0.0
        solve_seconds = perf_counter() - solve_start
        total_seconds = perf_counter() - start_time
        return GridSolveResult(
            values=np.asarray(u),
            stats=GridSolveStats(
                operator_backend=operator_backend,
                preconditioner=linear_solve.preconditioner,
                method=_stats_method_name(operator_backend, linear_solve, stepper.theta),
                steps=steps,
                grid_size=grid.size,
                operator_setup_seconds=operator_setup_seconds,
                solve_seconds=solve_seconds,
                total_seconds=total_seconds,
                krylov_iterations=krylov_iterations,
                krylov_iterations_per_step=tuple(krylov_iterations_per_step),
            ),
        )

    if operator_backend != "linear_operator":
        raise ValueError("operator_backend must be 'matrix' or 'linear_operator'")

    operator = model.build_linear_operator(grid, bc=bc)
    operator_setup_seconds = perf_counter() - setup_start

    def apply_rhs(vector: np.ndarray) -> np.ndarray:
        return vector + (1.0 - stepper.theta) * dt * (operator @ vector)

    if stepper.theta == 0.0:
        solve_start = perf_counter()
        for _ in range(steps):
            u = apply_rhs(u)
            if bc == "dirichlet":
                u[grid.boundary_mask()] = 0.0
        solve_seconds = perf_counter() - solve_start
        total_seconds = perf_counter() - start_time
        return GridSolveResult(
            values=np.asarray(u),
            stats=GridSolveStats(
                operator_backend=operator_backend,
                preconditioner=linear_solve.preconditioner,
                method=_stats_method_name(operator_backend, linear_solve, stepper.theta),
                steps=steps,
                grid_size=grid.size,
                operator_setup_seconds=operator_setup_seconds,
                solve_seconds=solve_seconds,
                total_seconds=total_seconds,
            ),
        )

    lhs = spla.LinearOperator(
        (grid.size, grid.size),
        matvec=lambda x: np.asarray(x, dtype=float).reshape(-1) - stepper.theta * dt * (operator @ x),
        dtype=float,
    )
    preconditioner = None
    if linear_solve.preconditioner == "ilu":
        raise ValueError("ilu preconditioning is only supported with operator_backend='matrix'")
    if linear_solve.preconditioner == "jacobi":
        lhs_diagonal = 1.0 - stepper.theta * dt * model.operator_diagonal(grid, bc=bc)
        safe_diagonal = lhs_diagonal.copy()
        near_zero = np.isclose(safe_diagonal, 0.0)
        safe_diagonal[near_zero] = 1.0
        inverse_diagonal = 1.0 / safe_diagonal
        inverse_diagonal[near_zero] = 1.0
        preconditioner = spla.LinearOperator(
            (grid.size, grid.size),
            matvec=lambda x: inverse_diagonal * np.asarray(x, dtype=float).reshape(-1),
            dtype=float,
        )

    solve_start = perf_counter()
    for _ in range(steps):
        u, iterations = _solve_iterative(lhs, apply_rhs(u), linear_solve, preconditioner=preconditioner)
        krylov_iterations += iterations
        krylov_iterations_per_step.append(iterations)
        if bc == "dirichlet":
            u[grid.boundary_mask()] = 0.0

    solve_seconds = perf_counter() - solve_start
    total_seconds = perf_counter() - start_time
    return GridSolveResult(
        values=np.asarray(u),
        stats=GridSolveStats(
            operator_backend=operator_backend,
            preconditioner=linear_solve.preconditioner,
            method=_stats_method_name(operator_backend, linear_solve, stepper.theta),
            steps=steps,
            grid_size=grid.size,
            operator_setup_seconds=operator_setup_seconds,
            solve_seconds=solve_seconds,
            total_seconds=total_seconds,
            krylov_iterations=krylov_iterations,
            krylov_iterations_per_step=tuple(krylov_iterations_per_step),
        ),
    )


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
    steps. For that path, ``LinearSolveConfig(preconditioner="jacobi")``
    enables a diagonal left preconditioner for the implicit system. For the
    assembled matrix path, ``preconditioner="ilu"`` uses SciPy's `spilu`.
    """

    return solve_on_grid_with_stats(
        model,
        grid,
        initial_condition,
        final_time=final_time,
        stepper=stepper,
        bc=bc,
        operator_backend=operator_backend,
        linear_solve=linear_solve,
    ).values
