"""Time stepping on one tensor component grid."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .fd import BoundaryCondition
from .grid import TensorGrid
from .models import LinearFokkerPlanck


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


def solve_on_grid(
    model: LinearFokkerPlanck,
    grid: TensorGrid,
    initial_condition,
    *,
    final_time: float,
    stepper: TimeStepper,
    bc: BoundaryCondition = "dirichlet",
) -> np.ndarray:
    """Solve one component-grid PDE problem and return a flat solution."""

    if final_time < 0.0:
        raise ValueError("final_time must be nonnegative")

    u = grid.values_from_callable(initial_condition).astype(float, copy=True)
    if bc == "dirichlet":
        u[grid.boundary_mask()] = 0.0
    if final_time == 0.0:
        return u

    operator = model.build_operator(grid, bc=bc)
    steps = max(1, ceil(final_time / stepper.dt))
    dt = final_time / steps

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

