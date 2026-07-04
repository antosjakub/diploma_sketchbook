"""Linear Fokker-Planck and convection-diffusion-reaction models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from .fd import (
    BoundaryCondition,
    apply_axis_operator,
    lift_axis_diagonal,
    tensor_derivative_1d_matrices,
    tensor_derivative_matrices,
)
from .grid import TensorGrid


class Potential(Protocol):
    dimension: int

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        """Return grad V with shape ``(dimension, npoints)``."""

    def laplacian(self, coords: np.ndarray) -> np.ndarray:
        """Return Delta V with shape ``(npoints,)``."""


class OperatorModel(Protocol):
    dimension: int

    def build_operator(
        self,
        grid: TensorGrid,
        *,
        bc: BoundaryCondition = "dirichlet",
    ) -> sparse.csr_matrix:
        """Assemble the semidiscrete operator on one tensor grid."""

    def build_linear_operator(
        self,
        grid: TensorGrid,
        *,
        bc: BoundaryCondition = "dirichlet",
    ) -> LinearOperator:
        """Build a matrix-free semidiscrete operator on one tensor grid."""

    def operator_diagonal(
        self,
        grid: TensorGrid,
        *,
        bc: BoundaryCondition = "dirichlet",
    ) -> np.ndarray:
        """Return the diagonal of the semidiscrete operator."""


def _diffusion_scalar_identity(diffusion: np.ndarray) -> float | None:
    """Return alpha when diffusion is alpha * I, else ``None``."""

    diagonal = np.diag(diffusion)
    if diagonal.size == 0:
        return 0.0
    alpha = float(diagonal[0])
    if np.allclose(diffusion, alpha * np.eye(diffusion.shape[0])):
        return alpha
    return None


def _active_diagonal_diffusion(diffusion: np.ndarray) -> list[tuple[int, float]]:
    """Return nonzero diagonal entries as ``(axis, coefficient)`` pairs."""

    diagonal = np.diag(diffusion)
    return [
        (axis, float(coefficient))
        for axis, coefficient in enumerate(diagonal)
        if coefficient != 0.0
    ]


def _active_mixed_diffusion(diffusion: np.ndarray) -> list[tuple[float, int, int]]:
    """Return nonzero off-diagonal diffusion entries."""

    active: list[tuple[float, int, int]] = []
    for i in range(diffusion.shape[0]):
        for j in range(diffusion.shape[1]):
            if i == j:
                continue
            coefficient = float(diffusion[i, j])
            if coefficient != 0.0:
                active.append((coefficient, i, j))
    return active


@dataclass(frozen=True)
class LinearFokkerPlanck:
    """Linear Fokker-Planck equation with constant diffusion.

    The represented equation is

        p_t = sum_ij diffusion_ij d_ij p - div(drift p)

    Expanding the divergence gives

        p_t = sum_ij diffusion_ij d_ij p
              - drift . grad p
              - div(drift) p
    """

    diffusion: np.ndarray
    dimension: int

    def __post_init__(self) -> None:
        diffusion = np.asarray(self.diffusion, dtype=float)
        if diffusion.shape != (self.dimension, self.dimension):
            raise ValueError("diffusion must have shape (dimension, dimension)")
        object.__setattr__(self, "diffusion", diffusion)

    def drift(self, coords: np.ndarray) -> np.ndarray:
        return np.zeros_like(coords)

    def divergence_drift(self, coords: np.ndarray) -> np.ndarray:
        return np.zeros(coords.shape[1], dtype=float)

    def _diffusion_structure(self) -> tuple[float | None, list[tuple[int, float]], list[tuple[float, int, int]]]:
        isotropic = _diffusion_scalar_identity(self.diffusion)
        diagonal = _active_diagonal_diffusion(self.diffusion)
        mixed = _active_mixed_diffusion(self.diffusion)
        return isotropic, diagonal, mixed

    def build_operator(
        self,
        grid: TensorGrid,
        *,
        bc: BoundaryCondition = "dirichlet",
    ) -> sparse.csr_matrix:
        if grid.dimension != self.dimension:
            raise ValueError("grid dimension does not match model dimension")

        d1, d2 = tensor_derivative_matrices(grid.shape, grid.spacing, bc)
        size = grid.size
        operator = sparse.csr_matrix((size, size), dtype=float)
        isotropic, diagonal_diffusion, mixed_diffusion = self._diffusion_structure()

        if isotropic is not None:
            for second in d2:
                operator = operator + isotropic * second
        else:
            for axis, coefficient in diagonal_diffusion:
                operator = operator + coefficient * d2[axis]
            for coefficient, i, j in mixed_diffusion:
                operator = operator + coefficient * (d1[i] @ d1[j])

        coords = grid.flat_coordinates()
        drift_values = np.asarray(self.drift(coords), dtype=float)
        if drift_values.shape != coords.shape:
            raise ValueError("drift must return shape (dimension, npoints)")

        for axis in range(self.dimension):
            values = -drift_values[axis]
            if np.any(values):
                operator = operator + sparse.diags(values, format="csr") @ d1[axis]

        div_values = -np.asarray(self.divergence_drift(coords), dtype=float)
        if div_values.shape != (size,):
            raise ValueError("divergence_drift must return shape (npoints,)")
        if np.any(div_values):
            operator = operator + sparse.diags(div_values, format="csr")

        if bc == "dirichlet":
            operator = operator.tolil()
            operator[grid.boundary_mask(), :] = 0.0
            operator = operator.tocsr()

        return operator

    def build_linear_operator(
        self,
        grid: TensorGrid,
        *,
        bc: BoundaryCondition = "dirichlet",
    ) -> LinearOperator:
        if grid.dimension != self.dimension:
            raise ValueError("grid dimension does not match model dimension")

        d1_1d, d2_1d = tensor_derivative_1d_matrices(grid.shape, grid.spacing, bc)
        coords = grid.flat_coordinates()
        drift_values = np.asarray(self.drift(coords), dtype=float)
        if drift_values.shape != coords.shape:
            raise ValueError("drift must return shape (dimension, npoints)")

        div_values = -np.asarray(self.divergence_drift(coords), dtype=float)
        if div_values.shape != (grid.size,):
            raise ValueError("divergence_drift must return shape (npoints,)")

        isotropic, diagonal_diffusion, mixed_diffusion = self._diffusion_structure()

        active_drift = [
            (axis, -drift_values[axis].copy())
            for axis in range(self.dimension)
            if np.any(drift_values[axis])
        ]
        has_divergence = np.any(div_values)
        boundary_mask = grid.boundary_mask() if bc == "dirichlet" else None

        def matvec(x: np.ndarray) -> np.ndarray:
            vector = np.asarray(x, dtype=float).reshape(-1)
            result = np.zeros_like(vector)

            if isotropic is not None:
                for axis, second in enumerate(d2_1d):
                    result += isotropic * apply_axis_operator(vector, second, grid.shape, axis)
            else:
                for axis, coefficient in diagonal_diffusion:
                    result += coefficient * apply_axis_operator(vector, d2_1d[axis], grid.shape, axis)
                for coefficient, i, j in mixed_diffusion:
                    mixed = apply_axis_operator(vector, d1_1d[j], grid.shape, j)
                    result += coefficient * apply_axis_operator(mixed, d1_1d[i], grid.shape, i)

            for axis, values in active_drift:
                result += values * apply_axis_operator(vector, d1_1d[axis], grid.shape, axis)

            if has_divergence:
                result += div_values * vector

            if boundary_mask is not None:
                result[boundary_mask] = 0.0
            return result

        return LinearOperator((grid.size, grid.size), matvec=matvec, dtype=float)

    def operator_diagonal(
        self,
        grid: TensorGrid,
        *,
        bc: BoundaryCondition = "dirichlet",
    ) -> np.ndarray:
        if grid.dimension != self.dimension:
            raise ValueError("grid dimension does not match model dimension")

        _, d2_1d = tensor_derivative_1d_matrices(grid.shape, grid.spacing, bc)
        diagonal = np.zeros(grid.size, dtype=float)
        isotropic, diagonal_diffusion, _ = self._diffusion_structure()

        if isotropic is not None:
            for axis, second in enumerate(d2_1d):
                diagonal += isotropic * lift_axis_diagonal(second.diagonal(), grid.shape, axis)
        else:
            for axis, coefficient in diagonal_diffusion:
                diagonal += coefficient * lift_axis_diagonal(d2_1d[axis].diagonal(), grid.shape, axis)

        coords = grid.flat_coordinates()
        div_values = -np.asarray(self.divergence_drift(coords), dtype=float)
        if div_values.shape != (grid.size,):
            raise ValueError("divergence_drift must return shape (npoints,)")
        diagonal += div_values

        if bc == "dirichlet":
            diagonal[grid.boundary_mask()] = 0.0

        return diagonal


@dataclass(frozen=True)
class ConvectionDiffusionReaction:
    """Model for ``p_t = a Δp + b(x) · ∇p + c(x) p``."""

    dimension: int
    diffusion: float
    drift_fn: Callable[[np.ndarray], np.ndarray] | None = None
    reaction_fn: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.diffusion < 0.0:
            raise ValueError("diffusion must be nonnegative")

    def drift(self, coords: np.ndarray) -> np.ndarray:
        if self.drift_fn is None:
            return np.zeros_like(coords)
        values = np.asarray(self.drift_fn(coords), dtype=float)
        if values.shape != coords.shape:
            raise ValueError("drift_fn must return shape (dimension, npoints)")
        return values

    def reaction(self, coords: np.ndarray) -> np.ndarray:
        if self.reaction_fn is None:
            return np.zeros(coords.shape[1], dtype=float)
        values = np.asarray(self.reaction_fn(coords), dtype=float).reshape(-1)
        if values.shape != (coords.shape[1],):
            raise ValueError("reaction_fn must return shape (npoints,)")
        return values

    def build_operator(
        self,
        grid: TensorGrid,
        *,
        bc: BoundaryCondition = "dirichlet",
    ) -> sparse.csr_matrix:
        if grid.dimension != self.dimension:
            raise ValueError("grid dimension does not match model dimension")

        d1, d2 = tensor_derivative_matrices(grid.shape, grid.spacing, bc)
        operator = sparse.csr_matrix((grid.size, grid.size), dtype=float)
        if self.diffusion:
            for second in d2:
                operator = operator + self.diffusion * second

        coords = grid.flat_coordinates()
        drift_values = self.drift(coords)
        for axis in range(self.dimension):
            values = drift_values[axis]
            if np.any(values):
                operator = operator + sparse.diags(values, format="csr") @ d1[axis]

        reaction_values = self.reaction(coords)
        if np.any(reaction_values):
            operator = operator + sparse.diags(reaction_values, format="csr")

        if bc == "dirichlet":
            operator = operator.tolil()
            operator[grid.boundary_mask(), :] = 0.0
            operator = operator.tocsr()

        return operator

    def build_linear_operator(
        self,
        grid: TensorGrid,
        *,
        bc: BoundaryCondition = "dirichlet",
    ) -> LinearOperator:
        if grid.dimension != self.dimension:
            raise ValueError("grid dimension does not match model dimension")

        d1_1d, d2_1d = tensor_derivative_1d_matrices(grid.shape, grid.spacing, bc)
        coords = grid.flat_coordinates()
        drift_values = self.drift(coords)
        reaction_values = self.reaction(coords)
        active_drift = [
            (axis, drift_values[axis].copy())
            for axis in range(self.dimension)
            if np.any(drift_values[axis])
        ]
        has_reaction = np.any(reaction_values)
        boundary_mask = grid.boundary_mask() if bc == "dirichlet" else None

        def matvec(x: np.ndarray) -> np.ndarray:
            vector = np.asarray(x, dtype=float).reshape(-1)
            result = np.zeros_like(vector)

            if self.diffusion:
                for axis, second in enumerate(d2_1d):
                    result += self.diffusion * apply_axis_operator(vector, second, grid.shape, axis)

            for axis, values in active_drift:
                result += values * apply_axis_operator(vector, d1_1d[axis], grid.shape, axis)

            if has_reaction:
                result += reaction_values * vector

            if boundary_mask is not None:
                result[boundary_mask] = 0.0
            return result

        return LinearOperator((grid.size, grid.size), matvec=matvec, dtype=float)

    def operator_diagonal(
        self,
        grid: TensorGrid,
        *,
        bc: BoundaryCondition = "dirichlet",
    ) -> np.ndarray:
        if grid.dimension != self.dimension:
            raise ValueError("grid dimension does not match model dimension")

        _, d2_1d = tensor_derivative_1d_matrices(grid.shape, grid.spacing, bc)
        diagonal = np.zeros(grid.size, dtype=float)

        if self.diffusion:
            for axis in range(self.dimension):
                diagonal += self.diffusion * lift_axis_diagonal(d2_1d[axis].diagonal(), grid.shape, axis)

        coords = grid.flat_coordinates()
        reaction_values = self.reaction(coords)
        diagonal += reaction_values

        if bc == "dirichlet":
            diagonal[grid.boundary_mask()] = 0.0

        return diagonal


@dataclass(frozen=True)
class OrnsteinUhlenbeck(LinearFokkerPlanck):
    """Ornstein-Uhlenbeck model dX = -0.5 X dt + Sigma^{1/2} dW."""

    covariance: np.ndarray

    def __init__(self, covariance: np.ndarray):
        covariance = np.asarray(covariance, dtype=float)
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("covariance must be a square matrix")
        object.__setattr__(self, "covariance", covariance)
        super().__init__(diffusion=0.5 * covariance, dimension=covariance.shape[0])

    def drift(self, coords: np.ndarray) -> np.ndarray:
        return -0.5 * coords

    def divergence_drift(self, coords: np.ndarray) -> np.ndarray:
        return np.full(coords.shape[1], -0.5 * self.dimension, dtype=float)


@dataclass(frozen=True)
class QuadraticPotential:
    matrix: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "dimension", matrix.shape[0])

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        return self.matrix @ coords

    def laplacian(self, coords: np.ndarray) -> np.ndarray:
        return np.full(coords.shape[1], np.trace(self.matrix), dtype=float)


@dataclass(frozen=True)
class DoubleWellPotential:
    wells: np.ndarray

    def __post_init__(self) -> None:
        wells = np.asarray(self.wells, dtype=float)
        if wells.ndim != 1:
            raise ValueError("wells must be a one-dimensional array")
        object.__setattr__(self, "wells", wells)
        object.__setattr__(self, "dimension", wells.size)

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        return coords * (coords * coords - self.wells[:, None] ** 2)

    def laplacian(self, coords: np.ndarray) -> np.ndarray:
        values = 3.0 * coords * coords - self.wells[:, None] ** 2
        return np.sum(values, axis=0)


@dataclass(frozen=True)
class RastriginPotential:
    amplitude: float
    gamma: np.ndarray

    def __post_init__(self) -> None:
        gamma = np.asarray(self.gamma, dtype=float)
        if gamma.ndim != 1:
            raise ValueError("gamma must be a one-dimensional array")
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "dimension", gamma.size)

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        return 2.0 * coords + self.amplitude * self.gamma[:, None] * np.sin(
            self.gamma[:, None] * coords
        )

    def laplacian(self, coords: np.ndarray) -> np.ndarray:
        values = 2.0 + self.amplitude * self.gamma[:, None] ** 2 * np.cos(
            self.gamma[:, None] * coords
        )
        return np.sum(values, axis=0)


@dataclass(frozen=True)
class Smoluchowski(LinearFokkerPlanck):
    """Smoluchowski equation rho_t = beta^-1 Delta rho + div(rho grad V)."""

    potential: Potential
    beta: float = 1.0

    def __init__(self, potential: Potential, beta: float = 1.0):
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        dimension = potential.dimension
        diffusion = (1.0 / beta) * np.eye(dimension)
        object.__setattr__(self, "potential", potential)
        object.__setattr__(self, "beta", beta)
        super().__init__(diffusion=diffusion, dimension=dimension)

    def drift(self, coords: np.ndarray) -> np.ndarray:
        return -self.potential.gradient(coords)

    def divergence_drift(self, coords: np.ndarray) -> np.ndarray:
        return -self.potential.laplacian(coords)


@dataclass(frozen=True)
class HeatEq:
    """Model for ``p_t = a Δp + b(x) · ∇p + c(x) p``."""
    """Model for u_t = a \Delta u """

    dimension: int
    diffusion: float

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.diffusion < 0.0:
            raise ValueError("diffusion must be nonnegative")

    def build_operator(
        self,
        grid: TensorGrid,
        *,
        bc: BoundaryCondition = "dirichlet",
    ) -> sparse.csr_matrix:
        if grid.dimension != self.dimension:
            raise ValueError("grid dimension does not match model dimension")

        d1, d2 = tensor_derivative_matrices(grid.shape, grid.spacing, bc)
        operator = sparse.csr_matrix((grid.size, grid.size), dtype=float)
        if self.diffusion:
            for second in d2:
                operator = operator + self.diffusion * second

        if bc == "dirichlet":
            operator = operator.tolil()
            operator[grid.boundary_mask(), :] = 0.0
            operator = operator.tocsr()

        return operator

    def build_linear_operator(
        self,
        grid: TensorGrid,
        *,
        bc: BoundaryCondition = "dirichlet",
    ) -> LinearOperator:
        if grid.dimension != self.dimension:
            raise ValueError("grid dimension does not match model dimension")

        d1_1d, d2_1d = tensor_derivative_1d_matrices(grid.shape, grid.spacing, bc)
        boundary_mask = grid.boundary_mask() if bc == "dirichlet" else None

        def matvec(x: np.ndarray) -> np.ndarray:
            vector = np.asarray(x, dtype=float).reshape(-1)
            result = np.zeros_like(vector)

            if self.diffusion:
                for axis, second in enumerate(d2_1d):
                    result += self.diffusion * apply_axis_operator(vector, second, grid.shape, axis)

            if boundary_mask is not None:
                result[boundary_mask] = 0.0
            return result

        return LinearOperator((grid.size, grid.size), matvec=matvec, dtype=float)

    def operator_diagonal(
        self,
        grid: TensorGrid,
        *,
        bc: BoundaryCondition = "dirichlet",
    ) -> np.ndarray:
        if grid.dimension != self.dimension:
            raise ValueError("grid dimension does not match model dimension")

        _, d2_1d = tensor_derivative_1d_matrices(grid.shape, grid.spacing, bc)
        diagonal = np.zeros(grid.size, dtype=float)

        if self.diffusion:
            for axis in range(self.dimension):
                diagonal += self.diffusion * lift_axis_diagonal(d2_1d[axis].diagonal(), grid.shape, axis)

        if bc == "dirichlet":
            diagonal[grid.boundary_mask()] = 0.0

        return diagonal