"""Finite-difference matrices for tensor grids."""

from __future__ import annotations

import numpy as np
from scipy import sparse


BoundaryCondition = str


def derivative_1d(n: int, h: float, order: int, bc: BoundaryCondition) -> sparse.csr_matrix:
    """Build a first- or second-derivative matrix on an endpoint grid."""

    if n < 3:
        raise ValueError("need at least three grid points")
    if order not in {1, 2}:
        raise ValueError("order must be 1 or 2")
    if bc not in {"dirichlet", "neumann"}:
        raise ValueError("bc must be 'dirichlet' or 'neumann'")

    mat = sparse.lil_matrix((n, n), dtype=float)

    if order == 1:
        scale = 1.0 / (2.0 * h)
        for i in range(1, n - 1):
            mat[i, i - 1] = -scale
            mat[i, i + 1] = scale
        if bc == "neumann":
            # Homogeneous Neumann closure: the normal derivative is zero.
            mat[0, :] = 0.0
            mat[n - 1, :] = 0.0
    else:
        scale = 1.0 / (h * h)
        for i in range(1, n - 1):
            mat[i, i - 1] = scale
            mat[i, i] = -2.0 * scale
            mat[i, i + 1] = scale
        if bc == "neumann":
            mat[0, 0] = -2.0 * scale
            mat[0, 1] = 2.0 * scale
            mat[n - 1, n - 2] = 2.0 * scale
            mat[n - 1, n - 1] = -2.0 * scale

    return mat.tocsr()


def tensor_derivative_1d_matrices(
    shape: tuple[int, ...],
    spacing: tuple[float, ...],
    bc: BoundaryCondition,
) -> tuple[list[sparse.csr_matrix], list[sparse.csr_matrix]]:
    d1_1d: list[sparse.csr_matrix] = []
    d2_1d: list[sparse.csr_matrix] = []
    for n, h in zip(shape, spacing, strict=True):
        d1_1d.append(derivative_1d(n, h, 1, bc))
        d2_1d.append(derivative_1d(n, h, 2, bc))
    return d1_1d, d2_1d


def kron_axis(operator: sparse.spmatrix, shape: tuple[int, ...], axis: int) -> sparse.csr_matrix:
    """Lift a 1D operator to a tensor-product grid."""

    factors: list[sparse.spmatrix] = []
    for j, n in enumerate(shape):
        factors.append(operator if j == axis else sparse.eye(n, format="csr"))

    result = factors[0]
    for factor in factors[1:]:
        result = sparse.kron(result, factor, format="csr")
    return result.tocsr()


def apply_axis_operator(
    values: np.ndarray,
    operator: sparse.csr_matrix,
    shape: tuple[int, ...],
    axis: int,
) -> np.ndarray:
    """Apply a 1D operator along one axis of a tensor-grid vector."""

    tensor = np.asarray(values, dtype=float).reshape(shape)
    moved = np.moveaxis(tensor, axis, 0)
    flattened = moved.reshape(shape[axis], -1)
    applied = operator @ flattened
    restored = np.moveaxis(applied.reshape(moved.shape), 0, axis)
    return np.asarray(restored, dtype=float).reshape(-1)


def tensor_derivative_matrices(
    shape: tuple[int, ...],
    spacing: tuple[float, ...],
    bc: BoundaryCondition,
) -> tuple[list[sparse.csr_matrix], list[sparse.csr_matrix]]:
    d1: list[sparse.csr_matrix] = []
    d2: list[sparse.csr_matrix] = []
    d1_1d, d2_1d = tensor_derivative_1d_matrices(shape, spacing, bc)
    for axis, (d1_axis, d2_axis) in enumerate(zip(d1_1d, d2_1d, strict=True)):
        d1.append(kron_axis(d1_axis, shape, axis))
        d2.append(kron_axis(d2_axis, shape, axis))
    return d1, d2
