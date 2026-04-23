"""Multi-index utilities for the classical combination technique."""

from __future__ import annotations

from math import comb
from typing import Iterator


def _compositions(total: int, length: int, minimum: int) -> Iterator[tuple[int, ...]]:
    if length == 1:
        if total >= minimum:
            yield (total,)
        return

    remaining_min = minimum * (length - 1)
    for value in range(minimum, total - remaining_min + 1):
        for tail in _compositions(total - value, length - 1, minimum):
            yield (value, *tail)


def combination_indices(
    level: int,
    dimension: int,
    *,
    min_level: int = 1,
) -> list[tuple[int, ...]]:
    """Return the classical combination-technique component-grid levels.

    The active set is

        {ell in N^d : level <= |ell|_1 <= level + dimension - 1}.

    ``min_level`` controls the smallest admissible one-dimensional level.
    The default ``1`` matches the usual sparse-grid convention.
    """

    if dimension < 1:
        raise ValueError("dimension must be positive")
    if level < dimension * min_level:
        raise ValueError(
            "level is too small for the requested dimension and min_level: "
            f"got level={level}, dimension={dimension}, min_level={min_level}"
        )

    indices: list[tuple[int, ...]] = []
    for total in range(level, level + dimension):
        indices.extend(_compositions(total, dimension, min_level))
    return indices


def combination_weight(level: int, dimension: int, ell: tuple[int, ...]) -> int:
    """Return the signed coefficient for one component grid."""

    total = sum(ell)
    if len(ell) != dimension:
        raise ValueError("ell length does not match dimension")
    if total < level or total > level + dimension - 1:
        return 0
    return (-1) ** (level + dimension - total - 1) * comb(dimension - 1, total - level)

