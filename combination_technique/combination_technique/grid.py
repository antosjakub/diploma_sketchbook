"""Endpoint-including anisotropic tensor grids."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class TensorGrid:
    """An anisotropic tensor grid on a box.

    Levels ``ell_j`` produce ``2**ell_j + 1`` endpoint-including nodes in
    direction ``j``.
    """

    levels: tuple[int, ...]
    bounds: tuple[tuple[float, float], ...]

    @classmethod
    def from_level(
        cls,
        levels: tuple[int, ...],
        *,
        domain_radius: tuple[float, ...] = (0.0, 1.0)
    ) -> "TensorGrid":
        #print("levels = ", levels)
        assert len(domain_radius) == 2
        int_min = float(domain_radius[0])
        int_max = float(domain_radius[1])
        bounds = tuple((int_min, int_max) for _ in levels)
        return cls(tuple(levels), bounds)

    @property
    def dimension(self) -> int:
        return len(self.levels)

    @cached_property
    def shape(self) -> tuple[int, ...]:
        return tuple(2**level + 1 for level in self.levels)

    @cached_property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @cached_property
    def axes(self) -> tuple[np.ndarray, ...]:
        return tuple(
            np.linspace(left, right, count)
            for (left, right), count in zip(self.bounds, self.shape, strict=True)
        )

    @cached_property
    def spacing(self) -> tuple[float, ...]:
        return tuple(
            (right - left) / (count - 1)
            for (left, right), count in zip(self.bounds, self.shape, strict=True)
        )

    def mesh(self, *, sparse: bool = False) -> tuple[np.ndarray, ...]:
        return np.meshgrid(*self.axes, indexing="ij", sparse=sparse)

    def flat_coordinates(self) -> np.ndarray:
        """Return coordinates as an array with shape ``(dimension, size)``."""

        meshes = self.mesh(sparse=False)
        return np.vstack([axis.reshape(1, -1) for axis in meshes])

    def values_from_callable(self, fn) -> np.ndarray:
        """Sample ``fn`` on this grid and return a flat vector."""

        coords = self.flat_coordinates()
        values = np.asarray(fn(coords), dtype=float)
        if values.shape == (self.size,):
            return values
        if values.shape == self.shape:
            return values.reshape(-1)
        if values.shape == (1, self.size):
            return values.reshape(-1)
        raise ValueError(
            "callable must return shape (size,), (1, size), or grid.shape; "
            f"got {values.shape}"
        )

    def boundary_mask(self) -> np.ndarray:
        mask = np.zeros(self.shape, dtype=bool)
        for axis in range(self.dimension):
            low = [slice(None)] * self.dimension
            high = [slice(None)] * self.dimension
            low[axis] = 0
            high[axis] = -1
            mask[tuple(low)] = True
            mask[tuple(high)] = True
        return mask.reshape(-1)

