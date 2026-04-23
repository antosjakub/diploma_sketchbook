"""Helpers to project combination-technique results onto SG++ sparse grids."""

from __future__ import annotations

from .sgpp import SGppInterpolant


def result_to_sgpp(
    result,
    *,
    level: int,
    bounds: tuple[tuple[float, float], ...] | None = None,
    boundary: bool = False,
) -> SGppInterpolant:
    if bounds is None:
        if not result.components:
            raise ValueError("cannot infer bounds from an empty combination result")
        bounds = result.components[0].grid.bounds

    return SGppInterpolant.from_function(
        lambda coords: result.evaluate(coords.T),
        dimension=result.dimension,
        level=level,
        bounds=bounds,
        boundary=boundary,
    )
