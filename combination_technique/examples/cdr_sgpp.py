"""General convection-diffusion-reaction example with SG++ interpolation.

Run from the repository root with:

    python examples/cdr_sgpp.py
"""

from __future__ import annotations

import os

import numpy as np

from combination_technique import (
    ConvectionDiffusionReaction,
    TimeStepper,
    gaussian_density,
    result_to_sgpp,
    solve_combination,
)


def drift(coords: np.ndarray) -> np.ndarray:
    return -0.35 * coords


def reaction(coords: np.ndarray) -> np.ndarray:
    radius2 = np.sum(coords * coords, axis=0)
    return 0.15 - 0.05 * radius2


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    model = ConvectionDiffusionReaction(
        dimension=3,
        diffusion=0.2,
        drift_fn=drift,
        reaction_fn=reaction,
    )
    result = solve_combination(
        model,
        level=4,
        initial_condition=gaussian_density,
        final_time=0.05,
        stepper=TimeStepper(dt=0.01, theta=1.0),
        domain_radius=4.0,
        bc="dirichlet",
        max_workers=1,
    )

    sparse = result_to_sgpp(result, level=4, boundary=True)
    values = sparse.evaluate(np.array([[0.0, 0.0, 0.0], [1.0, -0.5, 0.25]]))

    print(f"component grids: {len(result.components)}")
    print(f"sgpp points: {sparse.size}")
    print(f"mass from SG++ quadrature: {sparse.integral():.8f}")
    print("mean from SG++ quadrature:")
    print(sparse.mean())
    print("sparse-grid evaluations:")
    print(values)

    sparse.save_slice_plot("cdr_sgpp_slice.png", axes=(0, 1), fixed={2: 0.0}, resolution=120)
    print("saved plot: cdr_sgpp_slice.png")


if __name__ == "__main__":
    main()
