"""
run from top most folder root
"""


from __future__ import annotations

import numpy as np

from combination_technique import Smoluchowski, DoubleWellPotential, TensorGrid, TimeStepper, solve_combination, result_to_sgpp
from combination_technique.diagnostics import covariance, mass
from combination_technique.initial import gaussian_density


def main() -> None:
    d = 2
    a = np.ones(d, dtype=np.float64)
    pot = DoubleWellPotential(a)
    model = Smoluchowski(pot, 1.0)
    stepper = TimeStepper(dt=0.0001, theta=1.0)
    lvl = 12
    T = 0.8
    L = (-3.0, 3.0)
    n_workers = 10

    result = solve_combination(
        model,
        level=lvl,
        initial_condition=gaussian_density,
        final_time=T,
        stepper=stepper,
        domain_radius=L,
        bc="dirichlet",
        max_workers=n_workers,
    )

    target = TensorGrid.from_level(d*(lvl,), domain_radius=L)
    combined = result.combine_on_grid(target)
    print(f"components: {len(result.components)}")
    print(f"mass on target grid: {mass(target, combined):.8f}")
    print("covariance on target grid:")
    print(covariance(target, combined))


    # plotting
    sparse = result_to_sgpp(result, level=lvl, boundary=True)
    values = sparse.evaluate(np.array([[0.0, 0.0, 0.0], [1.0, -0.5, 0.25]]))

    print(f"component grids: {len(result.components)}")
    print(f"sgpp points: {sparse.size}")
    print(f"mass from SG++ quadrature: {sparse.integral():.8f}")
    print("mean from SG++ quadrature:")
    print(sparse.mean())
    print("sparse-grid evaluations:")
    print(values)

    sparse.save_slice_plot("smol_DW.png", axes=(0, 1), fixed={2: 0.0}, resolution=120)
    print("saved plot: smol_DW.png")


if __name__ == "__main__":
    main()


