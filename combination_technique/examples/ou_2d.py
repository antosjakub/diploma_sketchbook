"""Small OU combination-technique run.

Run from the repository root with:

    python examples/ou_2d.py
"""

from __future__ import annotations

import numpy as np

from combination_technique import OrnsteinUhlenbeck, TensorGrid, TimeStepper, solve_combination, result_to_sgpp
from combination_technique.diagnostics import covariance, mass
from combination_technique.initial import cauchy_density, gaussian_density, product_laplace_density


def main() -> None:
    #sigma = np.array([[1.0, 0.25], [0.25, 1.15]])
    sigma = np.array([[
                2.259658098220825,
                -0.48025619983673096
            ],
            [
                -0.48025619983673096,
                1.540342092514038
    ]])
    model = OrnsteinUhlenbeck(sigma)
    stepper = TimeStepper(dt=0.01, theta=1.0)
    lvl = 12
    L = 10.0

    result = solve_combination(
        model,
        level=lvl,
        #initial_condition=gaussian_density,
        initial_condition=cauchy_density,
        #initial_condition=product_laplace_density,
        final_time=1.0,
        #final_time=0.01,
        stepper=stepper,
        domain_radius=L,
        bc="dirichlet",
        max_workers=8,
    )

    target = TensorGrid.from_level((4, 4), domain_radius=L)
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

    sparse.save_slice_plot("ou_2d.png", axes=(0, 1), fixed={2: 0.0}, resolution=120)
    print("saved plot: ou_2d.png")


if __name__ == "__main__":
    main()

