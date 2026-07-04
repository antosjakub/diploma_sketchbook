"""
run from top most folder root
"""


from __future__ import annotations

import numpy as np

from combination_technique import HeatEq, TensorGrid, TimeStepper, solve_combination, result_to_sgpp
from combination_technique.initial import heat_eq_IC


def main() -> None:
    d = 2
    alpha = 0.01
    model = HeatEq(d, alpha)
    stepper = TimeStepper(dt=0.0001, theta=1.0)
    lvl = 8
    T = 1.0
    L = (0.0, 1.0)
    n_workers = 10

    result = solve_combination(
        model,
        level=lvl,
        initial_condition=heat_eq_IC,
        final_time=T,
        stepper=stepper,
        domain_radius=L,
        bc="dirichlet",
        max_workers=n_workers,
    )

    print(f"components: {len(result.components)}")


    # plotting
    sparse = result_to_sgpp(result, level=lvl, boundary=True)
    #values = sparse.evaluate(np.array([[0.0, 0.0, 0.0], [1.0, -0.5, 0.25]]))

    print(f"component grids: {len(result.components)}")
    print(f"sgpp points: {sparse.size}")
    print(f"mass from SG++ quadrature: {sparse.integral():.8f}")
    print("mean from SG++ quadrature:")
    print(sparse.mean())
    print("sparse-grid evaluations:")
    #print(values)

    sparse.save_slice_plot("HeatEq.png", axes=(0, 1), fixed={2: 0.5}, resolution=120)
    print("saved plot: HeatEq.png")


if __name__ == "__main__":
    main()


