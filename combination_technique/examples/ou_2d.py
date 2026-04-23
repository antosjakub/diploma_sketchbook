"""Small OU combination-technique run.

Run from the repository root with:

    python examples/ou_2d.py
"""

from __future__ import annotations

import numpy as np

from combination_technique import OrnsteinUhlenbeck, TensorGrid, TimeStepper, solve_combination
from combination_technique.diagnostics import covariance, mass
from combination_technique.initial import gaussian_density


def main() -> None:
    sigma = np.array([[1.0, 0.25], [0.25, 1.15]])
    model = OrnsteinUhlenbeck(sigma)
    stepper = TimeStepper(dt=0.02, theta=1.0)

    result = solve_combination(
        model,
        level=3,
        initial_condition=gaussian_density,
        final_time=0.1,
        stepper=stepper,
        domain_radius=4.0,
        bc="dirichlet",
        max_workers=1,
    )

    target = TensorGrid.from_level((4, 4), domain_radius=4.0)
    combined = result.combine_on_grid(target)
    print(f"components: {len(result.components)}")
    print(f"mass on target grid: {mass(target, combined):.8f}")
    print("covariance on target grid:")
    print(covariance(target, combined))


if __name__ == "__main__":
    main()

