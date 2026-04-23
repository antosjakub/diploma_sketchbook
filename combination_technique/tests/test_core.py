from __future__ import annotations

import unittest

import numpy as np

from combination_technique import (
    ConvectionDiffusionReaction,
    OrnsteinUhlenbeck,
    QuadraticPotential,
    SGppInterpolant,
    Smoluchowski,
    TensorGrid,
    TimeStepper,
    combination_indices,
    combination_weight,
    gaussian_density,
    result_to_sgpp,
    sgpp_available,
    solve_combination,
    solve_on_grid,
)


class CombinationIndexTests(unittest.TestCase):
    def test_two_dimensional_indices_and_weights(self) -> None:
        indices = combination_indices(3, 2)
        self.assertEqual(indices, [(1, 2), (2, 1), (1, 3), (2, 2), (3, 1)])
        weights = [combination_weight(3, 2, ell) for ell in indices]
        self.assertEqual(weights, [-1, -1, 1, 1, 1])

    def test_rejects_impossible_level(self) -> None:
        with self.assertRaises(ValueError):
            combination_indices(2, 3)


class GridAndSolverTests(unittest.TestCase):
    def test_grid_shape_and_spacing(self) -> None:
        grid = TensorGrid.from_level((2, 3), domain_radius=2.0)
        self.assertEqual(grid.shape, (5, 9))
        self.assertEqual(grid.size, 45)
        self.assertAlmostEqual(grid.spacing[0], 1.0)
        self.assertAlmostEqual(grid.spacing[1], 0.5)

    def test_zero_time_solve_samples_initial_condition(self) -> None:
        model = OrnsteinUhlenbeck(np.eye(2))
        grid = TensorGrid.from_level((2, 2), domain_radius=3.0)
        values = solve_on_grid(
            model,
            grid,
            gaussian_density,
            final_time=0.0,
            stepper=TimeStepper(dt=0.1),
            bc="neumann",
        )
        self.assertEqual(values.shape, (grid.size,))
        self.assertGreater(values.max(), 0.0)

    def test_smolu_operator_shape(self) -> None:
        potential = QuadraticPotential(np.eye(2))
        model = Smoluchowski(potential, beta=1.0)
        grid = TensorGrid.from_level((2, 2), domain_radius=2.0)
        operator = model.build_operator(grid, bc="dirichlet")
        self.assertEqual(operator.shape, (grid.size, grid.size))

    def test_general_cdr_operator_shape(self) -> None:
        model = ConvectionDiffusionReaction(
            dimension=2,
            diffusion=0.3,
            drift_fn=lambda coords: -coords,
            reaction_fn=lambda coords: np.sum(coords * 0.0, axis=0) + 0.2,
        )
        grid = TensorGrid.from_level((2, 2), domain_radius=2.0)
        operator = model.build_operator(grid, bc="dirichlet")
        self.assertEqual(operator.shape, (grid.size, grid.size))

    def test_combination_result_evaluates_points(self) -> None:
        model = OrnsteinUhlenbeck(np.eye(2))
        result = solve_combination(
            model,
            level=3,
            initial_condition=gaussian_density,
            final_time=0.0,
            stepper=TimeStepper(dt=0.1),
            domain_radius=3.0,
            bc="neumann",
            max_workers=1,
        )
        values = result.evaluate(np.array([[0.0, 0.0], [1.0, 0.0]]))
        self.assertEqual(values.shape, (2,))
        self.assertTrue(np.all(np.isfinite(values)))

    @unittest.skipUnless(sgpp_available(), "SG++ bindings are not available")
    def test_sgpp_interpolant_from_function(self) -> None:
        interpolant = SGppInterpolant.from_function(
            lambda coords: np.sum(coords, axis=0),
            dimension=2,
            level=3,
            domain_radius=1.0,
        )
        points = np.array([[0.0, 0.0], [0.5, -0.5]])
        values = interpolant.evaluate(points)
        self.assertEqual(values.shape, (2,))
        self.assertTrue(np.all(np.isfinite(values)))
        nodal = interpolant.nodal_values()
        self.assertEqual(nodal.shape, (interpolant.size,))
        self.assertTrue(np.all(np.isfinite(nodal)))

    @unittest.skipUnless(sgpp_available(), "SG++ bindings are not available")
    def test_sgpp_quadrature_and_moments(self) -> None:
        interpolant = SGppInterpolant.from_function(
            lambda coords: np.ones(coords.shape[1]),
            dimension=2,
            level=4,
            bounds=((-1.0, 1.0), (-2.0, 2.0)),
            boundary=True,
        )
        self.assertAlmostEqual(interpolant.integral(), 8.0, places=10)
        np.testing.assert_allclose(interpolant.mean(), np.zeros(2), atol=1e-10)
        covariance = interpolant.covariance()
        self.assertEqual(covariance.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(covariance)))

    @unittest.skipUnless(sgpp_available(), "SG++ bindings are not available")
    def test_combination_result_projects_to_sgpp(self) -> None:
        model = OrnsteinUhlenbeck(np.eye(2))
        result = solve_combination(
            model,
            level=3,
            initial_condition=gaussian_density,
            final_time=0.0,
            stepper=TimeStepper(dt=0.1),
            domain_radius=3.0,
            bc="neumann",
            max_workers=1,
        )
        sparse = result_to_sgpp(result, level=3)
        values = sparse.evaluate(np.array([[0.0, 0.0], [1.0, 0.0]]))
        self.assertEqual(values.shape, (2,))
        self.assertTrue(np.all(np.isfinite(values)))


if __name__ == "__main__":
    unittest.main()
