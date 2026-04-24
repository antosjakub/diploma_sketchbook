from __future__ import annotations

import csv
import tempfile
import unittest

import numpy as np

from combination_technique import (
    BenchmarkCase,
    ConvectionDiffusionReaction,
    LinearSolveConfig,
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
    run_backend_benchmark,
    sgpp_available,
    solve_combination,
    solve_on_grid,
    solve_on_grid_with_stats,
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
        matrix_free = model.build_linear_operator(grid, bc="dirichlet")
        self.assertEqual(matrix_free.shape, (grid.size, grid.size))

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

    def test_matrix_and_linear_operator_agree_on_matvec(self) -> None:
        model = OrnsteinUhlenbeck(np.array([[1.0, 0.2], [0.2, 0.8]]))
        grid = TensorGrid.from_level((2, 2), domain_radius=2.0)
        matrix = model.build_operator(grid, bc="dirichlet")
        operator = model.build_linear_operator(grid, bc="dirichlet")
        vector = np.linspace(0.0, 1.0, grid.size)
        np.testing.assert_allclose(matrix @ vector, operator @ vector, atol=1e-12, rtol=1e-12)

    def test_operator_diagonal_matches_assembled_matrix(self) -> None:
        model = OrnsteinUhlenbeck(np.array([[1.0, 0.2], [0.2, 0.8]]))
        grid = TensorGrid.from_level((2, 2), domain_radius=2.0)
        matrix = model.build_operator(grid, bc="dirichlet")
        diagonal = model.operator_diagonal(grid, bc="dirichlet")
        np.testing.assert_allclose(matrix.diagonal(), diagonal, atol=1e-12, rtol=1e-12)

    def test_matrix_and_linear_operator_solvers_agree(self) -> None:
        model = OrnsteinUhlenbeck(np.eye(2))
        grid = TensorGrid.from_level((2, 2), domain_radius=3.0)
        stepper = TimeStepper(dt=0.05, theta=1.0)
        matrix_solution = solve_on_grid(
            model,
            grid,
            gaussian_density,
            final_time=0.1,
            stepper=stepper,
            bc="neumann",
            operator_backend="matrix",
        )
        operator_solution = solve_on_grid(
            model,
            grid,
            gaussian_density,
            final_time=0.1,
            stepper=stepper,
            bc="neumann",
            operator_backend="linear_operator",
            linear_solve=LinearSolveConfig(rtol=1e-10, atol=1e-12, maxiter=200),
        )
        np.testing.assert_allclose(
            matrix_solution,
            operator_solution,
            atol=1e-8,
            rtol=1e-8,
        )

    def test_linear_operator_solver_with_jacobi_preconditioner(self) -> None:
        model = OrnsteinUhlenbeck(np.eye(2))
        grid = TensorGrid.from_level((2, 2), domain_radius=3.0)
        stepper = TimeStepper(dt=0.05, theta=1.0)
        solution = solve_on_grid(
            model,
            grid,
            gaussian_density,
            final_time=0.1,
            stepper=stepper,
            bc="neumann",
            operator_backend="linear_operator",
            linear_solve=LinearSolveConfig(
                method="gmres",
                preconditioner="jacobi",
                rtol=1e-10,
                atol=1e-12,
                maxiter=200,
            ),
        )
        self.assertEqual(solution.shape, (grid.size,))
        self.assertTrue(np.all(np.isfinite(solution)))

    def test_solve_on_grid_with_stats_reports_iterations_and_timing(self) -> None:
        model = OrnsteinUhlenbeck(np.eye(2))
        grid = TensorGrid.from_level((2, 2), domain_radius=3.0)
        result = solve_on_grid_with_stats(
            model,
            grid,
            gaussian_density,
            final_time=0.1,
            stepper=TimeStepper(dt=0.05, theta=1.0),
            bc="neumann",
            operator_backend="linear_operator",
            linear_solve=LinearSolveConfig(
                method="gmres",
                preconditioner="jacobi",
                rtol=1e-10,
                atol=1e-12,
                maxiter=200,
            ),
        )
        self.assertEqual(result.values.shape, (grid.size,))
        self.assertEqual(result.stats.grid_size, grid.size)
        self.assertEqual(result.stats.steps, 2)
        self.assertGreaterEqual(result.stats.total_seconds, 0.0)
        self.assertGreaterEqual(result.stats.operator_setup_seconds, 0.0)
        self.assertGreaterEqual(result.stats.solve_seconds, 0.0)
        self.assertGreater(result.stats.krylov_iterations, 0)
        self.assertEqual(len(result.stats.krylov_iterations_per_step), result.stats.steps)

    def test_matrix_solver_with_ilu_preconditioner(self) -> None:
        model = OrnsteinUhlenbeck(np.eye(2))
        grid = TensorGrid.from_level((2, 2), domain_radius=3.0)
        stepper = TimeStepper(dt=0.05, theta=1.0)
        solution = solve_on_grid(
            model,
            grid,
            gaussian_density,
            final_time=0.1,
            stepper=stepper,
            bc="neumann",
            operator_backend="matrix",
            linear_solve=LinearSolveConfig(
                method="gmres",
                preconditioner="ilu",
                ilu_drop_tol=1e-4,
                ilu_fill_factor=10.0,
                rtol=1e-10,
                atol=1e-12,
                maxiter=200,
            ),
        )
        self.assertEqual(solution.shape, (grid.size,))
        self.assertTrue(np.all(np.isfinite(solution)))

    def test_rejects_incompatible_preconditioner_backend_pair(self) -> None:
        model = OrnsteinUhlenbeck(np.eye(2))
        grid = TensorGrid.from_level((2, 2), domain_radius=3.0)
        stepper = TimeStepper(dt=0.05, theta=1.0)

        with self.assertRaises(ValueError):
            solve_on_grid(
                model,
                grid,
                gaussian_density,
                final_time=0.1,
                stepper=stepper,
                bc="neumann",
                operator_backend="matrix",
                linear_solve=LinearSolveConfig(preconditioner="jacobi"),
            )

        with self.assertRaises(ValueError):
            solve_on_grid(
                model,
                grid,
                gaussian_density,
                final_time=0.1,
                stepper=stepper,
                bc="neumann",
                operator_backend="linear_operator",
                linear_solve=LinearSolveConfig(preconditioner="ilu"),
            )

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
        self.assertEqual(len(result.component_stats()), len(result.components))
        self.assertGreaterEqual(result.total_component_time, 0.0)
        self.assertGreaterEqual(result.total_krylov_iterations, 0)

    def test_combination_accepts_linear_operator_backend(self) -> None:
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
            operator_backend="linear_operator",
        )
        values = result.evaluate(np.array([[0.0, 0.0], [1.0, 0.0]]))
        self.assertEqual(values.shape, (2,))
        self.assertTrue(np.all(np.isfinite(values)))

    def test_run_backend_benchmark_writes_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = f"{tmpdir}/benchmark.csv"
            rows = run_backend_benchmark(
                output_path=output,
                dimension=2,
                level=3,
                final_time=0.02,
                dt=0.01,
                domain_radius=2.0,
                rho=0.1,
                max_workers=1,
                repeats=1,
                cases=(
                    BenchmarkCase(
                        name="matrix_direct",
                        operator_backend="matrix",
                        linear_solve=LinearSolveConfig(preconditioner="none"),
                    ),
                    BenchmarkCase(
                        name="linear_operator_jacobi",
                        operator_backend="linear_operator",
                        linear_solve=LinearSolveConfig(
                            method="gmres",
                            preconditioner="jacobi",
                            maxiter=100,
                        ),
                    ),
                ),
            )
            self.assertGreater(len(rows), 0)
            with open(output, newline="", encoding="utf-8") as handle:
                written_rows = list(csv.DictReader(handle))
            self.assertEqual(len(written_rows), len(rows))
            self.assertEqual(sum(row["row_type"] == "summary" for row in written_rows), 2)

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
