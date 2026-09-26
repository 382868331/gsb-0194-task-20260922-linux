"""Tests for distortion.resample (forward area-weighted splatting).

References are independent of the adaptive path where it matters:
identity/affine cases are checked against exact polygon clipping, and the
distorted case is checked against a much finer independent run of the
same public entry point plus a hand-computed analytic expectation for the
identity case.
"""

import unittest

from distortion import (DistortionModel, forward_resample,
                        exact_quad_resample, ResampleResult)


def grid_total(grid):
    return sum(sum(row) for row in grid)


class IdentityExactTests(unittest.TestCase):
    """Identity model: forward_resample must match exact clipping."""

    def setUp(self):
        self.model = DistortionModel(0.0, 0.0, 0.0, 0.0)
        self.image = [[1.0, 2.0], [3.0, 4.0]]
        self.window = (0.0, 0.0, 2.0, 2.0)
        self.shape = (2, 2)

    def test_identity_matches_exact_clip(self):
        res = forward_resample(self.model, self.image, self.window,
                               self.shape, area_tol=1e-12)
        quads = [[(c, r), (c + 1, r), (c + 1, r + 1), (c, r + 1)]
                 for r in range(2) for c in range(2)]
        values = [self.image[r][c] for r in range(2) for c in range(2)]
        ref = exact_quad_resample(quads, values, self.window, self.shape)
        for row_e, row_r in zip(res.energy, ref.energy):
            for e, x in zip(row_e, row_r):
                self.assertAlmostEqual(e, x, places=9)
        self.assertTrue(res.converged)
        self.assertEqual(res.lost_energy, 0.0)

    def test_identity_constant_image_analytic(self):
        # Constant image v=5 over [0,2]^2 onto a 2x2 target grid: each
        # unit target cell receives exactly area 1, energy 5.
        res = forward_resample(self.model, [[5.0] * 2] * 2,
                               self.window, self.shape)
        for row in res.energy:
            for e in row:
                self.assertAlmostEqual(e, 5.0, places=9)
        for row in res.area:
            for a in row:
                self.assertAlmostEqual(a, 1.0, places=9)
        self.assertAlmostEqual(res.input_energy, 20.0)
        self.assertAlmostEqual(res.accounted_energy, 20.0)

    def test_affine_reference_shear(self):
        # Shear map (x, y) -> (x + 0.5*y, y) applied to one unit pixel is
        # a parallelogram of area exactly 1; clipping distributes it
        # exactly across the straddled target cells.
        quad = [(0.0, 0.0), (1.0, 0.0), (1.5, 1.0), (0.5, 1.0)]
        res = exact_quad_resample([quad], [7.0], (0.0, 0.0, 2.0, 1.0),
                                  (1, 2))
        self.assertAlmostEqual(grid_total(res.area), 1.0, places=12)
        self.assertAlmostEqual(grid_total(res.energy), 7.0, places=12)
        # Left cell [0,1]x[0,1] holds area 0.75, right cell 0.25.
        self.assertAlmostEqual(res.area[0][0], 0.75, places=12)
        self.assertAlmostEqual(res.area[0][1], 0.25, places=12)
        self.assertEqual(res.lost_energy, 0.0)


class DistortedConvergenceTests(unittest.TestCase):
    """Distorted model: coarse run must converge to a much finer run."""

    def test_barrel_converges_to_fine_reference(self):
        model = DistortionModel(k1=-0.2, k2=0.03, p1=0.0, p2=0.0)
        image = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        origin = (-1.5, -1.5)
        window = (-2.0, -2.0, 4.0, 4.0)
        shape = (8, 8)
        coarse = forward_resample(model, image, window, shape,
                                  origin=origin, area_tol=1e-5)
        fine = forward_resample(model, image, window, shape,
                                origin=origin, area_tol=1e-9, max_depth=14)
        self.assertTrue(coarse.converged)
        self.assertTrue(fine.converged)
        self.assertGreater(fine.leaves_used, coarse.leaves_used)
        # Total deposited energy converges.
        self.assertAlmostEqual(coarse.accounted_energy,
                               fine.accounted_energy, delta=0.05)
        # Per-cell energies converge too (loose bound: centroid splatting
        # of finite leaves has its own discretization floor).
        for rc, rf in zip(coarse.energy, fine.energy):
            for ec, ef in zip(rc, rf):
                self.assertAlmostEqual(ec, ef, delta=0.35)
        # Energy and area are separate grids; area is not a normalized
        # copy of energy for a non-constant image.
        self.assertNotAlmostEqual(grid_total(fine.energy),
                                  grid_total(fine.area), places=3)

    def test_energy_matches_independent_jacobian_integral(self):
        # Independent reference: the total mapped energy equals the
        # integral of value*|det J| over the source region, approximated
        # here by a dense midpoint rule with a finite-difference Jacobian
        # (no use of the resampler under test).
        model = DistortionModel(k1=-0.2, k2=0.03, p1=0.01, p2=-0.02)
        image = [[2.0, 3.0], [4.0, 5.0]]
        origin = (-1.0, -1.0)
        res = forward_resample(model, image, (-3.0, -3.0, 6.0, 6.0),
                               (12, 12), origin=origin,
                               area_tol=1e-8, max_depth=12)
        self.assertTrue(res.converged)
        self.assertEqual(res.lost_energy, 0.0)

        eps = 1e-6
        n = 60  # samples per pixel edge
        expected = 0.0
        for r in range(2):
            for c in range(2):
                value = image[r][c]
                for i in range(n):
                    for j in range(n):
                        x = origin[0] + c + (j + 0.5) / n
                        y = origin[1] + r + (i + 0.5) / n
                        fxp = model.distort(x + eps, y)
                        fxm = model.distort(x - eps, y)
                        fyp = model.distort(x, y + eps)
                        fym = model.distort(x, y - eps)
                        j00 = (fxp[0] - fxm[0]) / (2 * eps)
                        j01 = (fyp[0] - fym[0]) / (2 * eps)
                        j10 = (fxp[1] - fxm[1]) / (2 * eps)
                        j11 = (fyp[1] - fym[1]) / (2 * eps)
                        det = abs(j00 * j11 - j01 * j10)
                        expected += value * det / (n * n)
        self.assertAlmostEqual(res.accounted_energy, expected, delta=0.02)


class OutOfWindowTests(unittest.TestCase):
    def test_partial_coverage_counts_lost_energy(self):
        model = DistortionModel(0.0, 0.0, 0.0, 0.0)
        image = [[1.0, 1.0], [1.0, 1.0]]  # covers [0,2]x[0,2]
        # Window covers only [0,1]x[0,2]: half the area is lost.
        res = forward_resample(model, image, (0.0, 0.0, 1.0, 2.0), (2, 1))
        self.assertAlmostEqual(res.lost_energy, 2.0, places=9)
        self.assertAlmostEqual(grid_total(res.energy), 2.0, places=9)
        self.assertAlmostEqual(res.accounted_energy, res.input_energy,
                               places=9)

    def test_fully_outside_window(self):
        model = DistortionModel(0.0, 0.0, 0.0, 0.0)
        res = forward_resample(model, [[3.0]], (10.0, 10.0, 1.0, 1.0),
                               (1, 1))
        self.assertAlmostEqual(res.lost_energy, 3.0, places=12)
        self.assertEqual(grid_total(res.energy), 0.0)


class BudgetTests(unittest.TestCase):
    def test_budget_exhaustion_flags_not_converged(self):
        model = DistortionModel(k1=-0.35, k2=0.1, p1=0.05, p2=0.05)
        image = [[1.0, 2.0], [3.0, 4.0]]
        res = forward_resample(model, image, (-3.0, -3.0, 6.0, 6.0),
                               (8, 8), origin=(-1.0, -1.0),
                               area_tol=1e-12, max_depth=30, max_leaves=8)
        self.assertFalse(res.converged)
        self.assertLessEqual(res.leaves_used, 8 + 4 * 31)
        # Still returns usable grids and an error indicator; it does not
        # claim the tolerance was met.
        self.assertGreaterEqual(res.error_estimate, 0.0)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.model = DistortionModel(0.0, 0.0, 0.0, 0.0)
        self.image = [[1.0]]

    def test_near_zero_target_cell_rejected(self):
        with self.assertRaises(ValueError):
            forward_resample(self.model, self.image,
                             (0.0, 0.0, 1e-7, 1e-7), (1, 1))

    def test_nonpositive_window_rejected(self):
        for window in ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, -1.0)):
            with self.assertRaises(ValueError):
                forward_resample(self.model, self.image, window, (1, 1))

    def test_nan_and_inf_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                forward_resample(self.model, [[bad]], (0.0, 0.0, 1.0, 1.0),
                                 (1, 1))
        with self.assertRaises(ValueError):
            forward_resample(self.model, self.image,
                             (0.0, 0.0, float("nan"), 1.0), (1, 1))

    def test_bool_and_type_errors(self):
        with self.assertRaises(TypeError):
            forward_resample(self.model, [[True]], (0.0, 0.0, 1.0, 1.0),
                             (1, 1))
        with self.assertRaises(TypeError):
            forward_resample(self.model, self.image, (0.0, 0.0, 1.0, 1.0),
                             (1, True))
        with self.assertRaises(TypeError):
            forward_resample("not a model", self.image,
                             (0.0, 0.0, 1.0, 1.0), (1, 1))
        with self.assertRaises(TypeError):
            forward_resample(self.model, self.image, (0.0, 0.0, 1.0, 1.0),
                             (1, 1), max_leaves=True)

    def test_ragged_and_oversize_image_rejected(self):
        with self.assertRaises(ValueError):
            forward_resample(self.model, [[1.0], [1.0, 2.0]],
                             (0.0, 0.0, 2.0, 2.0), (1, 1))
        with self.assertRaises(ValueError):
            forward_resample(self.model, [[1.0]] * 65,
                             (0.0, 0.0, 1.0, 1.0), (1, 1))

    def test_exact_quad_resample_validation(self):
        with self.assertRaises(ValueError):
            exact_quad_resample([[(0, 0), (1, 0), (1, 1), (0, 1)]],
                                [1.0, 2.0], (0.0, 0.0, 1.0, 1.0), (1, 1))
        with self.assertRaises(TypeError):
            exact_quad_resample([[(0, 0), (1, 0), (1, 1)]], [1.0],
                                (0.0, 0.0, 1.0, 1.0), (1, 1))

    def test_result_shape_and_types(self):
        res = forward_resample(self.model, self.image, (0.0, 0.0, 1.0, 1.0),
                               (1, 1))
        self.assertIsInstance(res, ResampleResult)
        self.assertEqual(len(res.energy), 1)
        self.assertEqual(len(res.energy[0]), 1)


if __name__ == "__main__":
    unittest.main()
