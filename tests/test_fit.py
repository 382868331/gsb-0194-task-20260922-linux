"""Tests for the weighted single-coefficient radial fit.

Independence rule: every expected value here is re-derived in this file
from the spec formulas -- pencil-and-paper numbers, an independent
closed-form weighted-least-squares transcription (k = b/A with b and A
computed directly from the raw inputs), and a deterministic local
linear-congruential noise stream.  The package under test is never used
to manufacture its own expected values; re-projecting through the
returned model is a supplementary consistency check only.

Run: python -m unittest discover -s tests -v
"""

import math
import unittest

from distortion import (DistortionModel, RadialFitResult, fit_radial_k1,
                        finite_real)


# ---------------------------------------------------------------------------
# Independent references (derived directly from the spec)
# ---------------------------------------------------------------------------

def ref_pixel_project(k1, f, cx, cy, x, y):
    """u = f*x*(1+k1 r^2)+cx etc., transcribed independently here."""
    r2 = x * x + y * y
    s = 1.0 + k1 * r2
    return f * x * s + cx, f * y * s + cy


def ref_weighted_k1(points, weights, f, cx, cy):
    """Closed-form weighted-LS optimum, coded straight from the algebra.

    A = sum w f^2 r^6 ; b = sum w f r^2 (x*t_x + y*t_y),
    t_x = u - cx - f x, t_y = v - cy - f y;  k* = b / A.
    """
    A = 0.0
    b = 0.0
    for (x, y, u, v), w in zip(points, weights):
        r2 = x * x + y * y
        tx = u - cx - f * x
        ty = v - cy - f * y
        A += w * f * f * r2 * r2 * r2
        b += w * f * r2 * (x * tx + y * ty)
    return b / A, A, b


def ref_weighted_cost(points, weights, k1, f, cx, cy):
    """C = sum w [(u-u_hat)^2 + (v-v_hat)^2], independent transcription."""
    total = 0.0
    wsum = 0.0
    for (x, y, u, v), w in zip(points, weights):
        uh, vh = ref_pixel_project(k1, f, cx, cy, x, y)
        total += w * ((u - uh) ** 2 + (v - vh) ** 2)
        wsum += w
    return total, wsum


def det_noise(seed=1234567):
    """Deterministic fixed-sequence pseudo-noise in roughly [-1, 1]."""
    state = seed
    while True:
        state = (1103515245 * state + 12345) & 0x7fffffff
        yield state / 0x7fffffff * 2.0 - 1.0


# ---------------------------------------------------------------------------
# Normal behavior
# ---------------------------------------------------------------------------

class TestFitNormal(unittest.TestCase):
    def test_hand_computed_single_point(self):
        # Pencil and paper: x=0.5, y=0, r2=0.25, f=100, no principal
        # offset, k1=-0.2 -> xp=0.5*(1-0.05)=0.475 -> u=47.5.
        # Independently: A = 100^2 * 0.5^2 * 0.25^2 = 156.25,
        # t_x = 47.5-50 = -2.5, b = 100*0.5*0.25*(-2.5) = -31.25,
        # k* = b/A = -0.2 exactly.
        res = fit_radial_k1([((0.5, 0.0), (47.5, 0.0))],
                            f=100.0, cx=0.0, cy=0.0)
        self.assertIsInstance(res, RadialFitResult)
        self.assertAlmostEqual(res.k1, -0.2, places=12)
        self.assertEqual(res.status, "converged")
        self.assertTrue(res.converged)
        self.assertLessEqual(res.iterations, res.max_iterations)
        self.assertGreaterEqual(res.iterations, 1)
        # The optimum is b/A in floating point; residual round-off at
        # f=100 is a few 1e-8 px (documented FP scale), not exact zero.
        self.assertLessEqual(res.cost, 1e-13)
        self.assertLessEqual(res.rms, 1e-7)
        # The returned model is the existing 4-coefficient structure
        # with only k1 nonzero.
        self.assertIsInstance(res.model, DistortionModel)
        self.assertEqual((res.model.k2, res.model.p1, res.model.p2),
                         (0.0, 0.0, 0.0))
        # Supplementary consistency: pixel prediction from model.distort.
        xn, yn = res.model.distort(0.5, 0.0)
        self.assertAlmostEqual(100.0 * xn + 0.0, 47.5, places=9)

    def test_no_distortion_recovers_zero(self):
        f, cx, cy = 500.0, 320.0, 240.0
        norm = [(-0.4, 0.3), (0.25, -0.1), (0.0, 0.5), (0.45, 0.45)]
        obs = [ref_pixel_project(0.0, f, cx, cy, x, y) for x, y in norm]
        corrs = [((x, y), (u, v)) for (x, y), (u, v) in zip(norm, obs)]
        res = fit_radial_k1(corrs, f, cx, cy)
        self.assertEqual(res.status, "converged")
        self.assertAlmostEqual(res.k1, 0.0, places=12)
        self.assertAlmostEqual(res.rms, 0.0, places=8)
        self.assertEqual(res.n_points, 4)
        self.assertAlmostEqual(res.total_weight, 4.0, places=12)

    def test_synthetic_coefficient_recovery_matches_closed_form(self):
        f, cx, cy = 700.0, 319.5, 239.5
        k_true = -0.28
        norm = [(-0.5, 0.2), (0.4, -0.35), (0.1, 0.6), (-0.3, -0.45),
                (0.55, 0.05), (0.0, 0.0)]
        obs = [ref_pixel_project(k_true, f, cx, cy, x, y)
               for x, y in norm]
        corrs = [((x, y), (u, v)) for (x, y), (u, v) in zip(norm, obs)]
        pts = [(x, y, u, v) for (x, y), (u, v) in zip(norm, obs)]
        k_exp, A_exp, _ = ref_weighted_k1(pts, [1.0] * len(pts),
                                          f, cx, cy)
        self.assertAlmostEqual(k_exp, k_true, places=10)  # sanity on ref

        res = fit_radial_k1(corrs, f, cx, cy)
        self.assertEqual(res.status, "converged")
        self.assertAlmostEqual(res.k1, k_exp, places=12)
        self.assertAlmostEqual(res.k1, k_true, delta=1e-10)
        self.assertAlmostEqual(res.observability, math.sqrt(A_exp),
                               places=6)
        # Exact synthetic data: pixel residual vanishes.
        self.assertLessEqual(res.rms, 1e-8)

    def test_linear_model_converges_in_one_step_from_any_start(self):
        # The model is linear in k1: one Gauss-Newton step reaches the
        # global minimum even when the budget is a single iteration.
        f, cx, cy = 400.0, 100.0, 100.0
        norm = [(-0.3, 0.4), (0.5, -0.2), (0.2, 0.6)]
        obs = [ref_pixel_project(0.15, f, cx, cy, x, y)
               for x, y in norm]
        corrs = [((x, y), (u, v)) for (x, y), (u, v) in zip(norm, obs)]
        res = fit_radial_k1(corrs, f, cx, cy, initial_k1=0.7,
                            max_iterations=1)
        self.assertEqual(res.status, "converged")
        self.assertEqual(res.iterations, 1)
        self.assertAlmostEqual(res.k1, 0.15, places=11)

    def test_weights_match_independent_weighted_closed_form(self):
        f, cx, cy = 600.0, 0.0, 0.0
        k_true = 0.1
        norm = [(-0.45, 0.3), (0.35, 0.4), (0.5, -0.25)]
        clean = [ref_pixel_project(k_true, f, cx, cy, x, y)
                 for x, y in norm]
        weights = [1000.0, 1.0, 0.001]
        # Corrupt the two low-weight observations by 30+ pixels; the
        # high-weight one stays exact.  Expected answer from the
        # independent weighted formula, not from the fitter.
        noisy = [(clean[0][0], clean[0][1]),
                 (clean[1][0] + 30.0, clean[1][1] - 20.0),
                 (clean[2][0] - 40.0, clean[2][1] + 25.0)]
        corrs = [((x, y), (u, v)) for (x, y), (u, v)
                 in zip(norm, noisy)]
        pts = [(x, y, u, v) for (x, y), (u, v) in zip(norm, noisy)]
        k_exp, _, _ = ref_weighted_k1(pts, weights, f, cx, cy)
        cost_exp, wsum_exp = ref_weighted_cost(pts, weights, k_exp,
                                               f, cx, cy)

        res = fit_radial_k1(corrs, f, cx, cy, weights=weights)
        self.assertEqual(res.status, "converged")
        # Must match the independently coded weighted optimum...
        self.assertAlmostEqual(res.k1, k_exp, places=11)
        # ...and sit close to the generating value: the weight-1 point
        # still carries 30 px of injected error, so the independently
        # computed 4.6e-5 shift from k_true is accepted rather than
        # guessed here.
        self.assertAlmostEqual(res.k1, k_true, delta=1e-4)
        self.assertAlmostEqual(res.cost, cost_exp, places=6)
        self.assertAlmostEqual(res.total_weight, wsum_exp, places=9)

    def test_noisy_observations_near_truth(self):
        # Deterministic noise stream (no RNG object from the package);
        # sigma roughly 0.3 px.  The fitter must match the independent
        # weighted-LS optimum and sit close to the generating k1.
        f, cx, cy = 800.0, 500.0, 300.0
        k_true = -0.18
        rng = det_noise()
        norm = []
        for i in range(12):
            ang = 2.0 * math.pi * i / 12.0
            rad = 0.15 + 0.45 * ((i * 37) % 13) / 13.0
            norm.append((rad * math.cos(ang), rad * math.sin(ang)))
        pts = []
        for x, y in norm:
            u0, v0 = ref_pixel_project(k_true, f, cx, cy, x, y)
            pts.append((x, y, u0 + 0.3 * next(rng),
                        v0 + 0.3 * next(rng)))
        corrs = [((x, y), (u, v)) for (x, y, u, v) in pts]
        k_exp, _, _ = ref_weighted_k1(pts, [1.0] * len(pts), f, cx, cy)
        cost_exp, wsum = ref_weighted_cost(pts, [1.0] * len(pts),
                                           k_exp, f, cx, cy)
        res = fit_radial_k1(corrs, f, cx, cy)
        self.assertEqual(res.status, "converged")
        self.assertAlmostEqual(res.k1, k_exp, places=12)
        self.assertAlmostEqual(res.k1, k_true, delta=5e-3)
        self.assertAlmostEqual(res.rms, math.sqrt(cost_exp / wsum),
                               places=9)
        # The fit must reduce the residual relative to k1 = 0.
        c0, _ = ref_weighted_cost(pts, [1.0] * len(pts), 0.0, f, cx, cy)
        self.assertLess(res.cost, c0)

    def test_deterministic_and_stateless(self):
        f, cx, cy = 450.0, 10.0, 20.0
        norm = [(-0.2, 0.5), (0.4, 0.1)]
        obs = [ref_pixel_project(-0.05, f, cx, cy, x, y)
               for x, y in norm]
        corrs = [((x, y), (u, v)) for (x, y), (u, v) in zip(norm, obs)]
        r1 = fit_radial_k1(corrs, f, cx, cy)
        r2 = fit_radial_k1(corrs, f, cx, cy)
        self.assertEqual(r1.k1, r2.k1)
        self.assertEqual(r1.cost, r2.cost)
        self.assertEqual(r1.iterations, r2.iterations)


# ---------------------------------------------------------------------------
# Boundary / degenerate geometry
# ---------------------------------------------------------------------------

class TestFitDegenerate(unittest.TestCase):
    def test_all_points_at_origin_is_degenerate(self):
        # k1 never enters the prediction at r = 0; A = 0 exactly.
        corrs = [((0.0, 0.0), (320.0, 240.0)),
                 ((0.0, 0.0), (320.0, 240.0))]
        res = fit_radial_k1(corrs, f=500.0, cx=320.0, cy=240.0,
                            initial_k1=-0.3)
        self.assertEqual(res.status, "degenerate")
        self.assertFalse(res.converged)
        self.assertEqual(res.iterations, 0)
        self.assertEqual(res.k1, -0.3)  # kept at the initial value
        self.assertEqual(res.observability, 0.0)
        self.assertAlmostEqual(res.rms, 0.0, places=9)
        self.assertIn("degenerate", res.diagnostic)

    def test_degenerate_with_misregistration_reports_residual(self):
        # The single origin point is observed 10 px off the predicted
        # principal point: degenerate for k1, but the RMS residual says
        # how badly the (unfittable) data fit.
        res = fit_radial_k1([((0.0, 0.0), (10.0, 0.0))],
                            f=100.0, cx=0.0, cy=0.0)
        self.assertEqual(res.status, "degenerate")
        self.assertAlmostEqual(res.rms, 10.0, places=9)
        self.assertAlmostEqual(res.cost, 100.0, places=8)
        self.assertEqual(res.n_points, 1)

    def test_single_off_center_point_solvable(self):
        # One correspondence away from the origin fully determines k1.
        f, cx, cy = 250.0, 5.0, -7.0
        x, y = 0.3, -0.4
        u, v = ref_pixel_project(0.22, f, cx, cy, x, y)
        res = fit_radial_k1([((x, y), (u, v))], f, cx, cy)
        self.assertEqual(res.status, "converged")
        self.assertAlmostEqual(res.k1, 0.22, places=11)
        self.assertLessEqual(res.rms, 1e-9)

    def test_integer_inputs_accepted(self):
        res = fit_radial_k1([((1, 0), (120, 0))], f=100, cx=0, cy=0)
        # r2=1: u = 100*(1+k1) = 120 -> k1 = 0.2 by hand.
        self.assertEqual(res.status, "converged")
        self.assertAlmostEqual(res.k1, 0.2, places=12)
        self.assertEqual(finite_real(res.k1), res.k1)


# ---------------------------------------------------------------------------
# Invalid input: clear errors, no partial output
# ---------------------------------------------------------------------------

class TestFitValidation(unittest.TestCase):
    def good(self):
        return [((0.3, 0.4), (315.0, 420.0)),
                ((-0.2, 0.1), (-95.0, 105.0))]

    def test_structure_errors(self):
        with self.assertRaises(TypeError):
            fit_radial_k1("not-a-list", 500.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            fit_radial_k1([], 500.0, 0.0, 0.0)
        with self.assertRaises(TypeError):
            fit_radial_k1([((0.0, 0.0),)], 500.0, 0.0, 0.0)
        with self.assertRaises(TypeError):
            fit_radial_k1([((0.0, 0.0, 0.1), (0.0, 0.0))],
                          500.0, 0.0, 0.0)
        with self.assertRaises(TypeError):
            fit_radial_k1([((0.0, 0.0), (0.0,))], 500.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            fit_radial_k1([((0.0, float("nan")), (0.0, 0.0))],
                          500.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            fit_radial_k1([((0.0, 0.0), (0.0, float("inf")))],
                          500.0, 0.0, 0.0)

    def test_batch_limit_atomicity(self):
        big = [((0.1 * (i % 9), 0.0), (1.0, 0.0)) for i in range(101)]
        with self.assertRaises(ValueError):
            fit_radial_k1(big, 500.0, 0.0, 0.0)
        # The failure happens before any fit object exists: the call is
        # a pure function, nothing mutable could be half-updated.
        self.assertEqual(fit_radial_k1(self.good(), 500.0, 0.0,
                                       0.0).status, "converged")

    def test_intrinsics_errors(self):
        for bad in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaises((ValueError, TypeError)):
                fit_radial_k1(self.good(), bad, 0.0, 0.0)
        with self.assertRaises(ValueError):
            fit_radial_k1(self.good(), 500.0, float("inf"), 0.0)
        with self.assertRaises(TypeError):
            fit_radial_k1(self.good(), True, 0.0, 0.0)
        with self.assertRaises(TypeError):
            fit_radial_k1(self.good(), "500", 0.0, 0.0)

    def test_weight_errors(self):
        with self.assertRaises(ValueError):
            fit_radial_k1(self.good(), 500.0, 0.0, 0.0,
                          weights=[1.0])  # length mismatch
        with self.assertRaises(ValueError):
            fit_radial_k1(self.good(), 500.0, 0.0, 0.0,
                          weights=[1.0, 0.0])  # non-positive
        with self.assertRaises(ValueError):
            fit_radial_k1(self.good(), 500.0, 0.0, 0.0,
                          weights=[1.0, -2.0])
        with self.assertRaises(ValueError):
            fit_radial_k1(self.good(), 500.0, 0.0, 0.0,
                          weights=[1.0, float("nan")])
        with self.assertRaises(TypeError):
            fit_radial_k1(self.good(), 500.0, 0.0, 0.0,
                          weights=2.0)

    def test_initial_k1_and_budget_errors(self):
        with self.assertRaises(ValueError):
            fit_radial_k1(self.good(), 500.0, 0.0, 0.0,
                          initial_k1=float("inf"))
        with self.assertRaises(TypeError):
            fit_radial_k1(self.good(), 500.0, 0.0, 0.0,
                          max_iterations=True)
        with self.assertRaises(ValueError):
            fit_radial_k1(self.good(), 500.0, 0.0, 0.0,
                          max_iterations=0)
        with self.assertRaises(ValueError):
            fit_radial_k1(self.good(), 500.0, 0.0, 0.0,
                          max_iterations=10001)

    def test_error_messages_are_locatable(self):
        try:
            fit_radial_k1([((float("nan"), 0.0), (0.0, 0.0))],
                          500.0, 0.0, 0.0)
        except ValueError as exc:
            self.assertIn("correspondences[0].x", str(exc))
        else:
            self.fail("expected ValueError")
        try:
            fit_radial_k1(self.good(), -1.0, 0.0, 0.0)
        except ValueError as exc:
            self.assertIn("f", str(exc))
        else:
            self.fail("expected ValueError")


if __name__ == "__main__":
    unittest.main()
