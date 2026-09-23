"""Tests for the distortion package.

Independence rule: expected values in this file come from the task
specification re-derived here by hand (analytic cases, independent
critical-point quadratic, dense sign-change scans, and central finite
differences).  The package under test is never used to manufacture its
own expected values; distort/undistort round trips appear only as
supplementary assertions.

Run: python -m unittest discover -s tests -v
"""

import math
import unittest

from distortion import DistortionModel, Candidate, InverseResult, finite_real

EPS = 2.220446049250313e-16


# ---------------------------------------------------------------------------
# Independent references (derived directly from the task spec)
# ---------------------------------------------------------------------------

def ref_distort(k1, k2, p1, p2, x, y):
    """Independent transcription of the spec's forward formulas."""
    r2 = x * x + y * y
    s = 1.0 + k1 * r2 + k2 * r2 * r2
    xp = x * s + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yp = y * s + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return xp, yp


def fd_jacobian(k1, k2, p1, p2, x, y, h=1e-6):
    """Independent central-difference Jacobian of the forward map."""
    def f(xx, yy):
        return ref_distort(k1, k2, p1, p2, xx, yy)

    xp_xp, yp_xp = f(x + h, y)
    xp_xm, yp_xm = f(x - h, y)
    xp_yp, yp_yp = f(x, y + h)
    xp_ym, yp_ym = f(x, y - h)
    return (((xp_xp - xp_xm) / (2 * h), (xp_yp - xp_ym) / (2 * h)),
            ((yp_xp - yp_xm) / (2 * h), (yp_yp - yp_ym) / (2 * h)))


def ref_g(k1, k2, r):
    """Scalar radial map g(r) = r(1 + k1 r^2 + k2 r^4)."""
    r2 = r * r
    return r * (1.0 + k1 * r2 + k2 * r2 * r2)


def ref_critical_radii(k1, k2, r_hi):
    """Positive roots of g'(r)=0 in (0, r_hi), from the spec quadratic."""
    out = []
    if k2 == 0.0:
        if k1 != 0.0:
            q = -1.0 / (3.0 * k1)
            if q > 0.0:
                r = math.sqrt(q)
                if r < r_hi:
                    out.append(r)
        return out
    a, b, c = 5.0 * k2, 3.0 * k1, 1.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return out
    root = math.sqrt(disc)
    for q in ((-b - root) / (2 * a), (-b + root) / (2 * a)):
        if q > 0.0:
            r = math.sqrt(q)
            if r < r_hi:
                out.append(r)
    return sorted(out)


def count_axis_preimages(k1, k2, s, r_hi):
    """Independent expected preimage count of target (s, 0), s != 0.

    Preimages on the x-axis solve g(r) = s at (r, 0) or g(r) = -s at
    (r, 0).  Counted by a dense sign-change scan over [0, r_hi] with
    explicit checks for tangential hits at the independently computed
    critical points -- the same mathematics as the core certificate,
    but evaluated by brute force rather than segmented solving.
    """
    n = 600001
    dr = r_hi / (n - 1)
    band = 64.0 * EPS * max(1.0, abs(s), r_hi ** 5 * (abs(k1) + abs(k2)))

    def scan(target):
        count = 0
        prev = ref_g(k1, k2, 0.0) - target
        for i in range(1, n):
            cur = ref_g(k1, k2, i * dr) - target
            if prev * cur < 0.0 or (cur == 0.0 and prev != 0.0):
                count += 1  # sign change, or an exact grid-point root
            prev = cur
        for rc in ref_critical_radii(k1, k2, r_hi):
            if abs(ref_g(k1, k2, rc) - target) <= band:
                count += 1  # tangential (even-multiplicity) touch
        if abs(ref_g(k1, k2, r_hi) - target) <= band:
            count += 1
        return count

    return scan(s) + scan(-s)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestForward(unittest.TestCase):
    def test_hand_computed_examples(self):
        # (coeffs, point, expected) computed with pencil and paper:
        #   identity everywhere;
        #   k1-only at (1,0): xp = 1+k1;
        #   all coeffs at (1,0), r2=1:
        #     xp = 1+k1+k2 + 2 p1*0 + p2(1+2) = 1+k1+k2+3p2
        #     yp = 0 + p1(1+0) + 0 = p1
        cases = [
            (0.0, 0.0, 0.0, 0.0, 0.3, -0.4, 0.3, -0.4),
            (-0.2, 0.0, 0.0, 0.0, 1.0, 0.0, 0.8, 0.0),
            (-0.1, 0.02, 0.03, -0.04, 1.0, 0.0,
             1.0 - 0.1 + 0.02 + 3.0 * -0.04, 0.03),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ]
        for k1, k2, p1, p2, x, y, exp_x, exp_y in cases:
            with self.subTest(x=x, y=y, k1=k1):
                m = DistortionModel(k1, k2, p1, p2)
                xp, yp = m.distort(x, y)
                self.assertAlmostEqual(xp, exp_x, places=12)
                self.assertAlmostEqual(yp, exp_y, places=12)

    def test_matches_independent_transcription(self):
        for k1, k2, p1, p2 in ((0, 0, 0, 0),
                               (-0.31, 0.07, 0.02, -0.015),
                               (0.2, -0.01, -0.03, 0.04)):
            m = DistortionModel(k1, k2, p1, p2)
            for xx in (-0.7, 0.23, 0.91):
                for yy in (-0.55, 0.4, 0.8):
                    rx, ry = ref_distort(k1, k2, p1, p2, xx, yy)
                    ax, ay = m.distort(xx, yy)
                    self.assertAlmostEqual(rx, ax, places=12)
                    self.assertAlmostEqual(ry, ay, places=12)

    def test_int_inputs_accepted(self):
        m = DistortionModel(0, 0, 0, 0)
        self.assertEqual(m.distort(1, -2), (1.0, -2.0))

    def test_batch_limits_and_atomicity(self):
        m = DistortionModel(k1=-0.1)
        pts = [(i * 0.01, 0.0) for i in range(100)]
        self.assertEqual(len(m.distort_all(pts)), 100)
        with self.assertRaises(ValueError):
            m.distort_all(pts + [(0.0, 0.0)])  # 101 > limit
        with self.assertRaises(TypeError):
            m.distort_all([(0.0, 0.0), "nope"])
        with self.assertRaises(ValueError):
            m.distort_all([(float("nan"), 0.0)])

        tg = [(0.01 * i, 0.2) for i in range(100)]
        self.assertEqual(len(m.undistort_all(tg, 1.0)), 100)
        with self.assertRaises(ValueError):
            m.undistort_all(tg + [(0.0, 0.0)], 1.0)
        with self.assertRaises(ValueError):
            m.undistort_all(tg, 0.0)
        with self.assertRaises(TypeError):
            m.undistort_all([(True, 0.0)], 1.0)


class TestJacobian(unittest.TestCase):
    def test_analytic_vs_finite_differences(self):
        m = DistortionModel(k1=-0.31, k2=0.07, p1=0.02, p2=-0.015)
        for x, y in ((0.37, -0.21), (-0.8, 0.55), (0.01, 0.99),
                     (0.0, 0.0), (-0.5, -0.5)):
            with self.subTest(x=x, y=y):
                _, _, j00, j01, j11 = m.residual_and_jacobian(x, y, 0, 0)
                ref = fd_jacobian(m.k1, m.k2, m.p1, m.p2, x, y)
                self.assertAlmostEqual(j00, ref[0][0], delta=2e-6)
                self.assertAlmostEqual(j01, ref[0][1], delta=2e-6)
                self.assertAlmostEqual(j01, ref[1][0], delta=2e-6)
                self.assertAlmostEqual(j11, ref[1][1], delta=2e-6)


class TestRadialInverse(unittest.TestCase):
    K1, K2, R = -0.8, 0.1, 3.0

    def setUp(self):
        self.m = DistortionModel(k1=self.K1, k2=self.K2)

    def test_identity_inverse_unique(self):
        m = DistortionModel()
        for tx, ty in ((0.5, 0.0), (-0.3, 0.4), (0.99, -0.99)):
            r = m.undistort(tx, ty, R=1.0)
            self.assertEqual(len(r.candidates), 1)
            self.assertTrue(r.unique)
            self.assertFalse(r.proven_no_solution)
            c = r.candidates[0]
            self.assertAlmostEqual(c.x, tx, delta=1e-11)
            self.assertAlmostEqual(c.y, ty, delta=1e-11)
            self.assertLessEqual(c.residual, 1e-10)

    def test_center(self):
        # Small box straddles the first interior critical point
        # (r ~= 0.679); r=0 must be present and nothing leaves the box.
        r = self.m.undistort(0.0, 0.0, R=1.0)
        xs = [round(c.x, 9) for c in r.candidates]
        self.assertIn(0.0, xs)
        self.assertTrue(all(-1.0 <= c.x <= 1.0 and -1.0 <= c.y <= 1.0
                            for c in r.candidates))

        # Big box: g(r)=0 beyond origin solves 0.1 q^2 - 0.8 q + 1 = 0
        # with q = 4 +/- sqrt(6), checked analytically.
        r_big = self.m.undistort(0.0, 0.0, R=self.R)
        radii = sorted(round(math.hypot(c.x, c.y), 9)
                       for c in r_big.candidates)
        expected = sorted([0.0,
                           math.sqrt(4.0 - math.sqrt(6.0)),
                           math.sqrt(4.0 + math.sqrt(6.0))])
        self.assertEqual(len(radii), 3)
        for got, exp in zip(radii, expected):
            self.assertAlmostEqual(got, exp, delta=1e-8)
        self.assertFalse(r_big.unique)  # circle continua collapse to 0

    def test_box_outside_proven_no_solution(self):
        m = DistortionModel()
        for tx, ty in ((1.5, 0.0), (0.0, -1.0001), (1.1, 1.1)):
            r = m.undistort(tx, ty, R=1.0)
            self.assertEqual(r.candidates, [])
            self.assertTrue(r.proven_no_solution)
            self.assertFalse(r.budget_exhausted)

        # Folding map on a face-normal ray: independent max|g| bound.
        crit = ref_critical_radii(self.K1, self.K2, self.R)
        gmax = max([abs(ref_g(self.K1, self.K2, rr)) for rr in crit]
                   + [abs(ref_g(self.K1, self.K2, self.R)), 0.0])
        r = self.m.undistort(gmax + 0.5, 0.0, R=self.R)
        self.assertTrue(r.proven_no_solution)
        self.assertEqual(r.candidates, [])

    def test_folded_multiplicity_matches_independent_scan(self):
        # Target genuinely produced by the forward map at r = 0.8.
        s, _ = ref_distort(self.K1, self.K2, 0.0, 0.0, 0.8, 0.0)
        expected = count_axis_preimages(self.K1, self.K2, s, self.R)
        r = self.m.undistort(s, 0.0, R=self.R)
        self.assertGreaterEqual(expected, 3)  # the map really folds here
        self.assertEqual(len(r.candidates), expected)
        self.assertFalse(r.unique)
        # Must not stop at the first root: both ray directions appear.
        signs = {1 if c.x > 0 else -1 for c in r.candidates
                 if abs(c.x) > 1e-9}
        self.assertEqual(signs, {-1, 1})
        for c in r.candidates:
            self.assertLessEqual(c.residual, 1e-10)
            self.assertTrue(-self.R <= c.x <= self.R)
            self.assertTrue(-self.R <= c.y <= self.R)
        self.assertEqual(len(r.critical_radii),
                         len(ref_critical_radii(self.K1, self.K2, self.R)))

    def test_monotone_barrel_unique_certificate(self):
        m = DistortionModel(k1=-0.1)
        s = abs(ref_g(-0.1, 0.0, 0.6))
        r = m.undistort(s, 0.0, R=1.0)
        self.assertEqual(len(r.candidates), 1)
        self.assertTrue(r.unique)
        self.assertEqual(r.critical_radii, [])

    def test_singular_critical_value_point(self):
        # At the first local extremum g'(rc) = 0; independently the FD
        # Jacobian determinant vanishes there and the tangent root must
        # still be returned (one root, not a pair straddling rc).
        rc = ref_critical_radii(self.K1, self.K2, self.R)[0]
        s = ref_g(self.K1, self.K2, rc)
        r = self.m.undistort(s, 0.0, R=self.R)
        near = [c for c in r.candidates if abs(c.x - rc) < 1e-5]
        self.assertEqual(len(near), 1)
        self.assertLessEqual(near[0].residual, 1e-9)
        J = fd_jacobian(self.K1, self.K2, 0.0, 0.0, rc, 0.0)
        det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
        self.assertAlmostEqual(det, 0.0, delta=1e-5)

    def test_residual_ordering(self):
        s, _ = ref_distort(self.K1, self.K2, 0.0, 0.0, 2.0, 0.0)
        r = self.m.undistort(s, 0.0, R=self.R)
        resids = [c.residual for c in r.candidates]
        self.assertEqual(resids, sorted(resids))
        self.assertLessEqual(max(resids), 1e-10)

    def test_square_box_geometry_on_diagonal_ray(self):
        # The search box is the SQUARE [-R,R]^2: along the 45-degree
        # ray it reaches radius R*sqrt(2) (the corners).  Identity map:
        # (0.9, 0.9) is inside the square although its radius > R, and
        # (1.2, 1.2) lies beyond even the corner reach.
        m = DistortionModel()
        r_in = m.undistort(0.9, 0.9, R=1.0)
        self.assertEqual(len(r_in.candidates), 1)
        c = r_in.candidates[0]
        self.assertAlmostEqual(c.x, 0.9, delta=1e-11)
        self.assertAlmostEqual(c.y, 0.9, delta=1e-11)

        r_out = m.undistort(1.2, 1.2, R=1.0)
        self.assertTrue(r_out.proven_no_solution)
        self.assertEqual(r_out.candidates, [])


class TestGeneralInverse(unittest.TestCase):
    def test_tangential_roundtrip_supplementary(self):
        # Supplementary only: round trip does not prove correctness by
        # itself, but every genuine forward point must be recoverable.
        m = DistortionModel(k1=-0.15, k2=0.02, p1=0.01, p2=-0.008)
        for x, y in ((0.3, -0.4), (0.9, 0.9), (-0.7, 0.2), (0.0, 0.0)):
            xp, yp = m.distort(x, y)
            r = m.undistort(xp, yp, R=1.2)
            self.assertEqual(len(r), 1)
            c = r.candidates[0]
            self.assertAlmostEqual(c.x, x, delta=1e-9)
            self.assertAlmostEqual(c.y, y, delta=1e-9)
            self.assertFalse(r.proven_no_solution)

    def test_fold_with_tangential_finds_all_basins(self):
        k1, k2, R = -0.8, 0.1, 3.0
        m = DistortionModel(k1=k1, k2=k2, p1=1e-5, p2=-1e-5)
        s, _ = ref_distort(k1, k2, 0.0, 0.0, 0.8, 0.0)
        expected = count_axis_preimages(k1, k2, s, R)
        r = m.undistort(s, 0.0, R=R)
        self.assertFalse(r.radial)
        self.assertEqual(len(r.candidates), expected)
        for c in r.candidates:
            self.assertLessEqual(c.residual, 1e-10)

    def test_budget_flag_distinct_from_no_solution(self):
        m = DistortionModel(k1=-0.8, k2=0.1, p1=1e-4)
        r = m.undistort(100.0, 0.0, R=1.0)
        self.assertFalse(r.proven_no_solution)  # search never certifies
        self.assertEqual(r.candidates, [])
        r2 = m.undistort(0.3, 0.0, R=1.0, max_iterations=1,
                         starts_per_axis=2)
        self.assertFalse(r2.proven_no_solution)
        self.assertTrue(r2.budget_exhausted)

    def test_result_types(self):
        m = DistortionModel()
        r = m.undistort(0.2, 0.1, 1.0)
        self.assertIsInstance(r, InverseResult)
        self.assertIsInstance(r.candidates[0], Candidate)
        x, y = tuple(r.candidates[0])
        self.assertAlmostEqual(x, 0.2, places=10)
        self.assertAlmostEqual(y, 0.1, places=10)


class TestValidation(unittest.TestCase):
    def test_reject_non_finite(self):
        m = DistortionModel()
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                DistortionModel(bad, 0.0)
            with self.assertRaises(ValueError):
                m.distort(bad, 0.0)
            with self.assertRaises(ValueError):
                m.undistort(bad, 0.0, 1.0)
            with self.assertRaises(ValueError):
                m.undistort(0.0, 0.0, bad)

    def test_reject_bool_and_wrong_types(self):
        with self.assertRaises(TypeError):
            DistortionModel(True, 0.0)
        m = DistortionModel()
        with self.assertRaises(TypeError):
            m.distort(True, 0.0)
        with self.assertRaises(TypeError):
            m.undistort(0.0, 0.0, 1.0, starts_per_axis=True)
        with self.assertRaises(TypeError):
            m.undistort("0", 0.0, 1.0)
        with self.assertRaises(TypeError):
            m.distort_all((0.0, 0.0))

    def test_bad_box_and_tolerance(self):
        m = DistortionModel()
        for bad_R in (0.0, -1.0):
            with self.assertRaises(ValueError):
                m.undistort(0.0, 0.0, bad_R)
        with self.assertRaises(ValueError):
            m.undistort(0.0, 0.0, 1.0, residual_tol=0.0)
        with self.assertRaises(ValueError):
            m.undistort(0.0, 0.0, 1.0, starts_per_axis=0)
        with self.assertRaises(ValueError):
            m.undistort(0.0, 0.0, 1.0, starts_per_axis=13)

    def test_finite_real_helper(self):
        self.assertEqual(finite_real(2), 2.0)
        with self.assertRaises(TypeError):
            finite_real(True)
        with self.assertRaises(TypeError):
            finite_real(1 + 2j)
        with self.assertRaises(ValueError):
            finite_real(float("inf"))


if __name__ == "__main__":
    unittest.main()
