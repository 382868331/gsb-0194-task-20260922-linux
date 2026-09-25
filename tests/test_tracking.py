"""Tests for continuous-branch tracking (distortion.tracking).

Independence rule (as in test_distortion.py): expected undistorted
roots come from an INDEPENDENT solver -- decimal.Decimal high-precision
segmented bisection of the scalar radial map g(r) -- never from the
package under test.  Distorted targets are generated with the forward
formula re-transcribed here rather than taken from a solver result.

Run: python -m unittest discover -s tests -v
"""

import unittest
from decimal import Decimal, getcontext

from distortion import (DistortionModel, track_undistorted,
                        OK, AMBIGUOUS, NO_CANDIDATE, OVER_STEP,
                        INVALID_ANCHOR, NEEDS_ANCHOR)


# ---------------------------------------------------------------------------
# Independent high-precision radial reference (stdlib Decimal only)
# ---------------------------------------------------------------------------

def decimal_roots_g_eq(k1, k2, target, r_hi, prec=50):
    """All r in [0, r_hi] with g(r) = target, by scan + bisection.

    g(r) = r*(1 + k1*r^2 + k2*r^4).  A 20001-point sign-change scan
    localises crossings; each bracket is bisected at 50 decimal digits.
    Independent of radial.py's critical-point partition.
    """
    getcontext().prec = prec
    K1 = Decimal(str(k1))
    K2 = Decimal(str(k2))
    T = Decimal(str(target))
    RH = Decimal(str(r_hi))

    def g(r):
        r2 = r * r
        return r * (Decimal(1) + (K1 + K2 * r2) * r2)

    n = 20001
    grid = [RH * Decimal(i) / Decimal(n - 1) for i in range(n)]
    vals = [g(r) - T for r in grid]
    roots = []

    def bisect(lo, hi):
        # Sign convention must follow the branch's monotonicity: g may
        # fold several times on [0, r_hi], so decide the direction from
        # the sign change at the bracket midpoint.
        sample = g((lo + hi) / 2) - T

        def accept(value, reference):
            return value >= reference if sample > 0 else value < reference

        for _ in range(180):
            mid = (lo + hi) / 2
            if accept(g(mid) - T, 0):
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    for i in range(1, n):
        a, b = vals[i - 1], vals[i]
        if a == 0:
            roots.append(grid[i - 1])
        if a * b < 0:
            roots.append(bisect(grid[i - 1], grid[i]))
    if vals[-1] == 0:
        roots.append(grid[-1])
    out = []
    for r in roots:
        if not out or abs(r - out[-1]) > Decimal("1e-12"):
            out.append(r)
    return out


def ref_distort(k1, k2, x, y):
    """Forward formulas re-transcribed from the spec (float)."""
    r2 = x * x + y * y
    s = 1.0 + k1 * r2 + k2 * r2 * r2
    return x * s, y * s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestUniqueBranch(unittest.TestCase):
    """Smooth monotone region: tracking follows the single branch."""

    def test_barrel_small_radius_unique_branch(self):
        # k1 = -0.3 is monotone in r for r < rc ~= 1.054; within that
        # region the small positive root is unique on the near segment.
        m = DistortionModel(k1=-0.3)
        src = [(0.20 + 0.03 * i, 0.0) for i in range(8)]
        targets = [ref_distort(-0.3, 0.0, x, 0.0) for (x, _) in src]
        res = track_undistorted(m, targets, (src[0][0], 0.0), R=2.0,
                                max_step=0.1)
        self.assertTrue(all(step.status == OK for step in res))
        self.assertTrue(res.active)
        self.assertIsNone(res.break_index)
        for step, (ex, _) in zip(res, src):
            self.assertAlmostEqual(step.x, ex, delta=1e-11)
            self.assertEqual(step.y, 0.0)
            self.assertLessEqual(step.residual, 1e-10)
            self.assertLessEqual(step.distance, 0.1)
        self.assertGreaterEqual(len(res.steps[-1].candidates), 1)


class TestNearFold(unittest.TestCase):
    """Approaching and crossing a fold: the branch nearest the previous
    point is selected, not the smallest-residual / first list root."""

    K1, K2, R = -0.8, 0.1, 3.0

    def setUp(self):
        self.m = DistortionModel(k1=self.K1, k2=self.K2)

    def test_follow_branch_through_turning_point(self):
        # Two preimage branches coexist near rc1 ~= 0.679 where g
        # turns over.  Approaching from OUTSIDE along the outer branch
        # the nearest-candidate rule keeps the outer root even though
        # the inner root is also residual-acceptable every frame.
        radii = [0.80, 0.77, 0.74, 0.71, 0.69]
        targets = [ref_distort(self.K1, self.K2, r, 0.0)
                   for r in radii]
        res = track_undistorted(self.m, targets, (radii[0], 0.0),
                                R=self.R, max_step=0.05)
        self.assertEqual([s.status for s in res], [OK] * len(radii))
        for step, r in zip(res, radii):
            self.assertAlmostEqual(step.x, r, delta=1e-9)
            self.assertLessEqual(step.distance, 0.05)
        # The inner-fold root is present on every late frame but never
        # selected: it is farther from the previous outer-branch point.
        late = res.steps[-1]
        inner = [c for c in late.candidates if c.x < 0.679]
        self.assertTrue(inner)
        self.assertGreater(abs(inner[0].x - late.x), late.distance)
        # Targets move *up* in distorted value as r decreases toward rc,
        # while the tracked preimage keeps moving inward (no jump).
        sx = [t[0] for t in targets]
        self.assertLess(sx[0], sx[-1])
        self.assertLess(res.steps[-1].x, res.steps[0].x)
        self.assertTrue(all(step.radial for step in res))
        self.assertFalse(any(step.budget_exhausted for step in res))

    def test_selected_roots_match_independent_decimal_bisection(self):
        radii = [0.80, 0.77, 0.74, 0.71]
        targets = [ref_distort(self.K1, self.K2, r, 0.0)
                   for r in radii]
        res = track_undistorted(self.m, targets, (radii[0], 0.0),
                                R=self.R, max_step=0.05)
        for step, r in zip(res, radii):
            s = abs(ref_distort(self.K1, self.K2, r, 0.0)[0])
            ref = ([float(x) for x in
                    decimal_roots_g_eq(self.K1, self.K2, s, self.R)]
                   + [float(x) for x in
                      decimal_roots_g_eq(self.K1, self.K2, -s, self.R)])
            self.assertTrue(any(abs(step.x - rr) < 1e-9 for rr in ref))
            self.assertAlmostEqual(step.x, r, delta=1e-9)


class TestAmbiguousTie(unittest.TestCase):
    """Two roots equidistant from the previous point within tolerance."""

    def test_tie_returns_ambiguous_and_keeps_candidates(self):
        m = DistortionModel(k1=-0.3)
        # g(r) = r(1 - 0.3 r^2), rc ~= 1.054.  For target s = g(1.35)
        # the two positive-ray roots are r1 (~0.5) and r2 = 1.35.
        s = ref_distort(-0.3, 0.0, 1.35, 0.0)[0]
        roots = decimal_roots_g_eq(-0.3, 0.0, s, 2.0)
        r1 = float(roots[0])
        r2 = 1.35
        anchor_r = 0.5 * (r1 + r2)
        t0 = ref_distort(-0.3, 0.0, anchor_r, 0.0)
        res = track_undistorted(
            m, [t0, (s, 0.0)], (anchor_r, 0.0), R=2.0,
            max_step=1.0, tie_tol=1e-7)
        self.assertEqual(res.steps[0].status, OK)
        step = res.steps[1]
        self.assertEqual(step.status, AMBIGUOUS)
        self.assertIsNone(step.x)
        self.assertIsNone(step.y)
        self.assertTrue(res.broken)
        self.assertEqual(res.break_index, 1)
        xs = [c.x for c in step.candidates]
        self.assertTrue(any(abs(x - r1) < 1e-8 for x in xs))
        self.assertTrue(any(abs(x - r2) < 1e-8 for x in xs))
        self.assertIsNone(step.method)  # no list-order pick

    def test_tie_resolved_with_tighter_tolerance(self):
        m = DistortionModel(k1=-0.3)
        s = ref_distort(-0.3, 0.0, 1.35, 0.0)[0]
        r1 = float(decimal_roots_g_eq(-0.3, 0.0, s, 2.0)[0])
        anchor_r = 0.5 * (r1 + 1.35)
        t0 = ref_distort(-0.3, 0.0, anchor_r, 0.0)
        res = track_undistorted(
            m, [t0, (s, 0.0)], (anchor_r, 0.0), R=2.0,
            max_step=1.0, tie_tol=1e-15)
        self.assertEqual(res.steps[1].status, OK)
        self.assertIn(round(res.steps[1].x, 8),
                      (round(r1, 8), 1.35))


class TestAnchorValidation(unittest.TestCase):
    def test_wrong_anchor_breaks_at_frame_zero(self):
        m = DistortionModel(k1=-0.8, k2=0.1)
        target = ref_distort(-0.8, 0.1, 0.5, 0.0)
        res = track_undistorted(m, [target], (0.6, 0.0), R=3.0,
                                max_step=0.2, anchor_tol=1e-8)
        step = res.steps[0]
        self.assertEqual(step.status, INVALID_ANCHOR)
        self.assertGreater(step.residual, 1e-8)
        self.assertFalse(res.active)
        self.assertEqual(res.break_index, 0)
        self.assertIsNone(res.last)
        self.assertTrue(step.candidates)

    def test_anchor_outside_box_rejected_upfront(self):
        m = DistortionModel()
        with self.assertRaises(ValueError):
            track_undistorted(m, [(0.0, 0.0)], (1.5, 0.0), R=1.0,
                              max_step=0.1)

    def test_bad_resume_anchor_stays_broken(self):
        m = DistortionModel(k1=-0.8, k2=0.1)
        t0 = ref_distort(-0.8, 0.1, 0.5, 0.0)
        far = ref_distort(-0.8, 0.1, 2.2, 0.0)
        targets = [t0, far, far, t0]
        res = track_undistorted(
            m, targets, (0.5, 0.0), R=3.0, max_step=0.2,
            resume_anchors={2: (2.0, 0.0)})
        self.assertEqual(res.steps[1].status, OVER_STEP)
        self.assertEqual(res.steps[2].status, INVALID_ANCHOR)
        self.assertEqual(res.steps[3].status, NEEDS_ANCHOR)


class TestBreakAndResume(unittest.TestCase):
    def test_no_candidate_breaks_then_needs_anchor(self):
        # Identity model, box R=1: target 1.5 is certifiably absent.
        m = DistortionModel()
        targets = [(0.2, 0.0), (1.5, 0.0), (0.3, 0.0)]
        res = track_undistorted(m, targets, (0.2, 0.0), R=1.0,
                                max_step=0.5)
        self.assertEqual([s.status for s in res],
                         [OK, NO_CANDIDATE, NEEDS_ANCHOR])
        brk = res.steps[1]
        self.assertTrue(brk.proven_no_solution)
        self.assertEqual(brk.candidates, [])
        self.assertEqual(res.steps[2].candidates, [])
        self.assertFalse(res.active)

    def test_over_step_break_then_resume_with_new_anchor(self):
        m = DistortionModel(k1=-0.8, k2=0.1)
        near = ref_distort(-0.8, 0.1, 0.5, 0.0)
        far = ref_distort(-0.8, 0.1, 2.2, 0.0)
        targets = [near, far, far]
        res = track_undistorted(
            m, targets, (0.5, 0.0), R=3.0, max_step=0.2,
            resume_anchors={2: (2.2, 0.0)})
        self.assertEqual(res.steps[0].status, OK)
        self.assertEqual(res.steps[1].status, OVER_STEP)
        self.assertTrue(res.steps[1].candidates)
        self.assertEqual(res.steps[2].status, OK)
        self.assertEqual(res.steps[2].method, "anchor")
        self.assertTrue(res.active)
        self.assertEqual(res.last, (2.2, 0.0))


class TestTrackingValidation(unittest.TestCase):
    def test_bounds_and_types(self):
        m = DistortionModel(k1=-0.3)
        targets = [(0.2, 0.0), (0.25, 0.0)]
        with self.assertRaises(ValueError):
            track_undistorted(m, targets, (0.2, 0.0), R=2.0, max_step=0.0)
        with self.assertRaises(ValueError):
            track_undistorted(m, targets, (0.2, 0.0), R=2.0,
                              max_step=0.1, tie_tol=0.0)
        with self.assertRaises(ValueError):
            track_undistorted(m, [], (0.2, 0.0), R=2.0, max_step=0.1)
        with self.assertRaises(ValueError):
            track_undistorted(m, targets, (0.2, float("nan")),
                              R=2.0, max_step=0.1)
        with self.assertRaises(TypeError):
            track_undistorted(m, targets, (0.2, 0.0), R=2.0,
                              max_step=0.1, resume_anchors=[])
        with self.assertRaises(ValueError):
            track_undistorted(m, targets, (0.2, 0.0), R=2.0,
                              max_step=0.1,
                              resume_anchors={5: (0.2, 0.0)})

    def test_batch_limit_kept(self):
        m = DistortionModel()
        targets = [(0.0, 0.0)] * 101
        with self.assertRaises(ValueError):
            track_undistorted(m, targets, (0.0, 0.0), R=1.0,
                              max_step=0.1)


if __name__ == "__main__":
    unittest.main()
