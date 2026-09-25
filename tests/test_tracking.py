"""Tests for the continuous-branch tracking entry point.

Independence rule: the expected root sets here come from an
INDEPENDENT high-precision one-dimensional segmented root finder
written in this file (dense sign-change scan + 300-iteration
bisection per crossing, tangencies at independently computed critical
points).  The package under test is never used to manufacture
expected values; forward re-projection is only a supplementary check.

Run: python -m unittest discover -s tests -v
"""

import math
import unittest

from distortion import (
    DistortionModel,
    InverseTracker,
    track_points,
    ANCHORED,
    OK,
    AMBIGUOUS,
    NO_CANDIDATE,
    STEP_TOO_LARGE,
    ANCHOR_REJECTED,
    BROKEN_NEEDS_ANCHOR,
)

K1, K2, R = -0.8, 0.1, 3.0
EPS = 2.220446049250313e-16


# ---------------------------------------------------------------------------
# Independent high-precision 1-D reference for the radial cross-section
# ---------------------------------------------------------------------------

def ref_g(r):
    r2 = r * r
    return r * (1.0 + K1 * r2 + K2 * r2 * r2)


def ref_critical_radii(r_hi):
    """Positive roots of g'(r)=0 in (0,r_hi), from 5k2 q^2+3k1 q+1=0."""
    a, b, c = 5.0 * K2, 3.0 * K1, 1.0
    disc = b * b - 4.0 * a * c
    assert disc >= 0.0
    root = math.sqrt(disc)
    out = []
    for q in ((-b - root) / (2 * a), (-b + root) / (2 * a)):
        if q > 0.0:
            r = math.sqrt(q)
            if r < r_hi:
                out.append(r)
    return sorted(out)


def bisect_refine(lo, hi, target, iters=300):
    """300-iteration bisection of g(r)=target on a bracketed crossing."""
    flo = ref_g(lo) - target
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = ref_g(mid) - target
        if flo * fm <= 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def independent_axis_roots(s, r_hi=R):
    """All x-axis preimages of target (s, 0), s != 0, as signed x.

    Independent segmented search: dense sign-change scan over [0,r_hi],
    300 bisection steps per crossing, plus explicit tangency roots at
    the independently computed critical points.
    """
    crit = ref_critical_radii(r_hi)
    band = 64.0 * EPS * max(1.0, abs(s), r_hi ** 5 * (abs(K1) + abs(K2)))
    roots = []

    def scan(target, sign):
        n = 300001
        dr = r_hi / (n - 1)
        lo = 0.0
        prev = ref_g(lo) - target
        for i in range(1, n):
            r = i * dr
            cur = ref_g(r) - target
            if prev * cur < 0.0:
                roots.append(sign * bisect_refine(lo, r, target))
            elif cur == 0.0 and prev != 0.0:
                roots.append(sign * r)  # exact grid-point crossing
            lo, prev = r, cur
        for rc in crit:
            if abs(ref_g(rc) - target) <= band:
                rr = sign * rc
                if all(abs(rr - q) > 1e-9 for q in roots):
                    roots.append(rr)
        if abs(ref_g(r_hi) - target) <= band:
            rr = sign * r_hi
            if all(abs(rr - q) > 1e-9 for q in roots):
                roots.append(rr)

    scan(s, 1.0)      # g(r) = +s -> preimage at (+r, 0)
    scan(-s, -1.0)    # g(r) = -s -> preimage at (-r, 0)
    return sorted(roots)


def independent_pair_around_fold(s, r_hi=R):
    """The two roots straddling the first fold (near r_c), ascending."""
    rc = ref_critical_radii(r_hi)[0]
    near = sorted(x for x in independent_axis_roots(s, r_hi)
                  if abs(abs(x) - rc) < 0.2)
    assert len(near) == 2, near
    return near[0], near[1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestUniqueBranch(unittest.TestCase):
    def setUp(self):
        self.m = DistortionModel(k1=K1, k2=K2)

    def test_fold_track_stays_on_source_branch(self):
        # Every frame genuinely has several preimages (the map folds);
        # the tracker keeps the continuous inner branch instead of
        # silently taking candidates[0].
        sources = [0.30, 0.38, 0.46, 0.54]
        targets = [(ref_g(r), 0.0) for r in sources]
        steps = track_points(self.m, targets, (sources[0], 0.0), R,
                             max_step=0.2)
        self.assertEqual([s.status for s in steps],
                         [ANCHORED] + [OK] * 3)
        for step, source in zip(steps, sources):
            x = step.point[0]
            self.assertAlmostEqual(x, source, delta=1e-9)
            ref_roots = independent_axis_roots(step.target[0])
            self.assertTrue(any(abs(x - rr) <= 1e-9 for rr in ref_roots),
                            (x, ref_roots))
            self.assertLessEqual(step.selected.residual, 1e-10)

    def test_identity_unique_branch(self):
        m = DistortionModel()
        targets = [(0.10, 0.0), (0.15, 0.05), (0.20, 0.10)]
        steps = track_points(m, targets, (0.10, 0.0), 1.0, max_step=0.2)
        self.assertEqual([s.status for s in steps], [ANCHORED, OK, OK])
        for step, t in zip(steps, targets):
            self.assertAlmostEqual(step.point[0], t[0], delta=1e-11)
            self.assertAlmostEqual(step.point[1], t[1], delta=1e-11)


class TestNearFold(unittest.TestCase):
    def setUp(self):
        self.m = DistortionModel(k1=K1, k2=K2)
        self.rc = ref_critical_radii(R)[0]

    def test_track_approaches_turning_point_continuously(self):
        # Sources walk up the inner branch toward r_c ~= 0.679 but stay
        # just below it; selected roots stay on that inner branch and
        # move monotonically with the source.
        sources = [0.55, 0.60, 0.63, 0.66]
        targets = [(ref_g(r), 0.0) for r in sources]
        steps = track_points(self.m, targets, (sources[0], 0.0), R,
                             max_step=0.1)
        self.assertTrue(all(s.status in (ANCHORED, OK) for s in steps))
        chosen = [s.point[0] for s in steps]
        for a, b in zip(chosen, chosen[1:]):
            self.assertGreater(b, a)
        for x in chosen:
            self.assertLess(x, self.rc)
        for step in steps[1:]:
            self.assertGreaterEqual(len(step.candidates), 3)
            ref_roots = independent_axis_roots(step.target[0])
            self.assertGreaterEqual(len(ref_roots), 3)
            self.assertTrue(
                any(abs(step.point[0] - rr) <= 1e-9 for rr in ref_roots))



class TestAmbiguousTie(unittest.TestCase):
    def setUp(self):
        self.m = DistortionModel(k1=K1, k2=K2)

    def test_equidistant_roots_return_ambiguous_with_both_kept(self):
        # Exact tie, constructed independently: anchor at the midpoint
        # m = (r_lo + r_hi)/2 of the two roots bracketing r_c at target
        # s1; at that frame the two nearest candidates are equidistant
        # from m by construction.
        s1 = ref_g(ref_critical_radii(R)[0] - 0.10)
        r_lo, r_hi = independent_pair_around_fold(s1)
        mid = 0.5 * (r_lo + r_hi)
        s0 = ref_g(mid)
        self.assertAlmostEqual(r_hi - mid, mid - r_lo, delta=1e-12)

        steps = track_points(self.m, [(s0, 0.0), (s1, 0.0)],
                             (mid, 0.0), R,
                             max_step=0.5, equal_tol=1e-9)
        self.assertEqual(steps[0].status, ANCHORED)
        tie = steps[1]
        self.assertEqual(tie.status, AMBIGUOUS)
        self.assertIsNone(tie.selected)
        self.assertEqual(len(tie.tied), 2)
        tied_x = sorted(c.x for c in tie.tied)
        self.assertAlmostEqual(tied_x[0], r_lo, delta=1e-8)
        self.assertAlmostEqual(tied_x[1], r_hi, delta=1e-8)
        tied_d = [dd for c, dd in zip(tie.candidates, tie.distances)
                  if c in tie.tied]
        self.assertEqual(len(tied_d), 2)
        self.assertAlmostEqual(tied_d[0], tied_d[1],
                               delta=1e-9 * max(1.0, tied_d[0]))
        # Every candidate retained, not only the tied pair.
        self.assertGreaterEqual(len(tie.candidates), 4)
        # A tie never depends on candidate list order: reversing the
        # raw candidate list leaves the same ambiguous pair.
        tracker = InverseTracker(self.m, R, max_step=0.5, equal_tol=1e-9)
        tracker.start(s0, 0.0, mid, 0.0)
        raw = tracker._solve(s1, 0.0)
        raw.candidates[:] = list(reversed(raw.candidates))
        tracker._solve = lambda xp, yp: raw
        swapped = tracker.update(1, s1, 0.0)
        self.assertEqual(swapped.status, AMBIGUOUS)
        self.assertEqual(sorted(c.x for c in swapped.tied), tied_x)


class TestWrongAnchor(unittest.TestCase):
    def setUp(self):
        self.m = DistortionModel(k1=K1, k2=K2)

    def test_anchor_must_forward_project_to_first_target(self):
        steps = track_points(
            self.m, [(ref_g(0.60), 0.0), (ref_g(0.62), 0.0)],
            (0.50, 0.0), R, max_step=0.1)
        self.assertEqual(steps[0].status, ANCHOR_REJECTED)
        self.assertIsNone(steps[0].selected)
        self.assertGreaterEqual(len(steps[0].candidates), 1)
        self.assertEqual(steps[1].status, BROKEN_NEEDS_ANCHOR)

    def test_anchor_outside_box_rejected(self):
        m = DistortionModel()
        steps = track_points(m, [(1.5, 0.0)], (1.5, 0.0), 1.0,
                             max_step=0.1)
        self.assertEqual(steps[0].status, ANCHOR_REJECTED)
        self.assertIn("outside the search box", steps[0].reason)


class TestMidTrackBreak(unittest.TestCase):
    def setUp(self):
        self.m = DistortionModel(k1=K1, k2=K2)

    def test_step_over_limit_breaks_and_requires_new_anchor(self):
        # Anchor at r=0.45; the frame at r=0.62 folds back so the
        # nearest candidate is ~0.17 away while max_step is 0.02.
        steps = track_points(
            self.m, [(ref_g(0.45), 0.0), (ref_g(0.62), 0.0),
                     (ref_g(0.60), 0.0)],
            (0.45, 0.0), R, max_step=0.02)
        self.assertEqual(steps[0].status, ANCHORED)
        self.assertEqual(steps[1].status, STEP_TOO_LARGE)
        self.assertIsNone(steps[1].selected)
        self.assertGreater(min(steps[1].distances), 0.02)
        self.assertEqual(steps[2].status, BROKEN_NEEDS_ANCHOR)

        # A fresh forward-verified anchor resumes tracking.
        tracker = InverseTracker(self.m, R, max_step=0.02)
        tracker.start(ref_g(0.45), 0.0, 0.45, 0.0)
        broken = tracker.update(1, ref_g(0.62), 0.0)
        self.assertEqual(broken.status, STEP_TOO_LARGE)
        blocked = tracker.update(2, ref_g(0.60), 0.0)
        self.assertEqual(blocked.status, BROKEN_NEEDS_ANCHOR)
        restart = tracker.start(ref_g(0.62), 0.0, 0.62, 0.0)
        self.assertEqual(restart.status, ANCHORED)
        cont = tracker.update(3, ref_g(0.60), 0.0)
        self.assertEqual(cont.status, OK)
        self.assertAlmostEqual(cont.point[0], 0.60, delta=1e-9)

    def test_no_candidate_breaks_track(self):
        # Identity in [-1,1]^2: (1.5, 0) is certified absent.
        m = DistortionModel()
        steps = track_points(
            m, [(0.5, 0.0), (1.5, 0.0), (0.6, 0.0)],
            (0.5, 0.0), 1.0, max_step=0.5)
        self.assertEqual(steps[0].status, ANCHORED)
        self.assertEqual(steps[1].status, NO_CANDIDATE)
        self.assertTrue(steps[1].result.proven_no_solution)
        self.assertEqual(steps[1].candidates, [])
        self.assertEqual(steps[2].status, BROKEN_NEEDS_ANCHOR)


class TestTrackingValidation(unittest.TestCase):
    def test_bounds_and_finite_inputs(self):
        m = DistortionModel(k1=K1, k2=K2)
        with self.assertRaises(ValueError):
            InverseTracker(m, R, max_step=0.0)
        with self.assertRaises(ValueError):
            InverseTracker(m, R, max_step=0.1, equal_tol=0.0)
        with self.assertRaises(ValueError):
            InverseTracker(m, R, max_step=0.1, anchor_tol=0.0)
        with self.assertRaises(ValueError):
            InverseTracker(m, 0.0, max_step=0.1)
        with self.assertRaises(ValueError):
            track_points(m, [], (0.5, 0.0), R, max_step=0.1)
        with self.assertRaises(ValueError):
            track_points(m, [(0.0, 0.0)] * 101, (0.0, 0.0), R,
                         max_step=0.1)
        with self.assertRaises(TypeError):
            track_points(m, [(0.0, 0.0)], (0.0,), R, max_step=0.1)
        with self.assertRaises(TypeError):
            track_points(m, "ab", (0.0, 0.0), R, max_step=0.1)
        with self.assertRaises(ValueError):
            track_points(m, [(float("nan"), 0.0)], (0.0, 0.0), R,
                         max_step=0.1)
        with self.assertRaises(ValueError):
            track_points(m, [(0.0, 0.0)], (float("inf"), 0.0), R,
                         max_step=0.1)
        with self.assertRaises(TypeError):
            InverseTracker(m, R, max_step=0.1, starts_per_axis=True)


if __name__ == "__main__":
    unittest.main()
