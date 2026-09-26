"""Tests for the forward area-conserving resampler.

Independence rule: expected values come from hand-derived analytic
cases, an independent re-implementation of box clipping / shoelace
area written here, and dense independent grid integration of the
Jacobian.  The package under test never manufactures its own expected
values.

Run: python -m unittest discover -s tests -v
"""

import math
import unittest

from distortion import DistortionModel, forward_resample, ForwardResult


# ---------------------------------------------------------------------------
# Independent references (not imported from the package)
# ---------------------------------------------------------------------------

def ref_clip(poly, xmin, ymin, xmax, ymax):
    """Independent Sutherland-Hodgman clipper (re-derived here)."""
    def inside(p, edge, v):
        return {"l": p[0] >= v, "r": p[0] <= v,
                "b": p[1] >= v, "t": p[1] <= v}[edge]

    def cut(poly, edge, v):
        out = []
        if not poly:
            return out
        s, sin = poly[-1], inside(poly[-1], edge, v)
        for e in poly:
            ein = inside(e, edge, v)
            if ein != sin:
                if edge in "lr":
                    t = (v - s[0]) / (e[0] - s[0])
                    out.append((v, s[1] + t * (e[1] - s[1])))
                else:
                    t = (v - s[1]) / (e[1] - s[1])
                    out.append((s[0] + t * (e[0] - s[0]), v))
            if ein:
                out.append(e)
            s, sin = e, ein
        return out

    for edge, v in (("l", xmin), ("r", xmax), ("b", ymin), ("t", ymax)):
        poly = cut(poly, edge, v)
    return poly


def ref_area(poly):
    return 0.5 * abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1]
                         - poly[(i + 1) % len(poly)][0] * poly[i][1]
                         for i in range(len(poly)))) if len(poly) >= 3 else 0.0


def ref_distort(m, x, y):
    r2 = x * x + y * y
    s = 1.0 + m.k1 * r2 + m.k2 * r2 * r2
    return (x * s + 2 * m.p1 * x * y + m.p2 * (r2 + 2 * x * x),
            y * s + m.p1 * (r2 + 2 * y * y) + 2 * m.p2 * x * y)


def exact_affine_reference(model, pixels, src, dst, shape):
    """Exact reference for affine maps: map each source cell's 4 corners
    (exact for affine maps) and clip against every target cell."""
    sx0, sx1, sy0, sy1 = src
    dx0, dx1, dy0, dy1 = dst
    rows, cols = len(pixels), len(pixels[0])
    dh, dw = shape
    cw, ch = (sx1 - sx0) / cols, (sy1 - sy0) / rows
    gw, gh = (dx1 - dx0) / dw, (dy1 - dy0) / dh
    energy = [[0.0] * dw for _ in range(dh)]
    area = [[0.0] * dw for _ in range(dh)]
    lost = 0.0
    for r in range(rows):
        y1 = sy1 - r * ch
        y0 = y1 - ch
        for c in range(cols):
            x0 = sx0 + c * cw
            x1 = x0 + cw
            v = pixels[r][c]
            quad = [ref_distort(model, x0, y0), ref_distort(model, x1, y0),
                    ref_distort(model, x1, y1), ref_distort(model, x0, y1)]
            qa = ref_area(quad)
            clipped = ref_clip(quad, dx0, dy0, dx1, dy1)
            lost += v * (qa - ref_area(clipped))
            for gr in range(dh):
                for gc in range(dw):
                    piece = ref_clip(clipped, dx0 + gc * gw, dy0 + gr * gh,
                                     dx0 + (gc + 1) * gw, dy0 + (gr + 1) * gh)
                    a = ref_area(piece)
                    area[gr][gc] += a
                    energy[gr][gc] += v * a
    return energy, area, lost


def dense_jacobian_integral(model, value_fn, src, n=600):
    """Independent dense midpoint integration of value*|det J|."""
    sx0, sx1, sy0, sy1 = src
    h = 1e-6
    total = 0.0
    for i in range(n):
        y = sy0 + (sy1 - sy0) * (i + 0.5) / n
        for j in range(n):
            x = sx0 + (sx1 - sx0) * (j + 0.5) / n
            fxe, fye = ref_distort(model, x + h, y)
            fxw, fyw = ref_distort(model, x - h, y)
            fxn, fyn = ref_distort(model, x, y + h)
            fxs, fys = ref_distort(model, x, y - h)
            j00 = (fxe - fxw) / (2 * h)
            j10 = (fye - fyw) / (2 * h)
            j01 = (fxn - fxs) / (2 * h)
            j11 = (fyn - fys) / (2 * h)
            total += value_fn(x, y) * abs(j00 * j11 - j01 * j10)
    return total * ((sx1 - sx0) / n) * ((sy1 - sy0) / n)


def grid_sum(g):
    return sum(sum(row) for row in g)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIdentityAndAffine(unittest.TestCase):
    def setUp(self):
        self.model = DistortionModel()  # identity: the affine degenerate case

    def test_identity_constant_aligned_grid(self):
        img = [[2.0] * 4 for _ in range(4)]
        res = forward_resample(self.model, img, (-1, 1, -1, 1),
                               (-1, 1, -1, 1), (4, 4))
        self.assertIsInstance(res, ForwardResult)
        for r in range(4):
            for c in range(4):
                self.assertAlmostEqual(res.area[r][c], 0.25, places=12)
                self.assertAlmostEqual(res.energy[r][c], 0.5, places=12)
        self.assertAlmostEqual(res.lost_energy, 0.0, places=12)
        self.assertAlmostEqual(res.total_energy, 8.0, places=12)
        self.assertFalse(res.budget_exhausted)

    def test_identity_matches_exact_clipping_reference(self):
        img = [[0.25 * (r + c) + 0.5 for c in range(3)] for r in range(3)]
        src = (-0.9, 0.9, -0.6, 0.6)
        dst = (-1.0, 1.0, -1.0, 1.0)
        shape = (5, 4)
        res = forward_resample(self.model, img, src, dst, shape, tol=1e-12)
        e_ref, a_ref, lost_ref = exact_affine_reference(
            self.model, img, src, dst, shape)
        for r in range(shape[0]):
            for c in range(shape[1]):
                self.assertAlmostEqual(res.energy[r][c], e_ref[r][c],
                                       places=10)
                self.assertAlmostEqual(res.area[r][c], a_ref[r][c],
                                       places=10)
        self.assertAlmostEqual(res.lost_energy, lost_ref, places=10)

    def test_identity_unaligned_grid_exact_split(self):
        # One source pixel of value 3 covering [-1,1]^2; target grid
        # 4x4 over the same window: each cell gets exactly 3/16 of 4.
        res = forward_resample(self.model, [[3.0]], (-1, 1, -1, 1),
                               (-1, 1, -1, 1), (4, 4))
        for r in range(4):
            for c in range(4):
                self.assertAlmostEqual(res.energy[r][c], 3.0 * 0.25,
                                       places=12)
        self.assertAlmostEqual(grid_sum(res.energy), 3.0 * 4.0, places=12)

    def test_energy_and_area_are_separate_outputs(self):
        img = [[0.0, 5.0], [0.0, 0.0]]
        res = forward_resample(self.model, img, (-1, 1, -1, 1),
                               (-1, 1, -1, 1), (2, 2))
        # Area is covered everywhere; energy only where value != 0.
        self.assertAlmostEqual(grid_sum(res.area), 4.0, places=12)
        self.assertAlmostEqual(grid_sum(res.energy), 5.0, places=12)
        self.assertAlmostEqual(res.energy[1][1], 5.0, places=12)
        self.assertAlmostEqual(res.energy[0][0], 0.0, places=12)


class TestOutOfWindow(unittest.TestCase):
    def test_partial_window_lost_energy(self):
        model = DistortionModel()
        img = [[1.0] * 2 for _ in range(2)]
        # Target window covers only x,y >= 0 quarter of the source.
        res = forward_resample(model, img, (-1, 1, -1, 1),
                               (0.0, 1.0, 0.0, 1.0), (2, 2))
        self.assertAlmostEqual(grid_sum(res.energy), 1.0, places=12)
        self.assertAlmostEqual(res.lost_energy, 3.0, places=12)
        self.assertAlmostEqual(res.lost_area, 3.0, places=12)
        self.assertAlmostEqual(grid_sum(res.energy) + res.lost_energy,
                               res.total_energy, places=12)

    def test_fully_outside_window(self):
        model = DistortionModel()
        res = forward_resample(model, [[2.0]], (-1, 1, -1, 1),
                               (5.0, 6.0, 5.0, 6.0), (3, 3))
        self.assertAlmostEqual(grid_sum(res.energy), 0.0, places=15)
        self.assertAlmostEqual(res.lost_energy, 8.0, places=12)


class TestDistortedConvergence(unittest.TestCase):
    def test_radial_distortion_matches_dense_jacobian(self):
        model = DistortionModel(k1=-0.1)
        value = lambda x, y: 1.0 + 0.3 * x - 0.2 * y
        img = [[value(-1 + (c + 0.5) / 4, 1 - (r + 0.5) / 4)
                for c in range(8)] for r in range(8)]
        res = forward_resample(model, img, (-1, 1, -1, 1),
                               (-1.5, 1.5, -1.5, 1.5), (16, 16), tol=1e-5)
        accounted = grid_sum(res.energy) + res.lost_energy
        expected = dense_jacobian_integral(model, value, (-1, 1, -1, 1))
        self.assertFalse(res.budget_exhausted)
        self.assertLessEqual(res.subdivision_error, 1e-5)
        self.assertAlmostEqual(accounted, expected, delta=2e-3)

    def test_tangential_distortion_matches_dense_jacobian(self):
        model = DistortionModel(k1=0.05, p1=0.002, p2=-0.001)
        img = [[0.8] * 6 for _ in range(6)]
        res = forward_resample(model, img, (-0.8, 0.8, -0.8, 0.8),
                               (-1.2, 1.2, -1.2, 1.2), (12, 12), tol=1e-5)
        accounted = grid_sum(res.energy) + res.lost_energy
        expected = dense_jacobian_integral(model, lambda x, y: 0.8,
                                           (-0.8, 0.8, -0.8, 0.8))
        self.assertFalse(res.budget_exhausted)
        self.assertAlmostEqual(accounted, expected, delta=2e-3)

    def test_refinement_converges(self):
        # Independent check: a finer independent sampling (smaller tol)
        # changes the result only within a shrinking tolerance.
        model = DistortionModel(k1=-0.1)
        img = [[1.0] * 4 for _ in range(4)]
        totals = []
        for tol in (1e-3, 1e-4, 1e-5):
            res = forward_resample(model, img, (-1, 1, -1, 1),
                                   (-1.5, 1.5, -1.5, 1.5), (8, 8), tol=tol)
            self.assertFalse(res.budget_exhausted)
            totals.append(grid_sum(res.energy) + res.lost_energy)
        diffs = [abs(totals[i + 1] - totals[i]) for i in range(2)]
        self.assertLess(diffs[1], diffs[0] * 0.5 + 1e-9)


class TestBudgetAndErrors(unittest.TestCase):
    def test_budget_exhaustion_flagged_not_hidden(self):
        model = DistortionModel(k1=-0.1)
        img = [[1.0] * 4 for _ in range(4)]
        res = forward_resample(model, img, (-1, 1, -1, 1),
                               (-1.5, 1.5, -1.5, 1.5), (8, 8),
                               tol=1e-9, max_quads=50)
        self.assertTrue(res.budget_exhausted)
        self.assertGreater(res.subdivision_error, 1e-9)
        self.assertIn("NOT", res.certificate)

    def test_near_zero_cell_area_rejected(self):
        model = DistortionModel()
        with self.assertRaises(ValueError):
            forward_resample(model, [[1.0]], (0.0, 1e-14, 0.0, 1e-14),
                             (-1, 1, -1, 1), (2, 2))

    def test_invalid_inputs(self):
        model = DistortionModel()
        good = dict(src_extent=(-1, 1, -1, 1), dst_extent=(-1, 1, -1, 1),
                    dst_shape=(2, 2))
        with self.assertRaises(ValueError):
            forward_resample(model, [[float("nan")]], **good)
        with self.assertRaises(TypeError):
            forward_resample(model, [[True]], **good)
        with self.assertRaises(ValueError):
            forward_resample(model, [[1.0, 2.0], [1.0]], **good)
        with self.assertRaises(TypeError):
            forward_resample(model, [], **good)
        with self.assertRaises(ValueError):
            forward_resample(model, [[1.0]], (1, 1, -1, 1),
                             (-1, 1, -1, 1), (2, 2))
        with self.assertRaises(TypeError):
            forward_resample(model, [[1.0]], (-1, 1, -1, 1),
                             (-1, 1, -1, 1), (2, True))
        with self.assertRaises(ValueError):
            forward_resample(model, [[1.0]], tol=-1.0, **good)
        with self.assertRaises(ValueError):
            forward_resample(model, [[1.0] * 65] * 65, **good)


if __name__ == "__main__":
    unittest.main()
