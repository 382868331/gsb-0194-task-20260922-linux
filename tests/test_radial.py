"""径向证书与 solve_radial 的测试（独立参考：测试内手写二分）。"""

import math
import unittest

from lensdistort import (
    DistortionModel,
    critical_radii,
    monotonic_segments,
    radial_uniqueness_certificate,
    solve_radial,
)
from lensdistort.errors import InvalidInputError


def _ref_roots_g(k1, k2, rho, r_max, n=40000):
    """独立参考：扫描+二分求 g(r)=rho 在 [0, r_max] 的全部根。"""

    def g(r):
        return r * (1.0 + k1 * r * r + k2 * r ** 4)

    roots = []
    prev_r, prev_v = 0.0, g(0.0) - rho
    for i in range(1, n + 1):
        r = r_max * i / n
        v = g(r) - rho
        if v == 0.0 or (prev_v < 0) != (v < 0):
            a, b = prev_r, r
            fa = g(a) - rho
            for _ in range(200):
                m = 0.5 * (a + b)
                fm = g(m) - rho
                if (fa <= 0) == (fm <= 0):
                    a, fa = m, fm
                else:
                    b = m
            roots.append(0.5 * (a + b))
        prev_r, prev_v = r, v
    out = []
    for r in roots:
        if not out or abs(r - out[-1]) > 1e-7:
            out.append(r)
    return out


class TestCriticalRadii(unittest.TestCase):
    def test_folding_model_analytic(self):
        # k1=-2, k2=1: g'(r)=1-6r²+5r⁴=0 → u=r² ∈ {0.2, 1} → r ∈ {1/√5, 1}
        m = DistortionModel(k1=-2.0, k2=1.0)
        crit = critical_radii(m)
        self.assertEqual(len(crit), 2)
        self.assertAlmostEqual(crit[0], 1.0 / math.sqrt(5.0), places=14)
        self.assertAlmostEqual(crit[1], 1.0, places=14)

    def test_monotone_model_no_critical(self):
        m = DistortionModel(k1=0.1, k2=0.02)
        self.assertEqual(critical_radii(m), ())

    def test_identity_no_critical(self):
        self.assertEqual(critical_radii(DistortionModel()), ())


class TestCertificate(unittest.TestCase):
    def test_injective_certificate(self):
        m = DistortionModel(k1=0.1, k2=0.02)
        cert = radial_uniqueness_certificate(m, 1.0)
        self.assertTrue(cert.is_globally_injective)
        self.assertEqual(cert.max_candidate_radii, 1)
        self.assertEqual(len(cert.segments), 1)
        self.assertEqual(cert.segments[0].direction, "increasing")
        self.assertAlmostEqual(cert.r_max, math.sqrt(2.0))

    def test_folding_certificate(self):
        m = DistortionModel(k1=-2.0, k2=1.0)
        cert = radial_uniqueness_certificate(m, 1.5)
        self.assertFalse(cert.is_globally_injective)
        self.assertEqual(cert.max_candidate_radii, 3)
        self.assertEqual(len(cert.segments), 3)
        self.assertEqual([s.direction for s in cert.segments],
                         ["increasing", "decreasing", "increasing"])
        self.assertAlmostEqual(cert.critical_radii[0], 1 / math.sqrt(5.0), places=14)
        self.assertAlmostEqual(cert.critical_radii[1], 1.0, places=14)

    def test_certificate_rejects_non_radial(self):
        with self.assertRaises(InvalidInputError):
            radial_uniqueness_certificate(DistortionModel(k1=0.1, p1=0.01), 1.0)

    def test_certificate_rejects_bad_radius(self):
        m = DistortionModel()
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(InvalidInputError):
                radial_uniqueness_certificate(m, bad)


class TestSolveRadial(unittest.TestCase):
    def test_folding_three_roots(self):
        # k1=-2, k2=1, R=1.5, r_max=R*sqrt(2)≈2.121
        m = DistortionModel(k1=-2.0, k2=1.0)
        R, r_max = 1.5, 1.5 * math.sqrt(2.0)
        rho = 0.3 * (1 - 2 * 0.09 + 0.09 ** 2)  # g(0.3) ≈ 0.24843
        roots = solve_radial(m, rho, r_max)
        self.assertEqual(len(roots), 3)
        ref = _ref_roots_g(-2.0, 1.0, rho, r_max)
        self.assertEqual(len(roots), len(ref))
        for got, exp in zip(roots, ref):
            self.assertAlmostEqual(got, exp, places=10)

    def test_monotone_unique_root(self):
        m = DistortionModel(k1=0.1, k2=0.02)
        cert = radial_uniqueness_certificate(m, 1.0)
        rho = 0.5 * (1 + 0.1 * 0.25 + 0.02 * 0.0625)
        roots = solve_radial(m, rho, cert.r_max)
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0], 0.5, places=12)

    def test_no_root_when_rho_out_of_range(self):
        m = DistortionModel(k1=0.5)
        cert = radial_uniqueness_certificate(m, 1.0)
        rho = cert.r_max * (1 + 0.5 * cert.r_max ** 2) + 0.1
        self.assertEqual(solve_radial(m, rho, cert.r_max), ())

    def test_zero_rho_includes_fold_radius(self):
        # k1=-2,k2=1: g(r)=r(1-r²)² → 根 r=0 与 r=1（二重/切触）
        m = DistortionModel(k1=-2.0, k2=1.0)
        roots = solve_radial(m, 0.0, 1.5 * math.sqrt(2.0))
        self.assertIn(0.0, roots)
        self.assertTrue(any(abs(r - 1.0) < 1e-9 for r in roots))


if __name__ == "__main__":
    unittest.main()
