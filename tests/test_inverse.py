"""逆映射测试：无畸变解析、折叠多根、盒外、中心、奇异点、Newton 路径。"""

import math
import unittest

from lensdistort import (
    STATUS_CONVERGED,
    STATUS_EXHAUSTED,
    STATUS_PROVEN_NO_SOLUTION,
    DistortionModel,
    InvalidInputError,
    distort_point,
    undistort_point,
    undistort_points,
)


def _ref_roots_g(k1, k2, rho, r_max, n=40000):
    """独立参考：扫描+二分求 g(r)=rho 的根（不调用被测库）。"""

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


class TestIdentity(unittest.TestCase):
    def test_identity_roundtrip_exact(self):
        m = DistortionModel()
        res = undistort_point(m, 0.3, -0.4, 1.0)
        self.assertEqual(res.status, STATUS_CONVERGED)
        self.assertEqual(len(res.candidates), 1)
        c = res.candidates[0]
        self.assertAlmostEqual(c.x, 0.3, places=12)
        self.assertAlmostEqual(c.y, -0.4, places=12)
        self.assertLessEqual(c.residual, 1e-12)


class TestFoldingRadialMultiRoot(unittest.TestCase):
    """构造折叠径向模型 k1=-2, k2=1：g(r)=r(1-r²)²，一个目标多个原像。"""

    def setUp(self):
        self.m = DistortionModel(k1=-2.0, k2=1.0)
        self.R = 1.5
        self.r_max = self.R * math.sqrt(2.0)

    def test_fold_target_gives_three_candidates(self):
        xd, yd = distort_point(self.m, 0.3, 0.0)
        res = undistort_point(self.m, xd, yd, self.R)
        self.assertEqual(res.status, STATUS_CONVERGED)
        self.assertEqual(len(res.candidates), 3)
        radii = sorted(math.hypot(c.x, c.y) for c in res.candidates)
        ref = _ref_roots_g(-2.0, 1.0, math.hypot(xd, yd), self.r_max)
        self.assertEqual(len(radii), len(ref))
        for got, exp in zip(radii, ref):
            self.assertAlmostEqual(got, exp, places=10)
        for c in res.candidates:
            self.assertLessEqual(c.residual, 1e-9)

    def test_newton_path_also_finds_multiple(self):
        # 强制 Newton 路径（method="newton"）：多起点应找到 ≥2 个根
        xd, yd = distort_point(self.m, 0.3, 0.0)
        res = undistort_point(self.m, xd, yd, self.R, method="newton")
        self.assertEqual(res.status, STATUS_CONVERGED)
        self.assertGreaterEqual(len(res.candidates), 2)
        self.assertIn("不证明完备", res.completeness)

    def test_center_target_origin_and_circle(self):
        # 目标原点：候选含 (0,0)，且 notes 指出 r=1 整圆也是原像
        res = undistort_point(self.m, 0.0, 0.0, self.R)
        self.assertEqual(res.status, STATUS_CONVERGED)
        self.assertTrue(any(abs(c.x) < 1e-15 and abs(c.y) < 1e-15
                            for c in res.candidates))
        self.assertTrue(any("整圆" in n for n in res.notes))

    def test_singular_fold_point(self):
        # 折叠点 r=1/√5 处 g'(r)=0（Jacobian 奇异）：仍能返回该候选
        r_s = 1.0 / math.sqrt(5.0)
        xd, yd = distort_point(self.m, r_s, 0.0)
        res = undistort_point(self.m, xd, yd, self.R)
        self.assertEqual(res.status, STATUS_CONVERGED)
        self.assertTrue(any(abs(math.hypot(c.x, c.y) - r_s) < 1e-9
                            for c in res.candidates))


class TestOutOfBoxAndProvenNoSolution(unittest.TestCase):
    def test_target_beyond_range_proven_no_solution(self):
        # 单调模型 k1=0.1, k2=0.02，目标半径超出 g(r_max) → proven_no_solution
        m = DistortionModel(k1=0.1, k2=0.02)
        R = 1.0
        r_max = R * math.sqrt(2.0)
        rho_max = r_max * (1 + 0.1 * r_max**2 + 0.02 * r_max**4)
        res = undistort_point(m, rho_max + 0.5, 0.0, R)
        self.assertEqual(res.status, STATUS_PROVEN_NO_SOLUTION)
        self.assertEqual(res.candidates, ())
        self.assertIn("证明无原像", res.completeness)

    def test_status_exhausted_distinct_from_proven(self):
        # 非径向模型 + 预算耗尽：status 必须是 exhausted，不能冒充 proven
        m = DistortionModel(k1=0.1, k2=0.02, p1=0.05, p2=-0.03)
        res = undistort_point(m, 5.0, 5.0, 1.0,
                              max_starts=2, max_iterations=3)
        self.assertEqual(res.status, STATUS_EXHAUSTED)
        self.assertEqual(res.candidates, ())
        self.assertNotEqual(res.status, STATUS_PROVEN_NO_SOLUTION)


class TestTangentialModelNewton(unittest.TestCase):
    def test_roundtrip_supplement(self):
        # 往返一致性仅作补充验证
        m = DistortionModel(k1=0.1, k2=0.02, p1=0.05, p2=-0.03)
        x0, y0 = 0.4, 0.2
        xd, yd = distort_point(m, x0, y0)
        res = undistort_point(m, xd, yd, 1.0)
        self.assertEqual(res.status, STATUS_CONVERGED)
        self.assertTrue(res.candidates)
        best = min(res.candidates,
                   key=lambda c: (c.x - x0) ** 2 + (c.y - y0) ** 2)
        self.assertAlmostEqual(best.x, x0, places=9)
        self.assertAlmostEqual(best.y, y0, places=9)
        self.assertLessEqual(best.residual, 1e-9)


class TestBatchAndLimits(unittest.TestCase):
    def test_batch_100_points_ok(self):
        m = DistortionModel(k1=0.1)
        pts = [(0.01 * i, 0.01 * i) for i in range(100)]
        res = undistort_points(m, pts, 1.0)
        self.assertEqual(len(res), 100)
        for r in res:
            self.assertEqual(r.status, STATUS_CONVERGED)

    def test_batch_rejects_101_points(self):
        pts = [(0.0, 0.0)] * 101
        with self.assertRaises(InvalidInputError):
            undistort_points(DistortionModel(), pts, 1.0)

    def test_batch_atomic_no_partial_results(self):
        # 批次中任一点非法 → 整体拒绝，不返回部分结果
        m = DistortionModel()
        pts = [(0.1, 0.2)] * 5 + [(float("nan"), 0.0)]
        with self.assertRaises(InvalidInputError):
            undistort_points(m, pts, 1.0)


class TestInputValidation(unittest.TestCase):
    def setUp(self):
        self.m = DistortionModel()

    def test_nan_inf_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(InvalidInputError):
                undistort_point(self.m, bad, 0.0, 1.0)
            with self.assertRaises(InvalidInputError):
                undistort_point(self.m, 0.0, bad, 1.0)

    def test_bool_rejected_for_coefficients(self):
        with self.assertRaises(InvalidInputError):
            DistortionModel(k1=True)
        with self.assertRaises(InvalidInputError):
            DistortionModel(p2=False)

    def test_bool_rejected_for_int_params(self):
        with self.assertRaises(InvalidInputError):
            undistort_point(self.m, 0.0, 0.0, 1.0, max_starts=True)
        with self.assertRaises(InvalidInputError):
            undistort_point(self.m, 0.0, 0.0, 1.0, max_iterations=False)

    def test_bad_box_radius(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(InvalidInputError):
                undistort_point(self.m, 0.0, 0.0, bad)

    def test_error_message_locates_param(self):
        try:
            undistort_point(self.m, float("nan"), 0.0, 1.0)
        except InvalidInputError as e:
            self.assertIn("xd", str(e))
        else:
            self.fail("expected InvalidInputError")

    def test_radial_method_rejects_tangential_model(self):
        m = DistortionModel(k1=0.1, p1=0.01)
        with self.assertRaises(InvalidInputError):
            undistort_point(m, 0.1, 0.0, 1.0, method="radial")


class TestEdgeCases(unittest.TestCase):
    def test_center_target(self):
        # 无畸变模型：目标原点 → 唯一候选 (0,0)
        m = DistortionModel()
        res = undistort_point(m, 0.0, 0.0, 1.0)
        self.assertEqual(res.status, STATUS_CONVERGED)
        self.assertEqual(len(res.candidates), 1)
        self.assertEqual((res.candidates[0].x, res.candidates[0].y), (0.0, 0.0))

    def test_out_of_box_target_proven_no_solution(self):
        # 目标点在搜索盒外（径向单调模型）：proven_no_solution
        m = DistortionModel(k1=0.1)
        R = 1.0
        r_max = R * math.sqrt(2.0)
        rho_max = r_max * (1 + 0.1 * r_max**2)
        res = undistort_point(m, rho_max + 0.1, 0.0, R)
        self.assertEqual(res.status, STATUS_PROVEN_NO_SOLUTION)
        self.assertEqual(res.candidates, ())

    def test_boundary_point_on_box_edge(self):
        # 目标恰好是盒边界点的像 → 应找到该候选
        m = DistortionModel(k1=0.1)
        x0 = 1.0  # 盒边界上的点
        xd, yd = distort_point(m, x0, 0.0)
        res = undistort_point(m, xd, yd, 1.0)
        self.assertEqual(res.status, STATUS_CONVERGED)
        self.assertTrue(any(abs(c.x - 1.0) < 1e-9 for c in res.candidates))


if __name__ == "__main__":
    unittest.main()
