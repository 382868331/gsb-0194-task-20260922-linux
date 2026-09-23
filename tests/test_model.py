"""正向模型与解析 Jacobian 的测试。

独立参考：期望值由测试内手算公式 / 有限差分给出，不调用被测核心。
"""

import math
import unittest

from lensdistort import DistortionModel, distort_point, jacobian


def _fd_jacobian(model, x, y, h=1e-6):
    """独立参考：中心有限差分 Jacobian（直接按定义数值微分 distort）。"""
    # 注意：这里调用 distort_point 仅为生成函数值；
    # 参考方法（有限差分）独立于被测的解析 jacobian 实现。
    from lensdistort import distort_point as f

    fxph, fxmh = f(model, x + h, y)[0], f(model, x - h, y)[0]
    fyph, fymh = f(model, x, y + h)[0], f(model, x, y - h)[0]
    fxph2, fxmh2 = f(model, x + h, y)[1], f(model, x - h, y)[1]
    fyph2, fymh2 = f(model, x, y + h)[1], f(model, x, y - h)[1]
    return (
        ((fxph - fxmh) / (2 * h), (fyph - fymh) / (2 * h)),
        ((fxph2 - fxmh2) / (2 * h), (fyph2 - fymh2) / (2 * h)),
    )


class TestDistortForward(unittest.TestCase):
    def test_identity_model(self):
        m = DistortionModel()
        self.assertEqual(distort_point(m, 0.3, -0.4), (0.3, -0.4))
        self.assertEqual(distort_point(m, 0.0, 0.0), (0.0, 0.0))

    def test_hand_computed_radial(self):
        # k1=0.1, k2=0.01, p1=p2=0, 点 (0.5, -0.25)
        m = DistortionModel(k1=0.1, k2=0.01)
        x, y = 0.5, -0.25
        r2 = x * x + y * y  # 0.3125
        s = 1.0 + 0.1 * r2 + 0.01 * r2 * r2
        exp = (x * s, y * s)
        got = distort_point(m, x, y)
        self.assertAlmostEqual(got[0], exp[0], places=15)
        self.assertAlmostEqual(got[1], exp[1], places=15)

    def test_hand_computed_tangential(self):
        m = DistortionModel(k1=0.1, k2=0.01, p1=0.02, p2=-0.03)
        x, y = 0.5, -0.25
        r2 = x * x + y * y
        s = 1.0 + 0.1 * r2 + 0.01 * r2 * r2
        xd = x * s + 2 * 0.02 * x * y + (-0.03) * (r2 + 2 * x * x)
        yd = y * s + 0.02 * (r2 + 2 * y * y) + 2 * (-0.03) * x * y
        got = distort_point(m, x, y)
        self.assertAlmostEqual(got[0], xd, places=15)
        self.assertAlmostEqual(got[1], yd, places=15)

    def test_symmetry_of_tangential_terms(self):
        # 交换 x,y 与 p1,p2 的角色应对称：distort 的 x' 与 y' 公式对称
        m = DistortionModel(k1=0.0, k2=0.0, p1=0.05, p2=-0.02)
        xd, yd = distort_point(m, 0.3, 0.4)
        # 手算：r2=0.25, s=1
        # xd = x + 2 p1 x y + p2 (r2 + 2x^2) = 0.3 + 0.012 + (-0.02)*0.43
        self.assertAlmostEqual(xd, 0.3 + 2*0.05*0.3*0.4 - 0.02*0.43, places=15)
        # yd = y + p1 (r2 + 2 y^2) + 2 p2 x y
        self.assertAlmostEqual(yd, 0.4 + 0.05*0.57 + 2*(-0.02)*0.3*0.4, places=15)


class TestJacobian(unittest.TestCase):
    def test_jacobian_vs_finite_difference(self):
        cases = [
            (DistortionModel(), (0.3, -0.4)),
            (DistortionModel(k1=0.1, k2=0.01), (0.5, -0.25)),
            (DistortionModel(k1=-2.0, k2=1.0), (0.6, 0.0)),
            (DistortionModel(k1=0.1, k2=0.01, p1=0.02, p2=-0.03), (0.3, 0.4)),
            (DistortionModel(k1=0.0, k2=0.0, p1=0.05, p2=-0.02), (-0.2, 0.35)),
        ]
        for m, (x, y) in cases:
            with self.subTest(m=m, pt=(x, y)):
                (a, b), (c, d) = jacobian(m, x, y)
                (a2, b2), (c2, d2) = _fd_jacobian(m, x, y)
                for got, exp, name in [
                    (a, a2, "dxd/dx"), (b, b2, "dxd/dy"),
                    (c, c2, "dyd/dx"), (d, d2, "dyd/dy"),
                ]:
                    self.assertAlmostEqual(
                        got, exp, delta=1e-5 * max(1.0, abs(exp)),
                        msg=f"{name}: analytic={got}, fd={exp}",
                    )

    def test_jacobian_at_origin_is_identity(self):
        m = DistortionModel(k1=0.3, k2=-0.1, p1=0.05, p2=0.02)
        (a, b), (c, d) = jacobian(m, 0.0, 0.0)
        self.assertEqual((a, b, c, d), (1.0, 0.0, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
