#!/usr/bin/env python3
"""demo：展示多解（折叠径向）与一次实际触发的输入拒绝。离线、确定性、<8s。"""

import math

from lensdistort import (
    DistortionModel,
    InvalidInputError,
    distort_point,
    undistort_point,
)


def main() -> None:
    print("=== 透镜畸变逆映射 demo（离线、确定性） ===\n")

    # ---- 1) 折叠径向模型：一个目标 → 多个原像（多解展示）----
    m = DistortionModel(k1=-2.0, k2=1.0)  # g(r)=r(1-r²)²，折叠
    R = 1.5
    x0 = 0.3
    xd, yd = distort_point(m, x0, 0.0)
    print(f"[1] 折叠径向模型 k1=-2, k2=1；目标 = distort({x0}, 0)")
    print(f"    目标点: ({xd:.12f}, {yd:.12f})")
    res = undistort_point(m, xd, yd, R)
    print(f"    status: {res.status}")
    print(f"    候选数: {len(res.candidates)}")
    for i, c in enumerate(res.candidates):
        r = math.hypot(c.x, c.y)
        print(f"      候选{i}: ({c.x:.12f}, {c.y:.12f})  r={r:.12f}  残差={c.residual:.2e}")
    print(f"    完备性说明: {res.completeness}")
    print()

    # ---- 2) 径向唯一性证书（单调模型）----
    m2 = DistortionModel(k1=0.1, k2=0.02)
    from lensdistort import radial_uniqueness_certificate
    cert = radial_uniqueness_certificate(m2, 1.0)
    print(f"[2] 径向唯一性证书（单调模型 k1=0.1, k2=0.02）")
    print(f"    is_globally_injective: {cert.is_globally_injective}")
    print(f"    单调段数: {len(cert.segments)}")
    print(f"    最大候选半径数: {cert.max_candidate_radii}")
    print(f"    陈述: {cert.statement}")
    print()

    # ---- 3) 盒外目标：证明无解（非预算耗尽）----
    m3 = DistortionModel(k1=0.1, k2=0.02)
    cert3 = radial_uniqueness_certificate(m3, 1.0)
    rho_max = cert3.r_max * (1 + 0.1 * cert3.r_max**2 + 0.02 * cert3.r_max**4)
    res3 = undistort_point(m3, rho_max + 0.5, 0.0, 1.0)
    print(f"[3] 盒外目标（径向单调模型）：status={res3.status}")
    print(f"    候选数: {len(res3.candidates)}")
    print(f"    说明: {res3.completeness}")
    print()

    # ---- 4) 实际触发的拒绝：NaN 输入 ----
    print("[4] 触发输入拒绝：undistort_point(m, float('nan'), 0.0, 1.0)")
    try:
        undistort_point(m, float("nan"), 0.0, 1.0)
    except InvalidInputError as e:
        print(f"    捕获 InvalidInputError: {e}")
    print()

    # ---- 5) 切向畸变模型（Newton 路径）：往返一致性补充 ----
    m4 = DistortionModel(k1=0.1, k2=0.02, p1=0.05, p2=-0.03)
    x0, y0 = 0.4, 0.2
    xd4, yd4 = distort_point(m4, x0, y0)
    res4 = undistort_point(m4, xd4, yd4, 1.0)
    print(f"[4b] 切向模型（Newton）：目标=distort({x0},{y0})，候选数={len(res4.candidates)}")
    for i, c in enumerate(res4.candidates):
        print(f"      候选{i}: ({c.x:.12f}, {c.y:.12f})  残差={c.residual:.2e}")
    print(f"    说明: {res4.completeness}")


if __name__ == "__main__":
    main()
