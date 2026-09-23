"""p1=p2=0 径向特例：g(r) = r·(1 + k1·r² + k2·r⁴) 的分段分析与证书。

g'(r) = 1 + 3·k1·r² + 5·k2·r⁴。令 u = r²，则 g'(r)=0 化为
5·k2·u² + 3·k1·u + 1 = 0，最多两个正根 ⇒ g 在 [0, r_max] 上
最多 3 个单调段。每个单调段上 g 是双射，故给定 ρ 每段至多一个
原像半径 ⇒ 逐段枚举是完备的（在径向模型假设下），不是"取首根"。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ._validation import check_box_radius, check_finite_float
from .errors import InvalidInputError
from .model import DistortionModel

_BISECT_ITERS = 200  # 二分迭代次数：区间缩小 2^-200，远超 float64 精度


def radial_g(model: DistortionModel, r: float) -> float:
    """g(r) = r·(1 + k1·r² + k2·r⁴)。"""
    r2 = r * r
    return r * (1.0 + model.k1 * r2 + model.k2 * r2 * r2)


def radial_gp(model: DistortionModel, r: float) -> float:
    """g'(r) = 1 + 3·k1·r² + 5·k2·r⁴。"""
    r2 = r * r
    return 1.0 + 3.0 * model.k1 * r2 + 5.0 * model.k2 * r2 * r2


def critical_radii(model: DistortionModel) -> tuple[float, ...]:
    """g'(r)=0 的正根（升序）。二重根不改变单调性，不计入。"""
    a = 5.0 * model.k2
    b = 3.0 * model.k1
    # 方程 a·u² + b·u + 1 = 0，u = r²
    roots: list[float] = []
    if a == 0.0:
        if b != 0.0:
            u = -1.0 / b
            if u > 0.0:
                roots.append(math.sqrt(u))
    else:
        disc = b * b - 4.0 * a
        if disc > 0.0:
            sq = math.sqrt(disc)
            for u in ((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)):
                if u > 0.0:
                    roots.append(math.sqrt(u))
    return tuple(sorted(roots))


@dataclass(frozen=True)
class RadialSegment:
    """g 的一个单调段 [r_lo, r_hi]，direction ∈ increasing/decreasing/flat。"""

    r_lo: float
    r_hi: float
    direction: str  # "increasing" | "decreasing" | "flat"
    g_lo: float  # g(r_lo)
    g_hi: float  # g(r_hi)


def monotonic_segments(model: DistortionModel, r_max: float) -> tuple[RadialSegment, ...]:
    """把 [0, r_max] 按 g' 的零点切成单调段。"""
    r_max = check_finite_float("r_max", r_max)
    if r_max <= 0.0:
        raise InvalidInputError(f"r_max: 必须 > 0，得到 {r_max!r}")
    bounds = [0.0]
    for rc in critical_radii(model):
        if 0.0 < rc < r_max:
            bounds.append(rc)
    bounds.append(r_max)
    segs: list[RadialSegment] = []
    for lo, hi in zip(bounds, bounds[1:]):
        mid = 0.5 * (lo + hi)
        gp = radial_gp(model, mid)
        if gp > 0.0:
            direction = "increasing"
        elif gp < 0.0:
            direction = "decreasing"
        else:
            direction = "flat"
        segs.append(
            RadialSegment(lo, hi, direction, radial_g(model, lo), radial_g(model, hi))
        )
    return tuple(segs)


@dataclass(frozen=True)
class RadialCertificate:
    """径向唯一性证书。is_globally_injective=True ⇒ 盒内逆映射唯一或证明无解。"""

    box_radius: float
    r_max: float
    critical_radii: tuple[float, ...]
    segments: tuple[RadialSegment, ...]
    is_globally_injective: bool
    max_candidate_radii: int
    statement: str


def radial_uniqueness_certificate(
    model: DistortionModel, box_radius: float
) -> RadialCertificate:
    """对 p1=p2=0 的模型给出盒 [-R,R]² 上的唯一性证书。"""
    if not isinstance(model, DistortionModel):
        raise InvalidInputError(
            f"model: 期望 DistortionModel，得到 {type(model).__name__}"
        )
    if not model.is_radial:
        raise InvalidInputError(
            "radial_uniqueness_certificate: 仅适用于 p1=p2=0 的径向模型，"
            f"得到 p1={model.p1!r}, p2={model.p2!r}"
        )
    R = check_box_radius("box_radius", box_radius)
    r_max = R * math.sqrt(2.0)
    crit = tuple(rc for rc in critical_radii(model) if 0.0 < rc < r_max)
    segs = monotonic_segments(model, r_max)
    injective = len(crit) == 0 and segs[0].direction == "increasing"
    if injective:
        stmt = (
            f"g'(r)=1+3·k1·r²+5·k2·r⁴ 在 (0, {r_max:.6g}) 内无零点，"
            "g 在 [0, r_max] 严格递增；径向映射保方向，故盒 [-R,R]² 内"
            "每个目标点至多一个原像：唯一或证明无解。"
        )
    else:
        stmt = (
            f"g 在 [0, r_max] 上有 {len(segs)} 个单调段"
            f"（临界点 r={tuple(round(c, 6) for c in crit)}）；"
            f"每个目标半径 ρ 在每段上至多一个原像半径，故盒内候选半径"
            f"不超过 {len(segs)} 个，逐段枚举即完备（径向模型假设下）。"
        )
    return RadialCertificate(
        box_radius=R,
        r_max=r_max,
        critical_radii=tuple(rc for rc in critical_radii(model)),
        segments=segs,
        is_globally_injective=injective,
        max_candidate_radii=len(segs),
        statement=stmt,
    )


def solve_radial(model: DistortionModel, rho: float, r_max: float) -> tuple[float, ...]:
    """返回 [0, r_max] 上所有满足 g(r)=ρ 的 r（去重、升序）。

    策略：
    - 每个单调段的开区间内的单根：二分（全精度）。
    - 临界点/端点处的根（含切触重根）：因 g 在重根附近浮点下溢，
      二分只能到 √eps 精度，故直接判定 |g(p)-ρ| 并取精确 p。
    逐段枚举是完备的（径向模型假设下），不取首根。
    ρ 必须 >= 0。
    """
    rho = check_finite_float("rho", rho)
    if rho < 0.0:
        raise InvalidInputError(f"rho: 必须 >= 0，得到 {rho!r}")
    r_max = check_finite_float("r_max", r_max)
    if r_max <= 0.0:
        raise InvalidInputError(f"r_max: 必须 > 0，得到 {r_max!r}")

    segs = monotonic_segments(model, r_max)
    out: list[float] = []

    # 1) 各单调段开区间内的单根：二分
    for seg in segs:
        lo_v, hi_v = sorted((seg.g_lo, seg.g_hi))
        if not (lo_v < rho < hi_v):
            continue
        a, b = seg.r_lo, seg.r_hi
        fa = radial_g(model, a) - rho
        for _ in range(_BISECT_ITERS):
            m = 0.5 * (a + b)
            fm = radial_g(model, m) - rho
            if (fa <= 0.0) == (fm <= 0.0):
                a, fa = m, fm
            else:
                b = m
        out.append(0.5 * (a + b))

    # 2) 特殊点（0、临界点、r_max）：精确判定是否恰为根
    crit = [rc for rc in critical_radii(model) if 0.0 < rc < r_max]
    for p in [0.0] + crit + [r_max]:
        gp = radial_g(model, p)
        if abs(gp - rho) <= 1e-10 * max(1.0, abs(rho), abs(gp)):
            out.append(p)

    # 3) 排序去重（相邻段共享端点/临界点可能重复）
    out.sort()
    deduped: list[float] = []
    for r in out:
        if r < 1e-12:
            r = 0.0
        if not deduped or r - deduped[-1] > 1e-9 * max(1.0, r):
            deduped.append(r)
    return tuple(deduped)
