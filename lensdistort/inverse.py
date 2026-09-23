"""逆映射：多起点阻尼 Newton + 残差检验 + 去重；径向特例走证书路径。

状态语义（UndistortResult.status）：
- "converged": 找到 ≥1 个通过残差检验的候选。
- "exhausted": Newton 预算耗尽仍未找到合格候选（不证明无解）。
- "proven_no_solution": 径向证书证明盒内无原像（仅径向模型可达）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ._validation import (
    check_box_radius,
    check_finite_float,
    check_int,
    check_points,
    check_pos_float,
)
from .errors import InvalidInputError
from .model import DistortionModel, distort_point, jacobian
from .radial import radial_uniqueness_certificate, solve_radial

STATUS_CONVERGED = "converged"
STATUS_EXHAUSTED = "exhausted"
STATUS_PROVEN_NO_SOLUTION = "proven_no_solution"

_COMPLETENESS_NUMERIC = (
    "数值多起点搜索：不证明完备性，盒内可能存在未找到的根。"
)
_COMPLETENESS_RADIAL = (
    "径向证书：在 p1=p2=0 模型与浮点精度意义下，候选集完备。"
)


@dataclass(frozen=True)
class Candidate:
    """一个通过残差检验的逆解候选。residual = max(|F(z)-target|_∞)。"""

    x: float
    y: float
    residual: float


@dataclass(frozen=True)
class UndistortResult:
    """undistort 结果。status ∈ {converged, exhausted, proven_no_solution}。"""

    target: tuple[float, float]
    box_radius: float
    status: str
    candidates: tuple["Candidate", ...]
    completeness: str
    notes: tuple[str, ...] = ()


def _F(model, x, y, tx, ty):
    xd, yd = distort_point(model, x, y)
    return xd - tx, yd - ty


def _starts(tx, ty, R):
    """确定性起点集：裁剪目标点、原点、5×5 网格。"""
    starts: list[tuple[float, float]] = []

    def _add(p):
        if p not in starts:
            starts.append(p)

    _add((min(max(tx, -R), R), min(max(ty, -R), R)))  # 裁剪目标点
    _add((0.0, 0.0))  # 中心
    for i in range(5):
        for j in range(5):
            _add((-R + 2.0 * R * i / 4.0, -R + 2.0 * R * j / 4.0))
    return starts


def _newton(model, tx, ty, x0, y0, R, max_iter, tol):
    """阻尼 Newton：返回 (x, y, 残差, 是否收敛)。奇异 Jacobian 正则化。"""
    x, y = x0, y0
    fx, fy = _F(model, x, y, tx, ty)
    norm = max(abs(fx), abs(fy))
    for _ in range(max_iter):
        if norm <= tol:
            return x, y, norm, True
        (a, b), (c, d) = jacobian(model, x, y)
        det = a * d - b * c
        scale = max(abs(a), abs(b), abs(c), abs(d), 1.0)
        if abs(det) <= 1e-14 * scale * scale:
            lam = 1e-10 * scale  # Levenberg 式正则化：奇异点附近不崩溃
            a += lam
            d += lam
            det = a * d - b * c
            if abs(det) <= 1e-16 * scale * scale:
                return x, y, norm, False
        dx = (d * fx - b * fy) / det
        dy = (-c * fx + a * fy) / det
        t = 1.0
        accepted = False
        nx = ny = 0.0
        for _ls in range(40):
            nx, ny = x - t * dx, y - t * dy
            if abs(nx) <= R and abs(ny) <= R:
                nfx, nfy = _F(model, nx, ny, tx, ty)
                if max(abs(nfx), abs(nfy)) < norm:
                    accepted = True
                    break
            t *= 0.5
        if not accepted:
            return x, y, norm, False
        x, y = nx, ny
        fx, fy = _F(model, x, y, tx, ty)
        norm = max(abs(fx), abs(fy))
    return x, y, norm, norm <= tol


def _dedup(cands, tol):
    """按残差升序贪心去重：距离 ≤ tol 视为同一根，保留残差最小者。"""
    out: list = []
    for cand in sorted(cands, key=lambda c: c.residual):
        if all(max(abs(cand.x - o.x), abs(cand.y - o.y)) > tol for o in out):
            out.append(cand)
    return tuple(out)


def undistort_point(
    model: DistortionModel,
    xd, yd, box_radius,
    *,
    method: str = "auto",
    max_starts: int = 64,
    max_iterations: int = 60,
    tol: float = 1e-12,
    accept_tol: float = 1e-9,
    dedup_tol: float = 1e-7,
) -> "UndistortResult":
    """逆映射：返回盒 [-R,R]² 内全部通过残差检验且去重的候选。

    method: "auto"（径向模型走证书，否则 Newton）/ "newton" / "radial"。
    "radial" 仅当 p1=p2=0 合法。Newton 路径完备性不被证明；
    径向路径在 p1=p2=0 下按分段单调性给出完备候选。
    """
    if not isinstance(model, DistortionModel):
        raise InvalidInputError(
            f"model: 期望 DistortionModel，得到 {type(model).__name__}"
        )
    tx = check_finite_float("xd", xd)
    ty = check_finite_float("yd", yd)
    R = check_box_radius("box_radius", box_radius)
    if method not in ("auto", "newton", "radial"):
        raise InvalidInputError(
            f"method: 期望 'auto'/'newton'/'radial'，得到 {method!r}"
        )
    max_starts = check_int("max_starts", max_starts, 1)
    max_iterations = check_int("max_iterations", max_iterations, 1)
    tol = check_pos_float("tol", tol)
    accept_tol = check_pos_float("accept_tol", accept_tol)
    dedup_tol = check_pos_float("dedup_tol", dedup_tol)

    if model.is_radial and method in ("auto", "radial"):
        return _undistort_radial(model, tx, ty, R, accept_tol, dedup_tol)
    if method == "radial" and not model.is_radial:
        raise InvalidInputError(
            "method='radial' 仅适用于 p1=p2=0 的模型；"
            f"得到 p1={model.p1!r}, p2={model.p2!r}"
        )
    return _undistort_newton(
        model, tx, ty, R, max_starts, max_iterations, tol, accept_tol, dedup_tol
    )


def _undistort_radial(model, tx, ty, R, accept_tol, dedup_tol):
    """p1=p2=0：用径向证书求解，候选集完备（该模型假设下）。"""
    cert = radial_uniqueness_certificate(model, R)
    rho = math.hypot(tx, ty)
    notes: list[str] = []
    if rho == 0.0:
        # 原点：r=0 恒为解；g(r)=0 的正根对应整圆原像（连续统）
        roots = solve_radial(model, 0.0, cert.r_max)
        pos = [r for r in roots if r > 1e-12]
        cands = [Candidate(0.0, 0.0, 0.0)]
        if pos:
            notes.append(
                f"除原点外，半径 {tuple(round(r,12) for r in pos)} 的整圆"
                "均映射到原点（g(r)=0），逆解不唯一（连续统）。"
            )
        return UndistortResult(
            target=(tx, ty),
            box_radius=R,
            status=STATUS_CONVERGED,
            candidates=(Candidate(0.0, 0.0, 0.0),),
            completeness=_COMPLETENESS_RADIAL
            + (" 注意：原点存在整圆原像。" if pos else ""),
            notes=tuple(notes),
        )
    ux, uy = tx / rho, ty / rho
    cands: list[Candidate] = []
    for r in solve_radial(model, rho, cert.r_max):
        x, y = ux * r, uy * r
        if abs(x) <= R * (1.0 + 1e-12) and abs(y) <= R * (1.0 + 1e-12):
            xd2, yd2 = distort_point(model, x, y)
            res = max(abs(xd2 - tx), abs(yd2 - ty))
            if res <= accept_tol:
                cands.append(Candidate(x, y, res))
    cands = list(_dedup(cands, dedup_tol))
    if cands:
        status = STATUS_CONVERGED
        completeness = _COMPLETENESS_RADIAL
    else:
        status = STATUS_PROVEN_NO_SOLUTION
        completeness = "径向证书：g 值域不含该 ρ，盒内证明无原像。"
    return UndistortResult(
        target=(tx, ty),
        box_radius=R,
        status=status,
        candidates=tuple(cands),
        completeness=completeness,
        notes=tuple(notes),
    )


def _undistort_newton(model, tx, ty, R, max_starts, max_iterations, tol,
                      accept_tol, dedup_tol):
    starts = _starts(tx, ty, R)[:max_starts]
    cands: list[Candidate] = []
    for sx, sy in starts:
        x, y, res, ok = _newton(model, tx, ty, sx, sy, R, max_iterations, tol)
        if not ok and res > accept_tol:
            continue
        if abs(x) > R * (1.0 + 1e-12) or abs(y) > R * (1.0 + 1e-12):
            continue
        if res > accept_tol:
            continue
        cands.append(Candidate(x, y, res))
    deduped = _dedup(cands, dedup_tol)
    status = STATUS_CONVERGED if deduped else STATUS_EXHAUSTED
    return UndistortResult(
        target=(tx, ty),
        box_radius=R,
        status=status,
        candidates=tuple(deduped),
        completeness=_COMPLETENESS_NUMERIC,
        notes=(),
    )


def undistort_points(model, points, box_radius, **kw) -> list[UndistortResult]:
    """批量逆映射。先整体校验（失败则不产生任何部分结果），再逐点求解。"""
    pts = check_points("points", points)
    return [undistort_point(model, p[0], p[1], box_radius, **kw) for p in pts]
