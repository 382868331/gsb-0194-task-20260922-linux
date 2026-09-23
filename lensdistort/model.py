"""正向畸变模型与解析 Jacobian。

模型（归一化平面坐标，r² = x² + y²）：

    s = 1 + k1·r² + k2·r⁴
    x' = x·s + 2·p1·x·y + p2·(r² + 2·x²)
    y' = y·s + p1·(r² + 2·y²) + 2·p2·x·y
"""

from __future__ import annotations

from dataclasses import dataclass

from ._validation import check_finite_float, check_point, check_points


@dataclass(frozen=True)
class DistortionModel:
    """畸变系数。构造时即校验：拒绝 NaN/Inf 与 bool。"""

    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    def __post_init__(self) -> None:
        for name in ("k1", "k2", "p1", "p2"):
            object.__setattr__(
                self, name, check_finite_float(name, getattr(self, name))
            )

    @property
    def is_radial(self) -> bool:
        """p1 == p2 == 0 时为纯径向模型，可走径向证书路径。"""
        return self.p1 == 0.0 and self.p2 == 0.0


def distort_point(model: DistortionModel, x, y) -> tuple[float, float]:
    """正向映射 (x, y) -> (x', y')。"""
    if not isinstance(model, DistortionModel):
        raise TypeError(f"model: 期望 DistortionModel，得到 {type(model).__name__}")
    x = check_finite_float("x", x)
    y = check_finite_float("y", y)
    r2 = x * x + y * y
    s = 1.0 + model.k1 * r2 + model.k2 * r2 * r2
    xd = x * s + 2.0 * model.p1 * x * y + model.p2 * (r2 + 2.0 * x * x)
    yd = y * s + model.p1 * (r2 + 2.0 * y * y) + 2.0 * model.p2 * x * y
    return (xd, yd)


def distort_points(
    model: DistortionModel, points
) -> list[tuple[float, float]]:
    """批量正向映射。最多 100 点；任一非法则整体拒绝（不部分返回）。"""
    pts = check_points("points", points)
    return [distort_point(model, px, py) for px, py in pts]


def jacobian(model: DistortionModel, x, y) -> tuple[tuple[float, float], tuple[float, float]]:
    """解析 Jacobian：((∂x'/∂x, ∂x'/∂y), (∂y'/∂x, ∂y'/∂y))。"""
    if not isinstance(model, DistortionModel):
        raise TypeError(f"model: 期望 DistortionModel，得到 {type(model).__name__}")
    x = check_finite_float("x", x)
    y = check_finite_float("y", y)
    r2 = x * x + y * y
    s = 1.0 + model.k1 * r2 + model.k2 * r2 * r2
    t = model.k1 + 2.0 * model.k2 * r2  # ds/dr² 的 2 倍关系见下
    # ∂s/∂x = 2x·(k1 + 2·k2·r²)，∂s/∂y = 2y·(k1 + 2·k2·r²)
    a = s + 2.0 * x * x * t + 2.0 * model.p1 * y + 6.0 * model.p2 * x
    b = 2.0 * x * y * t + 2.0 * model.p1 * x + 2.0 * model.p2 * y
    c = 2.0 * x * y * t + 2.0 * model.p1 * x + 2.0 * model.p2 * y
    d = s + 2.0 * y * y * t + 6.0 * model.p1 * y + 2.0 * model.p2 * x
    return ((a, b), (c, d))
