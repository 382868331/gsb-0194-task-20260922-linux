"""内部输入校验助手：统一错误语义，不静默纠正。"""

from __future__ import annotations

import math

from .errors import InvalidInputError

MAX_POINTS = 100


def check_finite_float(name: str, value) -> float:
    """接受 int/float（拒绝 bool），拒绝 NaN/Inf，返回 float。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInputError(
            f"{name}: 期望有限实数，得到 {type(value).__name__}({value!r})"
        )
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        raise InvalidInputError(f"{name}: 必须为有限值，得到 {v!r}")
    return v


def check_int(name: str, value, minimum: int) -> int:
    """接受 int（拒绝 bool，bool 是 int 子类必须显式排除）。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidInputError(
            f"{name}: 期望整数，得到 {type(value).__name__}({value!r})"
        )
    if value < minimum:
        raise InvalidInputError(f"{name}: 必须 >= {minimum}，得到 {value}")
    return value


def check_point(name: str, point) -> tuple[float, float]:
    """校验单个 (x, y) 点。"""
    try:
        x_raw, y_raw = point
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(
            f"{name}: 期望 (x, y) 二元组，得到 {point!r}"
        ) from exc
    x = check_finite_float(f"{name}.x", x_raw)
    y = check_finite_float(f"{name}.y", y_raw)
    return (x, y)


def check_points(name: str, points) -> list[tuple[float, float]]:
    """校验点批次：最多 MAX_POINTS 个；全部校验通过才返回（不部分接受）。"""
    if isinstance(points, (str, bytes)):
        raise InvalidInputError(f"{name}: 期望点的序列，得到 {type(points).__name__}")
    try:
        items = list(points)
    except TypeError as exc:
        raise InvalidInputError(f"{name}: 期望可迭代的点序列") from exc
    if len(items) > MAX_POINTS:
        raise InvalidInputError(
            f"{name}: 最多 {MAX_POINTS} 个点，得到 {len(items)} 个"
        )
    return [check_point(f"{name}[{i}]", p) for i, p in enumerate(items)]


def check_box_radius(name: str, value) -> float:
    v = check_finite_float(name, value)
    if v <= 0.0:
        raise InvalidInputError(f"{name}: 搜索盒半径必须 > 0，得到 {v!r}")
    return v


def check_pos_float(name: str, value) -> float:
    v = check_finite_float(name, value)
    if v <= 0.0:
        raise InvalidInputError(f"{name}: 必须 > 0，得到 {v!r}")
    return v
