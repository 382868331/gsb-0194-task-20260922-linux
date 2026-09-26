"""Exact polygon clipping and signed area (Sutherland-Hodgman).

These primitives are used both for the exact affine/identity reference
path and for distributing the leaves of the adaptive forward
subdivision.  Nothing here performs point sampling: a source pixel is
an area-carrying constant cell and overlap area is computed exactly
for the polygon used to represent it.
"""

__all__ = ["clip_polygon_to_box", "polygon_area"]


def _inside(point, edge):
    x, y = point
    side, value = edge
    if side == "left":
        return x >= value
    if side == "right":
        return x <= value
    if side == "bottom":
        return y >= value
    return y <= value  # "top"


def _intersect(a, b, edge):
    side, value = edge
    ax, ay = a
    bx, by = b
    if side in ("left", "right"):
        t = (value - ax) / (bx - ax)
        return value, ay + t * (by - ay)
    t = (value - ay) / (by - ay)
    return ax + t * (bx - ax), value


def clip_polygon_to_box(polygon, xmin, ymin, xmax, ymax):
    """Clip a convex polygon against an axis-aligned closed box.

    The input is a list of ``(x, y)`` vertices in boundary order.
    Returns the clipped polygon (possibly empty).  Sutherland-Hodgman
    with four half-planes.
    """
    edges = (("left", xmin), ("right", xmax),
             ("bottom", ymin), ("top", ymax))
    output = [(float(x), float(y)) for (x, y) in polygon]
    for edge in edges:
        if not output:
            return []
        inp = output
        output = []
        s = inp[-1]
        s_in = _inside(s, edge)
        for e in inp:
            e_in = _inside(e, edge)
            if e_in:
                if not s_in:
                    output.append(_intersect(s, e, edge))
                output.append(e)
            elif s_in:
                output.append(_intersect(s, e, edge))
            s, s_in = e, e_in
    return output


def polygon_area(polygon):
    """Absolute shoelace area of a simple polygon (0 for < 3 verts)."""
    n = len(polygon)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return 0.5 * abs(total)
