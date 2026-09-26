"""Forward (splatting) resampling of a small grayscale image.

A source pixel is treated as a constant-valued *area* cell, not a
point sample.  Each cell is pushed through the distortion map; its
curved image is approximated by adaptive quadtree subdivision of the
source cell, and every leaf quad is clipped exactly (Sutherland-
Hodgman) against the target grid cells it overlaps.  Per-target-cell
energy (value x overlap area) and covered area are returned as two
separate grids -- no point sampling, no after-the-fact normalization
factor.

The caller must have verified that the source region is single-valued
under the map (no folding); the routine does not prove that.  It does
report a subdivision error metric, the energy that fell outside the
target window, and an honest budget-exhausted flag when the quad
budget or depth limit stopped refinement before the tolerance was met.
"""

import math

from .numeric import finite_real, checked_int
from .clip import clip_polygon_to_box, polygon_area

__all__ = ["ForwardResult", "forward_resample"]

# Interface bounds: this entry point is for *small* grayscale images.
MAX_IMAGE_SIDE = 64
MAX_GRID_SIDE = 512
# A source pixel cell smaller than this is rejected as degenerate.
MIN_CELL_AREA = 1e-24


class ForwardResult:
    """Outcome of forward_resample.

    Attributes:
        energy            -- rows (dst_h x dst_w) of received energy
        area              -- rows (dst_h x dst_w) of covered area
        lost_energy       -- source energy mapped outside the dst window
        lost_area         -- source area mapped outside the dst window
        total_energy      -- sum of source value x cell area (input)
        subdivision_error -- largest boundary deviation accepted by any
                             leaf (normalized-plane units)
        budget_exhausted  -- True when the quad budget or depth limit
                             stopped refinement before tol was met; the
                             result is then NOT claimed to satisfy it
        quads_used        -- number of leaf quads distributed
        certificate       -- human-readable summary
    """

    __slots__ = ("energy", "area", "lost_energy", "lost_area",
                 "total_energy", "subdivision_error", "budget_exhausted",
                 "quads_used", "certificate")

    def __init__(self, energy, area, lost_energy, lost_area, total_energy,
                 subdivision_error, budget_exhausted, quads_used,
                 certificate):
        self.energy = energy
        self.area = area
        self.lost_energy = lost_energy
        self.lost_area = lost_area
        self.total_energy = total_energy
        self.subdivision_error = subdivision_error
        self.budget_exhausted = budget_exhausted
        self.quads_used = quads_used
        self.certificate = certificate

    def __repr__(self):
        return (f"ForwardResult(quads={self.quads_used}, "
                f"lost_energy={self.lost_energy:.6g}, "
                f"subdivision_error={self.subdivision_error:.3e}, "
                f"budget_exhausted={self.budget_exhausted})")


def _check_image(pixels):
    if not isinstance(pixels, (list, tuple)) or len(pixels) == 0:
        raise TypeError("pixels: expected a non-empty list of rows")
    if len(pixels) > MAX_IMAGE_SIDE:
        raise ValueError(
            f"pixels: {len(pixels)} rows exceed the small-image limit "
            f"of {MAX_IMAGE_SIDE}")
    width = None
    cleaned = []
    for i, row in enumerate(pixels):
        if not isinstance(row, (list, tuple)) or len(row) == 0:
            raise TypeError(f"pixels[{i}]: expected a non-empty row")
        if width is None:
            width = len(row)
            if width > MAX_IMAGE_SIDE:
                raise ValueError(
                    f"pixels: row width {width} exceeds the small-image "
                    f"limit of {MAX_IMAGE_SIDE}")
        elif len(row) != width:
            raise ValueError(
                f"pixels[{i}]: ragged image, row has {len(row)} value(s), "
                f"expected {width}")
        cleaned.append([finite_real(v, f"pixels[{i}][{j}]")
                        for j, v in enumerate(row)])
    return cleaned, len(cleaned), width


def _check_extent(extent, name):
    if not isinstance(extent, (list, tuple)) or len(extent) != 4:
        raise TypeError(f"{name}: expected (x0, x1, y0, y1)")
    x0, x1, y0, y1 = (finite_real(v, f"{name}[{k}]")
                      for k, v in enumerate(extent))
    if not x1 > x0:
        raise ValueError(f"{name}: x1 must be > x0, got ({x0!r}, {x1!r})")
    if not y1 > y0:
        raise ValueError(f"{name}: y1 must be > y0, got ({y0!r}, {y1!r})")
    return x0, x1, y0, y1


class _Distributor:
    """Accumulates leaf-quad energy/area into the target grid."""

    def __init__(self, dst, dst_h, dst_w):
        self.x0, self.x1, self.y0, self.y1 = dst
        self.h = dst_h
        self.w = dst_w
        self.cw = (self.x1 - self.x0) / dst_w
        self.ch = (self.y1 - self.y0) / dst_h
        self.energy = [[0.0] * dst_w for _ in range(dst_h)]
        self.area = [[0.0] * dst_w for _ in range(dst_h)]
        self.lost_energy = 0.0
        self.lost_area = 0.0

    def add_quad(self, quad, value):
        quad_area = polygon_area(quad)
        if quad_area == 0.0:
            return
        clipped = clip_polygon_to_box(quad, self.x0, self.y0,
                                      self.x1, self.y1)
        inside = polygon_area(clipped)
        outside = quad_area - inside
        if outside > 0.0:
            self.lost_area += outside
            self.lost_energy += value * outside
        if inside == 0.0:
            return
        xs = [p[0] for p in clipped]
        ys = [p[1] for p in clipped]
        c0 = max(0, int((min(xs) - self.x0) / self.cw))
        c1 = min(self.w - 1, int((max(xs) - self.x0) / self.cw))
        r0 = max(0, int((min(ys) - self.y0) / self.ch))
        r1 = min(self.h - 1, int((max(ys) - self.y0) / self.ch))
        for r in range(r0, r1 + 1):
            cy0 = self.y0 + r * self.ch
            for c in range(c0, c1 + 1):
                cx0 = self.x0 + c * self.cw
                piece = clip_polygon_to_box(
                    clipped, cx0, cy0, cx0 + self.cw, cy0 + self.ch)
                a = polygon_area(piece)
                if a > 0.0:
                    self.area[r][c] += a
                    self.energy[r][c] += value * a


def forward_resample(model, pixels, src_extent, dst_extent, dst_shape, *,
                     tol=1e-6, max_depth=12, max_quads=200000):
    """Splat a small constant-cell grayscale image through the map.

    pixels is a list of rows (top row first) of finite gray values.
    src_extent = (x0, x1, y0, y1) places the image on the normalized
    source plane; each pixel is a constant cell of area
    ((x1-x0)/cols) * ((y1-y0)/rows).  dst_extent and
    dst_shape = (rows, cols) define the target window and grid.

    Returns a ForwardResult.  Energy and covered area are reported
    separately; energy outside the target window is reported as
    lost_energy and is NOT redistributed.
    """
    cleaned, rows, cols = _check_image(pixels)
    sx0, sx1, sy0, sy1 = _check_extent(src_extent, "src_extent")
    dst = _check_extent(dst_extent, "dst_extent")
    if not isinstance(dst_shape, (list, tuple)) or len(dst_shape) != 2:
        raise TypeError("dst_shape: expected (rows, cols)")
    dst_h = checked_int(dst_shape[0], "dst_shape[0]",
                        minimum=1, maximum=MAX_GRID_SIDE)
    dst_w = checked_int(dst_shape[1], "dst_shape[1]",
                        minimum=1, maximum=MAX_GRID_SIDE)
    tol = finite_real(tol, "tol")
    if tol <= 0.0:
        raise ValueError(f"tol: must be > 0, got {tol!r}")
    depth_limit = checked_int(max_depth, "max_depth", minimum=0, maximum=30)
    quad_budget = checked_int(max_quads, "max_quads",
                              minimum=1, maximum=10000000)

    cell_w = (sx1 - sx0) / cols
    cell_h = (sy1 - sy0) / rows
    cell_area = cell_w * cell_h
    if not math.isfinite(cell_area) or cell_area < MIN_CELL_AREA:
        raise ValueError(
            f"src_extent: source pixel cell area {cell_area!r} is "
            f"degenerate (below {MIN_CELL_AREA}); refusing to resample")

    dist = _Distributor(dst, dst_h, dst_w)
    state = {"quads": 0, "error": 0.0, "exhausted": False}

    def deviation(mapped, mids):
        # Max distance between true mapped midpoints and the bilinear
        # interpolation of the mapped corners (curvature indicator).
        (m00, m10, m11, m01) = mapped
        (em_s, em_e, em_n, em_w, cm) = mids
        worst = 0.0
        for true, a, b in ((em_s, m00, m10), (em_e, m10, m11),
                           (em_n, m01, m11), (em_w, m00, m01)):
            d = math.hypot(true[0] - 0.5 * (a[0] + b[0]),
                           true[1] - 0.5 * (a[1] + b[1]))
            if d > worst:
                worst = d
        cx = 0.25 * (m00[0] + m10[0] + m11[0] + m01[0])
        cy = 0.25 * (m00[1] + m10[1] + m11[1] + m01[1])
        d = math.hypot(cm[0] - cx, cm[1] - cy)
        return max(worst, d)

    def subdivide(corners, mapped, value, depth):
        (x0, y0), (x1, y1) = corners
        mx = 0.5 * (x0 + x1)
        my = 0.5 * (y0 + y1)
        mids = (model._distort(mx, y0), model._distort(x1, my),
                model._distort(mx, y1), model._distort(x0, my),
                model._distort(mx, my))
        dev = deviation(mapped, mids)
        if dev > tol and depth < depth_limit and \
                state["quads"] + 4 <= quad_budget:
            (m00, m10, m11, m01) = mapped
            (em_s, em_e, em_n, em_w, cm) = mids
            subdivide(((x0, y0), (mx, my)), (m00, em_s, cm, em_w),
                      value, depth + 1)
            subdivide(((mx, y0), (x1, my)), (em_s, m10, em_e, cm),
                      value, depth + 1)
            subdivide(((mx, my), (x1, y1)), (cm, em_e, m11, em_n),
                      value, depth + 1)
            subdivide(((x0, my), (mx, y1)), (em_w, cm, em_n, m01),
                      value, depth + 1)
            return
        if dev > tol:
            # Wanted to refine but the depth limit or quad budget said
            # no: the tolerance is NOT claimed to be met.
            state["exhausted"] = True
        if dev > state["error"]:
            state["error"] = dev
        state["quads"] += 1
        dist.add_quad([mapped[0], mapped[1], mapped[2], mapped[3]], value)

    total_energy = 0.0
    for r in range(rows):
        # Row 0 is the top of the image (largest y).
        y1 = sy1 - r * cell_h
        y0 = y1 - cell_h
        for c in range(cols):
            x0 = sx0 + c * cell_w
            x1 = x0 + cell_w
            value = cleaned[r][c]
            total_energy += value * cell_area
            mapped = (model._distort(x0, y0), model._distort(x1, y0),
                      model._distort(x1, y1), model._distort(x0, y1))
            subdivide(((x0, y0), (x1, y1)), mapped, value, 0)

    certificate = (
        f"forward area resampling: {rows}x{cols} source cells, "
        f"{state['quads']} leaf quad(s), tol={tol:.1e}, "
        f"max accepted boundary deviation "
        f"{state['error']:.3e}; energy is value x exact clipped overlap "
        f"area (no point sampling, no normalization factor). "
        + ("Budget/depth exhausted before tol was met: the subdivision "
           "error claim does NOT hold."
           if state["exhausted"] else
           "Refinement met the requested tolerance on every leaf."))

    return ForwardResult(
        dist.energy, dist.area, dist.lost_energy, dist.lost_area,
        total_energy, state["error"], state["exhausted"],
        state["quads"], certificate)
