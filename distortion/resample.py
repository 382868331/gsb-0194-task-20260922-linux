"""Forward resampling: splat source pixels as constant-value area cells.

Each source pixel is treated as a unit square of constant value (not a
point sample).  The square is pushed through the forward distortion map
with adaptive quadtree subdivision so that the curved image of the pixel
boundary is approximated to a requested area tolerance.  Every accepted
leaf contributes its mapped area times the pixel value to exactly one
target cell (the one containing the mapped centroid); the accumulated
energy and the accumulated covered area are reported as two separate
grids.  Energy whose mapped centroid falls outside the target window is
accumulated in 'lost_energy' -- it is never silently dropped or folded
back into the window.

This is NOT point sampling plus an after-the-fact normalization factor:
the area of every leaf is measured in the target plane from the mapped
polygon itself.

An exact polygon-clipping reference ('exact_quad_resample') is provided
for the degenerate cases where the forward map sends squares to
parallelograms exactly (identity / affine maps); there Sutherland-Hodgman
clipping against every target cell gives the exact covered areas.

Only Python standard library is used.
"""

import math

from .numeric import finite_real, finite_real_pair, checked_int
from .model import DistortionModel

__all__ = [
    "ResampleResult",
    "forward_resample",
    "exact_quad_resample",
    "MAX_IMAGE_SIDE",
    "MAX_TARGET_CELLS",
    "MIN_CELL_AREA",
]

# Interface bounds for this entry point (small grayscale images only).
MAX_IMAGE_SIDE = 64
MAX_TARGET_CELLS = 4096
# Target cells smaller than this are rejected: area bookkeeping would be
# meaningless at that scale.
MIN_CELL_AREA = 1e-12


class ResampleResult:
    """Outcome of one forward-resampling run.

    Attributes:
        energy         -- rows x cols grid of accumulated value*area
        area           -- rows x cols grid of accumulated covered area
        lost_energy    -- value*area whose centroid left the target window
        input_energy   -- sum of value*area over all source pixels
        error_estimate -- sum of per-leaf |quad area - refined area|
                          subdivision error estimates (target-plane area
                          units); an indicator, not a strict bound
        converged      -- False when the leaf budget was exhausted before
                          every leaf met the area tolerance
        leaves_used    -- number of accepted quadtree leaves
    """

    __slots__ = ("energy", "area", "lost_energy", "input_energy",
                 "error_estimate", "converged", "leaves_used")

    def __init__(self, energy, area, lost_energy, input_energy,
                 error_estimate, converged, leaves_used):
        self.energy = energy
        self.area = area
        self.lost_energy = lost_energy
        self.input_energy = input_energy
        self.error_estimate = error_estimate
        self.converged = converged
        self.leaves_used = leaves_used

    @property
    def accounted_energy(self):
        """Energy deposited in the window plus energy lost outside it."""
        return sum(sum(row) for row in self.energy) + self.lost_energy

    def __repr__(self):
        return ("ResampleResult(lost_energy=%r, input_energy=%r, "
                "error_estimate=%r, converged=%r, leaves_used=%r)"
                % (self.lost_energy, self.input_energy,
                   self.error_estimate, self.converged, self.leaves_used))


def _check_image(image):
    if not isinstance(image, (list, tuple)) or len(image) == 0:
        raise TypeError("image: expected a non-empty list of rows")
    if len(image) > MAX_IMAGE_SIDE:
        raise ValueError(
            "image: %d rows exceeds the limit of %d"
            % (len(image), MAX_IMAGE_SIDE))
    rows = []
    width = None
    for i, row in enumerate(image):
        if not isinstance(row, (list, tuple)) or len(row) == 0:
            raise TypeError("image[%d]: expected a non-empty row" % i)
        if len(row) > MAX_IMAGE_SIDE:
            raise ValueError(
                "image[%d]: %d columns exceeds the limit of %d"
                % (i, len(row), MAX_IMAGE_SIDE))
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(
                "image[%d]: ragged image, row has %d columns, expected %d"
                % (i, len(row), width))
        rows.append([finite_real(v, "image[%d][]" % i) for v in row])
    return rows, len(rows), width


def _check_target(window, shape):
    if not isinstance(window, (list, tuple)) or len(window) != 4:
        raise TypeError("window: expected (x0, y0, width, height)")
    x0, y0 = finite_real_pair(window[0], window[1],
                              ("window.x0", "window.y0"))
    w = finite_real(window[2], "window.width")
    h = finite_real(window[3], "window.height")
    if w <= 0.0 or h <= 0.0:
        raise ValueError(
            "window: width/height must be > 0, got (%r, %r)" % (w, h))
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        raise TypeError("shape: expected (rows, cols)")
    rows = checked_int(shape[0], "shape.rows", minimum=1)
    cols = checked_int(shape[1], "shape.cols", minimum=1)
    if rows * cols > MAX_TARGET_CELLS:
        raise ValueError(
            "shape: %dx%d exceeds the limit of %d target cells"
            % (rows, cols, MAX_TARGET_CELLS))
    cell_area = (w / cols) * (h / rows)
    if cell_area < MIN_CELL_AREA:
        raise ValueError(
            "window/shape: target cell area %r is below the minimum of "
            "%r; refusing near-zero-area cells" % (cell_area, MIN_CELL_AREA))
    return x0, y0, w, h, rows, cols


def _polygon_area_and_centroid(points):
    """Signed-area magnitude and area-weighted centroid of a polygon."""
    twice_area = 0.0
    cx = 0.0
    cy = 0.0
    n = len(points)
    for i in range(n):
        xa, ya = points[i]
        xb, yb = points[(i + 1) % n]
        cross = xa * yb - xb * ya
        twice_area += cross
        cx += (xa + xb) * cross
        cy += (ya + yb) * cross
    if twice_area == 0.0:
        # Degenerate (zero-area) polygon: fall back to the vertex mean so
        # the caller can still place it, with area 0.
        mx = sum(p[0] for p in points) / n
        my = sum(p[1] for p in points) / n
        return 0.0, mx, my
    area = abs(twice_area) / 2.0
    return area, cx / (3.0 * twice_area), cy / (3.0 * twice_area)


def _mapped_octagon(model, x0, y0, size):
    """Image of a source square under the map: 4 corners + 4 edge midpoints."""
    half = size / 2.0
    sources = ((x0, y0), (x0 + half, y0), (x0 + size, y0),
               (x0 + size, y0 + half), (x0 + size, y0 + size),
               (x0 + half, y0 + size), (x0, y0 + size),
               (x0, y0 + half))
    return [model._distort(sx, sy) for (sx, sy) in sources]


def forward_resample(model, image, window, shape, *,
                     origin=(0.0, 0.0), area_tol=1e-6, max_depth=8,
                     max_leaves=200000):
    """Splat a small constant-cell grayscale image through the forward map.

    image[r][c] is the constant value of the source pixel covering
    [ox+c, ox+c+1] x [oy+r, oy+r+1] in the normalized plane, where
    (ox, oy) = origin.  window = (x0, y0, width, height) and
    shape = (rows, cols) define the target grid.  Each source square is
    subdivided (quadtree, up to max_depth) until the mapped 8-gon's area
    agrees with the mapped corner quad to area_tol (target-plane area
    units), or the leaf budget max_leaves runs out -- in which case
    converged is False and no strict conservation error is claimed.

    Returns a ResampleResult.  All inputs are validated before any
    computation, so a rejected call leaves no partial state.
    """
    if not isinstance(model, DistortionModel):
        raise TypeError(
            "model: expected DistortionModel, got %s" % type(model).__name__)
    rows_img, n_rows, n_cols = _check_image(image)
    if not isinstance(origin, (list, tuple)) or len(origin) != 2:
        raise TypeError("origin: expected an (x, y) pair")
    ox, oy = finite_real_pair(origin[0], origin[1], ("origin.x", "origin.y"))
    x0, y0, w, h, t_rows, t_cols = _check_target(window, shape)
    area_tol = finite_real(area_tol, "area_tol")
    if area_tol <= 0.0:
        raise ValueError("area_tol: must be > 0, got %r" % (area_tol,))
    max_depth = checked_int(max_depth, "max_depth", minimum=0, maximum=30)
    max_leaves = checked_int(max_leaves, "max_leaves", minimum=1,
                             maximum=10_000_000)

    energy = [[0.0] * t_cols for _ in range(t_rows)]
    area = [[0.0] * t_cols for _ in range(t_rows)]
    cell_w = w / t_cols
    cell_h = h / t_rows
    lost = 0.0
    error_estimate = 0.0
    converged = True
    leaves_used = 0
    nodes_visited = 0
    input_energy = 0.0

    def splat(leaf_area, cx, cy, value):
        nonlocal lost
        if leaf_area == 0.0:
            return
        col = int((cx - x0) / cell_w)
        row = int((cy - y0) / cell_h)
        if 0 <= row < t_rows and 0 <= col < t_cols:
            energy[row][col] += value * leaf_area
            area[row][col] += leaf_area
        else:
            lost += value * leaf_area

    def subdivide(sx, sy, size, value, depth):
        nonlocal error_estimate, converged, leaves_used, nodes_visited
        nodes_visited += 1
        octagon = _mapped_octagon(model, sx, sy, size)
        quad = (octagon[0], octagon[2], octagon[4], octagon[6])
        oct_area, cx, cy = _polygon_area_and_centroid(octagon)
        quad_area, _, _ = _polygon_area_and_centroid(quad)
        err = abs(oct_area - quad_area)
        if err <= area_tol or depth >= max_depth:
            accept = True
        elif nodes_visited + 4 > max_leaves:
            # Budget exhausted: accept at this node and flag it; do NOT
            # claim the tolerance (hence strict conservation) was met.
            accept = True
            converged = False
        else:
            accept = False
        if accept:
            leaves_used += 1
            error_estimate += err
            splat(oct_area, cx, cy, value)
            return
        half = size / 2.0
        subdivide(sx, sy, half, value, depth + 1)
        subdivide(sx + half, sy, half, value, depth + 1)
        subdivide(sx, sy + half, half, value, depth + 1)
        subdivide(sx + half, sy + half, half, value, depth + 1)

    for r in range(n_rows):
        for c in range(n_cols):
            value = rows_img[r][c]
            input_energy += value * 1.0
            if value == 0.0:
                continue
            subdivide(ox + c, oy + r, 1.0, value, 0)

    return ResampleResult(energy, area, lost, input_energy,
                          error_estimate, converged, leaves_used)


def _clip_polygon_to_rect(points, rx0, ry0, rx1, ry1):
    """Sutherland-Hodgman clip of a polygon against an axis-aligned rect."""

    def clip_edge(poly, inside, intersect):
        if not poly:
            return poly
        out = []
        prev = poly[-1]
        prev_in = inside(prev)
        for cur in poly:
            cur_in = inside(cur)
            if cur_in:
                if not prev_in:
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif prev_in:
                out.append(intersect(prev, cur))
            prev, prev_in = cur, cur_in
        return out

    def make(edge, coord, keep_greater):
        def inside(p):
            return p[coord] >= edge if keep_greater else p[coord] <= edge

        def intersect(a, b):
            t = ((edge - a[coord]) / (b[coord] - a[coord])
                 if b[coord] != a[coord] else 0.0)
            other = 1 - coord
            pt = [0.0, 0.0]
            pt[coord] = edge
            pt[other] = a[other] + t * (b[other] - a[other])
            return (pt[0], pt[1])

        return inside, intersect

    poly = list(points)
    for edge, coord, keep in ((rx0, 0, True), (rx1, 0, False),
                              (ry0, 1, True), (ry1, 1, False)):
        inside, intersect = make(edge, coord, keep)
        poly = clip_edge(poly, inside, intersect)
        if not poly:
            break
    return poly


def exact_quad_resample(quads, values, window, shape):
    """Exact reference: clip mapped quads against every target cell.

    quads is a list of 4-vertex polygons (e.g. source pixels mapped by an
    exactly affine forward map, where squares become parallelograms);
    values[i] is the constant value of quad i.  Covered area and energy
    per target cell are computed exactly by polygon clipping -- no
    subdivision, no tolerance.  Area outside the window accumulates into
    lost_energy via the unclipped remainder.

    Returns a ResampleResult with error_estimate == 0.0 and
    converged == True.
    """
    if not isinstance(quads, (list, tuple)):
        raise TypeError("quads: expected a list of 4-vertex polygons")
    if not isinstance(values, (list, tuple)):
        raise TypeError("values: expected a list of finite reals")
    if len(quads) != len(values):
        raise ValueError(
            "quads/values: length mismatch %d != %d"
            % (len(quads), len(values)))
    cleaned_quads = []
    for i, q in enumerate(quads):
        if not isinstance(q, (list, tuple)) or len(q) != 4:
            raise TypeError("quads[%d]: expected 4 vertices" % i)
        cleaned_quads.append(
            [finite_real_pair(p[0], p[1],
                              ("quads[%d].x" % i, "quads[%d].y" % i))
             for p in q])
    cleaned_values = [finite_real(v, "values[%d]" % i)
                      for i, v in enumerate(values)]
    x0, y0, w, h, t_rows, t_cols = _check_target(window, shape)

    energy = [[0.0] * t_cols for _ in range(t_rows)]
    area = [[0.0] * t_cols for _ in range(t_rows)]
    cell_w = w / t_cols
    cell_h = h / t_rows
    lost = 0.0
    input_energy = 0.0

    for quad, value in zip(cleaned_quads, cleaned_values):
        quad_area, _, _ = _polygon_area_and_centroid(quad)
        input_energy += value * quad_area
        covered = 0.0
        for row in range(t_rows):
            ry0 = y0 + row * cell_h
            ry1 = ry0 + cell_h
            for col in range(t_cols):
                rx0 = x0 + col * cell_w
                rx1 = rx0 + cell_w
                piece = _clip_polygon_to_rect(quad, rx0, ry0, rx1, ry1)
                if not piece:
                    continue
                piece_area, _, _ = _polygon_area_and_centroid(piece)
                if piece_area > 0.0:
                    energy[row][col] += value * piece_area
                    area[row][col] += piece_area
                    covered += piece_area
        lost += value * max(quad_area - covered, 0.0)

    return ResampleResult(energy, area, lost, input_energy, 0.0, True,
                          len(cleaned_quads))
