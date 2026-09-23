"""Pure-radial special case (p1 == p2 == 0): segmented root solving.

For the radial model the forward map is

    (x', y') = g(r) * (x/r, y/r),   g(r) = r*(1 + k1*r^2 + k2*r^4),

so inversion along a target ray reduces to the scalar equation
``g(r) = +/- s`` on the closed interval ``r in [0, R]``.

The map g is C^1 with

    g'(r) = 1 + 3*k1*r^2 + 5*k2*r^4.

Writing q = r^2 turns g'(r) = 0 into the quadratic
``5*k2*q^2 + 3*k1*q + 1 = 0``.  Every positive root q with q < R^2
is an interior critical point.  Between consecutive critical points
(and the endpoints 0 and R) g is strictly monotone, so each segment
contains at most one solution of g(r) = c and a bisection decides
existence exactly (up to the documented floating-point error model).

Because every monotone segment is probed, this yields a finite
uniqueness / multiplicity certificate rather than "the first root
found".  It also proves "no solution" when the target radius exceeds
max|g| over [0,R] (that maximum is read off at the endpoints and
critical points).
"""

import math

__all__ = ["radial_invert"]


def _critical_radii(k1, k2, R):
    """All r in (0, R) where g'(r) = 0, sorted ascending."""
    eps = 0.0
    out = []
    if k2 == 0.0:
        if k1 != 0.0:
            q = -1.0 / (3.0 * k1)
            if q > eps:
                r = math.sqrt(q)
                if 0.0 < r < R:
                    out.append(r)
    else:
        a = 5.0 * k2
        b = 3.0 * k1
        c = 1.0
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            root = math.sqrt(disc)
            for q in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
                if q > eps:
                    r = math.sqrt(q)
                    if 0.0 < r < R:
                        out.append(r)
    out.sort()
    # Merge duplicates created by floating-point round-off near a
    # double root of g'.
    merged = []
    for r in out:
        if not merged or r - merged[-1] > 1e-12 * max(1.0, R):
            merged.append(r)
        else:
            merged[-1] = 0.5 * (merged[-1] + r)
    return merged


def radial_invert(model, xp, yp, R, *, residual_tol=1e-10,
                  origin_direction=(1.0, 0.0)):
    """Certified inversion for the radial-only model.

    Returns a dict with keys:
        candidates          -- list of (x, y, residual, method)
        unique              -- bool, multiplicity certificate
        proven_no_solution  -- bool, box-feasibility certificate
        critical_radii      -- list of the g\' critical radii in (0,R)
        segments            -- number of monotone segments probed
        detail              -- human-readable certificate explanation
        target_radius       -- s = hypot(xp, yp)
    """
    k1, k2 = model.k1, model.k2
    s = math.hypot(xp, yp)

    def g(r):
        r2 = r * r
        return r * (1.0 + (k1 + k2 * r2) * r2)

    crit = _critical_radii(k1, k2, R)
    grid = [0.0] + crit + [R]
    g_at = [g(r) for r in grid]

    # max |g(r)| on [0,R]: extrema of |g| occur at endpoints of the
    # monotone segments of g, i.e. at 0, R and the critical points.
    max_abs = max(abs(v) for v in g_at)

    result = {
        "candidates": [],
        "unique": False,
        "proven_no_solution": False,
        "critical_radii": crit,
        "segments": len(grid) - 1,
        "detail": "",
        "target_radius": s,
    }

    # Floating-point noise band for polynomial evaluations.  Exact
    # mathematics is certified *up to* this band (documented in the
    # README error model): a target within the band of a critical value
    # is classified as the tangent (double) root; a target beyond the
    # band of max|g| is certified absent.
    def band(value):
        return 64.0 * 2.220446049250313e-16 * max(1.0, abs(value))

    if s > max_abs + band(max(max_abs, s)):
        result["proven_no_solution"] = True
        result["detail"] = (
            f"target radius {s!r} exceeds max|g(r)|={max_abs!r} on "
            f"[0,R={R!r}] by more than the floating-point noise band "
            f"{band(max(max_abs, s)):.3e}; |g| is continuous with "
            f"g(0)=0, so the intermediate-value bound is tight across "
            f"the {len(grid) - 1} monotone segment(s): no inverse "
            f"exists inside the box."
        )
        return result

    if s == 0.0:
        ux, uy = origin_direction
        unorm = math.hypot(ux, uy)
        ux, uy = ux / unorm, uy / unorm
    else:
        ux, uy = xp / s, yp / s

    radii = []  # (r, sign) with sign in {+1,-1}: candidate at sign*r*u

    def add_root(r, sign):
        for rr, _ in radii:
            if abs(rr - r) <= 1e-11 * max(1.0, R):
                return
        radii.append((r, sign))

    def bisect_crossing(i, target, sign):
        """Strict sign-change root inside segment i (endpoints excluded)."""
        lo, hi = grid[i], grid[i + 1]
        glo, ghi = g_at[i], g_at[i + 1]
        if (glo - target) * (ghi - target) >= 0.0:
            return  # no strict sign change in the segment interior
        lo_f, hi_f = lo, hi
        flo, fhi = glo - target, ghi - target
        for _ in range(200):
            mid = 0.5 * (lo_f + hi_f)
            fm = g(mid) - target
            if fm == 0.0 or (hi_f - lo_f) <= 1e-14 * max(1.0, R):
                lo_f = hi_f = mid
                break
            if flo * fm <= 0.0:
                hi_f, fhi = mid, fm
            else:
                lo_f, flo = mid, fm
        add_root(0.5 * (lo_f + hi_f), sign)

    def handle_critical(j, target, sign):
        """Classify g(r_c) vs target at interior critical point grid[j]."""
        rc = grid[j]
        gc = g_at[j]
        gap = gc - target
        b = band(max(abs(gc), abs(target)))
        if abs(gap) <= b:
            # Tangent root at the critical value (within FP noise).
            # Suppress the two nearby crossing roots the bisections may
            # produce on either side and report the single extremum root.
            gpp = abs(6.0 * k1 * rc + 20.0 * k2 * rc ** 3)
            hood = 2.0 * math.sqrt(2.0 * b / gpp) if gpp > 0.0 else b
            hood = max(hood, 1e-11 * max(1.0, R))
            radii[:] = [(r, sg) for (r, sg) in radii if abs(r - rc) > hood]
            add_root(rc, sign)
            return
        lo, hi = grid[j - 1], grid[j + 1]
        glo, ghi = g_at[j - 1], g_at[j + 1]
        # Strictly above/below: the monotone neighbour segments carry
        # ordinary crossings; nothing special at rc.

    def solve_target(target, sign):
        # Ordinary crossings in every segment interior.
        for i in range(len(grid) - 1):
            bisect_crossing(i, target, sign)
        # Exact endpoint hits (with the same noise band).
        for r_endpoint, g_endpoint in ((0.0, g_at[0]), (R, g_at[-1])):
            if abs(g_endpoint - target) <= band(max(abs(g_endpoint),
                                                    abs(target))):
                add_root(r_endpoint, sign)
        # Tangencies at the interior critical points.
        for j in range(1, len(grid) - 1):
            handle_critical(j, target, sign)

    if s == 0.0:
        # r = 0 always maps to the origin.  Positive radii with g(r)=0
        # are whole circles collapsing onto it: probe every segment and
        # the critical points for g(r) = 0.
        add_root(0.0, 1)
        solve_target(0.0, 1)
        # r=0 is an endpoint, not a tangency; de-dup if re-added.
    else:
        # candidate at +r along u (g(r) = +s) and at r along -u
        # (g(r) = -s, i.e. the folded branch maps -r*u to +s*u).
        solve_target(s, 1)
        solve_target(-s, -1)

    for r, sign in radii:
        x = sign * r * ux
        y = sign * r * uy
        xp_chk, yp_chk = model._distort(x, y)
        residual = math.hypot(xp_chk - xp, yp_chk - yp)
        result["candidates"].append((x, y, residual, "radial_certified"))

    n = len(radii)
    result["unique"] = n == 1
    if s == 0.0 and n == 1:
        reason = (
            "g(r) != 0 for every r in (0,R]; r=0 is the sole preimage"
        )
    else:
        reason = (
            f"{n} root(s) of g(r)=+/-{s!r} found by probing all "
            f"{len(grid) - 1} strictly-monotone segment(s); uniqueness "
            f"requires exactly one"
        )
    result["detail"] = (
        f"radial certificate: {len(crit)} interior critical point(s) "
        f"{[round(v, 12) for v in crit]} partition [0,{R!r}] into "
        f"{len(grid) - 1} monotone segment(s); {reason}."
    )
    return result
