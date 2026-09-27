"""Bounded-box inversion: result types and the inverse solvers.

Two paths are provided:

* ``p1 == p2 == 0`` -- the radial solver in :mod:`distortion.radial`
  partitions g(r) at its critical points and gives a finite
  multiplicity / no-solution certificate.
* general coefficients -- multi-start damped Newton (Levenberg style)
  with an Armijo back-tracking line search, strictly confined to the
  search box [-R, R]^2.

Numerical search never *proves* completeness.  The result therefore
keeps ``proven_no_solution`` (only the radial certificate can set it)
strictly separate from ``budget_exhausted`` (some start could not be
resolved within its iteration budget).
"""

import math

from .numeric import finite_real, finite_real_pair, checked_int
from .radial import radial_invert

__all__ = [
    "Candidate",
    "InverseResult",
    "TrajectoryPoint",
    "TrajectoryResult",
    "undistort_points",
    "undistort_trajectory",
]


class Candidate:
    """One accepted inverse candidate.

    Attributes:
        x, y     -- preimage coordinates inside (or on) [-R, R]^2
        residual -- ||distort(x,y) - (xp,yp)||_2, normalized-plane units
        method   -- "radial_certified" or "damped_newton"
    """

    __slots__ = ("x", "y", "residual", "method")

    def __init__(self, x, y, residual, method):
        self.x = x
        self.y = y
        self.residual = residual
        self.method = method

    def __iter__(self):
        yield self.x
        yield self.y

    def __eq__(self, other):
        if not isinstance(other, Candidate):
            return NotImplemented
        return (self.x == other.x and self.y == other.y
                and self.residual == other.residual
                and self.method == other.method)

    def __repr__(self):
        return (f"Candidate(x={self.x!r}, y={self.y!r}, "
                f"residual={self.residual:.3e}, method={self.method!r})")


class InverseResult:
    """Outcome of a bounded inverse query.

    Attributes:
        xp, yp, R          -- validated query
        candidates         -- de-duplicated accepted candidates,
                              sorted by ascending residual
        unique             -- exactly one accepted candidate
        proven_no_solution -- a certificate says the box has no preimage
        budget_exhausted   -- some numerical start could not be resolved;
                              completeness is therefore not established
                              (even when candidates were found)
        radial             -- the radial-certified path was used
        certificate        -- human-readable explanation
        critical_radii     -- radial path only: g'(r)=0 points in (0, R)
    """

    __slots__ = ("xp", "yp", "R", "candidates", "unique",
                 "proven_no_solution", "budget_exhausted", "radial",
                 "certificate", "critical_radii")

    def __init__(self, xp, yp, R, candidates, *, unique,
                 proven_no_solution, budget_exhausted, radial,
                 certificate, critical_radii=()):
        self.xp = xp
        self.yp = yp
        self.R = R
        self.candidates = candidates
        self.unique = unique
        self.proven_no_solution = proven_no_solution
        self.budget_exhausted = budget_exhausted
        self.radial = radial
        self.certificate = certificate
        self.critical_radii = list(critical_radii)

    def __len__(self):
        return len(self.candidates)

    def __iter__(self):
        return iter(self.candidates)

    def __repr__(self):
        return (f"InverseResult(target=({self.xp!r},{self.yp!r}), R={self.R!r}, "
                f"n={len(self.candidates)}, unique={self.unique}, "
                f"proven_no_solution={self.proven_no_solution}, "
                f"budget_exhausted={self.budget_exhausted})")


class TrajectoryPoint:
    """One target in a branch-tracked trajectory.

    Attributes:
        index     -- zero-based position in the target sequence
        xp, yp    -- requested distorted target
        result    -- complete ordinary single-point inverse result
        candidate -- verified candidate on the tracked branch, or ``None``
                     when tracking was interrupted at this target
        status    -- ``"started"``, ``"tracking"`` or ``"interrupted"``
        reason    -- machine-readable explanation code
        message   -- human-readable explanation
    """

    __slots__ = ("index", "xp", "yp", "result", "candidate",
                 "status", "reason", "message")

    def __init__(self, index, xp, yp, result, candidate, status, reason,
                 message):
        self.index = index
        self.xp = xp
        self.yp = yp
        self.result = result
        self.candidate = candidate
        self.status = status
        self.reason = reason
        self.message = message

    @property
    def tracking(self):
        """True when this point has a candidate on the tracked branch."""
        return self.candidate is not None

    def __iter__(self):
        if self.candidate is None:
            raise TypeError(
                "an interrupted TrajectoryPoint has no (x, y) candidate"
            )
        yield self.candidate.x
        yield self.candidate.y

    def __repr__(self):
        return (f"TrajectoryPoint(index={self.index}, "
                f"target=({self.xp!r},{self.yp!r}), status={self.status!r}, "
                f"reason={self.reason!r})")


class TrajectoryResult:
    """Result of tracking one explicitly selected inverse branch.

    Trajectory processing stops at the first target where the selected
    branch cannot be continued and verified.  The ordinary search result
    at that target remains available, but another candidate is never
    silently selected.
    """

    __slots__ = ("R", "residual_tol", "points", "active",
                 "interrupted", "interruption")

    def __init__(self, R, residual_tol, points):
        self.R = R
        self.residual_tol = residual_tol
        self.points = points
        self.interruption = next((p for p in points
                                  if p.status == "interrupted"), None)
        self.interrupted = self.interruption is not None
        self.active = not self.interrupted

    def __len__(self):
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    @property
    def candidates(self):
        """Verified tracked candidates, in trajectory order."""
        return [p.candidate for p in self.points if p.candidate is not None]

    @property
    def points_xy(self):
        """Tracked ``(x, y)`` pairs; raises if the trajectory interrupted."""
        if self.interrupted:
            p = self.interruption
            raise ValueError(
                f"trajectory interrupted at index {p.index}: {p.message}"
            )
        return [(p.candidate.x, p.candidate.y) for p in self.points]

    def __repr__(self):
        return (f"TrajectoryResult(n={len(self.points)}, "
                f"active={self.active}, interrupted={self.interrupted})")


# ---------------------------------------------------------------------------
# Multi-start damped Newton for the general (tangential) model
# ---------------------------------------------------------------------------

def _backtrack(model, x, y, dx, dy, fx, fy, resid2, xp, yp, R):
    """Armijo back-tracking along (dx, dy) with a hard box constraint.

    Steps leaving the box are shortened (not clipped), so a boundary
    attractor is approached honestly.  Returns
    ``(nx, ny, nfx, nfy, nres2, alpha)`` or ``None``.
    """
    alpha = 1.0
    for _ in range(40):
        nx = x + alpha * dx
        ny = y + alpha * dy
        if -R <= nx <= R and -R <= ny <= R:
            nfx, nfy, *_ = model.residual_and_jacobian(nx, ny, xp, yp)
            nres2 = nfx * nfx + nfy * nfy
            if nres2 < resid2 * (1.0 - 1e-4 * alpha):
                return nx, ny, nfx, nfy, nres2, alpha
        alpha *= 0.5
        if alpha < 1e-16:
            return None
    return None


def _newton_solve(model, start, xp, yp, R, *, max_iterations, residual_tol):
    """Damped Newton from one start, confined to [-R, R]^2.

    A Levenberg-style diagonal damping keeps the normal equations
    positive; when the Newton step cannot make Armijo progress the
    iteration falls back to a normalized steepest-descent direction
    -J^T f, which escapes saddle/flat regions instead of stalling.

    Returns ``(x, y, residual, status)`` where status is "converged",
    "stagnated" or "budget".
    """
    x, y = start
    fx, fy, j00, j01, j11 = model.residual_and_jacobian(x, y, xp, yp)
    resid2 = fx * fx + fy * fy
    damp = 1e-12

    for _ in range(max_iterations):
        if resid2 <= residual_tol * residual_tol:
            return x, y, math.sqrt(resid2), "converged"

        # Normal equations  (J^T J + damp*I) d = -J^T f.
        a00 = j00 * j00 + j01 * j01 + damp
        a01 = j00 * j01 + j01 * j11
        a11 = j01 * j01 + j11 * j11 + damp
        b0 = -(j00 * fx + j01 * fy)
        b1 = -(j01 * fx + j11 * fy)
        det = a00 * a11 - a01 * a01
        scale = max(1.0, a00, a11, abs(a01))

        step = None
        if abs(det) > 1e-18 * scale * scale and math.isfinite(det):
            dx = (b0 * a11 - a01 * b1) / det
            dy = (a00 * b1 - b0 * a01) / det
            if math.isfinite(dx) and math.isfinite(dy):
                step = _backtrack(model, x, y, dx, dy, fx, fy, resid2,
                                  xp, yp, R)

        if step is None:
            # Fallback: normalized steepest descent on the least-squares
            # objective, with a box-aware initial step length.
            gx, gy = -b0, -b1  # J^T f
            gnorm = math.hypot(gx, gy)
            if gnorm <= 1e-300 or not math.isfinite(gnorm):
                damp *= 10.0
                if damp > 1e8:
                    return x, y, math.sqrt(resid2), "stagnated"
                continue
            length = min(0.5 * R, 0.5 * math.sqrt(resid2)
                         / max(1e-30, gnorm))
            length = max(length, 1e-12 * max(1.0, R))
            step = _backtrack(model, x, y,
                              -length * gx / gnorm,
                              -length * gy / gnorm,
                              fx, fy, resid2, xp, yp, R)
            if step is None:
                damp *= 10.0
                if damp > 1e8:
                    return x, y, math.sqrt(resid2), "stagnated"
                continue

        nx, ny, nfx, nfy, nres2, _alpha = step
        move = math.hypot(nx - x, ny - y)
        if move <= 1e-15 * max(1.0, R):
            # Numerical standstill: accept if residual already tiny.
            x, y, fx, fy, resid2 = nx, ny, nfx, nfy, nres2
            if resid2 <= residual_tol * residual_tol:
                return x, y, math.sqrt(resid2), "converged"
            return x, y, math.sqrt(resid2), "stagnated"

        x, y, fx, fy, resid2 = nx, ny, nfx, nfy, nres2
        fx, fy, j00, j01, j11 = model.residual_and_jacobian(x, y, xp, yp)
        damp = max(1e-12, damp * 0.5)

    return x, y, math.sqrt(resid2), "budget"


def _build_starts(xp, yp, R, n):
    """Deterministic multi-start set: n x n grid, rings and target."""
    starts = []
    coords = [-R + 2.0 * R * i / (n - 1) for i in range(n)] if n > 1 else [0.0]
    for a in coords:
        for b in coords:
            starts.append((a, b))

    # Concentric rings add angular density so close-by folded roots on
    # the same ray get distinct basins.
    for frac, angles in ((0.30, 8), (0.55, 12), (0.80, 12), (0.97, 8)):
        rad = frac * R
        for k in range(angles):
            theta = 2.0 * math.pi * k / angles
            starts.append((rad * math.cos(theta), rad * math.sin(theta)))

    # The target itself (clipped) and small perturbations of it; the
    # identity map is then solved in one step.
    cx = min(R, max(-R, xp))
    cy = min(R, max(-R, yp))
    delta = 0.02 * R
    starts.append((cx, cy))
    for dx, dy in ((delta, 0.0), (-delta, 0.0), (0.0, delta), (0.0, -delta)):
        px, py = cx + dx, cy + dy
        if -R <= px <= R and -R <= py <= R:
            starts.append((px, py))
    starts.append((0.0, 0.0))
    return starts


def _general_invert(model, xp, yp, R, *, residual_tol, starts_per_axis,
                    max_iterations):
    starts = _build_starts(xp, yp, R, starts_per_axis)
    accepted = []
    endpoints = []          # (x, y, residual) of non-converged starts
    dedup_tol = 1e-8 * max(1.0, R)

    for start in starts:
        x, y, resid, status = _newton_solve(
            model, start, xp, yp, R,
            max_iterations=max_iterations, residual_tol=residual_tol)
        if status == "converged":
            for c in accepted:
                if abs(c.x - x) <= dedup_tol and abs(c.y - y) <= dedup_tol:
                    if resid < c.residual:
                        c.x, c.y, c.residual = x, y, resid
                    break
            else:
                accepted.append(Candidate(x, y, resid, "damped_newton"))
        else:
            endpoints.append((x, y, resid))

    accepted.sort(key=lambda c: c.residual)

    # A stalled start only matters when it stalls at an UNKNOWN point:
    # a stall on top of an accepted candidate (or at an already tiny
    # residual) is a basin that did lead to a known root.  A stall far
    # from every accepted candidate keeps the budget flag honest.
    unresolved = 0
    for x, y, resid in endpoints:
        if resid <= 1e3 * residual_tol:
            continue
        if any(abs(c.x - x) <= 1e-6 * max(1.0, R)
               and abs(c.y - y) <= 1e-6 * max(1.0, R)
               for c in accepted):
            continue
        unresolved += 1

    budget_exhausted = unresolved > 0
    certificate = (
        f"general-model multi-start damped Newton: {len(starts)} "
        f"deterministic starts, residual tolerance {residual_tol:.0e}, "
        f"{len(accepted)} candidate(s) passed the residual test and "
        f"de-duplication; {unresolved} start(s) did not resolve. "
        f"Numerical search does NOT prove completeness -- further "
        f"preimages may exist outside the sampled basins of attraction."
    )
    return accepted, budget_exhausted, certificate


def undistort_points(model, xp, yp, R, *, residual_tol=1e-10,
                     starts_per_axis=5, max_iterations=80):
    """Validate one query and run the appropriate inverse path."""
    xp, yp = finite_real_pair(xp, yp, ("xp", "yp"))
    R = finite_real(R, "R")
    if R <= 0.0:
        raise ValueError(f"R: search box half-width must be > 0, got {R!r}")
    residual_tol = finite_real(residual_tol, "residual_tol")
    if residual_tol <= 0.0:
        raise ValueError(
            f"residual_tol: must be > 0, got {residual_tol!r}")
    n = checked_int(starts_per_axis, "starts_per_axis", minimum=1, maximum=12)
    iters = checked_int(max_iterations, "max_iterations",
                        minimum=1, maximum=10000)

    if model.is_radial:
        s = math.hypot(xp, yp)
        if s > 0.0:
            ux, uy = xp / s, yp / s
            # A ray stays inside the square box up to this radius, which
            # can exceed R near the box diagonal/corners.
            r_limit = R / max(abs(ux), abs(uy))
        else:
            # Preimages of the origin are whole circles g(r)=0.  A
            # circle of radius r meets the square iff r <= R*sqrt(2)
            # (the box corners), so existence is decided on [0, R*sqrt2]
            # and each intersecting circle is represented by its point
            # on the 45-degree ray, which lies inside the square.
            ux = uy = math.sqrt(0.5)
            r_limit = R * math.sqrt(2.0)

        cert = radial_invert(
            model, xp, yp, r_limit,
            residual_tol=residual_tol,
            origin_direction=(ux, uy))
        accepted = []
        for x, y, resid, method in cert["candidates"]:
            if (resid <= residual_tol
                    and -R <= x <= R and -R <= y <= R):
                accepted.append(Candidate(x, y, resid, method))
        accepted.sort(key=lambda c: c.residual)
        note = ""
        if s == 0.0 and len(accepted) > 1:
            note = (" Target (0,0) with a folding radial map has whole "
                    "circles of preimages (a continuum); one "
                    "representative per circle (on the 45-degree ray) "
                    "is returned, not the continuum.")
        return InverseResult(
            xp, yp, R, accepted,
            unique=cert["unique"] and not (s == 0.0 and len(accepted) > 1),
            proven_no_solution=cert["proven_no_solution"],
            budget_exhausted=False,
            radial=True,
            certificate=cert["detail"] + note,
            critical_radii=cert["critical_radii"],
        )

    accepted, budget_exhausted, certificate = _general_invert(
        model, xp, yp, R, residual_tol=residual_tol,
        starts_per_axis=n, max_iterations=iters)
    return InverseResult(
        xp, yp, R, accepted,
        unique=len(accepted) == 1,
        proven_no_solution=False,
        budget_exhausted=budget_exhausted,
        radial=False,
        certificate=certificate,
    )


def _jacobian_determinant(model, x, y, xp, yp):
    _, _, j00, j01, j11 = model.residual_and_jacobian(x, y, xp, yp)
    return j00 * j11 - j01 * j01


def _candidate_in_result(candidate, result, tolerance):
    for accepted in result.candidates:
        if (abs(accepted.x - candidate.x) <= tolerance
                and abs(accepted.y - candidate.y) <= tolerance):
            return accepted
    return None


def _orientation_compatible(model, previous, candidate, target, R):
    """Check the local sheet orientation when the prior point is nonsingular.

    Near a fold the determinant is numerically unreliable; proximity then
    remains the caller-controlled continuity test, while every accepted
    point is still independently residual-checked.
    """
    xp, yp = target
    old_det = _jacobian_determinant(model, previous.x, previous.y, xp, yp)
    new_det = _jacobian_determinant(model, candidate.x, candidate.y, xp, yp)
    singular_scale = 1e-9 * max(1.0, R)
    if abs(old_det) <= singular_scale or abs(new_det) <= singular_scale:
        return True
    return old_det * new_det > 0.0


def undistort_trajectory(model, targets, R, start, *,
                         residual_tol=1e-10,
                         starts_per_axis=5,
                         max_iterations=80,
                         continuation_tol=None):
    """Track one explicitly selected inverse branch through ``targets``.

    ``start`` must be an accepted :class:`Candidate` for the first target.
    It selects the branch; the function never chooses the lowest-residual
    candidate automatically.  At every target the ordinary single-point
    search is run first, so all available branches at that target are
    independently residual-checked and returned in the point's ``result``.

    Continuation is predicted by one confined Newton step sequence from
    the previous tracked point and is accepted only if that prediction
    reaches a verified candidate within ``continuation_tol`` (default
    ``1e-6 * max(1, R)``) with compatible nonsingular sheet orientation.
    If the branch disappears, cannot be locally continued, or splits far
    enough to exceed the caller's tolerance, processing stops with status
    ``"interrupted"``.  The other candidates found at that target are
    retained for an explicit caller restart; they are not silently used.

    An empty general-model candidate set means the finite numerical
    search found no candidate.  It is reported as
    ``"search_no_candidate"`` and is not a proof that no inverse exists.
    Only the radial certificate can report ``"proven_no_solution"``.
    """
    if not isinstance(start, Candidate):
        raise TypeError(
            "start: expected an existing Candidate for the first target, "
            f"got {type(start).__name__}"
        )
    if not isinstance(targets, (list, tuple)):
        raise TypeError(
            f"targets: expected list/tuple of (xp, yp) pairs, "
            f"got {type(targets).__name__}"
        )
    if len(targets) == 0:
        raise ValueError("targets: trajectory must contain at least one target")
    if len(targets) > 100:
        raise ValueError(
            f"targets: batch size {len(targets)} exceeds the interface "
            "limit of 100"
        )

    cleaned = []
    for i, target in enumerate(targets):
        if not isinstance(target, (list, tuple)) or len(target) != 2:
            raise TypeError(
                f"targets[{i}]: expected a (xp, yp) pair of length 2"
            )
        cleaned.append(finite_real_pair(
            target[0], target[1],
            (f"targets[{i}].xp", f"targets[{i}].yp")))

    R = finite_real(R, "R")
    if R <= 0.0:
        raise ValueError(f"R: search box half-width must be > 0, got {R!r}")
    residual_tol = finite_real(residual_tol, "residual_tol")
    if residual_tol <= 0.0:
        raise ValueError(
            f"residual_tol: must be > 0, got {residual_tol!r}")
    if continuation_tol is None:
        continuation_tol = 1e-6 * max(1.0, R)
    continuation_tol = finite_real(continuation_tol, "continuation_tol")
    if continuation_tol <= 0.0:
        raise ValueError(
            "continuation_tol: must be > 0, "
            f"got {continuation_tol!r}")
    start_x = finite_real(start.x, "start.x")
    start_y = finite_real(start.y, "start.y")
    start_residual = finite_real(start.residual, "start.residual")
    if not (-R <= start_x <= R and -R <= start_y <= R):
        raise ValueError(
            f"start: candidate ({start_x!r}, {start_y!r}) is outside "
            f"[-{R!r}, {R!r}]^2"
        )
    if start_residual < 0.0:
        raise ValueError(
            f"start.residual: must be >= 0, got {start_residual!r}")
    if not isinstance(start.method, str):
        raise TypeError(
            "start.method: expected str, "
            f"got {type(start.method).__name__}")

    points = []
    match_tol = 1e-8 * max(1.0, R)
    previous = None

    for index, (xp, yp) in enumerate(cleaned):
        result = undistort_points(
            model, xp, yp, R,
            residual_tol=residual_tol,
            starts_per_axis=starts_per_axis,
            max_iterations=max_iterations,
        )

        if index == 0:
            matched = _candidate_in_result(start, result, match_tol)
            if matched is None:
                point = TrajectoryPoint(
                    index, xp, yp, result, None, "interrupted",
                    "start_not_verified",
                    "the supplied start candidate was not found among "
                    "the independently verified candidates for the first "
                    "target; no branch was started")
                points.append(point)
                return TrajectoryResult(R, residual_tol, points)
            point = TrajectoryPoint(
                index, xp, yp, result, matched, "started", "started",
                "trajectory started on the caller-selected, verified "
                "candidate")
            points.append(point)
            previous = matched
            continue

        if not result.candidates:
            if result.proven_no_solution:
                reason = "proven_no_solution"
                message = ("the radial certificate proves that this target "
                           "has no preimage in the search box; the tracked "
                           "branch has ended")
            else:
                reason = "search_no_candidate"
                message = ("the finite single-point numerical search found "
                           "no residual-verified candidate; this does NOT "
                           "prove that no inverse exists")
            points.append(TrajectoryPoint(
                index, xp, yp, result, None, "interrupted", reason,
                message))
            return TrajectoryResult(R, residual_tol, points)

        x, y, local_residual, local_status = _newton_solve(
            model, (previous.x, previous.y), xp, yp, R,
            max_iterations=max_iterations,
            residual_tol=residual_tol,
        )

        if local_status != "converged":
            point = TrajectoryPoint(
                index, xp, yp, result, None, "interrupted",
                "local_continuation_failed",
                f"local continuation from the previous branch candidate "
                f"ended with status {local_status!r} and residual "
                f"{local_residual:.3e}; the verified single-point result "
                "is retained, but no candidate was automatically selected")
            points.append(point)
            return TrajectoryResult(R, residual_tol, points)

        predicted = Candidate(x, y, local_residual, "damped_newton")
        matched = _candidate_in_result(predicted, result, match_tol)
        if matched is None:
            points.append(TrajectoryPoint(
                index, xp, yp, result, None, "interrupted",
                "continuation_not_verified",
                "local continuation converged numerically, but its endpoint "
                "was not present in the independently verified single-point "
                "candidate set; tracking was not continued"))
            return TrajectoryResult(R, residual_tol, points)

        distance = math.hypot(matched.x - previous.x,
                              matched.y - previous.y)
        if distance > continuation_tol:
            points.append(TrajectoryPoint(
                index, xp, yp, result, None, "interrupted",
                "branch_gap",
                f"the verified continuation candidate moves "
                f"{distance:.3e}, exceeding continuation_tol="
                f"{continuation_tol:.3e}; branch continuation is not "
                "claimed. Inspect result.candidates and restart explicitly "
                "if another branch is desired"))
            return TrajectoryResult(R, residual_tol, points)

        if not _orientation_compatible(model, previous, matched,
                                       (xp, yp), R):
            points.append(TrajectoryPoint(
                index, xp, yp, result, None, "interrupted",
                "branch_changed",
                "the nearest residual-verified candidate lies across a "
                "Jacobian fold (its determinant orientation changed); "
                "the old branch was not silently replaced by another one"))
            return TrajectoryResult(R, residual_tol, points)

        points.append(TrajectoryPoint(
            index, xp, yp, result, matched, "tracking", "tracking",
            "continued on the previously selected branch; the candidate "
            "also appears in the independently verified single-point result"))
        previous = matched

    return TrajectoryResult(R, residual_tol, points)
