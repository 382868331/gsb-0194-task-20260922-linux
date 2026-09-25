"""Continuous-branch tracking over a sequence of distorted targets.

The bounded inverse solver (undistort_points) returns *every*
residual-acceptable candidate; when the forward map folds, one
distorted point can have several preimages.  Picking the first list
entry frame after frame is meaningless: list order follows residuals,
not geometry.

This module adds a *tracking* entry point.  The caller supplies the
distorted-coordinate sequence and an exact first undistorted anchor.
For every frame the original solver is still run unchanged, and among
its accepted candidates the root closest to the previous tracked point
is chosen -- but only if its distance does not exceed an explicit
per-frame step limit.  Two candidates equally close (within a given
tolerance) make the frame ambiguous: the tie is never broken by list
order.  A frame with no candidate, with every candidate beyond the
step limit, or with an ambiguous tie, *breaks* the trajectory;
tracking can only resume at a fresh caller-supplied anchor.

Tracking is a branch-continuity heuristic.  It does NOT make the
inverse unique: the completeness limitations of the multi-start
Newton path and the radial special-case certificate are produced
exactly as before and remain visible on every step.
"""

import math

from .numeric import finite_real, finite_real_pair
from .inverse import undistort_points

__all__ = ["TrackStep", "TrackResult", "track_undistorted"]

# Frame-level statuses.
OK = "ok"
AMBIGUOUS = "ambiguous"
NO_CANDIDATE = "no_candidate"
OVER_STEP = "over_step"
INVALID_ANCHOR = "invalid_anchor"
NEEDS_ANCHOR = "needs_anchor"

_BREAK_STATUSES = frozenset((AMBIGUOUS, NO_CANDIDATE, OVER_STEP,
                             INVALID_ANCHOR))


class TrackStep:
    """Outcome for one tracked frame."""

    __slots__ = ("index", "status", "x", "y", "residual", "method",
                 "distance", "candidates", "radial",
                 "proven_no_solution", "budget_exhausted", "reason")

    def __init__(self, index, status, *, x=None, y=None, residual=None,
                 method=None, distance=None, candidates=(), radial=False,
                 proven_no_solution=False, budget_exhausted=False,
                 reason=""):
        self.index = index
        self.status = status
        self.x = x
        self.y = y
        self.residual = residual
        self.method = method
        self.distance = distance
        self.candidates = list(candidates)
        self.radial = radial
        self.proven_no_solution = proven_no_solution
        self.budget_exhausted = budget_exhausted
        self.reason = reason

    @property
    def ok(self):
        """True when a continuous-branch root was selected."""
        return self.status == OK

    @property
    def broken(self):
        """True when the trajectory was broken at this frame."""
        return self.status in _BREAK_STATUSES

    def __iter__(self):
        if self.x is None:
            raise TypeError("this step has no selected root")
        yield self.x
        yield self.y

    def __repr__(self):
        return (f"TrackStep(index={self.index}, status={self.status!r}, "
                f"x={self.x!r}, y={self.y!r}, distance={self.distance!r})")


class TrackResult:
    """Outcome of a whole tracking run."""

    __slots__ = ("steps", "active", "last", "broken", "break_index",
                 "max_step", "tie_tol", "anchor_tol")

    def __init__(self, steps, max_step, tie_tol, anchor_tol):
        self.steps = steps
        self.max_step = max_step
        self.tie_tol = tie_tol
        self.anchor_tol = anchor_tol
        self.last = None
        self.break_index = None
        for step in steps:
            if step.status == OK:
                self.last = (step.x, step.y)
            elif step.status in _BREAK_STATUSES and self.break_index is None:
                self.break_index = step.index
        self.broken = self.break_index is not None
        self.active = bool(steps) and steps[-1].status == OK

    def __len__(self):
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def __repr__(self):
        return (f"TrackResult(frames={len(self.steps)}, active={self.active}, "
                f"broken={self.broken}, break_index={self.break_index!r}, "
                f"last={self.last!r})")


def _validate_anchor(anchor, R, name):
    ax, ay = finite_real_pair(anchor[0], anchor[1],
                              (f"{name}.x", f"{name}.y"))
    if not (-R <= ax <= R and -R <= ay <= R):
        raise ValueError(
            f"{name}: undistorted anchor ({ax!r}, {ay!r}) must lie "
            f"inside the closed search box [-{R}, {R}]^2")
    return ax, ay


def _anchor_step(index, model, target, anchor, R, anchor_tol, solver_kwargs):
    """Validate a caller-supplied anchor against its target.

    The anchor is accepted only when its forward image matches the
    target within anchor_tol.  The solver is still run, so the full
    candidate set and certificates are retained on the step.
    """
    ax, ay = anchor
    fx, fy = model._distort(ax, ay)
    tx, ty = target
    residual = math.hypot(fx - tx, fy - ty)
    result = undistort_points(model, tx, ty, R, **solver_kwargs)
    common = dict(candidates=result.candidates, radial=result.radial,
                  proven_no_solution=result.proven_no_solution,
                  budget_exhausted=result.budget_exhausted)
    if residual > anchor_tol:
        return TrackStep(
            index, INVALID_ANCHOR, residual=residual,
            reason=(
                f"anchor ({ax!r}, {ay!r}) fails forward validation: "
                f"||distort(anchor) - target|| = {residual:.3e} exceeds "
                f"anchor_tol = {anchor_tol:.3e}; trajectory broken, a "
                f"new anchor is required to continue"),
            **common)
    return TrackStep(
        index, OK, x=ax, y=ay, residual=residual, method="anchor",
        distance=0.0,
        reason=(f"anchor accepted: forward residual {residual:.3e} "
                f"<= anchor_tol {anchor_tol:.3e}"),
        **common)


def track_undistorted(model, targets, anchor, R, *, max_step,
                      tie_tol=1e-9, anchor_tol=1e-8,
                      resume_anchors=None, **solver_kwargs):
    """Track one continuous inverse branch across targets.

    Args:
        model:    DistortionModel
        targets:  sequence of distorted (xp, yp) points (1..100)
        anchor:   undistorted (x, y) preimage of targets[0], inside
                  the closed box; forward-checked against the target
        R:        search box half-width
        max_step: explicit per-frame step upper bound in undistorted
                  coordinates; a candidate farther than this cannot be
                  selected and breaks the trajectory
        tie_tol:  distance differences within this tolerance are a tie;
                  a tie of two closest roots gives AMBIGUOUS, never a
                  list-order pick
        anchor_tol: forward residual tolerance for anchors
        resume_anchors: optional {frame_index: (x, y)} giving a NEW
                  anchor after a break; forward-validated likewise
        **solver_kwargs: residual_tol / starts_per_axis /
                  max_iterations, passed unchanged to the solver

    After a break, later frames without a resume anchor are reported
    NEEDS_ANCHOR and the solver is not run for them.
    """
    cleaned = model._check_batch(targets)
    if not cleaned:
        raise ValueError("targets: at least one distorted point is required")
    R = finite_real(R, "R")
    if R <= 0.0:
        raise ValueError(f"R: search box half-width must be > 0, got {R!r}")
    max_step = finite_real(max_step, "max_step")
    if max_step <= 0.0:
        raise ValueError(f"max_step: must be > 0, got {max_step!r}")
    tie_tol = finite_real(tie_tol, "tie_tol")
    if tie_tol <= 0.0:
        raise ValueError(f"tie_tol: must be > 0, got {tie_tol!r}")
    anchor_tol = finite_real(anchor_tol, "anchor_tol")
    if anchor_tol <= 0.0:
        raise ValueError(f"anchor_tol: must be > 0, got {anchor_tol!r}")

    first_anchor = _validate_anchor(anchor, R, "anchor")

    resumes = {}
    if resume_anchors is not None:
        if not isinstance(resume_anchors, dict):
            raise TypeError(
                "resume_anchors: expected {frame_index: (x, y)} dict, "
                f"got {type(resume_anchors).__name__}")
        n = len(cleaned)
        for key, point in resume_anchors.items():
            if isinstance(key, bool) or not isinstance(key, int):
                raise TypeError(
                    "resume_anchors key: expected int frame index, "
                    f"got {type(key).__name__}")
            if not (0 <= key < n):
                raise ValueError(
                    f"resume_anchors: frame index {key} out of range "
                    f"[0, {n - 1}]")
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise TypeError(
                    f"resume_anchors[{key}]: expected an (x, y) pair")
            resumes[key] = _validate_anchor(
                point, R, f"resume_anchors[{key}]")

    steps = []
    prev = None
    for i, target in enumerate(cleaned):
        if i == 0:
            step = _anchor_step(i, model, target, first_anchor, R,
                                anchor_tol, solver_kwargs)
        elif i in resumes:
            step = _anchor_step(i, model, target, resumes[i], R,
                                anchor_tol, solver_kwargs)
        elif prev is None:
            step = TrackStep(
                i, NEEDS_ANCHOR,
                reason=("trajectory is broken before this frame; the "
                        "solver is not run -- supply a new anchor via "
                        "resume_anchors to continue"))
        else:
            tx, ty = target
            result = undistort_points(model, tx, ty, R, **solver_kwargs)
            candidates = result.candidates
            px, py = prev
            distances = [math.hypot(c.x - px, c.y - py)
                         for c in candidates]
            common = dict(candidates=candidates, radial=result.radial,
                          proven_no_solution=result.proven_no_solution,
                          budget_exhausted=result.budget_exhausted)
            if not candidates:
                if result.proven_no_solution:
                    why = ("no residual-acceptable candidate at this frame; "
                           "radial certificate: proven no solution")
                else:
                    why = ("no residual-acceptable candidate at this frame; "
                           "numerical search did not establish completeness "
                           f"(budget_exhausted={result.budget_exhausted})")
                step = TrackStep(
                    i, NO_CANDIDATE,
                    reason=why + "; trajectory broken, a new anchor is "
                                 "required to continue",
                    **common)
            else:
                dmin = min(distances)
                n_closest = sum(1 for d in distances
                                if abs(d - dmin) <= tie_tol)
                if dmin > max_step:
                    step = TrackStep(
                        i, OVER_STEP,
                        reason=(
                            f"nearest candidate is {dmin:.6e} from the "
                            f"previous point, exceeding max_step = "
                            f"{max_step:.6e}; trajectory broken rather than "
                            f"jumped, a new anchor is required to continue"),
                        **common)
                elif n_closest >= 2:
                    step = TrackStep(
                        i, AMBIGUOUS,
                        reason=(
                            f"{n_closest} candidates tie as closest to the "
                            f"previous point at distance {dmin:.6e} "
                            f"(within tie_tol = {tie_tol:.3e}); no root is "
                            f"chosen and list order is not used; trajectory "
                            f"broken, a new anchor is required to continue"),
                        **common)
                else:
                    chosen_idx = distances.index(dmin)
                    chosen = candidates[chosen_idx]
                    step = TrackStep(
                        i, OK, x=chosen.x, y=chosen.y,
                        residual=chosen.residual, method=chosen.method,
                        distance=dmin,
                        reason=(f"nearest of {len(candidates)} candidate(s) "
                                f"at distance {dmin:.6e} <= max_step "
                                f"{max_step:.6e}; branch continued"),
                        **common)

        steps.append(step)
        if step.status == OK:
            prev = (step.x, step.y)
        elif step.broken:
            prev = None

    return TrackResult(steps, max_step, tie_tol, anchor_tol)
