"""Trajectory mode: follow ONE inverse branch across a target sequence.

The single-point inverse (distortion.inverse.undistort_points)
returns every accepted candidate; for a continuous trajectory the
caller usually wants to stay on the same solution branch instead of
jumping between branches.  This module layers that policy on top of
the single-point solver WITHOUT weakening its guarantees:

* every step still runs the full single-point inverse and every
  reported point is a residual-checked candidate of that step;
* the start point is chosen by the caller from the accepted candidates
  of the first target and is verified against them;
* when the tracked branch disappears (no candidate near the previous
  point) the trajectory is interrupted with an explicit
  branch_changed / no_candidate status -- it never silently
  hops to another branch and calls it continuous;
* a step whose single-point search found nothing is reported as
  "search found no candidate"; only the radial certificate path may
  call it a proven no-solution, exactly as in the single-point API.
"""

import math

from .numeric import finite_real, finite_real_pair, checked_int
from .inverse import Candidate, undistort_points

__all__ = ["TrajectoryStep", "TrajectoryResult", "undistort_trajectory"]

#: Step statuses.
STATUS_START = "start"                    # caller-chosen anchor candidate
STATUS_CONTINUED = "continued"            # branch followed within max_step
STATUS_BRANCH_CHANGED = "branch_changed"  # nearest candidate beyond max_step
STATUS_NO_CANDIDATE = "no_candidate"      # step search accepted nothing


class TrajectoryStep:
    """One tracked step of a trajectory.

    Attributes:
        index     -- position in the target sequence
        xp, yp    -- the target of this step
        result    -- the full single-point InverseResult of this step
        candidate -- the tracked Candidate, or None when the branch was lost
        distance  -- distance from the previous tracked point (None for
                     the first step and for steps without a candidate)
        status    -- one of "start", "continued", "branch_changed",
                     "no_candidate"
    """

    __slots__ = ("index", "xp", "yp", "result", "candidate",
                 "distance", "status")

    def __init__(self, index, xp, yp, result, candidate, distance, status):
        self.index = index
        self.xp = xp
        self.yp = yp
        self.result = result
        self.candidate = candidate
        self.distance = distance
        self.status = status

    def __repr__(self):
        return ("TrajectoryStep(index=%r, target=(%r,%r), status=%r, "
                "distance=%r)" % (self.index, self.xp, self.yp,
                                  self.status, self.distance))


class TrajectoryResult:
    """Outcome of a trajectory query.

    Attributes:
        steps       -- list of TrajectoryStep, one per processed target
        complete    -- every target was tracked without interruption
        stop_reason -- human-readable reason the trajectory stopped, or
                       None when complete is True
    """

    __slots__ = ("steps", "complete", "stop_reason")

    def __init__(self, steps, complete, stop_reason):
        self.steps = steps
        self.complete = complete
        self.stop_reason = stop_reason

    def __len__(self):
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    @property
    def points(self):
        """Tracked (x, y) positions of the steps that have a candidate."""
        return [(s.candidate.x, s.candidate.y)
                for s in self.steps if s.candidate is not None]

    def __repr__(self):
        return ("TrajectoryResult(steps=%r, complete=%r, stop_reason=%r)"
                % (len(self.steps), self.complete, self.stop_reason))


def _nearest(candidates, x, y):
    """(candidate, distance) closest to (x, y), or (None, None)."""
    best, best_d = None, None
    for c in candidates:
        d = math.hypot(c.x - x, c.y - y)
        if best_d is None or d < best_d:
            best, best_d = c, d
    return best, best_d


def undistort_trajectory(model, targets, R, start, *, max_step,
                         residual_tol=1e-10, starts_per_axis=5,
                         max_iterations=80, follow_jumps=False,
                         start_match_tol=None):
    """Track one inverse branch across a sequence of targets.

    Parameters:
        targets     -- 1..100 (xp, yp) pairs (same batch limit as the
                       single-point batch API)
        R           -- search box half-width, shared by all steps
        start       -- caller-chosen anchor: a Candidate or (x, y) pair
                       that must match an accepted candidate of the
                       FIRST target within start_match_tol
        max_step    -- maximum allowed distance between consecutive
                       tracked points; a larger nearest-candidate
                       distance means the tracked branch disappeared
        follow_jumps-- False (default): stop at the first branch change;
                       True: keep going on the nearest branch but flag
                       the step as "branch_changed" (never silently
                       relabelled as continuous)
        start_match_tol -- match tolerance for the anchor, default
                       1e-6 * max(1, R)
        residual_tol, starts_per_axis, max_iterations -- forwarded to
                       the single-point solver of every step

    Everything is validated before any solver runs, so a rejected call
    never leaves partially computed steps.
    """
    cleaned = model._check_batch(targets)
    if not cleaned:
        raise ValueError("targets: trajectory needs at least 1 point")
    R = finite_real(R, "R")
    if R <= 0.0:
        raise ValueError("R: search box half-width must be > 0, got %r" % (R,))
    if isinstance(start, Candidate):
        sx, sy = start.x, start.y
    elif isinstance(start, (list, tuple)) and len(start) == 2:
        sx, sy = start
    else:
        raise TypeError(
            "start: expected a Candidate or (x, y) pair, got %s"
            % type(start).__name__
        )
    sx, sy = finite_real_pair(sx, sy, ("start.x", "start.y"))
    max_step = finite_real(max_step, "max_step")
    if max_step <= 0.0:
        raise ValueError("max_step: must be > 0, got %r" % (max_step,))
    residual_tol = finite_real(residual_tol, "residual_tol")
    if residual_tol <= 0.0:
        raise ValueError("residual_tol: must be > 0, got %r" % (residual_tol,))
    n = checked_int(starts_per_axis, "starts_per_axis", minimum=1, maximum=12)
    iters = checked_int(max_iterations, "max_iterations",
                        minimum=1, maximum=10000)
    if not isinstance(follow_jumps, bool):
        raise TypeError(
            "follow_jumps: expected bool, got %s" % type(follow_jumps).__name__)
    if start_match_tol is None:
        match_tol = 1e-6 * max(1.0, R)
    else:
        match_tol = finite_real(start_match_tol, "start_match_tol")
        if match_tol <= 0.0:
            raise ValueError(
                "start_match_tol: must be > 0, got %r" % (start_match_tol,))

    kwargs = dict(residual_tol=residual_tol, starts_per_axis=n,
                  max_iterations=iters)

    steps = []
    prev = None
    for index, (xp, yp) in enumerate(cleaned):
        result = undistort_points(model, xp, yp, R, **kwargs)

        if index == 0:
            anchor, dist = _nearest(result.candidates, sx, sy)
            if anchor is None or dist > match_tol:
                raise ValueError(
                    "start: (%r, %r) does not match any accepted candidate "
                    "of the first target within start_match_tol=%g "
                    "(nearest distance %r); choose the start from the "
                    "candidates of undistort() on the first target"
                    % (sx, sy, match_tol, dist)
                )
            steps.append(TrajectoryStep(0, xp, yp, result, anchor,
                                        None, STATUS_START))
            prev = anchor
            continue

        if not result.candidates:
            steps.append(TrajectoryStep(index, xp, yp, result, None,
                                        None, STATUS_NO_CANDIDATE))
            if result.proven_no_solution:
                reason = ("step %d: the radial certificate PROVES the box "
                          "has no preimage of (%r, %r); the tracked branch "
                          "provably ends here" % (index, xp, yp))
            else:
                reason = ("step %d: the search found NO candidate for "
                          "(%r, %r); this is NOT a proof that no preimage "
                          "exists -- the tracked branch is lost and the "
                          "trajectory stops" % (index, xp, yp))
            return TrajectoryResult(steps, False, reason)

        cand, dist = _nearest(result.candidates, prev.x, prev.y)
        if dist <= max_step:
            steps.append(TrajectoryStep(index, xp, yp, result, cand,
                                        dist, STATUS_CONTINUED))
            prev = cand
            continue

        # The branch being tracked has no continuation within max_step.
        steps.append(TrajectoryStep(index, xp, yp, result, cand,
                                    dist, STATUS_BRANCH_CHANGED))
        reason = ("step %d: nearest accepted candidate is %.6g away from "
                  "the tracked point, beyond max_step=%.6g; the tracked "
                  "branch disappeared (a different branch may exist at "
                  "(%r, %r))" % (index, dist, max_step, cand.x, cand.y))
        if not follow_jumps:
            return TrajectoryResult(steps, False, reason)
        # Explicitly follow the other branch; the step stays flagged.
        prev = cand

    return TrajectoryResult(steps, True, None)

