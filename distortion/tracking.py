"""Continuous-branch tracking across a sequence of distorted targets.

The bounded inverse solver (distortion.inverse.undistort_points)
returns *every* residual-acceptable candidate; when the forward map
folds, one distorted point can legitimately have several preimages.
Choosing candidates[0] independently for every frame silently hops
between branches.

This module adds a tracking entry point.  The caller supplies a
sequence of distorted coordinates plus the first *undistorted* anchor.
Each frame still runs the original completeness-limited solver; the
tracker then selects the accepted candidate nearest to the previous
result, provided that distance does not exceed an explicit max_step.
Ties within equal_tol are reported as "ambiguous" with every tied
candidate retained -- list order is never used to break a tie.

Branch continuity is a tracking heuristic, NOT a uniqueness theorem:
the underlying solver keeps all of its documented completeness
limitations (general-model multi-start search is not exhaustive) and
the radial special case keeps its segment certificate.  After an
ambiguous / no_candidate / step_too_large frame the track is broken
and a new forward-verified anchor is required before any further frame
can be selected.
"""

import math

from .numeric import finite_real, finite_real_pair, checked_int
from .inverse import Candidate, undistort_points

__all__ = [
    "TrackStep",
    "InverseTracker",
    "track_points",
    "ANCHORED",
    "OK",
    "AMBIGUOUS",
    "NO_CANDIDATE",
    "STEP_TOO_LARGE",
    "ANCHOR_REJECTED",
    "BROKEN_NEEDS_ANCHOR",
]

ANCHORED = "anchored"
OK = "ok"
AMBIGUOUS = "ambiguous"
NO_CANDIDATE = "no_candidate"
STEP_TOO_LARGE = "step_too_large"
ANCHOR_REJECTED = "anchor_rejected"
BROKEN_NEEDS_ANCHOR = "broken_needs_anchor"


class TrackStep:
    """Outcome of one tracked frame.

    Attributes:
        index      -- frame number (0 is the anchor frame)
        target     -- distorted coordinate (xp, yp)
        status     -- one of the module-level status constants
        selected   -- chosen Candidate, or None when nothing was chosen
        previous   -- (x, y) the distance was measured from, or None
        candidates -- all residual-accepted solver candidates, retained
                      (including on ambiguous / broken frames)
        distances  -- distance from the previous point per candidate
        tied       -- candidates tied for the minimum within equal_tol
        reason     -- human-readable explanation / break reason
        result     -- raw InverseResult of the original solver, so its
                      completeness flags and radial certificate survive
    """

    __slots__ = ("index", "target", "status", "selected", "previous",
                 "candidates", "distances", "tied", "reason", "result")

    def __init__(self, index, target, status, *, selected=None,
                 previous=None, candidates=(), distances=(), tied=(),
                 reason="", result=None):
        self.index = index
        self.target = target
        self.status = status
        self.selected = selected
        self.previous = previous
        self.candidates = list(candidates)
        self.distances = list(distances)
        self.tied = list(tied)
        self.reason = reason
        self.result = result

    @property
    def point(self):
        """Selected (x, y), or None when the frame was not resolved."""
        if self.selected is None:
            return None
        return self.selected.x, self.selected.y

    def __repr__(self):
        if self.selected is None:
            pt = "None"
        else:
            pt = f"({self.selected.x!r}, {self.selected.y!r})"
        return (f"TrackStep(index={self.index}, status={self.status!r}, "
                f"selected={pt}, n_candidates={len(self.candidates)})")


class InverseTracker:
    """Stateful continuous-branch tracker.

    Parameters:
        model      -- DistortionModel
        R          -- search box half-width (same bounded inverse)
        max_step   -- hard upper bound on the undistorted-space step
                      between consecutive selected roots (> 0)
        equal_tol  -- relative gap within which two distances count as a
                      tie: |d1-d2| <= equal_tol*max(1, d_min)
        anchor_tol -- forward re-projection tolerance for an anchor
        residual_tol, starts_per_axis, max_iterations -- forwarded to
                      the original inverse solver unchanged

    Lifecycle: start() verifies the anchor; subsequent targets are fed
    with update().  Once a frame breaks the track, further updates
    return BROKEN_NEEDS_ANCHOR until start() is called again with a new
    anchor.
    """

    def __init__(self, model, R, *, max_step, equal_tol=1e-9,
                 anchor_tol=1e-7, residual_tol=1e-10,
                 starts_per_axis=5, max_iterations=80):
        self.model = model
        self.R = finite_real(R, "R")
        if self.R <= 0.0:
            raise ValueError(f"R: must be > 0, got {R!r}")
        self.max_step = finite_real(max_step, "max_step")
        if self.max_step <= 0.0:
            raise ValueError(f"max_step: must be > 0, got {max_step!r}")
        self.equal_tol = finite_real(equal_tol, "equal_tol")
        if self.equal_tol <= 0.0:
            raise ValueError(f"equal_tol: must be > 0, got {equal_tol!r}")
        self.anchor_tol = finite_real(anchor_tol, "anchor_tol")
        if self.anchor_tol <= 0.0:
            raise ValueError(f"anchor_tol: must be > 0, got {anchor_tol!r}")
        self.residual_tol = finite_real(residual_tol, "residual_tol")
        if self.residual_tol <= 0.0:
            raise ValueError(
                f"residual_tol: must be > 0, got {residual_tol!r}")
        self.starts_per_axis = checked_int(
            starts_per_axis, "starts_per_axis", minimum=1, maximum=12)
        self.max_iterations = checked_int(
            max_iterations, "max_iterations",
            minimum=1, maximum=10000)
        self.active = False
        self.last = None

    def _solve(self, xp, yp):
        return undistort_points(
            self.model, xp, yp, self.R,
            residual_tol=self.residual_tol,
            starts_per_axis=self.starts_per_axis,
            max_iterations=self.max_iterations)

    def start(self, xp, yp, anchor_x, anchor_y):
        """Verify (anchor_x, anchor_y) against target (xp, yp).

        The anchor must lie inside the box and re-project onto the
        target within anchor_tol.  Otherwise the frame returns with
        status ANCHOR_REJECTED (the solver candidates are retained for
        diagnostics) and the tracker stays inactive.
        """
        xp, yp = finite_real_pair(xp, yp, ("xp", "yp"))
        ax, ay = finite_real_pair(anchor_x, anchor_y,
                                  ("anchor_x", "anchor_y"))
        result = self._solve(xp, yp)
        if not (-self.R <= ax <= self.R and -self.R <= ay <= self.R):
            self.active = False
            self.last = None
            return TrackStep(
                0, (xp, yp), ANCHOR_REJECTED,
                candidates=result.candidates,
                reason=(f"anchor ({ax!r}, {ay!r}) lies outside the "
                        f"search box [-{self.R!r}, {self.R!r}]^2; no "
                        f"track started"),
                result=result)
        rx, ry = self.model.distort(ax, ay)
        residual = math.hypot(rx - xp, ry - yp)
        if residual > self.anchor_tol:
            self.active = False
            self.last = None
            return TrackStep(
                0, (xp, yp), ANCHOR_REJECTED,
                candidates=result.candidates,
                reason=(f"anchor forward check failed: "
                        f"distort({ax!r}, {ay!r}) = ({rx!r}, {ry!r}), "
                        f"residual {residual:.3e} exceeds anchor_tol "
                        f"{self.anchor_tol:.0e}; anchor is not a "
                        f"preimage of the first target"),
                result=result)
        anchor = Candidate(ax, ay, residual, "anchor_forward_check")
        self.last = (ax, ay)
        self.active = True
        return TrackStep(
            0, (xp, yp), ANCHORED, selected=anchor,
            previous=None,
            candidates=[anchor] + list(result.candidates),
            reason=(f"anchor accepted: forward residual {residual:.3e} "
                    f"<= anchor_tol {self.anchor_tol:.0e}"),
            result=result)

    def update(self, index, xp, yp):
        """Process one non-anchor frame; see the module docstring."""
        xp, yp = finite_real_pair(xp, yp, ("xp", "yp"))
        if not self.active:
            return TrackStep(
                index, (xp, yp), BROKEN_NEEDS_ANCHOR,
                reason=("track is broken; a new forward-verified anchor "
                        "is required before further frames can be "
                        "selected"))

        result = self._solve(xp, yp)
        px, py = self.last
        candidates = list(result.candidates)
        pairs = [(c, math.hypot(c.x - px, c.y - py)) for c in candidates]

        if not pairs:
            self.active = False
            self.last = None
            reason = ("solver returned no residual-acceptable "
                      "candidate; track broken")
            if result.proven_no_solution:
                reason += " (radial certificate: no preimage in box)"
            elif result.budget_exhausted:
                reason += (" (general-model search budget exhausted; "
                           "completeness was never proven)")
            return TrackStep(index, (xp, yp), NO_CANDIDATE,
                             previous=(px, py), candidates=candidates,
                             distances=[d for _, d in pairs],
                             reason=reason, result=result)

        feasible = [(c, d) for c, d in pairs if d <= self.max_step]
        if not feasible:
            self.active = False
            self.last = None
            nearest = min(d for _, d in pairs)
            reason = (
                f"nearest accepted candidate is {nearest:.3e} from the "
                f"previous result, exceeding max_step {self.max_step:g}; "
                f"track broken (a jump to another folded branch is not "
                f"taken silently)")
            return TrackStep(index, (xp, yp), STEP_TOO_LARGE,
                             previous=(px, py), candidates=candidates,
                             distances=[d for _, d in pairs],
                             reason=reason, result=result)

        dmin = min(d for _, d in feasible)
        tie_band = self.equal_tol * max(1.0, dmin)
        tied = [c for c, d in feasible if abs(d - dmin) <= tie_band]
        if len(tied) >= 2:
            # Never break a tie by list order: retain all tied roots
            # and force the caller to supply a new anchor.
            self.active = False
            self.last = None
            reason = (
                f"{len(tied)} feasible candidates tie for nearest at "
                f"distance {dmin:.6e} (within equal_tol band "
                f"{tie_band:.2e}); ambiguous -- no candidate selected "
                f"by list order, track broken, supply a new anchor")
            return TrackStep(index, (xp, yp), AMBIGUOUS,
                             previous=(px, py), candidates=candidates,
                             distances=[d for _, d in pairs], tied=tied,
                             reason=reason, result=result)

        chosen, chosen_d = min(feasible, key=lambda cd: cd[1])
        self.last = (chosen.x, chosen.y)
        return TrackStep(
            index, (xp, yp), OK, selected=chosen,
            previous=(px, py), candidates=candidates,
            distances=[d for _, d in pairs],
            reason=(f"nearest of {len(candidates)} accepted "
                    f"candidate(s): step {chosen_d:.6e} <= max_step "
                    f"{self.max_step:g}"),
            result=result)


def track_points(model, targets, anchor, R, *, max_step, **kwargs):
    """Track one continuous branch over a sequence of distorted points.

    targets is a non-empty list/tuple of at most 100 distorted
    (xp, yp) pairs; anchor is the caller-supplied undistorted (x, y)
    of the FIRST target.  Returns one TrackStep per target (index 0 is
    the anchor frame).  After a break, remaining frames come back as
    BROKEN_NEEDS_ANCHOR; resuming requires a fresh anchor (another
    call or InverseTracker.start).
    """
    if not isinstance(targets, (list, tuple)):
        raise TypeError(
            f"targets: expected list/tuple of (xp, yp) pairs, got "
            f"{type(targets).__name__}")
    if not 1 <= len(targets) <= 100:
        raise ValueError(
            f"targets: need between 1 and 100 points, got {len(targets)}")
    cleaned = []
    for i, p in enumerate(targets):
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise TypeError(f"targets[{i}]: expected a pair of length 2")
        cleaned.append(finite_real_pair(
            p[0], p[1], (f"targets[{i}].xp", f"targets[{i}].yp")))
    if not isinstance(anchor, (list, tuple)) or len(anchor) != 2:
        raise TypeError("anchor: expected an (anchor_x, anchor_y) pair")
    ax, ay = finite_real_pair(anchor[0], anchor[1],
                              ("anchor_x", "anchor_y"))

    tracker = InverseTracker(model, R, max_step=max_step, **kwargs)
    steps = [tracker.start(cleaned[0][0], cleaned[0][1], ax, ay)]
    for i in range(1, len(cleaned)):
        if tracker.active:
            steps.append(tracker.update(i, cleaned[i][0], cleaned[i][1]))
        else:
            steps.append(TrackStep(
                i, cleaned[i], BROKEN_NEEDS_ANCHOR,
                reason=("earlier frame broke the track; a new anchor is "
                        "required (no candidate is chosen here)")))
    return steps
