"""Weighted fit of the single radial coefficient k1.

The focal length ``f`` and principal point ``(cx, cy)`` are fixed and
known; the tangential coefficients and k2 are held at zero (this
iteration fits exactly one radial coefficient).  Reusing the forward
formula of :mod:`distortion.model`, a normalized-plane point ``(x, y)``
with r^2 = x^2 + y^2 is predicted at pixel coordinates

    u = f * x * (1 + k1 * r^2) + cx
    v = f * y * (1 + k1 * r^2) + cy.

Given weighted correspondences ((x, y), (u, v)) the fit minimizes

    C(k1) = sum_i w_i * [(u_i - u_i_hat)^2 + (v_i - v_i_hat)^2].

With the correspondences fixed this objective is *linear* in k1, so it
has a unique global weighted-least-squares minimum whenever the normal
scalar A = f^2 * sum_i w_i r_i^6 is positive.  The minimization is
nevertheless carried out by a genuine iterative Gauss-Newton loop with
Armijo back-tracking (one accepted step reaches the linear minimum); the
result honestly reports the iteration budget, the iterations used, the
residual and the convergence status.  A == 0 (every correspondence has
r = 0, so k1 is unobservable) is a *degenerate geometry*: the caller
gets a diagnostic result instead of an exception, while structurally
invalid input (NaN, negative weight, ...) raises before any iteration.

Only the Python 3.14 standard library is used.
"""

import math

from .model import DistortionModel, MAX_POINTS
from .numeric import finite_real, checked_int

__all__ = ["RadialFitResult", "fit_radial_k1"]

# Convergence test on the magnitude of the Gauss-Newton step that would
# be taken next: |dk| <= XTOL * max(1, |k1|).
XTOL = 1e-12
_DEFAULT_MAX_ITERATIONS = 20


class RadialFitResult:
    """Outcome of :func:`fit_radial_k1`.

    Attributes:
        k1             -- fitted coefficient (equals ``initial_k1`` when
                          the geometry is degenerate)
        initial_k1     -- starting value of the iteration
        f, cx, cy      -- validated, fixed intrinsics used by the fit
        model          -- DistortionModel(k1=k1, k2=p1=p2=0) reusing the
                          existing forward model
        n_points       -- number of correspondences
        total_weight   -- sum of the per-point weights
        cost           -- weighted sum of squared pixel residuals
        rms            -- sqrt(cost / total_weight): weighted RMS
                          point-to-point distance, in pixels
        iterations     -- accepted parameter updates (0 if degenerate)
        max_iterations -- the iteration budget
        converged      -- True exactly when status == "converged"
        status         -- one of "converged", "max_iterations",
                          "degenerate"
        observability  -- sqrt(A) = f*sqrt(sum w_i r_i^6); zero means
                          k1 is unobservable; larger means better
                          conditioned in the presence of pixel noise
        diagnostic     -- human-readable explanation / certificate
    """

    __slots__ = ("k1", "initial_k1", "f", "cx", "cy", "model",
                 "n_points", "total_weight", "cost", "rms",
                 "iterations", "max_iterations", "converged", "status",
                 "observability", "diagnostic")

    def __init__(self, **kw):
        for attr in self.__slots__:
            setattr(self, attr, kw[attr])

    def __repr__(self):
        return (f"RadialFitResult(k1={self.k1!r}, rms={self.rms:.3e}, "
                f"iterations={self.iterations}/{self.max_iterations}, "
                f"status={self.status!r}, n={self.n_points})")


def _check_correspondences(correspondences):
    """Validate the whole batch up front; nothing is built on failure."""
    if not isinstance(correspondences, (list, tuple)):
        raise TypeError(
            "correspondences: expected list/tuple of ((x, y), (u, v)) "
            f"pairs, got {type(correspondences).__name__}"
        )
    n = len(correspondences)
    if n == 0:
        raise ValueError(
            "correspondences: at least one point pair is required, got 0"
        )
    if n > MAX_POINTS:
        raise ValueError(
            f"correspondences: batch size {n} exceeds the interface "
            f"limit of {MAX_POINTS}"
        )

    cleaned = []
    for i, item in enumerate(correspondences):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise TypeError(
                f"correspondences[{i}]: expected an ((x, y), (u, v)) "
                "pair of length 2"
            )
        xy, uv = item
        if not isinstance(xy, (list, tuple)) or len(xy) != 2:
            raise TypeError(
                f"correspondences[{i}].normalized: expected an (x, y) "
                "pair of length 2"
            )
        if not isinstance(uv, (list, tuple)) or len(uv) != 2:
            raise TypeError(
                f"correspondences[{i}].observed: expected a (u, v) "
                "pair of length 2"
            )
        x = finite_real(xy[0], f"correspondences[{i}].x")
        y = finite_real(xy[1], f"correspondences[{i}].y")
        u = finite_real(uv[0], f"correspondences[{i}].u")
        v = finite_real(uv[1], f"correspondences[{i}].v")
        cleaned.append((x, y, u, v))
    return cleaned


def _check_weights(weights, n):
    """Return n positive finite floats; default weight is 1.0."""
    if weights is None:
        return [1.0] * n
    if not isinstance(weights, (list, tuple)):
        raise TypeError(
            "weights: expected None or a list/tuple of positive "
            f"numbers, got {type(weights).__name__}"
        )
    if len(weights) != n:
        raise ValueError(
            f"weights: length {len(weights)} does not match the "
            f"{n} correspondences"
        )
    out = []
    for i, w in enumerate(weights):
        wf = finite_real(w, f"weights[{i}]")
        if wf <= 0.0:
            raise ValueError(
                f"weights[{i}]: must be > 0, got {wf!r}")
        out.append(wf)
    return out


def fit_radial_k1(correspondences, f, cx, cy, *, weights=None,
                  initial_k1=0.0, max_iterations=_DEFAULT_MAX_ITERATIONS):
    """Fit one radial coefficient k1 from weighted correspondences.

    Arguments:
        correspondences -- list/tuple of ``((x, y), (u, v))`` pairs,
                           1..100 items; ``(x, y)`` are normalized-plane
                           coordinates (the input domain of
                           ``DistortionModel.distort``) and ``(u, v)``
                           the observed pixel coordinates
        f               -- fixed focal length, a finite real > 0
        cx, cy          -- fixed principal point, finite reals
        weights         -- None (all ones) or one positive finite
                           weight per correspondence
        initial_k1      -- finite starting value for the iteration
        max_iterations  -- genuine int in [1, 10000], the iteration
                           budget (default 20)

    Returns a :class:`RadialFitResult`; the fitter is a pure function,
    so an invalid call raises (TypeError/ValueError) before any state is
    produced and never returns a partially computed result.
    """
    points = _check_correspondences(correspondences)
    f = finite_real(f, "f")
    if f <= 0.0:
        raise ValueError(f"f: focal length must be > 0, got {f!r}")
    cx = finite_real(cx, "cx")
    cy = finite_real(cy, "cy")
    w = _check_weights(weights, len(points))
    k0 = finite_real(initial_k1, "initial_k1")
    budget = checked_int(max_iterations, "max_iterations",
                         minimum=1, maximum=10000)

    # Scalar weighted least squares, with
    #   jx = f*x*r^2, jy = f*y*r^2          (d pred / d k1)
    #   tx = u - cx - f*x, ty = v - cy - f*y  (residual at k1 = 0)
    # A = sum w (jx^2+jy^2) = f^2 sum w r^6
    # b = sum w (jx*tx + jy*ty)
    # cost(k) = c0 - 2 k b + k^2 A,  optimum k* = b / A
    A = 0.0
    b = 0.0
    c0 = 0.0
    total_weight = 0.0
    for (x, y, u, v), wi in zip(points, w):
        r2 = x * x + y * y
        jx = f * x * r2
        jy = f * y * r2
        tx = u - cx - f * x
        ty = v - cy - f * y
        A += wi * (jx * jx + jy * jy)
        b += wi * (jx * tx + jy * ty)
        c0 += wi * (tx * tx + ty * ty)
        total_weight += wi

    n = len(points)

    def cost_quadratic(k):
        # Closed quadratic form -- cheap inside the line search, but
        # c0 - 2 k b + k^2 A suffers catastrophic cancellation near the
        # minimum, so the reported final cost is recomputed directly.
        return c0 - 2.0 * k * b + k * k * A

    def cost_direct(k):
        """Weighted residual sum evaluated straight from the raw points."""
        total = 0.0
        for (x, y, u, v), wi in zip(points, w):
            s = 1.0 + k * (x * x + y * y)
            du = u - (cx + f * x * s)
            dv = v - (cy + f * y * s)
            total += wi * (du * du + dv * dv)
        return total

    if A == 0.0:
        # Every positively weighted point has r = 0: k1 does not enter
        # the prediction.  Report the geometry as degenerate; the
        # residual at the initial value is still meaningful.
        k = k0
        resid_cost = cost_direct(k)
        rms = math.sqrt(resid_cost / total_weight)
        diagnostic = (
            f"degenerate geometry: all {n} correspondence(es) have "
            "normalized radius r = 0 (x = y = 0), so the normal scalar "
            "A = f^2 * sum(w_i r_i^6) is exactly 0 and k1 is "
            "unobservable; no iteration was run and k1 is kept at its "
            f"initial value {k0!r}. Add at least one correspondence "
            "away from the normalized origin. Weighted RMS residual at "
            f"the principal-point prediction is {rms:.6g} px."
        )
        return RadialFitResult(
            k1=k, initial_k1=k0, f=f, cx=cx, cy=cy,
            model=DistortionModel(k1=k),
            n_points=n, total_weight=total_weight,
            cost=resid_cost, rms=rms,
            iterations=0, max_iterations=budget,
            converged=False, status="degenerate",
            observability=0.0, diagnostic=diagnostic)

    # Gauss-Newton with Armijo back-tracking on the single parameter.
    # The model is linear in k1, so the unit step k <- b/A lands on the
    # global weighted-LS minimum; the loop machinery is what honors the
    # declared iteration budget and convergence contract.
    k = k0
    cur = cost_quadratic(k)
    iterations = 0
    status = "max_iterations"
    for it in range(1, budget + 1):
        g = 2.0 * (A * k - b)          # d cost / d k1
        dk = -g / (2.0 * A)            # Gauss-Newton step
        alpha = 1.0
        accepted = None
        for _ in range(80):
            kn = k + alpha * dk
            cn = cost_quadratic(kn)
            # Armijo sufficient decrease; g*dk <= 0 by construction.
            if cn <= cur + 1e-4 * alpha * g * dk:
                accepted = (kn, cn)
                break
            alpha *= 0.5
            if alpha < 1e-16:
                break
        if accepted is None:
            status = "max_iterations"
            break
        k, cur = accepted
        iterations = it
        next_step = (b - A * k) / A
        if abs(next_step) <= XTOL * max(1.0, abs(k)):
            status = "converged"
            break

    # Recompute the reported residual directly from the raw points, so
    # cancellation in the quadratic form never inflates an exact fit.
    cur = cost_direct(k)
    rms = math.sqrt(cur / total_weight)
    if status == "converged":
        diagnostic = (
            f"weighted Gauss-Newton converged after {iterations} "
            f"accepted step(s) of budget {budget}; weighted RMS point "
            f"residual {rms:.6g} px over {n} correspondence(es), total "
            f"weight {total_weight:g}. The single-coefficient model is "
            "linear in k1 for fixed correspondences, so the accepted "
            "step equals the unique global weighted-least-squares "
            "minimum (A = f^2 sum w r^6 = "
            f"{A:.6g})."
        )
    else:
        diagnostic = (
            f"weighted Gauss-Newton used all {iterations} accepted "
            f"step(s) of budget {budget} without meeting the step "
            f"tolerance {XTOL:g}; the returned k1 and residual are the "
            "best reached so far but convergence is NOT established. "
            f"Weighted RMS point residual is {rms:.6g} px."
        )

    return RadialFitResult(
        k1=k, initial_k1=k0, f=f, cx=cx, cy=cy,
        model=DistortionModel(k1=k),
        n_points=n, total_weight=total_weight,
        cost=cur, rms=rms,
        iterations=iterations, max_iterations=budget,
        converged=status == "converged", status=status,
        observability=math.sqrt(A), diagnostic=diagnostic)
