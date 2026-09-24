"""Weighted fit of a single radial coefficient from point correspondences.

The focal length (fx, fy) and principal point (cx, cy) are FIXED and
known; only the first radial coefficient ``k1`` is estimated, using the
same radial formula as :class:`distortion.model.DistortionModel` with
``k2 = p1 = p2 = 0``::

    x' = x*(1 + k1*r^2),  y' = y*(1 + k1*r^2),  r^2 = x^2 + y^2

on the normalized plane, where normalized coordinates are obtained from
pixel coordinates by ``a = (u - cx)/fx``, ``b = (v - cy)/fy``.

Each correspondence is an (undistorted, observed-distorted) pair with an
optional non-negative weight; the estimator minimizes the weighted sum
of squared residuals

    SSE(k1) = sum_i w_i * ((ad_i - a_i(1 + k1 q_i))^2
                           + (bd_i - b_i(1 + k1 q_i))^2),  q_i = a_i^2+b_i^2.

Because the model is LINEAR in k1, the Gauss-Newton normal equation is a
scalar and one full step reaches the weighted least-squares optimum
exactly (up to round-off); an Armijo back-tracking guard is kept for
extreme starting values.  Degenerate inputs (zero total weight, every
undistorted point at the principal point, or no numerically resolvable
curvature leverage) are NOT exceptions: they return a
:class:`RadialFitResult` with ``status == "degenerate"`` carrying a
human-readable diagnostic, leaving the caller's state untouched.
"""

import math

from .model import DistortionModel
from .numeric import finite_real, checked_int

__all__ = ["RadialFitResult", "fit_radial_coefficient"]

_EPS = 2.220446049250313e-16

# Status strings.
CONVERGED = "converged"
MAX_ITERATIONS = "max_iterations"
DEGENERATE = "degenerate"
STALLED = "stalled"


class RadialFitResult:
    """Outcome of :func:`fit_radial_coefficient`.

    Attributes:
        k1            -- estimated coefficient (equals initial_k1 on a
                         degenerate/stalled result that never moved)
        initial_k1    -- the start value used by the iteration
        fx, fy, cx, cy -- validated fixed intrinsics echoed back
        n_points      -- number of correspondences supplied
        weight_sum    -- sum of the supplied weights
        iterations    -- accepted parameter updates performed
        max_iterations-- the iteration budget
        converged     -- True exactly when status == "converged"
        status        -- one of "converged", "max_iterations",
                         "degenerate", "stalled"
        sse           -- weighted sum of squared residuals at k1
        rms_residual  -- sqrt(SSE / (2 * weight_sum)); per-coordinate
                         weighted RMS; 0.0 when weight_sum == 0
        max_residual  -- max unweighted ||(predicted)-(observed)||_2
                         over the correspondences, in normalized units
        final_step    -- magnitude of the last attempted Gauss-Newton
                         step (0.0 on a degenerate result)
        diagnostic    -- human-readable explanation of the outcome
    """

    __slots__ = ("k1", "initial_k1", "fx", "fy", "cx", "cy",
                 "n_points", "weight_sum", "iterations", "max_iterations",
                 "converged", "status", "sse", "rms_residual",
                 "max_residual", "final_step", "diagnostic")

    def __init__(self, **kw):
        for name in self.__slots__:
            setattr(self, name, kw[name])

    @property
    def model(self):
        """A radial-only :class:`DistortionModel` built from k1."""
        return DistortionModel(k1=self.k1)

    def __repr__(self):
        return (f"RadialFitResult(k1={self.k1!r}, status={self.status!r}, "
                f"iterations={self.iterations}/{self.max_iterations}, "
                f"rms_residual={self.rms_residual:.3e})")


def _validate_correspondences(correspondences):
    """Validate shape and finiteness before anything is computed.

    Returns a list of (u, v, ud, vd, w) tuples.  The whole batch is
    checked first, so an invalid call never produces a partial result.
    """
    if not isinstance(correspondences, (list, tuple)):
        raise TypeError(
            "correspondences: expected a list/tuple of "
            "(u, v, ud, vd) or (u, v, ud, vd, w) records, got "
            + type(correspondences).__name__
        )
    if len(correspondences) == 0:
        raise ValueError("correspondences: at least one pair is required")
    cleaned = []
    for i, rec in enumerate(correspondences):
        if not isinstance(rec, (list, tuple)) or len(rec) not in (4, 5):
            raise TypeError(
                f"correspondences[{i}]: expected a record of 4 or 5 "
                f"numbers (u, v, ud, vd[, weight]), got length "
                f"{len(rec) if isinstance(rec, (list, tuple)) else '?'}"
            )
        u = finite_real(rec[0], f"correspondences[{i}].u")
        v = finite_real(rec[1], f"correspondences[{i}].v")
        ud = finite_real(rec[2], f"correspondences[{i}].ud")
        vd = finite_real(rec[3], f"correspondences[{i}].vd")
        w = 1.0 if len(rec) == 4 else finite_real(rec[4],
                                                  f"correspondences[{i}].w")
        if w < 0.0:
            raise ValueError(
                f"correspondences[{i}].w: weight must be >= 0, got {w!r}")
        cleaned.append((u, v, ud, vd, w))
    return cleaned


def _metrics(data, k):
    """Weighted SSE, max unweighted residual norm for coefficient k."""
    sse = 0.0
    max_res = 0.0
    for a, b, ad, bd, w in data:
        q = a * a + b * b
        f_a = ad - a * (1.0 + k * q)
        f_b = bd - b * (1.0 + k * q)
        sse += w * (f_a * f_a + f_b * f_b)
        rn = math.hypot(f_a, f_b)
        if rn > max_res:
            max_res = rn
    return sse, max_res


def fit_radial_coefficient(correspondences, *, fx=1.0, fy=1.0,
                           cx=0.0, cy=0.0, initial_k1=0.0,
                           max_iterations=50, xtol=1e-12):
    """Fit one radial coefficient ``k1`` from weighted correspondences.

    Parameters
    ----------
    correspondences : list/tuple of records
        Each record is ``(u, v, ud, vd)`` (unit weight) or
        ``(u, v, ud, vd, w)`` with ``w >= 0``.  ``(u, v)`` is the
        undistorted point and ``(ud, vd)`` the observed distorted point,
        both in the SAME pixel (or already-normalized with fx=fy=1 and
        cx=cy=0) coordinate frame.
    fx, fy : float, strictly positive and finite
        Fixed known focal lengths.
    cx, cy : float, finite
        Fixed known principal point.
    initial_k1 : float, finite
        Starting value of the iteration.
    max_iterations : genuine int in [1, 10000] (bool rejected)
        Accepted-update budget.
    xtol : float, strictly positive and finite
        Convergence threshold on the parameter step: converged when the
        Gauss-Newton step magnitude is at most
        ``xtol * max(1, |k1|)``.

    Returns
    -------
    RadialFitResult
        A degenerate point set returns status ``"degenerate"`` with a
        diagnostic instead of raising.  Type/shape/NaN errors and
        out-of-range intrinsics or options raise TypeError/ValueError
        before any computation.
    """
    raw = _validate_correspondences(correspondences)

    fx = finite_real(fx, "fx")
    fy = finite_real(fy, "fy")
    cx = finite_real(cx, "cx")
    cy = finite_real(cy, "cy")
    if fx <= 0.0:
        raise ValueError(f"fx: focal length must be > 0, got {fx!r}")
    if fy <= 0.0:
        raise ValueError(f"fy: focal length must be > 0, got {fy!r}")
    k = finite_real(initial_k1, "initial_k1")
    k0 = k
    budget = checked_int(max_iterations, "max_iterations",
                         minimum=1, maximum=10000)
    xtol = finite_real(xtol, "xtol")
    if xtol <= 0.0:
        raise ValueError(f"xtol: must be > 0, got {xtol!r}")

    # Normalize once; the fit is entirely in normalized coordinates.
    data = []
    weight_sum = 0.0
    max_radius = 0.0  # of the UNDISTORTED points with positive weight
    for u, v, ud, vd, w in raw:
        a = (u - cx) / fx
        b = (v - cy) / fy
        ad = (ud - cx) / fx
        bd = (vd - cy) / fy
        data.append((a, b, ad, bd, w))
        weight_sum += w
        if w > 0.0:
            rad = math.hypot(a, b)
            if rad > max_radius:
                max_radius = rad

    n_points = len(data)

    def finish(kk, iterations, status, step, diagnostic):
        sse, max_res = _metrics(data, kk)
        rms = math.sqrt(sse / (2.0 * weight_sum)) if weight_sum > 0.0 else 0.0
        return RadialFitResult(
            k1=kk, initial_k1=k0, fx=fx, fy=fy, cx=cx, cy=cy,
            n_points=n_points, weight_sum=weight_sum,
            iterations=iterations, max_iterations=budget,
            converged=(status == CONVERGED), status=status,
            sse=sse, rms_residual=rms, max_residual=max_res,
            final_step=step, diagnostic=diagnostic,
        )

    # --- Degeneracy diagnosis (done before touching the iterate) -------
    if weight_sum <= 0.0:
        return finish(k, 0, DEGENERATE, 0.0,
                      "degenerate fit: every correspondence has zero "
                      "weight; the weighted normal equation is empty "
                      "and k1 is unobservable.")
    if not any(w > 0.0 and (a != 0.0 or b != 0.0)
               for a, b, ad, bd, w in data):
        return finish(k, 0, DEGENERATE, 0.0,
                      "degenerate fit: every positively weighted "
                      "undistorted point is at the principal point "
                      "(a=b=0); radial scaling leaves the origin fixed "
                      "so k1 is unobservable. Residual offsets at the "
                      "principal point cannot be explained by this "
                      "model.")

    # A = sum w q^2 (a^2+b^2) = sum w q^3 is the scalar normal-equation
    # coefficient.  When it does not exceed round-off of its summands,
    # solving for k1 has no reliable digits.
    def normal_terms(kk):
        aa = 0.0   # J^T J  (J = -a q per coordinate)
        gg = 0.0   # J^T f
        for a, b, ad, bd, w in data:
            if w == 0.0 or (a == 0.0 and b == 0.0):
                continue
            q = a * a + b * b
            f_a = ad - a * (1.0 + kk * q)
            f_b = bd - b * (1.0 + kk * q)
            aa += w * q * q * (a * a + b * b)
            gg += w * (-q) * (a * f_a + b * f_b)
        return aa, gg

    floor = 64.0 * _EPS * weight_sum * max(1.0, max_radius ** 6)
    a_nn, _ = normal_terms(k)
    if not (a_nn > floor):
        return finish(k, 0, DEGENERATE, 0.0,
                      f"degenerate fit: normal-equation curvature "
                      f"{a_nn:.3e} is at the round-off floor "
                      f"{floor:.3e}; the correspondences carry no "
                      f"numerically resolvable leverage on k1.")

    # --- Damped-free Gauss-Newton with Armijo back-tracking ------------
    sse, _ = _metrics(data, k)
    iterations = 0
    last_step = 0.0
    while True:
        a_nn, g = normal_terms(k)
        delta = -g / a_nn
        last_step = abs(delta)
        if last_step <= xtol * max(1.0, abs(k)):
            return finish(
                k, iterations, CONVERGED, last_step,
                f"converged: Gauss-Newton step {last_step:.3e} <= xtol "
                f"{xtol:.3e} * max(1, |k1|) after {iterations} accepted "
                f"update(s); weighted RMS residual reported.")
        if iterations >= budget:
            return finish(
                k, iterations, MAX_ITERATIONS, last_step,
                f"iteration budget {budget} exhausted before the step "
                f"convergence threshold (last step {last_step:.3e}); "
                f"k1 and residuals are reported at the final iterate, "
                f"not at a stationary point.")

        # The model is linear in k1, so the full step is the exact LS
        # update; back-tracking only guards cancellation at extreme k.
        directional = g * delta  # negative: local predicted decrease
        alpha = 1.0
        moved = False
        for _ in range(60):
            k_try = k + alpha * delta
            cost_try, _ = _metrics(data, k_try)
            if (math.isfinite(cost_try)
                    and cost_try <= sse + 1e-4 * alpha * directional):
                k, sse = k_try, cost_try
                moved = True
                break
            alpha *= 0.5
        if not moved:
            return finish(
                k, iterations, STALLED, last_step,
                f"stalled: no Armijo-acceptable descent at the Gauss-"
                f"Newton step after {iterations} update(s); reported k1 "
                f"is the best iterate found.")
        iterations += 1
