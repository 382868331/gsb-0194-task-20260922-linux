"""Forward lens-distortion model.

Implements the polynomial model of this task:

    x' = x*(1 + k1*r^2 + k2*r^4) + 2*p1*x*y + p2*(r^2 + 2*x^2)
    y' = y*(1 + k1*r^2 + k2*r^4) + p1*(r^2 + 2*y^2) + 2*p2*x*y

with r^2 = x^2 + y^2, on the normalized image plane.

Only this exact 4-coefficient subset is implemented; no claim of
compatibility with OpenCV, KB, or any other industry standard is made.
"""

import math

from .numeric import finite_real, finite_real_pair

# Hard interface bound stated by the task: at most 100 points per batch.
MAX_POINTS = 100


class DistortionModel:
    """Container of the four distortion coefficients (all finite reals)."""

    __slots__ = ("k1", "k2", "p1", "p2")

    def __init__(self, k1=0.0, k2=0.0, p1=0.0, p2=0.0):
        self.k1 = finite_real(k1, "k1")
        self.k2 = finite_real(k2, "k2")
        self.p1 = finite_real(p1, "p1")
        self.p2 = finite_real(p2, "p2")

    @property
    def is_radial(self):
        """True when the tangential terms vanish (p1 == p2 == 0)."""
        return self.p1 == 0.0 and self.p2 == 0.0

    def distort(self, x, y):
        """Forward map.  ``x``, ``y`` are normalized-plane coordinates."""
        x, y = finite_real_pair(x, y)
        return self._distort(x, y)

    @staticmethod
    def _check_batch(points):
        if not isinstance(points, (list, tuple)):
            raise TypeError(
                f"points: expected list/tuple of (x, y) pairs, "
                f"got {type(points).__name__}"
            )
        if len(points) > MAX_POINTS:
            raise ValueError(
                f"points: batch size {len(points)} exceeds the interface "
                f"limit of {MAX_POINTS}"
            )
        cleaned = []
        for i, p in enumerate(points):
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                raise TypeError(
                    f"points[{i}]: expected an (x, y) pair of length 2"
                )
            cleaned.append(finite_real_pair(p[0], p[1], (f"points[{i}].x",
                                                          f"points[{i}].y")))
        return cleaned

    def distort_all(self, points):
        """Forward map for a batch of at most ``MAX_POINTS`` pairs.

        The whole call is validated before any result is produced, so an
        invalid batch never yields a partially written output.
        """
        cleaned = self._check_batch(points)
        return [self._distort(x, y) for (x, y) in cleaned]

    def undistort(self, xp, yp, R, **kwargs):
        """Bounded inverse query; see :func:`distortion.inverse.undistort_points`.

        Returns an :class:`~distortion.inverse.InverseResult`.  The
        search budget (``starts_per_axis``, ``max_iterations``) and
        ``residual_tol`` are keyword arguments.
        """
        from .inverse import undistort_points
        return undistort_points(self, xp, yp, R, **kwargs)

    def undistort_all(self, targets, R, **kwargs):
        """Inverse query for a batch of at most ``MAX_POINTS`` targets.

        Every target and the shared box half-width ``R`` are validated
        before any solver runs, so a failure never leaves a partially
        computed result list.
        """
        from .inverse import undistort_points
        cleaned = self._check_batch(targets)
        R = finite_real(R, "R")
        if R <= 0.0:
            raise ValueError(f"R: search box half-width must be > 0, got {R!r}")
        return [undistort_points(self, tx, ty, R, **kwargs)
                for (tx, ty) in cleaned]

    def track(self, targets, anchor, R, *, max_step, **kwargs):
        """Track one continuous inverse branch; see
        distortion.tracking.track_undistorted.

        Returns a distortion.tracking.TrackResult.
        """
        from .tracking import track_undistorted
        return track_undistorted(self, targets, anchor, R,
                                 max_step=max_step, **kwargs)

    def _distort(self, x, y):
        r2 = x * x + y * y
        radial = 1.0 + (self.k1 + self.k2 * r2) * r2
        xp = x * radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x)
        yp = y * radial + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y
        return xp, yp

    def residual_and_jacobian(self, x, y, xp, yp):
        """Residual f = distort(x,y) - (xp,yp) and the analytic Jacobian.

        The Jacobian entries (verified against finite differences in the
        test suite) are::

            d = k1 + 2*k2*r^2
            J00 = D + 2*x^2*d + 2*p1*y + 6*p2*x
            J01 = 2*x*y*d + 2*p1*x + 2*p2*y
            J10 = J01
            J11 = D + 2*y^2*d + 6*p1*y + 2*p2*x
        """
        r2 = x * x + y * y
        d = self.k1 + 2.0 * self.k2 * r2
        radial = 1.0 + (self.k1 + self.k2 * r2) * r2

        fx, fy = self._distort(x, y)
        fx -= xp
        fy -= yp

        j00 = radial + 2.0 * x * x * d + 2.0 * self.p1 * y + 6.0 * self.p2 * x
        j01 = 2.0 * x * y * d + 2.0 * self.p1 * x + 2.0 * self.p2 * y
        j11 = radial + 2.0 * y * y * d + 6.0 * self.p1 * y + 2.0 * self.p2 * x
        return fx, fy, j00, j01, j11

    def radial_scale(self, radius):
        """1 + k1*r^2 + k2*r^4 for the pure-radial special case."""
        r2 = radius * radius
        return 1.0 + (self.k1 + self.k2 * r2) * r2

    def radial_scale_derivative(self, radius):
        """Analytic derivative of :meth:`radial_scale`."""
        r2 = radius * radius
        return 2.0 * radius * (self.k1 + 2.0 * self.k2 * r2)

    def __repr__(self):
        return (f"DistortionModel(k1={self.k1!r}, k2={self.k2!r}, "
                f"p1={self.p1!r}, p2={self.p2!r})")
