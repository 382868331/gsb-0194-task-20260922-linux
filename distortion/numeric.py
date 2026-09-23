"""Numeric validation primitives shared by the package.

Contract (see README):
* Every float-shaped input must be a finite real number: NaN and the
  infinities raise ValueError carrying the offending field name.
* Integer parameters must be genuine ints -- bool is rejected even though
  it is an int subclass in Python.
"""

import math

__all__ = [
    "finite_real",
    "finite_real_pair",
    "checked_int",
]


def finite_real(value, name="value"):
    """Return ``value`` as a finite float, else raise a locatable error.

    ``bool`` is rejected (it is not a real-number input).  ``int`` and
    ``float`` with a finite value are accepted and returned as ``float``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{name}: expected a real number (int/float), got {type(value).__name__}"
        )
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"{name}: must be finite (NaN/Infinity rejected), got {value!r}")
    return f


def finite_real_pair(x, y, names=("x", "y")):
    """Validate an ordered pair of finite reals; return ``(fx, fy)``."""
    return finite_real(x, names[0]), finite_real(y, names[1])


def checked_int(value, name="value", *, minimum=None, maximum=None):
    """Validate a genuine integer parameter (bool rejected) with bounds."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{name}: expected int, got {type(value).__name__}"
        )
    if minimum is not None and value < minimum:
        raise ValueError(f"{name}: must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name}: must be <= {maximum}, got {value}")
    return value
