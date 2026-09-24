"""Lens distortion mapping and bounded-box invertibility checking.

Public API:
    DistortionModel  -- forward/distort and inverse/undistort operations
    Candidate        -- one accepted inverse candidate
    InverseResult    -- all accepted, de-duplicated candidates plus flags
    RadialFitResult  -- result of the weighted k1 coefficient fit
    fit_radial_k1    -- fit one radial coefficient with fixed intrinsics
    finite_real      -- input validation helper

Only Python 3.14 standard library is used.  This module implements the
contract of this task only; it does NOT claim industry-standard camera
model compatibility.
"""

from .model import DistortionModel
from .inverse import Candidate, InverseResult
from .fit import RadialFitResult, fit_radial_k1
from .numeric import finite_real

__all__ = [
    "DistortionModel",
    "Candidate",
    "InverseResult",
    "RadialFitResult",
    "fit_radial_k1",
    "finite_real",
]
