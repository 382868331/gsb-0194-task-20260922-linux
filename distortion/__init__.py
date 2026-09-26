"""Lens distortion mapping, bounded-box invertibility checking, and
forward area-conserving resampling of small grayscale images.

Public API:
    DistortionModel  -- forward/distort and inverse/undistort operations
    Candidate        -- one accepted inverse candidate
    InverseResult    -- all accepted, de-duplicated candidates plus flags
    ForwardResult    -- forward splatting result (energy + area grids)
    forward_resample -- splat constant-area source cells through the map
    finite_real      -- input validation helper

Only Python 3.14 standard library is used.  This module implements the
contract of this task only; it does NOT claim industry-standard camera
model compatibility.
"""

from .model import DistortionModel
from .inverse import Candidate, InverseResult
from .numeric import finite_real
from .forward import ForwardResult, forward_resample

__all__ = [
    "DistortionModel",
    "Candidate",
    "InverseResult",
    "ForwardResult",
    "forward_resample",
    "finite_real",
]
