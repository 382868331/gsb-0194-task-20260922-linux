"""Lens distortion mapping, invertibility checking, and forward resampling.

Public API:
    DistortionModel     -- forward/distort and inverse/undistort operations
    Candidate           -- one accepted inverse candidate
    InverseResult       -- all accepted, de-duplicated candidates plus flags
    finite_real         -- input validation helper
    forward_resample    -- area-weighted forward splatting of a small
                           constant-cell grayscale image onto a target grid
    exact_quad_resample -- exact polygon-clipping reference for quads
                           (identity / affine degenerate cases)
    ResampleResult      -- energy/area grids plus error and loss flags

Only Python 3.14 standard library is used.  This module implements the
contract of this task only; it does NOT claim industry-standard camera
model compatibility.
"""

from .model import DistortionModel
from .inverse import Candidate, InverseResult
from .numeric import finite_real
from .resample import ResampleResult, forward_resample, exact_quad_resample

__all__ = [
    "DistortionModel",
    "Candidate",
    "InverseResult",
    "finite_real",
    "ResampleResult",
    "forward_resample",
    "exact_quad_resample",
]
