"""Lens distortion mapping and bounded-box invertibility checking.

Public API:
    DistortionModel  -- forward/distort and inverse/undistort operations
    Candidate            -- one accepted inverse candidate
    InverseResult        -- single-point candidates plus flags
    TrajectoryResult     -- branch-tracked inverse trajectory result
    finite_real      -- input validation helper

Only Python 3.14 standard library is used.  This module implements the
contract of this task only; it does NOT claim industry-standard camera
model compatibility.
"""

from .model import DistortionModel
from .inverse import (Candidate, InverseResult, TrajectoryPoint,
                      TrajectoryResult, undistort_trajectory)
from .numeric import finite_real

__all__ = [
    "DistortionModel",
    "Candidate",
    "InverseResult",
    "TrajectoryPoint",
    "TrajectoryResult",
    "undistort_trajectory",
    "finite_real",
]
