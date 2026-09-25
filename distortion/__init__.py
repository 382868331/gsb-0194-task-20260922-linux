"""Lens distortion mapping and bounded-box invertibility checking.

Public API:
    DistortionModel  -- forward/distort and inverse/undistort operations
    Candidate        -- one accepted inverse candidate
    InverseResult    -- all accepted, de-duplicated candidates plus flags
    finite_real      -- input validation helper

Only Python 3.14 standard library is used.  This module implements the
contract of this task only; it does NOT claim industry-standard camera
model compatibility.
"""

from .model import DistortionModel
from .inverse import Candidate, InverseResult
from .numeric import finite_real
from .tracking import (
    TrackStep,
    InverseTracker,
    track_points,
    ANCHORED,
    OK,
    AMBIGUOUS,
    NO_CANDIDATE,
    STEP_TOO_LARGE,
    ANCHOR_REJECTED,
    BROKEN_NEEDS_ANCHOR,
)

__all__ = [
    "DistortionModel",
    "Candidate",
    "InverseResult",
    "finite_real",
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
