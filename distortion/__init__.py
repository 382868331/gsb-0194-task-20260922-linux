"""Lens distortion mapping and bounded-box invertibility checking.

Public API:
    DistortionModel  -- forward/distort and inverse/undistort operations
    Candidate        -- one accepted inverse candidate
    InverseResult    -- all accepted, de-duplicated candidates plus flags
    TrackStep        -- one frame of continuous-branch tracking
    TrackResult      -- a whole tracked trajectory
    track_undistorted -- nearest-continuous-root sequence entry point
    finite_real      -- input validation helper

Only Python 3.14 standard library is used.  This module implements the
contract of this task only; it does NOT claim industry-standard camera
model compatibility.
"""

from .model import DistortionModel
from .inverse import Candidate, InverseResult
from .tracking import (TrackStep, TrackResult, track_undistorted,
                       OK, AMBIGUOUS, NO_CANDIDATE, OVER_STEP,
                       INVALID_ANCHOR, NEEDS_ANCHOR)
from .numeric import finite_real

__all__ = [
    "DistortionModel",
    "Candidate",
    "InverseResult",
    "TrackStep",
    "TrackResult",
    "track_undistorted",
    "OK",
    "AMBIGUOUS",
    "NO_CANDIDATE",
    "OVER_STEP",
    "INVALID_ANCHOR",
    "NEEDS_ANCHOR",
    "finite_real",
]
