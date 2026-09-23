"""透镜畸变映射的可逆区域检查库（纯标准库）。

公开接口：
- DistortionModel(k1, k2, p1, p2)
- distort_point / distort_points
- jacobian
- undistort_point / undistort_points
- radial_uniqueness_certificate / solve_radial / critical_radii / monotonic_segments
- 数据类：Candidate, UndistortResult, RadialCertificate, RadialSegment
- 常量：MAX_POINTS, STATUS_CONVERGED / STATUS_EXHAUSTED / STATUS_PROVEN_NO_SOLUTION
- 异常：LensDistortError, InvalidInputError
"""

from ._validation import MAX_POINTS
from .errors import InvalidInputError, LensDistortError
from .inverse import (
    STATUS_CONVERGED,
    STATUS_EXHAUSTED,
    STATUS_PROVEN_NO_SOLUTION,
    Candidate,
    UndistortResult,
    undistort_point,
    undistort_points,
)
from .model import DistortionModel, distort_point, distort_points, jacobian
from .radial import (
    RadialCertificate,
    RadialSegment,
    critical_radii,
    monotonic_segments,
    radial_g,
    radial_gp,
    radial_uniqueness_certificate,
    solve_radial,
)

__all__ = [
    "Candidate",
    "DistortionModel",
    "InvalidInputError",
    "LensDistortError",
    "MAX_POINTS",
    "RadialCertificate",
    "RadialSegment",
    "STATUS_CONVERGED",
    "STATUS_EXHAUSTED",
    "STATUS_PROVEN_NO_SOLUTION",
    "UndistortResult",
    "critical_radii",
    "distort_point",
    "distort_points",
    "jacobian",
    "monotonic_segments",
    "radial_g",
    "radial_gp",
    "radial_uniqueness_certificate",
    "solve_radial",
    "undistort_point",
    "undistort_points",
]

__version__ = "1.0.0"
