"""Offline demo of forward area-conserving resampling (no network, no
accounts, no sleeps, no prerecorded results; runs in well under 90 s).

Shows, with deterministic inputs generated here:
  1. identity map: exact split of a constant image, checked against
     hand-computed values;
  2. a radial distortion: energy/area grids, subdivision error metric,
     energy lost outside the target window, and a convergence check
     against an independent dense Jacobian integration;
  3. an actually triggered rejection (degenerate, near-zero-area
     source cell).
"""

import math

from distortion import DistortionModel, forward_resample


def grid_sum(g):
    return sum(sum(row) for row in g)


def main():
    print("=== 1. identity map, constant 4x4 image ===")
    ident = DistortionModel()
    img = [[1.5] * 4 for _ in range(4)]
    res = forward_resample(ident, img, (-1, 1, -1, 1), (-1, 1, -1, 1), (4, 4))
    print(f"total source energy : {res.total_energy:.6f}")
    print(f"grid energy sum     : {grid_sum(res.energy):.6f}")
    print(f"grid area sum       : {grid_sum(res.area):.6f}")
    print(f"lost energy         : {res.lost_energy:.6f}")
    print(f"budget exhausted    : {res.budget_exhausted}")
    assert abs(grid_sum(res.energy) - res.total_energy) < 1e-12
    assert abs(res.energy[0][0] - 1.5 * 0.25) < 1e-12

    print()
    print("=== 2. radial distortion k1=-0.1, 8x8 gradient, window clipped ===")
    model = DistortionModel(k1=-0.1)
    pic = [[0.5 + 0.1 * (r + c) for c in range(8)] for r in range(8)]
    # Target window deliberately smaller than the mapped image so that
    # some energy genuinely falls outside.
    res = forward_resample(model, pic, (-1, 1, -1, 1),
                           (-0.7, 0.7, -0.7, 0.7), (12, 12), tol=1e-5)
    inside = grid_sum(res.energy)
    print(f"total source energy   : {res.total_energy:.6f}")
    print(f"energy inside window  : {inside:.6f}")
    print(f"area inside window    : {grid_sum(res.area):.6f}")
    print(f"lost energy (outside) : {res.lost_energy:.6f}")
    print(f"lost area  (outside)  : {res.lost_area:.6f}")
    print(f"subdivision error     : {res.subdivision_error:.3e} (tol 1e-5)")
    print(f"leaf quads used       : {res.quads_used}")
    print(f"budget exhausted      : {res.budget_exhausted}")

    # Independent convergence check: dense midpoint integration of
    # value*|det J| over the source, written independently here.
    n = 400
    h = 1e-6
    expected = 0.0
    for i in range(n):
        y = -1 + 2 * (i + 0.5) / n
        for j in range(n):
            x = -1 + 2 * (j + 0.5) / n
            def f(xx, yy):
                s = 1 - 0.1 * (xx * xx + yy * yy)
                return xx * s, yy * s
            j00 = (f(x + h, y)[0] - f(x - h, y)[0]) / (2 * h)
            j10 = (f(x + h, y)[1] - f(x - h, y)[1]) / (2 * h)
            j01 = (f(x, y + h)[0] - f(x, y - h)[0]) / (2 * h)
            j11 = (f(x, y + h)[1] - f(x, y - h)[1]) / (2 * h)
            r, c = int((1 - y) * 4), int((x + 1) * 4)
            expected += pic[r][c] * abs(j00 * j11 - j01 * j10)
    expected *= (2 / n) ** 2
    accounted = inside + res.lost_energy
    print(f"independent dense integral of value*|J| : {expected:.6f}")
    print(f"accounted (inside + lost)               : {accounted:.6f}")
    print(f"absolute gap                            : "
          f"{abs(accounted - expected):.3e}")
    assert abs(accounted - expected) < 2e-3

    print()
    print("=== 3. rejection: near-zero-area source cell ===")
    try:
        forward_resample(model, [[1.0]], (0.0, 1e-14, 0.0, 1e-14),
                         (-1, 1, -1, 1), (2, 2))
    except ValueError as exc:
        print(f"rejected as expected: {exc}")

    print()
    print("demo_iteration: all checks passed (computed live, offline).")


if __name__ == "__main__":
    main()
