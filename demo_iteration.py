"""Offline demo: forward area-weighted resampling through lens distortion.

Runs three real computations (no sleep, no prerecorded results, no
network, no third-party packages) in well under 90 seconds:

1. identity model checked against the exact polygon-clipping reference;
2. a barrel-distorted 6x6 grayscale image splatted onto a target grid,
   reporting energy/area separation, the subdivision error estimate and
   convergence against a finer independent run;
3. a window that is too small on purpose, showing lost energy, and a
   deliberately starved leaf budget, showing the not-converged flag.
"""

import time

from distortion import DistortionModel, forward_resample, exact_quad_resample


def total(grid):
    return sum(sum(row) for row in grid)


def main():
    t0 = time.monotonic()

    print("=== 1. identity model vs exact polygon clipping ===")
    ident = DistortionModel(0.0, 0.0, 0.0, 0.0)
    image = [[float(r * 4 + c + 1) for c in range(4)] for r in range(4)]
    window = (0.0, 0.0, 4.0, 4.0)
    shape = (4, 4)
    res = forward_resample(ident, image, window, shape, area_tol=1e-12)
    quads = [[(c, r), (c + 1, r), (c + 1, r + 1), (c, r + 1)]
             for r in range(4) for c in range(4)]
    values = [image[r][c] for r in range(4) for c in range(4)]
    ref = exact_quad_resample(quads, values, window, shape)
    max_dev = max(abs(a - b)
                  for ra, rb in zip(res.energy, ref.energy)
                  for a, b in zip(ra, rb))
    print(f"  max |energy - exact clip| = {max_dev:.3e}  "
          f"(converged={res.converged}, leaves={res.leaves_used})")

    print("=== 2. barrel distortion, energy and area tracked separately ===")
    model = DistortionModel(k1=-0.05, k2=0.001, p1=0.005, p2=-0.008)
    image = [[float(10 + 5 * r + c) for c in range(6)] for r in range(6)]
    origin = (-2.0, -2.0)
    window = (-2.5, -2.5, 5.0, 5.0)
    shape = (14, 14)
    coarse = forward_resample(model, image, window, shape, origin=origin,
                              area_tol=1e-6)
    fine = forward_resample(model, image, window, shape, origin=origin,
                            area_tol=1e-9, max_depth=14)
    print(f"  input energy      = {coarse.input_energy:.6f}")
    print(f"  accounted energy  = {coarse.accounted_energy:.6f} "
          f"(window {total(coarse.energy):.6f} + lost {coarse.lost_energy:.3e})")
    print(f"  covered area      = {total(coarse.area):.6f} "
          f"(source area 36, shrunk by the distortion Jacobian)")
    print(f"  subdivision error estimate = {coarse.error_estimate:.3e} "
          f"(converged={coarse.converged}, leaves={coarse.leaves_used})")
    dev = max(abs(a - b)
              for ra, rb in zip(coarse.energy, fine.energy)
              for a, b in zip(ra, rb))
    print(f"  max per-cell |coarse - fine(reference)| = {dev:.3e} "
          f"(fine leaves={fine.leaves_used})")

    print("=== 3. out-of-window loss and budget exhaustion ===")
    small = forward_resample(model, image, (-1.0, -1.0, 2.0, 2.0), (4, 4),
                             origin=origin, area_tol=1e-6)
    print(f"  tight window: lost energy = {small.lost_energy:.4f} "
          f"of {small.input_energy:.1f} "
          f"(deposited {total(small.energy):.4f})")
    starved = forward_resample(model, image, window, shape, origin=origin,
                               area_tol=1e-12, max_depth=30, max_leaves=64)
    print(f"  starved budget: converged={starved.converged} "
          f"leaves={starved.leaves_used} "
          f"-- flag is set, no strict conservation claim is made")

    print(f"--- demo finished in {time.monotonic() - t0:.2f}s ---")


if __name__ == "__main__":
    main()
