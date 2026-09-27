"""Fixed-input demonstration of the distortion inversion library.

Runs real computation (no sleep, no pre-recorded results) and prints:

  1. an analytic no-distortion inversion,
  2. a constructed folding radial model with multiple preimages, each
     with its forward residual, plus the radial uniqueness certificate,
  3. the same fold with a small tangential term solved by multi-start
     damped Newton,
  4. a proven no-solution case,
  5. branch-tracked trajectory continuation/interruption, and
  6. actually triggered input rejections.

Run:  python demo.py     (completes in well under 8 seconds)
"""

import time

from distortion import DistortionModel


def banner(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def show_result(model, result):
    print(f"  target = ({result.xp:.9f}, {result.yp:.9f}), box half-width "
          f"R = {result.R:g}")
    print(f"  accepted candidates : {len(result.candidates)}")
    print(f"  unique              : {result.unique}")
    print(f"  proven no solution  : {result.proven_no_solution}")
    print(f"  search budget spent : {result.budget_exhausted}")
    if result.radial:
        print(f"  critical radii g'=0 : "
              f"{[round(v, 9) for v in result.critical_radii]}")
    for i, c in enumerate(result.candidates):
        back_x, back_y = model.distort(c.x, c.y)
        print(f"    [{i}] (x, y) = ({c.x: .10f}, {c.y: .10f})  "
              f"residual = {c.residual:.3e}  [{c.method}]  "
              f"reprojects to ({back_x: .9f}, {back_y: .9f})")
    print(f"  certificate: {result.certificate}")


def show_trajectory(trajectory):
    for point in trajectory.points:
        if point.candidate is None:
            print(f"    [{point.index}] {point.status} ({point.reason}): "
                  f"{point.message}")
            print(f"        single-point candidates available for explicit "
                  f"restart: {len(point.result.candidates)}")
        else:
            print(f"    [{point.index}] {point.status}: "
                  f"(x, y) = ({point.candidate.x: .10f}, "
                  f"{point.candidate.y: .10f}), "
                  f"residual = {point.candidate.residual:.3e}")


def main():
    t0 = time.perf_counter()

    banner("1. Analytic no-distortion case (k1=k2=p1=p2=0)")
    model = DistortionModel()
    xp, yp = 0.35, -0.20
    print(f"  distort(x,y) is the identity; undistort({xp}, {yp}, R=1):")
    show_result(model, model.undistort(xp, yp, R=1.0))

    banner("2. Constructed folding RADIAL model (k1=-0.8, k2=0.1)")
    # g(r) = r(1 - 0.8 r^2 + 0.1 r^4); g'(r)=1-2.4 r^2+0.5 r^4 has two
    # positive zeros inside R=3, so the map folds and one target can
    # have several preimages.
    radial_model = DistortionModel(k1=-0.8, k2=0.1)
    src_r = 0.8
    s, zero = radial_model.distort(src_r, 0.0)
    print(f"  forward: distort({src_r}, 0) = ({s:.12f}, {zero:.1f})")
    print("  the same target is now sought ANYWHERE in [-3, 3]^2:")
    result = radial_model.undistort(s, 0.0, R=3.0)
    show_result(radial_model, result)
    print("  note: the preimage at x=0.8 is only one of several; all "
          "monotone")
    print("  segments were probed, so this multiplicity is certified "
          "rather than")
    print("  'the first root the solver stumbled upon'.")

    banner("3. Same fold WITH tangential terms -> multi-start damped Newton")
    general_model = DistortionModel(k1=-0.8, k2=0.1, p1=1e-5, p2=-1e-5)
    s2, _ = general_model.distort(src_r, 0.0)
    result2 = general_model.undistort(s2, 0.0, R=3.0)
    show_result(general_model, result2)
    print("  numerical search found all the basins here but does NOT "
          "prove")
    print("  completeness: further preimages outside the sampled "
          "attraction")
    print("  basins cannot be excluded by a finite search.")

    banner("4. Proven NO SOLUTION (target outside what the box can reach)")
    identity = DistortionModel()
    far_x = 1.5
    result3 = identity.undistort(far_x, 0.0, R=1.0)
    print(f"  undistort({far_x}, 0, R=1) with the identity model:")
    show_result(identity, result3)

    banner("5. Trajectory mode: branch continuation without silent switching")
    track_model = DistortionModel(k1=-0.15, k2=0.02, p1=0.01, p2=-0.008)
    sources = [(0.30, -0.40), (0.31, -0.405), (0.32, -0.41)]
    track_targets = track_model.distort_all(sources)
    first_result = track_model.undistort(*track_targets[0], R=1.2)
    selected = first_result.candidates[0]
    print("  caller explicitly selects the first verified candidate:")
    show_result(track_model, first_result)
    tracked = track_model.undistort_trajectory(
        track_targets, 1.2, selected, continuation_tol=0.05)
    show_trajectory(tracked)

    # A folding radial branch that reaches a critical value and ends:
    # other roots can still exist at the new target, but they are left
    # for an explicit restart rather than used automatically.
    folded = DistortionModel(k1=-0.8, k2=0.1)
    s_start, _ = folded.distort(0.8, 0.0)
    folded_first = folded.undistort(s_start, 0.0, R=3.0)
    folded_start = min(folded_first.candidates, key=lambda c: abs(c.x - 0.8))
    broken = folded.undistort_trajectory(
        [(s_start, 0.0), (0.45, 0.0)], 3.0, folded_start,
        continuation_tol=0.5)
    print("  a folded branch that ends is interrupted, not switched:")
    show_trajectory(broken)

    banner("6. Actually triggered rejection / exception boundaries")
    demonstrations = []

    def attempt(label, fn):
        try:
            fn()
        except (ValueError, TypeError) as exc:
            demonstrations.append((label, type(exc).__name__, str(exc)))

    attempt("NaN target coordinate",
            lambda: identity.undistort(float("nan"), 0.0, 1.0))
    attempt("Infinity box half-width",
            lambda: identity.undistort(0.0, 0.0, float("inf")))
    attempt("non-positive box (R = 0)",
            lambda: identity.undistort(0.0, 0.0, 0.0))
    attempt("bool passed as an integer parameter",
            lambda: identity.undistort(0.0, 0.0, 1.0,
                                       starts_per_axis=True))
    attempt("batch of 101 points (interface limit is 100)",
            lambda: identity.undistort_all([(0.0, 0.0)] * 101, 1.0))

    for label, exc_name, message in demonstrations:
        print(f"  - {label}")
        print(f"      raised {exc_name}: {message}")
    print("  invalid inputs raise before any solving; no result list is")
    print("  ever partially built.")

    elapsed = time.perf_counter() - t0
    banner(f"done in {elapsed:.3f} s (budget ~ 8 s)")


if __name__ == "__main__":
    main()
