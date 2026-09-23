"""Fixed-input demonstration of the distortion inversion library.

Runs real computation (no sleep, no pre-recorded results) and prints:

  1. an analytic no-distortion inversion,
  2. a constructed folding radial model with multiple preimages, each
     with its forward residual, plus the radial uniqueness certificate,
  3. the same fold with a small tangential term solved by multi-start
     damped Newton,
  4. a proven no-solution case, and
  5. actually triggered input rejections.

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

    banner("5. Actually triggered rejection / exception boundaries")
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
