"""Fixed-input demonstration of the distortion inversion library.

Runs real computation (no sleep, no pre-recorded results) and prints:

  1. an analytic no-distortion inversion,
  2. a constructed folding radial model with multiple preimages, each
     with its forward residual, plus the radial uniqueness certificate,
  3. the same fold with a small tangential term solved by multi-start
     damped Newton,
  4. a proven no-solution case,
  5. actually triggered input rejections,
  6. weighted fitting of a single radial coefficient from fixed
     intrinsics (synthetic recovery, then with deterministic noise),
  7. a degenerate point set reported as a diagnostic (not an exception),
  8. an actually triggered invalid-input rejection for the fitter.

Run:  python demo.py     (completes in well under 90 seconds)
"""

import time

from distortion import DistortionModel, fit_radial_coefficient


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

    # ------------------------------------------------------------------
    # Single-coefficient weighted radial fit with fixed intrinsics
    # ------------------------------------------------------------------
    fx, fy, cx, cy = 800.0, 800.0, 320.0, 240.0
    k_true = -0.28
    undistorted_px = [(200.0, 120.0), (430.0, 300.0), (520.0, 90.0),
                      (90.0, 380.0), (360.0, 470.0), (610.0, 250.0),
                      (150.0, 300.0), (470.0, 420.0)]
    weights = [1.0, 2.0, 0.5, 1.5, 1.0, 0.25, 3.0, 1.0]

    def make_correspondences(noise_amp=0.0):
        """Pixel-frame records from the exact radial formula (input data)."""
        recs = []
        for i, (u, v) in enumerate(undistorted_px):
            a, b = (u - cx) / fx, (v - cy) / fy
            q = a * a + b * b
            c, d = a * (1.0 + k_true * q), b * (1.0 + k_true * q)
            if noise_amp:
                # Deterministic small offsets, ~noise_amp pixels.
                c += noise_amp * (((i * 37 + 11) % 13) / 6.0 - 1.0) / fx
                d += noise_amp * (((i * 53 + 7) % 11) / 5.0 - 1.0) / fy
            recs.append((u, v, fx * c + cx, fy * d + cy, weights[i]))
        return recs

    def show_fit(title, recs, **opts):
        banner(title)
        print(f"  fixed intrinsics: fx=fy={fx:g}, cx={cx:g}, cy={cy:g}; "
              f"{len(recs)} weighted pairs")
        r = fit_radial_coefficient(recs, fx=fx, fy=fy, cx=cx, cy=cy, **opts)
        print(f"  status        : {r.status} (converged={r.converged})")
        print(f"  fitted k1     : {r.k1:.10f}   (data generated with "
              f"{k_true:g})")
        print(f"  iterations    : {r.iterations}/{r.max_iterations}, "
              f"last step = {r.final_step:.3e}")
        print(f"  weighted SSE  : {r.sse:.6e}")
        print(f"  weighted RMS  : {r.rms_residual:.6e} normalized "
              f"({r.rms_residual * fx:.4f} px)")
        print(f"  max residual  : {r.max_residual:.6e} normalized "
              f"({r.max_residual * fx:.4f} px)")
        print(f"  sum of weights: {r.weight_sum:g}")
        print(f"  diagnostic    : {r.diagnostic}")
        return r

    banner("6. Weighted fit of ONE radial coefficient (fixed focal/principal)")
    print("  exact synthetic correspondences (8 weighted pairs, pixel "
          "frame):")
    r6 = show_fit("6a. Exact observations -> k1 recovered to machine "
                  "precision", make_correspondences())
    assert r6.status == "converged" and abs(r6.k1 - k_true) < 1e-10
    show_fit("6b. Same pairs with deterministic ~0.3 px observation noise",
             make_correspondences(noise_amp=0.3))

    banner("7. Degenerate point set -> diagnostic, not an exception")
    degenerate = [(cx, cy, cx + 0.4, cy - 0.2, 1.0),
                  (cx, cy, cx - 0.1, cy + 0.3, 2.0)]
    r7 = fit_radial_coefficient(degenerate, fx=fx, fy=fy, cx=cx, cy=cy)
    print(f"  every undistorted point is at the principal point:")
    print(f"  status       : {r7.status} (converged={r7.converged})")
    print(f"  k1 unchanged : {r7.k1:g} (echoes initial value; no "
          f"half-update)")
    print(f"  iterations   : {r7.iterations}")
    print(f"  diagnostic   : {r7.diagnostic}")

    banner("8. Fitter input rejection (actually triggered)")
    fit_rejections = []

    def attempt_fit(label, recs, **opts):
        try:
            fit_radial_coefficient(recs, **opts)
        except (ValueError, TypeError) as exc:
            fit_rejections.append((label, type(exc).__name__, str(exc)))

    good = make_correspondences()
    attempt_fit("non-positive focal length fx = 0", good,
                fx=0.0, fy=fy, cx=cx, cy=cy)
    attempt_fit("negative point weight",
                good[:-1] + [good[-1][:4] + (-1.0,)],
                fx=fx, fy=fy, cx=cx, cy=cy)
    attempt_fit("NaN observed coordinate",
                good[:-1] + [(good[-1][0], good[-1][1], float("nan"),
                              good[-1][3], good[-1][4])],
                fx=fx, fy=fy, cx=cx, cy=cy)
    attempt_fit("bool as max_iterations", good,
                fx=fx, fy=fy, cx=cx, cy=cy, max_iterations=True)
    attempt_fit("empty correspondence list", [], fx=fx, fy=fy,
                cx=cx, cy=cy)
    for label, exc_name, message in fit_rejections:
        print(f"  - {label}")
        print(f"      raised {exc_name}: {message}")

    elapsed = time.perf_counter() - t0
    banner(f"done in {elapsed:.3f} s (budget ~ 90 s)")


if __name__ == "__main__":
    main()
