"""Fixed-input demonstration of the distortion inversion library.

Runs real computation (no sleep, no pre-recorded results) and prints:

  1. an analytic no-distortion inversion,
  2. a constructed folding radial model with multiple preimages, each
     with its forward residual, plus the radial uniqueness certificate,
  3. the same fold with a small tangential term solved by multi-start
     damped Newton,
  4. a proven no-solution case,
  5. continuous-branch tracking (candidates per frame, selected root,
     an ambiguous tie and track breaks), and
  6. actually triggered input rejections.

Run:  python demo.py     (completes in well under 90 seconds)
"""

import time

from distortion import (
    DistortionModel,
    track_points,
    ANCHORED,
    OK,
    AMBIGUOUS,
    NO_CANDIDATE,
    STEP_TOO_LARGE,
    ANCHOR_REJECTED,
    BROKEN_NEEDS_ANCHOR,
)


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

    banner("5. Continuous-branch tracking over a folding radial map")
    track_model = DistortionModel(k1=-0.8, k2=0.1)

    def g(r):
        return r * (1.0 - 0.8 * r * r + 0.1 * r ** 4)

    def show_steps(title, steps):
        print(f"  {title}")
        for st in steps:
            xp, yp = st.target
            head = (f"    frame {st.index}: target=({xp: .9f}, {yp: .9f}) "
                    f"-> {st.status}")
            print(head)
            for j, (c, d) in enumerate(zip(st.candidates, st.distances)):
                mark = ""
                if st.status == OK and c is st.selected:
                    mark = "  <== selected (nearest, within max_step)"
                elif c in st.tied:
                    mark = "  <== tied nearest (ambiguous, retained)"
                print(f"        cand[{j}] (x={c.x: .10f}, y={c.y: .3g}) "
                      f"resid={c.residual:.2e} step={d:.6f}{mark}")
            if st.previous is not None and not st.distances:
                print("        no residual-accepted candidates")
            if st.status == ANCHORED and st.selected is not None:
                print(f"        anchor forward residual = "
                      f"{st.selected.residual:.3e}")
            print(f"        reason: {st.reason}")

    # 5a. Unique branch: the source path stays on the inner fold branch.
    sources = [0.30, 0.38, 0.46, 0.54]
    targets = [(g(r), 0.0) for r in sources]
    steps_ok = track_points(track_model, targets, (sources[0], 0.0),
                            R=3.0, max_step=0.2)
    show_steps("5a. unique branch: several candidates per frame, the "
               "tracked one follows the source", steps_ok)

    # 5b. Near a fold: ambiguous equidistant pair breaks the track.
    rc = 0.6789168263001503
    s_near = g(rc - 0.10)
    r = track_model.undistort(s_near, 0.0, R=3.0)
    inner = sorted(c.x for c in r.candidates
                   if abs(abs(c.x) - rc) < 0.2)
    mid = 0.5 * (inner[0] + inner[1])
    steps_tie = track_points(track_model, [(g(mid), 0.0), (s_near, 0.0)],
                             (mid, 0.0), R=3.0, max_step=0.5,
                             equal_tol=1e-9)
    show_steps("5b. equidistant roots: ambiguous, neither root picked "
               "by list order", steps_tie)

    # 5c. Mid-track breaks: step exceeds the limit; later frames need a
    # new anchor; then a certified no-candidate break.
    steps_break = track_points(
        track_model,
        [(g(0.45), 0.0), (g(0.62), 0.0), (g(0.60), 0.0)],
        (0.45, 0.0), R=3.0, max_step=0.02)
    show_steps("5c. step_too_large breaks the track; following frame "
               "needs a new anchor", steps_break)
    steps_gap = track_points(DistortionModel(),
                             [(0.5, 0.0), (1.5, 0.0), (0.6, 0.0)],
                             (0.5, 0.0), R=1.0, max_step=0.5)
    show_steps("5d. no_candidate: target outside the box (certified)",
               steps_gap)

    # 5e. A wrong anchor is rejected by forward verification.
    steps_bad_anchor = track_points(
        track_model, [(g(0.60), 0.0)], (0.50, 0.0), R=3.0,
        max_step=0.1)
    show_steps("5e. wrong anchor: forward check fails, no track started",
               steps_bad_anchor)

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
    banner(f"done in {elapsed:.3f} s (budget 90 s)")


if __name__ == "__main__":
    main()
