"""Regression guard on the real-data shadow-kinematics detector.

The detector's claim is that a caster which stays put accumulates votes from
many illuminations, while a dark patch painted on the ground does not. These
tests plant both, with known positions and a known height, and demand the
detector tell them apart. No ISIS, no network, runs in seconds.

Two of the checks exist because the first version of the code failed them.

The voter's up-sun extreme lands one to three pixels from the true base
depending on the sun angle, so an accumulator that counted exact pixel
coincidences found nothing at all, even for a caster planted at a fixed spot.
Votes are dilated over a small disc so agreement means "within a couple of
pixels".

And each frame votes at most once per place. Left as a tally, a frame that broke
one shadow into two regions would agree with itself, which is evidence of
nothing.

    python tests/test_kinematics.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import shadow_kinematics_real as K  # noqa: E402

S = 300
BASE = (150, 150)
DECOY = (60, 220)
SHADOW_PX = 34
AZIMUTHS = [20.0, 78.0, 141.0, 205.0, 268.0]     # a five-frame sweep, like the real one
ELEV = 5.0
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def brush(m, r, c):
    m[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = True


def scene(with_decoy: bool = True):
    """Frames of one fixed caster plus one painted streak that never moves."""
    frames = []
    for a in AZIMUTHS:
        m = np.zeros((S, S), bool)
        ar = math.radians(a)
        srow, scol = -math.cos(ar), math.sin(ar)
        for t in range(0, SHADOW_PX + 1):            # shadow runs anti-sun from the base
            brush(m, int(round(BASE[0] - t * srow)), int(round(BASE[1] - t * scol)))
        if with_decoy:
            for t in range(-16, 17):                 # a stain, fixed in place and angle
                brush(m, DECOY[0] + int(round(t * 0.5)), DECOY[1] + t)
        frames.append({"mask": m, "az_map": a, "elev": ELEV})
    return frames


def main() -> None:
    print("shadow kinematics, real-data detector")

    # the voter must return the base it was given, at every sun angle
    worst = 0.0
    for a in (0.0, 37.0, 90.0, 143.0, 218.0, 305.0):
        m = np.zeros((S, S), bool)
        ar = math.radians(a)
        srow, scol = -math.cos(ar), math.sin(ar)
        for t in range(0, SHADOW_PX + 1):
            brush(m, int(round(BASE[0] - t * srow)), int(round(BASE[1] - t * scol)))
        v, h, cast = K.vote(m, a, ELEV, 4, 1.8, radius=0)
        rr, cc = np.nonzero(v)
        d = math.hypot(rr[0] - BASE[0], cc[0] - BASE[1]) if rr.size else 99.0
        worst = max(worst, d)
    check("the voter recovers a planted base at every sun angle", worst <= 3.0,
          f"worst miss {worst:.1f} px")

    # a round blob is not a cast shadow and must not vote
    m = np.zeros((S, S), bool)
    yy, xx = np.ogrid[-8:9, -8:9]
    m[142:159, 142:159] = (yy * yy + xx * xx <= 64)
    _, _, cast = K.vote(m, 90.0, ELEV, 4, 1.8)
    check("a round blob does not vote, only anti-solar elongated regions do", cast == 0)

    # the caster converges, and lands where it was planted
    frames = scene()
    conf, ev, hm, dark, casts = K.accumulate(frames, (S, S), 4, 1.8, 2)
    need = max(2, int(math.ceil(0.6 * len(frames))))
    hits, mass = K.concentration(ev, len(frames))
    rr, cc = np.nonzero(ev >= need)
    check("a fixed caster accumulates votes across the sweep", hits > 0,
          f"{hits} pixels with at least {need} of {len(frames)} frames")
    if hits:
        i = int(np.argmax(ev[rr, cc]))
        r, c = int(rr[i]), int(cc[i])
        off = math.hypot(r - BASE[0], c - BASE[1])
        check("...at the place it was planted", off <= 2.0, f"peak {off:.1f} px away")
        check("...seen by every frame in the sweep", ev[r, c] == len(frames),
              f"{ev[r, c]:.0f}/{len(frames)}")
        truth = SHADOW_PX * K.RES * math.tan(math.radians(ELEV))
        check("...with a height near L*tan(e)", abs(hm[r, c] - truth) < 0.5,
              f"recovered {hm[r, c]:.2f} m against {truth:.2f} m planted")

    # the painted streak must not converge, which is the whole point of the sweep
    near = [1 for r, c in zip(rr, cc)
            if math.hypot(r - DECOY[0], c - DECOY[1]) <= 25]
    check("a stain painted on the ground does not converge", not near,
          f"{len(near)} pixels converged on the decoy")

    # and the convergence must be driven by the sun, not by chance clustering
    rng = np.random.default_rng(0)
    null = []
    for _ in range(300):
        o = rng.permutation(len(frames))
        if np.all(o == np.arange(len(frames))):
            continue
        _, e2, _, _, _ = K.accumulate(frames, (S, S), 4, 1.8, 2, o)
        null.append(K.concentration(e2, len(frames))[0])
    null = np.array(null, float)
    p = float((null >= hits).mean())
    z = (hits - null.mean()) / max(null.std(), 1e-9)
    check("it beats a shuffled-azimuth null", p < 0.05 and hits > null.mean(),
          f"real {hits} vs null {null.mean():.1f} +/- {null.std():.1f}, z {z:+.1f}, p {p:.3f}")

    # The shadow of a sub-metre boulder is one pixel wide at 0.9 m per pixel, and
    # a 3x3 morphological opening erases anything narrower than three. The
    # detector used to open the mask, so it was deleting the shadow of every
    # obstacle below about 2.7 m: the entire population this project exists to
    # find. Injected 0.3 and 0.5 m boulders were recovered 0% of the time before
    # this and 100% after. Cleaning by area instead keeps thin lines.
    dn = np.full((200, 200), 120.0)
    for t in range(16):
        dn[60 + t, 100] = 25.0
    thin = K.detect(dn, 0.5, 45, 4)
    check("a one-pixel-wide shadow survives detection", int(thin.sum()) >= 12,
          f"{int(thin.sum())} of 16 planted pixels kept")

    noise = np.full((200, 200), 120.0)
    noise[100, 100] = 25.0
    check("...while an isolated noise pixel is still removed",
          int(K.detect(noise, 0.5, 45, 4).sum()) == 0)

    # and the whole chain, image to vote, must recover a planted sub-metre boulder
    from scipy.ndimage import gaussian_filter
    rng2 = np.random.default_rng(2)
    az = [17.2, 23.3, 47.7, 267.4, 277.7, 286.3]
    el = [4.0, 3.5, 2.8, 1.25, 2.9, 2.6]

    class _A:
        shadow_frac, bg_win, min_area, elongation, vote_radius = 0.5, 45, 4, 1.8, 5

    real = [{"dn": gaussian_filter(rng2.normal(size=(400, 400)), 3) * 20 + 120,
             "az_map": a, "elev": e} for a, e in zip(az, el)]
    for f in real:
        f["mask"] = K.detect(f["dn"], 0.5, 45, 4)
    _, ev0, _, _, casts0 = K.accumulate(real, (400, 400), 4, 1.8, 5)
    check("an empty scene produces no false convergence",
          int(K.hits_by_k(ev0, len(real))[2]) == 0, f"{sum(casts0)} votes cast")

    planted = [(int(r), int(c), 0.5)
               for r, c in rng2.integers(110, 290, (25, 2))]
    frac = K.recovery(real, (400, 400), planted, _A, k=3, tol=8.0)
    check("half-metre boulders injected into the imagery are recovered",
          frac >= 0.8, f"{100 * frac:.0f}% of {len(planted)}")

    # the agreement threshold must not get stricter as the sweep grows
    ev = np.zeros((50, 50), np.float32)
    ev[10, 10] = 4
    curve = K.hits_by_k(ev, 16)
    check("hits_by_k reports every threshold, not one fixed fraction",
          len(curve) == 16 and curve[3] == 1 and curve[4] == 0,
          "a pixel with 4 votes counts at k<=4 and not above")

    # A vote count cannot separate a boulder from a crater rim: both converge.
    # Fitting a fixed point against a point that walks with the sun can.
    az8 = [3.5, 14.4, 24.0, 43.5, 52.5, 73.7, 345.9, 354.6]
    rng3 = np.random.default_rng(0)
    fixed = [(i, 200 + rng3.normal(0, 1), 300 + rng3.normal(0, 1))
             for i in range(len(az8))]
    cp = K.classify(fixed, az8)
    check("a fixed caster fits a point, so the arc radius comes back near zero",
          cp["radius_px"] < 3.0, f"radius {cp['radius_px']:.2f} px")

    for R in (12.0, 30.0):
        arc = []
        for i, a in enumerate(az8):
            ar = math.radians(a)
            arc.append((i, 200 + R * -math.cos(ar) + rng3.normal(0, 1),
                        300 + R * math.sin(ar) + rng3.normal(0, 1)))
        ca = K.classify(arc, az8)
        check(f"a rim of radius {R:.0f} px is recovered as an arc, not a point",
              abs(ca["radius_px"] - R) < 3.0 and ca["rms_arc"] < 0.4 * ca["rms_point"],
              f"radius {ca['radius_px']:.1f} px, rms {ca['rms_point']:.1f} -> "
              f"{ca['rms_arc']:.1f}")

    # The negative control. Without it the recovery column is half a result: a
    # detector that fires on everything scores 100%. Craters and ridges are the
    # two things that vote at a fixed terrain corner and mimic a caster.
    from scipy.ndimage import gaussian_filter as _gf
    rng4 = np.random.default_rng(5)
    SS = 420
    el8 = [3.29, 3.52, 3.68, 3.59, 3.34, 3.38, 4.79, 3.63]
    bare = [{"dn": _gf(rng4.normal(size=(SS, SS)), 3) * 20 + 120,
             "az_map": a, "elev": e} for a, e in zip(az8, el8)]

    class _B:
        shadow_frac, bg_win, min_area = 0.5, 45, 4
        elongation, vote_radius, max_width_px = 1.8, 3, 4.0

    spots = [(140, 140), (260, 280), (330, 130)]
    ev_b = K._run_injected(bare, (SS, SS), _B, lambda d, a, e: K.inject_shadows(
        d, [(r, c, 0.5) for r, c in spots], a, e, 45))
    ev_c = K._run_injected(bare, (SS, SS), _B, lambda d, a, e: K.inject_craters(
        d, [(r, c, 14, 3.0) for r, c in spots], a, e, 45))
    ev_r = K._run_injected(bare, (SS, SS), _B, lambda d, a, e: K.inject_ridges(
        d, [(r, c, 80, 35.0, 4.0) for r, c in spots], a, e, 45))
    tp = K.hits_near(ev_b, spots, 3, 20.0)
    fc = K.hits_near(ev_c, spots, 3, 20.0)
    fr = K.hits_near(ev_r, spots, 3, 20.0)
    check("injected half-metre boulders are called casters", tp >= 2, f"{tp} of 3")
    check("injected craters are NOT called casters", fc == 0, f"{fc} of 3")
    check("injected ridges are NOT called casters", fr == 0, f"{fr} of 3")

    # the sun bearing must be the one we think it is
    lat, lon = -84.7906, 29.1957
    check("sub-solar point on the same meridian gives a bearing of due north",
          abs(K.sun_azimuth(lat, lon, 0.0, lon)) < 1e-6)
    check("sub-solar point past the pole gives due south",
          abs(K.sun_azimuth(lat, lon, -89.99, lon - 180) - 180.0) < 1e-6)
    e = K.sun_azimuth(lat, lon, 0.0, lon + 90)
    w = K.sun_azimuth(lat, lon, 0.0, lon - 90)
    check("a sun to the east reads easterly and one to the west reads westerly",
          0 < e < 180 < w < 360, f"east {e:.1f}, west {w:.1f}")

    print()
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: " + ", ".join(FAILS))
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
