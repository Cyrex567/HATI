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
    conf, ev, hm, dark = K.accumulate(frames, (S, S), 4, 1.8, 2)
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
        _, e2, _, _ = K.accumulate(frames, (S, S), 4, 1.8, 2, o)
        null.append(K.concentration(e2, len(frames))[0])
    null = np.array(null, float)
    p = float((null >= hits).mean())
    z = (hits - null.mean()) / max(null.std(), 1e-9)
    check("it beats a shuffled-azimuth null", p < 0.05 and hits > null.mean(),
          f"real {hits} vs null {null.mean():.1f} +/- {null.std():.1f}, z {z:+.1f}, p {p:.3f}")

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
