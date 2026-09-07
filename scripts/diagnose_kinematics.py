"""Why did the real sweep produce no convergence?

The first real run cast 507 votes and stacked none of them. That has several
possible causes needing opposite fixes, and turning thresholds until something
appears would manufacture a detection rather than find one. This asks the
questions that separate the causes, and computes nothing that could be mistaken
for a result.

1. IS THE SUN WHERE WE THINK IT IS? The shadows know. Every detected region is
   elongated along the sun line, so the measured orientation of the regions is
   an independent measurement of the solar azimuth in the map frame. If it
   disagrees with the predicted azimuth, the bug is in the geometry: the
   sub-solar bearing, the polar-stereographic rotation, or the raster sign
   convention. Nothing downstream can work until that matches.

2. WHAT ARE WE ACTUALLY VOTING ON? A boulder is a compact caster and its votes
   converge on a point. A crater rim is an extended caster: as the sun turns,
   the up-sun end of its shadow slides ALONG the rim, so it converges on a line
   and never on a pixel. If the regions are large, the detector is looking at
   terrain rather than at obstacles, and the method has nothing to find.

3. HOW FAR APART DO THE VOTES LAND? If frames agree to within a handful of
   pixels the vote radius is simply too tight for real data, where the up-sun
   extreme is a single ragged pixel rather than the clean tip of a synthetic
   line. If they land tens of pixels apart there is no correspondence at all.

    python scripts/diagnose_kinematics.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import shadow_kinematics_real as K  # noqa: E402


def axis_azimuth(rr: np.ndarray, cc: np.ndarray) -> float | None:
    """Map-frame azimuth of a region's long axis, degrees, folded to [0, 180).

    Taken from the coordinate covariance rather than skimage's orientation, so
    the convention matches the voter exactly: the voter treats a sun azimuth A
    as the raster direction (-cos A, sin A), so a raster direction (dr, dc)
    corresponds to azimuth atan2(dc, -dr).
    """
    if rr.size < 3:
        return None
    d = np.vstack([rr - rr.mean(), cc - cc.mean()])
    cov = np.cov(d)
    if not np.all(np.isfinite(cov)):
        return None
    w, v = np.linalg.eigh(cov)
    dr, dc = v[:, int(np.argmax(w))]
    return math.degrees(math.atan2(dc, -dr)) % 180.0


def circ_diff_180(a: float, b: float) -> float:
    """Separation of two axis directions, degrees in [0, 90]."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", type=int, default=400)
    ap.add_argument("--shadow-frac", type=float, default=0.5)
    ap.add_argument("--bg-win", type=int, default=45)
    ap.add_argument("--min-area", type=int, default=4)
    ap.add_argument("--elongation", type=float, default=1.8)
    args = ap.parse_args()

    from skimage.measure import label, regionprops
    import athena_counterfactual as ac

    if not K.MANIFEST.exists():
        sys.exit("no manifest; run the ingest first")
    man = json.loads(K.MANIFEST.read_text())

    frames = []
    for e in man:
        if not e.get("shift_px"):
            continue
        g = K.geometry_for(e["pid"], ac.TD_LAT, ac.TD_LON, rebuild=False)
        if g is None:
            print(f"  {e['pid']}: no geometry sidecar, run the kinematics once first")
            continue
        lev2 = Path(e["lev2"])
        if not lev2.exists():
            lev2 = K.SWEEP_DIR / lev2.name
        w = K.frame_window(lev2, e["shift_px"], args.half, ac)
        if w is None:
            continue
        frames.append({"pid": e["pid"].split(".")[-1].upper(),
                       "dn": w, "az": g["az"], "elev": g["elev"],
                       "az_map": (g["az"] + ac.TD_LON) % 360.0})
    if len(frames) < 2:
        sys.exit("need at least two frames")

    # ---------------------------------------------------------- 1. geometry
    print("=" * 74)
    print("1. DOES THE MEASURED SHADOW DIRECTION MATCH THE PREDICTED SUN DIRECTION?")
    print("   The regions are elongated along the sun line, so their orientation is")
    print("   an independent measurement of it. These two columns must agree.\n")
    print(f"   {'frame':<18}{'predicted':>11}{'measured':>10}{'disagree':>10}"
          f"{'regions':>9}   verdict")
    worst = 0.0
    for f in frames:
        f["mask"] = K.detect(f["dn"], args.shadow_frac, args.bg_win)
        lab = label(f["mask"])
        axes, areas, elong = [], [], []
        for p in regionprops(lab):
            if p.area < args.min_area:
                continue
            rr = p.coords[:, 0].astype(float)
            cc = p.coords[:, 1].astype(float)
            a = axis_azimuth(rr, cc)
            if a is not None:
                axes.append(a)
            areas.append(p.area)
            ar = math.radians(f["az_map"])
            srow, scol = -math.cos(ar), math.sin(ar)
            proj = rr * srow + cc * scol
            perp = rr * (-scol) + cc * srow
            along = (proj.max() - proj.min()) * K.RES
            wide = (perp.max() - perp.min()) * K.RES
            elong.append(along / max(wide, K.RES))
        f["areas"] = np.array(areas) if areas else np.array([0])
        f["elong"] = np.array(elong) if elong else np.array([0.0])
        if not axes:
            print(f"   {f['pid']:<18}  no regions")
            continue
        # circular median on a 180-degree axis: double the angle, average, halve
        ang = np.radians(np.array(axes) * 2.0)
        meas = (math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2.0) % 180.0
        pred = f["az_map"] % 180.0
        d = circ_diff_180(meas, pred)
        worst = max(worst, d)
        v = "matches" if d < 20 else ("SUSPECT" if d < 40 else "WRONG")
        f["axis_meas"] = meas
        print(f"   {f['pid']:<18}{pred:>10.1f}{meas:>10.1f}{d:>10.1f}"
              f"{len(areas):>9}   {v}")
    print(f"\n   worst disagreement: {worst:.1f} deg")
    if worst < 20:
        print("   -> the sun geometry is right. The zero is not a rotation bug.")
    elif worst > 40:
        print("   -> the geometry is wrong. Every vote points the wrong way, so no")
        print("      threshold change can help. Check the A + lambda map rotation,")
        print("      the sub-solar bearing, and the raster row sign.")
    else:
        print("   -> borderline. Look at whether the disagreement is a constant")
        print("      offset (a convention bug) or scattered (just noisy regions).")

    # ------------------------------------------------------ 2. what we vote on
    print()
    print("=" * 74)
    print("2. WHAT ARE THE REGIONS? A compact caster converges on a point. An")
    print("   extended one, a crater rim, converges on a line and never a pixel.\n")
    print(f"   {'frame':<18}{'regions':>9}{'med area':>10}{'p90 area':>10}"
          f"{'med elong':>11}{'voted':>8}")
    for f in frames:
        v, _, cast = K.vote(f["mask"], f["az_map"], f["elev"],
                            args.min_area, args.elongation, radius=0)
        f["votes"] = np.array(np.nonzero(v)).T
        print(f"   {f['pid']:<18}{f['areas'].size:>9}{np.median(f['areas']):>10.0f}"
              f"{np.percentile(f['areas'], 90):>10.0f}"
              f"{np.median(f['elong']):>11.2f}{cast:>8}")
    allm = np.concatenate([f["areas"] for f in frames])
    print(f"\n   median region {np.median(allm):.0f} px = "
          f"{np.median(allm) * K.RES * K.RES:.1f} m2; a 0.3 m boulder at 5 deg sun")
    print(f"   should cast roughly 4 to 12 px. Regions in the hundreds are terrain.")

    # ------------------------------------------------------- 3. vote distances
    print()
    print("=" * 74)
    print("3. HOW CLOSE DO DIFFERENT FRAMES PUT THEIR CASTERS?")
    print("   Nearest vote in frame B for each vote in frame A. If this is a few")
    print("   pixels the vote radius is too tight; if it is tens, there is no")
    print("   correspondence to find.\n")
    print(f"   {'pair':<24}{'d_az':>7}{'median':>9}{'p25':>8}{'within 2px':>12}"
          f"{'within 6px':>12}")
    meds = []
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            a, b = frames[i], frames[j]
            if a["votes"].size == 0 or b["votes"].size == 0:
                continue
            d = np.sqrt(((a["votes"][:, None, :] - b["votes"][None, :, :]) ** 2).sum(-1))
            nn = d.min(axis=1)
            daz = circ_diff_180(a["az_map"], b["az_map"])
            meds.append(np.median(nn))
            print(f"   {a['pid'][-10:]}/{b['pid'][-10:]:<13}{daz:>7.0f}"
                  f"{np.median(nn):>9.1f}{np.percentile(nn, 25):>8.1f}"
                  f"{100*(nn <= 2).mean():>11.0f}%{100*(nn <= 6).mean():>11.0f}%")
    if meds:
        m = float(np.median(meds))
        print(f"\n   median over all pairs: {m:.1f} px = {m * K.RES:.1f} m")
        if m <= 8:
            print("   -> frames broadly agree. The vote radius of 2 px is too tight for")
            print("      real shadows, whose up-sun extreme is one ragged pixel. Widen it,")
            print("      or estimate the base robustly instead of by argmax.")
        else:
            print("   -> the frames do not agree on where the casters are. That is not a")
            print("      tuning problem; either these are extended casters or there is")
            print("      nothing compact in this window to find.")

    print()
    print("=" * 74)
    print("Read the three sections in order. Section 1 gates the other two: if the")
    print("sun direction is wrong nothing else means anything.")


if __name__ == "__main__":
    main()
