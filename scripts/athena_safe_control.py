"""Safe-ground control for the HATI heatmap: the 'trained eye'.

Contrasts the Athena touchdown against the smoothest patch in the SAME NOBILE03
scene, at the SAME 4 m/px resolution, with the SAME config. For each Tier-1
channel it reports the z-score at the hazard vs at safe ground, and the
discrimination (z_hazard - z_safe): how much each channel separates the two.

Why within-scene, not an external mare: roughness is scale-dependent, so a
control DEM at a different pixel size (e.g. the 1.5 m/px Apollo 17 model on
disk) would compare different physical scales; and z-scores are normalised
per scene, so they are not comparable across two separate runs. Holding the
scene, product and resolution fixed removes both confounds. An external mare
control is still worthwhile but must match resolution and compare raw (or
commonly-normalised) values, not per-scene z.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.heatmap import dem_features, fusion  # noqa: E402
from athena_counterfactual import (  # noqa: E402
    load_dtm, fill_nearest, build_weights, dtm_pixel)

OUT = ROOT / "output" / "athena"
WINDOW = 7
MDS = (2, 3, 5, 8)
PRETTY = {
    "rms_slope": "RMS slope", "iqr_slope": "Slope IQR",
    "iqr_curvature": "Curvature IQR", "rms_planar_dev": "Planar deviation",
    "tpi_abs": "|TPI|", "tri": "TRI",
    "mds_L2": "MDS @ 8 m", "mds_L3": "MDS @ 12 m",
    "mds_L5": "MDS @ 20 m", "mds_L8": "MDS @ 32 m",
}


def main():
    dem, nod, scale = load_dtm()
    demf = fill_nearest(dem, nod)
    ch = dem_features.compute_tier1_stack(
        demf, scale_m=scale, window_px=WINDOW, mds_scales_px=MDS, verbose=False)
    reach = int(np.ceil(8.0 * max(MDS) / 3.0)) + WINDOW // 2 + 1
    nod_dil = ndi.binary_dilation(nod, iterations=reach)
    for k in ch:
        ch[k] = np.where(nod_dil, np.nan, ch[k]).astype(np.float32)
    res = fusion.fuse(ch, weights=build_weights(MDS))
    H = np.where(nod_dil, np.nan, res.heatmap).astype(np.float32)
    slope = np.degrees(dem_features._slope_magnitude(demf, scale))

    tr, tc = dtm_pixel()

    # safe reference: robustly-low-H patch, well inside valid data
    core = ndi.binary_erosion(~nod_dil, iterations=20)
    finite = np.isfinite(H)
    num = ndi.uniform_filter(np.where(finite, H, 0.0), 11)
    den = ndi.uniform_filter(finite.astype(np.float32), 11)
    meanH = np.where(den > 0.5, num / np.maximum(den, 1e-6), np.inf)
    meanH[~core] = np.inf
    sr, sc = np.unravel_index(np.argmin(meanH), meanH.shape)

    print(f"touchdown  (row {tr}, col {tc})  H={H[tr,tc]:.3f}  slope={slope[tr,tc]:.1f} deg")
    print(f"safe ref   (row {sr}, col {sc})  H={H[sr,sc]:.3f}  slope={slope[sr,sc]:.1f} deg "
          f"(11x11 mean H={meanH[sr,sc]:.3f})")
    sep_km = np.hypot(tr - sr, tc - sc) * scale / 1000.0
    print(f"separation: {sep_km:.2f} km within the same scene\n")

    rows = []
    for name in res.weights:
        ztd = float(res.channel_zscores[name][tr, tc])
        zsf = float(res.channel_zscores[name][sr, sc])
        rows.append((name, ztd, zsf, ztd - zsf))
    rows.sort(key=lambda r: -r[1])

    print(f"{'channel':<16}{'z_hazard':>9}{'z_safe':>8}{'discrim.':>9}")
    for n, a, b, d in rows:
        print(f"{PRETTY[n]:<16}{a:>9.2f}{b:>8.2f}{d:>9.2f}")
    disc = sorted(rows, key=lambda r: -r[3])
    print("\nstrongest discriminators (z_hazard - z_safe):")
    for n, a, b, d in disc[:4]:
        print(f"  {PRETTY[n]:<16} {d:+.2f}")
    print("weakest (along for the ride here):")
    for n, a, b, d in disc[-2:]:
        print(f"  {PRETTY[n]:<16} {d:+.2f}")

    # ---- figure: grouped horizontal bars, z_hazard vs z_safe
    names = [PRETTY[r[0]] for r in rows]
    ztd = [r[1] for r in rows]
    zsf = [r[2] for r in rows]
    y = np.arange(len(names))[::-1]
    h = 0.38
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax.barh(y + h/2, ztd, height=h, color="#B23A2E", edgecolor="black",
            linewidth=0.4, label="at Athena touchdown (hazard)")
    ax.barh(y - h/2, zsf, height=h, color="#1B7A6E", edgecolor="black",
            linewidth=0.4, label="at smoothest in-scene patch (safe)")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("channel z-score (same scene, same 4 m/px, same config)", fontsize=9)
    ax.set_title("Safe-ground control: which channels actually discriminate\n"
                 "wide red–teal gap = a real discriminator; both low = along for the ride",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
    ax.margins(y=0.02)
    fig.tight_layout()
    fig.savefig(OUT / "athena_safe_control.png", dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"\n  -> {OUT / 'athena_safe_control.png'}")


if __name__ == "__main__":
    main()
