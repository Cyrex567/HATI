"""Per-channel decomposition of the HATI heatmap AT the Athena touchdown pixel.

Answers: which Tier-1 channels actually drove the hazard score at the landing
point, by how much, and in which direction? Reproduces the representative
config (window=7, MDS 8-32 m) exactly, then reads each channel's z-score,
weight, normalised contribution to the pre-sigmoid logit, and scene percentile
at the touchdown. Saves a sorted contribution bar chart.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.heatmap import dem_features, fusion  # noqa: E402
from athena_counterfactual import (  # noqa: E402
    load_dtm, fill_nearest, build_weights, dtm_pixel)
from scipy import ndimage as ndi  # noqa: E402

OUT = ROOT / "output" / "athena"
WINDOW = 7
MDS = (2, 3, 5, 8)

PRETTY = {
    "rms_slope": "RMS slope",
    "iqr_slope": "Slope IQR",
    "iqr_curvature": "Curvature IQR",
    "rms_planar_dev": "Planar deviation",
    "tpi_abs": "|TPI| (bump/dip)",
    "tri": "TRI (jaggedness)",
    "mds_L2": "MDS @ 8 m",
    "mds_L3": "MDS @ 12 m",
    "mds_L5": "MDS @ 20 m",
    "mds_L8": "MDS @ 32 m",
}


def main():
    dem, nod, scale = load_dtm()
    dem_f = fill_nearest(dem, nod)
    channels = dem_features.compute_tier1_stack(
        dem_f, scale_m=scale, window_px=WINDOW, mds_scales_px=MDS, verbose=False)
    reach = int(np.ceil(8.0 * max(MDS) / 3.0)) + WINDOW // 2 + 1
    nod_dil = ndi.binary_dilation(nod, iterations=reach)
    for k in channels:
        channels[k] = np.where(nod_dil, np.nan, channels[k]).astype(np.float32)

    weights = build_weights(MDS)
    res = fusion.fuse(channels, weights=weights)
    tr, tc = dtm_pixel()
    total_w = sum(abs(w) for w in res.weights.values())

    rows = []
    for name in res.weights:
        z = float(res.channel_zscores[name][tr, tc])
        w = res.weights[name]
        contrib = w * z / total_w           # additive piece of the pre-sigmoid
        raw = channels[name]
        finite = np.isfinite(raw)
        pctile = float((raw[finite] < raw[tr, tc]).mean() * 100)
        rows.append((name, w, z, contrib, pctile))

    rows.sort(key=lambda r: -r[3])
    presig = sum(r[3] for r in rows) + res.bias
    H = 1.0 / (1.0 + np.exp(-presig))

    print(f"Touchdown ({tr},{tc})  config window={WINDOW} MDS={MDS}")
    print(f"{'channel':<18}{'weight':>7}{'z@td':>8}{'contrib':>9}{'value pctile':>14}")
    for name, w, z, c, p in rows:
        print(f"{PRETTY[name]:<18}{w:>7.2f}{z:>8.2f}{c:>9.3f}{p:>13.1f}%")
    print(f"{'sum of contribs':<18}{'':>7}{'':>8}{sum(r[3] for r in rows):>9.3f}")
    print(f"bias {res.bias:+.2f}  ->  pre-sigmoid {presig:.3f}  ->  H = {H:.3f}")

    # --- bar chart, sorted by contribution
    names = [PRETTY[r[0]] for r in rows]
    contribs = [r[3] for r in rows]
    colors = ["#c0392b" if c >= 0 else "#2980b9" for c in contribs]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    y = np.arange(len(names))[::-1]
    ax.barh(y, contribs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("contribution to the touchdown logit  (weight x z-score / total weight)",
                  fontsize=9)
    ax.set_title("What drove the hazard score at the IM-2 Athena touchdown\n"
                 f"(channels summed + bias {res.bias:+.1f}  ->  H = {H:.2f}, "
                 "96th percentile)", fontsize=10.5, fontweight="bold")
    for yi, c in zip(y, contribs):
        ax.text(c + (0.005 if c >= 0 else -0.005), yi, f"{c:+.3f}",
                va="center", ha="left" if c >= 0 else "right", fontsize=8)
    ax.margins(x=0.18)
    fig.tight_layout()
    fig.savefig(OUT / "athena_channel_influence.png", dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  -> {OUT / 'athena_channel_influence.png'}")


if __name__ == "__main__":
    main()
