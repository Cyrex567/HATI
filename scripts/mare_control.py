"""Smooth-mare control: is the Athena touchdown objectively rough, or only rough
relative to its own (rough) neighbourhood? And how often does benign ground flag?

We take a genuinely smooth reference -- the Apollo 11 NAC DTM of Mare Tranquillitatis,
degraded to NOBILE03's 4 m -- and use ITS statistics as a common baseline (never the
per-scene z that the operational heatmap uses). Then:
  * express the Athena touchdown's roughness in mare-relative units (how many
    safe-ground spreads above safe-ground-normal it sits, per channel and fused);
  * apply that mare-calibrated heatmap to the mare itself -> the false-alarm rate on
    benign ground;
  * and to the Mons Mouton scene -> the smooth-vs-rough separability.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from scipy import ndimage as ndi
from skimage.measure import block_reduce

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.heatmap import dem_features  # noqa: E402
from athena_counterfactual import load_dtm, fill_nearest, build_weights, dtm_pixel  # noqa: E402

MARE = ROOT / "data" / "mare" / "NAC_DTM_APOLLO11.TIF"
OUT = ROOT / "output" / "athena"
SCALE, WINDOW, MDS = 4.0, 7, (2, 3, 5, 8)
WEIGHTS = build_weights(MDS)
BIAS = -1.0
REACH = int(np.ceil(8.0 * max(MDS) / 3.0)) + WINDOW // 2 + 1
PRETTY = {"rms_slope": "RMS slope", "iqr_slope": "Slope IQR", "iqr_curvature": "Curvature IQR",
          "rms_planar_dev": "Planar dev", "tpi_abs": "|TPI|", "tri": "TRI",
          "mds_L2": "MDS@8m", "mds_L3": "MDS@12m", "mds_L5": "MDS@20m", "mds_L8": "MDS@32m"}


def g_channels(z, nod):
    """log1p of the Tier-1 stack on a filled DEM; plus the dilated-nodata mask."""
    zf = fill_nearest(z, nod)
    ch = dem_features.compute_tier1_stack(zf, scale_m=SCALE, window_px=WINDOW,
                                          mds_scales_px=MDS, verbose=False)
    g = {k: np.log1p(np.maximum(v, 0.0)).astype(np.float32) for k, v in ch.items()}
    return g, ndi.binary_dilation(nod, iterations=REACH)


def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-np.clip(a, -30, 30)))


def main():
    totw = sum(abs(WEIGHTS[k]) for k in WEIGHTS)

    # ---- mare: degrade 2 m -> 4 m, compute channels, build the safe baseline
    ms = rasterio.open(MARE)
    m2 = ms.read(1).astype(np.float32)
    mnod2 = (m2 <= -1e30) | ~np.isfinite(m2)
    m2f = fill_nearest(m2, mnod2)
    mare4 = block_reduce(m2f, (2, 2), np.mean).astype(np.float32)
    mnod4 = block_reduce(mnod2.astype(np.float32), (2, 2), np.max) > 0
    mg, mbad = g_channels(mare4, mnod4)
    mvalid = ~mbad
    base = {k: (float(np.median(mg[k][mvalid])),
                max(float(np.subtract(*np.percentile(mg[k][mvalid], [75, 25]))) / 1.349, 1e-9))
            for k in mg}
    print(f"mare (Apollo 11, 4 m): {mvalid.sum()} valid cells -> safe baseline built")

    # ---- NOBILE03 (4 m): channels at the Athena touchdown
    dem, dnod, _ = load_dtm()
    dg, dbad = g_channels(dem, dnod)
    tr, tc = dtm_pixel()

    # ---- mare-relative z at the touchdown (per channel), unclipped + clipped-fused
    print("\nMare-relative roughness at the Athena touchdown (safe-ground spreads above safe-normal):")
    rows = []
    for k in WEIGHTS:
        z = (dg[k][tr, tc] - base[k][0]) / base[k][1]
        rows.append((PRETTY[k], z, WEIGHTS[k]))
    for name, z, w in sorted(rows, key=lambda r: -r[1]):
        print(f"   {name:<14} z_mare = {z:+6.1f}   (w={w:.2f})")
    a_td = sum(WEIGHTS[k] * np.clip((dg[k][tr, tc] - base[k][0]) / base[k][1], -4, 4)
               for k in WEIGHTS) / totw + BIAS
    H_td = float(sigmoid(a_td))

    # ---- mare-calibrated heatmap applied to mare (false alarms) and NOBILE03 (contrast)
    def marecal_H(gdict):
        acc = None
        for k in WEIGHTS:
            z = np.clip((gdict[k] - base[k][0]) / base[k][1], -4, 4)
            acc = WEIGHTS[k] * z if acc is None else acc + WEIGHTS[k] * z
        return sigmoid(acc / totw + BIAS)

    Hmare = marecal_H(mg)[mvalid]
    Hnob = marecal_H(dg)[~dbad]
    fa_half = float((Hmare >= 0.5).mean())
    fa_td = float((Hmare >= H_td).mean())
    td_pct = float((Hmare < H_td).mean() * 100)

    # separability: mare(0) vs Mons Mouton(1) under the mare-calibrated heatmap
    scores = np.concatenate([Hmare, Hnob])
    labels = np.concatenate([np.zeros(len(Hmare)), np.ones(len(Hnob))]).astype(bool)
    order = np.argsort(-scores); t = labels[order]
    P, N = int(t.sum()), int((~t).sum())
    tpr = np.concatenate([[0], np.cumsum(t) / P]); fpr = np.concatenate([[0], np.cumsum(~t) / N])
    auc = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2.0))

    print(f"\nMare-calibrated hazard H:")
    print(f"   touchdown H            = {H_td:.3f}")
    print(f"   mare median / p99 / max= {np.median(Hmare):.3f} / {np.percentile(Hmare,99):.3f} / {Hmare.max():.3f}")
    print(f"   touchdown percentile within the mare = {td_pct:.2f}%")
    print(f"   false alarms on mare: H>=0.5 -> {100*fa_half:.2f}% ;  H>=touchdown -> {100*fa_td:.3f}%")
    print(f"   smooth-mare vs Mons Mouton separability AUC = {auc:.3f}")

    # ---- figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    # A: per-channel mare distribution (box) + touchdown marker
    order_k = [k for k in WEIGHTS]
    data = [mg[k][mvalid] for k in order_k]
    bp = ax[0].boxplot(data, vert=False, showfliers=False, widths=0.6,
                       patch_artist=True, labels=[PRETTY[k] for k in order_k])
    for b in bp["boxes"]:
        b.set(facecolor="#1B7A6E", alpha=0.5)
    for i, k in enumerate(order_k):
        ax[0].plot(dg[k][tr, tc], i + 1, "D", color="#B23A2E", ms=7, zorder=5)
    ax[0].set_title("Per channel: smooth-mare spread (green) vs the Athena touchdown (red)",
                    fontsize=10, fontweight="bold")
    ax[0].set_xlabel(r"$\log(1+\mathrm{channel})$ value")
    # B: mare-calibrated H distributions
    ax[1].hist(Hmare, bins=60, density=True, color="#1B7A6E", alpha=0.6, label="Mare Tranquillitatis (safe)")
    ax[1].hist(Hnob, bins=60, density=True, color="#888", alpha=0.55, label="Mons Mouton scene")
    ax[1].axvline(H_td, color="#B23A2E", lw=2.4, label=f"Athena touchdown (H={H_td:.2f})")
    ax[1].axvline(0.5, color="black", ls=":", lw=1)
    ax[1].set_title(f"Mare-calibrated hazard: false alarm on mare = {100*fa_half:.1f}% (H$\\geq$0.5)\n"
                    f"smooth-vs-rough AUC = {auc:.2f}", fontsize=10, fontweight="bold")
    ax[1].set_xlabel("mare-calibrated $H$"); ax[1].set_ylabel("density"); ax[1].legend(fontsize=8)
    fig.suptitle("Smooth-mare control: is the touchdown objectively rough, and how often does benign ground flag?",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "mare_control.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  -> {OUT/'mare_control.png'}")


if __name__ == "__main__":
    main()
