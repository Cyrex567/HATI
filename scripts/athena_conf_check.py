"""Stereo-confidence check at the Athena touchdown.

Reads NAC_DTM_NOBILE03_CONF (the SOCET stereo confidence map) at the touchdown
and at the safe-reference patch, to test whether Channel 1's fine-scale
roughness flag rests on a well-correlated DEM solution or on matching noise.
Higher confidence = more reliable stereo match (Henriksen et al. 2017).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "athena"
OUT = ROOT / "output" / "athena"
CONF = DATA / "NAC_DTM_NOBILE03_CONF.IMG"
DTM = DATA / "NAC_DTM_NOBILE03.TIF"
ROWS, COLS = 1473, 1172
CONF_ULX, CONF_ULY, SC = 75564.0, 142712.0, 4.0   # magnitude (positive-x) convention

TD_DTM = (1181, 387)     # touchdown, DTM grid
SAFE_DTM = (1173, 541)   # smoothest in-scene patch, DTM grid


def conf_px(x, y):
    return int((CONF_ULY - y) / SC), int((x - CONF_ULX) / SC)


def sample(conf, r, c, k=2):
    win = conf[max(0, r - k):r + k + 1, max(0, c - k):c + k + 1]
    w = win[win > 0]
    return float(conf[r, c]), (float(w.mean()) if w.size else float("nan"))


def main():
    n = ROWS * COLS
    off = os.path.getsize(CONF) - n
    print(f"CONF offset {off}  (file {os.path.getsize(CONF)}, data {n})")
    conf = np.fromfile(CONF, dtype=np.uint8, count=n, offset=off).reshape(ROWS, COLS)
    valid = conf != 0
    v = conf[valid]
    pcts = np.percentile(v, [1, 5, 25, 50, 75, 95, 99]).astype(int)
    print(f"valid {100*valid.mean():.1f}%   min/mean/max {v.min()}/{v.mean():.1f}/{v.max()}")
    print(f"valid CONF pcts 1/5/25/50/75/95/99: {pcts}")

    with rasterio.open(DTM) as src:
        tx, ty = src.xy(*TD_DTM)
        sx, sy = src.xy(*SAFE_DTM)
    tr, tc = conf_px(tx, ty)
    sr, sc = conf_px(sx, sy)
    print(f"touchdown CONF px {(tr,tc)}   safe-ref CONF px {(sr,sc)}")

    tval, tmean = sample(conf, tr, tc)
    sval, smean = sample(conf, sr, sc)
    tpct = float((v < conf[tr, tc]).mean() * 100)
    spct = float((v < conf[sr, sc]).mean() * 100)
    print(f"\n  touchdown : CONF {tval:.0f}  (5x5 mean {tmean:.1f})  -> {tpct:.1f}th percentile")
    print(f"  safe-ref  : CONF {sval:.0f}  (5x5 mean {smean:.1f})  -> {spct:.1f}th percentile")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    m = np.ma.masked_where(~valid, conf)
    im = ax[0].imshow(m, cmap="viridis")
    ax[0].plot(tc, tr, "+", color="cyan", ms=16, mew=2.6, label="touchdown")
    ax[0].plot(sc, sr, "+", color="lime", ms=16, mew=2.6, label="safe ref")
    ax[0].set_title("NOBILE03 stereo-confidence map (higher = better)")
    ax[0].axis("off")
    ax[0].legend(loc="lower left", fontsize=8)
    plt.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].hist(v, bins=50, color="#555")
    ax[1].axvline(conf[tr, tc], color="cyan", lw=2.2, label=f"touchdown ({int(conf[tr,tc])})")
    ax[1].axvline(conf[sr, sc], color="green", lw=2.2, label=f"safe ref ({int(conf[sr,sc])})")
    ax[1].axvline(np.median(v), color="black", lw=1, ls="--", label=f"scene median ({int(np.median(v))})")
    ax[1].set_title("confidence distribution (valid pixels)")
    ax[1].set_xlabel("confidence DN")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "athena_conf_check.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {OUT / 'athena_conf_check.png'}")


if __name__ == "__main__":
    main()
