"""HATI v2.5 -- absolute layer, first increment (roadmap Phases 0-2).

Phase 0: match EFFECTIVE resolution (common 8 m posting + Gaussian to ~16 m FWHM),
         not just posting -- fixes the confound flagged in the mare-control review.
Phase 1: physical-unit channels: footprint-baseline slope (deg) at 16 & 32 m,
         signed relief (max +, max -, p95-p5, m), RMS deviation (m).
Phase 2: LanderSpec presets + deterministic slope gate -> mare false-alarm rate,
         massif rate, Athena-touchdown verdict.

Honest scope: the rock/relief gate at h_crit~0.3 m is below the DEM's resolution,
so at DEM scale the gate is the SLOPE gate; the sub-metre rock gate is the shadow
channel's job (Phase 3). We expect the Athena touchdown (gentle, ~2.7 deg) to PASS
the slope gate -- its hazard is a sub-resolution crater.
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
from athena_counterfactual import load_dtm, fill_nearest, dtm_pixel  # noqa

MARE = ROOT / "data" / "mare" / "NAC_DTM_APOLLO11.TIF"
OUT = ROOT / "output" / "athena"
POST = 8.0                 # common posting (m)
EFF_SIGMA = 1.0            # px -> ~16-19 m effective FWHM, common to both
PRESETS = {"strict (5deg/0.2m)": (5.0, 0.2), "nominal (8deg/0.3m)": (8.0, 0.3),
           "permissive (15deg/0.5m)": (15.0, 0.5)}


def auc(score, lab):
    lab = np.asarray(lab, bool); P, N = int(lab.sum()), int((~lab).sum())
    if P == 0 or N == 0: return float("nan")
    o = np.argsort(-np.asarray(score, float)); t = lab[o]
    tpr = np.concatenate([[0], np.cumsum(t)/P]); fpr = np.concatenate([[0], np.cumsum(~t)/N])
    return float(np.sum(np.diff(fpr)*(tpr[:-1]+tpr[1:])/2))


def common_eff(z, nod, native):
    f = int(round(POST/native))
    z8 = block_reduce(fill_nearest(z, nod), (f, f), np.mean).astype(np.float32)
    n8 = block_reduce(nod.astype(np.float32), (f, f), np.max) > 0
    zb = ndi.gaussian_filter(z8, EFF_SIGMA)
    return zb, ~ndi.binary_dilation(n8, iterations=4)


def slope_deg(z, base_px):
    zs = ndi.gaussian_filter(z, base_px/2.355) if base_px > 1.5 else z
    gy, gx = np.gradient(zs)
    return np.degrees(np.arctan(np.hypot(gx, gy)/POST))


def relief(z, win):
    r = z - ndi.uniform_filter(z, win)
    return (ndi.maximum_filter(r, win), -ndi.minimum_filter(r, win),
            ndi.maximum_filter(z, win) - ndi.minimum_filter(z, win))


def rmsdev(z, win):
    m = ndi.uniform_filter(z, win); ms = ndi.uniform_filter(z*z, win)
    return np.sqrt(np.maximum(ms - m*m, 0))


def chans(z):
    s16 = slope_deg(z, 2.0); s32 = slope_deg(z, 4.0)
    rp, rn, amp = relief(z, 4)          # 32 m window
    return {"slope16 (deg)": s16, "slope32 (deg)": s32,
            "relief+ (m)": rp, "relief- (m)": rn, "relief p2p (m)": amp,
            "RMSdev (m)": rmsdev(z, 4)}


def main():
    ms = rasterio.open(MARE); m = ms.read(1).astype(np.float32)
    mb, mv = common_eff(m, (m <= -1e30) | ~np.isfinite(m), 2.0)
    dem, dnod, _ = load_dtm(); db, dv = common_eff(dem, dnod, 4.0)
    tr, tc = dtm_pixel(); t8 = (tr//2, tc//2)
    mc, dc = chans(mb), chans(db)

    print(f"common eff res ~16-19 m | mare valid {mv.sum()} | massif valid {dv.sum()} | touchdown 8m px {t8}")
    print("\nPhase-1 gate: per-channel CROSS-SITE AUC (massif>mare), at matched effective resolution:")
    keys = list(mc)
    for k in keys:
        a = auc(np.concatenate([dc[k][dv], mc[k][mv]]),
                np.concatenate([np.ones(int(dv.sum())), np.zeros(int(mv.sum()))]))
        flag = "PASS" if a >= 0.80 else ("~"   if a >= 0.70 else "fail")
        print(f"   {k:<16} AUC={a:.3f}  [{flag}]   massif med={np.median(dc[k][dv]):.2g}"
              f"  mare med={np.median(mc[k][mv]):.2g}  touchdown={dc[k][t8]:.2g}")

    print("\nPhase-2 deterministic SLOPE gate (slope16 > theta_max): mare false-alarm + Athena verdict")
    s_m = mc["slope16 (deg)"][mv]; s_d_td = float(dc["slope16 (deg)"][t8])
    rp_td = float(dc["relief+ (m)"][t8]); rn_td = float(dc["relief- (m)"][t8])
    for name, (th, hc) in PRESETS.items():
        fa = float((s_m > th).mean())
        massif_haz = float((dc["slope16 (deg)"][dv] > th).mean())
        verdict = "HAZARD" if s_d_td > th else "pass (not flagged)"
        print(f"   {name:<22} mare FA={100*fa:5.2f}%  massif flagged={100*massif_haz:5.1f}%"
              f"  | touchdown slope={s_d_td:.1f}deg -> {verdict}")
    print(f"\n   touchdown signed relief (32 m window): +{rp_td:.2f} m / -{rn_td:.2f} m"
          f"   (rock gate h_crit~0.3 m is sub-DEM -> shadow channel's job)")

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    aucs = [auc(np.concatenate([dc[k][dv], mc[k][mv]]),
                np.concatenate([np.ones(int(dv.sum())), np.zeros(int(mv.sum()))])) for k in keys]
    ax[0].barh(range(len(keys)), aucs, color=["#1B7A6E" if a >= 0.8 else "#B23A2E" for a in aucs])
    ax[0].axvline(0.8, color="k", ls="--", lw=1); ax[0].axvline(0.5, color="k", ls=":", lw=1)
    ax[0].set_yticks(range(len(keys))); ax[0].set_yticklabels(keys)
    ax[0].set_xlim(0.4, 1.0); ax[0].set_xlabel("cross-site AUC (massif>mare)")
    ax[0].set_title("Phase 1: which physical channels transfer (>=0.8 = green)", fontsize=10, fontweight="bold")
    ax[1].hist(s_m, bins=60, density=True, color="#1B7A6E", alpha=0.6, label="mare (safe)")
    ax[1].hist(dc["slope16 (deg)"][dv], bins=60, density=True, color="#888", alpha=0.55, label="Mons Mouton")
    for name, (th, _) in PRESETS.items():
        ax[1].axvline(th, color="#444", ls="--", lw=1)
        ax[1].text(th, ax[1].get_ylim()[1]*0.9, f"{th:.0f}deg", fontsize=7, rotation=90, va="top")
    ax[1].axvline(s_d_td, color="#B23A2E", lw=2.4, label=f"Athena touchdown ({s_d_td:.1f}deg)")
    ax[1].set_xlim(0, 20); ax[1].set_xlabel("slope at 16 m baseline (deg)"); ax[1].set_ylabel("density")
    ax[1].set_title("Phase 2: the absolute slope gate", fontsize=10, fontweight="bold"); ax[1].legend(fontsize=8)
    fig.suptitle("HATI v2.5 absolute layer: physical channels transfer; slope gate is conservative; "
                 "Athena (gentle) passes -> a sub-resolution / shadow-channel hazard",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT/"hati25_absolute_gate.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"\n  -> {OUT/'hati25_absolute_gate.png'}")


if __name__ == "__main__":
    main()
