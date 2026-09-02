"""Resolution-ladder validation of the HATI pipelines.

Core question: if you only had COARSER data, would HATI still flag the hazards
that are directly visible in FINER data of the same place?

Method (controlled, single variable = resolution):
  Shadow pipeline -- treat the 0.9 m ortho as ground truth; area-average it to
    1.8/2.7/3.6/5.4/7.2 m; run the same shadow detector at each level; measure
    how well the coarse detections recover the full-res ones (ROC/AUC per level,
    and recovery vs obstacle size).
  Heatmap pipeline -- treat the 4 m DTM as ground truth; area-average to 8/16 m;
    run the heatmap on the coarse DTM; test whether it flags cells that hold
    real sub-cell relief in the fine DTM (ROC/AUC).

Degradation is by AREA-AVERAGING (anti-aliased), which models a real coarser
sensor's blur; naive subsampling would alias and manufacture fake roughness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.measure import block_reduce, label, regionprops

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.heatmap import dem_features, fusion  # noqa: E402
from athena_counterfactual import (  # noqa: E402
    load_ortho, ortho_pixel, load_dtm, fill_nearest, build_weights, ORTHO_SCALE)

OUT = ROOT / "output" / "athena"


# ---------------------------------------------------------------------------
def roc_auc(scores, truth):
    truth = np.asarray(truth, bool)
    P, N = int(truth.sum()), int((~truth).sum())
    if P == 0 or N == 0:
        return float("nan"), (np.array([0, 1]), np.array([0, 1]))
    order = np.argsort(-np.asarray(scores, float))
    t = truth[order]
    tp = np.cumsum(t); fp = np.cumsum(~t)
    tpr = np.concatenate([[0.0], tp / P])
    fpr = np.concatenate([[0.0], fp / N])
    auc = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2.0))  # trapezoid
    return auc, (fpr, tpr)


def downsample_curve(fpr, tpr, n=300):
    if len(fpr) <= n:
        return fpr, tpr
    idx = np.linspace(0, len(fpr) - 1, n).astype(int)
    return fpr[idx], tpr[idx]


def shadow_score_and_mask(img, scale_m, frac=0.5, bg_win_m=40.0, min_area=3):
    """Return (continuous darkness score in [0,1], cleaned binary shadow mask)."""
    cov = img > 0
    fillv = np.median(img[cov]) if cov.any() else 0.0
    bg_in = np.where(cov, img, fillv).astype(np.float32)
    win = max(3, int(round(bg_win_m / scale_m)))
    if win % 2 == 0:
        win += 1
    bg = ndi.median_filter(bg_in, size=win)
    score = np.where(bg > 0, (bg - img) / np.maximum(bg, 1e-6), 0.0)
    score = np.clip(score, 0.0, 1.0) * cov
    mask = cov & (img < frac * bg)
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_closing(mask, iterations=1)
    # drop tiny components
    lab = label(mask)
    keep = np.zeros_like(mask)
    for rp in regionprops(lab):
        if rp.area >= min_area:
            keep[lab == rp.label] = True
    return score, keep


# ===========================================================================
def shadow_ladder():
    print("=== SHADOW LADDER ===")
    img = load_ortho().astype(np.float32)
    r, c = ortho_pixel()
    H = 600                       # 1200x1200 px = 1.08 km^2, divisible by 2,3,4,6,8
    crop = img[r - H:r + H, c - H:c + H]
    covf = crop > 0
    sc_fine = 0.9

    score_f, mask_f = shadow_score_and_mask(crop, sc_fine)
    # fine obstacles + footprints (for size-stratified recovery)
    labf = label(mask_f)
    obstacles = []
    for rp in regionprops(labf):
        if rp.area < 3:
            continue
        foot = rp.axis_minor_length * sc_fine
        if foot < sc_fine:
            foot = rp.equivalent_diameter_area * sc_fine
        obstacles.append((int(rp.centroid[0]), int(rp.centroid[1]), foot))
    print(f"  fine ({sc_fine} m): {len(obstacles)} obstacles over 1.08 km^2")

    factors = [2, 3, 4, 6, 8]
    bins = [(0, 2), (2, 4), (4, 8), (8, 16), (16, 1e9)]
    rocs = {}
    recall = {b: [] for b in bins}
    res_list = []
    coarse_masks = {}
    for f in factors:
        sc = sc_fine * f
        res_list.append(sc)
        coarse = block_reduce(crop, (f, f), np.mean)
        cov_c = block_reduce(covf.astype(np.float32), (f, f), np.min) > 0.5  # fully covered cells
        score_c, mask_c = shadow_score_and_mask(coarse, sc)
        # upsample to fine grid
        up = lambda a: np.kron(a, np.ones((f, f)))[:crop.shape[0], :crop.shape[1]]
        score_up = up(score_c)
        mask_up = up(mask_c.astype(np.float32)) > 0.5
        cov_up = up(cov_c.astype(np.float32)) > 0.5
        use = covf & cov_up
        auc, (fpr, tpr) = roc_auc(score_up[use], mask_f[use])
        rocs[sc] = (auc, downsample_curve(fpr, tpr))
        # size-stratified recovery (centroid hit)
        per = {b: [0, 0] for b in bins}
        for (cy, cx, foot) in obstacles:
            for b in bins:
                if b[0] <= foot < b[1]:
                    per[b][1] += 1
                    if cov_up[cy, cx] and mask_up[cy, cx]:
                        per[b][0] += 1
                    break
        for b in bins:
            recall[b].append(per[b][0] / per[b][1] if per[b][1] else np.nan)
        coarse_masks[sc] = (coarse, mask_c, cov_c)
        print(f"  {sc:.1f} m: AUC={auc:.3f}  recall by size "
              + " ".join(f"{int(b[0])}-{('inf' if b[1]>1e8 else int(b[1]))}m:{recall[b][-1]:.2f}" for b in bins))

    return dict(crop=crop, mask_f=mask_f, sc_fine=sc_fine, rocs=rocs,
                recall=recall, bins=bins, res_list=res_list, coarse=coarse_masks,
                n_obst=len(obstacles))


# ===========================================================================
def heatmap_ladder():
    print("=== HEATMAP LADDER ===")
    dem, nod, scale = load_dtm()            # 4 m
    z = fill_nearest(dem, nod).astype(np.float32)
    out = {}
    for f in [2, 4]:
        zc = block_reduce(z, (f, f), np.mean)
        msq = block_reduce(z ** 2, (f, f), np.mean)
        relief = np.sqrt(np.maximum(msq - block_reduce(z, (f, f), np.mean) ** 2, 0.0))
        nod_c = block_reduce(nod.astype(np.float32), (f, f), np.max) > 0  # touches nodata
        sc = scale * f
        ch = dem_features.compute_tier1_stack(zc, scale_m=sc, window_px=7,
                                              mds_scales_px=(2, 3, 5, 8), verbose=False)
        Hc = fusion.fuse(ch, weights=build_weights((2, 3, 5, 8))).heatmap
        # valid cells: away from nodata (dilate a little), finite
        bad = ndi.binary_dilation(nod_c, iterations=4)
        valid = (~bad) & np.isfinite(Hc) & np.isfinite(relief)
        rv = relief[valid]
        thr = np.quantile(rv, 0.75)         # top-quartile sub-cell relief = "real hazard"
        truth = rv > thr
        auc, (fpr, tpr) = roc_auc(Hc[valid], truth)
        out[sc] = (auc, downsample_curve(fpr, tpr))
        print(f"  coarse {sc:.0f} m DTM predicting sub-cell relief in 4 m DTM:"
              f"  AUC={auc:.3f}  (n={valid.sum()} cells, hazard thr={thr:.2f} m)")
    return out


# ===========================================================================
def make_figures(s, h):
    # ---- curves figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for sc, (auc, (fpr, tpr)) in s["rocs"].items():
        ax[0].plot(fpr, tpr, lw=1.8, label=f"{sc:.1f} m  (AUC {auc:.2f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax[0].set_title("Shadow pipeline: recover full-res shadows from coarse data",
                    fontsize=10, fontweight="bold")
    ax[0].set_xlabel("false-positive rate"); ax[0].set_ylabel("true-positive rate")
    ax[0].legend(fontsize=8, title="degraded to", title_fontsize=8); ax[0].set_aspect("equal")

    for b in s["bins"]:
        lab = f"{int(b[0])}-{'inf' if b[1] > 1e8 else int(b[1])} m"
        ax[1].plot(s["res_list"], s["recall"][b], "o-", lw=1.8, ms=4, label=lab)
    ax[1].axvline(8.0, color="red", ls="--", lw=1, alpha=0.6)
    ax[1].text(8.1, 0.05, "4 m DEM\nfloor (8 m)", color="red", fontsize=7, va="bottom")
    ax[1].set_title("Shadow recovery vs resolution, by obstacle footprint",
                    fontsize=10, fontweight="bold")
    ax[1].set_xlabel("degraded resolution (m/px)"); ax[1].set_ylabel("fraction of full-res obstacles recovered")
    ax[1].set_ylim(-0.03, 1.03); ax[1].legend(fontsize=8, title="footprint", title_fontsize=8)

    for sc, (auc, (fpr, tpr)) in h.items():
        ax[2].plot(fpr, tpr, lw=1.8, label=f"{sc:.0f} m DTM  (AUC {auc:.2f})")
    ax[2].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax[2].set_title("Heatmap: coarse-DTM roughness predicts\nsub-cell relief in the 4 m DTM",
                    fontsize=10, fontweight="bold")
    ax[2].set_xlabel("false-positive rate"); ax[2].set_ylabel("true-positive rate")
    ax[2].legend(fontsize=8); ax[2].set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT / "validation_curves.png", dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {OUT/'validation_curves.png'}")

    # ---- visual ladder figure (fine vs two coarse levels, with shadow overlay)
    crop = s["crop"]; vis = crop[crop > 0]
    vlo, vhi = np.percentile(vis, 2), np.percentile(vis, 98)
    levels = [(s["sc_fine"], crop, s["mask_f"])]
    for sc in (3.6, 7.2):
        if sc in s["coarse"]:
            coarse, mask_c, _ = s["coarse"][sc]
            levels.append((sc, coarse, mask_c))
    fig, ax = plt.subplots(1, len(levels), figsize=(5 * len(levels), 5))
    for k, (sc, im, m) in enumerate(levels):
        a = ax[k]
        a.imshow(im, cmap="gray", vmin=vlo, vmax=vhi)
        ov = np.zeros((*im.shape, 4)); ov[m] = [1, 0.25, 0.25, 0.8]
        a.imshow(ov)
        tag = "full res (ground truth)" if k == 0 else "degraded"
        a.set_title(f"{sc:.1f} m/px  -- {tag}", fontsize=10, fontweight="bold")
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("The same patch at three resolutions, with detected shadows (red). "
                 "Small shadows wash out as the data coarsens; large ones survive.",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "validation_ladder.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {OUT/'validation_ladder.png'}")


def main():
    s = shadow_ladder()
    h = heatmap_ladder()
    make_figures(s, h)
    # summary line
    print("\n=== SUMMARY ===")
    print("shadow AUC by resolution:",
          {f"{k:.1f}m": round(v[0], 3) for k, v in s["rocs"].items()})
    print("heatmap AUC by coarse DTM:",
          {f"{k:.0f}m": round(v[0], 3) for k, v in h.items()})


if __name__ == "__main__":
    main()
