"""Proper resolution-ladder validation of the HATI heatmap, using a finer
(1.5 m) NAC DTM as ground truth -- the DEM analogue of the shadow ladder.

The earlier heatmap test returned chance because its "truth" was the sub-cell
relief that area-averaging *erases* (unrecoverable by construction). Here the
truth is the real, resolved relief features the FINE 1.5 m DTM shows. We degrade
the fine DTM to 3/6/12 m, run the heatmap on each coarse version, and ask how
well it recovers the fine-data hazards. Features bigger than the coarse floor
survive averaging and imprint on coarse roughness; smaller ones wash out -- so we
expect a real, declining detection curve, not a flat 0.5.

Site: Apollo 17 (Taurus-Littrow), a mid-latitude valley+massif scene -- different
terrain from the Athena polar site, which is good for generality. (It is also the
scene the heatmap window/scale config was first demonstrated on; the weights are
literature-fixed, so no fitting happens here, but an independent site would
further harden the result.)
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
from src.heatmap import dem_features, fusion  # noqa: E402
from athena_counterfactual import build_weights  # noqa: E402

DTM = ROOT / "data" / "APOLLO17_DTM_150CM.tiff"
OUT = ROOT / "output" / "athena"
FINE = 1.5                       # m/px
TRUTH_WIN_M = 9.0                # relief window (lander-scale features)
WINDOW, MDS = 7, (2, 3, 5, 8)    # same heatmap config as the Athena representative run
FACTORS = [2, 4, 8]              # -> 3, 6, 12 m


def roc_auc(scores, truth):
    truth = np.asarray(truth, bool)
    P, N = int(truth.sum()), int((~truth).sum())
    if P == 0 or N == 0:
        return float("nan"), (np.array([0, 1]), np.array([0, 1]))
    order = np.argsort(-np.asarray(scores, float))
    t = truth[order]
    tp = np.cumsum(t); fp = np.cumsum(~t)
    tpr = np.concatenate([[0.0], tp / P]); fpr = np.concatenate([[0.0], fp / N])
    auc = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2.0))
    idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 300)).astype(int)
    return auc, (fpr[idx], tpr[idx])


def main():
    with rasterio.open(DTM) as src:
        full = src.read(1).astype(np.float32)
        H, W = full.shape
    # representative 2400x2400 crop (valley floor + massif slopes), divisible by 2,4,8
    r0, c0, S = H // 2 - 1200, W // 2 - 1200, 2400
    z = full[r0:r0 + S, c0:c0 + S].copy()
    print(f"Apollo17 fine DTM crop {z.shape} @ {FINE} m/px (3.6 km), "
          f"relief {z.max()-z.min():.0f} m")

    # --- ground truth: resolved relief features in the FINE DTM (peak-to-peak)
    wpx = int(round(TRUTH_WIN_M / FINE))
    relief = (ndi.maximum_filter(z, wpx) - ndi.minimum_filter(z, wpx))
    truth_thr = np.quantile(relief, 0.75)
    truth_fine = relief > truth_thr
    print(f"  truth = fine {TRUTH_WIN_M:.0f} m relief > {truth_thr:.1f} m "
          f"(top quartile, {100*truth_fine.mean():.0f}% of pixels)")

    rocs, det_rate, res_list = {}, [], []
    for f in FACTORS:
        sc = FINE * f
        res_list.append(sc)
        zc = block_reduce(z, (f, f), np.mean)
        ch = dem_features.compute_tier1_stack(zc, scale_m=sc, window_px=WINDOW,
                                              mds_scales_px=MDS, verbose=False)
        Hc = fusion.fuse(ch, weights=build_weights(MDS)).heatmap
        # upsample coarse heatmap to the fine grid; score against FIXED fine truth
        Hup = np.kron(Hc, np.ones((f, f)))[:S, :S]
        mfine = 12 * f                      # drop coarse convolution-edge margin (fine px)
        interior = np.zeros((S, S), bool); interior[mfine:-mfine, mfine:-mfine] = True
        valid = interior & np.isfinite(Hup)
        auc, curve = roc_auc(Hup[valid], truth_fine[valid])
        op = np.quantile(Hup[valid], 0.75)  # flag top-quartile H; recall of fine hazards
        flagged = Hup >= op
        dr = float((flagged & truth_fine & valid).sum() / max((truth_fine & valid).sum(), 1))
        det_rate.append(dr)
        rocs[sc] = (auc, curve)
        print(f"  coarse {sc:>4.0f} m  (floor {2*sc:.0f} m):  AUC={auc:.3f}   "
              f"recall@top-quartile-H={dr:.2f}   fine-px scored={valid.sum()}")

    # ---- figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for sc, (auc, (fpr, tpr)) in rocs.items():
        ax[0].plot(fpr, tpr, lw=1.9, label=f"{sc:.0f} m DTM  (AUC {auc:.2f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax[0].set_title("Heatmap on a degraded DTM recovers fine-data hazards",
                    fontsize=10.5, fontweight="bold")
    ax[0].set_xlabel("false-positive rate"); ax[0].set_ylabel("true-positive rate")
    ax[0].legend(fontsize=8, title="degraded to", title_fontsize=8); ax[0].set_aspect("equal")

    aucs = [rocs[s][0] for s in res_list]
    ax[1].plot(res_list, aucs, "o-", lw=2, ms=6, color="#1F3A5F", label="detection AUC")
    ax[1].plot(res_list, det_rate, "s--", lw=1.6, ms=5, color="#B23A2E", label="recall @ top-quartile H")
    ax[1].axhline(0.5, color="black", ls=":", lw=1, alpha=0.5)
    ax[1].set_title("Heatmap reach: performance vs DTM resolution",
                    fontsize=10.5, fontweight="bold")
    ax[1].set_xlabel("degraded resolution (m/px)"); ax[1].set_ylabel("score")
    ax[1].set_ylim(0.3, 1.02); ax[1].legend(fontsize=8.5)
    fig.suptitle("Heatmap validation on a 1.5 m NAC DTM (Apollo 17): degrade, score against full-res relief features",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "heatmap_validation.png", dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {OUT/'heatmap_validation.png'}")
    print("\nSUMMARY heatmap AUC by resolution:",
          {f"{s:.0f}m": round(rocs[s][0], 3) for s in res_list})


if __name__ == "__main__":
    main()
