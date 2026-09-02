"""Mare control, corrected (per Fable 5's review). Tests whether the earlier
AUC<0.5 was a confound (the +-4 clip + unmatched effective resolution) or real.

Fixes: (1) bring BOTH DEMs to a common 8 m posting (area-average) so effective
resolution is matched, not the raw posting; (2) compare in RAW physical units with
NO clip; (3) per-channel cross-site AUC (does the channel rank the rough massif above
the smooth mare?) including new physical channels slope-degrees and relief-amplitude;
(4) an 8-16 m band-power check to see if the mare is genuinely texture-saturated or
just carrying 2 m-native content the 4 m massif lacks.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import rasterio
from scipy import ndimage as ndi
from skimage.measure import block_reduce

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.heatmap import dem_features  # noqa
from athena_counterfactual import load_dtm, fill_nearest, build_weights, dtm_pixel  # noqa

MARE = ROOT / "data" / "mare" / "NAC_DTM_APOLLO11.TIF"
COMMON = 8.0  # m/px common posting
WIN = 7
MDS = (2, 3, 5, 8)
PRETTY = {"rms_slope": "RMS slope", "iqr_slope": "Slope IQR", "iqr_curvature": "Curv IQR",
          "rms_planar_dev": "Planar dev", "tpi_abs": "|TPI|", "tri": "TRI",
          "mds_L2": "MDS@16m", "mds_L3": "MDS@24m", "mds_L5": "MDS@40m", "mds_L8": "MDS@64m"}


def auc(score, lab):  # lab True = massif (should score higher)
    lab = np.asarray(lab, bool); P, N = int(lab.sum()), int((~lab).sum())
    if P == 0 or N == 0: return float("nan")
    o = np.argsort(-np.asarray(score, float)); t = lab[o]
    tpr = np.concatenate([[0], np.cumsum(t) / P]); fpr = np.concatenate([[0], np.cumsum(~t) / N])
    return float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2))


def slope_deg(z, s):
    gy, gx = np.gradient(z); return np.degrees(np.arctan(np.hypot(gx, gy) / s))


def relief_amp(z, k=3):  # peak-to-peak over (2k+1) window, metres
    return ndi.maximum_filter(z, 2 * k + 1) - ndi.minimum_filter(z, 2 * k + 1)


def to_common(z, nod, native):
    f = int(round(COMMON / native))
    zf = fill_nearest(z, nod)
    z8 = block_reduce(zf, (f, f), np.mean).astype(np.float32)
    n8 = block_reduce(nod.astype(np.float32), (f, f), np.max) > 0
    return z8, ndi.binary_dilation(n8, iterations=WIN)


def chans(z8):
    c = dem_features.compute_tier1_stack(z8, scale_m=COMMON, window_px=WIN,
                                         mds_scales_px=MDS, verbose=False)
    c["slope_deg"] = slope_deg(z8, COMMON)
    c["relief_m"] = relief_amp(z8, 3)  # 48 m window
    return c


def bandpower_8_16(z8, valid):  # variance of (G8 - G16): power in the ~8-16 m band
    bp = ndi.gaussian_filter(z8, 1.0) - ndi.gaussian_filter(z8, 2.0)
    return float(np.var(bp[valid]))


def main():
    # mare (2 m) -> 8 m
    ms = rasterio.open(MARE); m = ms.read(1).astype(np.float32)
    mnod = (m <= -1e30) | ~np.isfinite(m)
    m8, mbad = to_common(m, mnod, 2.0)
    # massif NOBILE03 (4 m) -> 8 m
    dem, dnod, _ = load_dtm(); d8, dbad = to_common(dem, dnod, 4.0)
    mv, dv = ~mbad, ~dbad
    tr, tc = dtm_pixel(); tr8, tc8 = tr // 2, tc // 2  # touchdown at 8 m

    print(f"common posting {COMMON:.0f} m | mare {m8.shape} valid {mv.sum()} | massif {d8.shape} valid {dv.sum()}")
    bpm, bpd = bandpower_8_16(m8, mv), bandpower_8_16(d8, dv)
    print(f"8-16 m band POWER: mare={bpm:.3e}  massif={bpd:.3e}  ratio massif/mare={bpd/bpm:.2f}")
    print("  (>1 => massif genuinely rougher in-band; <1 => mare carries more fine power)")

    mc, dc = chans(m8), chans(d8)
    keys = list(PRETTY) + ["slope_deg", "relief_m"]
    name = {**PRETTY, "slope_deg": "slope (deg)", "relief_m": "relief 48m (m)"}
    print("\nper-channel CROSS-SITE AUC (massif>mare) + touchdown vs mare (no clip):")
    print(f"  {'channel':<14}{'AUC':>6}{'mare_med':>10}{'mare_p95':>10}{'touchdown':>11}{'z_mare(td)':>11}")
    res = {}
    for k in keys:
        mvals, dvals = mc[k][mv], dc[k][dv]
        a = auc(np.concatenate([dvals, mvals]),
                np.concatenate([np.ones(len(dvals)), np.zeros(len(mvals))]))
        med = float(np.median(mvals)); p95 = float(np.percentile(mvals, 95))
        iqr = max(float(np.subtract(*np.percentile(mvals, [75, 25]))) / 1.349, 1e-9)
        tdv = float(dc[k][tr8, tc8]); ztd = (tdv - med) / iqr
        res[k] = a
        print(f"  {name[k]:<14}{a:>6.2f}{med:>10.3g}{p95:>10.3g}{tdv:>11.3g}{ztd:>+11.1f}")

    # fused mare-calibrated H, NO CLIP
    w = build_weights(MDS); totw = sum(abs(v) for v in w.values())
    base = {k: (np.median(mc[k][mv]),
                max(float(np.subtract(*np.percentile(mc[k][mv], [75, 25]))) / 1.349, 1e-9)) for k in w}

    def fused(cd, noclip=True):
        acc = None
        for k in w:
            z = (cd[k] - base[k][0]) / base[k][1]
            if not noclip: z = np.clip(z, -4, 4)
            acc = w[k] * z if acc is None else acc + w[k] * z
        return 1 / (1 + np.exp(-np.clip(acc / totw - 1.0, -30, 30)))
    for tag, nc in [("NO clip", True), ("+-4 clip", False)]:
        Hm = fused(mc, nc)[mv]; Hd = fused(dc, nc)
        Htd = float(Hd[tr8, tc8]); fa = float((Hm >= 0.5).mean())
        a = auc(np.concatenate([Hd[dv], Hm]), np.concatenate([np.ones(int(dv.sum())), np.zeros(len(Hm))]))
        print(f"\nfused mare-calibrated H [{tag}]: touchdown={Htd:.3f}  td-pct-in-mare={100*(Hm<Htd).mean():.1f}%"
              f"  mare-FA(>=0.5)={100*fa:.1f}%  massif>mare AUC={a:.3f}")


if __name__ == "__main__":
    main()
