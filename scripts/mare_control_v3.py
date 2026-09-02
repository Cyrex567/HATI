"""Mare control v3 -- isolating WHY the texture channels failed to transfer.

v2 found texture/variability channels do not transfer between the smooth Apollo-11
mare and the rough Mons Mouton massif (cross-site AUC 0.21-0.42) while physical
channels do (0.78-0.82). Two candidate explanations were conflated:

  (1) EFFECTIVE RESOLUTION mismatch. v2 matched the *posting* (both to 8 m) but not
      the effective resolution. The mare is 2 m native block-reduced 4x, so it still
      carries genuine 8 m texture; NOBILE03 is 4 m native (effective ~10 m) reduced
      only 2x, so it does not. That asymmetry inflates the mare's texture channels.

  (2) PIXEL-DEFINED CHANNELS (audit finding A0.1). Windows and the np.gradient slope
      baseline are pixel quantities, so they only measure the same physical thing when
      the postings match. (They did match in v2 -- so this is expected to matter less
      here than it does for general cross-DEM use.)

Three configurations isolate them:

  A  posting-matched only, legacy pixel channels          <- reproduces v2
  B  posting + EFFECTIVE-RESOLUTION matched, legacy       <- adds fix (1)
  C  posting + effective-resolution matched, PHYSICAL     <- adds fix (2)

A->B measures the resolution-matching effect. B->C measures the physical-definition
effect. Per-channel cross-site AUC (massif ranked above mare) is reported for each.

Honest scope: this tests whether the negative result is an artifact of measurement
setup. It does not, by itself, establish that texture channels are useful.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage as ndi
from skimage.measure import block_reduce

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.heatmap import dem_features as df                      # noqa: E402
from athena_counterfactual import load_dtm, fill_nearest         # noqa: E402

MARE = ROOT / "data" / "mare" / "NAC_DTM_APOLLO11.TIF"
COMMON = 8.0            # common posting (m/px)
EFF_FWHM = 20.0         # common effective resolution for configs B and C (m)
WIN_PX = 7              # legacy window (px)  -> 56 m at 8 m posting
MDS_PX = (2, 3, 5, 8)   # legacy MDS scales (px)
WIN_M = WIN_PX * COMMON                     # 56 m
MDS_M = tuple(p * COMMON for p in MDS_PX)   # 16, 24, 40, 64 m
SLOPE_BASE_M = 2 * COMMON                   # 16 m -- matches the legacy 2-px baseline

PRETTY = {"rms_slope": "RMS slope", "iqr_slope": "Slope IQR", "iqr_curvature": "Curv IQR",
          "rms_planar_dev": "Planar dev", "tpi_abs": "|TPI|", "tri": "TRI"}
TEXTURE = {"iqr_slope", "iqr_curvature", "rms_planar_dev", "tpi_abs"}


def auc(score: np.ndarray, lab: np.ndarray) -> float:
    """P(random massif pixel scores above a random mare pixel)."""
    lab = np.asarray(lab, bool)
    P, N = int(lab.sum()), int((~lab).sum())
    if P == 0 or N == 0:
        return float("nan")
    t = lab[np.argsort(-np.asarray(score, float))]
    tpr = np.concatenate([[0], np.cumsum(t) / P])
    fpr = np.concatenate([[0], np.cumsum(~t) / N])
    return float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2))


def to_common(z: np.ndarray, nod: np.ndarray, native: float,
              eff_fwhm: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Block-average to the common posting; optionally blur to a common effective
    resolution so both scenes carry the same amount of real detail."""
    f = int(round(COMMON / native))
    z8 = block_reduce(fill_nearest(z, nod), (f, f), np.mean).astype(np.float32)
    n8 = block_reduce(nod.astype(np.float32), (f, f), np.max) > 0
    if eff_fwhm:
        # blur each scene from its OWN native effective resolution up to the target
        own = max(2.0 * native, COMMON)
        extra = max(eff_fwhm ** 2 - own ** 2, 0.0) ** 0.5
        if extra > 0:
            z8 = ndi.gaussian_filter(z8, (extra / COMMON) / 2.3548)
    valid = ~ndi.binary_dilation(n8, iterations=WIN_PX)
    return z8, valid


def stack(z: np.ndarray, physical: bool) -> dict[str, np.ndarray]:
    if physical:
        return df.compute_tier1_stack(z, scale_m=COMMON, window_m=WIN_M,
                                      mds_baselines_m=MDS_M, slope_baseline_m=SLOPE_BASE_M,
                                      strict=False, verbose=False)
    return df.compute_tier1_stack(z, scale_m=COMMON, window_px=WIN_PX,
                                  mds_scales_px=MDS_PX, verbose=False)


def base_name(k: str) -> str:
    """Map legacy mds_L2 and physical mds_B16m onto one comparable label."""
    if k.startswith("mds_L"):
        return f"MDS@{int(k[5:]) * COMMON:.0f}m"
    if k.startswith("mds_B"):
        return f"MDS@{float(k[5:-1]):.0f}m"   # strip the trailing 'm' only
    return PRETTY.get(k, k)


def run(mare: np.ndarray, mvalid: np.ndarray, mas: np.ndarray, dvalid: np.ndarray,
        physical: bool, tag: str) -> dict[str, float]:
    mc, dc = stack(mare, physical), stack(mas, physical)
    lab = np.concatenate([np.ones(int(dvalid.sum())), np.zeros(int(mvalid.sum()))])
    out = {}
    for k in mc:
        out[base_name(k)] = auc(np.concatenate([dc[k][dvalid], mc[k][mvalid]]), lab)
    print(f"  [{tag}] {len(out)} channels scored "
          f"({int(dvalid.sum())} massif / {int(mvalid.sum())} mare px)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mare-window-px", type=int, default=2400,
                    help="native-resolution mare crop side (2 m/px); 0 = full scene")
    args = ap.parse_args()

    print("loading NOBILE03 (massif, 4 m) ...", flush=True)
    dem, dnod, _ = load_dtm()
    mas_A, dvalid_A = to_common(dem, dnod, 4.0, None)
    mas_B, dvalid_B = to_common(dem, dnod, 4.0, EFF_FWHM)

    print(f"loading Apollo 11 mare (2 m), crop {args.mare_window_px or 'full'} px ...", flush=True)
    with rasterio.open(MARE) as src:
        if args.mare_window_px:
            s = args.mare_window_px
            r0, c0 = max((src.height - s) // 2, 0), max((src.width - s) // 2, 0)
            m = src.read(1, window=Window(c0, r0, min(s, src.width), min(s, src.height)))
        else:
            m = src.read(1)
    m = m.astype(np.float32)
    mnod = (m <= -1e30) | ~np.isfinite(m)
    mare_A, mvalid_A = to_common(m, mnod, 2.0, None)
    mare_B, mvalid_B = to_common(m, mnod, 2.0, EFF_FWHM)

    print(f"\ncommon posting {COMMON:.0f} m | effective-resolution target {EFF_FWHM:.0f} m FWHM"
          f" | window {WIN_M:.0f} m | slope baseline {SLOPE_BASE_M:.0f} m\n", flush=True)

    A = run(mare_A, mvalid_A, mas_A, dvalid_A, False, "A posting-matched, legacy")
    B = run(mare_B, mvalid_B, mas_B, dvalid_B, False, "B eff-res matched, legacy")
    C = run(mare_B, mvalid_B, mas_B, dvalid_B, True, "C eff-res matched, PHYSICAL")

    keys = list(A)
    print(f"\n{'channel':<14}{'A v2-repro':>12}{'B eff-res':>11}{'C physical':>12}"
          f"{'A->C':>8}   verdict")
    print("  (0.50 = coin flip. Below 0.50 the channel ranks SMOOTH ground above ROUGH.)")
    print("-" * 84)
    for k in keys:
        a, b, c = A.get(k, np.nan), B.get(k, np.nan), C.get(k, np.nan)
        d = c - a
        # A verdict must be decided by WHERE the channel lands, never by how far
        # it moved. A channel can improve a lot and still be below a coin flip.
        if c >= 0.70:
            v = "KEEP transfers"
        elif c >= 0.55:
            v = "weak, under the 0.70 bar"
        elif c >= 0.45:
            v = "CUT no signal (~chance)"
        else:
            v = "CUT below chance, ranks smooth above rough"
            if d > 0.10:
                v += f" (rose {d:+.2f}, still under 0.50)"
        print(f"{k:<14}{a:>12.3f}{b:>11.3f}{c:>12.3f}{d:>+8.3f}   {v}")

    tex = [k for k in keys if any(k.startswith(p) for p in ("Slope IQR", "Curv IQR",
                                                            "Planar dev", "|TPI|", "MDS@"))]
    tex = [k for k in tex if k in A and k in C]          # only channels present in both
    mA = float(np.nanmean([A[k] for k in tex])); mC = float(np.nanmean([C[k] for k in tex]))
    print(f"\ntexture-channel mean AUC:  A {mA:.3f}  ->  C {mC:.3f}   ({mC - mA:+.3f})")
    if mC >= 0.60:
        print("VERDICT: the v2 negative result was substantially a MEASUREMENT ARTIFACT.")
    elif mC - mA > 0.10:
        print("VERDICT: partially artifact -- texture recovers but stays weak. Report both.")
    else:
        print("VERDICT: the negative result is INTRINSIC. Texture channels do not transfer;\n"
              "         cutting the texture layer is the honest simplification.")


if __name__ == "__main__":
    main()
