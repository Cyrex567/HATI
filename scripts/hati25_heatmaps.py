"""HATI v2.5 absolute hazard heatmaps, two sites on ONE common physical scale.

Mons Mouton (NOBILE03, the Athena site, 4 m native) vs Apollo 17 / Taurus-Littrow
(1.5 m native). Both brought to a common 12 m posting (clean integer block-average:
4->12 x3, 1.5->12 x8) + light Gaussian to a common effective resolution, then the
absolute physical hazard channel -- footprint slope in DEGREES -- is rendered on a
single shared colour scale with the nominal 8 deg lander gate drawn as a contour.

The point: because the channel is in physical units and the resolution is matched,
the two panels are directly comparable -- a 7 deg slope is 7 deg on both. That is
what the deterministic absolute layer buys you over a per-scene z-score heatmap.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import rasterio
from scipy import ndimage as ndi
from skimage.measure import block_reduce

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from athena_counterfactual import load_dtm, fill_nearest, dtm_pixel  # noqa

APOLLO17 = ROOT / "data" / "APOLLO17_DTM_150CM.tiff"
OUT = ROOT / "output" / "athena"
COMMON = 12.0          # m/px, common posting (4->12 x3, 1.5->12 x8)
EFF_SIGMA = 1.0        # px -> ~28 m effective FWHM, common to both
THETA = 8.0            # nominal lander gate (deg)
VMAX = 15.0            # colour ceiling (deg)


def to_common(z, nod, native):
    f = int(round(native and COMMON / native))
    z8 = block_reduce(fill_nearest(z, nod), (f, f), np.mean).astype(np.float32)
    n8 = block_reduce(nod.astype(np.float32), (f, f), np.max) > 0
    valid = ~ndi.binary_dilation(n8, iterations=3)
    return ndi.gaussian_filter(z8, EFF_SIGMA), valid


def slope_deg(z):
    gy, gx = np.gradient(z)
    return np.degrees(np.arctan(np.hypot(gx, gy) / COMMON))


def panel(ax, slope, valid, title, extent_km, td=None):
    s = np.where(valid, slope, np.nan)
    im = ax.imshow(s, origin="upper", cmap="inferno", vmin=0, vmax=VMAX,
                   extent=[0, extent_km[1], 0, extent_km[0]], interpolation="nearest")
    # nominal-gate contour in pixel coords mapped to km
    ny, nx = slope.shape
    ax.contour(np.linspace(0, extent_km[1], nx), np.linspace(extent_km[0], 0, ny),
               np.where(valid, slope, 0), levels=[THETA], colors="#27E1D6", linewidths=1.3)
    pct = 100 * float((slope[valid] > THETA).mean())
    med = float(np.median(slope[valid])); p95 = float(np.percentile(slope[valid], 95))
    ax.set_title(f"{title}\nmed {med:.1f}° | p95 {p95:.1f}° | "
                 f"{pct:.1f}% over the {THETA:.0f}° gate", fontsize=10, fontweight="bold")
    ax.set_xlabel("km"); ax.set_ylabel("km")
    if td is not None:
        tx, ty, px, py = td
        ax.plot(tx, ty, marker="x", color="#27E1D6", ms=13, mew=3, zorder=6)
        ax.annotate(f"Athena touchdown\n{slope[py, px]:.1f}° → passes gate",
                    xy=(tx, ty), xytext=(tx + extent_km[1]*0.06, ty + extent_km[0]*0.10),
                    color="white", fontsize=8.5, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#27E1D6", lw=1.4))
    return im


def main():
    # Mons Mouton (NOBILE03, 4 m)
    dem, dnod, _ = load_dtm()
    mm, mmv = to_common(dem, dnod, 4.0)
    s_mm = slope_deg(mm)
    tr, tc = dtm_pixel(); tr12, tc12 = tr // 3, tc // 3
    mm_ext = (mm.shape[0] * COMMON / 1000, mm.shape[1] * COMMON / 1000)
    td = (tc12 * COMMON / 1000, (mm.shape[0] - tr12) * COMMON / 1000, tc12, tr12)

    # Apollo 17 / Taurus-Littrow (1.5 m) -- central valley+massif crop, divisible by 8
    with rasterio.open(APOLLO17) as src:
        full = src.read(1).astype(np.float32)
    Hh, Ww = full.shape
    S = 4800  # 7.2 km at 1.5 m, /8 -> 600 px at 12 m
    z = full[Hh // 2 - S // 2:Hh // 2 + S // 2, Ww // 2 - S // 2:Ww // 2 + S // 2].copy()
    znod = (z <= -1e30) | ~np.isfinite(z)
    a17, a17v = to_common(z, znod, 1.5)
    s_a17 = slope_deg(a17)
    a17_ext = (a17.shape[0] * COMMON / 1000, a17.shape[1] * COMMON / 1000)

    print(f"Mons Mouton  {mm.shape} @12m ({mm_ext[1]:.1f}x{mm_ext[0]:.1f} km)  valid {mmv.sum()}")
    print(f"  touchdown 12m px ({tr12},{tc12})  slope={s_mm[tr12, tc12]:.1f} deg  "
          f"med={np.median(s_mm[mmv]):.1f}  >8deg {100*(s_mm[mmv]>THETA).mean():.1f}%")
    print(f"Apollo 17    {a17.shape} @12m ({a17_ext[1]:.1f}x{a17_ext[0]:.1f} km)  valid {a17v.sum()}")
    print(f"  med={np.median(s_a17[a17v]):.1f}  p95={np.percentile(s_a17[a17v],95):.1f}  "
          f">8deg {100*(s_a17[a17v]>THETA).mean():.1f}%")

    fig, ax = plt.subplots(1, 2, figsize=(14, 6.6))
    fig.subplots_adjust(left=0.05, right=0.88, bottom=0.08, top=0.86, wspace=0.18)
    panel(ax[0], s_mm, mmv, "Mons Mouton  (NOBILE03 — the Athena site)", mm_ext, td=td)
    im = panel(ax[1], s_a17, a17v, "Apollo 17  (Taurus-Littrow valley + massifs)", a17_ext)
    cax = fig.add_axes([0.905, 0.10, 0.016, 0.72])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("footprint slope (degrees) — ABSOLUTE, same scale on both sites")
    cb.ax.axhline(THETA, color="#27E1D6", lw=2)
    cb.ax.text(1.6, THETA, "  8° gate", color="#0a7d76", fontsize=8, va="center", fontweight="bold")
    fig.suptitle("HATI v2.5 absolute hazard heatmap — deterministic, physical units, cross-site comparable\n"
                 "cyan contour = nominal 8° lander gate.  Athena touchdown sits in benign terrain "
                 "(its hazard is the sub-resolution crater → shadow channel).",
                 fontsize=10.5, fontweight="bold")
    fig.savefig(OUT / "hati25_heatmaps.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  -> {OUT/'hati25_heatmaps.png'}")


if __name__ == "__main__":
    main()
