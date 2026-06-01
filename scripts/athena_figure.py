"""Combined figure for the Athena (IM-2) counterfactual: both HATI pipelines
on the actual touchdown point, using pre-landing NOBILE03 data.

Reads the cached outputs from athena_counterfactual.py:
    output/athena/athena_dem.npz      (heatmap, slope, touchdown px)
    output/athena/athena_shadow.npz   (ortho crop, shadow mask, density, ...)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Patch
import rasterio

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "athena"
DTM = ROOT / "data" / "athena" / "NAC_DTM_NOBILE03.TIF"

# Headline numbers from the representative DEM config + shadow census.
DEM_H = 0.726
DEM_PCT = 96.3
DEM_SLOPE = 2.8
SHADOW_SUBRES_KM2 = 1417
SHADOW_TOTAL_KM2 = 1514


def hillshade(dem, az=315.0, alt=30.0):
    az, alt = np.radians(az), np.radians(alt)
    gy, gx = np.gradient(dem.astype(np.float32))
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    sh = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip((sh + 1) / 2, 0, 1)


def main():
    dem_npz = np.load(OUT / "athena_dem.npz")
    H = dem_npz["heatmap"]
    nod = dem_npz["nodata"]
    tr, tc = int(dem_npz["td_row"]), int(dem_npz["td_col"])

    with rasterio.open(DTM) as src:
        dem = src.read(1).astype(np.float32)
    demf = dem.copy()
    demf[nod] = np.nanmedian(dem[~nod])
    hs = hillshade(demf)
    Hm = np.ma.masked_invalid(H)
    maskcol = (0.74, 0.80, 0.90)  # inert light slate = "no DEM / masked"
    mask_ov = np.zeros((*H.shape, 4))
    _bad = ~np.isfinite(H)
    mask_ov[_bad, 0], mask_ov[_bad, 1], mask_ov[_bad, 2] = maskcol
    mask_ov[_bad, 3] = 1.0

    sh = np.load(OUT / "athena_shadow.npz")
    crop = sh["crop"]
    mask = sh["mask"]
    density = sh["density"]
    ws = sh["widths_m"]
    sr, sc = int(sh["td_row"]), int(sh["td_col"])
    cov = crop > 0
    vis = crop[cov]
    vlo, vhi = np.percentile(vis, 2), np.percentile(vis, 98)

    fig = plt.figure(figsize=(16.5, 9.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.18, wspace=0.16)

    # --- Panel 1: DEM heatmap, full massif
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1)
    ax.imshow(mask_ov)
    im = ax.imshow(Hm, cmap="magma", vmin=0, vmax=1, alpha=0.6)
    ax.plot(tc, tr, "+", color="#00e5ff", ms=18, mew=2.6)
    ax.legend(handles=[Patch(facecolor=maskcol, edgecolor="0.4",
              linewidth=0.4, label="no DEM / masked")],
              loc="lower right", fontsize=7, framealpha=0.92)
    z = 110
    ax.add_patch(Rectangle((tc - z, tr - z), 2 * z, 2 * z, fill=False,
                           edgecolor="#00e5ff", lw=1.3))
    ax.set_title("Pipeline 1 - DEM hazard heatmap H(x)\n"
                 "NOBILE03 DTM, 4 m/px (pre-landing 2012)", fontsize=10.5,
                 fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("H(x)", fontsize=9)

    # --- Panel 2: DEM heatmap zoom on touchdown
    ax = fig.add_subplot(gs[0, 1])
    r0, r1 = tr - z, tr + z
    c0, c1 = tc - z, tc + z
    ax.imshow(hs[r0:r1, c0:c1], cmap="gray", vmin=0, vmax=1,
              extent=[c0, c1, r1, r0])
    ax.imshow(mask_ov[r0:r1, c0:c1], extent=[c0, c1, r1, r0])
    ax.imshow(Hm[r0:r1, c0:c1], cmap="magma", vmin=0, vmax=1, alpha=0.62,
              extent=[c0, c1, r1, r0])
    ax.plot(tc, tr, "+", color="#00e5ff", ms=22, mew=3)
    ax.set_title(f"Touchdown zoom (880 m): H={DEM_H:.2f}, "
                 f"{DEM_PCT:.0f}th pct\nslope only {DEM_SLOPE:.1f} deg "
                 "-> flag is fine-scale roughness", fontsize=10.5,
                 fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    # --- Panel 3: verdict text
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    ax.text(0.0, 1.0,
            "IM-2 'Athena' counterfactual\n"
            "84.7906 S, 29.1957 E - Mons Mouton\n"
            "(lost 2025-03-06, tipped at a ~20 m crater)\n\n"
            "Question: with ONLY pre-landing data,\n"
            "would HATI flag this exact point?\n\n"
            "PIPELINE 1 (DEM heatmap): FLAG\n"
            f"  H = {DEM_H:.2f}, {DEM_PCT:.0f}th percentile of the\n"
            "  Mons Mouton massif; robust across\n"
            "  scale configs (93-97th pct). Driven by\n"
            f"  fine-scale roughness, not slope ({DEM_SLOPE:.1f} deg).\n\n"
            "PIPELINE 2 (NAC shadow census):\n"
            f"  {SHADOW_SUBRES_KM2}/km2 obstacles BELOW the 4 m/px\n"
            "  DEM floor (94% of all detected); 3 within\n"
            "  the 25 m dispersion. Pervasive sub-\n"
            "  resolution hazard the DEM is blind to.\n\n"
            "VERDICT: >=1 pipeline flags -> SUCCESS.\n"
            "One pinpoints the site; both confirm it\n"
            "is hazardous at AND below planning res.",
            fontsize=9.6, va="top", ha="left", family="monospace",
            linespacing=1.32)

    # --- Panel 4: ortho crop
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(crop, cmap="gray", vmin=vlo, vmax=vhi)
    ax.plot(sc, sr, "+", color="#ff3838", ms=18, mew=2.6)
    ax.set_title("Pipeline 2 - NAC ortho, 0.9 m/px\n"
                 "(co-registered, pre-landing 2012)", fontsize=10.5,
                 fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    # --- Panel 5: ortho + shadow overlay + dispersion ring
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(crop, cmap="gray", vmin=vlo, vmax=vhi)
    ov = np.zeros((*crop.shape, 4))
    ov[mask] = [1, 0.25, 0.25, 0.85]
    ax.imshow(ov)
    ax.plot(sc, sr, "+", color="#00e5ff", ms=18, mew=2.6)
    ax.add_patch(Circle((sc, sr), 25 / 0.9, fill=False, edgecolor="#00e5ff",
                        lw=1.6, ls="--"))
    ax.set_title("Cast shadows (red) + 25 m dispersion ring\n"
                 f"{SHADOW_TOTAL_KM2}/km2 shadows, 94% sub-8m footprint",
                 fontsize=10.5, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    # --- Panel 6: footprint SFD
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(ws[ws < 30], bins=40, color="#444", edgecolor="none")
    ax.axvline(8.0, color="#ff3838", lw=2)
    ax.text(8.3, ax.get_ylim()[1] * 0.9, "4 m/px DEM\nfloor (8 m)",
            color="#ff3838", fontsize=9, va="top")
    ax.set_xlabel("cross-sun footprint (m)", fontsize=9)
    ax.set_ylabel("shadow count", fontsize=9)
    ax.set_title("Obstacle footprint SFD\n94% below the DEM resolving floor",
                 fontsize=10.5, fontweight="bold")
    ax.tick_params(labelsize=8)

    fig.suptitle("HATI v2.0 - south-polar forensic check: both pipelines on the "
                 "actual IM-2 Athena touchdown point (pre-landing data only)",
                 fontsize=13, fontweight="bold", y=0.985)
    fig.text(0.5, 0.005,
             "Data: LROC NAC DTM NOBILE03 (ASU / M. Robinson), source frames "
             "M1101075756/097181/118606, 2012-08-31. "
             "Analysis: G. Csaba Morvai, HATI v2.0.",
             ha="center", fontsize=8, style="italic", color="#555")

    out = OUT / "athena_counterfactual.png"
    fig.savefig(out, dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
