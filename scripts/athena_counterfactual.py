"""HATI v2.0 -- Athena (IM-2) landing-site counterfactual.

Question: would HATI's two pipelines have flagged the actual IM-2 touchdown
point as hazardous, using ONLY pre-landing data?

Touchdown:  84.7906 deg S, 29.1957 deg E  (Mons Mouton, lunar south pole)
Data:       LROC NAC DTM NOBILE03 (ASU / Robinson), source frames 2012-08-31
            - DTM   NAC_DTM_NOBILE03.TIF          4.0 m/px  (stereo elevation)
            - ortho NAC_DTM_NOBILE03_M1101075756_90CM.IMG  0.9 m/px (co-reg.)
            Both products predate the 2025-03-06 landing -> no circularity.

Two pipelines, success if EITHER flags the touchdown:
  Pipeline 1 (DEM heatmap): Tier-1 roughness stack -> fused hazard H(x) in
    [0,1]. Read H and its scene percentile at the touchdown pixel.
  Pipeline 2 (NAC shadow):  grazing-sun shadow detection on the 0.9 m/px
    ortho; a shadow at the touchdown = a sub-DEM-resolution obstacle the
    4 m/px DTM cannot resolve.

The DTM is 4 m/px vs the heatmap's 1.5 m/px Apollo tuning. We run the
*identical* validated pipeline + weights (no re-tuning); the physical scales
auto-shift coarser (MDS L=3..20 px -> 12..80 m). A window=7 sensitivity run
checks the touchdown percentile is not an artifact of window choice.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi
from skimage.measure import label, regionprops

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.heatmap import dem_features, fusion  # noqa: E402

DATA = ROOT / "data" / "athena"
OUT = ROOT / "output" / "athena"
OUT.mkdir(parents=True, exist_ok=True)

DTM_PATH = DATA / "NAC_DTM_NOBILE03.TIF"
ORTHO_IMG = DATA / "NAC_DTM_NOBILE03_M1101075756_90CM.IMG"
ORTHO_XML = DATA / "NAC_DTM_NOBILE03_M1101075756_90CM.xml"

# Athena touchdown (planetocentric; Moon is a sphere so == planetographic)
TD_LAT = -84.7906
TD_LON = 29.1957
R_MOON = 1737400.0

# Ortho georef (from PDS4 label; |x| matches the DTM corner -- the label's
# negative upperleft_corner_x is a polar-stereo axis-sign convention, so we
# use the magnitude to stay in the DTM's positive-x frame, verified to
# reproduce the DTM touchdown pixel exactly).
ORTHO_ULX = 75559.35
ORTHO_ULY = 142716.65
ORTHO_SCALE = 0.9
ORTHO_LINES = 6547
ORTHO_SAMPLES = 5209
ORTHO_DTYPE = "<u2"

NODATA_BELOW = -1e30


# ---------------------------------------------------------------------------
# Georeferencing
# ---------------------------------------------------------------------------


def touchdown_xy() -> tuple[float, float]:
    """Closed-form spherical south-polar-stereographic (lat<0), central
    meridian 0, positive-east. Returns projected (x, y) in metres."""
    phi = np.radians(TD_LAT)
    lam = np.radians(TD_LON)
    rho = 2.0 * R_MOON * np.tan(np.pi / 4.0 + phi / 2.0)
    return rho * np.sin(lam), rho * np.cos(lam)


def dtm_pixel() -> tuple[int, int]:
    x, y = touchdown_xy()
    with rasterio.open(DTM_PATH) as src:
        row, col = src.index(x, y)
    return int(row), int(col)


def ortho_pixel() -> tuple[int, int]:
    x, y = touchdown_xy()
    col = (x - ORTHO_ULX) / ORTHO_SCALE
    row = (ORTHO_ULY - y) / ORTHO_SCALE
    return int(round(row)), int(round(col))


# ---------------------------------------------------------------------------
# DEM pipeline
# ---------------------------------------------------------------------------


def load_dtm() -> tuple[np.ndarray, np.ndarray, float]:
    with rasterio.open(DTM_PATH) as src:
        dem = src.read(1).astype(np.float32)
        scale = float(src.transform.a)
    nod = dem <= NODATA_BELOW
    return dem, nod, scale


def fill_nearest(dem: np.ndarray, nod: np.ndarray) -> np.ndarray:
    """Replace nodata with the nearest valid value (no artificial cliffs,
    so sliding-window slope/curvature stay finite and unbiased)."""
    if not nod.any():
        return dem
    idx = ndi.distance_transform_edt(nod, return_distances=False,
                                     return_indices=True)
    return dem[tuple(idx)]


@dataclass
class DemResult:
    heatmap: np.ndarray          # NaN at nodata
    slope_deg: np.ndarray        # NaN at nodata
    nodata: np.ndarray
    td_rc: tuple[int, int]
    window_px: int
    mds_scales_px: tuple[int, ...]


def run_dem_pipeline(window_px: int, mds_scales_px: tuple[int, ...],
                     weights: dict | None) -> DemResult:
    dem, nod, scale = load_dtm()
    print(f"  DTM {dem.shape}  scale={scale} m/px  nodata={100*nod.mean():.1f}%")
    dem_f = fill_nearest(dem, nod)

    print(f"  Tier-1 stack  window={window_px}px  mds={mds_scales_px}px"
          f"  (= {tuple(round(s*scale) for s in mds_scales_px)} m)")
    channels = dem_features.compute_tier1_stack(
        dem_f, scale_m=scale, window_px=window_px,
        mds_scales_px=mds_scales_px, verbose=False)

    # Mask channels near nodata so z-score stats use only valid terrain and
    # the output is undefined on filled regions. The dominant filter reach is
    # the MDS Gaussian support: scipy truncates at 4 sigma, sigma_2L = 2L/3,
    # so support ~= 8*maxL/3, plus the median/uniform window half-width.
    reach = int(np.ceil(8.0 * max(mds_scales_px) / 3.0)) + window_px // 2 + 1
    nod_dil = ndi.binary_dilation(nod, iterations=reach)
    for k in channels:
        channels[k] = np.where(nod_dil, np.nan, channels[k]).astype(np.float32)

    result = fusion.fuse(channels, weights=weights)
    heat = np.where(nod_dil, np.nan, result.heatmap).astype(np.float32)

    slope = np.degrees(dem_features._slope_magnitude(dem_f, scale))
    slope = np.where(nod_dil, np.nan, slope).astype(np.float32)

    return DemResult(heat, slope, nod, dtm_pixel(), window_px, mds_scales_px)


def evaluate_dem(res: DemResult, label: str) -> dict:
    r, c = res.td_rc
    H = res.heatmap
    valid = np.isfinite(H)
    Htd = float(H[r, c])
    p50, p75, p90, p95 = np.nanpercentile(H, [50, 75, 90, 95])
    s = res.slope_deg
    Std = float(s[r, c])
    rr = slice(max(0, r - 5), r + 6)
    cc = slice(max(0, c - 5), c + 6)
    s_local = s[rr, cc][np.isfinite(s[rr, cc])]
    H_local = H[rr, cc][np.isfinite(H[rr, cc])]

    print(f"\n  [{label}]  touchdown pixel (row={r}, col={c})")
    if not np.isfinite(Htd):
        print("    H(touchdown)      = NaN -- uncomputable: this config's "
              "coarsest scale reaches the DTM coverage edge (116 m away)")
        return {"label": label, "Htd": np.nan, "pct": np.nan,
                "p50": float(p50), "p75": float(p75), "p90": float(p90),
                "p95": float(p95), "slope_td": Std,
                "slope_local_max": float(s_local.max()) if s_local.size else np.nan,
                "H_local_max": float(H_local.max()) if H_local.size else np.nan,
                "flag_p75": False, "flag_p90": False}

    pct = float((H[valid] < Htd).mean() * 100.0)
    print(f"    H(touchdown)      = {Htd:.3f}")
    print(f"    scene percentile  = {pct:.1f}%   (higher = rougher/worse)")
    print(f"    scene H p50/p75/p90/p95 = "
          f"{p50:.3f} / {p75:.3f} / {p90:.3f} / {p95:.3f}")
    print(f"    slope(touchdown)  = {Std:.1f} deg")
    print(f"    local 44 m box: slope max={s_local.max():.1f} deg "
          f"mean={s_local.mean():.1f} deg ; H max={H_local.max():.3f}")
    flag_p75 = Htd >= p75
    flag_p90 = Htd >= p90
    print(f"    FLAG vs scene p75 (top 25% roughest): {flag_p75}")
    print(f"    FLAG vs scene p90 (top 10% roughest): {flag_p90}")
    return {
        "label": label, "Htd": Htd, "pct": pct,
        "p50": float(p50), "p75": float(p75), "p90": float(p90),
        "p95": float(p95), "slope_td": Std,
        "slope_local_max": float(s_local.max()),
        "H_local_max": float(H_local.max()),
        "flag_p75": bool(flag_p75), "flag_p90": bool(flag_p90),
    }


def build_weights(mds_scales_px: tuple[int, ...]) -> dict:
    """Literature-anchored weights for an arbitrary MDS scale set. The rule is
    the SAME as fusion.DEFAULT_WEIGHTS: base channels fixed, MDS weights
    decreasing with scale (smaller scale, nearer Nyquist -> higher weight).
    Reproduces DEFAULT_WEIGHTS exactly for mds=(3,5,10,20). Deterministic, not
    tuned to the touchdown."""
    w = {"rms_slope": 1.0, "iqr_slope": 1.0, "iqr_curvature": 1.0,
         "rms_planar_dev": 0.7, "tpi_abs": 0.7, "tri": 0.7}
    ladder = [1.0, 0.9, 0.8, 0.6]
    for i, L in enumerate(sorted(mds_scales_px)):
        w[f"mds_L{L}"] = ladder[min(i, len(ladder) - 1)]
    return w


# Configs swept to show the touchdown flag is robust, not config-fished.
# All but the first are edge-computable (filter reach < 29 px to the nodata
# margin). The first is the *exact* validated Apollo config; its 80 m coarse
# scale cannot be evaluated 116 m from the DTM edge -> documented as NaN.
SWEEP = [
    ("Apollo default  w=11  mds 12-80m", 11, (3, 5, 10, 20)),
    ("fine            w=5   mds 8-32m",   5, (2, 3, 5, 8)),
    ("fine            w=7   mds 8-32m",   7, (2, 3, 5, 8)),
    ("fine            w=9   mds 8-24m",   9, (2, 3, 4, 6)),
    ("finest          w=7   mds 8-20m",   7, (2, 3, 4, 5)),
]
# The representative config whose heatmap is saved + visualised.
REPRESENTATIVE = "fine            w=7   mds 8-32m"


def main_dem():
    print("=" * 64)
    print("PIPELINE 1 -- DEM heatmap on NOBILE03 (4 m/px)")
    print("=" * 64)
    t0 = time.perf_counter()
    rows = []
    saved = None
    for label, win, mds in SWEEP:
        res = run_dem_pipeline(win, mds, weights=build_weights(mds))
        ev = evaluate_dem(res, label)
        rows.append(ev)
        if label == REPRESENTATIVE:
            saved = res

    print("\n" + "=" * 64)
    print("ROBUSTNESS SWEEP -- touchdown vs scene (NaN = uncomputable at edge)")
    print("=" * 64)
    print(f"  {'config':<34} {'H_td':>6} {'pct':>6}  flag>p75 flag>p90")
    for ev in rows:
        h = "nan" if np.isnan(ev["Htd"]) else f"{ev['Htd']:.3f}"
        p = "nan" if np.isnan(ev["pct"]) else f"{ev['pct']:.1f}"
        print(f"  {ev['label']:<34} {h:>6} {p:>6}  "
              f"{str(ev['flag_p75']):>7} {str(ev['flag_p90']):>7}")
    print(f"\n  DEM pipeline elapsed {time.perf_counter() - t0:.1f} s")

    np.savez_compressed(
        OUT / "athena_dem.npz",
        heatmap=saved.heatmap, slope_deg=saved.slope_deg,
        nodata=saved.nodata, td_row=saved.td_rc[0], td_col=saved.td_rc[1],
    )
    print(f"  -> {OUT / 'athena_dem.npz'}  (config: {REPRESENTATIVE})")
    return rows


# ---------------------------------------------------------------------------
# Shadow pipeline (Pipeline 2) -- on the 0.9 m/px ortho
# ---------------------------------------------------------------------------

SHADOW_FRAC = 0.5        # shadow if DN < 0.5 * local sunlit background
BG_WIN_PX = 45           # 40 m median window for the local background
SHADOW_MIN_AREA = 4      # px
DEM_FLOOR_M = 8.0        # 2-px Nyquist of the 4 m/px DTM -> sub-resolution
SHADOW_REGION = 1200     # px analysis box (1080 m) centred on touchdown
DENSITY_RAD_PX = 28      # 25 m radius for local obstacle-density map


def load_ortho() -> np.ndarray:
    n = ORTHO_LINES * ORTHO_SAMPLES
    off = ORTHO_IMG.stat().st_size - n * 2  # data is the trailing block
    return np.fromfile(ORTHO_IMG, dtype=ORTHO_DTYPE, count=n,
                       offset=off).reshape(ORTHO_LINES, ORTHO_SAMPLES)


@dataclass
class ShadowResult:
    crop: np.ndarray
    mask: np.ndarray
    widths_m: np.ndarray         # cross-sun footprint per shadow
    lengths_m: np.ndarray        # along-sun shadow length per shadow
    cx: np.ndarray               # centroid col within crop
    cy: np.ndarray               # centroid row within crop
    density: np.ndarray          # sub-floor obstacles within DENSITY_RAD
    td_local: tuple[int, int]    # touchdown (row,col) within crop
    modal_orient_deg: float


def detect_shadows_local(crop: np.ndarray):
    cov = crop > 0
    bg_in = np.where(cov, crop, np.median(crop[cov])).astype(np.float32)
    bg = ndi.median_filter(bg_in, size=BG_WIN_PX)
    mask = cov & (crop < SHADOW_FRAC * bg)
    mask = ndi.binary_opening(mask, iterations=1)
    mask = ndi.binary_closing(mask, iterations=1)
    lab = label(mask)
    ws, ls, cx, cy, ang = [], [], [], [], []
    for p in regionprops(lab):
        if p.area < SHADOW_MIN_AREA:
            continue
        w = p.axis_minor_length * ORTHO_SCALE
        L = p.axis_major_length * ORTHO_SCALE
        if w < ORTHO_SCALE:
            w = p.equivalent_diameter_area * ORTHO_SCALE
        ws.append(w); ls.append(L)
        cy.append(p.centroid[0]); cx.append(p.centroid[1])
        if p.axis_major_length >= 4 and \
           p.axis_major_length > 1.8 * max(p.axis_minor_length, 1e-6):
            ang.append(p.orientation)
    ang = np.array(ang)
    modal = float(np.degrees(0.5 * np.arctan2(
        np.mean(np.sin(2 * ang)), np.mean(np.cos(2 * ang))))) if ang.size else float("nan")
    return (mask, np.array(ws), np.array(ls),
            np.array(cx), np.array(cy), modal)


def run_shadow_pipeline() -> ShadowResult:
    print("=" * 64)
    print("PIPELINE 2 -- NAC shadow census on ortho (0.9 m/px)")
    print("=" * 64)
    img = load_ortho()
    r, c = ortho_pixel()
    h = SHADOW_REGION // 2
    r0, c0 = r - h, c - h
    crop = img[r0:r + h, c0:c + h].astype(np.float32)
    td_local = (r - r0, c - c0)
    cov_frac = float((crop > 0).mean())
    print(f"  region {crop.shape} ({SHADOW_REGION*ORTHO_SCALE:.0f} m)  "
          f"coverage {100*cov_frac:.1f}%")

    mask, ws, ls, cx, cy, modal = detect_shadows_local(crop)
    area_km2 = (crop > 0).sum() * (ORTHO_SCALE ** 2) / 1e6
    sub = ws < DEM_FLOOR_M
    print(f"  shadows (>= {SHADOW_MIN_AREA}px): {len(ws)}  over {area_km2:.3f} km^2"
          f"  -> {len(ws)/area_km2:.0f}/km^2")
    print(f"  footprint < {DEM_FLOOR_M:.0f} m (sub-DEM): {int(sub.sum())} "
          f"({100*sub.mean():.0f}%)  -> {int(sub.sum())/area_km2:.0f}/km^2")
    print(f"  modal shadow orientation: {modal:.1f} deg "
          f"(unimodal => illumination-driven, not random albedo)")

    # Local sub-resolution obstacle density: count sub-floor centroids within
    # DENSITY_RAD of each pixel (disk convolution).
    pts = np.zeros(crop.shape, np.float32)
    for yy, xx in zip(cy[sub].astype(int), cx[sub].astype(int)):
        pts[yy, xx] += 1
    yy, xx = np.ogrid[-DENSITY_RAD_PX:DENSITY_RAD_PX + 1,
                      -DENSITY_RAD_PX:DENSITY_RAD_PX + 1]
    disk = (yy * yy + xx * xx <= DENSITY_RAD_PX ** 2).astype(np.float32)
    density = ndi.convolve(pts, disk, mode="constant")

    # Rank touchdown density only where the full disk sits in coverage.
    covf = ndi.binary_erosion(crop > 0, iterations=DENSITY_RAD_PX)
    td_dens = float(density[td_local])
    dvalid = density[covf]
    pct = float((dvalid < td_dens).mean() * 100.0)
    print(f"\n  touchdown local density: {td_dens:.0f} sub-res obstacles "
          f"within {DENSITY_RAD_PX*ORTHO_SCALE:.0f} m")
    print(f"  density percentile in region: {pct:.1f}%  "
          f"(p50={np.percentile(dvalid,50):.0f}, p90={np.percentile(dvalid,90):.0f})")

    # Census within landing-dispersion radii of the exact touchdown point.
    d = np.hypot(cy - td_local[0], cx - td_local[1]) * ORTHO_SCALE
    print("  obstacle census around exact touchdown point:")
    for rad in (5, 12, 25, 50):
        within = d < rad
        print(f"    within {rad:2d} m: {int(within.sum()):2d} shadows, "
              f"{int((within & sub).sum()):2d} sub-{DEM_FLOOR_M:.0f}m")
    if d.size:
        i = d.argmin()
        print(f"  nearest shadow: {d[i]:.1f} m away, footprint {ws[i]:.1f} m")

    flag = td_dens >= np.percentile(dvalid, 75)
    print(f"\n  PIPELINE 2 FLAG (density >= region p75): {flag}")

    np.savez_compressed(
        OUT / "athena_shadow.npz", crop=crop, mask=mask, density=density,
        widths_m=ws, lengths_m=ls, cx=cx, cy=cy,
        td_row=td_local[0], td_col=td_local[1], modal=modal,
        td_density=td_dens, density_pct=pct,
    )
    print(f"  -> {OUT / 'athena_shadow.npz'}")
    return ShadowResult(crop, mask, ws, ls, cx, cy, density, td_local, modal)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["dem", "shadow", "both"],
                    default="both")
    args = ap.parse_args()
    if args.stage in ("dem", "both"):
        main_dem()
    if args.stage in ("shadow", "both"):
        run_shadow_pipeline()
