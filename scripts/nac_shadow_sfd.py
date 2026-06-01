"""HATI v2.0 NAC pipeline -- Stage B: calibrated shadow size-frequency
distribution of the sub-resolution hazard population.

Goal
----
Stage A showed that NAC shadows flag obstacles the 1.5 m/px DEM cannot
resolve. Stage B turns that qualitative observation into a *quantified*
hazard population:

  1. Detect cast shadows (dark connected components) in a native-resolution
     NAC crop.  We work in raw EDR pixel space on purpose: map-projection
     resampling would smear the sub-pixel shadows we are trying to count.
  2. Fit each shadow with an ellipse (skimage regionprops) to separate two
     physically distinct measurements:
       * cross-sun width  (minor axis)  -> object FOOTPRINT.  Independent of
         sun elevation.  Saturates near the 1-pixel floor for sub-pixel
         objects (you cannot resolve a width below a pixel).
       * along-sun length (major axis)  -> object RELIEF via h = L*tan(e).
         This is what reaches into the sub-pixel regime: a low boulder still
         throws a long, detectable shadow under a low sun.
  3. Recover the solar azimuth from the data itself (the modal orientation of
     elongated shadows points anti-solar) as an internal validity check that
     these are illumination-driven shadows, not random dark patches.
  4. Build the size-frequency distribution and count the hazard population in
     the 0.5-3 m relief band -- the gap between the DEM's ~3 m resolving floor
     (2x the 1.5 m posting, Nyquist) and the lander-killing scale.

The sun elevation e is the ONLY quantity we cannot read from the EDR label
(geometry is added downstream at CDR/RDR via SPICE).  Because relief scales
as tan(e), we present the relief SFD across a plausible band e in
{25, 40, 55} deg and show the sub-DEM-floor conclusion is robust to it.
Detection and footprint widths do not depend on e at all.

Outputs (output/nac/):
  M1412631647RE_stageB_imagery.png      NAC crop + shadows coloured by relief
  M1412631647RE_stageB_statistics.png   width SFD | relief SFD | orientation
and a console summary with hazard densities per km^2.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "nac"
OUT = ROOT / "output" / "nac"
OUT.mkdir(parents=True, exist_ok=True)

FRAME = "M1412631647RE"
IMG = DATA / f"{FRAME}.IMG"
LINES = 52224
SAMPLES = 5064
HEADER_OFFSET = 5064
PIXEL_SCALE_M = 0.808            # native NAC scale adopted for this frame

# Analysis crop: a large valley-floor patch (Stage A located many boulders
# near line 26000).  Big enough for a statistically meaningful SFD, clear of
# the massif slopes that would conflate boulders with topographic shadow.
CROP_LINE0 = 25400
CROP_SAMPLE0 = 1800
CROP_SIZE = 1600

SHADOW_DN = 28                   # primary shadow threshold (Stage A bimodal split)
DN_SWEEP = (24, 28, 32)          # robustness sweep
MIN_AREA_PX = 3                  # ellipse fit unstable below this

DEM_FLOOR_M = 3.0                # 1.5 m/px DEM resolves features >~ 2 posts
HAZARD_BAND_M = (0.5, 3.0)       # lander-relevant sub-resolution band
SUN_ELEV_BAND_DEG = (25.0, 40.0, 55.0)
SUN_ELEV_CENTRAL_DEG = 40.0


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_nac() -> np.ndarray:
    with IMG.open("rb") as f:
        f.seek(HEADER_OFFSET)
        return np.frombuffer(f.read(LINES * SAMPLES), dtype=np.uint8).reshape(
            LINES, SAMPLES
        )


@dataclass
class Shadow:
    area_px: int
    width_px: float        # minor axis  -> cross-sun footprint
    length_px: float       # major axis  -> along-sun shadow length
    orient_deg: float      # major-axis angle, image frame, [-90, 90]
    cy: float
    cx: float

    @property
    def width_m(self) -> float:
        return self.width_px * PIXEL_SCALE_M

    @property
    def length_m(self) -> float:
        return self.length_px * PIXEL_SCALE_M

    def relief_m(self, elev_deg: float) -> float:
        # h = L_shadow * tan(elevation); shadow length measured along-sun.
        return self.length_m * np.tan(np.radians(elev_deg))


def detect_shadows(crop: np.ndarray, dn: int) -> list[Shadow]:
    """Threshold, clean, label, drop border-touching regions, fit ellipses."""
    mask = crop < dn
    mask = ndi.binary_closing(mask, iterations=1)
    mask = clear_border(mask)                    # truncated shadows bias sizes
    lab = label(mask)
    shadows: list[Shadow] = []
    for rp in regionprops(lab):
        if rp.area < MIN_AREA_PX:
            continue
        width = float(rp.axis_minor_length)
        length = float(rp.axis_major_length)
        if length <= 0:
            continue
        # degenerate minor axis (thin 1-px-wide shadow): floor at the
        # equivalent disc so a real but narrow footprint isn't logged as 0.
        if width < 1.0:
            width = float(rp.equivalent_diameter_area)
        cy, cx = rp.centroid
        shadows.append(Shadow(
            area_px=int(rp.area), width_px=width, length_px=length,
            orient_deg=float(np.degrees(rp.orientation)),
            cy=float(cy), cx=float(cx),
        ))
    return shadows


def modal_orientation(shadows: list[Shadow]) -> float:
    """Dominant shadow-elongation axis (deg).  Elongated shadows only; an
    illumination-driven population is unimodal and points anti-solar."""
    elong = [s.orient_deg for s in shadows
             if s.length_px >= 4 and s.length_px > 1.8 * s.width_px]
    if not elong:
        return float("nan")
    # circular-ish stat on a 180-deg axis: double the angle, average on circle
    ang = np.radians(np.array(elong) * 2.0)
    mean = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    return float(np.degrees(mean) / 2.0)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render_imagery(crop, shadows, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.4))
    p1, p99 = np.percentile(crop, [0.5, 99])
    for ax in axes:
        ax.imshow(crop, cmap="gray", vmin=p1, vmax=p99, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].set_title(
        f"NAC raw, native {PIXEL_SCALE_M:.2f} m/px "
        f"({crop.shape[1] * PIXEL_SCALE_M / 1000:.2f} x "
        f"{crop.shape[0] * PIXEL_SCALE_M / 1000:.2f} km)",
        fontsize=12, fontweight="bold")

    # colour each shadow by inferred relief at the central sun elevation
    relief = np.array([s.relief_m(SUN_ELEV_CENTRAL_DEG) for s in shadows])
    norm = Normalize(vmin=0.0, vmax=4.0)
    cmap = plt.get_cmap("turbo")
    for s, r in zip(shadows, relief):
        in_band = HAZARD_BAND_M[0] <= r < HAZARD_BAND_M[1]
        circ = plt.Circle(
            (s.cx, s.cy), radius=max(4.0, 0.6 * s.length_px),
            fill=False, edgecolor=cmap(norm(r)),
            linewidth=1.6 if in_band else 0.8,
            alpha=0.95 if in_band else 0.45)
        axes[1].add_patch(circ)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.02)
    cb.set_label(f"inferred relief (m) at sun elev {SUN_ELEV_CENTRAL_DEG:.0f} deg",
                 fontsize=10)
    n_band = int(((relief >= HAZARD_BAND_M[0]) & (relief < HAZARD_BAND_M[1])).sum())
    axes[1].set_title(
        f"Shadow detections coloured by relief\n"
        f"bold = {HAZARD_BAND_M[0]:.1f}-{HAZARD_BAND_M[1]:.0f} m hazard band "
        f"({n_band} obstacles the DEM is blind to)",
        fontsize=12, fontweight="bold")

    # scale bar
    bar_m = 100.0
    bar_px = bar_m / PIXEL_SCALE_M
    x0 = crop.shape[1] - bar_px - 40
    y0 = crop.shape[0] - 40
    for ax in axes:
        ax.plot([x0, x0 + bar_px], [y0, y0], color="white", lw=4,
                solid_capstyle="butt")
        ax.text(x0 + bar_px / 2, y0 - 16, f"{bar_m:.0f} m", color="white",
                ha="center", fontsize=10,
                bbox=dict(facecolor="black", alpha=0.6, pad=2, edgecolor="none"))

    fig.suptitle(
        f"{FRAME}  --  HATI v2.0 NAC Stage B: sub-resolution hazard population "
        "(Apollo 17 valley floor)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _cumulative(sizes_m: np.ndarray, area_km2: float):
    """Cumulative number density N(>=d) per km^2, sorted ascending."""
    s = np.sort(sizes_m)
    n_ge = (len(s) - np.arange(len(s))) / area_km2
    return s, n_ge


def render_statistics(shadows, area_km2, modal_az, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))

    widths = np.array([s.width_m for s in shadows])
    pixel_floor_m = PIXEL_SCALE_M

    # --- Panel 1: footprint-width SFD (elevation independent) ---
    ax = axes[0]
    s, n_ge = _cumulative(widths, area_km2)
    ax.loglog(s, n_ge, drawstyle="steps-post", color="#1f77b4", lw=2)
    ax.axvspan(s.min() if len(s) else 0.1, pixel_floor_m, color="0.85",
               label=f"sub-pixel width (<{pixel_floor_m:.2f} m)")
    ax.axvline(DEM_FLOOR_M, color="#d62728", ls="--", lw=1.8,
               label=f"DEM resolving floor ({DEM_FLOOR_M:.0f} m)")
    ax.set_xlabel("cross-sun footprint width (m)")
    ax.set_ylabel("cumulative count >= size  (per km$^2$)")
    ax.set_title("Footprint-width SFD\n(elevation-independent)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    # --- Panel 2: relief SFD across the sun-elevation band ---
    ax = axes[1]
    colors = ["#2ca02c", "#ff7f0e", "#9467bd"]
    for elev, c in zip(SUN_ELEV_BAND_DEG, colors):
        relief = np.array([sh.relief_m(elev) for sh in shadows])
        s, n_ge = _cumulative(relief, area_km2)
        ax.loglog(s, n_ge, drawstyle="steps-post", color=c, lw=2,
                  label=f"sun elev {elev:.0f} deg")
    ax.axvspan(*HAZARD_BAND_M, color="#ffe08a", alpha=0.5,
               label=f"hazard band {HAZARD_BAND_M[0]:.1f}-{HAZARD_BAND_M[1]:.0f} m")
    ax.axvline(DEM_FLOOR_M, color="#d62728", ls="--", lw=1.8)
    ax.set_xlabel("inferred relief / obstacle height (m)")
    ax.set_ylabel("cumulative count >= size  (per km$^2$)")
    ax.set_title("Relief SFD from shadow length\n(robust across sun-elevation band)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    # --- Panel 3: orientation rose (validity check) ---
    ax = axes[2]
    ax.remove()
    ax = fig.add_subplot(1, 3, 3, projection="polar")
    orients = np.array([s.orient_deg for s in shadows
                        if s.length_px >= 4 and s.length_px > 1.8 * s.width_px])
    if len(orients):
        # plot on a full circle by mirroring the 180-deg axis
        ang = np.radians(np.concatenate([orients, orients + 180.0]))
        ax.hist(ang % (2 * np.pi), bins=36, color="#444", alpha=0.8)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
    ax.set_title(
        f"Shadow-elongation rose\nmodal axis = {modal_az:+.0f} deg "
        "(unimodal = illumination-driven)",
        fontsize=11, fontweight="bold", pad=18)

    fig.suptitle(
        f"{FRAME}  --  shadow size-frequency distribution  "
        f"({len(shadows)} shadows over {area_km2:.2f} km$^2$)",
        fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def summarise(shadows, area_km2) -> None:
    widths = np.array([s.width_m for s in shadows])
    print(f"\n  detected shadows (area >= {MIN_AREA_PX} px, border-clean): {len(shadows)}")
    print(f"  crop area: {area_km2:.3f} km^2   ->  {len(shadows) / area_km2:.0f} shadows/km^2")
    print(f"\n  footprint width (cross-sun, elevation-independent):")
    for lo, hi, lbl in [(0, 1.5, "sub-pixel/marginal <1.5 m"),
                        (1.5, 3.0, "1.5-3 m"),
                        (3.0, 1e9, ">3 m (DEM-resolved)")]:
        n = int(((widths >= lo) & (widths < hi)).sum())
        print(f"     {lbl:28s}: {n:5d}  ({n / area_km2:7.0f} /km^2)")
    print(f"\n  relief / height (from shadow length, per sun elevation):")
    for elev in SUN_ELEV_BAND_DEG:
        relief = np.array([s.relief_m(elev) for s in shadows])
        n_haz = int(((relief >= HAZARD_BAND_M[0]) & (relief < HAZARD_BAND_M[1])).sum())
        n_sub = int((relief < HAZARD_BAND_M[0]).sum())
        print(f"     elev {elev:4.0f} deg:  "
              f"{HAZARD_BAND_M[0]:.1f}-{HAZARD_BAND_M[1]:.0f} m hazard band = "
              f"{n_haz:5d} ({n_haz / area_km2:7.0f}/km^2)   "
              f"<{HAZARD_BAND_M[0]:.1f} m = {n_sub} ({n_sub / area_km2:.0f}/km^2)")


def main() -> None:
    print(f"Loading {FRAME}")
    arr = load_nac()
    crop = arr[CROP_LINE0:CROP_LINE0 + CROP_SIZE,
               CROP_SAMPLE0:CROP_SAMPLE0 + CROP_SIZE].copy()
    print(f"crop {crop.shape} at line={CROP_LINE0} sample={CROP_SAMPLE0}  "
          f"(DN mean {crop.mean():.1f}, p1 {np.percentile(crop,1):.0f}, "
          f"shadow-frac<{SHADOW_DN} = {(crop < SHADOW_DN).mean()*100:.1f}%)")
    area_km2 = (CROP_SIZE * PIXEL_SCALE_M / 1000.0) ** 2

    # robustness sweep over threshold
    print("\nthreshold sweep (sensitivity check):")
    for dn in DN_SWEEP:
        sh = detect_shadows(crop, dn)
        w = np.array([s.width_m for s in sh])
        rel40 = np.array([s.relief_m(SUN_ELEV_CENTRAL_DEG) for s in sh])
        n_haz = int(((rel40 >= HAZARD_BAND_M[0]) & (rel40 < HAZARD_BAND_M[1])).sum())
        print(f"   DN<{dn}: {len(sh):5d} shadows   "
              f"hazard-band@{SUN_ELEV_CENTRAL_DEG:.0f}deg = {n_haz:5d} "
              f"({n_haz / area_km2:7.0f}/km^2)")

    shadows = detect_shadows(crop, SHADOW_DN)
    modal_az = modal_orientation(shadows)
    summarise(shadows, area_km2)
    print(f"\n  modal shadow-elongation axis: {modal_az:+.1f} deg (image frame)")

    img_path = OUT / f"{FRAME}_stageB_imagery.png"
    stat_path = OUT / f"{FRAME}_stageB_statistics.png"
    render_imagery(crop, shadows, img_path)
    render_statistics(shadows, area_km2, modal_az, stat_path)
    print(f"\n  -> {img_path}")
    print(f"  -> {stat_path}")


if __name__ == "__main__":
    main()
