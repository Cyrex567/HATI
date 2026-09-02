"""Multi-frame consensus shadow census on the real Athena touchdown.

We now hold all three NOBILE03 orthophotos co-registered on the DTM grid, plus the
measured solar elevation of each frame (from the ODE solar-sweep query). They share
one illumination (azimuth 173-179 deg), so this is NOT the shadow-kinematics map --
it cannot sense motion or reject albedo. What it CAN do, and what the single-frame
v2.0 census could not:
  * denoise -- keep only shadows seen in >=2 of 3 independent acquisitions, rejecting
    per-frame detector noise / cosmic-ray hits / transient artefacts;
  * calibrate height -- convert each consensus shadow length to an obstacle height
    h = L*tan(e) with the MEASURED elevation, not an assumed one;
  * quantify -- obstacle areal density and a Poisson strike probability at the touchdown.
The kinematics gains (sign, albedo rejection, slope-corrected height) wait for an
azimuth-diverse sweep (ISIS ingestion).
"""
from __future__ import annotations
import sys
import glob
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import athena_counterfactual as ac  # constants + detect_shadows_local + ortho_pixel

OUT = ROOT / "output" / "athena"
SWEEP_CSV = sorted(OUT.glob("solar_sweep_*EDRNAC4.csv"))
H_CRIT = 0.3      # m, lander belly clearance threshold
FOOTPRINTS_M = [1.0, 3.5, 6.0]   # candidate lander footprint diameters


def load_ortho_path(path: Path) -> np.ndarray:
    n = ac.ORTHO_LINES * ac.ORTHO_SAMPLES
    off = path.stat().st_size - n * 2
    return np.fromfile(path, dtype=ac.ORTHO_DTYPE, count=n,
                       offset=off).reshape(ac.ORTHO_LINES, ac.ORTHO_SAMPLES)


def elev_lookup() -> dict:
    d = {}
    if SWEEP_CSV:
        for ln in SWEEP_CSV[0].read_text().splitlines()[1:]:
            f = ln.split(",")
            if len(f) > 3 and f[0].startswith("nac."):
                m = f[0].replace("nac.", "").rstrip("lre").upper()  # m1101075756
                try:
                    d.setdefault("M" + m[1:], float(f[3]))
                except ValueError:
                    pass
    return d


def main():
    paths = sorted(Path(ac.DATA).glob("NAC_DTM_NOBILE03_M*_90CM.IMG"))
    if len(paths) < 2:
        print(f"need >=2 orthos, found {len(paths)}"); sys.exit(1)
    elevs = elev_lookup()
    r, c = ac.ortho_pixel()
    h = ac.SHADOW_REGION // 2
    r0, c0 = r - h, c - h
    td_local = (r - r0, c - c0)

    masks, crops, used_elev = [], [], []
    print(f"consensus shadow census at touchdown ({ac.SHADOW_REGION*ac.ORTHO_SCALE:.0f} m box, {ac.ORTHO_SCALE} m/px)")
    for p in paths:
        mnum = p.name.split("_")[3]               # M1101075756
        e = elevs.get(mnum, 5.4)
        crop = load_ortho_path(p)[r0:r + h, c0:c + h].astype(np.float32)
        mask = ac.detect_shadows_local(crop)[0]
        masks.append(mask); crops.append(crop); used_elev.append(e)
        print(f"  {mnum}  sun elev {e:.2f} deg  shadow px {int(mask.sum())}  ({100*mask.mean():.2f}% of box)")

    stack = np.sum(masks, axis=0)
    consensus = stack >= 2                          # seen in >=2 of N frames
    union = stack >= 1
    mean_e = float(np.mean(used_elev))
    tan_e = math.tan(math.radians(mean_e))

    # single-frame (frame 0) vs consensus obstacle counts
    def census(mask, e_tan):
        rows = []
        for pr in regionprops(label(mask)):
            if pr.area < ac.SHADOW_MIN_AREA:
                continue
            L = pr.axis_major_length * ac.ORTHO_SCALE
            rows.append((L, L * e_tan, pr.centroid))    # length, height, (row,col)
        return rows
    sf = census(masks[0], math.tan(math.radians(used_elev[0])))
    cons = census(consensus, tan_e)
    heights = np.array([h for _, h, _ in cons])

    cov = np.mean([cr > 0 for cr in crops], axis=0) > 0.5
    area_km2 = float(cov.sum()) * (ac.ORTHO_SCALE ** 2) / 1e6
    n_tall = int((heights >= H_CRIT).sum())
    lam = n_tall / max(area_km2 * 1e6, 1.0)         # obstacles per m^2 above clearance

    # touchdown-local density
    cy = np.array([cc[0] for _, _, cc in cons]); cx = np.array([cc[1] for _, _, cc in cons])
    dlocal = np.hypot(cy - td_local[0], cx - td_local[1]) * ac.ORTHO_SCALE
    near = int((dlocal <= ac.DENSITY_RAD_PX * ac.ORTHO_SCALE).sum())

    print(f"\nsingle-frame obstacles: {len(sf)}   ->   consensus (>=2/{len(paths)}): {len(cons)}  "
          f"(denoise: {100*(1-len(cons)/max(len(sf),1)):.0f}% of single-frame detections rejected as non-repeating)")
    if heights.size:
        print(f"consensus obstacle heights (h = L*tan {mean_e:.1f} deg): median {np.median(heights):.2f}  "
              f"p90 {np.percentile(heights,90):.2f}  max {heights.max():.2f} m  | >= {H_CRIT} m clearance: {n_tall}")
    print(f"obstacle density >= {H_CRIT} m: {lam*1e6:.0f} per km^2  over {area_km2:.3f} km^2 observed")
    print(f"touchdown-local: {near} obstacles within {ac.DENSITY_RAD_PX*ac.ORTHO_SCALE:.0f} m")
    print("Poisson strike probability P = 1 - exp(-lambda * A):")
    for d in FOOTPRINTS_M:
        A = math.pi * (d / 2) ** 2
        print(f"   footprint d={d:.1f} m (A={A:.1f} m^2):  P = {1 - math.exp(-lam * A):.3f}")

    # figure
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.6))
    vmax = float(np.percentile(crops[0][crops[0] > 0], 99))
    for i in range(3):
        a = ax[0, i]; a.imshow(crops[i], cmap="gray", vmin=0, vmax=vmax)
        ov = np.zeros((*crops[i].shape, 4)); ov[masks[i]] = (1, 0.25, 0.1, 0.6); a.imshow(ov)
        a.set_title(f"{paths[i].name.split('_')[3]}  (elev {used_elev[i]:.2f}°)  shadows", fontsize=9)
        a.plot(td_local[1], td_local[0], "x", color="#27E1D6", ms=11, mew=2.2); a.set_xticks([]); a.set_yticks([])
    meanimg = np.mean(crops, axis=0)
    ax[1, 0].imshow(meanimg, cmap="gray", vmin=0, vmax=vmax)
    ovc = np.zeros((*consensus.shape, 4)); ovc[consensus] = (0.1, 0.9, 0.5, 0.8); ax[1, 0].imshow(ovc)
    ax[1, 0].plot(td_local[1], td_local[0], "x", color="#B23A2E", ms=12, mew=2.5)
    ax[1, 0].set_title(f"consensus shadows (≥2/{len(paths)} frames)\n{len(cons)} confident obstacles", fontsize=9)
    ax[1, 0].set_xticks([]); ax[1, 0].set_yticks([])
    sd = np.zeros((*stack.shape, 3))                # vote-count map
    sd[..., 1] = stack / max(stack.max(), 1)
    ax[1, 1].imshow(sd); ax[1, 1].set_title("shadow agreement (brighter = more frames agree)", fontsize=9)
    ax[1, 1].set_xticks([]); ax[1, 1].set_yticks([])
    if heights.size:
        ax[1, 2].hist(heights, bins=30, color="#1B7A6E", alpha=0.8)
        ax[1, 2].axvline(H_CRIT, color="#B23A2E", ls="--", lw=1.5, label=f"{H_CRIT} m clearance")
        ax[1, 2].set_xlabel("obstacle height (m)  from L·tan(e)"); ax[1, 2].set_ylabel("count")
        ax[1, 2].set_title(f"consensus obstacle heights\nmedian {np.median(heights):.2f} m, "
                           f"{n_tall} ≥ clearance", fontsize=9); ax[1, 2].legend(fontsize=8)
    fig.suptitle("HATI shadow channel, refreshed — 3-frame consensus census on the real Athena touchdown "
                 "(one illumination; denoised + height-calibrated)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "shadow_consensus.png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  -> {OUT/'shadow_consensus.png'}")


if __name__ == "__main__":
    main()
