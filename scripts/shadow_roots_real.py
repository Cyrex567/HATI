"""Run the experimental v2.5 likelihood search on an audited Athena sweep.

Reads projected products and geometry sidecars only. This program NEVER invokes
ISIS or downloads imagery. The generic core supports either pole; this adapter
retains the explicitly named Athena reference and site of the ingest pipeline.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.hati_core.shadow_likelihood import (
    ShadowConfig, gaussian_search_calibration, map_sun_azimuth, search_stack)
from sweep_contract import DEFAULT_BEFORE, PROCESSING_VERSION, predates


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/sweep/manifest.json")
    ap.add_argument("--output", type=Path, default=ROOT / "output/athena/shadow_roots_v25")
    ap.add_argument("--before", default=DEFAULT_BEFORE)
    ap.add_argument("--half", type=int, default=256, help="central pilot half-width; must fit audited window")
    ap.add_argument("--noise-sigma", type=float, default=0.03,
                    help="assumed standard deviation in median-normalized radiance, NOT a calibration")
    ap.add_argument("--psf-sigma-px", type=float, default=0.6)
    ap.add_argument("--registration-sigma-px", type=float, required=True,
                    help="local alignment sigma or documented sensitivity assumption, not closure median")
    ap.add_argument("--max-candidates", type=int, default=150)
    ap.add_argument("--null-trials", type=int, default=0, help="optional full-search Gaussian MODEL check; >=19")
    ap.add_argument("--seed", type=int, default=2718)
    ap.add_argument("--slope-row", type=float, default=0., help="local receiving-plane dz/drow-distance in m/m")
    ap.add_argument("--slope-col", type=float, default=0., help="local receiving-plane dz/dcolumn-distance in m/m")
    args = ap.parse_args()
    import rasterio
    import athena_counterfactual as ac
    from shadow_kinematics_real import frame_window, geometry_for
    man = json.loads(args.manifest.read_text())
    if len(man) < 3 or any(e.get("processing_version") != PROCESSING_VERSION or
                           not e.get("gate_pass") or not predates(e.get("utc", ""), args.before)
                           for e in man):
        sys.exit("requires >=3 audited, pre-cutoff, gate-passing manifest frames")
    if args.half < 16 or any(args.half > int(e.get("half_px", 0)) for e in man):
        sys.exit("pilot must fit inside the ingested registration window")
    # The fitted translation was estimated on each frame's site-centred pixel
    # window, which is the read convention retained here. It is not proof of
    # spatially uniform subpixel accuracy inside that window.
    with rasterio.open(ac.ORTHO_IMG) as reference:
        rr, cc = ac.ortho_pixel()
        transform = reference.window_transform(rasterio.windows.Window(
            cc-args.half, rr-args.half, 2*args.half, 2*args.half))
        crs = reference.crs
        sx = np.hypot(transform.a, transform.d)
        sy = np.hypot(transform.b, transform.e)
        if not np.isclose(sx, sy) or abs(transform.a*transform.b+transform.d*transform.e) > 1e-8:
            sys.exit("template geometry requires square orthogonal pixels")
    frames, azimuths, elevations, provenance = [], [], [], []
    for e in man:
        pid = e["pid"]
        g = geometry_for(pid, ac.TD_LAT, ac.TD_LON, rebuild=False)
        if g is None or e.get("shift_px") is None:
            sys.exit(f"missing measured geometry/alignment for {pid}; rebuild on ISIS host")
        lev2 = Path(e["lev2"])
        if not lev2.exists():
            lev2 = args.manifest.parent / lev2.name
        # Read the same full window used to estimate the shift, then crop. This
        # avoids introducing fresh interpolation boundaries in the pilot area.
        half = int(e["half_px"])
        frame = frame_window(lev2, e["shift_px"], half, ac)
        if frame is None:
            sys.exit(f"unreadable projected frame: {pid}")
        frame = frame[half-args.half:half+args.half, half-args.half:half+args.half]
        good = np.isfinite(frame) & (frame > 0)
        if good.mean() < 0.8:
            sys.exit(f"{pid}: less than 80% positive radiance in pilot; no silent frame dropping")
        scale = float(np.median(frame[good]))
        frames.append(np.where(good, frame / scale, np.nan))
        azimuths.append(map_sun_azimuth(crs, transform, ac.TD_LAT, ac.TD_LON, g["az"]))
        elevations.append(g["elev"])
        provenance.append(dict(pid=pid, utc=e["utc"], measured_geometry=g,
                               normalization_median=scale, shift_px=e["shift_px"]))
    cfg = ShadowConfig(pixel_m=float(sx), psf_sigma_px=args.psf_sigma_px,
                       registration_sigma_px=args.registration_sigma_px,
                       max_candidates=args.max_candidates)
    print(f"Searching {len(frames)} frames, {2*args.half} px pilot; config {cfg.hash()}", flush=True)
    result = search_stack(np.stack(frames), azimuths, elevations, args.noise_sigma, cfg,
                          slope_rc=(args.slope_row, args.slope_col))
    if args.null_trials:
        print(f"Running {args.null_trials} full-search Gaussian model checks", flush=True)
        result['gaussian_model_check'] = gaussian_search_calibration(
            result, np.stack(frames), azimuths, elevations, args.noise_sigma, cfg,
            trials=args.null_trials, seed=args.seed, slope_rc=(args.slope_row,args.slope_col))
    for candidate in result["candidates"]:
        x, y = transform * (candidate["col_px"] + 0.5, candidate["row_px"] + 0.5)
        candidate.update(x_m=float(x), y_m=float(y))
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    result.update(revision=rev.stdout.strip() if rev.returncode == 0 else "unknown",
                  manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  frames=provenance, azimuths_map=azimuths, elevations=elevations,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  crs=crs.to_wkt(), transform=list(transform),
                  spatial_scope="Central Athena pilot only; local registration accuracy unverified",
                  physical_limits="Constant receiving plane, rectangular caster, Gaussian PSF, approximate solar disc; no resolved-terrain ray tracing")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    fields = ["row_px", "col_px", "x_m", "y_m", "score", "contrast", "identifiability",
              "template_height_m", "template_width_m", "endpoint_censored", "common_fraction"]
    with (args.output / "candidates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["candidates"])
    # A support raster records where all frames are observed. It is deliberately
    # not a probability or a dense candidate score interpolation.
    common = np.isfinite(np.stack(frames)).all(axis=0).astype("uint8")
    with rasterio.open(args.output / "common_support.tif", "w", driver="GTiff",
                       height=common.shape[0], width=common.shape[1], count=1,
                       dtype="uint8", crs=crs, transform=transform, compress="deflate") as dst:
        dst.write(common, 1)
        dst.update_tags(meaning="1=all frames observed; 0=unknown. Neither value establishes safety.")
    print(f"Saved {len(result['candidates'])} ranked candidates to {args.output}")
    print(f"Candidate cap reached: {result['search_truncated']}; common coverage: {result['common_fraction']:.1%}")
    print("Experimental rankings only. Template dimensions are not measured object dimensions.")


if __name__ == "__main__":
    main()
