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
    ShadowConfig, gaussian_search_calibration, search_stack)
from sweep_contract import DEFAULT_BEFORE


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
    from sweep_products import load_sweep
    sweep = load_sweep(args.manifest,args.half,args.before)
    frames = sweep['stack']
    azimuths,elevations = sweep['azimuths'].tolist(),sweep['elevations'].tolist()
    transform,crs,sx = sweep['transform'],sweep['crs'],sweep['pixel_m']
    provenance = sweep['frames']
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
