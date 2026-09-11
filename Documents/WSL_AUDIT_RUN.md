# HATI v2.5 WSL runbook

**Completed the audited ingestion already?** Use the [2.5.3 three-map postprocessing runbook](V253_LANDING_MAPS.md) and `bash scripts/run_landing_maps_wsl.sh`. It reuses those products and does not run ISIS. The full ingestion runner below now also invokes that postprocessing step after its legacy pilots.

Run on the Linux/WSL GPU box with the existing reference data. No ISIS ingestion was run on the Windows audit machine. This updates the checkout, tests the deterministic Python path, ingests a pre-landing sweep and produces experimental root rankings for a central 512-pixel pilot under three registration-blur assumptions.

```bash
cd ~/HATI_V2.0
set -euo pipefail
git switch feat/v2.0-heatmap
git pull --ff-only origin feat/v2.0-heatmap
conda activate isis
python -m pip install -r requirements-science.txt
bash scripts/run_v25_wsl.sh
```

Change the first line if your checkout is elsewhere. Use the Linux filesystem with its `data/` products. Logs record the Git revision, Python versions and stage output. `pipefail` stops on test/ingestion failure even through `tee`. The script requires `lronacecho`; do not remove the correction to reuse old projections. Twenty-four means selection bins, not a guarantee of 24 usable frames.

The first audited ingest regenerates legacy projections while retaining complete EDR downloads. Matching new caches remain reusable. `--rebuild` deliberately regenerates even matching products. Preserve the manifest, coregistration report, pairwise closure JSON, geometry sidecars, processing contracts and ISIS labels. Inspect `Kernels.ShapeModel` in `.isis-label.pvl`: a spherical map projection does not establish the camera shape surface.

Each pilot writes `run.json`, `candidates.csv` and `common_support.tif` under `output/athena/shadow_roots_v25/<run-id>/`. Registration sigmas 0.25, 0.5 and 1 pixel, radiance sigma 0.03, optical PSF sigma 0.6 pixel and a flat receiving plane are **sensitivity assumptions**, not measurements. Candidate dimensions are template parameters. Censored endpoints cannot supply measured heights. Missing support, candidate-cap truncation and per-frame contradictions are saved. No candidate means no clearance conclusion.

The new runner only reads products and measured geometry sidecars; it never calls ISIS. The core's projection helper supports both poles, but this ingestion adapter remains Athena-specific. The 40-degree emission cap is a heuristic, not a local parallax bound. The pilot lies inside the registration window, which does not prove local subpixel accuracy. These routines use the GPU *box* as a CPU host; CUDA acceleration is not implemented.

Optional synthetic characterization, without ISIS:

```bash
python scripts/benchmark_shadow_likelihood.py --seeds 12
```

Optional full-search Gaussian model diagnostic after ingestion (potentially slow):

```bash
python scripts/shadow_roots_real.py \
  --half 128 --registration-sigma-px 0.5 --noise-sigma 0.03 \
  --null-trials 99 --seed 2718 \
  --output output/athena/shadow_roots_v25/gaussian-model-check
```

This reruns proposals and the entire template search on every null draw, then uses the plus-one rank. Its global p-value assumes independent Gaussian noise with fixed known sigma and support. Correlated residuals, photometric normalization, slope mismatch and registration errors violate this simplified model. It is not a measured lunar false-positive rate.

The legacy spatial-vote diagnostic remains available:

```bash
python scripts/shadow_kinematics_real.py \
  --before 2025-03-06T00:00:00Z --half 0 \
  --trials 1000 --inject 60 --no-rebuild
```

Read the [baseline audit](AUDIT_2026-09-09.md) and [release assessment](V25_RELEASE.md) before interpreting either detector.
