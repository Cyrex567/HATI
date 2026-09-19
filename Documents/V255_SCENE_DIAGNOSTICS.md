# HATI 2.5.5 — root footprints and scene discrepancies

19 September 2026. This release addresses the supplied 2.5.4 diagnostic bundle. It preserves the separate terrain, shadow and fused maps and adds concrete checks on what their evidence means. It does not claim validated subpixel detections or a mission-loss probability.

## Changed behavior

Previously, every pixel in a four-pixel analysis cell inherited its strongest root score before footprint buffering. That could import a winning root from outside the nominal landing radius. The regional search now retains the maximum template score at **every sampled root**, before cell maxima or candidate deduplication. The shadow footprint takes the maximum only over roots inside the exact disk centred on the output pixel. A weaker root inside the disk remains available when a stronger root in its cell lies outside it.

Root coordinates are pixel-centre indices: add 0.5 before applying the raster transform. Attribution records the controlling root, its distance from the sampled map centre, and its separate distance from the exact test coordinate. These distances can differ because the map samples the test position at a pixel centre. Coverage still conservatively requires assessed cells throughout the disk. Known high evidence remains visible with incomplete coverage; missing low evidence stays unknown. Exact distances do not establish completeness between sampled roots or model the entire obstacle extent.

The new broad-darkness layer identifies image components with an observed inscribed core at least four image pixels across, intensity at most 20% of each frame's finite positive median, and darkness despite nominal DEM illumination in at least two frames. At least three nominally lit, observed frames are required to assess a pixel. The whole qualifying component is recorded. Holes and outside-image regions supply no dark support. These are explicit research settings, not calibrated thresholds or values fitted to the touchdown outcome.

A broad discrepancy may arise from terrain poorly represented by the DEM, albedo or registration. It is **not an object classifier**. Discrepancy or insufficient diagnostic support blocks qualified low-shadow evidence across the footprint, while preserving high warnings. Absence of a discrepancy does not validate the subpixel model. Terrain calculations and the existing template likelihood, noise assumptions and hazard score threshold remain unchanged. The albedo-gain experiment remains diagnostic only.

Local registration diagnostics select frame pairs with Sun-azimuth separation at most 8 degrees and elevation separation at most 1 degree. They search integer offsets within ±6 pixels on fixed 96-pixel tiles, after a two-pixel Gaussian high-pass operation. Every tested displacement uses the same mask, including filter and shift support, with at least 80% common pixels. Flat or unsupported tiles report no displacement. NCC, separated-peak gap and search-boundary flags accompany apparent offsets. Nothing shifts the input images or converts offsets into a registration sigma.

`SceneConfig` records these settings in JSON. The map CLI accepts `--scene-config`; portable review accepts `--config`. Freeze any alternative configuration before independent evaluation.

## Products

The original outputs remain: `terrain_hazard.tif/.png`, `shadow_hazard.tif/.png`, `fused_hazard.tif/.png`, `three_maps.png`, observability, ranking and counterfactual reports.

New outputs include:

- `shadow_root_evidence.npz`: all sampled root coordinates and their maximum template scores, including scores below threshold.
- `shadow_buffer_root_row.tif`, `shadow_buffer_root_col.tif`, `shadow_buffer_score.tif`: the exact footprint's controlling sampled root.
- `shadow_null_energy_per_dof.tif`, `shadow_best_contrast.tif`, `shadow_endpoint_censored.tif`: conditional cell-fit diagnostics, without a calibrated goodness-of-fit interpretation.
- `scene_dark_flag.tif`, `scene_dark_assessed.tif`, `scene_dark_fraction.tif`, `scene_dark_discrepant_frames.tif`, `scene_dark_lit_frames.tif`: discrepancy, support and frame-count layers.
- `local_registration.json`: every eligible pair and tile, including unsupported tiles.

Scenario comparison rejects mixing old cell-based buffering with exact-root buffering, or mixing scene settings, as a registration-sensitivity experiment.

## Stationary WSL run

Use the existing checkout and cached imagery. No downloads or ISIS ingestion are required. The Python code runs on CPU; CUDA is not used.

```bash
cd ~/HATI_V2.0
set -euo pipefail
git switch feat/v2.0-heatmap
git pull --ff-only origin feat/v2.0-heatmap
conda activate isis
bash scripts/run_landing_maps_wsl.sh
```

The wrapper runs nine offline suites, then regenerates all maps for the 0.25, 0.5 and 1.0-pixel registration assumptions, with source/input provenance. Results go to `output/athena/landing_maps_v255/<UTC timestamp>/`, leaving the earlier release's outputs intact. The final comparison is in its `comparison` subfolder.

## Portable diagnostics without cubes

The old diagnostic bundle is sufficient for replaying the new image/DEM discrepancy and registration checks:

```bash
python scripts/review_shadow_bundle.py \
  --bundle output/athena/landing_maps_v253/20260912T072640Z/registration-0.5/diagnostics_v254/hati_diagnostic_bundle.zip \
  --output output/athena/scene_diagnostics_v255/review-01
```

This verifies the aligned-array hash, frame identities, geometry and raster grid; reads only named archive members; and writes diagnostic GeoTIFFs, `scene_diagnostics.json`, a PNG and `SUMMARY.md`. Choose a new output directory for each review. This command does not regenerate the three production maps or new per-root evidence.

## Evidence from the supplied bundle

Default scene diagnostics assess 97.60% of the 512-by-512 window and flag broad-dark discrepancies in 4.27% of the entire window. The fixed touchdown pixel is flagged in all eight nominally lit frames. That supports a model-mismatch interpretation; it does not identify a boulder, measure crater dimensions or validate the mission-loss counterfactual.

Of two eligible similar-illumination frame pairs, only one tile meets the deliberately strict common-support mask, with apparent offset magnitude one pixel. The other pair has no supported tile. These results are insufficient to estimate a regional registration uncertainty or justify dropping frames.

The older albedo experiment produced warnings in 31/61 baseline patches versus 32/61 gain-model patches, with median null residual energy per degree of freedom 13.50 versus 12.37. It therefore has not solved the saturation problem. The new discrepancy layer and footprint correction must not be portrayed as a completed false-alarm calibration.

A complete 512-by-512 replay using the verified bundle and local native DEM completed in approximately 510 seconds, with no ISIS execution. Raw regional score rasters and terrain hazard rasters are identical to the supplied 2.5.3 run. The new bank retains 229,184 sampled-root scores. All buffered sources lie inside their nine-metre map-centre disks. The touchdown's controlling root is now 8.82 m from its exact coordinate (8.93 m from the sampled map centre), and its shadow index changes from 0.9377 to 0.9292. The available shadow map remains 100% above threshold and the raw touchdown cell remains unassessed. The footprint defect is corrected; saturation and scientific validation remain open.

## Stereo confidence correction

`scripts/athena_conf_check.py` now reads the product georeferencing and reports documented confidence categories in `output/athena/athena_conf_categories.json`. It no longer computes means/percentiles of category codes or treats a larger code as universally better. It samples actual touchdown coordinates instead of historical pixel constants.

The test position samples native row 1180, column 386 with code 14; all 25 pixels in the surrounding 5-by-5 window also have code 14. The [official NOBILE03 README](https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/NOBILE03/NAC_DTM_NOBILE03_README.TXT) defines 10–14 as successful correlation, 4 as interpolated/extrapolated and 15 as manually edited. Its 3.7 m SOCET precision and 0.42 m LOLA RMS are different quantities; neither directly supplies the local DEM error covariance required by HATI. This release does not substitute either for the existing assumed vertical sigma.

## Verification and remaining scientific work

Nine offline suites pass. New independent tests compare footprint maxima with brute-force distances, retain weaker in-radius evidence, verify coordinate conventions, distinguish broad and narrow dark structure, reject missing support, recover planted registration offsets with the correct sign, treat flat imagery as uninformative, verify categorical confidence and replay a portable ZIP while external executables are forbidden. Existing independent shadow rendering still agrees with every saved root score.

Next scientific work remains independent resolved-terrain modelling, model-adequacy and false-alarm calibration on separate regions/illumination, defensible spatial DEM uncertainty, and vehicle-specific limits. A qualified research index is not operational landing clearance.
