# Adaptive shadow development and workstation run

This implementation adds an experimental adaptive path to HATI v2.5. It tests whether larger observed context, a finer joint dimension bank and stronger background alternatives improve the current detector. It does not establish 10 cm accuracy, an operational false-alarm rate, landing clearance or a probability of loss of mission.

## Implementation and rationale

1. **Record missing context.** The baseline retains its scores and three maps. Additional arrays record the winning template height/width, dimension-bank boundaries, number of censored predicted endpoints and number without observed support. Adaptive histories distinguish patch cutoff, fitting-support cutoff, missing endpoint pixels, missing background beyond an endpoint and invalid geometry. These are predictions about support, not detected physical shadow edges.
2. **Queue regions deterministically.** Every assessed cell with a baseline warning, predicted cutoff or missing endpoint can request refinement. If at least 25% of assessed cells in a processing tile have predicted cutoffs, the remaining assessed cells in that tile also receive context checks. Adjacent requested cells form scheduling/display regions. Each cell retains its own original root grid; region grouping never pools them into one inferred rock. Baseline-unavailable cells remain unavailable. The workstation queue has no candidate cap.
3. **Expand real image context.** The baseline 25-pixel patch/6-pixel fitting radius becomes 49/12 and then 97/24. The code reads existing aligned pixels without resampling. Eligibility is fixed by the first experimental pass; subsequent passes cannot remove an inconvenient frame. Insufficient coverage, image edges or invalid terrain stop the cell with an unresolved status. Earlier fits remain in its history, but a failed last pass cannot supply final dimensions.
4. **Check receiving-plane limitations.** The maximum change in receiving gradient within the fitting disk, multiplied by its physical radius, must not exceed the declared 0.15 m departure proxy. This is a conservative model-domain check, not a measurement of DEM error or a complete terrain ray tracer. HATI reports unresolved terrain when this local-plane model is unsuitable. It does not invent a terrain extension beyond available data.
5. **Refine height and width together.** Coarse knots span 0.1–2.4 m in height and 0.2–2.4 m in width. For candidates above the declared score threshold, the engine fills the neighbouring intervals around every compatible coarse pair at 0.1 m spacing, retaining the coarse knots. All original cell root positions participate. Streaming one rendered shape at a time bounds memory. The stored surface profiles over roots; the finite-grid compatibility set uses a declared delta of 4 and is not a confidence interval. Coarse-to-fine selection can miss structure between unselected knots and therefore needs empirical testing.
6. **Stop conservatively.** Context is marked supported only when the compatible models' predicted endpoints and background lie on common observed pixels, the dimensions avoid the bank boundaries, both dimension spans are at most 0.4 m, and the winning dimensions change by no more than 0.2 m from the preceding scale. This remains an unvalidated model status. Other outcomes are low evidence/unqualified, unresolved, or queued/unprocessed. The algorithm never chooses the maximum raw score across different windows.
7. **Strengthen the competing explanation.** Adaptive fits use static texture plus an independent quadratic illumination surface per frame, with the existing registration covariance model. Data and templates pass through exactly the same nuisance operator. Baseline production inference keeps its previous linear illumination model. The comparison can reveal sensitivity lost to the stronger alternative; reduced scores alone are not success.
8. **Predict withheld illumination.** T11 evaluates a fixed lattice, all declared scales, every withheld frame and both linear/quadratic nuisance families. Training fixes object position, dimensions, contrast and covariance reference. The withheld intensities never select them or the window. Every prediction uses the same validity mask and nuisance treatment. Saved errors compare a static background, the measured Sun geometry and a 90-degree rotation of only the withheld prediction. This is conditional prediction inside one scene; T8 remains the independent annotated-scene test.
9. **Generate independent rocks.** The new renderer projects clipped triangular meshes onto receiving planes, samples seven positions on the solar disc, integrates pixels and applies anisotropic blur and correlated noise. It shares no rendering function with the detector. Seeds fix shape, yaw, burial, placement and noise. Scenarios include isolated rocks, overlap, changing backgrounds, ridges, missing context, registration stress, sloping planes and a terrain break. Photometry is simplified: rock-body reflectance, mutual illumination and multiple scattering are absent.
10. **Use actual lunar shape sources with explicit assumptions.** Two small convex proxies come from NASA Apollo samples 10017,15 and 10021,79. Their parent rocks are separated between development and evaluation. Original archive, OBJ and metadata hashes, reduction settings and NASA attribution accompany them. The 96-direction reduction removes concavities and fine detail. Scene dimensions and orientation are imposed. One evaluation rock and rescaled laboratory samples do not characterize the polar boulder population. The OBJ adapter accepts additional attributed, hash-verified catalogs and rejects shared parent identities or duplicate payloads across splits.
11. **Preserve observable outputs.** Baseline terrain, shadow and fused heatmaps remain separate under `stages/maps/maps/`. T9 adds GeoTIFFs, arrays, region IDs and per-cell histories. HATI Watch shows the actual expanding patch, predicted shadow, residual, endpoint reasons and dimension compatibility. A queued or unresolved area cannot acquire a completed status through display logic. Synthetic local coordinates are never overlaid on the real Athena image.
12. **Package and resume.** Campaign stages run sequentially. T9 uses four CPU processes with one numerical thread each, a bounded task queue and deterministic result order. Completed cells have input/configuration signatures and checksums; interrupted work resumes under the runner's frozen-source rules. All results, source code, mesh payloads, configuration, input hashes and summaries enter the ordinary verified ZIP. CPU workers do not imply GPU acceleration or bitwise equivalence across different numerical-library versions.

## Validation gates

Software checks cover real-pixel expansion, fixed target grids and frame sets, endpoint support reasons, missing data, image edges, terrain limits, explicit budget exhaustion, observer independence, serial/parallel equality, withheld-intensity leakage, mesh geometry, seed replay and source/split checks. Small complete workers verify T9–T11 exports and checkpoints.

The local validation on 23 September 2026 passed all 14 software suites, including 18 new targeted tests. A small command-line campaign ran the stage sequence, exported all three maps, verified 286 result-file hashes and ZIP integrity, and reused all completed stages on resume. T4/T8 correctly remained blocked without external data; no scientific stage crashed. Browser inspection confirmed the adaptive state/compatibility display and plot selection without console errors. These are implementation checks, not the full workstation experiment.

A 56-by-56 pixel cached Athena cutout reproduced all 16 existing baseline arrays and its candidate list exactly against the prior committed implementation. Four cells near the fixed touchdown exceeded the new receiving-plane tolerance (departure proxies about 0.20–0.29 m versus the declared 0.15 m limit). They remained unresolved. Low-evidence best-fit parameters stay in audit histories but do not populate object-dimension maps. The new full workstation run is still required to measure coverage, recovery and prediction behaviour across the scene. No ISIS ingestion ran on the laptop.

The workstation campaign measures the full adaptive request/stop procedure within predeclared synthetic central regions. Each scene receives a full-frame baseline; a fixed 3-by-3 cell region, chosen before seeing intensities, receives adaptive tests. This bounds cost while exposing unknown denominators. It is **not** a full-image false-alarm calibration. Synthetic height trials cover 0.2–1.2 m in 0.1 m steps, with two seeds, procedural and Apollo-derived shapes, and overlapping rocks. The nearest predeclared cell supplies dimension errors; the analysis does not select whichever cell happens to estimate the truth best.

Scientific acceptance requires fewer unresolved cutoffs without hiding unavailable coverage, retained recovery of small objects, acceptable changing-background response, useful withheld-frame predictions, and honestly reported dimension bias and missing-estimate fractions. No acceptance threshold is fitted to make the Athena touchdown look successful. T9–T11 remain PARTIAL until independent field evidence and calibration support stronger claims. Thermal data and annotated scenes, if absent, remain explicitly blocked in T4/T8.

## Run on the stationary WSL machine

Use a new run directory because sources and configuration have changed. The cached diagnostic bundle contains the aligned NAC images and illumination geometry; these stages do not call ISIS or download imagery. The added DEM checks and synthetic engine supplement the existing ingestion pipeline.

```bash
cd ~/HATI_V2.0
git switch feat/v2.0-heatmap
git pull --ff-only origin feat/v2.0-heatmap
conda activate isis
python -m pip install -r requirements-science.txt

bash scripts/run_saturation_campaign_wsl.sh \
  --bundle output/athena/landing_maps_v253/20260912T072640Z/registration-0.5/diagnostics_v254/hati_diagnostic_bundle.zip \
  --config configs/saturation_campaign_adaptive_workstation.json \
  --output output/athena/saturation_campaign/athena-adaptive-01 \
  --export-dir /mnt/c/Users/Public/Downloads/HATI
```

In a second WSL terminal:

```bash
cd ~/HATI_V2.0
conda activate isis
python dashboard/hati_watch.py \
  --run-dir output/athena/saturation_campaign/athena-adaptive-01
```

Open `http://localhost:8765`. After the campaign finishes, send `C:\Users\Public\Downloads\HATI\athena-adaptive-01_results.zip`. Extract the entire archive and open `START_HERE.html`. Add `--resume` to the unchanged campaign command after interruption. Exit code 2 means a missing prerequisite; the results ZIP is still produced. Missing external validation data must not be mistaken for an implementation crash.

To use another frozen local catalog, append `--rock-catalog /path/to/catalog.json`. The bundled catalog requires no download. `scripts/prepare_lunar_rock_catalog.py` is an explicit, separate provenance-building utility; the scientific runner never calls it.

Sources: [NASA mesh availability and credit](https://ares.jsc.nasa.gov/astromaterials3d/faqs.htm), [sample 10017,15](https://ares.jsc.nasa.gov/astromaterials3d/sample-details.htm?sample=10017-15), [sample 10021,79](https://ares.jsc.nasa.gov/astromaterials3d/sample-details.htm?sample=10021-79). [BOULDERING](https://zenodo.org/records/14250874) remains a possible source for image-outline priors; it is not incorporated in this release and supplies no automatic ground-truth height.
