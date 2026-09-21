# Sequential saturation campaign

The runner follows T1-T8 in `Documents/report/HATI_saturation_analysis.pdf`. It executes all ten offline software suites first, then the map replay and scientific stages one at a time. It uses the verified diagnostic ZIP already produced on the stationary machine. It does not download images, call ISIS, change the saved inputs or tune production thresholds. Numerical computation currently uses CPU NumPy/SciPy, including when run on the GPU workstation.

## Run on the stationary WSL machine

```bash
cd ~/HATI_V2.0
git switch feat/v2.0-heatmap
git pull --ff-only origin feat/v2.0-heatmap
conda activate isis
python -m pip install -r requirements-science.txt

bash scripts/run_saturation_campaign_wsl.sh \
  --bundle output/athena/landing_maps_v253/20260912T072640Z/registration-0.5/diagnostics_v254/hati_diagnostic_bundle.zip \
  --output output/athena/saturation_campaign/athena-01 \
  --export-dir /mnt/c/Users/Public/Downloads/HATI
```

The native DEM defaults to the existing Athena asset location. To select another copy explicitly, append `--dem /path/to/NAC_DTM_NOBILE03.TIF`. It must match the reference projection and native grid orientation. The DEM is hashed before the run. Missing DEM blocks only the map replay; the saved-array scientific diagnostics still run.

Do not run this in a `set -e` chain that treats all nonzero results as missing output. Exit codes are **0** for finished execution, **1** for a failed/interrupted stage, and **2** for a missing prerequisite. A verified results ZIP is still produced after a stage failure or Ctrl-C. A process killed forcibly or a machine shutdown cannot finish packaging; resume the run to recover it. A finished execution can still have a scientifically inconclusive result.

The campaign includes a full regional baseline, every leave-one-frame-out replay, two declared frame-group removals, three full geometry reassignments, an expanded-width regional replay and independent synthetic controls for those variants. Allow a long workstation session. No local timing estimate is used as a promise for another machine.

## Resume after interruption

Run the same command with the same output directory and add `--resume`. Inputs, configuration, Python/platform and scientific source hashes must match. Completed stage artifacts are hash-checked before reuse; completed regional subruns are also reusable inside an interrupted stage. Failed steps rerun. Changed inputs or code require a new output directory. Missing external data can be added to a new campaign, with a new frozen configuration.

## Retrieve and read the output

The command prints `SEND THIS FILE` and `WINDOWS/EXPORT COPY`. With the command above, send:

```text
C:\Users\Public\Downloads\HATI\athena-01_results.zip
```

Unzip the whole archive in Windows Explorer. Open `athena-01/START_HERE.html` in any browser. It works offline, links the results and displays the diagnostic figures. The same findings are in `VERDICT.md`; `results.csv`, `verdict.json` and `campaign.json` support programmatic review. `MANIFEST.sha256` lists the SHA-256 of every packaged file. The ZIP itself has a sibling `.zip.sha256` checksum and is checked for corruption before export.

The archive contains:

- The separate terrain, shadow and fused maps, their three-panel comparison, observability and fixed-touchdown attribution under `stages/maps/maps/`.
- Raw regional arrays, root coordinates and scores, signed candidate frame contributions, complete cell height/width profiles, paired comparisons, synthetic-control records and local registration outputs under the corresponding test folders.
- A log for every software suite and scientific stage, including errors.
- The frozen campaign configuration, verified source input ZIP, software environment and source snapshot under `inputs/`.

## What each stage measures

| Stage | Implemented run | Interpretation |
|---|---|---|
| Maps | Native DEM terrain/visibility replay and full three-map production path | Preserves separate maps and qualification; touchdown warning is attributed to its evidence source |
| T1 | Full regional baseline, signed per-frame contributions, independent rounded-caster/static/structured/ridge scenes under measured geometry and actual masks | No log transform of negative contributions; synthetic detection and warning counts remain conditional on their tested scenes |
| T2 | Every leave-one-out plus the two frame-ID groups in the config; same seeds and injection settings | Parent per-cell frame eligibility and common pixels are frozen. Fit frame count, geometry and identifiability can change. Compare matched roots and injection losses before interpreting changes |
| T3 | Three declared cyclic reassignments of measured azimuth/elevation pairs through the full regional search and matched control scenes | Observed masks and receiving slopes stay fixed. Stress control, not an exchangeable null or p-value |
| T4 | Optional independent thermal footprints aggregated at effective support | Missing thermal input is BLOCKED. Association is descriptive, without a generic conversion to sub-metre rock abundance |
| T5 | Adds the 2.4 m width, records full cell profiles, reruns matched synthetic/null scenes | Profiles maximize over sampled roots for each height/width pair. Wider-bank migration is not proof of a nonphysical signal |
| T6 | Fixed-grid height/root/width profiles at original and larger common supports; nominal endpoint and beyond-endpoint sample support; known-height and overlapping-caster controls | Records bias and empirical inclusion in a declared delta-score set. Status remains PARTIAL because the set has no validated confidence level, and observed shadow continuation is not inferred from a predicted endpoint |
| T7 | 64/96/128 px tile diagnostics, unchanged strict support masks, planted signed shifts with actual missing data and a brightness gradient | Reports unsupported/ambiguous/boundary cases. Does not apply image shifts or derive a regional registration sigma |
| T8 | Optional independent held-out bundle/annotation evaluation at the unchanged production score threshold | Missing annotation input is BLOCKED. Reports cell counts, unknown coverage and recall/false-alarm tradeoffs; does not count replicated pixels as independent cells |

All science settings are declared in `configs/saturation_campaign.json` and saved before experiments run. Frame groups refer to the documented Athena stack. An absent declared frame makes that group unavailable rather than silently substituting another frame. New sites need a new declared configuration.

The independent renderer does not call the production shadow renderer. It uses a rounded transverse caster profile, a point Sun, anisotropic optical blur, static texture and mixed white/correlated noise. It samples the actual saved masks and receiving slopes at fixed fractions of the map. These are deliberate transfer stresses, not independent field truth. Reusing seeds across variants makes comparisons paired. Repeated seeds, roots, neighbouring cells and overlapping thermal footprints are not independent statistical replicates.

## External data for T4

Supply `--thermal footprints.csv`. Each row must have:

```csv
source,footprint_id,row_start,row_stop,col_start,col_stop,rock_abundance,uncertainty
```

Coordinates are half-open pixel bounds on the source aligned grid, covering the **effective detector footprint**, not a fine resampled output pixel. `rock_abundance` is a fraction between zero and one; uncertainty is a nonnegative value in the same units. Use a documented valid polar product, include its source identifier and assign unique footprint IDs. At least the configured fraction of each full footprint must have finite HATI evidence. The worker reports supported footprints and a descriptive Spearman association when defined. It does not infer independence from the row count or validate the external product automatically. Larger-area comparisons need a correspondingly larger matched HATI input.

## External data for T8

Supply `--held-out scenes.json`:

```json
{
  "split": "held_out",
  "annotation_source": "documented independent annotation source and version",
  "independent_of_development": true,
  "scenes": [
    {"bundle": "scene-a/hati_diagnostic_bundle.zip", "labels": "scene-a/labels.npz"}
  ]
}
```

Paths are relative to the manifest. Every scene has its own verified bundle with slope fields. Each labels NPZ contains boolean `hazard` and `complete` arrays on that exact scene grid, plus scalar string `stack_sha256` binding labels to the bundle's aligned-stack hash. An incomplete cell is excluded from labelled counts; an unassessed annotated positive counts as missed in conservative recall. The development stack and duplicate held-out stack hashes are rejected. The same posting and frozen detector/noise settings are used; new settings must be selected on a separate calibration set.

The research targets default to recall at least 0.9 and an assessed-negative warning fraction at most 0.1. They are declared comparison criteria, not verified lander limits. Negative annotations must actually cover the target hazard class; an unannotated subpixel rock must not become a false-positive label. Geographically overlapping scenes require separate review even if their stack bytes differ.

## Verdict rules

Software checks can PASS or FAIL. Scientific stages report COMPLETE, PARTIAL, BLOCKED or FAILED. The execution and scientific verdicts are separate. Missing evidence, errors, partial height validation and unsupported diagnostics cannot produce a scientific pass. The summary lists measured saturation, paired ablation changes, geometry specificity, width preference, height support, registration coverage and any external results.

The runner does not force H1 or H2 from arbitrary cutoffs. They can coexist, and correct Sun dependence does not distinguish all resolved terrain shadows from unresolved casters. Operational use remains withheld: this campaign does not validate footpad dynamics, a height uncertainty model or a bound on loss of mission. The next run is designed to expose which explanations survive the declared controls and what evidence still prevents a final field verdict.
