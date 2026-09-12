# HATI 2.5.4 — attribution and shadow-model diagnosis

12 September 2026. This release responds to the completed real run `20260912T072640Z`. It adds explanations and diagnostic experiments; it does not claim that the saturated shadow map is corrected or validated.

The real 0.5-pixel scenario had 89.1% raw shadow exceedance and 100% buffered exceedance among available pixels. The touchdown's raw cell was unassessed (76.1% common support versus 80% required). Its buffered warning came from a regional-cell winner about 10.58 m from the exact location. The four-pixel cell maximum extends influence beyond the nominal nine-metre buffer. Neither that warning nor zero disagreement among saturated threshold maps establishes an independent hazard counterfactual.

## Stationary-machine command

Run in the existing checkout containing the completed sweep and its output:

```bash
cd ~/HATI_V2.0
set -euo pipefail
git switch feat/v2.0-heatmap
git pull --ff-only origin feat/v2.0-heatmap
conda activate isis
bash scripts/run_shadow_diagnostics_wsl.sh \
  output/athena/landing_maps_v253/20260912T072640Z/registration-0.5
```

The environment holds the Python dependencies. No ISIS command, download or new ingestion runs. The script reads cached cubes and geometry sidecars, verifies their manifest hash, identities, geometry and grid against the saved output, and exports the aligned stack before evaluating diagnostics. The original hazard maps remain intact.

Send this result:

```text
output/athena/landing_maps_v253/20260912T072640Z/registration-0.5/diagnostics_v254/hati_diagnostic_bundle.zip
```

It contains aligned intensities without extra numeric downcasting, measured illumination, nominal DEM visibility, receiving gradients, coordinate reference, source run metadata, attribution, model diagnostics and source/input hashes. This is the missing evidence needed to distinguish photometric mismatch from registration or other residual structure.

## Warning attribution

New map runs save every assessed cell's winning root coordinates before candidate deduplication. `counterfactual.json` and `warning_attribution.json` distinguish a regional warning with an unassessed direct cell from a general configured regional warning. They report direct support, raw score availability, regional exceedance fractions, the source cell, root distance and whether the root is inside the nominal buffer radius.

Historical output can be inspected without image cubes:

```bash
python scripts/diagnose_landing_run.py \
  --run-dir output/athena/landing_maps_v253/20260912T072640Z/registration-0.5 \
  --report-only
```

For older outputs, a deduplicated candidate is named as the source only when score and owned cell agree uniquely. Otherwise it stays unresolved. Cell/buffer geometry is recorded rather than reinterpreted as an exact root-distance exclusion. Terrain, shadow and fused map calculations and thresholds are unchanged.

## Diagnostic experiments

The default fixed grid has patch centres 64 pixels apart, retaining original image posting. Within each supported patch, 24 combinations sample four root offsets, three heights and two widths. This is a small patch-centre search, not the complete regional search or object completeness.

Two null models are compared: the existing static-albedo/per-frame-plane model with first-order registration covariance, and an extension permitting each frame to scale a frozen reference texture. The full-patch reference is a median image. Both data and shadow templates pass through exactly the same extended nuisance operator; registration covariance is rebuilt outside that nuisance space.

The extra texture-gain term is diagnostic only. It can absorb real signals as well as artifacts, so fewer warnings do not by themselves mean improvement. Residual energy per degree of freedom is conditional on a data-derived reference and assumed noise, not a calibrated chi-square test.

Each model receives three cyclic template-frame reassignment stress controls with observed masks fixed. These are not exchangeable null draws or calibrated false-alarm probabilities.

When at least six frames are usable, an alternating manifest-frame split selects root, template and contrast using training intensities only. Held-out frames keep these parameters fixed, refit nuisance terms, and report signed fit improvement. The texture/covariance reference also uses training frames only. Similar Sun directions may occur in both splits; this is not held-out-site validation.

Per-patch results and frame identities are retained. No diagnostic changes the production threshold or automatically chooses a model. The next decision requires real residual inspection and independent controls, not making the touchdown score favourable.

## Verification and remaining work

Eight offline suites pass, including seven new diagnostic regressions: dense least-squares equivalence of the gain nuisance, gain-only texture removal, survival of a planted moving-template signal, deterministic controls, no held-out leakage into training selection, unresolved sources, buffer attribution and an intensity export/bundle exercise with external executable calls forbidden after plotting initialization. These mathematical and synthetic tests do not establish lunar performance.

The attribution tool also reproduced the unassessed direct cell and historical winning root from the supplied real output without ingestion. Real intensity diagnostics remain pending until the command above runs on the host holding the cubes. Accurate photometric/noise and local-registration diagnosis, independently justified DEM relative-height uncertainty, and independent controls remain necessary before claiming a discriminating counterfactual.
