# HATI v2.0 Tier-1 heatmap — first results

Author: Gergő Csaba Morvai
Date: 2026-05-26
Status: working pipeline, validated on Apollo 17 site, ready for the v2.0
paper. Companion document to `heatmap_explained.md`.

## What was built

The Tier-1 sub-resolution hazard heatmap, as specified in
`heatmap_explained.md`, is now a runnable pipeline. The implementation
lives in `src/heatmap/`:

- `dem_features.py` — ten DEM-derived channel implementations (MDS at
  four scales, RMS slope, IQR slope, IQR profile curvature, RMS planar
  deviation, |TPI|, TRI).
- `fusion.py` — literature-anchored fusion: robust z-score with `log1p`
  pre-transform and clipping, normalised by total weight magnitude, plus
  bias, plus logistic sigmoid to [0, 1].
- `__init__.py` — module exports.

Runner scripts in `scripts/`:

- `run_heatmap_apollo17.py` — full pipeline on the LRO Apollo 17 DEM
  (full or `--subset` centre crop, with optional `--from-cache` to skip
  channel recomputation for fusion tuning).
- `validate_channels.py` — sanity validation (inter-channel correlation,
  heatmap distribution).

## Validation status

### Internal sanity (passes)

| Check | Result | What it tells us |
|---|---|---|
| Heatmap range | [0.07, 0.95] | No saturation, proper continuous probability surface |
| Heatmap mean | 0.36 | Consistent with mare-like terrain |
| High-hazard fraction | 20% above 0.5, 5% above 0.8 | Reasonable for Apollo 17 (crater-bearing mare) |
| Distribution shape | Unimodal, log-tailed | Proper probability distribution, not bimodal |
| Mean inter-channel correlation | +0.43 | Healthy range; channels agree but aren't redundant |

### Per-channel observations

`rms_slope` and `tri` correlate at r = 0.97. Effectively duplicate
channels. A future tuning pass should down-weight one or drop it.

`mds_L3` and `mds_L5` correlate at r = 0.94. Adjacent-scale channels in
the MDS family are expected to correlate. Keeping both still adds
slightly different sensitivity profiles around the Nyquist limit.

`iqr_curvature` correlates at 0.02 to 0.09 with every other channel.
This is the anomaly. Either it captures information no other channel
sees (a real positive) or it is noise-dominated (the Laplacian
amplifies high-frequency noise, and the IQR over a window then captures
that noise's spread). Needs investigation before the v2.0 paper. The
visual panel suggests it is more noise than signal.

`rms_planar_dev` correlates strongly with `mds_L5` (0.84) and `iqr_slope`
(0.63). It's measuring residual height after detrending, which is close
to what a small-scale MDS captures. Keep but consider as a partial
substitute if MDS is dropped.

### V3 (audit-based) validation: deferred

The 1,459-object forensic audit from v1.5 is the right validation set
for the heatmap. It requires the `titan_hazard_map.geojson` produced by
Phase 1 of the v1.5 pipeline. On this laptop, Phase 1 has not been run
(it requires GPU for the 7-minute Titan scan; CPU would take many
hours). When the GPU box is next available:

1. Run Phase 1 to regenerate the hazard map.
2. Run the v1.5 forensic audit (Phase 3) to get the 1,459-object
   relief classifications.
3. Compute AUC of heatmap centroid value as a discriminator between
   the ~1,306 valid hazards and ~153 flat-ground noise objects.
4. Target: AUC > 0.7 (gate); AUC > 0.8 would be strong.

The fact that the heatmap visually picks out exactly the crater rims
and the central massif on Apollo 17 is consistent with a positive V3
result, but the formal AUC number must wait.

## The figures

- `apollo17_heatmap_main_subset.png` — single-panel hillshade overlay,
  2000 × 2000 px centre crop of the Apollo 17 site. The deliverable
  figure for outreach communication.
- `apollo17_heatmap_overview_subset.png` — multi-panel diagnostic with
  DEM, hillshade, fused heatmap, pre-sigmoid, and each individual
  channel's z-score.
- `apollo17_channel_correlation.png` — Pearson correlation matrix of
  the ten channels.
- `apollo17_heatmap_distribution.png` — heatmap distribution histogram.

A full-DEM run (9240 × 7800 px, the entire Apollo 17 site) is also
available as `apollo17_heatmap_main.png` (if you see the file alongside
this document) at the time the README was last updated.

## What's still missing for the v2.0 paper

In rough priority order:

1. The NAC multi-illumination shadow channels (Tier-1.5 from the
   `heatmap_explained.md` plan). These give the heatmap genuine sub-
   resolution access. Requires LROC NAC fetching and coregistration.
2. The audit-based V3 validation, once the v1.5 pipeline can be re-run
   end-to-end on the GPU box.
3. The Ripley's K conditioning test (V1). More work to implement than
   V3.
4. The terrain-relative navigation (TRN) layer.
5. Application to the Mons Mouton terrain (the Athena counterfactual
   site), once we have the right LROC products for that region.

## Reproducibility

```bash
# From the repository root
pip install -r requirements.txt
pip install scikit-learn scikit-image      # not in the v1.5 requirements
python scripts/run_heatmap_apollo17.py --subset    # 2-3 min
python scripts/run_heatmap_apollo17.py             # 30 min CPU
python scripts/validate_channels.py
```

Everything in `src/heatmap/` is CPU-only and runs on a Galaxy Book-class
laptop with 16 GB+ RAM.
