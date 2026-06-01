# HATI v2.0 — IM-2 "Athena" landing-site counterfactual

**Question.** Using **only pre-landing data**, would HATI's two pipelines have
flagged the *actual* IM-2 Athena touchdown point as hazardous?

**Touchdown:** 84.7906° S, 29.1957° E — Mons Mouton, lunar south pole.
IM-2 landed 2025-03-06 and tipped over; the LROC team attributes the upset to
a ~20 m crater on sloped terrain at the touchdown.

**Data (all 2012 — predates the landing, so no circularity):**
LROC NAC DTM **NOBILE03** (ASU / M. Robinson), built from frames
M1101075756 / M1101097181 / M1101118606, 2012-08-31.
- DTM `NAC_DTM_NOBILE03.TIF` — **4.0 m/px** stereo elevation (1473×1172, 45% nodata diamond).
- Ortho `NAC_DTM_NOBILE03_M1101075756_90CM.IMG` — **0.9 m/px** co-registered NAC.

**Scripts:** `scripts/athena_counterfactual.py`, `scripts/athena_figure.py`.
**Figure:** `output/athena/athena_counterfactual.png`.

**Success criterion (set by the user):** if *either* pipeline flags the
touchdown, the counterfactual succeeds.

---

## Verdict: SUCCESS — Pipeline 1 flags the point; both confirm the site

| Pipeline | Measures | Result at touchdown | Flag? |
|---|---|---|---|
| **1 — DEM heatmap** | resolved fine-scale roughness (4 m/px) | H = **0.726**, **96.3rd percentile** of the massif | **YES** |
| **2 — NAC shadow census** | sub-DEM obstacles (0.9 m/px) | **1417 obstacles/km² below the DEM floor**; 3 within 25 m | corroborates* |

\* Pipeline 2 does not flag the point as a *local* density anomaly — see below.

---

## Pipeline 1 — DEM hazard heatmap (the flag)

The identical, literature-anchored Tier-1 heatmap from the Apollo 17 work,
run on the 4 m/px NOBILE03 DTM. The touchdown lands at the **96th percentile**
of fused hazard across the (already rugged) Mons Mouton massif.

**The flag is robust, not config-fished.** Across every scale config that is
*computable* at this near-edge site, the touchdown percentile is consistently
high:

| Config | MDS scales | H(touchdown) | scene percentile | flag>p75 / p90 |
|---|---|---|---|---|
| Apollo default w=11 | 12–80 m | *NaN (uncomputable)* | — | — |
| fine w=5 | 8–32 m | 0.757 | **97.1 %** | yes / yes |
| fine w=7 | 8–32 m | 0.726 | **96.3 %** | yes / yes |
| fine w=9 | 8–24 m | 0.645 | **93.3 %** | yes / yes |
| finest w=7 | 8–20 m | 0.780 | **97.4 %** | yes / yes |

**The flag is driven by fine-scale roughness, not gross slope.** Slope at the
touchdown is only **2.8°** (local 44 m max 6.4°) — below any lander GNC slope
limit. A slope threshold would pass this site; the multi-scale roughness /
curvature stack does not. **This is the core HATI thesis, demonstrated on the
real failure site:** the hazard is in the texture, below what a slope check or
the coarse DEM sees directly.

The ~20 m crater the LROC team identified falls squarely in the 8–32 m scale
band that carries the flag.

---

## Pipeline 2 — NAC shadow census (the sub-resolution layer)

Local-relative cast-shadow detection (DN < 0.5 × local sunlit background, the
standard approach under the strong slope-driven illumination gradient of polar
terrain) on the 0.9 m/px ortho, over a 1.08 km² box around the touchdown:

- **1514 shadows/km²** total; **1417/km² (94 %)** have cross-sun footprints
  **below the 4 m/px DEM's 8 m resolving floor** — a hazard population the
  planning DEM is physically blind to.
- **Modal shadow orientation 85°, unimodal** — confirms these are
  illumination-driven cast shadows, not random albedo/noise (validity check,
  as in Stage B).
- Within the **25 m landing dispersion** of the exact touchdown point: **3
  sub-resolution obstacles** (9 within 50 m). Nearest shadow 3.7 m away.

**Why this is corroboration, not an independent point-flag.** The touchdown's
local sub-resolution obstacle density is the **49.9th percentile (median)** of
the region — because the *entire* Mons Mouton site is uniformly carpeted with
sub-resolution obstacles. There is no "safe" ground nearby to stand out
against. That is arguably a *stronger* statement than a lucky pixel: the site
is pervasively hazardous below the planning resolution, and small retargeting
would not have helped.

---

## What this does and does not claim

**Does:** with pre-landing data alone, HATI's DEM heatmap independently ranks
the actual touchdown in the top ~4 % of an already-rough massif, robustly
across scale configs, on fine-scale roughness rather than slope; and the NAC
shadow census quantifies a dense sub-DEM-resolution obstacle field at the same
point. The site was **flaggable before the landing**.

**Does not:** claim HATI uniquely "found the 20 m crater," nor that it would
have changed IM-2's outcome (that needs the mission's actual hazard maps and
GNC). This is a retrospective demonstration of *detectability*, not a mission
post-mortem.

## Honest caveats

1. **Relative heatmap.** H is z-scored against the scene, so "96th percentile"
   means *rougher than 96 % of the Mons Mouton massif* — the baseline itself is
   hazardous. The flag is a *ranking*, not an absolute probability.
2. **Near the data edge.** The touchdown is 116 m from the DTM coverage edge,
   so the coarsest (80 m) MDS scale cannot be evaluated there (reported as
   NaN, not hidden). The 8–32 m scales — the correct band for a ~20 m crater —
   are fully inside valid data and carry the flag.
3. **Polar illumination.** A global DN threshold fails at grazing sun
   (slope-shading dominates); we use a local-relative threshold. Shadow counts
   scale with that fraction (~Stage B's DN sensitivity), so densities are
   *order ~10³/km²*, not false-precision single numbers.
4. **Orthorectified imagery.** The 0.9 m/px ortho is resampled (Stage B's
   sub-pixel-smear caveat); acceptable here because polar shadows span many
   pixels. Absolute shadow azimuth needs SPICE (deferred); only unimodality is
   used here.
5. **DEM floor = 8 m** (2-px Nyquist at 4 m/px).

## Why the south pole helps (ties back to Stage B)

Stage B showed shadow-based sub-resolution sensitivity *improves at low sun*.
Mons Mouton has permanently grazing illumination, which is exactly why the
0.9 m/px ortho resolves such a dense shadow-obstacle field. The mid-latitude
Apollo 17 demonstration was conservative; polar performance is better — and the
pole is where the high-value (and high-loss) landings happen.

## Provenance / reproducibility

- DTM + ortho georeferenced in south-polar-stereographic (sphere R=1737.4 km,
  central meridian 0, planetocentric). Touchdown → DTM px (row 1181, col 387),
  ortho px (row 5248, col 1721); closed-form and GDAL transforms agree to 0 px.
- Caches: `output/athena/athena_dem.npz`, `athena_shadow.npz`.
- Pre-landing source frames (2012-08-31) guarantee no post-landing circularity.
