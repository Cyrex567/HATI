# HATI v2.0 — NAC pipeline, Stage B results

**Frame:** M1412631647RE (LROC NAC EDR, Apollo 17 / Taurus–Littrow)
**Crop:** 1600×1600 px at line 25400, sample 1800 — valley floor, 1.29×1.29 km (1.671 km²)
**Native scale:** 0.808 m/px (worked in raw EDR pixel space — see "Why EDR space")
**Script:** `scripts/nac_shadow_sfd.py`
**Outputs:** `M1412631647RE_stageB_imagery.png`, `M1412631647RE_stageB_statistics.png`

## What Stage B does

Stage A showed (qualitatively) that NAC shadows flag obstacles the 1.5 m/px
DEM cannot resolve. Stage B **quantifies that population** into a
size-frequency distribution (SFD), separating two physically distinct
measurements from each shadow's fitted ellipse:

| Measurement | From | Depends on sun elevation? | Tells us |
|---|---|---|---|
| Cross-sun **footprint width** | ellipse minor axis | **No** | object horizontal size |
| Along-sun **shadow length** → relief `h = L·tan(e)` | ellipse major axis | Yes (via `e`) | obstacle height / relief |

Solar elevation `e` is the **only** quantity not in the EDR label (geometry
is added downstream at CDR/RDR via SPICE). Because relief scales as `tan(e)`,
the relief SFD is shown across a plausible band `e ∈ {25°, 40°, 55°}`. The
detection itself and all footprint widths are independent of `e`.

## Headline result (elevation-independent)

Of **668** border-clean shadows (≥3 px) over 1.671 km²:

| Cross-sun footprint | count | per km² | vs DEM |
|---|---|---|---|
| < 1.5 m | 145 | 87 | below floor |
| 1.5–3 m | 315 | 188 | below floor |
| > 3 m | 208 | 124 | DEM-resolvable |

**≈ 69 % of shadow-flagged obstacles (275/km²) have footprints below the
DEM's ~3 m resolving floor** (2× the 1.5 m posting, Nyquist). This is the
sub-resolution hazard population HATI's heatmap is built to surface — measured
directly, with no illumination assumption.

## Relief result (depends on sun elevation)

Obstacles with inferred relief in the 0.5–3 m lander-hazard band:

| Sun elevation | hazard-band count | per km² |
|---|---|---|
| 25° (low) | 475 | 284 |
| 40° (central) | 273 | 163 |
| 55° (high) | 64 | 38 |

## Two honest caveats

1. **Detection floor, not a real cutoff.** The relief SFD curves stop near
   ~1 m and `<0.5 m = 0` — this is an *artifact*: a sub-0.5 m object throws a
   sub-3 px shadow we reject. The smallest detectable shadow (3 px ≈ 2.4 m
   long) corresponds to ~1.1 m relief at 25° sun, ~3.5 m at 55°. So Stage B
   robustly samples the **~1–3 m** part of the hazard band; the 0.5–1 m part
   sits at/below our floor and is *under*-counted, not absent.

2. **Threshold sensitivity ≈ 2×.** Shadow count scales with the DN cutoff:
   DN<24 → 457, DN<28 → 668, DN<32 → 984 shadows (hazard-band @40°:
   111 / 163 / 248 per km²). Report densities as *order ~10²/km²*, not
   false-precision single numbers.

## The key physics — why this matters for Athena

Relief sensitivity *improves as the sun gets lower*: grazing light turns a
sub-meter obstacle into a long, easily-detected shadow. At 25° sun the method
reaches ~1.1 m relief; at 55° only ~3.5 m. **The lunar south pole — Athena's
site — has permanently grazing illumination.** So shadow-based sub-resolution
hazard detection is *most* effective exactly where high-value landings happen
and where Athena was lost. The mid-latitude Apollo 17 frame is a conservative
demonstration; polar performance should be better, not worse.

## Internal validity check

The modal shadow-elongation axis is **unimodal at −11.5°** (image frame). A
population of *random* dark patches (albedo, noise) would have no preferred
orientation; a single dominant axis confirms these are **illumination-driven
cast shadows**. (Tying −11.5° image-frame to a compass azimuth needs the
north direction, i.e. SPICE — deferred. Unimodality is the point here.)

## Why EDR (native) space, not map-projected

Map-projection resampling smears sub-pixel shadows — the very signal we count.
For *sub-pixel hazard statistics* the raw EDR is the correct domain, so the
absence of a geotransform on the EDR is not a limitation for Stage B. It does
mean Stage B is a **statistical population** result, not a georeferenced map;
pixel-level NAC↔DEM fusion needs map-projected NAC (RDR) or SPICE, deferred.

## Where this sits in the pipeline

- **Stage A** (done): shadows reveal sub-Nyquist obstacles; bimodal DN split at ~28.
- **Stage B** (this): quantified SFD — 275/km² obstacles below the DEM floor; low-sun→deeper sensitivity.
- **Next (laptop-doable):** multi-crop / multi-terrain SFD (valley vs massif vs crater ejecta) to show the population varies with terrain — the basis for a NAC hazard *channel*.
- **Fusion with DEM** (deferred): needs map-projected NAC (RDR) or SPICE for co-registration.
- **Athena counterfactual** (GPU-gated): Mons Mouton DEM + polar NAC.
