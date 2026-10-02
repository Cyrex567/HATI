# The DEM and the noise the shadow detector sees

The relief correction (T14, and the relief option of the landing maps) fits a metre-scale surface by shape from shading and removes its predicted shading before the detector runs. It starts from flat ground: the solver linearises every slope around zero, and its surface has no absolute tilt, which a brightness plane per frame absorbs. The 4 m NAC DTM (NOBILE03) never enters it; the landing maps merge it in afterwards, for shadow tracing only (`relief_terrain.merged_surface`).

At Athena's Sun that is a real gap. The DTM's slopes on the 512 px crop have a median of 2.8 degrees and a 90th percentile of 5.7, as large as the Sun elevation in the eight frames (3.3 to 4.3 degrees), and at a 3 to 4 degree Sun a 1 degree tilt moves a facet's brightness by about 30%. So the question was whether starting from the DEM's macro shape, and letting shape from shading add only the detail, leaves less noise for the detector.

## How it was measured

On the f03 stack (8 frames, 0.9 m pixels), each correction was scored by the residual the detector's own null leaves: a static field plus a quadratic per frame, removed in 24 px patches, with the per-frame estimator of `noise_scale`. Patches are classed by DEM slope and are the same for every correction; pixels a model leaves uncorrected keep their shading and count. The bundle carries the DTM's slopes on the image grid, not its heights, so the DEM surface is their least-squares integral (its slopes come back to 0.002 m/m rms). Every solver ran at the workstation campaign's settings (1 px grid, smoothness 1, two passes for the linearised solver), the settings the existing correction was made with.

| Correction | Flat, below 3 degrees | 3 to 6 degrees | Steeper than 6 degrees |
|---|---|---|---|
| None | 0.139 | 0.148 | 0.181 |
| Shape from shading from flat ground (current) | 0.090 | 0.092 | 0.114 |
| DEM shading subtracted, then shape from shading | 0.091 | 0.094 | 0.103 |
| The same with the DEM smoothed to 9 m first | 0.091 | 0.095 | 0.102 |
| Gauss-Newton shape from shading from flat ground | 0.092 | 0.094 | 0.115 |
| Gauss-Newton started on the DEM | 0.095 | 0.098 | 0.104 |
| DEM shading alone, subtracted | 0.140 | 0.148 | 0.176 |

47, 39 and 19 patches. The coarse settings the first runs used (2 px grid, smoothness 3, one pass) leave 0.095, 0.097 and 0.120 without any DEM, so settings must match before corrections are compared; an earlier version of this table did not, and understated the DEM.

## What it shows

- Subtracting the DEM's predicted shading before shape from shading cuts the residual on ground steeper than 6 degrees by 10% and leaves flatter ground within 2%. Steep ground is where the residual was largest, and where the detector, scaled by the flat-ground noise, overstates scores by about a fifth.
- The DEM's shading alone removes almost nothing. Inside each detector window the macro shading is smooth, and the null's quadratic per frame already absorbs it.
- What remains after the correction is metre-scale structure, finer than the DTM's 4 m posting: neighbouring pixels of the residual correlate at 0.61, and its variance in 2 and 4 px blocks is 2.5 and 3.6 times what independent noise would give. Only shape from shading resolves that.
- Dividing by a predicted shading instead of subtracting it scales noise up wherever the model darkens a frame, and made every DEM variant look worse than it is.
- The Gauss-Newton solver gains on steep ground from a DEM start but loses a little elsewhere; the linearised solver on the DEM-subtracted images does as well on steep ground without that loss.

## What was built from it

`sfs_dem_prior` (T14) and `dem_prior` in the landing maps' relief settings: each frame's Lunar-Lambert shading on the DEM, relative to its mean over frames (`sfs.dem_shading_ratio`), is subtracted from the stack before shape from shading (`sfs.subtract_shading`), for the original solve and every planted-rock solve alike. Sizing still casts shadows on the solved relief, each window's plane set to the DEM tilt as before (why, below). The landing maps' merged surface already takes the DEM below its posting. The repository functions reproduce the experiment's corrected stack exactly. On a flat DEM the option changes nothing.

T14 reports `residual_by_slope`, the residual before and after the correction in the three slope classes above (`noise_scale.residual_by_slope`), and HATI Watch draws it, so every run shows what the correction does on slopes.

The noise also varies between frames, from 0.056 to 0.137 after the correction, while the detector weighted all frames alike; weighting each by its own noise is worth about a third more information on every fit. `RegistrationProjector` now takes unequal frame noise. Its displacement covariance acts on two image modes that are orthogonal to the brightness planes, so the whitened frame k scales mode i by 1/sqrt(1 + (s g_i / sigma_k)^2) and leaves the rest of the patch at 1/sigma_k, and the static albedo is projected out separately along each; this equals the full weighted least-squares projection (a test compares it with a dense Cholesky solution). Equal noise keeps the original arithmetic, so existing results do not move. `fit_patch` and `assess_regions` take one noise scale or one per frame, and with `noise_per_frame` T14 runs the detector, scores the planted rocks and sizes at each frame's measured residual scale (floored at half the pooled value).

Both options are off by default in the code.

## What planted rocks say about them

The laptop sizing bench planted the same 168 rocks (eight rounds of the T14 population on the 8-frame Athena stack) three times: as HATI was, with the DEM prior, and with the DEM prior and per-frame noise; sizing ran with the context guard, edge padding and the fourth window. A development run, not a study.

| | As before | DEM prior | DEM prior and per-frame noise |
|---|---|---|---|
| Found, 0.3 to 0.6 m | 19 of 55 | 17 of 53 | 21 of 54 |
| Found, 0.6 to 1.2 m and 1.2 to 2 m | 47 of 47, 30 of 31 | 46 of 46, 30 of 31 | 46 of 46, 29 of 30 |
| Context-supported estimates | 52 | 62 | 61 |
| Median error of the estimates, log of estimate over truth | 0.085 | 0.069 | 0.070 |
| Calibrated 90% estimate interval, high over low factor | 1.81 | 1.50 | 1.47 |
| Bound margin at 0.97 | 0.21 m | 0.25 m | 0.19 m |
| Fitted bounds holding, on the 102 rocks all three bounded | 82 | 84 | 86 |

The prior's gain is in sizing: more estimates, a fifth less error, a calibrated interval a sixth narrower. Per-frame noise adds detection: median scores of planted rocks rose 10 to 14% and four more 0.3 to 0.6 m rocks were found, while the empty ground at the planting sites warned at 8 of 168 sites against 7, so false warnings barely moved; its sizing matched the prior alone (51 of 57 shared estimates identical). The 0.97 bound margin is set by the few worst overshoots at this size, so its differences say little until a run has a hundred or more bounded rocks.

Under the prior, shadows were cast on the solved relief alone, as without it. Adding the DEM surface under them (the integral of the bundle's slopes, `sfs.integrate_slopes`) made five more of the first 31 fitted bounds overshoot: planted shadows are drawn on the DEM plane at their site, so the DEM's own curvature is not something they have.

`configs/saturation_campaign_v26_workstation.json` turns both on.

The scripts are development tools outside the repository (`dem_sfs_experiment.py`, `fine_sfs_check.py`, `additive_check.py`, and the sizing bench with its `--dem-prior`, `--frame-noise` and `--watch` options).
