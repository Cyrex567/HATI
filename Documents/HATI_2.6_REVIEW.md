# HATI 2.6 review: detection floor, sizing bias and a sweep classifier

Branch `feat/v2.6-classifier`, 30 September 2026. Every number below comes from code on this branch or from the repository's own reports. Synthetic results are labelled synthetic. Literature was found with the Consensus search engine; each claim from a paper cites it, and the reference list at the end links every paper.

## The short version

The engineering is in good shape. Modules are separated, missing data stays missing instead of turning into "safe", campaigns resume and hash their inputs, and there are 138 unit tests plus four script checks. Two of those tests failed on a fresh machine for a reason that has nothing to do with the science (matplotlib builds a font cache by calling an external program, which the tests forbid). That is fixed here.

The detection floor on real data is not set by noise. On the Athena stack 89.1% of assessed cells cross the score threshold of 8, and 84 to 86% still cross it when the Sun geometry is deliberately reassigned between frames. On clean synthetic ground the same detector should find isolated 0.1 m rocks at the assumed noise: they scored 6 to 9 at twice that noise, and the signal part of the score scales as one over the noise. The gap between those two facts is structure the null model does not describe: metre-scale relief, registration error and correlated noise. Anything that models that structure will lower the real floor more than anything that suppresses white noise.

The sizes come out too large for reasons I could reproduce one at a time, and the height model itself is not one of them: on clean flat ground it runs 0 to 14% low. The T14 path, which sizes casters on the shape-from-shading corrected images and casts their shadows onto the shape-from-shading surface, returned heights 13 to 61% too high, and its "lower bound" sat above the true height in 4 of 6 trials for both 0.3 m and 0.6 m rocks. That is the same 4 of 6 the commit history reports for injected 0.6 m rocks. Ground that falls away from the Sun by 1 degree more than the DEM says adds 15 to 22%. Width is the weakest dimension: blur, misregistration or a neighbouring rock that the model does not know about roughly triples it. And cells with no rock in them get sized anyway.

A physics-based classifier for boulders, hummocks, craters and other features can be built from the sweep, within limits set by resolution and Sun geometry. A first version is on this branch (`src/hati_core/sweep_classifier.py`, campaign stage T18). <<CLASSIFIER_SUMMARY>> These are synthetic error rates. A classifier that deserves the word clinical needs independently labelled lunar terrain, and HATI does not have that yet.

No remote-sensing detector can give absolute evidence. What HATI can give is an evidence chain for every call, with error rates fixed before the data are looked at. Section 6 describes that chain and which parts of it exist.

## 1. What I did

I read the core package (`src/hati_core/`, about 3,300 lines), the campaign scripts that drive it, the documentation in `Documents/` and the test suite. I ran the whole suite in a clean Python 3.11 environment with the science requirements installed. I ran new controlled experiments on the sizing path and on a new classifier, all with the repository's independent renderer (`relief_scenes.py`) so that the detector never sees scenes drawn by its own templates. I searched the peer-reviewed literature with Consensus for crater and boulder detection, noise and denoising, shape from shading, registration, shadow-height error, detection bias near a threshold, rejection-option classifiers, lunar surface textures and independent truth sources.

Nothing here touched ISIS, downloaded imagery or ran on the real Athena stack. The real-data stages (T18, the post-landing check) are wired up and tested on a small synthetic bundle, and they need the workstation.

## 2. The repository today

### 2.1 The pipeline

Ingestion (ISIS, WSL only) produces an aligned 8-frame NAC stack at 0.9 m, DEM visibility per frame and receiving slopes from the 4 m NOBILE03 DTM. Three maps come out of `scripts/landing_maps.py`: a terrain map from native DEM plane fits, a shadow map from the regional likelihood search, and a conservative fusion. The shadow search (`src/hati_core/regional_shadow.py`, `shadow_likelihood.py`) fits a nonnegative rectangular cast-shadow template at every root in every 4-pixel cell, after removing a static image and a brightness plane per frame from both data and templates, and scores the improvement in chi-square. A score of 8 is a warning. The saturation campaign (T1 to T16) then asks where those warnings come from: frame ablations, geometry reassignment, width and height profiles, registration, adaptive context and sizing (T9, T14), relief versus rock competition by withheld-frame prediction (T13), shape from shading as a structural null (T14), residual noise (T12) and no-caster nulls (T16). The post-landing check reruns all of it on frames taken after the landing.

### 2.2 Tests

| Test file | Tests | Result before | Result after |
|---|---:|---|---|
| test_adaptive_campaign | 2 | 1 failure on a fresh machine | pass |
| test_adaptive_shadow | 11, now 13 | pass | pass (2 new slope-search tests) |
| test_audit_regressions | 15 | pass | pass |
| test_channels, test_coreg, test_footprint, test_kinematics | script checks | pass | pass |
| test_landing_maps | 20 | 1 failure on a fresh machine | pass |
| test_live_watch | 11 | pass | pass |
| test_noise_scale | 9 | pass | pass |
| test_post_landing_check | 3 | pass | pass |
| test_relief | 19 | pass | pass |
| test_rock_scenes | 5 | pass | pass |
| test_saturation_campaign | 12 | pass | pass |
| test_scene_diagnostics | 10 | pass | pass |
| test_shadow_diagnostics | 7 | pass | pass |
| test_shadow_likelihood | 14 | pass | pass |
| test_sweep_classifier (new) | 15 | not present | pass |

The whole suite takes about two minutes here (four cores, 15 GB). I confirmed the fix by deleting the matplotlib cache before running each of the two tests.

### 2.3 Findings

1. Two tests failed on any machine without a matplotlib font cache. `tests/test_adaptive_campaign.py` and `tests/test_landing_maps.py` replace `subprocess.run` with a function that raises, to prove that no external program such as ISIS is called. matplotlib's first import calls `fc-list` through `subprocess` to build its font cache, so the guard fired. On the workstation the cache already exists, which is why the campaign never saw it. Both files now build the cache at import, before the guard is installed.

2. Registration error is probably larger than the detector assumes. The production run models 0.5 px of registration scatter (`registration_sigma_px`), while the same run's per-frame co-registration closure was 0.22 to 2.77 px (median over the pairs each frame belongs to, in the evidence snapshot). Closure between frames lit from very different directions also contains illumination-driven correlation bias, so it is an upper bound on registration error rather than a measurement of it. Even so, in my tests 1 px of unmodelled jitter tripled fitted shadow widths (section 4).

3. The T14 height "lower bound" is not a bound. `_caster_row` in `scripts/relief_experiments.py` takes the lower end of the set of heights whose fit is within a chi-square difference of 4 of the best. That set is only a confidence region if the template model is exactly right; with any systematic error it shrinks around the wrong value as the signal gets stronger. When the best template is censored by the fitting window, the bound switches to the window reach times the tangent of the lowest Sun, and whether the shadow is censored is itself decided by the best template. A template that is too long therefore produces a large "geometric lower bound" that the data never established.

4. The receiving slope is taken as exact. `PLAN_BREAKTHROUGH.md` (U3) proposes solving for height and local slope together from the sweep, because a shadow lengthens downslope and shortens upslope. The adaptive sizer never did this; it uses the DEM slope at the cell centre. At 3.5 degrees of Sun a 1 degree slope error moves the height by about 29%. This branch adds the joint fit as an option, and section 4.4 shows why it does not work with the Athena sweep's narrow range of Sun elevations.

5. There is no continuous integration. The software checks run only inside the campaign runner on the workstation, so a failure like finding 1 is discovered at the start of a multi-hour run. A GitHub Actions job running the test scripts on every push would catch it earlier. I did not add one, since it would start running jobs on your GitHub account without you asking.

6. Version and citation metadata were stale. The core said 2.5.5 and is now 2.6.0. `CITATION.cff` still says 1.5.0, released 31 January 2026, and pointed at `github.com/Cyrex567/YOUR_REPO_NAME`; I fixed the URL and left version and date for you to set when you tag 2.6. The README leads into the v1.5 history, including an 85% combined safety score and 99.5% boulder precision that the current path does not support. The README says so, but a reader skimming it could miss that.

7. Scripts that patch the old v1.5 notebook (`scripts/apply_fixes.py`, `scripts/apply_gpu_optimizations.py`) and superseded controls (`scripts/mare_control.py`, `mare_control_v2.py`) sit beside the current science scripts. Moving them to an archive folder would make the current path easier to audit. I left them where they are.

## 3. Pushing the detection floor down

### 3.1 Where the floor comes from

For one cell the detector fits a nonnegative shadow amplitude c to data that has passed through the null projection. If t is the projected template in noise units, the score is close to c times the norm of t, and the squared norm of t is a sum over frames and pixels of how much the shadow changes the image after the static part is removed. A cell warns at a score of 8. So four things set the floor: the shadow contrast, how many pixel-frames the shadow changes (longer shadows, more frames and more varied Sun directions all help), the noise in the denominator, and the threshold.

On clean synthetic ground with the eight Athena frames, the measured floor is low. At noise sigma 0.1, 40% of 0.2 m rocks and 80% of 0.25 m rocks were detected (40 seeds each). At sigma 0.06 a 0.1 m rock scored between 6.0 and 9.2. The signal part of the score scales as one over sigma, so at the assumed 0.03 an isolated 0.1 m rock on level ground should score roughly 12 to 18, well over the threshold. On the real stack the floor is set by structure, since nearly every cell already warns. That changes which approaches pay off.

### 3.2 Options, in the order I would try them

| Approach | What it acts on | Evidence | Effort | Where |
|---|---|---|---|---|
| Model relief inside the likelihood instead of subtracting shape from shading from the data | Structure the null misses | At the south pole, multi-image SfS with shadow constraints reconstructs relief pixel by pixel [@Jia2024] and adaptive weighting recovers crater depths [@Ye2025]; photometric-stereo error depends on the spread of Sun azimuth and zenith [@Liu2018]. Section 4 shows that subtracting SfS from the data inflates sizes | Medium | `RegistrationProjector` already takes extra nuisance columns (the albedo-gain option); an SfS shading basis per cell can go in the same way, as the T14 absorption rule in `relief_experiments.py` already suggests |
| Better registration and a measured registration uncertainty per frame and tile | Misregistration that looks like change | Phase-correlation decomposition keeps about 0.1 px accuracy under large illumination change [@Wan2019]; robust M-estimators give about tenfold better precision with local illumination change [@Fouad2012]; moving-window phase correlation with quality filtering reaches about 0.3 px [@Scheffler2017]; block bundle adjustment of south-pole NAC images with a DEM constraint brings image inconsistency below 0.5 px [@Chen2024] | Medium | `scripts/ingest_sweep.py`; feed per-tile sigma into `RegistrationProjector` instead of one scalar |
| More frames and more Sun geometry | Norm of t | Multi-illumination NAC crater counting finds 3 m craters with a uniform threshold by matching across images [@Cadogan2020]; slope error depends on the difference in Sun azimuth and zenith between images [@Liu2018] | Low (data exist) | The 24-frame post-landing plan spans 135 degrees of azimuth and 1.6 to 5.4 degrees of elevation against 75 degrees and 1.3x in cot(e) for the nine-frame plan. Oblique frames need orthorectification first |
| A real noise model | The denominator | The NAC is linear only from about 600 DN and needs a signal-dependent correction below it; the flight calibration names stray light as important for shadowed craters and spatial crosstalk for objects about 1 m across [@Tschimmel2016] | Low to medium | Use T12's per-frame sigma instead of one 0.03; weight pixels by a shot-noise model (dark shadow pixels are quieter than lit ones); whiten spatially correlated noise before the fit |
| Use the rock's lit face | Contrast | At 3 to 4 degrees of Sun, a rock face tilted 45 degrees toward the Sun is about 10 times brighter than level ground in the Lunar-Lambert model the renderer uses (L = 0.5); real photometry differs, so take that as an order of magnitude | Low | The detector only darkens. The classifier's boulder hypothesis already has a lit-face term |
| Require agreement between frames | False alarms from single frames | Matching across images removes false positives and recovers missed craters [@Cadogan2020] | Low | `frame_delta_chi2` is already stored per candidate |
| Calibrate the threshold instead of fixing it at 8 | The threshold | Injecting artificial sources measures completeness and corrects flux boosting [@Bethermin2010]; conformal prediction with a reject option gives distribution-free error rates, shown so far for binary classification [@HallbergSzabadvary2025] | Medium | `gaussian_search_calibration` already reruns the full search under white noise; do the same on real blank backgrounds |
| Deep-learning denoising or super-resolution | Visual quality, not the likelihood | A denoiser built on a physical NAC noise model and real noise samples outperforms the standard calibration on permanently shadowed images [@Moseley2021], and deep-learning post-processing revealed boulders and craters down to 3 m inside those regions [@Bickel2021]; multi-image restoration reached 5 cm from 25 cm HiRISE images [@GPT2017]; single-image super-resolution can invent structure [@Jyhne2025] | High | Only as a separate visual product, or rerun every injection test through it. It changes the noise statistics the score assumes |
| Independent truth | Validation, not detection | OHRC images at about 0.25 to 0.3 m have resolved boulders under 1 m around a young crater [@Dagar2022] and craters of about 1.8 m inside the permanently shadowed part of Cabeus [@Dagar2023]; a 2026 report (no journal listed in the search result) describes an open ISIS and Ames Stereo Pipeline workflow for 0.3 m OHRC DEMs [@Tungathurthi2026] | Depends on coverage | I could not check whether OHRC imaged the Athena site. If it did, T8 stops being blocked |

Two notes on the table. The denoising and super-resolution rows are there because they came up in the search, not because I recommend them for the detector. And Diviner rock abundance, the other independent check (T4), works at hundreds of metres and underestimates where small rocks dominate [@Powell2023] [@Amaro2026]; the standard product I found covers 60 degrees north to 60 degrees south [@Bandfield2011], so whether it exists at 84.8 degrees south needs checking.

### 3.3 Craters specifically

HATI's shadow detector is built for casters. Craters enter only as relief in T13 or through the DEM terrain map. At 3 to 4 degrees of Sun that misses the most visible thing about a small fresh crater: most of its interior is in shadow. A bowl with a depth-to-diameter ratio of 0.1 has walls up to 22 degrees, so its up-Sun rim shades most of the floor. Small craters at landing scale are often much shallower, though: 0.061 on average at the Chang'e-4 site [@Wu2020], and the ESA lunar lander study expected slopes at the scale of a lander footprint near the south pole to come mostly from mature, shallow craters with slopes of about 11 degrees [@DeRosa2012].

Four routes, roughly from cheapest to most work:

- The crater hypothesis on this branch. A paraboloid bowl with a closed-form interior shadow boundary, dark on the up-Sun side and bright on the down-Sun wall, checked against the repository's numerical horizon tracer (agreement everywhere except within 10 cm of the shadow edge). Its shadow pivot moves around the rim as the Sun azimuth changes, which is the opposite of a boulder, whose shadows all start at one point.
- Single-image highlight-and-shadow operators run on each frame, then required to agree across frames. Path-profile operators detect craters from 5 pixels [@Wang2022], and path-set detectors reached a detection rate of 0.89 below 5 m at 0.5 to 2 m per pixel [@Wang2024].
- Multi-image matching, which found craters down to 3 m and gave a uniform detection threshold [@Cadogan2020].
- Depressions in the shape-from-shading surface. SfS refinement can bring the effective resolution of LOLA and NAC DEMs close to 1 m per pixel [@Boatwright2024], and adaptive weighting recovers crater depths better than plain multi-image SfS [@Ye2025]. T14 already produces the surface; a flood-fill depression finder on it is a small addition.

Convolutional detectors are an option for larger craters, but one trained on NAC data overestimated crater diameters by 15% [@Fairweather2022], which is worth remembering for the next section.

## 4. Why the sizes come out too large

### 4.1 The experiment

`scripts/sizing_bias_experiments.py` renders one rock of known height (0.3, 0.6 or 1.2 m; 0.6 m wide, 0.81 m long) with the independent renderer under the eight Athena Sun geometries, adds the renderer's own noise mix at sigma 0.03, and sizes the centre cell with the adaptive refinement T14 uses (`refine_cell`, scales 1, 2 and 4, quadratic null), then applies T14's own lower-bound rule. Each condition changes one thing against the clean scene. Noise draws are shared across conditions for the same seed, so the comparisons are paired. Six seeds per condition and height; 40 seeds per height for the near-floor test. The run took 25 minutes on four cores.

### 4.2 Results

| Condition | Height, fitted / true (0.3, 0.6, 1.2 m rock) | T14 lower bound above truth (0.3, 0.6, 1.2 m) | Median fitted width (m) |
|---|---|---|---|
| Clean: flat ground, production blur and registration | 1.00, 0.86, 0.90 | 0/6, 0/6, 0/6 | 0.65, 0.70, 0.70 |
| 0.8 px more optical or resampling blur than the model assumes | 1.00, 0.89, 0.90 | 0/6, 0/6, 0/6 | 2.05, 2.05, 2.00 |
| 1 px registration jitter (model assumes 0.5 px) | 0.89, 0.86, 0.89 | 0/6, 0/6, 0/6 | 2.30, 2.35, 2.40 |
| Ripples, 6 m wavelength, 1 deg slopes, DEM says flat | 1.00, 0.97, 0.96 | 1/6, 1/6, 0/6 | 0.60, 0.70, 0.70 |
| The same ripples at 3 deg | 1.28, 0.97, 1.06 | 2/6, 3/6, 0/6 | 2.35, 1.55, 0.70 |
| Ground falls 1 deg along the mean shadow direction, DEM says flat | 1.22, 1.17, 1.15 | 4/6, 5/6, 0/6 | 0.70, 0.40, 0.50 |
| Ground rises 1 deg along the mean shadow direction, DEM says flat | 0.72, 0.69, 0.75 | 0/6, 0/6, 0/6 | 0.80, 0.70, 0.75 |
| Second rock of the same height 2.5 m down-Sun | 1.00, 0.94, 0.92 | 0/6, 0/6, 0/6 | 2.25, 2.25, 2.30 |
| Target inside a field of 0.2 to 1 m rocks covering 3% | 1.11, 1.19, 0.99 | 2/6, 1/6, 0/6 | 0.75, 0.60, 0.60 |
| Flat ground, then the T14 shape-from-shading correction | 1.00, 0.97, not run | 0/6, 0/6, not run | 0.30, 0.40, not run |
| 3 deg ripples, T14 path: corrected images and SfS surface | 1.61, 1.28, 1.13 | 4/6, 4/6, 0/6 | 0.40, 0.40, 0.40 |
| 3 deg ripples, corrected images on the DEM plane | 1.67, 1.28, not run | 3/6, 4/6, not run | 0.40, 0.40, not run |
| 3 deg ripples, original images on the SfS surface | 1.33, 0.97, not run | 2/6, 2/6, not run | 2.35, 0.70, not run |
| 3 deg ripples and no rock | 6 of 6 cells warn; median fitted height 0.20 m | all 6 bounds describe a rock that is not there | 2.40 |
| Boulder field, no rock at the cell centre | 3 of 6 cells warn; median fitted height 0.30 m | the 5 bounds describe neighbouring field rocks or their merged shadows, not a rock in the cell | 0.70 |

Heights are the final adaptive pass of cells that kept a warning score; "lower bound above truth" counts trials whose T14 bound exceeded the true height by more than 5 cm. Six seeds per row and height, so a single trial is 17 percentage points: read the rows as directions and rough sizes, not precise rates. `output/sizing_bias/sizing_bias.png` on the workstation, or the copy in `Documents/assets/v26_sizing_bias.png`, shows the same results.

Near the floor (fixed 5 cm height grid, first scale, noise sigma 0.1, 40 seeds per height):

| True height | Detected | Mean fitted height, all fits | Mean fitted height, detected only |
|---|---|---|---|
| 0.15 m | 5% | 0.144 m | 0.150 m |
| 0.20 m | 40% | 0.174 m | 0.188 m |
| 0.25 m | 80% | 0.219 m | 0.220 m |
| 0.30 m | 95% | 0.274 m | 0.272 m |
| 0.40 m | 100% | 0.426 m | 0.426 m |

### 4.3 What is going on

The height model itself is not the problem. On clean flat ground the fitted height runs 0 to 14% low, and T6 found the same direction (median errors of -0.05, -0.075 and -0.30 m for 0.3, 0.6 and 1.2 m casters). A rounded rock casts a tapered shadow, and a rectangle fitted to a shadow that fades toward its tip comes out short. So something has to push the estimate up, and it has to push hard.

The correction step in T14 is the largest upward push I found. Splitting it shows which half does it. Sizing the corrected images on the plain DEM plane gives 1.67 and 1.28, the same as the full T14 path (1.61 and 1.28). Sizing the original images on the shape-from-shading surface gives 1.33 and 0.97, the same as not correcting at all (1.28 and 0.97). On flat ground the correction is harmless (1.00 and 0.97). So the corrected images are the cause, the surface is not, and the trouble only appears when there is relief to correct. I measured how the solver treats a 0.6 m rock: on both flat and rippled ground it keeps about a quarter of the rock's shadowed pixel-frames as slope data (0.28 and 0.24), and after correction about 90% of the rock's amplitude survives, but roughly two thirds of the rock signal is moved around (0.66 and 0.69 of its absolute value). My reading is that the rock bends the local surface solution, so the relief around the rock is mis-corrected and the mis-correction reads as extra shadow length. I have not proved that last step; the split above is the measured part.

The receiving slope matters more than anything inside the shadow model. Shadow length is h / (tan e + beta), with beta the slope along the shadow. At 3.5 degrees of Sun tan e is 0.061 and a 1 degree slope is 0.017, so a 1 degree error moves the height by about 29%. The tilt tests bracket that: +15 to +22% when the ground falls away from the Sun more than the DEM says, -25 to -31% when it rises. Detection adds a lean: rocks on ground that falls away cast longer shadows and cross the threshold more easily, so detected rocks are drawn more from the overestimated side. For scale, the improved 5 m LOLA DEMs of south-pole landing sites carry slope uncertainties of 1.5 to 2.5 degrees RMS [@Barker2020]. I do not know the equivalent figure for NOBILE03 at the scale of one rock, and it needs measuring before any height is trusted to better than about a factor of 1.5.

Relief finer than the DEM acts the same way locally: 3 degree ripples added 28% to a 0.3 m rock; 1 degree ripples changed nothing measurable.

Width absorbs whatever the model does not know about. These rocks are 0.6 m wide, less than one 0.9 m pixel, so the width comes from the blurred profile across the shadow. Give the model 0.8 px less blur than the data have, or half the real registration scatter, or put a second rock beside the shadow, and the fitted width jumps from 0.65 to 0.7 m to between 2.0 and 2.4 m, the top of the template bank. If "size" in your results means plan-view extent, this is the mechanism. The real stack is resampled twice (map projection and co-registration) and its per-frame closure reaches 2.8 px, so I would expect inflated widths there.

Cells with no rock get sized. With 3 degree ripples and no rock, all six centre cells warned and were given a height of about 0.2 m and the maximum width. In a boulder field with no rock at the centre, half the cells warned and were sized at about 0.3 m from their neighbours' shadows. On a saturated map a large share of sized cells could be like this. T13 sends relief-like cells elsewhere, but ambiguous cells are still sized.

Near the floor, selection mattered less than I expected. The rocks that crossed the threshold had a slightly larger mean fitted height than all fits of the same rock (0.188 against 0.174 m at 0.2 m, where 40% were detected), which is the Eddington effect seen in astronomical source counts, where fluxes near the detection limit are inflated by positive noise fluctuations [@Massardi2008] [@Wang2004]. Here it is smaller than the estimator's own low bias: weighting the detections by a steep size distribution (cumulative exponent 3, an assumed value) still leaves detected rocks at 0.93 of their true height on average. Selection becomes a real problem when it combines with one of the upward pushes above, slopes especially.

The compatible range is not a confidence interval. It contained the true height in anywhere from 0 to 6 of 6 trials depending on the condition, and in 0 of 6 for clean 1.2 m rocks. The repository already says the range is descriptive; these numbers show how quickly it stops working as a bound once any bias is present.

Other automated detectors show the same tendency at larger scales, for what it is worth. A CNN crater detector on NAC images overestimated crater diameters by 15% [@Fairweather2022]. At the Mars Orbiter Camera resolution limit, dark pixels taken for boulders turned out in HiRISE to be mostly the shadows of clusters of smaller rocks [@Golombek2008], which is the width and neighbour mechanism above, one scale up. As rocks approach the pixel size, fewer and fewer are detected at all [@Golombek2022].

### 4.4 What to change

In order of how much I expect each to remove:

1. Stop sizing on the shape-from-shading corrected images. The split above shows that is where the extra height comes from. Size on the original images and carry relief as a nuisance that both hypotheses see, which is the rule T14's own absorption verdict already states, rather than subtracting it from the data. Keep the SfS surface for what it does well: terrain slopes and depressions.
2. Constrain the receiving slope, but not by fitting it from the Athena sweep alone. I added the joint fit that `PLAN_BREAKTHROUGH.md` calls U3: `AdaptiveConfig(slope_search_deg=(-1, 0, 1))` profiles a grid of plane tilts around the DEM slope together with height and width, and records the fitted tilt and its compatible range. On the matched renderer it recovers a 2 degree tilt exactly (a unit test). On independently rendered rocks under the Athena sweep it did not help. With a 3 by 3 grid of plus or minus 1 degree, 0.3 m rocks on ground tilted 1 degree away from the Sun came out at 1.33 of their height (1.22 with the plane fixed) and on ground tilted toward it at 0.67 (0.78 fixed); 0.6 m rocks were unchanged or slightly worse, and on flat ground the search only added scatter (three seeds each, scales 1 and 2). The cause is identifiability, not the code. Height and the along-Sun slope enter as h / (tan e_k + beta . d_k), and when every shadow lies within about 53 degrees of one direction and tan e varies only 1.3 times across frames, a taller rock and a steeper downslope look almost the same, so the extra freedom soaks up model mismatch instead. <<POST_SLOPE>> Until a test like that passes on real data, take the slope from an independent measurement (the SfS slope at the root on the original images, or a better DEM) and carry its uncertainty into the height interval: at 3.5 degrees of Sun, sigma_h / h is about sigma_beta / (tan e + beta). The option is off by default, leaves hashes and behaviour unchanged when off, and multiplies template renders by the grid size when on; no preset turns it on.
3. Treat width as an upper bound until the stack's blur and registration are measured. Measure the effective PSF per frame from sharp natural edges (on Earth, modelling the illumination gradient across the shadow edge brought shadow-derived heights of about 20 m to within 6% [@RadaGiacaman2022]), give `RegistrationProjector` measured per-tile registration scatter instead of one 0.5 px, and let the PSF width be a fitted nuisance.
4. Replace the compatible range with an interval calibrated by injection. T14 already plants rendered rocks into the real images. Record fitted against true height at many sites and backgrounds and build the interval and the bias correction from that distribution, as source-extraction pipelines do with artificial sources [@Bethermin2010]. Near the floor, deboost with a size-frequency prior [@Coppin2004].
5. Decide censoring from the shortest compatible fit, not the best one. Done on this branch: `fit_patch` reports `lowest_compatible_censored`, and `_caster_row` uses it, so the geometric bound applies only when every compatible explanation runs past the window.
6. Size only cells the classifier calls boulder with its margin met, and publish every size with the evidence chain in section 6.

## 5. A sweep classifier for boulders, hummocks, craters and other features

### 5.1 What the sweep can tell apart

A single low-Sun image cannot reliably separate a small boulder from a small crater: both are a dark patch next to a bright one. The sweep can, because the three physical cases respond differently when the Sun moves.

Along the Sun direction, going from the Sun side to the far side, a boulder is bright then dark, and its dark streak is much longer than the rock and grows as cot(e). A hummock is also bright then dark, but the pattern stays within its own footprint until its flanks are steeper than the Sun, and only then does it cast a shadow. A crater is the reverse: dark then bright, and its shadow never leaves the bowl. When the Sun azimuth changes, a boulder's shadows all rotate about one fixed point, while a crater's shadow starts wherever the rim faces the Sun, so its starting point walks around the rim. Elephant-hide texture and ripples respond to the Sun like relief but have no single centre; the texture is nearly everywhere on slopes steeper than about 5 to 8 degrees and its apparent pattern changes with illumination direction [@Bondarenko2022], and near the south pole it also appears on near-level ground at the foot of slopes [@Brown2026]. Albedo changes do not follow the Sun at all, and registration error follows image gradients rather than the Sun.

The sweep's limits come straight from this. The Athena pre-landing stack spans about 107 degrees of azimuth but only 3.3 to 4.3 degrees of elevation, so the cot(e) test is weak (a ratio of 1.3) and the azimuth tests carry most of the weight. The 24-frame post-landing set is much better on both counts. And a feature smaller than the combined blur of PSF, pixel and registration can only be classified through its shadow or shading pattern, never through its outline.

### 5.2 What is on the branch

`src/hati_core/sweep_classifier.py` competes four generative hypotheses for each cell, all passed through the detector's own null so that none of them can win by describing albedo or a brightness drift:

- boulder: the detector's cast-shadow template plus a lit face, a blurred point half a width toward the Sun;
- hummock: a paraboloid dome with Lambert shading and the attached and cast shadows derived in closed form;
- crater: a paraboloid bowl with Lambert shading and the interior shadow of its up-Sun rim, also closed form;
- extended relief: a free slope at every pixel, shaded to first order by each frame's Sun. This describes any gentle relief, compact or not, but spends two parameters per pixel.

Each compact hypothesis has two nonnegative amplitudes, one for brightening and one for shadow, fitted by an exact box-constrained solver. The comparison uses withheld-frame prediction: each frame is predicted from a fit to the others, and a test on this branch confirms that the withheld frame's intensities never choose the model that predicts them. A call needs the leading hypothesis to beat its rival by a calibrated margin, and the same model with the Sun directions reassigned among frames to predict worse by another calibrated margin. A compact class also has to explain a calibrated share of what the free slope field explains; otherwise the relief is called extended. Anything else is `ambiguous` (with the two leading hypotheses), `no_signal` or `non_solar_change`.

The margins are split-conformal quantiles: calibrated on scenes of known truth against declared error targets, then frozen and applied to new scenes. For scenes like the calibration scenes, each declared error rate then holds in expectation. That guarantee does not reach lunar terrain unlike the calibration scenes, which is the whole difficulty.

The dome and bowl models share no code with the renderer used to test them. The renderer draws Gaussian mounds and bowls with raised rims, with Lunar-Lambert photometry and a numerical horizon sweep; the classifier assumes paraboloids and Lambert photometry. Some of the shape mismatch in the results is deliberate.

Campaign stage T18 (`scripts/classifier_stage.py`) runs the classifier three ways on a real bundle: calibration and held-out test on generated scenes under the stack's own geometry and measured noise; rocks, bowls and mounds planted into the real images at declared sites, counted only where the site's own background was quiet; and every assessed cell near the touchdown plus a random sample fixed by seed. A boulder call carries its height and the lower end of its compatible range, a crater call its diameter and depth, a hummock call its flank slope. None of them clears ground.

### 5.3 Results on held-out synthetic scenes

Setup: the eight Athena Sun geometries, noise sigma 0.03, the classifier's default configuration (33 by 33 pixel window, 12 pixel fitting disc). Calibration used 165 scenes: 3 seeds of each of 35 scene types (rocks of 0.15, 0.3, 0.6 and 1.2 m on flat and on rippled ground; mounds and bowls of 3, 6 and 12 m with 2, 5 and 10 degree slopes; ripples of 3, 6 and 12 m wavelength with 1, 2 and 5 degree slopes) plus 30 blank and 30 stripes scenes. Declared targets: 5% for a blank called signal, a stripes scene passing the Sun check and a boulder or crater call on another truth; 10% for hummock and extended calls on another truth and for ripples called compact. The calibrated margins were: no-signal 0.008, Sun 0.51, Sun share 0.126, compactness 0.37, lead for boulder 0.085, hummock 0, crater 0 and extended 5.57 (withheld-frame gains, squared noise units per pixel). The large extended margin is the price of the deliberate shape mismatch: the free slope field often beats a paraboloid on a Gaussian mound by several units, so an extended call needs a big lead.

First test, original rule, 165 new scenes at the same noise. When the classifier made a call it was right almost every time: boulder 12 of 12 calls, hummock 24 of 25, crater 18 of 19, extended 13 of 13, and all 30 blanks and all 30 stripes scenes were labelled correctly. It made far fewer calls than it should have, though. Only 12 of 24 rocks, 18 of 27 bowls and 13 of 27 ripple scenes got their own class, and most of the misses came back as `non_solar_change`: 7 rocks, 3 mounds and 9 bowls. At twice the noise that became 23 of 24 rocks. That was a flaw in my rule, not in the physics. A weak change that cannot pass the Sun check was being reported as a change that ignores the Sun. I changed the rule so that `non_solar_change` needs the reassigned geometry to explain nearly as much as the measured geometry (a Sun margin below a calibrated share, 0.126, of the gain), and anything weaker is `ambiguous`. Because I made that change after looking at those test scenes, they no longer count as held out for the new rule, so the table below uses new seeds that nobody had looked at.

<<FRESH_TABLES>>

What it can and cannot separate at this noise and geometry, from the per-scene rows:

<<FRESH_SCENES>>

My reading: as calibrated here the classifier is conservative. Its calls are trustworthy on synthetic scenes, and it says ambiguous instead of guessing, which is the behaviour you want in front of a hazard map. But it does not yet classify the features the detector is best at finding, sub-pixel rocks of 0.15 to 0.3 m: their withheld-frame gains are too small to clear the Sun check with eight frames and three reassignments. Four things should move that floor: more frames (the 24-frame post-landing set), a proper permutation Sun test with many reassignments rather than three cyclic shifts, a fitting window matched to small features, and a noise model that is not white. The Sun threshold is also set by the stripes scenes, whose amplitude (0.12 in normalized radiance) I did not tune; stronger non-solar change in the calibration set would raise it. Those are the next experiments, not settled facts.

### 5.4 What "clinical" would take

In medicine a diagnostic test earns trust through sensitivity and specificity measured on patients whose true condition was established independently, on a population like the one it will be used on, with the decision threshold fixed beforehand. The same standard applied here needs:

1. Independent truth at the target scale. OHRC imagery or OHRC DEMs at 0.25 to 0.3 m [@Dagar2022] [@Tungathurthi2026], or a site with a lander or rover survey. Synthetic scenes, however careful, test the model against itself.
2. A calibration population separate from the test population, and both separate from development. T8's manifest already enforces this for annotated scenes; T18 does it for generated scenes.
3. Error rates with intervals, reported per class, including how often the classifier abstains. The confusion tables on this branch carry exact binomial 95% intervals.
4. An abstain class that is used. Conformal prediction with a reject option is the formal version of this [@HallbergSzabadvary2025], and cost-based and coverage-based rejection rules lead to the same optimal strategy [@JMLR2021]; selective risk control can be extended to distribution shift [@Bai2026].
5. A declared shift test. The double-noise test on this branch is a small example: margins calibrated at one noise level, applied at twice that. Real shifts are larger: other sites, other Sun geometries, other regolith.

My view: the physics is sound enough to build on, and the evidence so far is synthetic. The classifier should stay a research label (as T13's classes are) until one independently labelled polar scene exists.

## 6. Sub-pixel hazards with supporting evidence

"Absolute" evidence is not available to any orbital detector. Every call rests on a model of the camera, the Sun and the ground. What a detector can offer is an evidence chain in which each link is a separate, testable claim and the error rates were fixed before the data were examined. For a sub-pixel hazard call from HATI the chain would be:

1. The data support it: enough frames and common pixels, recorded per cell (exists).
2. Something changes with the Sun here: the score against a null with a calibrated false-alarm rate on real blank backgrounds (partly exists; the calibration on real backgrounds is the gap).
3. It follows the measured Sun, not reassigned geometry: the Sun margin (exists in T13 and T18).
4. It is a compact caster, not relief, albedo or misregistration: the classifier call with its margin (new, synthetic calibration only).
5. It predicts frames it was not fitted to (exists in T11, T13 and T18).
6. Its size is bounded with a calibrated interval that accounts for slope, blur and registration uncertainty (not yet: section 4.4).
7. It reappears in independent data: the post-landing frames, which the repository already treats as a held-out set (planned), and ideally OHRC.

Only when all seven hold would I call a sub-pixel hazard "supported". Today links 1, 3 and 5 are in place, 2 and 4 exist in synthetic form, and 6 and 7 are open.

## 7. Tests

### 7.1 What I ran here

All in this container: Python 3.11, numpy 2.4, scipy 1.17, four cores, no ISIS, no real imagery.

| What | Result | Time |
|---|---|---|
| Existing suite, fresh environment | 15 of 17 files passed; `test_adaptive_campaign` and `test_landing_maps` failed on the font-cache guard (finding 1) | about 2 min |
| Same two files after the fix, with the matplotlib cache deleted first | pass (2 and 20 tests) | under 1 min |
| `tests/test_adaptive_shadow.py` with two new slope-search tests | 13 tests pass; the fitted plane recovers a 2 degree tilt exactly and the fixed-plane fit overestimates the height | 2 s |
| `tests/test_sweep_classifier.py` (new): closed-form shadows against the numerical horizon tracer, box-constrained solver against scipy, conformal and binomial arithmetic, decision rules including weak signals, withheld-frame leakage, frame alignment, missing pixels, hazard quantities from full compatible ranges, one scene of each kind, and T18 end to end on a synthetic bundle with external programs forbidden | 15 tests pass | 35 s |
| Whole suite, matplotlib cache deleted first | all 18 files pass: 153 unit tests and 4 script checks; the two classifier tests added afterwards (hazard ranges, weak signals) and the files they touch were rerun and pass, 155 in all | about 4 min while experiments ran |
| Sizing ablation: 12 conditions, 3 heights, 6 seeds, plus 200 near-floor fits | section 4 | 25 min |
| Shape-from-shading split: 3 conditions, 2 heights, 6 seeds | section 4 | 9 min |
| Slope search against a fixed plane, paired: Athena sweep (3 conditions, 2 heights, 3 seeds) and the 24-frame post-landing sweep (3 conditions, 0.3 m, 3 seeds), scales 1 and 2 | section 4.4 | <<SLOPE_TIME>> |
| Classifier calibration and held-out test: <<CLASSIFIER_SCENES>> generated scenes | section 5.3 | <<CLASSIFIER_TIME>> |

Every experiment above is a script on this branch with a `--quick` or `--conditions` option, so it can be rerun and extended.

### 7.2 What to run on the workstation

In this order. The first three need nothing new; the rest need the stationary machine and the bundle it already has.

1. The software checks, which the campaign runner also runs first:

   ```bash
   for t in tests/test_*.py; do python "$t" > /dev/null 2>&1 && echo "ok   $t" || echo "FAIL $t"; done
   ```

2. The sizing experiments at more seeds (CPU only, synthetic):

   ```bash
   python scripts/sizing_bias_experiments.py --output output/sizing_bias --workers 8 --seeds 20
   python scripts/sizing_bias_experiments.py --conditions tilt_away_1deg,tilt_toward_1deg,ripples_3deg,clean \
       --slope-search=-1,-0.5,0,0.5,1 --scales 1,2 --output output/sizing_bias_slope --workers 8
   ```

3. The classifier on generated scenes at more seeds:

   ```bash
   python scripts/classifier_experiments.py --output output/sweep_classifier --workers 8 --seeds 6 --blanks 60
   ```

4. T18 on the real pre-landing bundle, with the stages it depends on:

   ```bash
   CONFIG=configs/saturation_campaign_classifier_workstation.json STAGES=T1,T12,T13,T18 bash scripts/run_noise_campaign_wsl.sh
   ```

   Read `stages/T18/result.json` for the margins and the held-out table, `planted.json` for planted-feature recovery on quiet real backgrounds, and `athena_cells.json` for the declared cells. The planted recovery is the most informative number in the whole run: it is the classifier on real noise, real relief and real registration error, with known truth.

5. T14 again, now that the geometric bound comes from the shortest compatible fit, in a new output folder, and compare `injected_rock_sizing` with `full-03`:

   ```bash
   STAGES=T1,T12,T13,T14 bash scripts/run_noise_campaign_wsl.sh
   ```

6. Both post-landing runs with T18 added:

   ```bash
   STAGES=T1,T12,T13,T14,T18 CONFIG=configs/saturation_campaign_classifier_workstation.json \
       bash scripts/run_post_landing_queue_wsl.sh
   ```

   `post_landing_check.py` compares T13 labels between the two stacks but not T18 labels yet. Adding the T18 labels to the repeat measures is a small change and the most direct test of whether the classes are physical: rocks and craters do not move between March 2025 and the pre-landing frames.

7. Registration: run T7 at the workstation preset and compare its measured offsets with the per-frame closure. If offsets near 1 px are common, set `registration_sigma_px` from them before trusting any width.

8. Look up OHRC coverage of the Athena site (84.79 S, 29.20 E). If it exists, a labelled patch there turns T8 from blocked into the first real test of both detector and classifier.

Tests I would add next, in code: a T14 variant that sizes the original images with the slope search instead of the corrected images; a detector variant with the lit-face term; a test that feeds measured per-tile registration scatter into `RegistrationProjector`; and T18 label repeatability in `post_landing_check.py`.

## 8. Files on this branch

New:

- `src/hati_core/sweep_classifier.py`: the classifier (hypotheses, closed-form dome and bowl shadows, box-constrained fits, withheld-frame comparison, conformal calibration, confusion tables with exact intervals).
- `scripts/classifier_stage.py`: campaign stage T18.
- `scripts/classifier_experiments.py`: calibration and held-out test on generated scenes.
- `scripts/sizing_bias_experiments.py`: the sizing ablation, the shape-from-shading split, the near-floor test and the slope-search comparison.
- `tests/test_sweep_classifier.py`: 15 tests.
- `configs/saturation_campaign_classifier_workstation.json`: the relief preset plus T18 settings.
- `Documents/HATI_2.6_REVIEW.md` (this file), `Documents/v26_results/` (result summaries) and two figures in `Documents/assets/`.

Changed:

- `src/hati_core/adaptive_shadow.py`: optional joint receiving-slope search (`slope_search_deg`, off by default, hash unchanged when off); the shortest compatible fit and whether it is censored are recorded.
- `scripts/relief_experiments.py`: T14's geometric lower bound applies only when the shortest compatible fit is censored.
- `scripts/saturation_experiments.py` and `scripts/run_saturation_campaign.py`: T18 registered, its configuration validated and its findings summarised in the verdict.
- `tests/test_adaptive_campaign.py`, `tests/test_landing_maps.py`: build the matplotlib font cache before forbidding external programs.
- `tests/test_adaptive_shadow.py`: two slope-search tests.
- `src/hati_core/__init__.py`: version 2.6.0. `CITATION.cff`: repository URL. `README.md`, `Documents/SATURATION_CAMPAIGN.md`: 2.6 and T18.

Unchanged on purpose: the production detector, its templates, the score threshold, the landing maps and every existing campaign preset. With the new options off, results match 2.5.5 except for T14's lower bound, which now uses the shortest compatible fit.

## References

<<REFERENCES>>
