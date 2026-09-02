# HATI → breakthrough, and Poseidon → maritime
### Master plan: code audit, scientific upgrades, execution
Gergő Csaba Morvai · June 2026 · v1.0

> **Citation warning.** Every reference below is given from memory (knowledge cutoff
> Jan 2026) and is **unverified**. Author, year, journal and volume are given so each can
> be located; **DOIs are deliberately omitted where I am not certain of them.** Verify every
> citation against the actual paper before it enters any manuscript or grant application.
> Entries are marked **[C]** confident, **[M]** moderately confident, **[V]** must verify.

---

# PART 0 — Code audit: what is actually in the source

I read `src/heatmap/dem_features.py` and `src/heatmap/fusion.py` line by line. Findings
ordered by consequence.

### A0.1 — **Every channel is defined in PIXELS, not metres.** (critical)

`compute_tier1_stack(dem, scale_m, window_px=11, mds_scales_px=(3,5,10,20))`. The
`scale_m` argument is used *only* to convert gradient units. The **window size, the MDS
scales, and the TRI neighbourhood are all pixel quantities**. Worse, `_slope_magnitude`
uses `np.gradient`, a central difference, so slope is *always* measured over a fixed
**2-pixel baseline** no matter what `scale_m` says.

Consequence: the same function call on two DEMs of different posting measures **different
physical quantities**. An 11-px window is 44 m on the 4 m NOBILE03 DEM and 16.5 m on the
1.5 m Apollo 17 DEM. A 2-px slope baseline is 8 m vs 3 m.

**This is the architectural root cause of the cross-site transfer failure**, and it means
the mare-control negative result is partly confounded: we compared texture channels that
were *not measuring the same thing* at each site. My earlier `hati25_absolute_gate.py`
sidestepped this by Gaussian-pre-smoothing to a common effective resolution before taking
gradients — which is exactly why the *physical* channels transferred (0.82–0.84) while the
raw pixel-defined texture channels did not (0.21–0.42).

**This is the single most important finding in the audit, and it is testable:** re-run the
mare control with all channels redefined at fixed physical baselines. Either the texture
channels partially recover (the negative result was substantially an artifact) or they do
not (the negative result is intrinsic and now *properly* established). **Both outcomes are
publishable, and we currently cannot tell which is true.**

### A0.2 — Per-scene normalisation makes the output crop-dependent

`fusion.robust_zscore` takes the median/IQR **of whatever array it is handed**. The heatmap
value of a fixed pixel therefore changes if you crop the tile differently. For a research
figure this is tolerable; for an operational product it is disqualifying. Already known as
"relative ranker", but the *crop-dependence* specifically has not been stated.

### A0.3 — Nodata contaminates the normalisation statistics

Scripts call `fill_nearest()` before feature computation, then dilate a mask afterwards.
But `robust_zscore` computes median and IQR over the **filled** array — invented values bias
the very baseline everything is measured against. Fix: pass a validity mask through the
stack and compute statistics on valid pixels only.

### A0.4 — `iqr_curvature` is a noise amplifier by construction

`_laplacian` is a 4-neighbour discrete Laplacian divided by `scale_m²`. Discrete Laplacians
amplify high-frequency content; on a stereo DEM that content is largely *matching noise*.
This matches the earlier channel-correlation finding that `iqr_curvature` correlates ~0.02–0.09
with everything else — it is measuring noise, not terrain. Fix: compute curvature at an
explicit smoothing scale and subtract the noise-floor contribution.

### A0.5 — Stereo confidence is never propagated

`data/athena/NAC_DTM_NOBILE03_CONF.IMG` is read by exactly one diagnostic script and never
enters the heatmap. The DEM ships with a per-pixel quality map and we throw it away. This is
the largest *unclaimed* asset in the repository.

### A0.6 — Performance: `percentile_filter` dominates

`iqr_slope` and `iqr_curvature` each call `ndi.percentile_filter` twice. This is O(N·k²)
with a sort per window and is the reason full-tile runs take minutes. It is the correct
target for the GPU box (and for any real-time ambition).

### A0.7 — Smaller items

* `FusionResult.contributions` stores `mean(|w·z|)` — a magnitude, **not** the signed
  attribution `w·z/Σ|w|` used in the papers. Two different quantities under one name.
* `fuse()` silently drops channels whose weights are missing (prints a note). Calling with
  a non-default `mds_scales_px` and default weights quietly halves the model.
* `terrain_ruggedness_index` does 8 full-array `ndi.shift` calls where slicing would do.
* `rms_planar_deviation`'s docstring warns about tilted terrain, but local-mean subtraction
  **does** remove a linear trend exactly for a symmetric window (the mean of a plane over a
  centred window equals the centre value). The channel is tilt-invariant; the docstring is
  overly pessimistic and should be corrected rather than the code.

---

# PART I — The breakthrough thesis

### What exists in the field today

| Product | Resolution | Limitation |
|---|---|---|
| LOLA global slope/roughness maps *(Rosenburg 2011 [C], Kreslavsky 2013 [C])* | ~1 km baseline | far coarser than a lander |
| Diviner rock abundance *(Bandfield 2011 [C])* | ~240 m/px | statistical, not per-obstacle |
| NAC stereo DEMs *(Henriksen 2017 [C])* | 2–5 m | blind below ~4× posting |
| Manual NAC boulder counting | 0.5–1 m | not scalable, not reproducible |
| ML boulder detectors (recent) | 0.5–1 m | black box, no error budget, needs labels |

### The gap

**There is no per-metre, physically-calibrated, uncertainty-bearing lunar landing hazard map
that transfers between sites without retraining and carries a stated error budget.** That is
the hole HATI is shaped to fill.

### The claim worth earning

> A physics-derived, uncertainty-quantified, sub-resolution lunar landing hazard map with an
> explicit error budget, transferable across sites without training data, and validated
> against a real landing outcome (IM-2 Athena).

Every clause is a deliverable. Today HATI has: the physics ✅, the sub-resolution channel
(synthetic only) ⚠️, the forensic case ✅, transferability ❌ (confounded by A0.1),
uncertainty ❌ (A0.5), calibration ❌.

**A plan is not a breakthrough.** The breakthrough is earned at exactly one gate: the
real-data sweep run producing an obstacle map with a defensible error budget. Everything in
Part II serves that gate or hardens what follows it.

---

# PART II — The five upgrades

## U1 — Scale-explicit channels *(fixes A0.1; highest value per hour)*

**Rationale.** Roughness is meaningless without a stated baseline; this is the central lesson
of the planetary roughness literature, where every metric is reported *at* a baseline
(Kreslavsky & Head 2000 [C] define differential slope explicitly at baseline pairs; Rosenburg
2011 [C] reports slope at stated baselines; Shannon 1949 [C] bounds what any sampled grid can
represent).

**Method.** Re-specify the API as physical:
```python
compute_tier1_stack(dem, scale_m, window_m=48.0, mds_baselines_m=(8, 16, 32, 64),
                    slope_baseline_m=8.0, valid=mask)
```
Pixel windows derived per-DEM as `round(window_m / scale_m)`; slope computed by Gaussian
pre-smoothing to the requested baseline (as `hati25_absolute_gate.py` already does), never
by bare `np.gradient`. **Refuse** any baseline below ~2× the DEM's effective resolution
rather than silently returning noise.

**Acceptance test.** Same physical baselines on both sites → re-run the mare control.
*Prediction to be tested, not assumed:* texture-channel cross-site AUC moves materially from
0.21–0.42. Report the result either way.

## U2 — Uncertainty propagation *(fixes A0.5)*

**Rationale.** A safety-critical map without error bars is not a safety product. The NAC
stereo DEM ships a per-pixel confidence map (Henriksen 2017 [C]); shadow-derived heights carry
a measurement error dominated by shadow-length quantisation and solar-elevation uncertainty.

**Method.** Propagate two variances to every hazard pixel: (i) DEM elevation variance from
CONF → slope/relief variance by linearised error propagation; (ii) shadow height variance
`σ_h² ≈ (tan e)²σ_L² + (L sec²e)²σ_e²` from `h = L·tan e`. Output **hazard ± CI**, and gate
on the *conservative bound*, not the point estimate.

**Acceptance test.** Every published hazard number carries a CI; the Athena gate decision is
re-stated as a bound.

## U3 — Joint slope-and-height inversion from the solar sweep *(the scientific core)*

**Rationale.** `h = L·tan e` assumes the shadow falls on flat ground. On real terrain the
shadow lengthens downslope and shortens upslope, biasing height directly — this is the
known limitation of single-image shadow measurement and the motivation for multi-illumination
methods (photometric stereo, Woodham 1980 [C]; shape-from-shading, Horn 1975 [M];
photoclinometry for planetary topography, Kirk et al. 2003 [M]; shadow-based shape recovery,
Savarese et al., "Shadow Carving", ICCV 2001 [V]). Lunar reflectance for any photometric term
is Hapke-family (Hapke 1984 [C]; resolved lunar Hapke maps, Sato et al. 2014 [C]).

**Method.** With N illuminations at known (azimuth φ_k, elevation e_k), each detected shadow
gives one equation in the unknowns (h, local slope components p, q). With N ≥ 3 well-spread
azimuths the system is over-determined: solve per-obstacle by least squares for **h and the
local slope jointly**, with residuals giving a per-obstacle goodness-of-fit that doubles as
the albedo-rejection statistic. The existing vote-convergence map becomes the *initialiser*
for this inversion rather than the final product.

**Why this is the novel part.** Shadow-based obstacle detection is old; the specific
contribution is a **deterministic, auditable, over-determined inversion driven by the natural
solar sweep, returning height *and* local slope *and* a per-obstacle residual**, at a polar
site where `cot(e) ≈ 8–30` gives extraordinary height leverage. The pole is uniquely
favourable: the Sun circles at a few degrees elevation for a whole synodic month (polar
illumination geometry, Mazarico et al. 2011 [C]; Speyerer & Robinson 2013 [M]).

**Acceptance test.** On synthetic terrain **with slopes** (not the current flat background),
joint inversion recovers heights with materially lower bias than the flat-ground law.

## U4 — Calibration to a defensible strike probability

**Rationale.** Obstacle counts become a landing risk only through a strike model. Poisson
strike probability `P = 1 − exp(−λA)` is standard given an areal density λ and footprint A;
the density's size-frequency form on planetary surfaces is well characterised (Golombek &
Rapp 1997 [C]; lunar rock abundance, Bandfield 2011 [C]).

**Method.** Fit the observed obstacle size-frequency distribution, integrate above the
lander's clearance to get λ, propagate U2 uncertainty into λ, and report **P with a CI**.
State plainly that with a handful of instrumented lunar landing outcomes, this is a
*calibrated hazard index*, not a validated P(loss of mission). Honesty here is a feature.

**Acceptance test.** λ from HATI shadows agrees with Diviner-derived rock abundance
(Bandfield 2011 [C]) within stated uncertainty, at matched scale. **Promote this cross-check
out of "hardening" — it is the first independent instrument check the census has ever had.**

## U5 — Held-out validation with frozen parameters

**Rationale.** Every current number was measured on sites used to develop the method.
Generalisation claims require untouched data. This is elementary and currently missing.

**Method.** Freeze every threshold and weight, hash the config, run unmodified on a site
never used in development, and report what comes out. No knob moves after the freeze.

**Acceptance test.** A committed config hash, and a result reported whether or not it flatters
the method.

### Also worth doing, lower priority
* **Texture ablation** — after U1, test whether texture adds anything over physical + shadow.
  If not, cut it; a two-layer pipeline is stronger than a three-layer one with a dead layer.
* **Weight-sensitivity sweep** — perturb the literature weights, show conclusions are stable.
  Half a day; closes a hole a reviewer finds in five minutes.

---

# PART III — HATI execution

| Phase | Work | Where | Gate to pass |
|---|---|---|---|
| **H0** | Repo hygiene: `hati_core` module, config-hash freezing, validity masks threaded through (A0.3) | laptop | imports clean, scripts unchanged in behaviour |
| **H1** | **U1 scale-explicit channels** + re-run mare control | laptop→box | cross-site AUC reported at matched physical baselines |
| **H2** | ISIS install + **ingest dry-run** | box | ISIS pill green, frame plan printed |
| **H3** | **Ingest execute** → `coreg_report.csv` | box | **median \|shift\| ≤ 1 px** — the error budget |
| **H4** | **Real-data kinematics** (vote map, then U3 inversion) | box | first real obstacle+height map |
| **H5** | U2 uncertainty + U4 calibration + Diviner cross-check | box | λ agrees with Diviner within CI |
| **H6** | U5 held-out site, frozen config | box | honest generalisation number |
| **H7** | v2.5 capstone paper | laptop | one honest end-to-end account |

**H3 is the hard gate.** If co-registration cannot reach ~1 px, U3 degrades and we say so
publicly rather than papering over it.

---

# PART IV — Poseidon: the maritime adaptation

### What actually transfers

Not "shadows". The transferable core is the **method**:

> Detect sub-resolution / low-contrast targets by accumulating evidence across multiple
> acquisition geometries, where a *physical forward model* predicts how a real target's
> signature must change between looks, and anything that fails to obey that model is rejected.

### The exact physical analogue — and it is a good one

Lunar: shadow length encodes height, `L = h·cot e`; the shadow **pivots** with solar azimuth
while albedo marks stay put.

Maritime SAR: a **moving** target is displaced along-azimuth in the image by
`Δx ≈ (R/V)·v_r` — slant range over platform velocity, times radial velocity. This is the
classic SAR moving-target azimuth shift (Raney 1971 [M]; standard in SAR ship literature,
Crisp 2004 [C]). So **the offset between a ship and its wake encodes its velocity exactly as
shadow length encodes obstacle height** — one physical parameter recovered from a geometric
displacement, deterministically, with an error budget.

That is not a metaphor. It is the same estimator shape, and it means the HATI core ports with
its auditability intact.

| HATI (lunar) | Poseidon (maritime) |
|---|---|
| Solar azimuth sweep | Multi-temporal Sentinel-1 revisits, varying incidence/orbit |
| Shadow length → height `h = L·tan e` | Azimuth displacement → radial velocity `v_r = Δx·V/R` |
| Albedo decoy (dark, doesn't move) | Sea clutter / speckle / azimuth ambiguity (doesn't obey the model) |
| Vote convergence at obstacle base | Track-consistent detection across looks |
| DEM slope gate (gross terrain) | CFAR sea-clutter gate (gross detection) *(Crisp 2004 [C])* |
| Poisson strike probability | Detection/false-alarm rate per km² per pass |
| Diviner cross-check | **AIS as ground truth** — and the residual is the product |

### Why the residual *is* the product

Correlate SAR detections against AIS. Matched → a cooperative vessel. **Unmatched → a dark
vessel.** The thing that is normally an error term becomes the deliverable. This gives
Poseidon something HATI can never have: **abundant, free, continuously-refreshed ground
truth.** Sentinel-1 is open (Torres et al. 2012 [C]); AIS is broadly available; the
vessel-detection literature is mature enough to benchmark against (Crisp 2004 [C];
Greidanus et al., SUMO ship detector, *Remote Sensing* 2017 [M]; optical vessel detection
survey, Kanjir et al., *RSE* 2018 [M]).

**The strategic consequence: Poseidon can be validated far faster and far harder than HATI.**
Lunar has ~one instrumented outcome; maritime has thousands of labelled ship-hours per week.
Poseidon may end up being where the *method* gets its statistical proof — and that proof
flows back to strengthen the lunar claim.

### What does *not* transfer (state it up front)
* Radar backscatter physics ≠ optical shadow physics. The detector front-end is new (CFAR on
  speckle statistics, not local-relative brightness thresholding).
* Sea surface is non-stationary; the "background" moves. Lunar terrain does not.
* Speckle is multiplicative, not additive.
* Ambiguities and azimuth artefacts are SAR-specific failure modes with no lunar analogue.

---

# PART V — Poseidon execution

| Phase | Work | Gate |
|---|---|---|
| **P0** | Pull one Sentinel-1 GRD scene over a Baltic box + matching AIS for the same window | data in hand, co-registered |
| **P1** | CFAR detector baseline on that scene *(Crisp 2004 [C])* | detections vs AIS: recall + false alarms per km² |
| **P2** | Multi-look accumulation across ≥5 revisits; reject detections that fail the physical model | measured false-alarm reduction vs single pass |
| **P3** | Azimuth-displacement velocity estimate on wake-bearing ships | velocity vs AIS-reported SOG, RMSE |
| **P4** | **Dark-vessel product**: detections with no AIS correlate, with confidence | a named dark-vessel list over a named box and window |
| **P5** | Package: the one-pager + P4 numbers → design-partner conversation (Danish Navy / coast guard / EMSA / Frontex) | one letter of interest |

**P4 is the Poseidon equivalent of H3/H4** — the moment it stops being a deck. Everything
before it is plumbing; everything after it is business.

### Sequencing HATI against Poseidon

Do **H1–H4 first.** Reason: H4 is the scientific credibility that makes Poseidon fundable,
and the U1/U2 work (scale-explicit, uncertainty) is *shared core* that Poseidon inherits.
Start P0–P1 in parallel only because they are cheap and data-gathering has latency. Do not
open P2+ until H4 is done, or both stall.

---

# PART VI — References to verify

**Unverified — locate and confirm each before citing.**

**Lunar / planetary**
1. Kreslavsky, M.A. & Head, J.W. (2000). Kilometer-scale roughness of Mars. *JGR* 105(E11):26695–26711. **[C]**
2. Rosenburg, M.A. et al. (2011). Global surface slopes and roughness of the Moon from LOLA. *JGR Planets* 116:E02001. **[C]**
3. Kreslavsky, M.A. et al. (2013). Lunar topographic roughness maps from LOLA. *Icarus* 226:52–66. **[C]**
4. Robinson, M.S. et al. (2010). LROC instrument overview. *Space Sci. Rev.* 150:81–124. **[C]**
5. Henriksen, M.R. et al. (2017). Extracting topography from LROC NAC stereo. *Icarus* 283:122–137. **[C]**
6. Hapke, B. (1984). Bidirectional reflectance spectroscopy 3: macroscopic roughness. *Icarus* 59:41–59. **[C]**
7. Sato, H. et al. (2014). Resolved Hapke parameter maps of the Moon. *JGR Planets* 119:1775–1805. **[C]**
8. Bandfield, J.L. et al. (2011). Lunar rock abundance from Diviner. *JGR Planets* 116:E00H02. **[C]**
9. Mazarico, E. et al. (2011). Illumination conditions of the lunar polar regions from LOLA. *Icarus* 211:1066–1081. **[C]**
10. Golombek, M. & Rapp, D. (1997). Size-frequency distributions of rocks. *JGR* 102(E2):4117–4129. **[C]**
11. Speyerer, E.J. & Robinson, M.S. (2013). Persistently illuminated regions at the lunar poles. *Icarus* 222:122–136. **[M]**
12. Barker, M.K. et al. (2016). LOLA + SELENE TC lunar DEM (SLDEM2015). *Icarus* 273:346–355. **[M]**
13. Cai, Z. & Fa, W. (2020). *JGR Planets*, doi:10.1029/2020JE006429 — already in `references.bib`. **[C]**
14. Kirk, R.L. et al. (2003). High-resolution topomapping of MER candidate sites. *JGR* 108(E12). **[M]**

**Vision / estimation**
15. Woodham, R.J. (1980). Photometric method for determining surface orientation. *Optical Engineering* 19(1):139–144. **[C]**
16. Horn, B.K.P. (1975). Obtaining shape from shading information. **[M]**
17. Savarese, S. et al. (2001). Shadow Carving. *ICCV*. **[V]**
18. Riley, S.J. et al. (1999). Terrain Ruggedness Index. *Intermountain J. Sciences* 5:23–27. **[M]**
19. Shannon, C.E. (1949). Communication in the presence of noise. *Proc. IRE* 37:10–21. **[C]**

**Maritime / SAR**
20. Torres, R. et al. (2012). GMES Sentinel-1 mission. *Remote Sensing of Environment* 120:9–24. **[C]**
21. Crisp, D.J. (2004). State-of-the-art in ship detection in SAR imagery. DSTO Research Report. **[C]**
22. Raney, R.K. (1971). Synthetic aperture imaging radar and moving targets. *IEEE Trans. AES*. **[M]**
23. Greidanus, H. et al. (2017). The SUMO ship detector algorithm. *Remote Sensing* 9. **[M]**
24. Kanjir, U. et al. (2018). Vessel detection from spaceborne optical imagery: a survey. *RSE* 207:1–26. **[M]**

---

# PART VII — What would falsify this

Stated in advance so the project cannot quietly move the goalposts.

1. **Co-registration cannot reach ~1 px** on real frames → U3 inversion is unreliable; the
   sub-resolution claim reduces to detection without calibrated height. *Report it.*
2. **Texture channels stay non-transferable after U1** → the negative result is intrinsic;
   cut the texture layer entirely and say so.
3. **HATI λ disagrees with Diviner** beyond stated uncertainty → the shadow census has a
   systematic error; find it before any strike probability is published.
4. **Held-out site performance collapses** → the method does not generalise; the honest
   product is a *site-specific analysis tool*, not a general hazard map.
5. **Maritime CFAR baseline already matches the multi-look accumulation** → the HATI core
   adds nothing over standard practice at sea; Poseidon's differentiator becomes auditability
   and integration alone, which is a weaker but still real position.

Any one of these is survivable. Concealing any one of them is not.
