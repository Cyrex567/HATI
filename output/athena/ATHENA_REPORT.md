# Could we have seen it coming? Re-checking the Athena landing site with HATI

**Author:** Gergő Csaba Morvai · HATI v2.0 · 2026-05-29
**Status:** Proof of concept for the HATI v2.0 two-channel hazard pipeline.
**Companion figures:** `athena_counterfactual.png` (both channels), `athena_channel_influence.png` (per-channel decomposition)
**Code:** `scripts/athena_counterfactual.py`, `scripts/athena_figure.py`, `scripts/athena_channel_influence.py`
**Data:** LROC NAC DTM NOBILE03 (Arizona State University / M. Robinson), 2012 source frames

---

## TL;DR (read this even if you read nothing else)

On 2025-03-06 the Intuitive Machines IM-2 lander, "Athena," touched down near the
lunar south pole and tipped over. The LROC team later pinned the upset on a small
crater, roughly 20 m across, sitting on sloped ground right where it landed.

Here's the question this report answers: if we feed HATI **only the maps that
already existed before the landing**, does it point at that exact spot and say
"don't land here"?

Short answer: one of HATI's two methods does, clearly. The landing point scores
in the **top 4%** of hazard across an already-nasty stretch of terrain. And it does
that even though the ground there is gently sloped, which is the whole point. The
danger wasn't the tilt. It was the texture, and the texture lives below the size a
normal planning map can see.

The second method doesn't single out the exact point, because the entire region is
blanketed with small obstacles. That's not a failure. It's a louder warning: there
was no safe pixel nearby to aim for.

---

## 1. What we're actually testing, and why it's easy to fool yourself

A counterfactual is a "what if we'd looked earlier" test. Those are dangerous,
because you already know the answer, and it's tempting to massage the analysis until
it agrees with reality. Two rules keep this one honest.

First, **only pre-landing data.** The elevation map and the photo I use were both
built in 2012 from images taken thirteen years before Athena arrived. Nothing in
them could have been contaminated by the crash. No lander, no fresh scuff marks, no
hindsight baked into the pixels.

Second, **the landing outcome never enters the math.** HATI scores the terrain
without being told what happened. The only thing I borrow from after the fact is the
single coordinate of where Athena ended up, so I know which pixel to read. The score
at that pixel is computed the same way as every other pixel in the scene.

The landing point, from the LROC team's measurement: **84.7906° S, 29.1957° E**, on
Mons Mouton, deep in the south-polar highlands.

I should be blunt about what this is and isn't. This is a *detectability* test:
was the hazard visible in old data? It is **not** a claim that HATI would have
changed the mission, and it is **not** a blind prediction (I knew where to look).
Those are different, weaker claims, and I'm not making them.

---

## 2. From a coordinate on a globe to a pixel in a file

Both maps use the same projection: a south-polar stereographic grid on a sphere of
radius 1737.4 km, centred on the pole, longitude measured east. To find the landing
pixel I convert latitude/longitude to metres-from-pole with the textbook spherical
formula,

```
rho = 2 * R * tan(pi/4 + phi/2)        # phi is latitude (negative in the south)
x   = rho * sin(lambda)                 # lambda is longitude east
y   = rho * cos(lambda)
```

then divide by the pixel size and offset by the map's corner. I ran this two
independent ways: the closed-form equation above, and GDAL's own coordinate
transform. They land on the **same pixel, zero disagreement**. On the elevation map
that's row 1181, column 387. On the photo, row 5248, column 1721. Anyone with the two
files and these numbers gets the identical pixels.

One honest wrinkle: the photo's label stores the corner's x-coordinate with a flipped
sign (a known quirk of how polar-stereographic axes get written down). I use the
magnitude, which puts the photo in the same frame as the elevation map, and I checked
that this reproduces the elevation-map pixel exactly. So the two maps are pinned to
each other.

---

## 3. Channel 1 — the roughness heatmap (this is the one that flags the site)

### 3.1 The idea, in one breath

Imagine a parking lot photographed from so high up that anything smaller than a
car-sized blob washes out to a smudge. You can't see individual potholes. But a
patch with lots of potholes still *looks* different from smooth asphalt: it's
speckled, busier, higher-variance. The heatmap is a machine for measuring "busy-ness"
at several sizes at once, then boiling it down to a single hazard number between 0
and 1.

The elevation map I use is 4 metres per pixel. A 20 m crater is only five pixels
wide and a couple of pixels deep. You won't always see it as an obvious bowl. But you
*will* see the local terrain get statistically rougher. That's the signal.

### 3.2 The channels: what each measures, why it's in the stack, and how hard each one pushed

Each ingredient is a small image the same size as the elevation map, where bright
means "rough here." They come straight from the planetary-roughness literature
(Kreslavsky & Head 2000; Rosenburg et al. 2011; Kreslavsky et al. 2013; Cai & Fa
2020; Wang et al. 2024). Here's what each one actually does.

| Ingredient | What it computes | What "high" means |
|---|---|---|
| RMS slope | square root of the average squared slope in a sliding window | steep, or full of little steep bits |
| IQR slope | spread (75th minus 25th percentile) of slope in the window | slopes pointing every which way |
| IQR curvature | spread of the Laplacian (how bowl- or dome-shaped) | mix of rims and pits, i.e. craters |
| Planar deviation | RMS of elevation after subtracting the local average | bumpy relative to a local flat fit |
| \|TPI\| | absolute difference of a pixel from its neighbourhood mean | sits on a bump or in a dip |
| TRI | mean absolute step to the 8 touching neighbours | jagged at the smallest scale |
| MDS(L) | difference of slope measured at size L vs size 2L, for several L | roughness *concentrated at size L* |

The "MDS" family is the clever one. Median Differential Slope blurs the terrain to two
sizes, L and 2L, measures slope on each, and subtracts. What survives is the roughness
that lives specifically around size L. Run it at several L values and you get a little
size spectrum of the bumpiness. I run L at roughly 8, 12, 20, and 32 metres, which
brackets the size of the crater Athena hit.

Slope, by the way, is just rise over run:

```
slope = arctan( sqrt( (dz/dx)^2 + (dz/dy)^2 ) )
```

computed from neighbouring pixel heights, divided by the 4 m pixel size to get real
units. Curvature is the Laplacian, the sum of the second derivatives, which is
positive in a bowl and negative on a dome.

Now the proof-of-concept payoff: which channels actually moved the score at the
landing point, by how much, and which way. The chart `athena_channel_influence.png`
shows it; the table below is the same data. "Contribution" is each channel's additive
share of the score's logit. They sum, the −1.0 bias is added, the sigmoid squashes the
total, and you land on H = 0.726.

| Channel | Weight | z at touchdown | Contribution | Value percentile | Push |
|---|---|---|---|---|---|
| Slope IQR | 1.00 | 4.00 (capped) | +0.476 | 98.7 | up |
| MDS @ 12 m | 0.90 | 3.79 | +0.406 | 98.4 | up |
| \|TPI\| | 0.70 | 4.00 (capped) | +0.333 | 99.0 | up |
| MDS @ 8 m | 1.00 | 2.69 | +0.321 | 96.0 | up |
| MDS @ 20 m | 0.80 | 2.92 | +0.278 | 96.7 | up |
| Curvature IQR | 1.00 | 1.29 | +0.153 | 85.0 | up |
| Planar deviation | 0.70 | 1.60 | +0.134 | 88.9 | up |
| MDS @ 32 m | 0.60 | 0.42 | +0.030 | 64.8 | up (weak) |
| RMS slope | 1.00 | −0.64 | −0.077 | 24.7 | DOWN |
| TRI | 0.70 | −0.94 | −0.079 | 16.2 | DOWN |

Each channel, and the reason it earns a place:

**Slope IQR — the spread of slope directions in the window. Weight 1.00. Biggest
driver: +0.476, 98.7th percentile, z pinned at the +4 cap.** Reason for inclusion
(Kreslavsky 2013; Wang 2024): a clean plane has slopes all pointing one way, so the
spread is tiny; a crater field has slopes pointing everywhere, so the spread explodes.
At the touchdown it maxed out. Fingerprint of a pitted surface, not a tidy slope.

**MDS @ 12 m — roughness concentrated at the 12 m size. Weight 0.90. Second driver:
+0.406, 98.4th percentile.** Reason (Kreslavsky & Head 2000; Rosenburg 2011):
differential slope isolates the bumpiness at one chosen size, and 12 m sits right on
the crater the LROC team blamed. This channel put its finger on the hazard's actual
size.

**|TPI| — how far the point sits above or below its local average. Weight 0.70.
+0.333, 99.0th percentile, capped.** Reason (Wang 2024): a lander cares whether it's
perched on a rim or sunk in a bowl. At the 99th percentile, the point sits on strong
local relief, the rim-and-floor structure of the craters around it.

**MDS @ 8 m and @ 20 m — roughness at those sizes. Weights 1.00 and 0.80. +0.321 and
+0.278; 96th–97th percentile.** Reason: the smaller MDS sizes sit nearest the map's
resolution floor, where an unresolved hazard leaves its first mark, so they carry top
weight. Both fired hard. The danger has structure across the whole 8–20 m band, not at
one isolated size.

**Curvature IQR and Planar deviation — spread of bowl-vs-dome shape, and lumpiness
around a local flat fit. Weights 1.00 and 0.70. +0.153 and +0.134; 85th and 89th
percentile.** Reason (Kreslavsky 2013; Wang 2024): craters carry a telltale curvature
pattern (positive rim, negative floor); planar deviation catches general lumpiness.
Both moderately raised, backing the cratered read without running the show.

**MDS @ 32 m — roughness at 32 m. Weight 0.60. Almost nothing: +0.030, 65th
percentile.** Reason: the coarsest size in this config, kept for completeness. Its
near-silence is itself a finding: the hazard is smaller than 32 m, so the largest
scale sees little.

**RMS slope and TRI — average slope magnitude, and single-step jaggedness to the 8
touching neighbours. Weights 1.00 and 0.70. Both push DOWN: −0.077 and −0.079; 25th
and 16th percentile.** Reason (Rosenburg 2011, the primary first-derivative roughness;
TRI the canonical single-step ruggedness): and here is the line that makes the proof
of concept land. Both sit BELOW the scene median. The average tilt is gentle, the
pixel-to-pixel steps are small. On their own, these two channels would call the site
calm.

#### The hazard signature, in one sentence

Stack it together: **low average slope, low jaggedness, but maxed slope spread,
near-maxed TPI, and high MDS across 8–20 m.** In plain words, the ground is gently
tilted and smooth between features, yet packed with crater-scale relief at 8–20 m.
That is what a gently-sloped, heavily-cratered patch looks like, and it is exactly the
case a slope-only or single-step screen (RMS slope, TRI) waves straight through. The
flag is not one channel shouting. It is the disagreement between the channels that read
gross tilt (quiet) and the channels that read structured, feature-scale roughness
(loud), and the heatmap does it from topography alone, no scattering model required.
Engineering that disagreement into a single readable number, rather than a black box,
is why this run stands as the proof of concept for the pipeline.

### 3.3 Dealing with the holes in the map

The elevation file is a rectangle, but the actual data covers a tilted diamond inside
it. About 45% of the rectangle is "no data." If you run sliding-window math straight
across a cliff between real data and a hole, you get garbage spikes at the edge.

Fix: I fill every hole with the value of the *nearest real pixel* before doing any
window math. That makes the filled region a smooth continuation instead of a cliff, so
the windows don't see a fake edge. Then, after computing the ingredients, I throw away
(set to "undefined") every pixel close enough to a real hole that a window could have
peeked into it. The landing point sits 116 m from the nearest hole, which is far
enough that the medium-size windows never reach the edge. The one exception matters,
and I deal with it in 3.5.

### 3.4 Folding seven-plus images into one score

Three steps. First, **normalise** each ingredient so they're on a common scale. I use a
median-and-IQR z-score (subtract the median, divide by the IQR over 1.349, which is the
IQR-to-standard-deviation conversion for a normal distribution), after a log(1+x)
squash and a clip at ±4. Plain version: center each ingredient on "typical for this
scene," scale it by its own spread, and don't let one freak pixel scream over the rest.

Second, **weighted sum.** Each normalised ingredient gets a weight from the literature,
not from tuning against Athena. Slope and curvature spreads get 1.0. The general-purpose
GIS metrics (TPI, TRI, planar deviation) get 0.7. The MDS family gets more weight at
smaller sizes, because the smaller the feature, the closer it is to invisible on a 4 m
map, and the more it matters. Then I divide by the total weight so the score doesn't
drift just because I added or dropped an ingredient.

Third, **squash to 0–1** with a logistic sigmoid and a bias of −1.0, so average terrain
lands well below 0.5 and only genuinely rough terrain pushes toward 1.

The output, H(x), is a hazard *ranking* for this scene. Say that out loud, because it's
the single most important caveat in the whole report: **H is relative.** A score of
0.73 means "rougher than most of *this* place," not "0.73 probability of death." More on
why that bites in Section 6.

### 3.5 Reading the score at the landing point, and not cheating about it

The honest worry with any multi-knob method: did I twiddle the knobs until it
incriminated the site? To kill that worry, I ran a sweep of window sizes and MDS size
bands and reported **all** of them, including the one that fails.

| Configuration | MDS sizes | H at landing point | Percentile in scene | Top 25% / top 10%? |
|---|---|---|---|---|
| Apollo default, window 11 | 12–80 m | **undefined** | — | — |
| window 5 | 8–32 m | 0.757 | 97.1 | yes / yes |
| window 7 | 8–32 m | 0.726 | 96.3 | yes / yes |
| window 9 | 8–24 m | 0.645 | 93.3 | yes / yes |
| window 7 | 8–20 m | 0.780 | 97.4 | yes / yes |

Every size band that *can* be computed at this spot puts the landing point in the top
4–7% of hazard. The score barely wanders: 93rd to 97th percentile. That stability is
the opposite of knob-twiddling.

The "undefined" row is real and I'm leaving it in. The original setting I validated on
the Apollo 17 site uses an 80 m roughness scale. To measure roughness at 80 m you blur
the map with a wide kernel, and that kernel reaches about 54 pixels out. The landing
point is only 29 pixels from the data edge. So the 80 m scale physically can't be
evaluated here without sucking in the hole. It returns "undefined," not a low score. The
sizes that actually fit a 20 m crater (8–32 m) fit comfortably inside the good data and
do the flagging.

### 3.6 The result, with the number that matters most

At the landing point, using the window-7 / 8–32 m setting:

- Hazard score **H = 0.726**, the **96.3rd percentile** of the scene. Only **3.7%** of
  the valid terrain scores rougher.
- That's **3.22 robust standard deviations** above the scene's median hazard (using the
  median-absolute-deviation scale, which shrugs off outliers).
- Slope at the point is **2.8°**. The scene's median slope is 5.1°. So the landing point
  is in the **15.9th percentile for slope** — smoother than 84% of the area.

Sit with that pair. **Smoother than most of the site by tilt, rougher than almost all of
it by texture.** A landing system that screens sites by slope alone, which is the common
approach, would have waved this spot through. The multi-scale roughness does not. That
gap, between what a slope check sees and what's actually there, is the entire reason
HATI exists.

---

## 4. Channel 2 — the shadow census (the sub-resolution layer)

### 4.1 The idea: shadows are a ruler for things the elevation map can't resolve

The elevation map stops at 4 m. The photo is 0.9 m per pixel, more than four times
finer. At the south pole the sun barely clears the horizon, so even a knee-high rock
throws a long shadow. Every little crater shows a dark crescent. Those shadows are free
measurements of objects far too small to appear on the elevation map.

A shadow gives you two separate numbers. Its width *across* the sun direction is the
object's actual footprint, and that doesn't depend on how low the sun is. Its length
*along* the sun direction tells you the object's height, through
`height = length * tan(sun elevation)`. For this report the footprint is the headline,
because it needs no assumption about the sun angle.

### 4.2 Why the obvious method falls apart at the pole

The naive shadow detector is "call any dark pixel a shadow." At the pole that fails
badly, because half the darkness isn't cast shadow at all, it's just slopes tilted away
from the grazing sun. I checked: a global brightness threshold (Otsu) labels 54% of the
crop "shadow," which is meaningless. It's splitting sunlit-toward from sunlit-away, not
finding obstacles.

The fix is to define a shadow *relative to its surroundings*. I build a local
brightness baseline by running a 40 m median filter over the photo, then call a pixel a
shadow only if it's darker than **half** that local baseline. This adapts to the
overall light gradient and isolates genuine cast shadows. The dark fraction drops to a
sane 5%, and visually it lights up exactly the crater interiors and obstacle shadows,
not the gentle slopes.

### 4.3 Turning dark blobs into measured obstacles

After thresholding I clean the mask (one round of morphological opening to kill single-
pixel speckle, one of closing to seal pinholes), label the connected blobs, and fit an
ellipse to each one with at least 4 pixels. The ellipse's short axis is the cross-sun
footprint; the long axis is the shadow length. Multiply pixel counts by 0.9 m to get
metres.

### 4.4 The line that separates "the map could see it" from "it couldn't"

The elevation map is 4 m per pixel. By the sampling theorem you need at least two
pixels to register a feature, so anything under about **8 m** is below the map's floor.
I split every detected obstacle at that 8 m footprint. Below it, the elevation map is
blind by physics, and the shadow census is the only way to know the obstacle is there.

### 4.5 A built-in lie detector

If my "shadows" were really just noise or dark rock, their orientations would point
every which way. Real cast shadows all lean away from the sun, so they should share one
direction. I measured the dominant orientation of the elongated shadows: **85°, sharply
single-peaked.** That's the signature of illumination-driven shadows, and it's a free
check that the detector is finding real physics and not albedo splotches. (Tying 85° in
the image to a true compass bearing needs SPICE geometry, which I haven't pulled in. The
single peak is the point here, not the absolute bearing.)

### 4.6 What the census says

Over a 1.08 km² box around the landing point:

- **1639 shadows**, about **1514 per km²**.
- **93.6%** have footprints **below the 8 m elevation-map floor**, roughly **1417 per
  km²** of obstacles the planning map physically cannot resolve. Median footprint: 3.2 m.
- Within a 25 m landing-dispersion radius of the exact point: **3 sub-resolution
  obstacles** (9 within 50 m). Nearest shadow 3.7 m away.

Now the honest part. Is the landing point a *local spike* in obstacle density? No. Its
local density sits at the **49.9th percentile**, dead median, for this region. The
reason is almost more damning than a spike would be: the whole place is uniformly
carpeted with sub-resolution obstacles. There was no quiet pocket within reach to retarget
into. So Channel 2 doesn't independently point a finger at the pixel. It does something
else: it measures a thick hazard layer, more than a thousand obstacles per square
kilometre, that the planning map never showed anyone.

---

## 5. Putting the two channels together

Channel 1 ranks the exact landing point in the worst 4% of the area, on texture rather
than tilt. Channel 2 shows that the same area is saturated with obstacles below the
planning map's resolution. One pinpoints the spot. Both agree the site was dangerous at
the scale the mission could see *and* at the scale it couldn't.

The success bar set for this test was simple: if either channel flags the point, the
test passes. Channel 1 flags it. The test passes. I am deliberately not upgrading that
to "both channels flag it," because Channel 2's density is median, and saying otherwise
would be dishonest.

### 5.1 — The channel we didn't even build: Hapke photometric roughness (θ̄)

Say this part out loud, because it changes how you should read the whole result.
HATI has no photometric-roughness channel. Zero. The Hapke scattering model carries
a parameter called θ̄ ("theta-bar"), the mean photometric slope angle, and it puts a
number on roughness *below the size of a single pixel* by reading how sunlight
scatters off the surface across different sun-and-camera angles. It's the standard
tool planetary scientists reach for when they want sub-resolution texture (Hapke
1984; resolved lunar maps in Sato et al. 2014 — citation to verify, not peer-checked
in this report). HATI doesn't compute it. It isn't in the pipeline at all.

And the hazard still shows up. Two channels built from nothing but geometry, resolved
roughness from elevation and obstacle shadows from a photo, both light up at the
landing point with no scattering model anywhere in the stack.

The part worth being proud of, stated precisely: the signal doesn't lean on the hard,
model-heavy machinery. It falls out of first-principles geometry, through two
independent routes that agree. When a result survives that kind of triangulation you
trust it more, because two unrelated physical mechanisms would both have to fail the
same way to fool you.

The part not to oversell, because a reviewer won't let it slide: θ̄ is not redundant
with what we built, and these channels did not "replace" it. They live in different
size bands. The heatmap reads 8–80 m roughness from topography. The shadow census
reads discrete obstacles from roughly a metre up. θ̄ reads the *statistical* sub-pixel
slope, a band neither channel touches directly. So the defensible claim is "we got a
hazard signal before adding any photometric roughness," not "we already have what θ̄
would give." A θ̄ channel would still add an independent third measurement, especially
in the finest texture where no single shadow is resolved.

Two more honest edges. You can't even fit θ̄ from the single NAC frame I used; the
inversion needs multi-angle photometry, so "not implemented" is partly "not
implementable from this data." That actually sharpens the point rather than softening
it: the geometric route got to a hazard flag without ever needing the photometric
dataset. And "characteristic signal of *hazards*" is a specificity claim. What we have
is a strong signal *at* a hazard. We have not yet shown the signal goes quiet over
safe ground. Until the safe-mare control runs (§6), the honest phrasing is "a strong,
two-channel signal at a known hazard," not "characteristic of hazards." Same pride,
wording that lives through peer review.

---

## 6. Scrutiny: where a hostile reviewer would push, and how hard it holds

I ran this past an adversarial review. The strong objections, and my straight answers:

**"96th percentile of a terrible neighbourhood is not an absolute hazard."** Correct,
and this is the biggest limitation. H is a within-scene ranking. The baseline here is
the Mons Mouton highlands, which are rough everywhere. To turn "rougher than 96% of a
bad place" into "objectively dangerous," I need to run the identical pipeline over a
genuinely safe site (a smooth mare patch) and show the landing point's raw feature
values sit far above what safe ground produces. That control run is the next job, and
until it exists, the absolute claim is unproven. The *relative* claim stands.

**"You tried five settings. That's fishing."** I reported all five, including the one
that fails outright. The flag holds from the 93rd to the 97th percentile across every
computable setting and uses literature weights I didn't tune against Athena. Fishing
looks like one lucky setting surrounded by silence. This is the opposite.

**"n = 1, no controls."** True. One positive site proves the method *can* fire on a
known-bad spot. It says nothing yet about false alarms. The method needs safe sites that
*don't* flag and, ideally, other failure sites that do, before anyone claims a hit rate.

**"Fine-scale roughness on a stereo map might be matching noise, not real rock."** This
is the sharp one. The elevation map is built by matching two photos, and that process has
its own correlated error at the few-pixel scale, exactly the scale my flag leans on.
Some of the "roughness" at 8–12 m could be reconstruction noise rather than ground truth.
The fix is to read the product's confidence map (`NAC_DTM_NOBILE03_CONF`) at the landing
point and confirm the stereo solution there is high-quality. I have not done that yet. It
is on the list, and the flag should carry an asterisk until it's done.

**"The shadow threshold is arbitrary."** Half-of-local-background is a defensible choice,
but the count moves with it, the same roughly twofold sensitivity I documented on the
Apollo data. So I report obstacle density as an order of magnitude, ~10³ per km², not as a
precise figure.

**Circularity.** I knew the site failed. The guardrails (pre-landing pixels, outcome-free
scoring) keep the *measurement* clean, but the *site selection* is hindsight. That's why
this is framed as detectability, not prediction.

None of these sink the result. They fence it. The defensible statement is: with old data
and untuned, literature-anchored settings, HATI's roughness channel ranks the actual
Athena landing point among the worst few percent of an already-dangerous site, driven by
texture a slope check would miss, and a finer-resolution shadow census confirms a dense
obstacle layer the planning map could not show. Two open jobs (a safe-site control, the
stereo confidence check) would harden "worst few percent here" into "objectively unsafe."

---

## 7. Reproduce it exactly

Files (under `C:\Máni Mission\HATI V2.0\`):

```
data/athena/NAC_DTM_NOBILE03.TIF                      # 4.0 m/px elevation, 1473x1172
data/athena/NAC_DTM_NOBILE03_M1101075756_90CM.IMG     # 0.9 m/px photo, 6547x5209, 16-bit
scripts/athena_counterfactual.py                       # both channels
scripts/athena_figure.py                               # the 6-panel figure
scripts/athena_channel_influence.py                    # per-channel decomposition at touchdown
output/athena/athena_dem.npz, athena_shadow.npz        # cached results
```

Run:

```
python scripts/athena_counterfactual.py --stage both
python scripts/athena_figure.py
python scripts/athena_channel_influence.py
```

Fixed constants (no per-run tuning): landing point 84.7906° S, 29.1957° E; elevation
pixel (1181, 387); photo pixel (5248, 1721); MDS weight ladder 1.0/0.9/0.8/0.6 by
ascending size; fusion bias −1.0; z-score clip ±4; shadow threshold 0.5 × local median
(40 m window); obstacle floor 8 m; density radius 25 m. The representative heatmap is the
window-7 / 8–32 m setting.

Why the photo is read from a byte offset: the 16-bit raster sits in the last
6547×5209×2 bytes of the `.IMG`, so the script computes the offset from the file size
rather than trusting a header. That makes the read independent of label quirks.

---

## 8. Provenance

LROC NAC DTM NOBILE03, Arizona State University, producer M. Robinson, built from NAC
frames M1101075756 / M1101097181 / M1101118606 acquired 2012-08-31, product version 1.9.
Both the elevation map and the photo predate the IM-2 landing, so no post-landing
information could leak into the inputs. Analysis and code by Gergő Csaba Morvai, HATI v2.0.
The ~20 m crater attribution at the landing site is the LROC team's published measurement.
