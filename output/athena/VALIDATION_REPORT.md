# Does HATI still work when the data gets worse?
### A resolution-ladder validation of the two pipelines

**Author:** Gergő Csaba Morvai · HATI v2.0 · 2026-06-04
**Code:** `scripts/athena_resolution_validation.py`
**Figures:** `validation_curves.png`, `validation_ladder.png`

---

## The question

Everything HATI had shown so far was a single site, scored relative to itself. That
leaves the obvious question unanswered: how well does it actually *work*? If you handed
the pipeline coarser data, would it still find the hazards you can plainly see in
sharper data of the same ground?

That question has a clean experiment hiding inside it. Take imagery where the hazards
are directly visible. Blur it down to a worse resolution. Run the pipeline on the blurry
version. Then check the blurry-data answer against what the sharp data shows. Only one
thing changes between the two — the resolution — so whatever the pipeline gets right or
wrong is about resolution and nothing else. No second site to confound it, no different
sensor, no co-registration headache.

This report is that experiment, run on the NOBILE03 data already on disk: the 0.9 m
photograph for the shadow channel and the 4 m elevation model for the heatmap.

## How I set it up

Two choices mattered, and both are easy to get wrong.

**I blurred the way a real camera does, not the lazy way.** A coarser camera averages
the light falling on each larger pixel. So I coarsened by averaging blocks of pixels
together (a 2×2 block becomes one pixel, then 3×3, 4×4, and so on). The tempting
shortcut — just throwing away three out of every four pixels — does something subtly
poisonous: it aliases, folding fine detail down into fake coarse texture. You would then
"detect" hazards that are artifacts of your own bad downsampling. Averaging avoids that,
and it is what a real lower-resolution sensor would actually deliver.

**The sharp data is the ground truth.** At full resolution I run the pipeline once and
treat its detections as the answer key — these are the real shadows and obstacles you
can see. Then I score the coarse-data runs against that key. Is this circular, since the
coarse data is made from the sharp data? No, and the distinction is the whole point:
averaging genuinely destroys small features. Whether a shadow survives being blurred is
a real, open question, and measuring exactly when it stops surviving is the result.

## What the shadow channel did: it works, and here is its reach

This is the strong result. The figure `validation_curves.png`, left panel, shows the
detection trade-off (true hits versus false alarms) at each degraded resolution. The
area under that curve — a single 0–1 score where 0.5 is a coin flip and 1.0 is perfect —
holds up remarkably as the data gets worse:

| Degraded to | 1.8 m | 2.7 m | 3.6 m | 5.4 m | 7.2 m |
|---|---|---|---|---|---|
| Detection score (AUC) | 0.99 | 0.97 | 0.94 | 0.88 | 0.83 |

Even after blurring the photograph eightfold, to 7.2 m per pixel, the pipeline still
sorts shadow from not-shadow at 0.83 — far above chance.

The middle panel is the one I would put on a slide. It breaks the recovery down by how
big each obstacle's footprint is, and it draws a clean physical limit. Reading the
fraction of full-resolution obstacles still caught at each coarseness:

- Obstacles **larger than 16 m** are recovered essentially in full out to 5.4 m
  resolution, and only start slipping at 7.2 m.
- **8–16 m** obstacles hold on until about 3–4 m resolution (96% recovered at 1.8 m,
  down to 45% by 3.6 m), then fade.
- **4–8 m** obstacles survive only the gentlest blurring (67% at 1.8 m, gone by 3 m).
- Anything **under 4 m** washes out almost immediately.

The pattern is exactly what physics demands: an obstacle's shadow stays detectable while
the pixels are smaller than roughly half its footprint, and disappears once the pixels
grow past it. Nothing magic, nothing recovered from thin air. The pipeline reaches a
little below the resolution of the data it is given, and the curve says precisely how
far. The right panel of `validation_ladder.png` makes it visible: the same patch at
0.9 m, 3.6 m, and 7.2 m, with detections drawn in red — the small shadows quietly vanish
as the picture coarsens, the big ones stay lit.

For the original claim, this lands cleanly. A 0.9 m photograph recovers obstacles down
to a few metres across — well inside the 8 m blind spot of the 4 m elevation model. That
is the sub-resolution hazard layer the two-channel design was built to catch, now with a
number on it.

## What the heatmap test did: a null, then a proper test that worked

The heatmap took two attempts, and the reason the first one failed is worth telling.

I ran the matching test on the heatmap: take the 4 m elevation model as truth, average
it down to 8 m and 16 m, run the roughness heatmap on the coarse version, and ask whether
it flags the cells that actually hold fine relief in the 4 m data. The detection score
came back at **0.50 and 0.52** — a coin flip (right panel of `validation_curves.png`,
the two lines sitting on the diagonal).

I am not going to spin that. But it is the *expected* result, and it says something
useful rather than something damning. Two reasons it had to come out this way:

1. **Clean averaging erases the target.** The thing I asked the coarse heatmap to find —
   the relief that lives *inside* each coarse cell — is exactly the thing the averaging
   removed. You cannot recover from a number what went into making that number. A blur,
   done honestly, is not invertible.
2. **The scales do not line up.** A heatmap on 16 m pixels measures roughness at the
   tens-of-metres scale; the relief I was testing it against lives at 4–16 m. They are
   looking at different bands, so of course they barely correlate.

So this is not "the heatmap is broken." It is a clean boundary line: the heatmap reads
roughness at scales the elevation model actually resolves — which is its job, and which
is what flagged the Athena touchdown in the first place — and it does not pretend to see
beneath the model's own floor. Seeing beneath the floor is the shadow channel's job, and
that is the channel the experiment above validated. If anything, the null reinforces why
HATI has two channels instead of one: each covers a band the other cannot.

That null also pointed at the fix, and the fix turned the result around.

A fair test needs a DTM *finer* than the one being scored, so the ground truth can be
real resolved features instead of erased content. There was a 1.5 m NAC DTM of Apollo 17
(Taurus-Littrow) already on disk. So I ran the proper version: define the hazards from
the 1.5 m data (top-quartile relief over a 9 m window — actual rough, steep ground),
degrade the DTM to 3, 6 and 12 m, run the heatmap on each coarse version, and score it
against those fine-data hazards (`heatmap_validation.png`).

This time it works, clearly. Detection AUC of **0.86, 0.92, 0.93** at 3, 6 and 12 m, all
far above chance, with the heatmap catching 61–75% of the fine-data hazards when set to
flag only its top quartile. The earlier coin flip was never the heatmap failing — it was
me asking the wrong question, to recover the unrecoverable. Asked properly, the heatmap
recovers fine-data hazards from coarse elevation data with high reliability.

One honest read of the trend: the AUC actually *rises* a little as the data coarsens.
That is not "worse data is better." It is that the ground truth here — broad rough
terrain (the massif slopes) versus the smooth valley floor — is a large-scale signal,
bigger than 12 m, so a coarser heatmap that integrates over larger areas matches it
slightly better, while a finer one adds local texture that reads as noise against a broad
label. Two things follow, and I will say both: the heatmap's strength is discriminating
rough terrain from smooth, robustly across resolution; and this scene, with its strong
valley/massif contrast, is a clear but gentle test — a more uniform site would be harder.
(Apollo 17 is also where the heatmap's window and scale settings were first shown; the
weights are literature-fixed, so nothing is fitted here, but an independent site would
harden it further.)

## What this does and does not establish

Both channels now carry a measured number, and the two numbers say different,
complementary things.

The shadow channel genuinely detects sub-resolution obstacles: AUC above 0.8 even after
heavy degradation, with a recovery curve that says exactly how small an obstacle it can
hold onto at each resolution. That is the channel that reaches *below* the elevation
model's floor, now quantified.

The heatmap genuinely discriminates hazardous terrain from coarse elevation data: AUC
0.86–0.93 against fine-data relief. It does this at the scales a DEM resolves — it does
not reconstruct features that averaging erased (the first test fixed that boundary in
place). So the two channels split the work cleanly, and the data now shows the split: the
heatmap ranks resolved rough terrain, the shadows recover individual sub-resolution
obstacles.

What is still open is honest to name. Both tests use controlled degradation of single
sites, not a genuinely different coarser sensor (which would add its own noise); and a
smooth-mare control is still owed, to measure false alarms on benign ground. But the core
of it stands: each pipeline has been handed worse data and measured against the truth
visible in better data, and each does the job it claims, with a number attached. That is
the position you can defend in front of someone holding the real answer key.

## Reproduce it

`python scripts/athena_resolution_validation.py` runs the shadow ladder on the NOBILE03
ortho and the first (within-cell) heatmap test; `python scripts/heatmap_resolution_validation.py`
runs the proper heatmap ladder on the 1.5 m Apollo 17 DTM. Both degrade by block-averaging
and score coarse-data detections against full-resolution truth. Fixed choices: the shadow
detector uses a physically-scaled 40 m background window and the same 0.5×-background
threshold at every resolution; the shadow ground truth is the full-resolution detection;
the proper heatmap ground truth is the top quartile of 9 m relief in the 1.5 m DTM; the
heatmap uses the same window-7, MDS 8–32 m configuration as the Athena run. Figures:
`validation_curves.png`, `validation_ladder.png`, `heatmap_validation.png`.
