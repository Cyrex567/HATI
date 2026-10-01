# Dynamic context: how HATI sizes a shadow it cannot see whole

A rock's shadow at the polar Sun is long: at Athena's lowest frame (3.28 degrees) a 0.6 m rock throws about 12 px of shadow and a 2 m rock about 39 px. The detector's first window is small, so sizing grows the window around each warning cell until the shadow fits (`refine_cell` in `src/hati_core/adaptive_shadow.py`).

## How the window grows

Each pass doubles the window (scale factors 1, 2, 4 by default; base radius 12 px, fitting support 6 px) and refits height and width over the whole template bank, refining around the compatible pairs. It stops with a height estimate when the fitted size is stable against the previous pass, the compatible range is narrow, the shadow ends and the ground beyond them are seen in every frame, and the fit is not at the edge of the bank. Otherwise the last pass gives only a lower bound: the lower end of the compatible range, or, when even the shortest compatible shadow runs past the window, the height whose shadow at the lowest Sun just reaches the window edge.

At Athena's lowest Sun the supports of the three passes hold shadows of rocks up to about 0.31, 0.62 and 1.24 m. A taller rock never fits whole and can only be bounded. With the 77-frame stack, whose lowest Sun is 1.63 degrees, shadows are twice as long.

## What can go wrong, and the settings for it

| Setting | Default | What it does |
|---|---|---|
| `context_guard` | off | Stops growing when a larger window contradicts a smaller one that saw where the shadow ends and the ground beyond it: the larger window claims a caster taller than the smaller window's whole compatible range plus `stability_m`. The smaller window's fit stands (state `context_conflict`), and T14 bounds from it. |
| `pad_edges` | off | Lets a window run past the image edge, the outside counted as missing data like any gap. The fit's own minimum valid support (85% per frame, 80% in common) still decides whether enough of the window is inside. Without it, any window that crosses the edge stops the cell (`unresolved_image_edge`). |
| `scale_factors` | 1, 2, 4 | A fourth pass (8) holds shadows of rocks up to about 2.5 m at Athena's Sun; it needs `pad_edges` to matter near the edge of a 512 px image. |
| `caster_profile` | `plate` | The caster behind every template. `plate` stands the full height across its whole width and casts a rectangle. `dome` is a half ellipsoid: its height falls off across the Sun as the square root of 1 minus (2 x across / width) squared, so its shadow tapers to the same tip the way a rounded rock's does. |

The guard answers a failure seen on planted rocks: a 0.53 m rock whose shadow the second window saw end was bounded at 1.13 m by the fourth window, which had taken in unrelated dark ground and explained it with one tall caster. A legitimate growth, where the smaller window's shortest compatible shadow was itself cut by its edge, is not a conflict and is left alone.

All four are opt-in in `AdaptiveConfig`, leave the hashes of existing configurations unchanged at their defaults, and can be set for T14's sizing alone through `sfs_sizing_adaptive`, which overrides the shared `adaptive` block without touching T9. `sfs_sizing_images: original` sizes on the original images instead of the relief-corrected ones; the surface from shape from shading stays the receiving ground. `sfs_sizing_subgrid` reads heights between grid steps (`subgrid_height`): the estimate is the vertex of the parabola through the best height of the profile over width and its two neighbours, and an uncensored bound the lower of that vertex and the compatible range's end interpolated where the profile crosses the threshold. At high signal the compatible range shrinks to one grid height, so the vertex carries the information.

`configs/saturation_campaign_v26_workstation.json` sets the guard, edge padding, the fourth window, widths up to 4.8 m and sub-grid reading for T14, with 20 planting rounds.

## What the laptop bench showed

Planted rocks drawn as T14 draws them (same seeds, NASA and procedural bodies, DEM tilt, whole shadows), multiplied into the 8-frame Athena stack, relief-corrected and sized at the noise measured after correction (0.096). A development run, not a study.

- The fourth window with edge padding and the guard (`guard8pad`) against the defaults, on the same 49 rocks: twice the context-supported estimates (14 against 7), no cell stopped at the image edge (10 had), and the same share of fitted bounds holding (28 of 32). The guard alone changed no rock.
- Over eight rounds with `guard8pad` (168 rocks): found 0 of 30 below 0.3 m, 19 of 55 from 0.3 to 0.6 m, 47 of 47 from 0.6 to 1.2 m and 30 of 31 from 1.2 to 2 m. Bounds calibrated at 0.97, cross-fitted by whole rounds on quiet sites, held for 98 of 99 rocks out of sample, at a cost of 0.21 m; the fitted bounds held for 82. Estimates read 6% short at the median, and 49 of 52 calibrated 90% intervals held. The 0.6 to 1.2 m bin is established (both shares at least 92% at 95% confidence); 1.2 to 2 m is not yet, one miss among 31, so no run of this size can state a height it is measurable from.
- The miss is a 1.3 m NASA body only 1.3 m wide. Its shadow is about 1.5 px across, the first window sees 6 px of it, and near the root the eight frames' shadows overlap and go to the static field. Narrow tall rocks lose most of their signal in the first window; T14 counts only that window as found, while the campaign's adaptive queue would also take the cell for its cut shadow.
- `caster_profile: dome` reads planted rocks alone on flat ground within 1% of their height at the median, where the plate reads 8% short, but scatters more: 90% of its readings span a factor of 1.24, the plate's 1.16. On the real images it gave fewer estimates (11 against 14), about 70% more scatter, and fitted bounds that held for 19 of 32 against 28. Calibration absorbs a bias and not a scatter, so `plate` stays.
- `sfs_sizing_images: original` gave fewer estimates and looser bounds on its one round.
- Reading heights between grid steps (`sfs_sizing_subgrid`) brought the 0.97 bound margin from 0.30 to 0.23 m and narrowed the widest estimate intervals; the typical estimate error did not change.
- Widths up to 4.8 m change no fitted height; they only lift wide tall rocks off the edge of the bank, which otherwise refuses their estimate.

## Not yet done

- Relief as a nuisance both hypotheses see. The 2.6 review's first recommendation is to stop subtracting relief from the data and instead carry the predicted relief shading as a per-frame nuisance. The projector applies one spatial basis to every frame today, so this needs a frame-dependent basis.
- Neighbouring casters as nuisance. A larger window can hold other rocks' shadows; fitting them alongside the target, instead of only guarding against them, would let the window keep growing safely.
