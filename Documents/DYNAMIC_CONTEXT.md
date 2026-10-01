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

All four are opt-in in `AdaptiveConfig`, leave the hashes of existing configurations unchanged at their defaults, and can be set for T14's sizing alone through `sfs_sizing_adaptive`, which overrides the shared `adaptive` block without touching T9. `sfs_sizing_images: original` sizes on the original images instead of the relief-corrected ones; the surface from shape from shading stays the receiving ground.

## Not yet done

- Relief as a nuisance both hypotheses see. The 2.6 review's first recommendation is to stop subtracting relief from the data and instead carry the predicted relief shading as a per-frame nuisance. The projector applies one spatial basis to every frame today, so this needs a frame-dependent basis.
- Neighbouring casters as nuisance. A larger window can hold other rocks' shadows; fitting them alongside the target, instead of only guarding against them, would let the window keep growing safely.
