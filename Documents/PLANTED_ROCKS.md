# Planted rocks and real rocks in HATI 2.6

HATI searches the real images for rocks. Planted rocks, rendered and multiplied into the same images, calibrate that search: how often a rock of a given height is found, and whether its sized height can be trusted. Version 2.6 changes both sides.

## The real rocks

T14 runs the detector on the relief-corrected stack, checks every warning cell against relief with T13's calibrated rule, and sizes the cells that stay rock-like or ambiguous. Those sized cells used to be the end product, and a cell is not a rock: one rock can set off several neighbouring cells along its shadow.

`stages/T14/real_rocks.json` now merges touching sized cells (8-connected on the cell grid) into candidate objects. Each object records:

- its cells, mean position and strongest cell (`peak_row_px`, `peak_col_px`, `peak_score`);
- its height: the largest context-supported estimate among its cells (`height_m`), otherwise the largest lower bound (`height_lower_bound_m`);
- its label: rock-like if any cell is, otherwise ambiguous, or unchecked where T13 did not run;
- its distance to the touchdown and whether it reaches the illustrative clearance;
- `calibration`: how many planted rocks of that height this run found after the relief correction, at the measured noise. Candidates shorter than any planted rock have none.

`result.json` summarises them under `real_rocks`. A candidate object may hold several rocks, and nothing here checks the candidates against independent truth.

## The planted rocks

Every planted rock used to be the same procedural body, 0.6 m wide with aspect 1.35, at one of three heights. A 1.2 m rock on that body is a narrow spire. With `planted_geometry: population` (the default), `src/hati_core/rock_population.py` gives every planted rock its own body:

| Property | How it is drawn | Basis |
|---|---|---|
| Exposed height | log-uniform, 0.15 to 2 m | calibration design: every part of the range gets rocks |
| Width over length | normal, mean 0.71, sd 0.12, clipped to 0.3 to 1 | laboratory impact fragments, axes about 2 : 1.4 : 1 (Fujiwara et al. 1978) |
| Height over maximum diameter | normal, mean 0.54, sd 0.1, clipped; never taller than wide | Moon rocks in Lunokhod, Apollo and LROC NAC images (Demidov and Basilevsky 2014) |
| Burial | uniform, 0 to 15% of the body | the same study found negligible penetration into the regolith |
| Yaw | uniform | |
| Body | NASA Apollo sample 10017 or a procedural convex body, half each | NASA Astromaterials 3D; sample 10021 stays in the evaluation split |

The means come from the sources; the spreads are assumptions, recorded with every run. Log-uniform heights are a calibration design, not the lunar size-frequency distribution, which small rocks dominate. The bodies are rescaled laboratory samples, not measured polar boulders.

Every planted rock stands on the DEM plane at its site, and its window holds its whole shadow on that plane (`shadow_reach_px`). Ground falling along a shadow lengthens it; a site whose shadow could reach the next site's scoring window is left empty and counted in `injection_rounds` as `dropped_long_shadow`.

`injection.json` records each rock's width, length, height over diameter, burial, yaw and body. T14 summarises recovery, relief absorption and sizing by height bin, and checks every height bound against that rock's own true height. T18 keeps its configured rock heights and gives each planted rock a body from the same population, scaled to that height; its mounds and bowls are unchanged.

## Settings

| Key | Default | Meaning |
|---|---|---|
| `planted_geometry` | `population` | `fixed` restores the 2.5 bodies at `sfs_injection_heights_m` |
| `planted_rock_prior` | see the table | any of `height_m`, `width_over_length`, `height_over_diameter`, `ratio_limits`, `burial`, `nasa_fraction` |
| `planted_height_bins_m` | `[0.15, 0.3, 0.6, 1.2, 2.0]` | the bins T14 reports by; they must span the planted height range |

The NASA bodies come from the campaign's rock catalog (`--rock-catalog`, by default `data/rock_shapes/apollo_proxy_v1`), development split only. More Apollo samples can be added to a new catalog with `scripts/prepare_lunar_rock_catalog.py`, run on its own; the science runner never downloads anything.

## In HATI Watch

The T14 report now opens with the real candidates, then shows the planted bodies (height against height over diameter; found, missed or not scored; triangles for NASA bodies), the calibration by height bin, measured against true height for every planted rock, a map of the real candidates sized by height with the touchdown marked, and the real candidates by height bin next to the share of planted rocks found at that height.

## References

- Fujiwara, A., Kamimoto, G. and Tsukamoto, A. (1978). Expected shape distribution of asteroids obtained from laboratory impact experiments. Nature 272, 602-603.
- Demidov, N. E. and Basilevsky, A. T. (2014). Height-to-diameter ratios of moon rocks from analysis of Lunokhod-1 and -2 and Apollo 11-17 panoramas and LROC NAC images. Solar System Research 48, 324-329.
- NASA Astromaterials 3D, Lunar Sample Laboratory Facility, NASA Acquisition and Curation Office: sample 10017,15 (https://ares.jsc.nasa.gov/astromaterials3d/sample-details.htm?sample=10017-15).
