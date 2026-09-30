# Post-landing check

LROC imaged the Athena site many times after the landing. Those frames can never be part of the counterfactual, but they are an independent second look at the same ground, which makes them a validation set. This check runs HATI on them and compares the result with the pre-landing campaign.

## What it tests

Rocks do not move. If HATI's detections are physical, the casters it measures, its warnings after the relief correction, the shape-from-shading detail and the rock-or-relief labels should repeat between the two image sets. Each comparison is set against shifted copies of the same maps, which keep their clustering, so repetition by chance is measured instead of assumed. The plan and its thresholds sit in `configs/post_landing_check.json` and were fixed on 30 September 2026, before any post-landing frame was analysed.

The lander itself is a weaker test than it first looks. LROC shows Athena on its side on the shadowed floor of a 20 m crater, so its own shadow mostly falls into existing darkness. The check therefore asks only that HATI warns there or says unknown, never that it reports quiet ground. Everything within 100 m of the lander is described on its own and left out of the repeat measures, because the landing changed the surface there.

## Keeping the two data sets apart

- `ingest_sweep.py --after <start> --before <end> --sweep-dir <folder>` reads only frames inside that window. The window must start after the counterfactual cutoff, and it refuses to write into `data/sweep`. Every manifest entry records its `after`.
- `sweep_products.load_sweep` refuses any entry that carries `after` unless the same window is requested, and refuses entries outside it. The counterfactual path is unchanged.
- `landing_maps.py --after` labels the run as post-landing validation in its provenance, and `diagnose_landing_run.py` passes the window on.

## Run on the stationary WSL machine

```bash
cd ~/HATI_V2.0
git pull --ff-only origin feat/v2.0-heatmap
conda activate isis
bash scripts/run_post_landing_wsl.sh
```

The default window is 7 to 21 March 2025, which excludes the landing day. With the audited gates (emission at most 40 degrees, Sun 1.5 to 7.5 degrees high, 600 m inside the strip) the archive table gives nine frames over 75 degrees of Sun azimuth, about 3.1 GB of downloads. The script then builds the maps and the bundle as for the pre-landing stack, runs T1, T12, T13 and T14 with HATI Watch, and compares with the pre-landing campaign in `output/athena/saturation_campaign/full-03` (set `PRE=` for another). `SKIP_INGEST=1` reuses frames already ingested into `data/sweep_post_landing`.

Send the two files it prints: the campaign results ZIP and the check ZIP. `REPORT.md` in the check lists each expectation as met or not met, with the numbers behind it.

## Every frame that shows the lander

The archive holds more than the nine frames the audited gates let through. As of 30 September 2026 it has 31 frames with the site inside the strip after the landing, all from 6 to 21 March 2025 except one right-camera frame from 26 February 2026. Leaving out the landing day and the frames with the Sun below 1.5 degrees leaves 24: the nine, two that sit only 427 m and 309 m inside their strips, and thirteen viewed at 41 to 60 degrees emission. Together they span 135 degrees of Sun azimuth instead of 75, and Sun elevations of 1.6 to 5.4 degrees, so shadows change length by a factor of 3.5 across the stack instead of 1.3.

`configs/post_landing_check_all_frames.json` declares that frame list, its window (to the end of February 2026) and the relaxed ingest gates, with the same expectations and thresholds as the nine-frame plan. It was written on 30 September 2026, before any post-landing frame was ingested. `configs/post_landing_all_frames.csv` is the archive catalogue the frames come from.

```bash
bash scripts/run_post_landing_all_frames_wsl.sh
```

It runs the same pipeline into its own folders (`data/sweep_post_landing_all`, an output folder ending in `-all-frames`, and exports under `Downloads\HATI\all-frames`), so it never overwrites the nine-frame run. The oblique frames were excluded for a reason: the projection does not remove all relief, so relief shifts an oblique frame by its height times tan(emission). The ingest's co-registration closure drops frames that do not fit, and says so; if it drops many, the next step is projecting onto the NAC terrain model.

## Both runs in one queue

```bash
cd ~/HATI_V2.0
git pull --ff-only origin feat/v2.0-heatmap
conda activate isis
bash scripts/run_post_landing_queue_wsl.sh
```

The queue runs the nine-frame check and then the 24-frame run on the same checkout, so both use the same code, and a failure in the first does not stop the second. `START_AT=22:00` makes it wait until then. Leave the terminal open until it prints its summary; the log is in `logs/post_landing_queue_<time>.log`.
