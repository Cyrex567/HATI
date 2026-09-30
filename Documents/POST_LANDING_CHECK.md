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
