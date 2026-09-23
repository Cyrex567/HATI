# HATI Watch

For the adaptive campaign, follow [the new workstation command](ADAPTIVE_SHADOW_PLAN.md) and point the viewer at `output/athena/saturation_campaign/athena-adaptive-01`. T9 shows expanding extraction patches, per-frame endpoint support reasons and uncalibrated dimension compatibility. The adaptive state legend differs from the baseline assessment legend and is shown in the viewer. T10 uses synthetic coordinates and therefore shows its scene figures separately; T11 reports progress through fixed withheld-frame tests. The original three maps remain available as separate products.

HATI Watch is a local browser window for observing a saturation campaign. It shows the aligned image frames, Sun geometry, intermediate shadow fits, developing maps, saved scientific figures and test logs. The campaign runs independently. Frame playback, stage selection and plot selection affect only the display.

## Run on the stationary WSL machine

Pull the update before starting a campaign. Do not update source files while an existing campaign is running. Changed source or configuration requires a new output folder; resume is reserved for the same frozen inputs, environment and source.

Terminal 1 starts the denser campaign on the saved eight-frame Athena bundle:

```bash
cd ~/HATI_V2.0
git switch feat/v2.0-heatmap
git pull --ff-only origin feat/v2.0-heatmap
conda activate isis
python -m pip install -r requirements-science.txt
bash scripts/run_saturation_campaign_wsl.sh \
  --bundle output/athena/landing_maps_v253/20260912T072640Z/registration-0.5/diagnostics_v254/hati_diagnostic_bundle.zip \
  --config configs/saturation_campaign_workstation.json \
  --output output/athena/saturation_campaign/athena-watch-01 \
  --export-dir /mnt/c/Users/Public/Downloads/HATI
```

Terminal 2 starts the viewer. The run folder may still be empty when it starts:

```bash
cd ~/HATI_V2.0
conda activate isis
python dashboard/hati_watch.py \
  --run-dir output/athena/saturation_campaign/athena-watch-01
```

Open **http://localhost:8765** in a browser on the same stationary machine. This uses WSL's localhost forwarding to Windows. If that forwarding is disabled on your installation, open the viewer from a browser inside WSL or restore localhost forwarding. The server deliberately listens only on `127.0.0.1`; it is not a public network service. Use `--port 8766` if 8765 is occupied, and open the matching URL.

Closing the tab or stopping terminal 2 leaves terminal 1 running. `Ctrl-C` in terminal 1 interrupts the campaign and normally packages the results collected so far. The archive is exported to `C:\Users\Public\Downloads\HATI\athena-watch-01_results.zip`, with a sibling checksum. See the [campaign runbook](SATURATION_CAMPAIGN.md) for resume, exit codes and external validation inputs.

## What the display means

- **Illumination sweep:** the actual aligned source frames, with a shared percentile stretch. Play cycles the saved frames once per second. It does not animate invented observations. The cyan box marks the most recently displayed regional patch and its cross marks the winning sampled root. White marks the fixed touchdown test coordinate. Synthetic inputs carry a visible demonstration badge.
- **Last regional fit:** an observed patch, predicted shadow coverage and the whitened residual after fitting. Switching frames selects the corresponding image and model planes. A frame excluded from that fit is explicitly unavailable. The displayed bank height and width are winning model parameters, not measured object dimensions.
- **Terrain physics:** slope and surface-RMS fields, plus slope, RMS and signed relief magnitudes at the fixed touchdown coordinate. Each measurement is divided by its configured limit; their maximum gives the local terrain index before navigation buffering. Native DEM posting, effective plane diameter and the configuration label remain visible. These measurements stay available while later shadow tests run.
- **Calculations:** null and fitted energy, their difference, shadow score, regional index and signed contributions from eligible frames. Negative contributions stay signed. The index is model evidence and is not a calibrated hazard probability. Geometry-stress stages also show the reassigned model Sun geometry.
- **Developing maps:** partial regional scores and assessment coverage; during map generation, terrain slope, DEM illumination and the fused field appear as those steps finish. Grey border/unvisited, green assessed, amber unavailable and pink nonidentifiable remain distinct. Height-profile and registration summaries appear during their stages.
- **Saved products:** individual terrain, shadow and fused maps, their three-panel comparison, observability and other completed figures. Select any saved plot or open it at full size.
- **Test sequence and verdict:** follow the active stage or inspect a completed stage's last snapshot and log. A completed test means its calculation finished; missing external evidence can still leave the overall verdict inconclusive.

The browser polls every two seconds. The computation publishes snapshots at tile boundaries, throttled to about one every three seconds, and at selected stage transitions. A long tile or operation may leave a picture unchanged while the separate heartbeat stays live. This is sampled feedback, not a recording of every operation. Only the latest snapshot per stage is retained; intermediate animation history is not archived. Completed regional snapshots remain inspectable while later synthetic checks in the same stage finish.

A heartbeat older than ten seconds is marked stale. A disconnected viewer keeps the last values visibly frozen. Older campaigns can show logs, completed plots and source images without changing their files, but cannot retroactively supply intermediate calculations they never recorded. Held-out scenes use a different grid; their fits are not overlaid on development images.

## Scientific separation and checks

The pipeline's display writer emits atomic JSON and PNG files. The server reads them and serves GET/HEAD requests; it has no start, stop, parameter-edit or shell route. Display I/O failures are caught and processing continues. Scientific arrays are not rounded or downsampled: only display copies are, so exported scientific arrays remain the numerical record.

Regression checks compare every regional scientific array and candidate with observation enabled and disabled, verify that the reported energy difference and signed contributions reproduce the score, check source-array immutability, simulate display failure, test stale/partial observations and exercise HTTP read-only and path boundaries. A synthetic end-to-end campaign covers software checks, map generation, T1-T8 dispatch and result packaging. This validates the observer integration, not lunar hazard-detection performance.

The denser workstation configuration uses CPU NumPy/SciPy. It increases diagnostic coverage without claiming GPU acceleration or changing production thresholds. Independent annotated scenes, calibrated uncertainties and landing-system validation are still needed for claims about missed hazards or loss of mission.
