# HATI transfer package: laptop -> GPU box

## What to transfer
Zip the whole `HATI V2.0` folder (or clone the repo + copy `data/`). Everything is
relative-path; no machine-specific config.

## On the GPU box, once
```bash
# python deps (only these; the dashboard itself is stdlib-only)
pip install numpy scipy scikit-image matplotlib rasterio

# ISIS3 (for the sweep ingestion; Linux/WSL/conda)
conda create -n isis -c usgs-astrogeology isis
conda activate isis
# then set ISISROOT/ISISDATA per the USGS instructions and download the LRO kernels
```

## Launch the dashboard
* **Windows:** double-click `HATI_Dashboard.bat`
* **Linux/WSL:** `./hati_dashboard.sh`
* opens `http://127.0.0.1:8737` with the mission dashboard: solar-sweep radar
  (real ODE data), live shadow-kinematics physics, job control, telemetry
  console, ingestion stepper, result-figure gallery, evidence ledger.
* Put your logo at `dashboard/static/assets/hati_logo.png` to brand the header.

## Optional: single-file executable
Run `build_exe.bat` **on the box** (PyInstaller output is platform-specific).
It produces `dist/HATI.exe` with the logo as its icon; place the exe in the
project root so it can find `scripts/`, `data/`, `output/`.

## Run order on the box (the buttons, top to bottom)
1. **Consensus shadow census** - first refreshed real product (minutes).
2. **Kinematics synthetic benchmark** - confirms environment parity (minutes).
3. **Ingest sweep DRY RUN** - prints the frame plan, checks ISIS (seconds).
4. **Ingest sweep EXECUTE** - downloads ~10 NAC EDRs (~3.5 GB) and runs
   lronac2isis -> spiceinit(web) -> lronaccal -> cam2map -> co-registration.
   Watch the stepper; the deliverable is `data/sweep/coreg_report.csv`
   (**the error budget**: gate = median |shift| <= 1 px) + `manifest.json`.
5. Then: the real-data kinematics adapter (next build) consumes `manifest.json`.

Note: the current pipeline is CPU/numpy; the GPU matters for future ML-free
acceleration and big rasters, not for correctness.
