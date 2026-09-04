# HATI on a Mac Mini: full execution runbook

Every command here is meant to be copied and pasted as written. Lines starting with
`#` are comments. Where a command needs a decision from you, it says so.

---

## 0. Will it run on a Mac Mini?

**The Python side: yes, without reservation.** The dashboard, the channel code, the
mare control, the shadow kinematics and the sweep query are pure Python (numpy, scipy,
scikit-image, matplotlib, rasterio) and run natively on both Intel and Apple Silicon.

**The ISIS side: yes, with one caveat you need to test.** USGS ISIS supports macOS. On
Apple Silicon it has historically been an x86_64 build, run through Rosetta 2. Newer
releases may ship a native ARM64 build. Section 4 tries native first and falls back to
Rosetta if that fails, so you do not need to know in advance which applies to you.

**What you actually need:**

| Resource | Minimum | Comfortable | Why |
|---|---|---|---|
| RAM | 8 GB | 16 GB | frames are processed one at a time, but ISIS cubes are large |
| Free disk | 60 GB | 120 GB | see the disk budget below |
| Internet | required | | the sweep frames download on demand, and `spiceinit` uses a web service |

Disk budget, roughly: the transfer is 1.2 GB, the ISIS data area is several to tens of
GB depending on what you fetch, the ten NAC frames are about 3.5 GB, and ISIS
intermediates peak around 2 to 3 GB per frame while it works. The ingest deletes each
intermediate cube as soon as the projected version exists, so it does not accumulate.

Check what you have before starting:

```bash
sysctl -n machdep.cpu.brand_string   # which chip
sysctl -n hw.memsize | awk '{print $1/1073741824 " GB RAM"}'
df -h ~ | tail -1                    # free space on your home volume
```

---

## 1. Transfer and unpack

Copy `HATI_transfer.zip` to the Mac (AirDrop, a USB stick, `scp`, whatever is easiest).
Then:

```bash
cd ~
unzip -q ~/Downloads/HATI_transfer.zip -d ~
cd ~/HATI_V2.0
chmod +x hati_dashboard.sh
ls
```

You should see `dashboard  data  paper  scripts  src  tests  output` among the entries.
The folder is named `HATI_V2.0` with no spaces on purpose, so nothing needs quoting.

---

## 2. Python environment

macOS ships a Python you should not install into. Use Homebrew's, or conda if you
prefer. Homebrew route:

```bash
# install Homebrew only if you do not have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install python@3.12
cd ~/HATI_V2.0
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install numpy scipy scikit-image matplotlib rasterio
```

Verify:

```bash
python3 -c "import numpy, scipy, skimage, matplotlib, rasterio; print('deps OK')"
python3 tests/test_channels.py
```

The test run should end with `ALL CHECKS PASSED`. If it does, the science core is
working on your machine and every number in the papers is reproducible here.

---

## 3. First run, no ISIS needed

Three of the four jobs need nothing but Python. Start the dashboard:

```bash
cd ~/HATI_V2.0
source .venv/bin/activate
./hati_dashboard.sh
```

It prints a URL and opens your browser at `http://127.0.0.1:8737`. You should see the
solar sweep radar turning with 696 real frames on it, the live shadow physics panel,
and a row of status pills. The ISIS pill will be red. That is expected until section 4.

In the dashboard, click these in order and watch the telemetry console:

1. **Mare control v3** confirms the channel decision on your machine. Expect RMS slope
   0.826 and TRI 0.808 to be kept, everything else cut.
2. **Kinematics synthetic benchmark** confirms the estimator. Expect AUC 0.990 and a
   height error of about 0.24 m.
3. **Consensus shadow census** runs the real three-frame census at the Athena site.

If you would rather not use the browser, the same jobs run from the terminal:

```bash
python3 scripts/mare_control_v3.py
python3 scripts/shadow_kinematics.py
python3 scripts/shadow_consensus.py
```

Leave the dashboard running in its own terminal tab. Open a second tab for everything
below.

---

## 4. Install ISIS

Miniforge is the cleanest conda on Apple Silicon:

```bash
brew install --cask miniforge
conda init "$(basename "$SHELL")"
exec "$SHELL"
```

**Do not pin a Python version.** ISIS depends on a specific Python, and pinning one
yourself produces a solver conflict along the lines of "pins are involved in the
conflict ... python=3.9 is not installable". Create the environment and install ISIS
in a single command so the solver picks a consistent Python for you.

Try the native build first. `mamba` ships with Miniforge and solves far faster than
`conda` on a graph this size, so prefer it:

```bash
mamba create -y -n isis -c usgs-astrogeology -c conda-forge isis
conda activate isis
```

If `mamba` is missing, `conda create -y -n isis -c usgs-astrogeology -c conda-forge isis`
does the same thing, just slower.

If that resolves and installs, you are done. **If it says it cannot find a package for
your platform**, you are on Apple Silicon without a native build. Remove the environment
and redo it under Rosetta:

```bash
conda deactivate
conda env remove -y -n isis
softwareupdate --install-rosetta --agree-to-license     # once per machine
CONDA_SUBDIR=osx-64 mamba create -y -n isis -c usgs-astrogeology -c conda-forge isis
conda activate isis
conda config --env --set subdir osx-64
```

Check which Python it chose, and do not try to change it:

```bash
python --version
```

### Putting all of it on an external SSD

ISISDATA plus the environment plus the downloaded frames can easily outgrow an internal
disk. Three things need redirecting: the conda environment, conda's package cache, and
ISISDATA.

**Check the filesystem first.** It must be APFS or Mac OS Extended. A conda environment
on exFAT or FAT32 fails in confusing ways later, because those filesystems do not carry
POSIX permissions or symlinks. Avoid spaces in the volume name.

```bash
diskutil info /Volumes/YOUR_SSD | grep -i "file system"
```

Then set one variable and let the rest follow:

```bash
SSD=/Volumes/YOUR_SSD              # edit this line only
mkdir -p "$SSD/conda/envs" "$SSD/conda/pkgs" "$SSD/isisdata"
conda config --add envs_dirs "$SSD/conda/envs"
conda config --add pkgs_dirs "$SSD/conda/pkgs"
conda env remove -y -n isis        # if one already exists on the internal disk
```

Create the environment as in section 4 above. It now lands on the SSD, and `-n isis`
still works normally. Point ISISDATA there as well, and move the project across so the
NAC downloads and ISIS intermediates land on the SSD too:

```bash
export ISISDATA="$SSD/isisdata"
mv ~/HATI_V2.0 "$SSD/HATI_V2.0" && cd "$SSD/HATI_V2.0"
```

Persist it, since `export` dies with the shell:

```bash
cat >> ~/.zshrc <<'EOF'
export SSD=/Volumes/YOUR_SSD
export ISISDATA="$SSD/isisdata"
export ISISROOT="$CONDA_PREFIX"
EOF
```

Verify early, after the first few hundred megabytes rather than at the end:

```bash
which lronac2isis            # should print a path under /Volumes/...
du -sh "$SSD/conda" "$SSD/isisdata"
df -h "$SSD" | tail -1       # confirm the SSD is filling, not the internal disk
```

The SSD must be mounted for any of this to work. Unplugged, `conda activate isis` fails
and ISISDATA points nowhere. Recoverable, but do not run the ingest over a flaky cable.

Then point ISIS at its data area and fetch what LRO needs:

```bash
export ISISROOT="$CONDA_PREFIX"
export ISISDATA="$HOME/isisdata"
mkdir -p "$ISISDATA"

downloadIsisData base "$ISISDATA"
downloadIsisData lro  "$ISISDATA"
```

That download is the big one. Watch it with `du -sh "$ISISDATA"` in another tab and stop
it if it grows beyond what your disk can take. We do not need the SPICE kernels locally
because the ingest calls `spiceinit web=yes`, which uses the USGS web service, but the
calibration files for `lronaccal` do have to be local.

Make the settings stick:

```bash
echo 'export ISISROOT="$CONDA_PREFIX"'      >> ~/.zshrc
echo 'export ISISDATA="$HOME/isisdata"'     >> ~/.zshrc
```

Confirm ISIS is on the path:

```bash
which lronac2isis spiceinit lronaccal cam2map
```

Four paths should print. If they do, refresh the dashboard in your browser and the ISIS
pill turns green.

---

## 5. The ingest

ISIS lives in its own conda environment, the science code in the venv. The ingest needs
both, so run it from the ISIS environment with the venv's packages installed there too:

```bash
conda activate isis
python -m pip install numpy scipy scikit-image matplotlib rasterio
cd ~/HATI_V2.0
```

**Dry run first. It downloads nothing.**

```bash
python scripts/ingest_sweep.py --n 10
```

It prints the ten frames it would fetch, their sun angles, and whether ISIS is visible.
Read that list before going further. If it says fewer than about six azimuth bins were
filled, widen the elevation band:

```bash
python scripts/ingest_sweep.py --n 10 --min-elev 1.0 --max-elev 9.0
```

**Then the real thing.** This downloads roughly 3.5 GB and runs the full chain. Expect
one to three hours depending on your connection.

```bash
python scripts/ingest_sweep.py --n 10 --execute 2>&1 | tee logs/ingest.log
```

Create the log directory first if it does not exist: `mkdir -p logs`.

You can watch the same output live in the dashboard, where the stepper lights up stage
by stage: SELECT, DOWNLOAD, LRONAC2ISIS, SPICEINIT, LRONACCAL, CAM2MAP, COREGISTER,
MANIFEST.

---

## 6. Read the gate before doing anything else

The ingest produces one file that decides whether the science holds:

```bash
column -s, -t data/sweep/coreg_report.csv
```

Each row is one frame with its measured shift, in pixels, against the reference
orthophoto. Compute the median absolute shift:

```bash
python3 - <<'PY'
import csv, statistics
rows=[r for r in csv.DictReader(open('data/sweep/coreg_report.csv')) if r['shift_row_px']]
s=[ (float(r['shift_row_px'])**2 + float(r['shift_col_px'])**2)**0.5 for r in rows]
print(f"{len(rows)} frames co-registered")
print(f"median |shift| = {statistics.median(s):.2f} px   (gate: <= 1.00 px)")
print(f"worst  |shift| = {max(s):.2f} px")
PY
```

**Median at or below 1.00 px: the gate passes.** The shadow height estimates are
trustworthy and the real-data kinematics run is on.

**Median above 1.00 px: stop and say so.** Do not run the science on top of it. A
mislocated shadow base blurs the confidence peak and biases every height. That is a
result worth reporting honestly, not a problem to work around.

---

## 7. Get the data back out

Run the harvester on the Mac. It skips the tens of gigabytes of intermediates and takes
only what you cannot regenerate:

```bash
cd ~/HATI_V2.0
python3 harvest_results.py --out ~/Desktop
```

You get `~/Desktop/HATI_results_<host>_<date>.zip`, roughly 50 to 150 MB, containing
every figure, every CSV, the reports, the compiled PDFs, the co-registration budget, the
sweep manifest and a `HARVEST_MANIFEST.txt` listing exactly what came along. Copy that
zip anywhere.

If you want the projected cubes as well, which is gigabytes:

```bash
python3 harvest_results.py --out ~/Desktop --include-cubes
```

To move results by network instead of a stick, from your other machine:

```bash
scp yourname@mac-mini.local:~/Desktop/HATI_results_*.zip .
```

And if you want the code changes back under version control, the repository came with
its history intact, so you can commit and push from the Mac. Set authorship through the
environment, never through `git config`:

```bash
cd ~/HATI_V2.0
GIT_AUTHOR_NAME="Cyrex567" GIT_AUTHOR_EMAIL="gergomorvai044@gmail.com" \
GIT_COMMITTER_NAME="Cyrex567" GIT_COMMITTER_EMAIL="gergomorvai044@gmail.com" \
git commit -am "real-data sweep ingest: co-registration budget and manifest"
git push origin feat/v2.0-heatmap
```

---

## 8. When things go wrong

**`./hati_dashboard.sh: Permission denied`**
`chmod +x hati_dashboard.sh`

**`rasterio` fails to install or import**
`python3 -m pip install --upgrade pip wheel` then reinstall. If it still fails, use conda
instead: `conda install -c conda-forge rasterio`.

**Port 8737 already in use**
`python3 dashboard/hati_dashboard.py --port 8800`

**`downloadIsisData` fills the disk**
Stop it. You only need `base` and `lro`. Check with `du -sh "$ISISDATA"`. Nothing else is
required for this pipeline.

**`spiceinit` fails with a network error**
It is calling the USGS web service. Retry; if it keeps failing the service may be down.
The ingest caches every completed frame, so re-running resumes rather than starting over.

**ISIS commands are not found even though the install worked**
You are in the wrong environment. `conda activate isis`, then check `echo $ISISROOT`.

**"pins are involved in the conflict" / "python=3.9 is not installable"**
You pinned a Python version that ISIS cannot use. Do not pin one. Remove the
environment and create it with ISIS in the same command, as in section 4, so the solver
chooses Python itself: `conda env remove -y -n isis` then
`mamba create -y -n isis -c usgs-astrogeology -c conda-forge isis`.

**"Solving environment" appears frozen**
Normal for ISIS, the dependency graph is large. Ten to twenty minutes with no visible
progress is expected with `conda`. If it runs much longer, cancel and use `mamba`, which
usually finishes in a couple of minutes.

**A frame fails partway through the ISIS chain**
The ingest logs which stage failed for which frame and carries on with the rest. A
handful of failures out of ten is survivable as long as the surviving frames still span
a good spread of sun azimuths. Check the spread in `data/sweep/manifest.json`.

**Everything ran but the numbers differ from the papers**
Run `python3 tests/test_channels.py` first. If that passes, the core is fine and the
difference is in the data or the configuration. `Config.hash()` in `src/hati_core`
prints a digest of every setting, which is the fastest way to find what moved.
