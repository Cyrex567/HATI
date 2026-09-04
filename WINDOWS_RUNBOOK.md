# HATI on a Windows PC: full execution runbook

Copy and paste every command as written. PowerShell unless a block says otherwise.

The one thing to know before you start: **the science code runs natively on Windows and
is already proven there, but ISIS does not run on Windows at all.** It is a Linux and
macOS program. So the work splits in two:

| Phase | Where | Needs |
|---|---|---|
| **1.** Dashboard, mare control, kinematics, shadow census, ingest dry-run | Windows, native | Python only |
| **2.** The sweep ingest (the real-data milestone) | WSL2 (Ubuntu inside Windows) | ISIS |

Phase 1 is not a compromise or a preview. Every number in the papers was produced on a
Windows machine exactly this way. You can do all of it before touching WSL.

---

## 0. What you need

| Resource | Minimum | Comfortable |
|---|---|---|
| RAM | 8 GB | 16 GB |
| Free disk | 60 GB | 120 GB |
| Windows | 10 (2004+) or 11 | 11 |

Check:

```powershell
Get-CimInstance Win32_ComputerSystem | ForEach-Object { "{0:N0} GB RAM" -f ($_.TotalPhysicalMemory/1GB) }
Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" | ForEach-Object { "{0} {1:N0} GB free" -f $_.DeviceID, ($_.FreeSpace/1GB) }
winver
```

---

## 1. Unpack

Right-click the zip, Extract All, or:

```powershell
Expand-Archive -Path "$HOME\Downloads\HATI_transfer.zip" -DestinationPath "$HOME" -Force
cd "$HOME\HATI_V2.0"
dir
```

The folder is named `HATI_V2.0` with no spaces on purpose, so nothing needs quoting.

---

## 2. Python

Install Python 3.11 or 3.12 from python.org, ticking **"Add python.exe to PATH"** during
setup. Then:

```powershell
cd "$HOME\HATI_V2.0"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy scipy scikit-image matplotlib rasterio
```

If PowerShell refuses to run the activation script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Verify, and treat this as the go/no-go:

```powershell
python -c "import numpy, scipy, skimage, matplotlib, rasterio; print('deps OK')"
python tests\test_channels.py
```

That must end with `ALL CHECKS PASSED`. If it does, the science core reproduces on your
machine and everything in the papers is verifiable here.

---

## 3. Phase 1: run it, no ISIS needed

```powershell
cd "$HOME\HATI_V2.0"
.\HATI_Dashboard.bat
```

The launcher picks up `.venv` automatically if it exists. Your browser opens at
`http://127.0.0.1:8737`. You should see the solar sweep radar turning with 696 real
frames, the live shadow physics panel, and status pills. The ISIS pill will be red,
which is expected until phase 2.

Click these in order and watch the telemetry console. Each heavy job asks for
confirmation first.

1. **Mare control v3** should reproduce RMS slope 0.826 and TRI 0.808 kept, the other
   eight cut. About a minute.
2. **Kinematics synthetic benchmark** should reproduce AUC 0.990 and 0.24 m height
   error. About a minute.
3. **Consensus shadow census** runs the real three-frame census at the Athena site.
4. **Ingest sweep DRY RUN** prints the ten frames it would fetch. Downloads nothing.

Or from the terminal, with the venv active:

```powershell
python scripts\mare_control_v3.py
python scripts\shadow_kinematics.py
python scripts\shadow_consensus.py
python scripts\ingest_sweep.py --n 10
```

Stop here if you like. Everything above needs nothing but what is in the zip.

---

## 4. Phase 2: WSL2, because ISIS needs Linux

In an **Administrator** PowerShell:

```powershell
wsl --install -d Ubuntu
```

Reboot when it asks. On first launch Ubuntu asks you to create a username and password;
these are separate from your Windows account and the password is invisible as you type.

Confirm you are on WSL 2, not 1:

```powershell
wsl -l -v
```

The VERSION column must say 2. If it says 1:

```powershell
wsl --set-version Ubuntu 2
```

### Get the project into Linux

Do the work inside the Linux filesystem, not on `/mnt/c`. Windows drives are visible from
WSL but file access across that boundary is slow, and ISIS moves gigabyte cubes around.

In the **Ubuntu** terminal:

```bash
cp -r /mnt/c/Users/$USER/HATI_V2.0 ~/
cd ~/HATI_V2.0
```

Adjust the path if your Windows username differs. Check it with `ls /mnt/c/Users`.

### Install conda and ISIS inside Ubuntu

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
~/miniforge3/bin/conda init bash
exec bash
```

**Do not pin a Python version.** ISIS depends on a particular one, and pinning your own
causes a solver conflict ("pins are involved in the conflict"). Create the environment
and install ISIS in a single command so the solver chooses:

```bash
mamba create -y -n isis -c usgs-astrogeology -c conda-forge isis
conda activate isis
python --version          # whatever ISIS picked, do not fight it
```

"Solving environment" can sit silent for a while. That is normal. `mamba` usually
finishes in a few minutes where `conda` would take much longer.

### ISIS data area

```bash
export ISISROOT="$CONDA_PREFIX"
export ISISDATA="$HOME/isisdata"
mkdir -p "$ISISDATA"
downloadIsisData base "$ISISDATA"
downloadIsisData lro  "$ISISDATA"
```

This is the big download, tens of GB and potentially hours. Watch it from a second
Ubuntu tab and stop it if it outgrows your disk:

```bash
watch -n 30 du -sh ~/isisdata
```

Make it stick:

```bash
echo 'export ISISROOT="$CONDA_PREFIX"'  >> ~/.bashrc
echo 'export ISISDATA="$HOME/isisdata"' >> ~/.bashrc
```

Confirm ISIS is on the path:

```bash
which lronac2isis spiceinit lronaccal cam2map
```

Four paths should print.

### Putting the heavy data on another drive

WSL2 keeps its filesystem in a virtual disk on `C:` that grows as you fill it. If `C:` is
tight, put ISISDATA on a second drive instead. From Ubuntu, a Windows drive `D:` is at
`/mnt/d`:

```bash
export ISISDATA=/mnt/d/isisdata
mkdir -p "$ISISDATA"
```

That trades speed for space, and it is the right trade for ISISDATA specifically, since
it is read occasionally rather than churned. Keep the **project** itself inside Linux
(`~/HATI_V2.0`), because that is where the gigabyte cubes get written and rewritten.

---

## 5. The ingest

The ingest needs both ISIS and the Python packages, so install the Python side into the
ISIS environment too:

```bash
conda activate isis
python -m pip install numpy scipy scikit-image matplotlib rasterio
cd ~/HATI_V2.0
mkdir -p logs
```

**Dry run first. It downloads nothing.**

```bash
python scripts/ingest_sweep.py --n 10
```

Read the printed frame list. If fewer than about six azimuth bins are filled, widen the
band:

```bash
python scripts/ingest_sweep.py --n 10 --min-elev 1.0 --max-elev 9.0
```

**Then the real run.** Roughly 3.5 GB of downloads plus the full ISIS chain. One to three
hours depending on your connection.

```bash
python scripts/ingest_sweep.py --n 10 --execute 2>&1 | tee logs/ingest.log
```

To watch the stepper light up while it runs, start the dashboard **inside Ubuntu** in
another tab. WSL2 forwards localhost, so your Windows browser reaches it normally:

```bash
conda activate isis
cd ~/HATI_V2.0
python dashboard/hati_dashboard.py --no-browser --port 8737
```

Then open `http://127.0.0.1:8737` in Windows. Run from inside WSL like this, every
button works including the ingest, because ISIS is on the path there.

---

## 6. Read the gate before anything else

The ingest produces one file that decides whether the science holds:

```bash
column -s, -t data/sweep/coreg_report.csv
```

Then compute the median shift:

```bash
python3 - <<'PY'
import csv, statistics
rows=[r for r in csv.DictReader(open('data/sweep/coreg_report.csv')) if r['shift_row_px']]
s=[(float(r['shift_row_px'])**2 + float(r['shift_col_px'])**2)**0.5 for r in rows]
print(f"{len(rows)} frames co-registered")
print(f"median |shift| = {statistics.median(s):.2f} px   (gate: <= 1.00 px)")
print(f"worst  |shift| = {max(s):.2f} px")
PY
```

**Median at or below 1.00 px: the gate passes.** Height estimates are trustworthy and the
real-data kinematics run is on.

**Median above 1.00 px: stop and say so.** Do not run the science on top of it. A
mislocated shadow base blurs the confidence peak and biases every height. That is a
result worth reporting honestly, not a problem to work around.

---

## 7. Get the results back to Windows

From Ubuntu:

```bash
cd ~/HATI_V2.0
python harvest_results.py --out /mnt/c/Users/$USER/Desktop
```

That writes `HATI_results_<host>_<date>.zip` straight to your Windows Desktop, roughly 50
to 150 MB, holding every figure, CSV, report, compiled PDF, the co-registration budget,
the sweep manifest, and a manifest listing exactly what came along. Add `--include-cubes`
if you also want the projected cubes, which is gigabytes.

You can also browse the Linux filesystem from Windows Explorer at `\\wsl$\Ubuntu\home\`.

To push code changes back, set authorship through the environment, never through
`git config`:

```bash
GIT_AUTHOR_NAME="Cyrex567" GIT_AUTHOR_EMAIL="gergomorvai044@gmail.com" \
GIT_COMMITTER_NAME="Cyrex567" GIT_COMMITTER_EMAIL="gergomorvai044@gmail.com" \
git commit -am "real-data sweep ingest: co-registration budget and manifest"
git push origin feat/v2.0-heatmap
```

---

## 8. When things go wrong

**`Activate.ps1 cannot be loaded because running scripts is disabled`**
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

**`python` not recognised**
Python was installed without the PATH option. Reinstall and tick "Add python.exe to
PATH", or use the `py` launcher instead.

**`rasterio` will not install on Windows**
`python -m pip install --upgrade pip wheel` then retry. Modern rasterio ships Windows
wheels, so this is rare.

**Port 8737 already in use**
`python dashboard\hati_dashboard.py --port 8800`

**`wsl --install` fails or Ubuntu will not start**
Virtualisation is probably off in BIOS. Enable Intel VT-x or AMD-V. Confirm with Task
Manager, Performance, CPU: "Virtualization: Enabled".

**WSL is eating the C: drive**
The virtual disk grows and does not shrink on its own. Move ISISDATA to `/mnt/d` as in
section 4, and check with `du -sh ~/isisdata`.

**"pins are involved in the conflict" during the ISIS install**
You pinned a Python version. Do not. `conda env remove -y -n isis`, then create the
environment and install ISIS in one command as shown in section 4.

**"Solving environment" appears frozen**
Normal for ISIS. Ten to twenty minutes with no output is expected with `conda`. Use
`mamba` instead, which is far faster.

**`spiceinit` fails with a network error**
It calls a USGS web service. Retry. The ingest caches each completed frame, so rerunning
resumes rather than starting over.

**ISIS commands not found even though the install worked**
Wrong environment. `conda activate isis`, then check `echo $ISISROOT`.

**Everything ran but the numbers differ from the papers**
Run `python tests/test_channels.py` first. If that passes, the core is fine and the
difference is in the data or configuration. `Config.hash()` in `src/hati_core` prints a
digest of every setting, which is the fastest way to find what moved.
