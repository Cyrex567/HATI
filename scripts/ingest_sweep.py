"""HATI v2.5 -- solar-sweep ingestion pipeline (EDR -> co-registered stack).

Turns an azimuth-spread subset of the archived LROC NAC frames (from the ODE
sweep CSV) into calibrated, map-projected, sub-pixel co-registered orthos on the
NOBILE03 polar-stereographic grid -- the input the shadow-kinematics detector
needs for its first real-data run.

Stages (each prints '##STAGE <NAME> <run|ok|fail|skip> ...' for the dashboard):
  SELECT       azimuth-stratified pick of N lit frames (elev in band, near 5 deg)
  DOWNLOAD     fetch NAC EDR .IMG from the LROC PDS node  (~250-450 MB each!)
  LRONAC2ISIS  EDR -> ISIS cube
  SPICEINIT    attach geometry (web=yes -> USGS kernel service)
  LRONACCAL    radiometric calibration
  CAM2MAP      project to south-polar stereographic, 0.9 m/px (sweep_polar.map)
  COREGISTER   phase-correlation shift vs the NOBILE03 reference ortho
               -> coreg_report.csv  (THE error budget for the kinematics claim)
  MANIFEST     data/sweep/manifest.json for the real-data kinematics run

Safety: DRY-RUN by default -- prints the full plan, checks ISIS, downloads
nothing. Pass --execute to actually run (GPU box). ISIS3 must be on PATH
(conda: `conda create -n isis -c usgs-astrogeology isis` + ISISDATA setup).
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "athena"
SWEEP_DIR = ROOT / "data" / "sweep"
REF_ORTHO = ROOT / "data" / "athena" / "NAC_DTM_NOBILE03_M1101075756_90CM.IMG"
ISIS_BIN = ["lronac2isis", "spiceinit", "lronaccal", "cam2map"]

MAP_PVL = """Group = Mapping
  ProjectionName     = PolarStereographic
  CenterLongitude    = 0.0
  CenterLatitude     = -90.0
  TargetName         = Moon
  EquatorialRadius   = 1737400.0 <meters>
  PolarRadius        = 1737400.0 <meters>
  LatitudeType       = Planetocentric
  LongitudeDirection = PositiveEast
  LongitudeDomain    = 360
  PixelResolution    = 0.9 <meters/pixel>
  MinimumLatitude    = -85.10
  MaximumLatitude    = -84.50
  MinimumLongitude   = 27.5
  MaximumLongitude   = 31.5
End_Group
End
"""


def stage(name: str, state: str, detail: str = "") -> None:
    print(f"##STAGE {name} {state} {detail}".rstrip(), flush=True)


def load_csv() -> list[dict]:
    """Read the sweep CSV, keeping only frames whose footprint contains the site.

    ODE's spatial query filters on the footprint BOUNDING BOX, which for a long
    diagonal polar strip is far larger than the strip. The first real ingest
    downloaded four frames on that basis and three of them imaged nothing at the
    touchdown. solar_sweep_query.py now writes a covers_site column; honour it.
    """
    import csv as _csv
    csvs = sorted(OUT.glob("solar_sweep_*.csv"))
    if not csvs:
        sys.exit("no solar_sweep CSV in output/athena -- run solar_sweep_query.py first")
    rows, skipped = [], 0
    with open(csvs[-1], encoding="utf-8", errors="replace") as fh:
        for d in _csv.DictReader(fh):
            if (d.get("covers_site") or "yes").strip() != "yes":
                skipped += 1
                continue
            try:
                rows.append({"pid": d["product"], "utc": d["utc"],
                             "elev": float(d["sun_elev_deg"]), "az": float(d["sun_az_deg"]),
                             "url": d["download_url"]})
            except (KeyError, ValueError):
                continue
    note = f", {skipped} skipped (footprint does not cover the site)" if skipped else ""
    print(f"sweep CSV: {csvs[-1].name}  ({len(rows)} usable frames{note})")
    if not rows:
        sys.exit("no frames cover the site; re-run solar_sweep_query.py to refresh the CSV")
    return rows


def select_frames(rows: list[dict], n: int, emin: float, emax: float, target: float) -> list[dict]:
    stage("SELECT", "run")
    lit = [r for r in rows if emin <= r["elev"] <= emax and r["url"].lower().endswith(".img")]
    picked = []
    for b in range(n):
        lo, hi = b * 360.0 / n, (b + 1) * 360.0 / n
        cand = [r for r in lit if lo <= (r["az"] % 360) < hi]
        if cand:
            picked.append(min(cand, key=lambda r: abs(r["elev"] - target)))
    print(f"{'pid':<22}{'az(proxy)':>10}{'elev':>7}   url")
    for r in picked:
        print(f"{r['pid']:<22}{r['az']:>10.1f}{r['elev']:>7.2f}   {r['url'][-48:]}")
    # What kinematics needs is azimuth SPREAD, not a full set of bins. Four frames
    # across 150 degrees is workable; ten frames inside 15 degrees is not. Gate on
    # the spread and the count, and say which one failed.
    azs = sorted(r["az"] % 360 for r in picked)
    span = (max(azs) - min(azs)) if len(azs) > 1 else 0.0
    gaps = [(azs[i + 1] - azs[i]) for i in range(len(azs) - 1)] + [360 - (azs[-1] - azs[0])]
    span = 360 - max(gaps)            # spread of the arc the frames actually occupy
    ok = len(picked) >= 4 and span >= 40.0
    stage("SELECT", "ok" if ok else "fail",
          f"{len(picked)}/{n} bins, azimuth spread {span:.0f} deg "
          f"(elev {emin}-{emax} deg, target {target})")
    if not ok:
        if len(picked) < 4:
            sys.exit(f"only {len(picked)} frames selected; need at least 4. "
                     f"Widen --min-elev/--max-elev or lower --n.")
        sys.exit(f"azimuth spread is only {span:.0f} deg; shadows barely move below about "
                 f"40 deg, so the kinematics cannot discriminate. Widen the elevation band.")
    return picked


def download(fr: dict, dest: Path) -> bool:
    f = dest / (fr["pid"].split(".")[-1].upper() + ".IMG")
    fr["img"] = f
    if f.exists() and f.stat().st_size > 10_000_000:
        stage("DOWNLOAD", "ok", f"{f.name} cached ({f.stat().st_size/1e6:.0f} MB)")
        return True
    stage("DOWNLOAD", "run", f.name)
    try:
        req = urllib.request.Request(fr["url"], headers={"User-Agent": "HATI-ingest/1.0"})
        with urllib.request.urlopen(req, timeout=300) as r, open(f, "wb") as fh:
            got = 0
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                fh.write(b); got += len(b)
                if got % (1 << 26) < (1 << 22):
                    print(f"   ... {got/1e6:.0f} MB", flush=True)
        stage("DOWNLOAD", "ok", f"{f.name} ({got/1e6:.0f} MB)")
        return True
    except Exception as e:  # noqa: BLE001
        stage("DOWNLOAD", "fail", f"{f.name}: {e}")
        return False


def isis(cmd: list[str]) -> tuple[bool, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        return r.returncode == 0, (r.stderr or r.stdout)[-400:]
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def process_frame(fr: dict, workdir: Path, mapfile: Path) -> bool:
    base = fr["img"].stem
    cub, cal, prj = (workdir / f"{base}.cub", workdir / f"{base}.cal.cub",
                     workdir / f"{base}.lev2.cub")
    fr["lev2"] = prj
    if prj.exists():
        for s in ("LRONAC2ISIS", "SPICEINIT", "LRONACCAL", "CAM2MAP"):
            stage(s, "ok", f"{base} cached")
        return True
    steps = [
        ("LRONAC2ISIS", ["lronac2isis", f"from={fr['img']}", f"to={cub}"]),
        ("SPICEINIT",   ["spiceinit", f"from={cub}", "web=yes"]),
        ("LRONACCAL",   ["lronaccal", f"from={cub}", f"to={cal}"]),
        ("CAM2MAP",     ["cam2map", f"from={cal}", f"map={mapfile}", f"to={prj}",
                         "pixres=map", "defaultrange=map"]),
    ]
    for name, cmd in steps:
        stage(name, "run", base)
        ok, tail = isis(cmd)
        stage(name, "ok" if ok else "fail", base if ok else f"{base}: {tail}")
        if not ok:
            return False
    cub.unlink(missing_ok=True); cal.unlink(missing_ok=True)   # keep only lev2
    return True


def coregister(frames: list[dict]) -> None:
    """Phase-correlation shift of each projected cube vs the reference ortho.
    This CSV is the co-registration error budget the kinematics claim rests on."""
    stage("COREGISTER", "run")
    try:
        import numpy as np
        import rasterio
        from rasterio.warp import transform as warp_transform
        from skimage.registration import phase_cross_correlation
        sys.path.insert(0, str(ROOT / "scripts"))
        import athena_counterfactual as ac

        MOON_GEOG = "+proj=longlat +R=1737400 +no_defs"

        def touchdown_rowcol(src):
            """Locate the touchdown in this raster.

            Prefer the raster's OWN crs and let proj do the transform, instead of
            hand-rolling the polar-stereographic formula and hoping its x sign
            matches what cam2map wrote. The sign convention differs between the
            NOBILE03 PDS4 label and ISIS output, which is what produced empty
            windows on the first real ingest.
            """
            if src.crs:
                try:
                    xs, ys = warp_transform(MOON_GEOG, src.crs, [ac.TD_LON], [ac.TD_LAT])
                    r, c = src.index(xs[0], ys[0])
                    if 0 <= r < src.height and 0 <= c < src.width:
                        return r, c, "cube crs"
                except Exception:  # noqa: BLE001
                    pass
            x, y = ac.touchdown_xy()          # fallback: try both x conventions
            b = src.bounds
            for sx in (1, -1):
                if b.left <= sx * x <= b.right and b.bottom <= y <= b.top:
                    r, c = src.index(sx * x, y)
                    return r, c, f"formula x{sx:+d}"
            raise ValueError("touchdown is outside this cube under either x sign; "
                             "widen the extent in sweep_polar.map and re-run cam2map")

        ref = ac.load_ortho().astype("float32")
        rr, rc_ = ac.ortho_pixel()
        h = 400
        refc = ref[rr - h:rr + h, rc_ - h:rc_ + h]
        rows = ["pid,shift_row_px,shift_col_px,error,note"]
        for fr in frames:
            try:
                with rasterio.open(fr["lev2"]) as src:        # GDAL reads ISIS3 cubes
                    r0, c0, how = touchdown_rowcol(src)
                    win = rasterio.windows.Window(c0 - h, r0 - h, 2 * h, 2 * h)
                    # float64: ISIS marks nodata with -FLT_MAX (-3.4e38), which is
                    # FINITE. Left unmasked it passes an isfinite() check and then
                    # overflows the variance to inf, which phase correlation returns
                    # as an exact (0,0) shift with a nan error, i.e. a fake perfect
                    # alignment. Mask it explicitly, as NODATA_BELOW does elsewhere.
                    mov = src.read(1, window=win, boundless=True,
                                   fill_value=float("nan")).astype("float64")
                    if src.nodata is not None:
                        mov[mov == src.nodata] = np.nan
                    mov[mov <= ac.NODATA_BELOW] = np.nan
                # A window of nodata is finite and uniform, so an .any() check passes
                # it and phase_cross_correlation then returns exactly (0,0) with a nan
                # error. That reads as a perfect alignment and sails through the gate.
                # Refuse anything that cannot carry a real measurement.
                if mov.shape != refc.shape:
                    raise ValueError(f"window shape {mov.shape} != reference {refc.shape}")
                finite = np.isfinite(mov)
                if finite.sum() < 0.5 * mov.size:
                    raise ValueError(f"only {100*finite.mean():.0f}% real data in the "
                                     f"window; this frame's strip may not cover the site")
                sd = float(np.nanstd(mov))
                if not np.isfinite(sd) or sd < 1e-6:
                    raise ValueError(f"window variance is {sd}, not a usable image")
                # fill gaps with the local mean rather than zero, so masked pixels do
                # not create an artificial step that dominates the correlation
                filled = np.where(finite, mov, np.nanmean(mov))
                sh, err, _ = phase_cross_correlation(refc.astype("float64"), filled,
                                                     upsample_factor=10)
                if not np.isfinite(err):
                    raise ValueError("correlation degenerate (nan error), not a measurement")
                fr["shift"] = [float(sh[0]), float(sh[1])]
                rows.append(f"{fr['pid']},{sh[0]:.2f},{sh[1]:.2f},{err:.3f},ok ({how})")
                stage("COREGISTER", "ok",
                      f"{fr['pid']} shift=({sh[0]:+.2f},{sh[1]:+.2f}) px err={err:.3f} [{how}]")
            except Exception as e:  # noqa: BLE001
                fr["shift"] = None
                rows.append(f"{fr['pid']},,,,{e}")
                stage("COREGISTER", "fail", f"{fr['pid']}: {e}")
        (SWEEP_DIR / "coreg_report.csv").write_text("\n".join(rows), encoding="utf-8")
        print(f"co-registration budget -> {SWEEP_DIR/'coreg_report.csv'}")
    except ImportError as e:
        stage("COREGISTER", "fail", f"missing python dep: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="azimuth bins / frames to ingest")
    ap.add_argument("--min-elev", type=float, default=1.5)
    ap.add_argument("--max-elev", type=float, default=7.5)
    ap.add_argument("--target-elev", type=float, default=5.0)
    ap.add_argument("--execute", action="store_true",
                    help="actually download + run ISIS (default: dry-run plan only)")
    args = ap.parse_args()

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    frames = select_frames(load_csv(), args.n, args.min_elev, args.max_elev, args.target_elev)

    have_isis = all(shutil.which(b) for b in ISIS_BIN)
    isis_msg = "YES" if have_isis else \
        "NO  (conda create -n isis -c usgs-astrogeology isis; set ISISROOT/ISISDATA)"
    print(f"ISIS3 on PATH: {isis_msg}")
    est = 0.35 * len(frames)
    print(f"plan: {len(frames)} frames, ~{est:.1f} GB download, ISIS chain, coregistration")

    if not args.execute:
        (SWEEP_DIR / "plan.json").write_text(json.dumps(
            [{k: str(v) if isinstance(v, Path) else v for k, v in f.items()} for f in frames],
            indent=1), encoding="utf-8")
        print(f"DRY RUN complete -> {SWEEP_DIR/'plan.json'}   (re-run with --execute on the GPU box)")
        for s in ("DOWNLOAD", "LRONAC2ISIS", "SPICEINIT", "LRONACCAL", "CAM2MAP",
                  "COREGISTER", "MANIFEST"):
            stage(s, "skip", "dry-run")
        return

    if not have_isis:
        sys.exit("--execute needs ISIS3 on PATH; install it first")
    mapfile = SWEEP_DIR / "sweep_polar.map"
    mapfile.write_text(MAP_PVL, encoding="utf-8")

    done = []
    for fr in frames:
        if download(fr, SWEEP_DIR) and process_frame(fr, SWEEP_DIR, mapfile):
            done.append(fr)
    coregister(done)

    stage("MANIFEST", "run")
    man = [{"pid": f["pid"], "az_proxy": f["az"], "elev": f["elev"],
            "lev2": str(f["lev2"]), "shift_px": f.get("shift")} for f in done]
    (SWEEP_DIR / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    stage("MANIFEST", "ok", f"{len(done)}/{len(frames)} frames -> data/sweep/manifest.json")
    print("\nNEXT: check coreg_report.csv (gate: median |shift| <= 1 px), then run the "
          "real-data kinematics adapter on manifest.json.")


if __name__ == "__main__":
    main()
