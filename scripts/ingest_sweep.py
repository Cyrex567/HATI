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
import re
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


def load_csv(min_margin_m: float = 600.0) -> list[dict]:
    """Read the sweep CSV, keeping only frames that really image the touchdown.

    ODE's spatial query filters on the footprint BOUNDING BOX, which for a long
    diagonal polar strip is far larger than the strip. The first real ingest
    downloaded four frames on that basis and three of them imaged nothing at the
    touchdown.

    The obvious fix -- test the footprint polygon instead of its box -- is only
    right if the test runs in a projected frame. Done in raw degrees it passed
    every one of the five frames of the second ingest, all of which then came back
    0% covered; measured properly they miss the site by 0.4 to 3.0 km.
    solar_sweep_query.py now writes a signed margin_m column, the distance from
    the site to the nearest strip edge. Require real clearance, not containment:
    co-registration reads a window 400 m across, so a frame that merely clips the
    site is still useless.
    """
    import csv as _csv
    csvs = sorted(OUT.glob("solar_sweep_*.csv"))
    if not csvs:
        sys.exit("no solar_sweep CSV in output/athena -- run solar_sweep_query.py first")
    rows, missed, clipped, legacy = [], 0, 0, False
    with open(csvs[-1], encoding="utf-8", errors="replace") as fh:
        for d in _csv.DictReader(fh):
            raw = (d.get("margin_m") or "").strip()
            if raw == "":
                # a CSV written before margins existed; fall back to the boolean
                legacy = True
                if (d.get("covers_site") or "yes").strip() != "yes":
                    missed += 1
                    continue
                margin = float("nan")
            else:
                try:
                    margin = float(raw)
                except ValueError:
                    continue
                if margin < 0:
                    missed += 1
                    continue
                if margin < min_margin_m:
                    clipped += 1
                    continue
            try:
                rows.append({"pid": d["product"], "utc": d["utc"],
                             "elev": float(d["sun_elev_deg"]), "az": float(d["sun_az_deg"]),
                             "url": d["download_url"], "margin": margin})
            except (KeyError, ValueError):
                continue
    print(f"sweep CSV: {csvs[-1].name}  ({len(rows)} usable frames; "
          f"{missed} miss the site, {clipped} clip it by under {min_margin_m:.0f} m)")
    if legacy:
        print("  NOTE: this CSV has no margin_m column, so frames were filtered by the old\n"
              "        boolean containment test, which is known to be over-permissive near\n"
              "        the pole. Re-run solar_sweep_query.py to regenerate it.")
    if not rows:
        sys.exit("no frames cover the site with enough clearance; re-run "
                 "solar_sweep_query.py, or lower --min-margin-m")
    return rows


def select_frames(rows: list[dict], n: int, emin: float, emax: float, target: float,
                  force: list[str] | None = None) -> list[dict]:
    stage("SELECT", "run")
    if force:
        want = {f.strip().lower().lstrip("nac.") for f in force if f.strip()}
        picked = [r for r in rows
                  if r["pid"].lower().split(".")[-1] in want or r["pid"].lower() in want]
        found = {r["pid"].lower().split(".")[-1] for r in picked}
        for w in sorted(want - found):
            print(f"  requested frame not in the CSV: {w}")
        print(f"forced frame list: {len(picked)} of {len(want)} requested frames found")
        for r in picked:
            print(f"{r['pid']:<22}{r['az']:>10.1f}{r['elev']:>7.2f}"
                  f"{r['margin']:>9.0f} m   {r['url'][-48:]}")
        stage("SELECT", "ok" if picked else "fail",
              f"{len(picked)} frames, forced by --frames (gates bypassed)")
        if not picked:
            sys.exit("none of the requested frames are in the sweep CSV")
        return picked

    lit = [r for r in rows if emin <= r["elev"] <= emax and r["url"].lower().endswith(".img")]
    picked = []
    for b in range(n):
        lo, hi = b * 360.0 / n, (b + 1) * 360.0 / n
        cand = [r for r in lit if lo <= (r["az"] % 360) < hi]
        if cand:
            # Within a bin, elevation near the target is what we want, but a frame
            # that barely clips the site is worth nothing however good its sun angle
            # is. Break ties toward clearance: prefer anything comfortably inside.
            picked.append(min(cand, key=lambda r: (abs(r["elev"] - target)
                                                   - 0.5 * min(r["margin"], 2000.0) / 1000.0)))
    print(f"{'pid':<22}{'az(proxy)':>10}{'elev':>7}{'margin':>11}   url")
    for r in picked:
        print(f"{r['pid']:<22}{r['az']:>10.1f}{r['elev']:>7.2f}"
              f"{r['margin']:>9.0f} m   {r['url'][-48:]}")
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


def campt_covers(cub: Path, lat: float, lon: float, base: str) -> bool | None:
    """Ask the camera model whether the site falls on this frame's detector.

    The ODE footprint is an index product: a coarse polygon, good enough to
    reject a strip that misses by kilometres, but it is not the camera. After
    spiceinit the real geometry is available, so ask it directly -- campt in
    ground mode reports the sample and line a lat/lon lands on, and fails when
    the point is off the image. That check costs seconds and sits before
    lronaccal and cam2map, which cost minutes and gigabytes each.

    Returns True/False, or None when campt is unavailable and the caller should
    fall back to the footprint decision rather than reject a good frame.
    """
    if not shutil.which("campt"):
        return None
    out = cub.with_suffix(".campt.txt")
    ok, tail = isis(["campt", f"from={cub}", "type=ground",
                     f"latitude={lat}", f"longitude={lon}",
                     f"to={out}", "format=pvl", "append=false"])
    txt = ""
    if out.exists():
        txt = out.read_text(errors="replace")
        out.unlink(missing_ok=True)
    if not ok:
        low = (tail or "").lower()
        if "outside" in low or "not visible" in low or "off the image" in low \
                or "does not intersect" in low or "no intersection" in low:
            stage("CAMPT", "fail", f"{base}: the site is not on this frame's detector")
            return False
        stage("CAMPT", "skip", f"{base}: campt errored ({tail[:120]}); "
                               f"falling back to the footprint decision")
        return None
    m = re.search(r"^\s*Sample\s*=\s*([-\d.]+)", txt, re.M)
    n = re.search(r"^\s*Line\s*=\s*([-\d.]+)", txt, re.M)
    if not (m and n):
        stage("CAMPT", "skip", f"{base}: campt gave no Sample/Line; using the footprint")
        return None
    s, l = float(m.group(1)), float(n.group(1))
    inside = s > 0 and l > 0
    stage("CAMPT", "ok" if inside else "fail",
          f"{base}: site at sample {s:.0f}, line {l:.0f}"
          + ("" if inside else "  -- off the detector"))
    return inside


def process_frame(fr: dict, workdir: Path, mapfile: Path,
                  skip_campt: bool = False) -> bool:
    base = fr["img"].stem
    cub, cal, prj = (workdir / f"{base}.cub", workdir / f"{base}.cal.cub",
                     workdir / f"{base}.lev2.cub")
    fr["lev2"] = prj
    if prj.exists():
        for s in ("LRONAC2ISIS", "SPICEINIT", "LRONACCAL", "CAM2MAP"):
            stage(s, "ok", f"{base} cached")
        return True
    early = [
        ("LRONAC2ISIS", ["lronac2isis", f"from={fr['img']}", f"to={cub}"]),
        ("SPICEINIT",   ["spiceinit", f"from={cub}", "web=yes"]),
    ]
    late = [
        ("LRONACCAL",   ["lronaccal", f"from={cub}", f"to={cal}"]),
        ("CAM2MAP",     ["cam2map", f"from={cal}", f"map={mapfile}", f"to={prj}",
                         "pixres=map", "defaultrange=map"]),
    ]
    for name, cmd in early:
        stage(name, "run", base)
        ok, tail = isis(cmd)
        stage(name, "ok" if ok else "fail", base if ok else f"{base}: {tail}")
        if not ok:
            return False

    if not skip_campt:
        sys.path.insert(0, str(ROOT / "scripts"))
        import athena_counterfactual as ac
        if campt_covers(cub, ac.TD_LAT, ac.TD_LON, base) is False:
            cub.unlink(missing_ok=True)
            return False

    for name, cmd in late:
        stage(name, "run", base)
        ok, tail = isis(cmd)
        stage(name, "ok" if ok else "fail", base if ok else f"{base}: {tail}")
        if not ok:
            return False
    cub.unlink(missing_ok=True); cal.unlink(missing_ok=True)   # keep only lev2
    return True


def where_is_the_data(fr: dict, np, rasterio, ac) -> str:
    """When a window comes back empty, say WHY in one line.

    An empty window has three quite different causes and they need opposite
    fixes, so guessing between them wastes a whole run. Read the cube decimated,
    find every valid pixel, and report how far the nearest one is from the
    touchdown:

      cube entirely empty      -> cam2map produced nothing; a projection problem
      data present, far away   -> the frame really does miss the site; selection
      data present, close by   -> the window is being placed wrong; a CRS problem

    Decimated to about 2000 pixels on the long axis, so this costs a second even
    on a full NAC strip.
    """
    try:
        from rasterio.warp import transform as warp_transform
        with rasterio.open(fr["lev2"]) as src:
            step = max(1, max(src.width, src.height) // 2000)
            a = src.read(1, out_shape=(1, max(1, src.height // step),
                                       max(1, src.width // step)))[0].astype("float64")
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
            a[a <= ac.NODATA_BELOW] = np.nan
            good = np.isfinite(a)
            fill = 100.0 * good.mean()
            if not good.any():
                return (f"the whole cube is empty ({fill:.1f}% valid) -- cam2map wrote no "
                        f"image data, so this is a projection/extent problem, not selection")
            xs, ys = warp_transform("+proj=longlat +R=1737400 +no_defs", src.crs,
                                    [ac.TD_LON], [ac.TD_LAT])
            gr, gc = np.nonzero(good)
            # decimated pixel centres back to map coordinates
            px, py = rasterio.transform.xy(src.transform, gr * step, gc * step)
            d = np.hypot(np.asarray(px) - xs[0], np.asarray(py) - ys[0])
            near = float(d.min())
            cx, cy = float(np.mean(px)), float(np.mean(py))
            verdict = ("the window is mislocated -- data sits essentially at the touchdown"
                       if near < 500 else
                       "the frame genuinely misses the site" if near > 2000 else
                       "the site is on the strip edge")
            return (f"cube {fill:.1f}% valid; nearest real pixel {near/1000:.2f} km from the "
                    f"touchdown; valid centroid {(cx-xs[0])/1000:+.1f},{(cy-ys[0])/1000:+.1f} km "
                    f"away -> {verdict}")
    except Exception as e:  # noqa: BLE001
        return f"(could not diagnose: {e})"


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
                diag = where_is_the_data(fr, np, rasterio, ac)
                rows.append(f"{fr['pid']},,,,{e}{'; ' + diag if diag else ''}")
                stage("COREGISTER", "fail", f"{fr['pid']}: {e}")
                if diag:
                    print(f"   {diag}", flush=True)
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
    ap.add_argument("--min-margin-m", type=float, default=600.0,
                    help="how far inside the strip the site must sit, in metres. "
                         "The co-registration window is 400 m across, so a frame that "
                         "only clips the site cannot be correlated. Default 600.")
    ap.add_argument("--frames", default="",
                    help="comma-separated product ids to ingest instead of selecting "
                         "by azimuth, e.g. M1101075756LE. Bypasses the elevation and "
                         "spread gates; for diagnosing the pipeline on known-good frames.")
    ap.add_argument("--no-campt", action="store_true",
                    help="skip the campt ground-point check after spiceinit")
    ap.add_argument("--execute", action="store_true",
                    help="actually download + run ISIS (default: dry-run plan only)")
    args = ap.parse_args()

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    forced = [s for s in args.frames.split(",") if s.strip()]
    frames = select_frames(load_csv(args.min_margin_m), args.n, args.min_elev,
                           args.max_elev, args.target_elev, force=forced)

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
        if download(fr, SWEEP_DIR) and process_frame(fr, SWEEP_DIR, mapfile,
                                                     skip_campt=args.no_campt):
            done.append(fr)
    if not done:
        sys.exit("no frame survived the ISIS chain; nothing to co-register")
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
