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
ALL_BY_PID: dict[str, dict] = {}      # every frame in the sweep CSV, filtered or not

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
    ALL_BY_PID.clear()
    with open(csvs[-1], encoding="utf-8", errors="replace") as fh:
        for d in _csv.DictReader(fh):
            # Keep an unfiltered index too. A frame rejected on its own footprint
            # can still be the sibling channel that rescues its partner, and the
            # footprint is per-product while the two optics share a swath.
            try:
                ALL_BY_PID[d["product"].split(".")[-1].upper()] = {
                    "pid": d["product"], "utc": d["utc"],
                    "elev": float(d["sun_elev_deg"]), "az": float(d["sun_az_deg"]),
                    "url": d["download_url"],
                    "margin": float(d.get("margin_m") or "nan")}
            except (KeyError, ValueError):
                pass
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
                  force: list[str] | None = None, alternates: int = 3) -> list[dict]:
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
        if not cand:
            continue
        # Within a bin, elevation near the target is what we want, but a frame that
        # barely clips the site is worth nothing however good its sun angle is.
        # Break ties toward clearance: prefer anything comfortably inside.
        cand.sort(key=lambda r: (abs(r["elev"] - target)
                                 - 0.5 * min(r["margin"], 2000.0) / 1000.0))
        # Carry alternates. The ODE footprint is a coarse index polygon describing
        # the observation, not the single optic we download, so it cannot tell
        # whether the site lands on this channel's 5064-sample detector -- only
        # campt can, and only after spiceinit. Three frames of the previous run put
        # the site 5, 989 and 1256 samples past the edge on footprints that read as
        # 1.3 to 2.3 km inside. Give each bin a queue and let campt pick the winner
        # rather than losing the bin to a frame the footprint mis-sold.
        head = dict(cand[0])
        head["alternates"] = [dict(c) for c in cand[1:1 + max(0, alternates)]]
        picked.append(head)
    print(f"{'pid':<22}{'az(proxy)':>10}{'elev':>7}{'margin':>11}{'alts':>6}   url")
    for r in picked:
        print(f"{r['pid']:<22}{r['az']:>10.1f}{r['elev']:>7.2f}"
              f"{r['margin']:>9.0f} m{len(r.get('alternates', [])):>6}   {r['url'][-48:]}")
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


def lev2_has_site(prj: Path, half: int = 400) -> bool | None:
    """Does this projected cube actually hold pixels at the touchdown?

    Cheap enough to run on every cached product. Returns None, meaning "cannot
    tell, assume it is fine", when the python imaging stack is unavailable, so a
    missing dependency never silently discards good work.
    """
    try:
        import numpy as np
        import rasterio
        from rasterio.warp import transform as warp_transform
        sys.path.insert(0, str(ROOT / "scripts"))
        import athena_counterfactual as ac
        with rasterio.open(prj) as src:
            xs, ys = warp_transform("+proj=longlat +R=1737400 +no_defs", src.crs,
                                    [ac.TD_LON], [ac.TD_LAT])
            r, c = src.index(xs[0], ys[0])
            w = rasterio.windows.Window(c - half, r - half, 2 * half, 2 * half)
            a = src.read(1, window=w, boundless=True,
                         fill_value=float("nan")).astype("float64")
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
            a[a <= ac.NODATA_BELOW] = np.nan
            return bool(np.isfinite(a).mean() >= 0.5)
    except Exception:  # noqa: BLE001
        return None


def cube_dims(cub: Path) -> tuple[int, int]:
    """Samples and lines of a cube, from the ISIS label. Falls back to NAC's 5064."""
    dims = []
    for key in ("Samples", "Lines"):
        ok, out = isis(["getkey", f"from={cub}", "grpname=Dimensions", f"keyword={key}"])
        m = re.search(r"(\d+)", out or "")
        dims.append(int(m.group(1)) if (ok and m) else 0)
    ns, nl = dims
    return (ns or 5064), (nl or 0)


def campt_covers(cub: Path, lat: float, lon: float, base: str,
                 edge_px: float = 400.0) -> str:
    """Ask the camera model where the site falls on this frame's detector.

    The ODE footprint is an index product: a coarse polygon, good enough to
    reject a strip that misses by kilometres, but it is not the camera, and it
    describes the observation rather than the single LE or RE channel we
    actually download. After spiceinit the real geometry is available, so ask it
    directly. That check costs seconds and sits before lronaccal and cam2map,
    which cost minutes and gigabytes each.

    The bound matters as much as the question. campt in ground mode happily
    extrapolates past the focal plane and reports a sample number off the end of
    the detector without erroring, so a sample has to be tested against the
    cube's real width, not merely against zero. Skipping that is what let three
    frames through with the site 5, 989 and 1256 samples beyond a 5064-sample
    NAC channel -- reported as covered, projected as empty.

    Returns one of:
      "inside"      the site is on the detector with room for the window
      "off_sample"  past the left or right edge; the sibling channel may have it
      "off_line"    before the start or after the end of the readout
      "unknown"     campt unavailable or unparseable; defer to the footprint
    """
    if not shutil.which("campt"):
        return "unknown"
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
        if any(k in low for k in ("outside", "not visible", "off the image",
                                  "does not intersect", "no intersection")):
            stage("CAMPT", "fail", f"{base}: the site is not on this frame's detector")
            return "off_line"
        stage("CAMPT", "skip", f"{base}: campt errored ({tail[:120]}); "
                               f"falling back to the footprint decision")
        return "unknown"
    m = re.search(r"^\s*Sample\s*=\s*([-\d.]+)", txt, re.M)
    n = re.search(r"^\s*Line\s*=\s*([-\d.]+)", txt, re.M)
    if not (m and n):
        stage("CAMPT", "skip", f"{base}: campt gave no Sample/Line; using the footprint")
        return "unknown"
    s, l = float(m.group(1)), float(n.group(1))
    ns, nl = cube_dims(cub)
    # the correlation window is 400 map pixels wide, so the site needs that much
    # detector either side of it, not merely a sample number inside the array
    if not (edge_px < s < ns - edge_px):
        stage("CAMPT", "fail",
              f"{base}: site at sample {s:.0f} of {ns} -- "
              + (f"{s - ns:.0f} past the edge of this channel"
                 if s > ns else f"{-s:.0f} before it" if s < 0
                 else f"only {min(s, ns - s):.0f} px from the edge, "
                      f"too close for a {edge_px:.0f} px window"))
        return "off_sample"
    if nl and not (0 < l < nl):
        stage("CAMPT", "fail",
              f"{base}: site at line {l:.0f} of {nl} -- outside the readout")
        return "off_line"
    stage("CAMPT", "ok",
          f"{base}: site at sample {s:.0f} of {ns}, line {l:.0f}"
          + (f" of {nl}" if nl else ""))
    return "inside"


def sibling_channel(pid: str) -> str | None:
    """The other optic of the same NAC observation: LE <-> RE.

    The two channels are separate products covering adjacent ground swaths. When
    the site lands just past the edge of one, it is usually well inside the
    other, so a frame rejected for being off the sample edge is worth one retry
    rather than a discard.
    """
    p = pid.strip().upper()
    if p.endswith("LE"):
        return p[:-2] + "RE"
    if p.endswith("RE"):
        return p[:-2] + "LE"
    return None


def process_frame(fr: dict, workdir: Path, mapfile: Path,
                  skip_campt: bool = False) -> bool:
    base = fr["img"].stem
    cub, cal, prj = (workdir / f"{base}.cub", workdir / f"{base}.cal.cub",
                     workdir / f"{base}.lev2.cub")
    fr["lev2"] = prj
    if prj.exists():
        # A cached cube skips campt, because the level-1 cube it needs is deleted
        # once the projection succeeds. That is how three known-empty products
        # survived a re-run untouched. Check the projection itself instead: if it
        # holds no data at the site, throw it away and rebuild from the EDR, which
        # is still on disk, so this costs ISIS time and no download.
        if lev2_has_site(prj) is False:
            print(f"   cached {prj.name} has no data at the touchdown; rebuilding it",
                  flush=True)
            prj.unlink(missing_ok=True)
        else:
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
        verdict = campt_covers(cub, ac.TD_LAT, ac.TD_LON, base)
        fr["campt"] = verdict
        if verdict in ("off_sample", "off_line"):
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
            # read(1, ...) with a scalar band index returns a 2-D array already;
            # indexing [0] off it silently reduced this to a single row, which is
            # why the diagnostic died unpacking a 1-D nonzero()
            a = src.read(1, out_shape=(max(1, src.height // step),
                                       max(1, src.width // step))).astype("float64")
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
        rows = ["pid,shift_row_px,shift_col_px,residual_px,ncc,skimage_error,note"]
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
                # normalization=None, not skimage's default 'phase'.
                #
                # Phase normalisation whitens the spectrum, which amplifies the
                # high-frequency bins. On imagery that has been resampled -- and
                # cam2map interpolates, which low-passes -- those bins hold numerical
                # noise rather than signal, and the correlation peak disappears: a
                # planted 3 px shift on a smoothed field comes back as exactly
                # (0.00, 0.00). It also fails outright when the two frames carry
                # different shadows, which for a solar sweep is the normal case, not
                # the exception. Plain cross-correlation recovers the planted shift to
                # 0.1 px in every regime tested: smooth, noisy, sharply textured, and
                # contrast-flipped illumination.
                #
                # It also restores a usable error. Under phase normalisation skimage's
                # error is ~1.000 whether the answer is right or hopeless, so it cannot
                # be read at all; unnormalised it runs about 0.5 when aligned and rises
                # toward 1.0 as the illumination diverges.
                #
                # Mean-subtract first: unnormalised correlation is otherwise dominated
                # by the DC term, and NAC DN values sit well above zero.
                r0f = refc.astype("float64")
                r0f = r0f - np.nanmean(r0f)
                filled = filled - np.nanmean(filled)
                sh, err, _ = phase_cross_correlation(r0f, filled, upsample_factor=10,
                                                     normalization=None)
                if not np.isfinite(err):
                    raise ValueError("correlation degenerate (nan error), not a measurement")

                # residual: apply the shift and correlate again. This is the quantity
                # the gate is actually about. The raw shift is the SPICE pointing
                # error, which is tens of metres and is meant to be corrected, not
                # gated on; what has to be sub-pixel is what is left after correcting.
                #
                # ncc: Pearson correlation of the aligned pair over real pixels.
                # Interpretable on its own, and the honest caveat is that frames at
                # different sun azimuth carry different shadows, so a modest ncc at a
                # small residual means illumination difference, not misalignment.
                from scipy.ndimage import shift as nd_shift
                aligned = nd_shift(filled, sh, order=1, mode="nearest")
                sh2, _, _ = phase_cross_correlation(r0f, aligned, upsample_factor=20,
                                                    normalization=None)
                resid = float(np.hypot(sh2[0], sh2[1]))
                m = np.isfinite(r0f) & np.isfinite(aligned)
                if m.sum() > 100:
                    a, b = r0f[m], aligned[m]
                    sa, sb = a.std(), b.std()
                    ncc = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)) \
                        if sa > 1e-9 and sb > 1e-9 else float("nan")
                else:
                    ncc = float("nan")

                fr["shift"] = [float(sh[0]), float(sh[1])]
                fr["residual_px"] = resid
                fr["ncc"] = ncc
                rows.append(f"{fr['pid']},{sh[0]:.2f},{sh[1]:.2f},{resid:.3f},"
                            f"{ncc:.3f},{err:.3f},ok ({how})")
                stage("COREGISTER", "ok",
                      f"{fr['pid']} shift=({sh[0]:+.2f},{sh[1]:+.2f}) px "
                      f"residual={resid:.2f} px ncc={ncc:+.3f} [{how}]")
            except Exception as e:  # noqa: BLE001
                fr["shift"] = None
                diag = where_is_the_data(fr, np, rasterio, ac)
                rows.append(f"{fr['pid']},,,,,,{e}{'; ' + diag if diag else ''}")
                stage("COREGISTER", "fail", f"{fr['pid']}: {e}")
                if diag:
                    print(f"   {diag}", flush=True)
        (SWEEP_DIR / "coreg_report.csv").write_text("\n".join(rows), encoding="utf-8")
        print(f"co-registration budget -> {SWEEP_DIR/'coreg_report.csv'}")

        ok_fr = [f for f in frames if f.get("residual_px") is not None]
        if not ok_fr:
            stage("GATE", "fail", "no frame produced a measurement")
            return
        res = sorted(f["residual_px"] for f in ok_fr)
        med = res[len(res) // 2] if len(res) % 2 else 0.5 * (res[len(res)//2 - 1]
                                                             + res[len(res)//2])
        raw = sorted(math.hypot(*f["shift"]) for f in ok_fr)
        rmed = raw[len(raw) // 2] if len(raw) % 2 else 0.5 * (raw[len(raw)//2 - 1]
                                                              + raw[len(raw)//2])
        nccs = [f["ncc"] for f in ok_fr if f.get("ncc") == f.get("ncc")]
        print()
        print(f"  frames measured        : {len(ok_fr)} of {len(frames)}")
        print(f"  median raw shift       : {rmed:.2f} px   "
              f"({rmed * 0.9:.0f} m at the 0.9 m/px map scale)")
        print(f"     -- this is the SPICE pointing error, which co-registration exists")
        print(f"        to remove. It is not the gate and is expected to be large.")
        print(f"  median residual        : {med:.2f} px   <-- THE GATE, passes at <= 1.00")
        if nccs:
            print(f"  median aligned ncc     : {sorted(nccs)[len(nccs)//2]:+.3f}   "
                  f"(low ncc at a low residual means the sun moved, not the frame)")
        gate = med <= 1.0
        stage("GATE", "ok" if gate else "fail",
              f"median residual {med:.2f} px over {len(ok_fr)} frames")
        if not gate:
            print("\n  Do not run kinematics on this. A residual above a pixel means the\n"
                  "  shadow motion we would measure is contaminated by frame motion.")
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
    ap.add_argument("--alternates", type=int, default=3,
                    help="fallback frames to keep per azimuth bin. The footprint "
                         "cannot tell whether the site lands on the downloaded "
                         "channel's detector; campt can, so each bin gets a queue "
                         "and campt picks the winner. 0 restores one-shot bins.")
    ap.add_argument("--no-sibling", action="store_true",
                    help="do not retry the other NAC channel when the site lands "
                         "just past the sample edge of the one selected")
    ap.add_argument("--execute", action="store_true",
                    help="actually download + run ISIS (default: dry-run plan only)")
    args = ap.parse_args()

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    forced = [s for s in args.frames.split(",") if s.strip()]
    frames = select_frames(load_csv(args.min_margin_m), args.n, args.min_elev,
                           args.max_elev, args.target_elev, force=forced,
                           alternates=args.alternates)

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

    done, tried = [], set()
    for fr in frames:
        queue = [fr] + list(fr.get("alternates") or [])
        # last resort: the other optic of the same observation, which images the
        # adjacent swath. ODE lists only one channel per observation here, but the
        # archive path differs by two characters, so it costs nothing to try.
        if not args.no_sibling:
            sib = sibling_channel(fr["pid"].split(".")[-1])
            if sib and sib not in ALL_BY_PID:
                alt = dict(fr)
                alt["pid"] = "nac." + sib.lower()
                alt["url"] = re.sub(r"(M\d+)(LE|RE)\.IMG$", rf"\g<1>{sib[-2:]}.IMG",
                                    fr["url"], flags=re.I)
                alt.pop("alternates", None)
                if alt["url"] != fr["url"]:
                    queue.append(alt)
        for i, f in enumerate(queue):
            key = f["pid"].split(".")[-1].upper()
            if key in tried:
                continue
            tried.add(key)
            if i:
                print(f"   -> {queue[i-1]['pid'].split('.')[-1].upper()} does not put "
                      f"the site on its detector; trying {key}", flush=True)
            if not download(f, SWEEP_DIR):
                continue
            if process_frame(f, SWEEP_DIR, mapfile, skip_campt=args.no_campt):
                done.append(f)
                break
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
