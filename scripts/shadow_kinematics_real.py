"""Shadow kinematics on the real ingested solar sweep.

The detector is the one proven on synthetic ground truth in shadow_kinematics.py:
accumulate, over many illuminations, one vote at the up-sun base of every
elongated dark region. A real obstacle casts from the same base every time, so
its votes converge. A low-albedo patch is painted on the ground, so as the sun
swings its assumed caster position swings with it and the votes smear.

This runs that detector on data/sweep/manifest.json, the co-registered stack the
ingest produces. Three things had to be got right that the synthetic version
never had to face.

SUN GEOMETRY IS MEASURED, NOT ASSUMED. The sweep catalogue's azimuth is a
synodic-time proxy, honest enough for asking how much azimuth diversity an
archive holds and useless for pointing a vote. Real azimuth and elevation come
from campt's sub-solar point, via spherical trigonometry we control rather than
an ISIS azimuth convention we would have to guess at. Without them the script
refuses to run.

THE MAP FRAME IS ROTATED. Votes are cast in the projected cube, and in south
polar stereographic the direction of north turns with longitude. At site
longitude lam, north in map coordinates is (sin lam, cos lam) and east is
(cos lam, -sin lam), so a sun azimuth A from north points along
(sin(lam+A), cos(lam+A)). The voter therefore gets A + lam, a 29.2 degree
correction at this site.

THERE IS NO GROUND TRUTH. Athena has no boulder catalogue, so no AUC can be
quoted and none is. What can be tested is whether convergence is driven by the
sun at all: re-run the accumulation with the azimuths shuffled between frames,
which keeps every image and every shadow and destroys only the correspondence
between them. If the real run does not beat that null, the sweep is not adding
anything.

    python scripts/shadow_kinematics_real.py
    python scripts/shadow_kinematics_real.py --half 500 --shadow-frac 0.55
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SWEEP_DIR = ROOT / "data" / "sweep"
OUT = ROOT / "output" / "athena"
MANIFEST = SWEEP_DIR / "manifest.json"
RES = 0.9                       # m/px, the map scale cam2map wrote
MOON_R = 1737400.0
MAP_CRS = ("+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +k=1 "
           "+x_0=0 +y_0=0 +R=1737400 +units=m +no_defs")


# --------------------------------------------------------------- sun geometry
def _pvl(txt: str, key: str) -> float | None:
    m = re.search(rf"^\s*{key}\s*=\s*\(?\s*([-\d.]+)", txt, re.M)
    return float(m.group(1)) if m else None


def sun_azimuth(site_lat: float, site_lon: float,
                sub_lat: float, sub_lon: float) -> float:
    """Ground azimuth of the sun from the site, degrees clockwise from north.

    Standard bearing on a sphere, computed here rather than read from ISIS so the
    convention is ours and unambiguous: ISIS reports image-plane azimuths, which
    is not what a vote cast in a map projection needs.
    """
    phi, phis = math.radians(site_lat), math.radians(sub_lat)
    dlon = math.radians(sub_lon - site_lon)
    y = math.sin(dlon) * math.cos(phis)
    x = math.cos(phi) * math.sin(phis) - math.sin(phi) * math.cos(phis) * math.cos(dlon)
    return math.degrees(math.atan2(y, x)) % 360.0


def geometry_for(pid: str, lat: float, lon: float, rebuild: bool) -> dict | None:
    """Real sun azimuth and elevation at the site for one frame.

    Cached in a sidecar next to the cube, because the level-1 cube campt needs is
    deleted once the projection succeeds, and regenerating it costs a couple of
    minutes. Frames ingested before sidecars existed are rebuilt from the EDR,
    which is still on disk, so this costs ISIS time and no download.
    """
    base = pid.split(".")[-1].upper()
    side = SWEEP_DIR / f"{base}.geom.json"
    if side.exists():
        try:
            g = json.loads(side.read_text())
            g["source"] = "sidecar"
            return g
        except Exception:  # noqa: BLE001
            pass

    cub = SWEEP_DIR / f"{base}.cub"
    made_here = False
    if not cub.exists():
        if not rebuild:
            return None
        edr = SWEEP_DIR / f"{base}.IMG"
        if not edr.exists() or not shutil.which("spiceinit"):
            return None
        print(f"   {base}: rebuilding a level-1 cube to measure the sun geometry",
              flush=True)
        for cmd in (["lronac2isis", f"from={edr}", f"to={cub}"],
                    ["spiceinit", f"from={cub}", "web=yes"]):
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if r.returncode != 0:
                cub.unlink(missing_ok=True)
                print(f"   {base}: {cmd[0]} failed: {(r.stderr or r.stdout)[-200:]}")
                return None
        made_here = True

    if not shutil.which("campt"):
        return None
    tmp = SWEEP_DIR / f"{base}.geom.pvl"
    r = subprocess.run(["campt", f"from={cub}", "type=ground",
                        f"latitude={lat}", f"longitude={lon}",
                        f"to={tmp}", "format=pvl", "append=false"],
                       capture_output=True, text=True, timeout=1800)
    txt = tmp.read_text(errors="replace") if tmp.exists() else ""
    tmp.unlink(missing_ok=True)
    if made_here:
        cub.unlink(missing_ok=True)
    if r.returncode != 0 or not txt:
        return None

    sub_lat = _pvl(txt, "SubSolarLatitude")
    sub_lon = _pvl(txt, "SubSolarLongitude")
    inc = _pvl(txt, "Incidence")
    if sub_lat is None or sub_lon is None or inc is None:
        return None
    g = {"az": sun_azimuth(lat, lon, sub_lat, sub_lon),
         "elev": 90.0 - inc,
         "incidence": inc, "sub_lat": sub_lat, "sub_lon": sub_lon}
    side.write_text(json.dumps(g, indent=1), encoding="utf-8")
    g["source"] = "campt"
    return g


# --------------------------------------------------------------- frame windows
def frame_window(lev2: Path, shift, half: int, ac) -> np.ndarray | None:
    """The touchdown window from one projected cube, put on the reference grid.

    The cubes already share a grid, since cam2map wrote them all against the same
    map file. The measured co-registration shift removes what is left, which is
    the spacecraft pointing error, tens of metres of it.
    """
    import rasterio
    from rasterio.warp import transform as warp_transform
    from scipy.ndimage import shift as nd_shift
    with rasterio.open(lev2) as src:
        if not src.crs:
            return None
        xs, ys = warp_transform("+proj=longlat +R=1737400 +no_defs", src.crs,
                                [ac.TD_LON], [ac.TD_LAT])
        r0, c0 = src.index(xs[0], ys[0])
        win = rasterio.windows.Window(c0 - half, r0 - half, 2 * half, 2 * half)
        a = src.read(1, window=win, boundless=True,
                     fill_value=float("nan")).astype("float64")
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
    a[a <= ac.NODATA_BELOW] = np.nan
    if shift:
        good = np.isfinite(a)
        filled = np.where(good, a, np.nanmedian(a))
        a = nd_shift(filled, shift, order=1, mode="nearest")
        keep = nd_shift(good.astype(float), shift, order=0, mode="constant", cval=0)
        a[keep < 0.5] = np.nan
    return a


# --------------------------------------------------------------- detect + vote
def detect(dn: np.ndarray, frac: float, bg_win: int, min_area: int = 4) -> np.ndarray:
    """Shadow where a pixel is much darker than its own neighbourhood.

    A local background, not a global threshold: at five degrees of sun elevation
    the scene brightness varies enormously across a few hundred metres, and a
    global cut would call one end of the window shadow and the other end ground.
    Nodata is filled before filtering, since one NaN poisons the whole kernel,
    and masked out again afterwards.

    Cleaned by AREA, not by morphological opening. Opening with a 3x3 element
    erases any feature narrower than three pixels, and at 0.9 m per pixel the
    shadow of a boulder is about as wide as the boulder: 1 px for anything under
    a metre, 2 px up to about 1.8 m. So the opening inherited from the synthetic
    version was deleting the shadow of every obstacle below roughly 2.7 m, which
    is the whole sub-resolution population this project exists to find. It never
    showed up on the synthetic benchmark because those boulders were rendered
    with 1 to 2 pixel footprints, giving 3 to 5 pixel shadows that survived.

    Removing small connected components does the job the opening was there for,
    killing isolated noise, while leaving a one-pixel-wide line intact.
    """
    from scipy import ndimage as ndi
    valid = np.isfinite(dn)
    if valid.sum() < 100:
        return np.zeros_like(dn, bool)
    filled = np.where(valid, dn, np.nanmedian(dn))
    bg = ndi.uniform_filter(filled, bg_win)
    mask = valid & (filled < frac * bg)
    # Done here rather than with skimage's remove_small_objects, whose parameter
    # and its meaning changed in 0.26. The rule has to match the voter's exactly,
    # so it is spelled out: drop components smaller than min_area, keep the rest
    # however thin they are.
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    area = np.bincount(lab.ravel())
    area[0] = 0
    return area[lab] >= max(2, min_area)


def regions_of(mask, min_area):
    """Coordinates of every region large enough to consider.

    Split out from voting because it is the expensive half and it does not
    depend on the sun. The shuffled-azimuth null re-votes the same images a
    couple of hundred times, and relabelling nine million pixels on every trial
    costs about half an hour on a 2.7 km window for no new information.
    """
    from skimage.measure import label, regionprops
    out = []
    for p in regionprops(label(mask)):
        if p.area < min_area:
            continue
        out.append((p.coords[:, 0].astype(np.float64),
                    p.coords[:, 1].astype(np.float64)))
    return out


def bases_from_regions(regions, az_map_deg, elev_deg, elong):
    """Where each cast-shadow-shaped region says its caster is."""
    az = math.radians(az_map_deg)
    tan_e = math.tan(math.radians(elev_deg))
    srow, scol = -math.cos(az), math.sin(az)          # toward the sun, in the raster
    prow, pcol = -scol, srow
    bases = []
    for rr, cc in regions:
        proj = rr * srow + cc * scol
        perp = rr * prow + cc * pcol
        along = (proj.max() - proj.min()) * RES
        wide = (perp.max() - perp.min()) * RES
        if along < elong * max(wide, RES):
            continue                                   # not a cast shadow shape
        i = int(np.argmax(proj))                       # up-sun extreme = the base
        bases.append((int(rr[i]), int(cc[i]), along * tan_e))
    return bases


def stamp(shape, bases, radius):
    """Binary vote map and height map for one frame, each vote spread over a disc.

    Writes only the discs, rather than filtering the whole array: a few hundred
    votes times a thirteen-pixel disc instead of nine million pixels.
    """
    v = np.zeros(shape, np.float32)
    h = np.zeros(shape, np.float32)
    H, W = shape
    if radius <= 0:
        for br, bc, hh in bases:
            if 0 <= br < H and 0 <= bc < W:
                v[br, bc] = 1.0
                h[br, bc] = max(h[br, bc], hh)
        return v, h
    R = radius
    yy, xx = np.ogrid[-R:R + 1, -R:R + 1]
    disc = (yy * yy + xx * xx) <= R * R
    for br, bc, hh in bases:
        r0, r1 = max(0, br - R), min(H, br + R + 1)
        c0, c1 = max(0, bc - R), min(W, bc + R + 1)
        if r0 >= r1 or c0 >= c1:
            continue
        d = disc[r0 - (br - R):r1 - (br - R), c0 - (bc - R):c1 - (bc - R)]
        v[r0:r1, c0:c1][d] = 1.0
        sub = h[r0:r1, c0:c1]
        np.maximum(sub, np.where(d, hh, 0.0), out=sub)
    return v, h


def vote(mask, az_map_deg, elev_deg, min_area, elong, radius=2):
    """One frame's votes: where does this illumination say the casters are?

    Returns a BINARY vote map, not a tally. Two reasons, both learned by testing
    the thing on planted shadows of known base.

    Agreement has to mean "within a couple of pixels", not "the same pixel". The
    up-sun extreme of a discretised shadow lands one to three pixels from the true
    base depending on the sun angle, so counting exact pixel coincidences finds
    nothing at all, even for a caster planted at a fixed spot. Each vote is
    therefore dilated over a small disc.

    And one frame must count once. Left as a tally, a frame that happens to break
    a shadow into two regions would agree with itself, which is not evidence of
    anything. Binary per frame, summed across frames, so a pixel holding k means
    k separate illuminations put a caster there.
    """
    regions = regions_of(mask, min_area)
    bases = bases_from_regions(regions, az_map_deg, elev_deg, elong)
    v, h = stamp(mask.shape, bases, radius)
    return v, h, len(bases)


def accumulate(frames, shape, min_area, elong, radius=2, order=None):
    """Vote over every frame. `order` permutes which geometry goes with which image."""
    from scipy import ndimage as ndi
    evidence = np.zeros(shape, np.float32)
    height_acc = np.zeros(shape, np.float32)
    dark = np.zeros(shape, np.float32)
    n = len(frames)
    idx = list(range(n)) if order is None else list(order)
    casts = []
    for k, f in enumerate(frames):
        # cached on the frame: the regions do not depend on the sun, and the null
        # re-votes these same images a couple of hundred times
        if "regions" not in f:
            f["regions"] = regions_of(f["mask"], min_area)
        g = frames[idx[k]]
        bases = bases_from_regions(f["regions"], g["az_map"], g["elev"], elong)
        v, h = stamp(shape, bases, radius)
        evidence += v
        height_acc += h
        dark += f["mask"]
        casts.append(len(bases))
    conf = ndi.gaussian_filter(evidence, 1.2)
    hmap = np.where(evidence > 0, height_acc / np.maximum(evidence, 1e-6), 0.0)
    return conf, evidence, hmap, dark / max(n, 1), casts


def hits_by_k(evidence: np.ndarray, n_frames: int) -> np.ndarray:
    """How many pixels carry votes from at least k frames, for every k.

    The whole curve, not one number at a fixed fraction of the sweep. The
    previous version asked for agreement by 60% of frames, which is a threshold
    that gets HARDER as the sweep grows: three of five, but ten of sixteen.
    That is backwards. Detection of any one caster is probabilistic, so the
    fraction of frames that see it does not rise with n; adding frames should
    buy statistical power, not a stricter test.

    Comparing the whole curve against the same curve computed under shuffled
    azimuths lets the data say which k separates signal from chance, instead of
    a constant chosen in advance.
    """
    return np.array([int((evidence >= k).sum()) for k in range(1, n_frames + 1)])


def concentration(evidence: np.ndarray, n_frames: int) -> tuple[int, float]:
    """Kept for the regression tests: hits and vote share at 60% agreement."""
    need = max(2, int(math.ceil(0.6 * n_frames)))
    hits = int((evidence >= need).sum())
    total = float(evidence.sum())
    mass = float(evidence[evidence >= need].sum()) / total if total > 0 else 0.0
    return hits, mass


# --------------------------------------------------------------- injection
def inject_shadows(dn, boulders, az_map_deg, elev_deg, bg_win, darkness=0.25):
    """Darken the ground where boulders of known height would cast, in one frame.

    Injected into the IMAGE, before detection, so the recovery number measures
    the whole chain: thresholding, region shaping, the elongation filter, the
    vote and the accumulation. Injecting into the mask instead would skip the
    step most likely to be wrong.

    The shadow runs anti-sun from the base for h*cot(e), with a width of about
    the boulder itself, and is set to a fraction of the local background rather
    than to a constant, because a real shadow is dark relative to its
    surroundings and the scene brightness varies across the window.
    """
    from scipy import ndimage as ndi
    out = dn.copy()
    valid = np.isfinite(out)
    bg = ndi.uniform_filter(np.where(valid, out, np.nanmedian(out)), bg_win)
    ar = math.radians(az_map_deg)
    srow, scol = -math.cos(ar), math.sin(ar)
    cot = 1.0 / math.tan(math.radians(elev_deg))
    H, W = out.shape
    for br, bc, h in boulders:
        length_px = h * cot / RES
        half_w = max(0, int(round(0.5 * h / RES)))
        for t in range(int(round(length_px)) + 1):
            r = int(round(br - t * srow))
            c = int(round(bc - t * scol))
            r0, r1 = max(0, r - half_w), min(H, r + half_w + 1)
            c0, c1 = max(0, c - half_w), min(W, c + half_w + 1)
            if r0 < r1 and c0 < c1:
                out[r0:r1, c0:c1] = darkness * bg[r0:r1, c0:c1]
    return out


def recovery(frames, shape, boulders, args, k: int, tol: float) -> float:
    """Fraction of planted boulders the detector finds, on the real imagery."""
    inj = []
    for f in frames:
        d = inject_shadows(f["dn"], boulders, f["az_map"], f["elev"], args.bg_win)
        inj.append({"mask": detect(d, args.shadow_frac, args.bg_win, args.min_area),
                    "az_map": f["az_map"], "elev": f["elev"]})
    _, ev, _, _, _ = accumulate(inj, shape, args.min_area, args.elongation,
                                args.vote_radius)
    rr, cc = np.nonzero(ev >= k)
    if rr.size == 0:
        return 0.0
    found = 0
    for br, bc, _ in boulders:
        if np.min((rr - br) ** 2 + (cc - bc) ** 2) <= tol * tol:
            found += 1
    return found / len(boulders)


# --------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", type=int, default=400,
                    help="half-window in pixels around the touchdown (0.9 m each)")
    ap.add_argument("--shadow-frac", type=float, default=0.5,
                    help="a pixel is shadow below this fraction of its local background")
    ap.add_argument("--bg-win", type=int, default=45, help="local background window, px")
    ap.add_argument("--min-area", type=int, default=4, help="smallest region to vote, px")
    ap.add_argument("--vote-radius", type=int, default=0,
                    help="how close two frames must agree to count as agreeing, px. "
                         "0 means derive it from the measured closure in the manifest, "
                         "which is the real frame-to-frame alignment.")
    ap.add_argument("--inject", type=int, default=60,
                    help="synthetic boulders planted per sensitivity trial. 0 skips it.")
    ap.add_argument("--inject-heights", default="0.3,0.5,1.0,2.0",
                    help="boulder heights in metres for the sensitivity table")
    ap.add_argument("--elongation", type=float, default=1.8,
                    help="how much longer than wide, along the sun line, to count as a "
                         "cast shadow rather than a blob")
    ap.add_argument("--no-rebuild", action="store_true",
                    help="do not regenerate a level-1 cube to measure sun geometry")
    ap.add_argument("--trials", type=int, default=200,
                    help="shuffled-azimuth null trials")
    args = ap.parse_args()

    if not MANIFEST.exists():
        sys.exit(f"no manifest at {MANIFEST}; run scripts/ingest_sweep.py --execute first")
    import athena_counterfactual as ac
    man = json.loads(MANIFEST.read_text())
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"manifest: {len(man)} frames from {MANIFEST}")
    print(f"site    : {ac.TD_LAT}, {ac.TD_LON}   window {2*args.half} px "
          f"({2*args.half*RES:.0f} m)   scale {RES} m/px\n")

    # ---- geometry and windows
    frames, dropped = [], []
    for e in man:
        pid = e["pid"]
        if not e.get("shift_px"):
            dropped.append((pid, "no co-registration measurement"))
            continue
        g = geometry_for(pid, ac.TD_LAT, ac.TD_LON, rebuild=not args.no_rebuild)
        if g is None:
            dropped.append((pid, "sun geometry could not be measured"))
            continue
        lev2 = Path(e["lev2"])
        if not lev2.exists():
            lev2 = SWEEP_DIR / lev2.name
        try:
            w = frame_window(lev2, e["shift_px"], args.half, ac)
        except Exception as exc:  # noqa: BLE001
            dropped.append((pid, f"window unreadable: {exc}"))
            continue
        if w is None or np.isfinite(w).mean() < 0.5:
            dropped.append((pid, "window is mostly empty"))
            continue
        frames.append({"pid": pid, "dn": w, "az": g["az"], "elev": g["elev"],
                       "az_map": (g["az"] + ac.TD_LON) % 360.0,
                       "src": g["source"], "fill": float(np.isfinite(w).mean())})

    for pid, why in dropped:
        print(f"  dropped {pid}: {why}")
    if len(frames) < 3:
        sys.exit(f"\nonly {len(frames)} usable frames; the kinematics needs at least 3. "
                 f"Re-run the ingest.")

    azs = sorted(f["az_map"] for f in frames)
    gaps = [azs[i + 1] - azs[i] for i in range(len(azs) - 1)] + [360 - (azs[-1] - azs[0])]
    spread = 360 - max(gaps)
    print(f"\n{'frame':<22}{'sun az':>8}{'map az':>8}{'elev':>7}{'cot(e)':>8}"
          f"{'fill':>7}  geometry")
    for f in sorted(frames, key=lambda x: x["az_map"]):
        print(f"{f['pid']:<22}{f['az']:>8.1f}{f['az_map']:>8.1f}{f['elev']:>7.2f}"
              f"{1/math.tan(math.radians(f['elev'])):>8.1f}{100*f['fill']:>6.0f}%  {f['src']}")
    print(f"\nazimuth spread in the map frame: {spread:.0f} deg over {len(frames)} frames")

    # The vote radius has to match how well the frames actually agree with each
    # other, which the co-registration closure measures. The residual is 0.1 px
    # against the reference, but frames lit from different directions close with
    # one another at a few pixels, and that is the number a vote has to tolerate.
    # Demanding 2 px agreement when the data only supports 4 finds nothing.
    #
    # This cannot inflate a result: the shuffled-azimuth null is accumulated at
    # the same radius, so a looser radius raises the null in step.
    if args.vote_radius <= 0:
        clo = [e.get("closure_px") for e in man if e.get("closure_px")]
        base = (sorted(clo)[len(clo) // 2] if clo else 2.0)
        args.vote_radius = int(max(2, min(10, round(base + 2))))
        src = (f"from the measured closure of {base:.1f} px" if clo
               else "defaulted; no closure in the manifest")
        print(f"vote radius: {args.vote_radius} px  ({src}, plus 2 px for the "
              f"shadow-base scatter)")
    if spread < 40:
        sys.exit("under 40 degrees of spread; shadows barely move, so the kinematics "
                 "cannot separate a caster from a stain.")

    # ---- detect
    print()
    for f in frames:
        f["mask"] = detect(f["dn"], args.shadow_frac, args.bg_win, args.min_area)
        sf = 100.0 * f["mask"].sum() / max(np.isfinite(f["dn"]).sum(), 1)
        f["shadow_pct"] = sf
        flag = "  <-- suspicious, check --shadow-frac" if (sf < 0.2 or sf > 45) else ""
        print(f"  {f['pid']:<22} shadow pixels {sf:5.1f}%{flag}")

    shape = frames[0]["dn"].shape
    conf, evidence, hmap, darkf, casts = accumulate(frames, shape, args.min_area,
                                                    args.elongation, args.vote_radius)
    hits, mass = concentration(evidence, len(frames))
    need = max(2, int(math.ceil(0.6 * len(frames))))

    silent = sum(1 for c in casts if c == 0)
    print(f"\n{'-'*68}\nCAST-SHADOW REGIONS FOUND")
    for f, c in zip(frames, casts):
        print(f"  {f['pid']:<22}{c:>6}")
    print(f"  {'total':<22}{sum(casts):>6}")
    if silent:
        print(f"\n  {silent} of {len(casts)} frames found no cast-shadow-shaped region at "
              f"all, so\n  agreement by {need} frames is arithmetically impossible. That is "
              f"a shortage of\n  obstacles in this window, not a threshold to tune. Search "
              f"more ground with\n  --half, or point the run at terrain that has boulders "
              f"in it.")
    if sum(casts) == 0:
        print("\n  No votes were cast anywhere. Nothing to test against a null.")
        write_products(conf, evidence, hmap, darkf, frames, ac, args, need)
        return

    # ---- the null: same images, same shadows, azimuths shuffled between them
    n = len(frames)
    rng = np.random.default_rng(11)
    real_curve = hits_by_k(evidence, n)
    null_curves = []
    for _ in range(args.trials):
        order = rng.permutation(n)
        if np.all(order == np.arange(n)):
            continue
        _, ev, _, _, _ = accumulate(frames, shape, args.min_area, args.elongation,
                                    args.vote_radius, order)
        null_curves.append(hits_by_k(ev, n))
    null_curves = np.array(null_curves, float)

    print(f"\n{'-'*68}\nCONVERGENCE, at every agreement threshold")
    print(f"  votes cast (regions)        : {sum(casts)}")
    print(f"  shuffled-azimuth trials     : {len(null_curves)}")
    print()
    print(f"  {'k frames agree':>15}{'real':>9}{'null mean':>11}{'null p99':>10}"
          f"{'p':>8}   ")
    best = None
    for i in range(1, n):                       # k = 2 .. n
        k = i + 1
        r = int(real_curve[i])
        col = null_curves[:, i] if null_curves.size else np.zeros(1)
        p = float((col >= r).mean()) if r > 0 else 1.0
        p99 = float(np.percentile(col, 99))
        flag = ""
        if r > 0 and p < 0.05 and r > col.mean():
            flag = "  <-- separates"
            if best is None or k > best[0]:
                best = (k, r, p)
        print(f"  {k:>15}{r:>9}{col.mean():>11.1f}{p99:>10.0f}{p:>8.3f}{flag}")
        if r == 0 and col.mean() == 0:
            break

    # the reporting threshold is whichever k the data supports, not a constant
    need = best[0] if best else max(2, int(math.ceil(0.6 * n)))
    hits = int(real_curve[need - 1])
    if best:
        print(f"\n  The sun drives convergence at k = {best[0]}: {best[1]} pixels, "
              f"p = {best[2]:.3f}.")
    else:
        print("\n  No agreement threshold separates the real sweep from shuffled "
              "azimuths.")

    # ---- sensitivity: what could this sweep have found if it were there?
    if args.inject:
        print(f"\n{'-'*68}\nSENSITIVITY, by injection into the real frames")
        print("  Boulders of known height are darkened into the imagery before")
        print("  detection, so this measures the whole chain and not just the voter.")
        print("  It turns a null result into a bound: not 'nothing is there' but")
        print("  'nothing this size would have been found'.\n")
        rr = np.random.default_rng(3)
        H, W = shape
        m = 120
        pos = np.column_stack([rr.integers(m, H - m, args.inject),
                               rr.integers(m, W - m, args.inject)])
        kk = need
        print(f"  {'boulder height':>15}{'shadow at 3 deg':>18}{'recovered':>12}")
        for h in [float(x) for x in args.inject_heights.split(",") if x.strip()]:
            b = [(int(p[0]), int(p[1]), h) for p in pos]
            frac = recovery(frames, shape, b, args, kk, tol=3.0 + args.vote_radius)
            print(f"  {h:>13.1f} m{h / math.tan(math.radians(3.0)):>15.1f} m"
                  f"{100 * frac:>11.0f}%")
        print(f"\n  Recovery is measured at k = {kk}, the same threshold the real")
        print("  result is reported at, and on the same frames.")

    # ---- heights at the converged pixels
    hs = hmap[evidence >= need]
    hs = hs[(hs > 0) & (hs < 20)]
    if hs.size:
        print(f"\nHEIGHTS at those pixels (h = L*tan(e))")
        print(f"  n {hs.size}   median {np.median(hs):.2f} m   "
              f"16-84 pct {np.percentile(hs,16):.2f} to {np.percentile(hs,84):.2f} m")

    # ---- write products
    write_products(conf, evidence, hmap, darkf, frames, ac, args, need)

    print(f"\n{'-'*68}")
    if best:
        print(f"Real-data shadow kinematics beat its own null at k = {best[0]}, p = "
              f"{best[2]:.3f}.\nThere is no boulder catalogue at this site, so this is a "
              f"convergence result and\nnot a validated detection rate: the AUC of 0.990 in "
              f"the papers is the synthetic\nbenchmark and stays labelled that way.")
    else:
        print("No detection. Read that against the sensitivity table above: if injected\n"
              "boulders of a given size are recovered and no real ones converge, the\n"
              "statement is that nothing that size is there, which is a result. If the\n"
              "injected ones are not recovered either, the sweep is simply not sensitive\n"
              "enough yet, and the answer is more frames rather than a looser threshold.")


def write_products(conf, evidence, hmap, darkf, frames, ac, args, need) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.transform import from_origin

    half = args.half
    rr, cc = ac.ortho_pixel()
    x0 = ac.ORTHO_ULX + (cc - half) * RES
    y0 = ac.ORTHO_ULY - (rr - half) * RES
    tr = from_origin(x0, y0, RES, RES)
    for name, arr in (("confidence", conf), ("height", hmap)):
        path = OUT / f"shadow_kinematics_real_{name}.tif"
        with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0],
                           width=arr.shape[1], count=1, dtype="float32",
                           crs=MAP_CRS, transform=tr, nodata=0.0) as dst:
            dst.write(arr.astype("float32"), 1)
        print(f"  -> {path}")

    ys, xs = np.nonzero(evidence >= need)
    rows = ["row,col,x_m,y_m,votes,height_m"]
    for r, c in zip(ys, xs):
        rows.append(f"{r},{c},{x0 + c*RES:.1f},{y0 - r*RES:.1f},"
                    f"{evidence[r,c]:.0f},{hmap[r,c]:.2f}")
    csv = OUT / "shadow_kinematics_real_detections.csv"
    csv.write_text("\n".join(rows), encoding="utf-8")
    print(f"  -> {csv}  ({len(ys)} candidates)")

    show = sorted(frames, key=lambda f: f["az_map"])
    pick = [show[0], show[len(show) // 2], show[-1]]
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.26, wspace=0.2)
    for i, f in enumerate(pick):
        ax = fig.add_subplot(gs[0, i])
        d = f["dn"]
        v = d[np.isfinite(d)]
        ax.imshow(d, cmap="gray",
                  vmin=np.percentile(v, 2), vmax=np.percentile(v, 98))
        ov = np.zeros((*d.shape, 4))
        ov[f["mask"]] = (1, 0.2, 0.1, 0.55)
        ax.imshow(ov)
        ax.set_title(f"{f['pid'].split('.')[-1].upper()}\nsun az {f['az']:.0f}deg  "
                     f"elev {f['elev']:.1f}deg  (shadows red)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    axc = fig.add_subplot(gs[1, 0])
    axc.imshow(conf, cmap="inferno")
    axc.set_title(f"accumulated confidence, {len(frames)} real frames\n"
                  f"votes converge on casters, smear on stains", fontsize=9)
    axc.set_xticks([]); axc.set_yticks([])

    axd = fig.add_subplot(gs[1, 1])
    axd.imshow(darkf, cmap="viridis", vmin=0, vmax=1)
    axd.set_title("fraction of frames each pixel was dark in\n"
                  "near 1.0 = permanently dark, not a moving shadow", fontsize=9)
    axd.set_xticks([]); axd.set_yticks([])

    axh = fig.add_subplot(gs[1, 2])
    hs = hmap[evidence >= need]
    hs = hs[(hs > 0) & (hs < 20)]
    if hs.size:
        axh.hist(hs, bins=24, color="#1B7A6E", edgecolor="white")
        axh.axvline(float(np.median(hs)), color="#C1121F", lw=1.5,
                    label=f"median {np.median(hs):.2f} m")
        axh.legend(fontsize=8)
    axh.set_xlabel("recovered height (m)")
    axh.set_ylabel("candidates")
    axh.set_title(f"heights at pixels with >= {need} votes\nfrom L*tan(e), no ground "
                  f"truth to check against", fontsize=9)
    axh.grid(alpha=0.3)

    fig.suptitle("HATI shadow kinematics on the real solar sweep — Athena site, "
                 "co-registered LROC NAC", fontsize=11, fontweight="bold")
    png = OUT / "shadow_kinematics_real.png"
    fig.savefig(png, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {png}")


if __name__ == "__main__":
    main()
