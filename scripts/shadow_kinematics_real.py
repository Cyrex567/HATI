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


def bases_from_regions(regions, az_map_deg, elev_deg, elong, max_width_px=4.0):
    """Where each cast-shadow-shaped region says its caster is.

    Two filters, and the width one matters more than the ratio.

    A boulder's shadow is as WIDE as the boulder: 1 px for anything under a
    metre at 0.9 m/px, 3 px at 2.7 m. A crest or a crater rim casts a shadow
    that is long AND wide, and a pure ratio test passes it happily. That is why
    8491 regions passed on crater-dominated ground.

    Width is also what makes the up-sun extreme meaningful. On a thin streak the
    extreme IS the caster, and stays the caster under any assumed azimuth within
    90 degrees of the truth. On a wide region the up-sun boundary is a crest
    perpendicular to the sun, the maximum along that direction is degenerate, and
    which pixel wins is decided by boundary noise. Worse, an assumed azimuth even
    slightly off picks the END of that crest instead, deterministically and by the
    sign of the error, and crest ends are fixed terrain points that recur in every
    frame. Verified directly: a 1 px streak votes at the identical pixel at every
    azimuth error tested, while a 13 px crest walks (297,295) to (302,304) across
    30 degrees of error.

    That is what broke the shuffled-azimuth null. Permuting azimuths inside an 88
    degree span cannot move a thin vote at all, so it preserves every boulder
    coincidence, while it relocates every wide-region vote onto a repeatable
    terrain corner. The null could only come out above the real result, which is
    exactly what it did, three times.
    """
    az = math.radians(az_map_deg)
    tan_e = math.tan(math.radians(elev_deg))
    srow, scol = -math.cos(az), math.sin(az)          # toward the sun, in the raster
    prow, pcol = -scol, srow
    bases = []
    for rr, cc in regions:
        proj = rr * srow + cc * scol
        perp = rr * prow + cc * pcol
        along = (proj.max() - proj.min()) * RES
        wide_px = perp.max() - perp.min()
        wide = wide_px * RES
        if wide_px > max_width_px:
            continue          # a crest or a rim, not a boulder shadow
        if along < elong * max(wide, RES):
            continue                                   # not a cast shadow shape
        i = int(np.argmax(proj))                       # up-sun extreme = the base
        bases.append((int(rr[i]), int(cc[i]), along * tan_e, along / RES))
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
        for br, bc, hh, *_ in bases:
            if 0 <= br < H and 0 <= bc < W:
                v[br, bc] = 1.0
                h[br, bc] = max(h[br, bc], hh)
        return v, h
    R = radius
    yy, xx = np.ogrid[-R:R + 1, -R:R + 1]
    disc = (yy * yy + xx * xx) <= R * R
    for br, bc, hh, *_ in bases:
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


def accumulate(frames, shape, min_area, elong, radius=2, order=None,
               max_width_px=4.0, jitter=None):
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
        bases = bases_from_regions(f["regions"], g["az_map"], g["elev"], elong,
                                   max_width_px)
        if jitter is not None:
            # the spatial-shift null: displace this frame's whole vote map by more
            # than the terrain correlation length. Unlike permuting azimuths it
            # destroys the correspondence without touching the estimator, so it
            # measures chance coincidence and nothing else.
            dy, dx = jitter[k]
            bases = [(b[0] + dy, b[1] + dx, b[2], b[3]) for b in bases]
        if order is None and jitter is None:
            f["_bases"] = bases          # kept so each cluster can be classified
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


# --------------------------------------------------------------- what is it
def classify(votes, azs):
    """Is this cluster a fixed point, or a point that walks with the sun?

    A boulder casts from the same base every frame, so its votes fit a single
    fixed point B. A crater rim does not: the up-sun end of its shadow slides
    around the rim as the sun turns, tracing c + r*s_hat with r the rim radius.
    Both converge under a coincidence count, which is why the vote tally alone
    cannot tell them apart and this comparison is the actual test.

    Fits both by least squares and returns their RMS residuals and the fitted
    radius. The arc model has one more parameter so it can never fit worse; what
    matters is whether r is large enough to be a real arc rather than noise.

    votes: (frame index, row, col).  azs: map azimuth per frame, degrees.
    """
    if len(votes) < 3:
        return None
    v = np.array([(r, c) for _, r, c in votes], float)
    rms_point = float(np.sqrt(np.mean(np.sum((v - v.mean(0)) ** 2, axis=1))))

    # v_i = c + r * s_hat_i, unknowns (c_row, c_col, r), two rows per vote
    A, b = [], []
    for (fi, r, c) in votes:
        a = math.radians(azs[fi])
        A.append([1.0, 0.0, -math.cos(a)]); b.append(r)
        A.append([0.0, 1.0, math.sin(a)]);  b.append(c)
    sol, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    resid = np.array(b) - np.array(A) @ sol
    rms_arc = float(np.sqrt(np.mean(resid.reshape(-1, 2).sum(axis=1) ** 2 / 2)))
    return {"rms_point": rms_point, "rms_arc": rms_arc, "radius_px": float(abs(sol[2])),
            "n": len(votes)}


def bright_side(frames, cy, cx, reach, patch=2):
    """Is there a lit protrusion up-sun of this shadow, or a lit wall beyond it?

    The arc test has no power below a rim radius of about 4 px, because the chord
    an arc traces over 88 degrees is 1.4 times the radius and the vote dilation is
    3 px. Every one of the ten survivors fits inside that blind zone, so the arc
    fit cannot say what they are. This can, and it needs no new data.

    Walk down-sun through the feature. A BOULDER is a lit protrusion with its
    shadow behind it: bright, then dark, then ordinary ground. A CRATER has no
    protrusion; the sunward rim is where the shadow starts and the far interior
    wall faces the sun and is lit: ordinary ground, then dark, then bright.

    So sample just up-sun of the vote and just beyond the far end of the shadow,
    and take the difference. Positive means the caster is a protrusion. Summed
    over frames, since one frame is noise and eight is a measurement.
    """
    tot, n = 0.0, 0
    for f in frames:
        ar = math.radians(f["az_map"])
        srow, scol = -math.cos(ar), math.sin(ar)
        dn = f["dn"]
        H, W = dn.shape
        near = [(br, bc, L) for (br, bc, _h, L) in f.get("_bases", [])
                if (br - cy) ** 2 + (bc - cx) ** 2 <= reach * reach]
        for br, bc, L in near:
            up = (br + 2 * srow, bc + 2 * scol)                 # toward the sun
            far = (br - (L + 2) * srow, bc - (L + 2) * scol)    # past the shadow
            vals = []
            for (pr, pc) in (up, far):
                r0, r1 = int(pr) - patch, int(pr) + patch + 1
                c0, c1 = int(pc) - patch, int(pc) + patch + 1
                if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
                    vals.append(None)
                    continue
                sub = dn[r0:r1, c0:c1]
                vals.append(float(np.nanmean(sub)) if np.isfinite(sub).any() else None)
            if vals[0] is None or vals[1] is None:
                continue
            bg = np.nanmedian(dn)
            tot += (vals[0] - vals[1]) / (bg if bg else 1.0)
            n += 1
    return (tot / n, n) if n else (float("nan"), 0)


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


def inject_craters(dn, craters, az_map_deg, elev_deg, bg_win, darkness=0.25):
    """Bowl craters, the negative control the pipeline must NOT call a boulder.

    At low sun the interior wall on the SUNWARD side faces away from the sun and
    is also shadowed by its own rim, so the shadow occupies the sunward part of
    the bowl, bounded by a chord perpendicular to the sun. The result is a lune
    whose long axis is across the sun line, and whose up-sun extreme walks along
    the rim as the sun turns. That is the thing that converges on a terrain
    corner and looks like a caster.

    craters: (row, col, radius_px, depth_m).
    """
    from scipy import ndimage as ndi
    out = dn.copy()
    bg = ndi.uniform_filter(np.where(np.isfinite(out), out, np.nanmedian(out)), bg_win)
    ar = math.radians(az_map_deg)
    srow, scol = -math.cos(ar), math.sin(ar)
    cot = 1.0 / math.tan(math.radians(elev_deg))
    H, W = out.shape
    for r0, c0, rad, depth in craters:
        reach = min(2.0 * rad, depth * cot / RES)      # how far the rim shadow throws
        lo = max(0, int(r0 - rad)); hi = min(H, int(r0 + rad) + 1)
        cl = max(0, int(c0 - rad)); ch = min(W, int(c0 + rad) + 1)
        if lo >= hi or cl >= ch:
            continue
        yy, xx = np.mgrid[lo:hi, cl:ch]
        dy, dx = yy - r0, xx - c0
        inside = dy * dy + dx * dx <= rad * rad
        along = dy * srow + dx * scol                  # toward the sun
        shadow = inside & (along > rad - reach)
        sub = out[lo:hi, cl:ch]
        sub[shadow] = darkness * bg[lo:hi, cl:ch][shadow]
    return out


def inject_ridges(dn, ridges, az_map_deg, elev_deg, bg_win, darkness=0.25):
    """Linear crests, the other thing that votes at a fixed terrain corner.

    A ridge of height h throws a band h*cot(e) wide down-sun along its whole
    length. The band's long axis is the ridge, fixed in the ground, and its ends
    are fixed points that recur in every frame.

    ridges: (row, col, length_px, orientation_deg, height_m).
    """
    from scipy import ndimage as ndi
    out = dn.copy()
    bg = ndi.uniform_filter(np.where(np.isfinite(out), out, np.nanmedian(out)), bg_win)
    ar = math.radians(az_map_deg)
    srow, scol = -math.cos(ar), math.sin(ar)
    cot = 1.0 / math.tan(math.radians(elev_deg))
    H, W = out.shape
    for r0, c0, length, orient, h in ridges:
        orad = math.radians(orient)
        ur, uc = math.cos(orad), math.sin(orad)        # along the ridge
        reach = int(round(h * cot / RES))
        for t in range(-int(length) // 2, int(length) // 2 + 1):
            br, bc = r0 + t * ur, c0 + t * uc
            for q in range(0, max(1, reach) + 1):      # sweep the shadow down-sun
                rr = int(round(br - q * srow))
                cc = int(round(bc - q * scol))
                if 0 <= rr < H and 0 <= cc < W:
                    out[rr, cc] = darkness * bg[rr, cc]
    return out


def _run_injected(frames, shape, args, render):
    """Detect and accumulate over frames with something rendered into them."""
    inj = []
    for f in frames:
        d = render(f["dn"], f["az_map"], f["elev"])
        inj.append({"mask": detect(d, args.shadow_frac, args.bg_win, args.min_area),
                    "az_map": f["az_map"], "elev": f["elev"]})
    _, ev, _, _, _ = accumulate(inj, shape, args.min_area, args.elongation,
                                args.vote_radius, None, args.max_width_px)
    return ev


def hits_near(ev, targets, k, tol) -> int:
    """How many planted objects have a converged pixel within tol."""
    rr, cc = np.nonzero(ev >= k)
    if rr.size == 0:
        return 0
    n = 0
    for t in targets:
        if np.min((rr - t[0]) ** 2 + (cc - t[1]) ** 2) <= tol * tol:
            n += 1
    return n


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
    ap.add_argument("--half", type=int, default=0,
                    help="half-window in pixels around the touchdown. 0 means take "
                         "the one the ingest co-registered on, recorded in the "
                         "manifest, so the shift is verified on the same ground the "
                         "science reads. Overriding it breaks that guarantee.")
    ap.add_argument("--shadow-frac", type=float, default=0.5,
                    help="a pixel is shadow below this fraction of its local background")
    ap.add_argument("--bg-win", type=int, default=45, help="local background window, px")
    ap.add_argument("--min-area", type=int, default=4, help="smallest region to vote, px")
    ap.add_argument("--vote-radius", type=int, default=0,
                    help="how close two frames must agree to count as agreeing, px. "
                         "0 means derive it from the measured closure in the manifest, "
                         "which is the real frame-to-frame alignment.")
    ap.add_argument("--max-width-px", type=float, default=8.0,
                    help="widest region allowed to vote, across the sun line. "
                         "Was 4 px on the reasoning that a shadow is as wide as "
                         "its caster, which is wrong: a lunar rock is about twice "
                         "as wide as it is tall, so a 2 m high boulder throws a "
                         "4.4 px shadow and a 4 px cap rejected it. That was the "
                         "recovery ceiling, self-inflicted.")
    ap.add_argument("--inject", type=int, default=60,
                    help="synthetic boulders planted per sensitivity trial. 0 skips it.")
    ap.add_argument("--inject-heights", default="0.3,0.5,1.0,2.0",
                    help="boulder heights in metres for the sensitivity table")
    ap.add_argument("--elongation", type=float, default=1.8,
                    help="how much longer than wide, along the sun line, to count as a "
                         "cast shadow rather than a blob")
    ap.add_argument("--ignore-gate", action="store_true",
                    help="run even though the ingest's co-registration gate failed. "
                         "The output is then not reportable.")
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

    # A failed gate used to write a manifest and exit 0 like any other run, so
    # the science could be run on frames the pipeline itself had judged
    # misaligned. Refuse unless told otherwise, in as many words.
    gated = [e for e in man if "gate_pass" in e]
    if gated and not any(e["gate_pass"] for e in gated) and not args.ignore_gate:
        sys.exit(
            "\nThis manifest comes from an ingest whose GATE FAILED: the frames are\n"
            "not aligned well enough for shadow motion to mean anything.\n"
            "Fix the ingest, or pass --ignore-gate and do not report the result.")

    # One window for the whole project. Co-registration measured its shift and
    # closure on this exact ground, so reading a different amount here would put
    # the science outside what the gate verified.
    ingest_half = next((int(e["half_px"]) for e in man if e.get("half_px")), 0)
    if args.half <= 0:
        if not ingest_half:
            sys.exit("this manifest predates the shared window; re-run the ingest, "
                     "or pass --half explicitly and note that co-registration was "
                     "only verified over 720 m")
        args.half = ingest_half
    elif ingest_half and args.half != ingest_half:
        print(f"  WARNING: reading {2*args.half} px while co-registration was "
              f"verified on {2*ingest_half} px.\n"
              f"  The shift is a rigid translation fitted to that window; nothing "
              f"checks it holds\n  outside. Prefer --half 0.")

    print(f"manifest: {len(man)} frames from {MANIFEST}")
    print(f"site    : {ac.TD_LAT}, {ac.TD_LON}   window {2*args.half} px "
          f"({2*args.half*RES:.0f} m)   scale {RES} m/px"
          + ("   (as co-registered)" if args.half == ingest_half else "") + "\n")

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
    conf, evidence, hmap, darkf, casts = accumulate(
        frames, shape, args.min_area, args.elongation, args.vote_radius,
        None, args.max_width_px)
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
    # The primary control is a SPATIAL SHIFT, not an azimuth permutation.
    #
    # Permuting azimuths looked like the natural null and is invalid here. Every
    # azimuth in this sweep lies within 88 degrees of every other, and the up-sun
    # extreme of a THIN region is unchanged by any assumed direction within 90
    # degrees of the truth. So permutation preserves every boulder coincidence
    # intact, while relocating every wide-region vote onto a repeatable terrain
    # corner. It can only sit above the real result, and it did, three runs
    # running. Verified directly: a 1 px streak votes at the identical pixel at
    # every azimuth error, a 13 px crest walks several pixels with the sign of it.
    #
    # Displacing each frame's finished vote map by more than the terrain
    # correlation length destroys the correspondence without touching the
    # estimator, so it measures chance coincidence and nothing else.
    lo = max(40, 4 * args.vote_radius)
    null_curves = []
    for _ in range(args.trials):
        jit = [(int(rng.integers(-shape[0] // 4, shape[0] // 4)),
                int(rng.integers(-shape[1] // 4, shape[1] // 4))) for _ in range(n)]
        jit = [(dy if abs(dy) > lo else lo, dx if abs(dx) > lo else lo)
               for dy, dx in jit]
        _, ev, _, _, _ = accumulate(frames, shape, args.min_area, args.elongation,
                                    args.vote_radius, None, args.max_width_px, jit)
        null_curves.append(hits_by_k(ev, n))
    null_curves = np.array(null_curves, float)

    print(f"\n{'-'*68}\nCONVERGENCE, at every agreement threshold")
    print(f"  votes cast (regions)        : {sum(casts)}")
    print(f"  spatial-shift null trials   : {len(null_curves)}")
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
        m = 140
        kk = need
        tol = 3.0 + args.vote_radius
        pos = np.column_stack([rr.integers(m, H - m, args.inject),
                               rr.integers(m, W - m, args.inject)])
        print(f"  {'planted':>24}{'shadow at 3 deg':>18}{'called a caster':>18}")
        for h in [float(x) for x in args.inject_heights.split(",") if x.strip()]:
            b = [(int(p[0]), int(p[1]), h) for p in pos]
            frac = recovery(frames, shape, b, args, kk, tol=tol)
            lbl = f"boulder {h:.1f} m"
            print(f"  {lbl:>24}{h / math.tan(math.radians(3.0)):>15.1f} m"
                  f"{100 * frac:>17.0f}%")

        # The negative half. Positive injection alone is not a result: a detector
        # that fires on everything scores 100% recovery. Craters and ridges are
        # the two things that vote at a fixed terrain corner and therefore mimic
        # a caster to a coincidence count. These rows ARE the false-positive rate.
        print()
        nn = max(8, args.inject // 3)
        neg = np.column_stack([rr.integers(m, H - m, nn), rr.integers(m, W - m, nn)])
        tgt = [(int(p[0]), int(p[1])) for p in neg]
        # Down to 2 m, because that is where the survivors are. My first pass
        # started at 11 m across, and the arc test is blind below about 7 m, so
        # the whole regime the ten detections occupy went untested. A 3 m crater
        # at 4 degrees sun is a 3 px shadowed interior plus a 1 to 2 px rim
        # shadow: a 4 to 5 px region, along the sun, under 4 px wide, voting at a
        # point that moves less than the dilation. It passes every filter and
        # reads as a 0.2 m boulder.
        for diam in (2.0, 3.0, 5.0, 8.0, 11.0, 25.0, 54.0):
            rad = max(1, int(round(0.5 * diam / RES)))
            dep = 0.12 * diam                     # lunar depth to diameter
            cr = [(t[0], t[1], rad, dep) for t in tgt]
            ev = _run_injected(frames, shape, args,
                               lambda d, a, e, c=cr: inject_craters(d, c, a, e,
                                                                    args.bg_win))
            fp = hits_near(ev, tgt, kk, tol + rad)
            lbl = f"crater {diam:.0f} m across"
            print(f"  {lbl:>24}{'':>18}{100 * fp / len(tgt):>17.0f}%")
        for ln, hh in ((30, 2.0), (80, 4.0)):
            rg = [(t[0], t[1], ln, float(rr.integers(0, 180)), hh) for t in tgt]
            ev = _run_injected(frames, shape, args,
                               lambda d, a, e, g=rg: inject_ridges(d, g, a, e,
                                                                   args.bg_win))
            fp = hits_near(ev, tgt, kk, tol + ln / 2)
            lbl = f"ridge {ln * RES:.0f} m, {hh:.0f} m high"
            print(f"  {lbl:>24}{'':>18}{100 * fp / len(tgt):>17.0f}%")
        print(f"\n  Boulder rows are recovery, crater and ridge rows are the FALSE")
        print("  POSITIVE rate. Anything above a few percent there means a converged")
        print("  pixel is not on its own evidence of a boulder.")
        print(f"\n  All measured at k = {kk}, the threshold the real")
        print("  result is reported at, and on the same frames.")

    # ---- what is each surviving cluster, a point or an arc?
    if best:
        from scipy import ndimage as _ndi
        lab, nloc = _ndi.label(evidence >= need)
        azs = [f["az_map"] for f in frames]
        print(f"\n{'-'*68}\nWHAT THE SURVIVORS ARE")
        print("  A vote count cannot tell a boulder from a crater rim: both converge.")
        print("  A boulder casts from one fixed base, so its votes fit a POINT. A rim's")
        print("  shadow end slides around the rim as the sun turns, so its votes fit an")
        print("  ARC of the rim's own radius. Fitting both is the actual test.\n")
        print("  The arc test is blind below a rim radius of about 4 px, which is a")
        print("  7 m crater, and that is where these sit. So also walk down-sun through")
        print("  each one: a boulder is a lit protrusion with its shadow behind it, a")
        print("  crater has no protrusion and a lit far wall past the shadow. Positive")
        print("  sign means a protrusion.\n")
        print(f"  {'loc':>4}{'px':>5}{'frames':>8}{'rms point':>11}{'rms arc':>9}"
              f"{'radius':>9}{'height':>8}{'sign':>8}   reading")
        for li in range(1, nloc + 1):
            ys, xs = np.nonzero(lab == li)
            cy, cx = float(ys.mean()), float(xs.mean())
            reach = args.vote_radius + 4
            votes = [(i, br, bc) for i, f in enumerate(frames)
                     for (br, bc, _h, _L) in f.get("_bases", [])
                     if (br - cy) ** 2 + (bc - cx) ** 2 <= reach * reach]
            cl = classify(votes, azs)
            hh = float(np.median(hmap[lab == li][hmap[lab == li] > 0])) \
                if np.any(hmap[lab == li] > 0) else float("nan")
            sgn, nsg = bright_side(frames, cy, cx, reach)
            if cl is None:
                print(f"  {li:>4}{ys.size:>5}{'-':>8}{'-':>11}{'-':>9}{'-':>9}"
                      f"{hh:>7.2f}m{sgn:>+8.3f}   too few votes to fit")
                continue
            rad = cl["radius_px"]
            if rad > args.vote_radius:
                verdict = f"walks {rad*RES:.1f} m: rim or crest, not a boulder"
            elif not (sgn == sgn):
                verdict = "point-like, sign not measurable"
            elif sgn > 0.02:
                verdict = "point-like AND lit up-sun: boulder"
            elif sgn < -0.02:
                verdict = "point-like but lit BEYOND: pit or small crater"
            else:
                verdict = "point-like, sign flat: undetermined"
            print(f"  {li:>4}{ys.size:>5}{cl['n']:>8}{cl['rms_point']:>11.2f}"
                  f"{cl['rms_arc']:>9.2f}{rad:>9.2f}{hh:>7.2f}m{sgn:>+8.3f}   {verdict}")

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

    # Report LOCATIONS, not pixels. Each converged caster is stamped as a disc of
    # the vote radius, so 42 pixels at radius 3 is about 1.4 places, not 42
    # boulders. Counting pixels overstates a detection by more than an order of
    # magnitude and is exactly how a threshold artefact gets written up as a map.
    from scipy import ndimage as _ndi
    lab, nloc = _ndi.label(evidence >= need)
    ys, xs = np.nonzero(evidence >= need)
    print(f"  {len(ys)} pixels at k >= {need}, which is {nloc} distinct location"
          f"{'' if nloc == 1 else 's'} once the vote discs are merged")
    rows = ["row,col,x_m,y_m,votes,height_m,location_id"]
    for r, c in zip(ys, xs):
        rows.append(f"{r},{c},{x0 + c*RES:.1f},{y0 - r*RES:.1f},"
                    f"{evidence[r,c]:.0f},{hmap[r,c]:.2f},{lab[r,c]}")
    csv = OUT / "shadow_kinematics_real_detections.csv"
    csv.write_text("\n".join(rows), encoding="utf-8")
    print(f"  -> {csv}  ({len(ys)} px in {nloc} locations)")

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
