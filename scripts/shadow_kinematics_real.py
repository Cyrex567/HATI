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
def detect(dn: np.ndarray, frac: float, bg_win: int) -> np.ndarray:
    """Shadow where a pixel is much darker than its own neighbourhood.

    A local background, not a global threshold: at five degrees of sun elevation
    the scene brightness varies enormously across a few hundred metres, and a
    global cut would call one end of the window shadow and the other end ground.
    Nodata is filled before filtering, since one NaN poisons the whole kernel,
    and masked out again afterwards.
    """
    from scipy import ndimage as ndi
    valid = np.isfinite(dn)
    if valid.sum() < 100:
        return np.zeros_like(dn, bool)
    filled = np.where(valid, dn, np.nanmedian(dn))
    bg = ndi.uniform_filter(filled, bg_win)
    mask = valid & (filled < frac * bg)
    return ndi.binary_opening(mask, iterations=1)


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
    from scipy import ndimage as ndi
    from skimage.measure import label, regionprops
    e = math.radians(elev_deg)
    az = math.radians(az_map_deg)
    srow, scol = -math.cos(az), math.sin(az)          # toward the sun, in the raster
    prow, pcol = -scol, srow
    tan_e = math.tan(e)
    v = np.zeros(mask.shape, np.float32)
    h = np.zeros(mask.shape, np.float32)
    cast = 0
    for p in regionprops(label(mask)):
        if p.area < min_area:
            continue
        rr = p.coords[:, 0].astype(np.float64)
        cc = p.coords[:, 1].astype(np.float64)
        proj = rr * srow + cc * scol
        perp = rr * prow + cc * pcol
        along = (proj.max() - proj.min()) * RES
        wide = (perp.max() - perp.min()) * RES
        if along < elong * max(wide, RES):
            continue                                   # not a cast shadow shape
        i = int(np.argmax(proj))                       # up-sun extreme = the base
        br, bc = int(rr[i]), int(cc[i])
        v[br, bc] = 1.0
        h[br, bc] = max(h[br, bc], along * tan_e)      # h = L*tan(e)
        cast += 1
    if radius > 0 and cast:
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        disc = (y * y + x * x) <= radius * radius
        v = (ndi.maximum_filter(v, footprint=disc) > 0).astype(np.float32)
        h = ndi.maximum_filter(h, footprint=disc)
    return v, h, cast


def accumulate(frames, shape, min_area, elong, radius=2, order=None):
    """Vote over every frame. `order` permutes which geometry goes with which image."""
    from scipy import ndimage as ndi
    evidence = np.zeros(shape, np.float32)
    height_acc = np.zeros(shape, np.float32)
    dark = np.zeros(shape, np.float32)
    n = len(frames)
    idx = list(range(n)) if order is None else list(order)
    for k, f in enumerate(frames):
        g = frames[idx[k]]
        v, h, _ = vote(f["mask"], g["az_map"], g["elev"], min_area, elong, radius)
        evidence += v
        height_acc += h
        dark += f["mask"]
    conf = ndi.gaussian_filter(evidence, 1.2)
    hmap = np.where(evidence > 0, height_acc / np.maximum(evidence, 1e-6), 0.0)
    return conf, evidence, hmap, dark / max(n, 1)


def concentration(evidence: np.ndarray, n_frames: int) -> tuple[int, float]:
    """How much of the vote mass lands on pixels seen by most of the sweep.

    With no ground truth this is the statistic that carries the claim. A pixel
    holding votes from most illuminations is a caster that stayed put. Smeared
    votes from albedo patches land one deep and almost never stack.
    """
    need = max(2, int(math.ceil(0.6 * n_frames)))
    hits = int((evidence >= need).sum())
    total = float(evidence.sum())
    mass = float(evidence[evidence >= need].sum()) / total if total > 0 else 0.0
    return hits, mass


# --------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", type=int, default=400,
                    help="half-window in pixels around the touchdown (0.9 m each)")
    ap.add_argument("--shadow-frac", type=float, default=0.5,
                    help="a pixel is shadow below this fraction of its local background")
    ap.add_argument("--bg-win", type=int, default=45, help="local background window, px")
    ap.add_argument("--min-area", type=int, default=4, help="smallest region to vote, px")
    ap.add_argument("--vote-radius", type=int, default=2,
                    help="how close two frames must agree to count as agreeing, px. "
                         "The up-sun extreme of a discretised shadow lands a pixel or "
                         "three from the true base, so exact coincidence finds nothing.")
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
    if spread < 40:
        sys.exit("under 40 degrees of spread; shadows barely move, so the kinematics "
                 "cannot separate a caster from a stain.")

    # ---- detect
    print()
    for f in frames:
        f["mask"] = detect(f["dn"], args.shadow_frac, args.bg_win)
        sf = 100.0 * f["mask"].sum() / max(np.isfinite(f["dn"]).sum(), 1)
        f["shadow_pct"] = sf
        flag = "  <-- suspicious, check --shadow-frac" if (sf < 0.2 or sf > 45) else ""
        print(f"  {f['pid']:<22} shadow pixels {sf:5.1f}%{flag}")

    shape = frames[0]["dn"].shape
    conf, evidence, hmap, darkf = accumulate(frames, shape, args.min_area,
                                            args.elongation, args.vote_radius)
    hits, mass = concentration(evidence, len(frames))
    need = max(2, int(math.ceil(0.6 * len(frames))))

    # ---- the null: same images, same shadows, azimuths shuffled between them
    rng = np.random.default_rng(11)
    null_hits, null_mass = [], []
    n = len(frames)
    for _ in range(args.trials):
        order = rng.permutation(n)
        if np.all(order == np.arange(n)):
            continue
        _, ev, _, _ = accumulate(frames, shape, args.min_area, args.elongation,
                                 args.vote_radius, order)
        h, m = concentration(ev, n)
        null_hits.append(h)
        null_mass.append(m)
    null_hits = np.array(null_hits, float)
    nh_mean, nh_sd = float(null_hits.mean()), float(null_hits.std())
    z = (hits - nh_mean) / nh_sd if nh_sd > 1e-9 else float("nan")
    p = float((null_hits >= hits).mean())

    print(f"\n{'-'*68}\nCONVERGENCE")
    print(f"  total votes cast            : {int(evidence.sum())}")
    print(f"  pixels with >= {need} votes      : {hits}")
    print(f"  share of votes in them      : {100*mass:.1f}%")
    print(f"  shuffled-azimuth null       : {nh_mean:.1f} +/- {nh_sd:.1f} pixels "
          f"({len(null_hits)} trials)")
    print(f"  z vs null                   : {z:+.1f}      p = {p:.3f}")
    verdict = ("the sun is driving the convergence" if (p < 0.05 and hits > nh_mean)
               else "NOT separable from chance; this sweep does not support a detection")
    print(f"  verdict                     : {verdict}")

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
    if p < 0.05 and hits > nh_mean:
        print("Real-data shadow kinematics ran and beat its own null. There is no boulder\n"
              "catalogue at this site, so this is a convergence result, not a validated\n"
              "detection rate: the AUC of 0.990 in the papers is the synthetic benchmark\n"
              "and stays labelled that way.")
    else:
        print("Do not report this as a detection. Widen the azimuth spread or add frames.")


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
