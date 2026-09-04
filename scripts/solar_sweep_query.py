"""Solar-sweep availability query.

Asks NASA PDS Orbital Data Explorer (ODE) for every LROC NAC frame whose
footprint intersects a requested area, and tabulates the illumination geometry
of each -- incidence (=> sun elevation), emission, phase, acquisition time, and
sun azimuth where available. The distribution of sun azimuths over the returned
frames IS the available solar sweep: how many distinct sun directions we can
actually assemble for a given site before paying for co-registration.

Source: ODE REST API, https://oderest.rsl.wustl.edu/  (target=moon, LROC NAC).
Geometry fields are the authoritative per-frame index values. Sun azimuth is
used if ODE returns it; otherwise a synodic-phase proxy is computed from the
acquisition time (monotonic in sub-solar longitude over a lunation) so we can
still measure azimuth DIVERSITY -- flagged clearly as a proxy.

Usage:
  python scripts/solar_sweep_query.py --lat -84.7906 --lon 29.1957 --halfwidth-km 3
  python scripts/solar_sweep_query.py --lat -84.7906 --lon 29.1957 --pt CDRNAC --probe
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import math
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "athena"
ODE = "https://oderest.rsl.wustl.edu/live2/"
R_MOON_KM = 1737.4
SYNODIC = 29.530588853
NEW_MOON = dt.datetime(2000, 1, 6, 18, 14, tzinfo=dt.timezone.utc)  # ref new moon


def synodic_angle(t: dt.datetime) -> float:
    days = (t - NEW_MOON).total_seconds() / 86400.0
    return (days % SYNODIC) / SYNODIC * 360.0


def parse_utc(s: str):
    if not s:
        return None
    s = s.strip().replace("Z", "")[:19]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(s, fmt).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
    return None


def get_any(d: dict, *keys, default=None):
    low = {k.lower(): v for k, v in d.items()}
    for k in keys:
        if k.lower() in low and low[k.lower()] not in ("", "N/A", "UNK", None):
            return low[k.lower()]
    # fuzzy: first key containing the token
    for tok in keys:
        for k, v in low.items():
            if tok.lower() in k and v not in ("", "N/A", "UNK", None):
                return v
    return default


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _polygon(wkt: str | None) -> list[tuple[float, float]] | None:
    """First ring of a WKT POLYGON as (lon, lat) vertices."""
    if not wkt:
        return None
    m = re.search(r"\(\(([^)]*)\)", wkt)
    if not m:
        return None
    pts = []
    for pair in m.group(1).split(","):
        a = pair.split()
        if len(a) >= 2:
            try:
                pts.append((float(a[0]), float(a[1])))
            except ValueError:
                pass
    return pts or None


def _contains(pt: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon."""
    x, y = pt
    inside, j = False, len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _project(lon: float, lat: float, ref_lat: float) -> tuple[float, float]:
    """Longitude/latitude to metres, in a frame where a NAC strip is a strip.

    Above 60 degrees this is polar stereographic about the nearer pole -- the
    same family of projection cam2map writes, so a distance computed here is
    directly comparable to the co-registration window size in metres. Below it,
    a local equirectangular scaling, where the distortion is mild.
    """
    lam, phi = math.radians(lon), math.radians(lat)
    if abs(ref_lat) >= 60.0:
        south = ref_lat < 0
        # rho: distance from the pole on the projection plane
        t = math.tan(math.pi / 4 + (phi if south else -phi) / 2)
        rho = 2 * R_MOON_KM * 1000.0 * t
        return rho * math.sin(lam), rho * math.cos(lam)
    k = R_MOON_KM * 1000.0
    return k * lam * math.cos(math.radians(ref_lat)), k * phi


def _edge_distance(pt: tuple[float, float], poly: list[tuple[float, float]]) -> float:
    """Distance from a point to the nearest polygon edge."""
    px, py = pt
    best = float("inf")
    for i in range(len(poly)):
        ax, ay = poly[i - 1]
        bx, by = poly[i]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        best = min(best, math.hypot(px - (ax + t * dx), py - (ay + t * dy)))
    return best


def site_margin_m(p: dict, lat: float, lon: float) -> float | None:
    """How far inside this frame's footprint the site sits, in metres.

    Positive means inside; negative means the strip misses the site by that far.
    None when the footprint is missing.

    ODE's spatial query filters on the footprint's BOUNDING BOX. A NAC strip is
    long and narrow, so near the pole a diagonal strip has a bounding box vastly
    larger than the strip, and the query returns frames that image nothing near
    the site.

    Testing the polygon itself is necessary but not sufficient: the test has to
    be done in a projected frame. Ray-casting in raw lon/lat degrees treats each
    edge as a straight line in a space that is wildly stretched near the pole, so
    a footprint spanning tens of degrees of longitude balloons into a shape that
    swallows the whole region. That is why the first polygon test passed almost
    every frame, including four that turned out to miss the touchdown by 0.4 to
    3.0 km. Projecting the vertices first separates them cleanly.

    Containment alone is still not enough, which is why this returns a distance
    rather than a boolean: co-registration reads a window hundreds of metres
    wide, so the site has to be that far clear of the strip edge, not merely
    inside it.
    """
    poly = _polygon(get_any(p, "Footprint_geometry", "Footprint_C0_geometry",
                            "Footprint_GL_geometry"))
    if not poly:
        return None
    prj = [_project(a, b, lat) for a, b in poly]
    pt = _project(lon, lat, lat)
    d = _edge_distance(pt, prj)
    return d if _contains(pt, prj) else -d


def best_url(p: dict) -> str:
    pf = (p.get("Product_files") or {}).get("Product_file")
    if isinstance(pf, dict):
        pf = [pf]
    urls = [f.get("URL") or f.get("FileURL") for f in (pf or []) if (f.get("URL") or f.get("FileURL"))]
    for u in urls:
        if u.lower().endswith(".img"):
            return u
    if urls:
        return urls[0]
    return get_any(p, "LabelURL", "ProductURL", "External_url", default="")


def fetch(params: dict) -> dict:
    url = ODE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "HATI-solar-sweep/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8", "replace")), url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=-84.7906)
    ap.add_argument("--lon", type=float, default=29.1957)
    ap.add_argument("--halfwidth-km", type=float, default=3.0)
    ap.add_argument("--pt", default="EDRNAC4", help="ODE product type (EDRNAC4 / CDRNAC4)")
    ap.add_argument("--probe", action="store_true", help="print raw keys of first product and exit")
    ap.add_argument("--max-frames", type=int, default=2000,
                    help="ODE result cap. Most returned frames are bounding-box false "
                         "positives, so the cap has to be well above the number of "
                         "frames actually wanted.")
    ap.add_argument("--all-frames", action="store_true",
                    help="keep frames whose footprint does NOT contain the site "
                         "(the old, over-permissive behaviour)")
    ap.add_argument("--min-margin-m", type=float, default=600.0,
                    help="how far inside the strip the site must sit. The "
                         "co-registration window is 400 m across at NAC scale, so a "
                         "frame that only clips the site is useless. Default 600 m.")
    args = ap.parse_args()

    dlat = args.halfwidth_km / (math.pi / 180 * R_MOON_KM)
    dlon = dlat / max(math.cos(math.radians(args.lat)), 1e-3)
    minlat, maxlat = args.lat - dlat, args.lat + dlat
    wlon, elon = (args.lon - dlon) % 360, (args.lon + dlon) % 360

    params = {
        "query": "product", "results": "fmpc", "output": "JSON",
        "target": "moon", "ihid": "lro", "iid": "lroc", "pt": args.pt,
        "minlat": f"{minlat:.4f}", "maxlat": f"{maxlat:.4f}",
        "westlon": f"{wlon:.4f}", "eastlon": f"{elon:.4f}",
        "limit": str(args.max_frames),
    }
    print(f"area: lat[{minlat:.3f},{maxlat:.3f}] lon[{wlon:.3f},{elon:.3f}]  pt={args.pt}")
    try:
        data, _ = fetch(params)
    except Exception as e:  # noqa
        print(f"NETWORK/QUERY ERROR: {e}")
        sys.exit(2)
    res = data.get("ODEResults", data)
    status = res.get("Status", "?")
    plist = (res.get("Products") or {}).get("Product")
    if not plist:
        print(f"status={status}; no products. keys={list(res.keys())} err={res.get('Error')}")
        sys.exit(1)
    if isinstance(plist, dict):
        plist = [plist]
    capped = " (capped — raise --max-frames)" if len(plist) >= args.max_frames else ""
    print(f"status={status}  frames collected={len(plist)}{capped}")

    if args.probe and plist:
        print("\n--- first product keys ---")
        for k in sorted(plist[0]):
            v = plist[0][k]
            print(f"  {k} = {str(v)[:90]}")
        return

    rows, az_real = [], False
    for p in plist:
        pid = get_any(p, "pdsid", "ProductId", "Product_Id", default="?")
        inc = to_float(get_any(p, "Incidence_angle", "incidence"))
        emi = to_float(get_any(p, "Emission_angle", "emission"))
        pha = to_float(get_any(p, "Phase_angle", "phase"))
        azf = to_float(get_any(p, "subsolar_azimuth", "solar_azimuth", "sun_azimuth", "azimuth"))
        t = parse_utc(get_any(p, "UTC_start_time", "Observation_time", "Start_time", "UTC_start"))
        clat = to_float(get_any(p, "Center_latitude", "Centerlatitude"))
        clon = to_float(get_any(p, "Center_longitude", "Centerlongitude"))
        if azf is not None:
            az_real = True
        az = azf if azf is not None else (synodic_angle(t) if t else None)
        elev = (90.0 - inc) if inc is not None else None
        mar = site_margin_m(p, args.lat, args.lon)
        rows.append(dict(pid=pid, utc=(t.isoformat() if t else ""), inc=inc, elev=elev,
                         emi=emi, pha=pha, az=az, az_src=("ODE" if azf is not None else "proxy"),
                         clat=clat, clon=clon, url=best_url(p),
                         margin=(None if mar is None else round(mar)),
                         cov=("unknown" if mar is None else
                              ("yes" if mar >= args.min_margin_m else "no"))))

    n_all = len(rows)
    n_yes = sum(1 for r in rows if r["cov"] == "yes")
    n_unk = sum(1 for r in rows if r["cov"] == "unknown")
    n_out = sum(1 for r in rows if r["margin"] is not None and r["margin"] < 0)
    print(f"\nfootprint containment: {n_yes}/{n_all} frames put the site at least "
          f"{args.min_margin_m:.0f} m inside the strip ({n_unk} undetermined).")
    print(f"  {n_out} of the {n_all} miss the site entirely, and "
          f"{n_all - n_yes - n_unk - n_out} only clip its edge.")
    print("  ODE filters on the footprint BOUNDING BOX. Near the pole a long diagonal NAC")
    print("  strip has a bounding box far larger than the strip, so the query returns many")
    print("  frames that image nothing at the site. The polygon test, done in a PROJECTED")
    print("  frame rather than in raw degrees, removes them.")
    if not args.all_frames:
        rows = [r for r in rows if r["cov"] == "yes"]
        print(f"keeping the {len(rows)} that contain it (pass --all-frames to keep the rest)")
    rows.sort(key=lambda r: (r["az"] if r["az"] is not None else 999))
    OUT.mkdir(parents=True, exist_ok=True)
    csv = OUT / f"solar_sweep_{args.lat:+.3f}_{args.lon:+.3f}_{args.pt}.csv"
    with open(csv, "w", encoding="utf-8") as f:
        f.write("product,utc,incidence_deg,sun_elev_deg,emission_deg,phase_deg,sun_az_deg,"
                "az_source,center_lat,center_lon,download_url,covers_site,margin_m\n")
        for r in rows:
            f.write(",".join(str(r[k]) if r[k] is not None else ""
                    for k in ("pid", "utc", "inc", "elev", "emi", "pha", "az", "az_src",
                              "clat", "clon", "url", "cov", "margin")) + "\n")

    # coverage summary
    azs = [r["az"] for r in rows if r["az"] is not None]
    bins = [0] * 12
    for a in azs:
        bins[int(a // 30) % 12] += 1
    filled = sum(1 for b in bins if b)
    elevs = [r["elev"] for r in rows if r["elev"] is not None]
    lit = [e for e in elevs if e and e > 0]
    print(f"\nazimuth source: {'ODE (real)' if az_real else 'synodic-time PROXY (no ODE azimuth field)'}")
    print(f"azimuth coverage: {filled}/12 thirty-degree sectors populated")
    print("  sector(deg) :", " ".join(f"{i*30:>3}:{bins[i]}" for i in range(12)))
    if elevs:
        print(f"sun elevation: min {min(elevs):.1f}  max {max(elevs):.1f} deg  | lit (elev>0): {len(lit)}/{len(elevs)}")
    if elevs:
        print(f"incidence: min {90-max(elevs):.1f} max {90-min(elevs):.1f} deg")
    print(f"\nsweep verdict: {'USABLE' if filled >= 4 and len(lit) >= 4 else 'THIN'} "
          f"({len(rows)} frames, {filled}/12 azimuth sectors, {len(lit)} lit)")
    print(f"  -> {csv}")

    # optional polar plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="polar")
        th = [math.radians(r["az"]) for r in rows if r["az"] is not None and r["elev"] is not None]
        rr = [max(r["elev"], 0) for r in rows if r["az"] is not None and r["elev"] is not None]
        yr = [parse_utc(r["utc"]).year if r["utc"] else 0 for r in rows if r["az"] is not None and r["elev"] is not None]
        sc = ax.scatter(th, rr, c=yr, cmap="viridis", s=45, edgecolor="k", linewidth=0.4)
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.set_title(f"Solar-sweep availability @ ({args.lat:.3f},{args.lon:.3f})\n"
                     f"{len(rows)} {args.pt} frames | radius=elevation° | angle=sun azimuth "
                     f"({'ODE' if az_real else 'time-proxy'})", fontsize=10)
        fig.colorbar(sc, ax=ax, label="acquisition year", fraction=0.045, pad=0.1)
        png = OUT / f"solar_sweep_{args.lat:+.3f}_{args.lon:+.3f}_{args.pt}.png"
        fig.savefig(png, dpi=140, bbox_inches="tight", facecolor="white")
        print(f"  -> {png}")
    except Exception as e:  # noqa
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
