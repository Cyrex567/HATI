"""Regression guard on the footprint containment test.

Two ingests were spent on frames that image nothing at the touchdown. The first
was ODE's bounding-box query; the fix was to test the footprint polygon. The
second was the polygon test itself, run in raw lon/lat degrees, where at 84.8 S
a strip edge spanning tens of degrees of longitude is a straight line in a space
stretched by a factor of ten, so the polygon balloons and swallows the site. All
five frames of that run passed and all five came back empty; measured in a
projected frame they miss by 0.4 to 3.0 km.

These tests build strips of known geometry, so they need no network and cannot
drift with the archive.

    python tests/test_footprint.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from solar_sweep_query import _project, site_margin_m  # noqa: E402

R = 1737400.0
SITE_LAT, SITE_LON = -84.7906, 29.1957
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def unproject(x: float, y: float) -> tuple[float, float]:
    """Inverse of _project's south polar branch, so we can build strips in metres."""
    rho = math.hypot(x, y)
    lat = math.degrees(2 * math.atan(rho / (2 * R)) - math.pi / 2)
    lon = math.degrees(math.atan2(x, y))
    return lon % 360.0, lat


def strip(centre_offset_m: float, half_width_m: float = 2500.0,
          length_m: float = 60000.0, bearing_deg: float = 35.0) -> dict:
    """A NAC-like strip, offset sideways from the site by a known distance.

    Built in projected metres and handed back as lon/lat WKT, which is the form
    ODE delivers. centre_offset_m is the perpendicular distance from the site to
    the strip's centreline: 0 puts the site down the middle.
    """
    sx, sy = _project(SITE_LON, SITE_LAT, SITE_LAT)
    b = math.radians(bearing_deg)
    ux, uy = math.cos(b), math.sin(b)        # along the strip
    nx, ny = -uy, ux                         # across it
    cx, cy = sx + nx * centre_offset_m, sy + ny * centre_offset_m
    corners = []
    for a, w in ((+1, +1), (+1, -1), (-1, -1), (-1, +1)):
        px = cx + ux * a * length_m / 2 + nx * w * half_width_m
        py = cy + uy * a * length_m / 2 + ny * w * half_width_m
        corners.append(unproject(px, py))
    corners.append(corners[0])
    wkt = "POLYGON ((" + ", ".join(f"{lo:.6f} {la:.6f}" for lo, la in corners) + "))"
    return {"Footprint_geometry": wkt}


def main() -> None:
    print("footprint containment")

    m = site_margin_m(strip(0.0), SITE_LAT, SITE_LON)
    check("site down the middle of a 5 km strip is ~2.5 km inside",
          m is not None and 2300 < m < 2700, f"margin {m:.0f} m")

    m = site_margin_m(strip(6000.0), SITE_LAT, SITE_LON)
    check("strip offset 6 km sideways reports the site outside",
          m is not None and m < 0, f"margin {m:.0f} m")

    m = site_margin_m(strip(6000.0), SITE_LAT, SITE_LON)
    check("...and reports the miss distance to within a few hundred metres",
          m is not None and -4000 < m < -3000, f"margin {m:.0f} m")

    m = site_margin_m(strip(2400.0), SITE_LAT, SITE_LON)
    check("a strip that only clips the site reports a small positive margin",
          m is not None and 0 < m < 400, f"margin {m:.0f} m")

    # The failure that cost a run, pinned with the real footprints ODE serves.
    # M1160078961RE was selected by the degree-space test and came back with no
    # data at all; M1101118606LE is one of the three frames the reference NOBILE03
    # DTM is built from, so it demonstrably images the site. Note how similar the
    # two polygons look in degrees -- both are slivers running from about 22-29 E
    # at 83-84 S down to 35-36 E at 86 S. That is the whole problem: in degree
    # space those long edges are straight lines, on the ground they bow by
    # kilometres, and the test cannot tell the two frames apart.
    from solar_sweep_query import _contains, _polygon
    real = {
        "M1160078961RE": ("POLYGON ((22.7 -83.36, 35.62 -85.98, 34.6 -86.02, "
                          "22.02 -83.39, 22.7 -83.36))", False),
        "M1101118606LE": ("POLYGON ((28.78 -84.54, 36.59 -86.39, 35.59 -86.41, "
                          "28.1 -84.55, 28.78 -84.54))", True),
    }
    for pid, (wkt, truth) in real.items():
        m = site_margin_m({"Footprint_geometry": wkt}, SITE_LAT, SITE_LON)
        check(f"{pid}: projected test agrees with the pixels "
              f"({'images' if truth else 'misses'} the site)",
              m is not None and (m > 0) == truth, f"margin {m:.0f} m")

    degrees_says = {
        pid: _contains((SITE_LON % 360, SITE_LAT), _polygon(wkt))
        for pid, (wkt, _) in real.items()
    }
    check("in raw degrees both frames read as covering the site, which is why the "
          "degree-space test passed a frame that imaged nothing",
          all(degrees_says.values()),
          ", ".join(f"{k}={'inside' if v else 'outside'}" for k, v in degrees_says.items()))

    # symmetry: nothing about the fix may depend on which side of the site we sit
    left = site_margin_m(strip(-6000.0), SITE_LAT, SITE_LON)
    right = site_margin_m(strip(6000.0), SITE_LAT, SITE_LON)
    check("a miss to either side is measured the same",
          abs(left - right) < 200, f"{left:.0f} m vs {right:.0f} m")

    # and the low-latitude branch must still work, for non-polar sites
    eq_site_lat, eq_site_lon = -12.0, 45.0
    sx, sy = _project(eq_site_lon, eq_site_lat, eq_site_lat)
    check("the sub-60-degree branch scales longitude by cos(lat)",
          abs(sx - R * math.radians(eq_site_lon) * math.cos(math.radians(eq_site_lat))) < 1.0,
          f"x={sx:.0f} m")

    m = site_margin_m({"Footprint_geometry": None}, SITE_LAT, SITE_LON)
    check("a missing footprint is undetermined, not a silent yes", m is None)

    print()
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: " + ", ".join(FAILS))
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
