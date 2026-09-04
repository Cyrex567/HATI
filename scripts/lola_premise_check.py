"""Independent check on the premise the mare control rests on.

The cross-site control assumes Mons Mouton (massif) is rough terrain and Mare
Tranquillitatis is smooth terrain, then scores channels on whether they rank the
massif above the mare. If that assumption were wrong, every AUC in the control
would be measuring the wrong thing.

This tests the assumption with a completely different instrument. LOLA is a laser
altimeter, not a camera, and its median-absolute-slope maps are in degrees and
cover the poles (unlike Diviner rock abundance and the WAC Hapke parameter maps,
both of which stop at 70 degrees latitude).

Resolution is ~1.9 km/px, far too coarse to score individual hazard pixels. It is
the right scale to answer one narrow question: at regional scale, is the massif
actually rougher than the mare?
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "lola"
BASE = ("https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
        "lrolol_1xxx/EXTRAS/fractal/img/")
PRODUCT = "mas_57m_16"          # median absolute slope, 57 m baseline

# (name, lat, lon_east, half-window in degrees lat)
SITES = [
    ("Mons Mouton (Athena, massif)", -84.7906, 29.1957, 0.25),
    ("Mare Tranquillitatis (Apollo 11)", 0.6740, 23.4730, 0.25),
]


def fetch(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / name
    if out.exists() and out.stat().st_size > 1000:
        print(f"  cached {name} ({out.stat().st_size/1e6:.1f} MB)")
        return out
    print(f"  downloading {name} ...", flush=True)
    req = urllib.request.Request(BASE + name, headers={"User-Agent": "HATI/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(out, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    print(f"  got {name} ({out.stat().st_size/1e6:.1f} MB)")
    return out


def label_value(txt: str, key: str, default=None):
    m = re.search(rf"^\s*{key}\s*=\s*([^\r\n]+)", txt, re.M | re.I)
    if not m:
        return default
    return m.group(1).strip().strip('"').split()[0]


def main() -> None:
    print("LOLA median absolute slope, independent premise check")
    lbl = fetch(PRODUCT + ".lbl").read_text(errors="replace")
    img = fetch(PRODUCT + ".img")

    lines = int(float(label_value(lbl, "LINES")))
    samples = int(float(label_value(lbl, "LINE_SAMPLES")))
    bits = int(float(label_value(lbl, "SAMPLE_BITS", 32)))
    stype = str(label_value(lbl, "SAMPLE_TYPE", "PC_REAL")).upper()
    res = float(label_value(lbl, "MAP_RESOLUTION", 16))          # px per degree
    clat = float(label_value(lbl, "CENTER_LATITUDE", 0))
    clon = float(label_value(lbl, "CENTER_LONGITUDE", 0))
    lproj = float(label_value(lbl, "LINE_PROJECTION_OFFSET", lines / 2))
    sproj = float(label_value(lbl, "SAMPLE_PROJECTION_OFFSET", samples / 2))
    scale = float(label_value(lbl, "SCALING_FACTOR", 1) or 1)
    offset = float(label_value(lbl, "OFFSET", 0) or 0)
    missing = label_value(lbl, "MISSING_CONSTANT")

    dtype = ("<f4" if "PC_REAL" in stype or "LSB" in stype else ">f4") if bits == 32 else "<i2"
    print(f"  grid {lines} x {samples} | {bits}-bit {stype} | {res} px/deg "
          f"| centre ({clat}, {clon}) | offsets L{lproj} S{sproj}")

    a = np.fromfile(img, dtype=dtype)
    if a.size < lines * samples:
        raise SystemExit(f"file too small: {a.size} values for {lines}x{samples}")
    a = a[:lines * samples].reshape(lines, samples).astype(np.float32)
    if missing is not None:
        try:
            a[a == float(missing)] = np.nan
        except ValueError:
            pass
    a[a < -1e30] = np.nan
    a = a * scale + offset

    print(f"\n{'site':<36}{'n px':>7}{'median':>9}{'p25':>8}{'p75':>8}   (degrees)")
    print("-" * 74)
    stats = {}
    for name, lat, lon, half in SITES:
        # PDS simple-cylindrical convention
        line = lproj + (clat - lat) * res
        samp = sproj + (lon - clon) * res
        dl = max(int(half * res), 1)
        ds = max(int(dl / max(np.cos(np.radians(lat)), 0.02)), 1)
        r0, r1 = int(line) - dl, int(line) + dl + 1
        c0, c1 = int(samp) - ds, int(samp) + ds + 1
        r0, r1 = max(r0, 0), min(r1, lines)
        cols = [c % samples for c in range(c0, c1)]
        win = a[r0:r1][:, cols]
        v = win[np.isfinite(win)]
        if v.size == 0:
            print(f"{name:<36}{'no data':>7}")
            continue
        stats[name] = float(np.median(v))
        print(f"{name:<36}{v.size:>7}{np.median(v):>9.2f}{np.percentile(v,25):>8.2f}"
              f"{np.percentile(v,75):>8.2f}")

    if len(stats) == 2:
        massif, mare = (stats[SITES[0][0]], stats[SITES[1][0]])
        print(f"\nmassif / mare ratio: {massif/max(mare,1e-6):.2f}x")
        if massif > mare:
            print("PREMISE HOLDS: an independent instrument agrees the massif is the rougher\n"
                  "site, so the cross-site control was scoring channels against a real contrast.")
        else:
            print("PREMISE FAILS: LOLA does not see the massif as rougher. The control's\n"
                  "labelling would need rethinking before its AUCs mean anything.")
    print("\nScale caveat: ~1.9 km per pixel. This checks the regional contrast the control\n"
          "assumes, not any individual hazard.")


if __name__ == "__main__":
    main()
