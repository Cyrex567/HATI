"""Find out why the co-registration windows came back empty.

The first real ingest produced four projected cubes and a co-registration report
of exactly (0.00, 0.00) with a nan error on every frame. That is not a perfect
alignment, it is what phase correlation returns when one input has no variance.
So the window being read from each projected cube contained no image data.

This asks, for every cube in the manifest, the only questions that matter:
where does the cube actually sit, where do we think the touchdown is, and do
those two agree. It reads pixels but computes nothing heavy.

    python scripts/diagnose_coreg.py

Prints, per cube: its CRS and bounds, whether the touchdown falls inside under
each of the four polar-stereographic sign conventions, and the statistics of the
window we would have correlated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import athena_counterfactual as ac  # noqa: E402

MANIFEST = ROOT / "data" / "sweep" / "manifest.json"
HALF = 400          # same half-window the ingest uses


def describe(tag: str, a: np.ndarray) -> str:
    if a.size == 0:
        return f"{tag}: EMPTY"
    fin = np.isfinite(a)
    if not fin.any():
        return f"{tag}: all non-finite"
    v = a[fin]
    return (f"{tag}: shape {a.shape} finite {100*fin.mean():5.1f}% "
            f"min {v.min():.4g} max {v.max():.4g} std {v.std():.4g} "
            f"{'<-- ZERO VARIANCE, no image data' if v.std() < 1e-6 else ''}")


def main() -> None:
    if not MANIFEST.exists():
        sys.exit(f"no manifest at {MANIFEST}; run the ingest first")
    frames = json.loads(MANIFEST.read_text())
    x, y = ac.touchdown_xy()
    print(f"touchdown lat/lon      : {ac.TD_LAT}, {ac.TD_LON}")
    print(f"touchdown_xy() as used : x={x:.1f}  y={y:.1f}  (metres, south polar stereographic)")

    # the reference ortho, for comparison
    try:
        ref = ac.load_ortho().astype("float32")
        rr, rc = ac.ortho_pixel()
        refc = ref[rr - HALF:rr + HALF, rc - HALF:rc + HALF]
        print("\nREFERENCE orthophoto")
        print("  " + describe("window", refc))
    except Exception as e:  # noqa: BLE001
        print(f"\nREFERENCE failed: {e}")

    for fr in frames:
        cub = Path(fr["lev2"])
        # the manifest stores the path from the machine that ran the ingest
        if not cub.exists():
            local = ROOT / "data" / "sweep" / cub.name
            cub = local if local.exists() else cub
        print(f"\n{'='*70}\n{fr['pid']}   az {fr.get('az_proxy', float('nan')):.1f}  "
              f"elev {fr.get('elev', float('nan')):.2f}")
        if not cub.exists():
            print(f"  cube not found: {cub}")
            continue
        with rasterio.open(cub) as src:
            b = src.bounds
            print(f"  size   : {src.width} x {src.height}   nodata={src.nodata}")
            print(f"  crs    : {str(src.crs)[:70]}")
            print(f"  bounds : x {b.left:.1f} .. {b.right:.1f}   y {b.bottom:.1f} .. {b.top:.1f}")

            # which sign convention puts the touchdown inside this cube?
            print("  touchdown inside bounds under each sign convention:")
            hit = None
            for sx, sy, name in [(1, 1, "( x,  y)"), (-1, 1, "(-x,  y)"),
                                 (1, -1, "( x, -y)"), (-1, -1, "(-x, -y)")]:
                tx, ty = sx * x, sy * y
                inside = (b.left <= tx <= b.right) and (b.bottom <= ty <= b.top)
                print(f"    {name} = ({tx:12.1f}, {ty:12.1f})  {'INSIDE' if inside else 'outside'}")
                if inside and hit is None:
                    hit = (tx, ty, name)

            if hit is None:
                print("  >> the touchdown is outside this cube under EVERY convention.")
                print("     the cam2map extent does not cover the landing site.")
                continue

            tx, ty, name = hit
            r0, c0 = src.index(tx, ty)
            print(f"  >> {name} lands at pixel row {r0}, col {c0}")
            win = Window(c0 - HALF, r0 - HALF, 2 * HALF, 2 * HALF)
            mov = src.read(1, window=win, boundless=True,
                           fill_value=float("nan")).astype("float32")
            print("  " + describe("window", mov))

    print(f"\n{'='*70}")
    print("Reading: if the touchdown only lands INSIDE under a flipped sign, the fix is a\n"
          "sign convention mismatch between touchdown_xy() and the cam2map output. If it is\n"
          "outside under every convention, widen the extent in scripts/sweep_polar.map (the\n"
          "MinimumLatitude/MaximumLatitude and longitude range) and re-run cam2map. If it is\n"
          "inside but the window has zero variance, the cube is valid but empty there, which\n"
          "means the frame does not actually image the site.")


if __name__ == "__main__":
    main()
