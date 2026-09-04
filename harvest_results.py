"""Collect every result the pipeline produced into one small zip you can carry home.

The working tree gets large (NAC frames, ISIS cubes, projected rasters). Almost none
of that is worth moving. This gathers only the things you cannot regenerate without
re-running: the figures, the reports, the co-registration budget, the sweep manifest,
the CSVs, the compiled PDFs and the run logs.

    python3 harvest_results.py                 # -> HATI_results_<host>_<date>.zip
    python3 harvest_results.py --out ~/Desktop # choose where it lands
    python3 harvest_results.py --include-cubes # also take the projected .lev2 cubes (GB)

Run it on the machine that did the work. Copy the resulting zip anywhere.
"""
from __future__ import annotations

import argparse
import datetime as dt
import platform
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (glob pattern, why it matters) relative to the project root
WANTED = [
    ("output/athena/*.png",        "figures"),
    ("output/athena/*.csv",        "sweep census and tables"),
    ("output/athena/*.md",         "written reports"),
    ("output/heatmap/*.png",       "heatmap figures"),
    ("data/sweep/coreg_report.csv", "co-registration error budget"),
    ("data/sweep/manifest.json",   "ingested sweep manifest"),
    ("data/sweep/plan.json",       "ingest dry-run plan"),
    ("paper/*.pdf",                "compiled documents"),
    ("logs/*.log",                 "run logs"),
    ("logs/*.txt",                 "run logs"),
]
BIG = [("data/sweep/*.lev2.cub", "projected cubes (large)")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".", help="directory to write the zip into")
    ap.add_argument("--include-cubes", action="store_true",
                    help="also include the projected ISIS cubes (adds gigabytes)")
    args = ap.parse_args()

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M")
    host = platform.node().split(".")[0] or "box"
    dest = Path(args.out).expanduser().resolve() / f"HATI_results_{host}_{stamp}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)

    patterns = WANTED + (BIG if args.include_cubes else [])
    total = 0
    manifest: list[str] = [f"HATI results harvest", f"host: {platform.node()}",
                           f"platform: {platform.platform()}", f"taken: {stamp}", ""]

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for pattern, why in patterns:
            hits = sorted(ROOT.glob(pattern))
            if not hits:
                manifest.append(f"[none] {pattern}  ({why})")
                continue
            for f in hits:
                if not f.is_file():
                    continue
                rel = f.relative_to(ROOT)
                z.write(f, str(rel))
                total += f.stat().st_size
                manifest.append(f"{f.stat().st_size/1e6:9.2f} MB  {rel}")
            print(f"  + {len(hits):3d}  {pattern}")
        z.writestr("HARVEST_MANIFEST.txt", "\n".join(manifest) + "\n")

    print(f"\ncollected {total/1e6:.1f} MB of results")
    print(f"  -> {dest}")
    if not args.include_cubes:
        print("  (projected ISIS cubes were skipped; add --include-cubes if you need them)")


if __name__ == "__main__":
    main()
