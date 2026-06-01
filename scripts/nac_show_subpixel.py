"""Annotate the sub-pixel hazards visible in a NAC frame.

Stage A produced a centre crop where individual sub-pixel boulders are
hard to see at a glance. This script:

1. Loads the same NAC frame.
2. Takes a much smaller, more zoomed-in crop (320 m square, 400x400 px).
3. Applies a contrast stretch that emphasises the dark end (where shadows
   live).
4. Runs a simple shadow detector: threshold + connected components.
5. Classifies each shadow blob by size:
     * 2-5 px area:   sub-pixel boulders (the population invisible to the
                       1.5 m/px DEM)
     * 6-20 px area:  marginal-resolution features (right at the DEM's
                       detection limit)
     * 21+ px area:   features the DEM already resolves
6. Renders the crop with each class annotated in a distinct colour, plus
   a side-by-side "raw image / shadow detections" comparison.

The sun azimuth was ~313 deg at acquisition, so shadows extend roughly
to the south-east in the image. Boulder shadow direction provides an
internal cross-check.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "nac"
OUT = ROOT / "output" / "nac"
OUT.mkdir(parents=True, exist_ok=True)

FRAME = "M1412631647RE"
IMG = DATA / f"{FRAME}.IMG"
LINES = 52224
SAMPLES = 5064
HEADER_OFFSET = 5064
PIXEL_SCALE_M = 0.808  # native NAC resolution


def load_nac() -> np.ndarray:
    with IMG.open("rb") as f:
        f.seek(HEADER_OFFSET)
        return np.frombuffer(f.read(LINES * SAMPLES), dtype=np.uint8).reshape(
            LINES, SAMPLES
        )


def find_shadow_blobs(crop: np.ndarray, threshold: int = 28):
    """Threshold the crop, label connected dark regions, compute size.

    Returns the labelled array and a list of (label_id, area_px, cy, cx).
    """
    shadow_mask = crop < threshold
    # Light morphological cleanup: close 1-px gaps so a smeared shadow
    # isn't counted as several tiny ones.
    shadow_mask = ndi.binary_closing(shadow_mask, iterations=1)
    labelled, n = ndi.label(shadow_mask)
    blobs = []
    for label_id in range(1, n + 1):
        mask = labelled == label_id
        area = int(mask.sum())
        if area < 2:
            continue
        ys, xs = np.where(mask)
        cy = float(ys.mean())
        cx = float(xs.mean())
        blobs.append((label_id, area, cy, cx))
    return labelled, blobs


def classify(area_px: int) -> str:
    if area_px <= 5:
        return "subpixel"      # boulder smaller than DEM resolution
    if area_px <= 20:
        return "marginal"      # right at DEM resolution limit
    return "resolved"          # DEM-resolved


CLASS_COLOR = {
    "subpixel": "#ff3838",     # red
    "marginal": "#ffcc00",     # yellow
    "resolved": "#00bfff",     # cyan
}


def render(crop, blobs, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    # Left: contrast-stretched raw NAC
    p1, p99 = np.percentile(crop, [0.5, 99])
    axes[0].imshow(crop, cmap="gray", vmin=p1, vmax=p99, interpolation="nearest")
    axes[0].set_title(
        "NAC raw, contrast-stretched\n"
        f"({crop.shape[1] * PIXEL_SCALE_M:.0f} m x {crop.shape[0] * PIXEL_SCALE_M:.0f} m "
        f"at {PIXEL_SCALE_M:.2f} m/px)",
        fontsize=11, fontweight="bold",
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Right: NAC + shadow blob annotations
    axes[1].imshow(crop, cmap="gray", vmin=p1, vmax=p99, interpolation="nearest")
    counts = {"subpixel": 0, "marginal": 0, "resolved": 0}
    for label_id, area, cy, cx in blobs:
        cls = classify(area)
        counts[cls] += 1
        color = CLASS_COLOR[cls]
        if cls == "subpixel":
            # Small red circle just big enough to be visible
            circle = Circle(
                (cx, cy), radius=6, fill=False, edgecolor=color,
                linewidth=1.2, alpha=0.9,
            )
            axes[1].add_patch(circle)
        elif cls == "marginal":
            # Yellow circle proportional to size
            r = max(8, np.sqrt(area / np.pi) * 2.5)
            circle = Circle(
                (cx, cy), radius=r, fill=False, edgecolor=color,
                linewidth=1.4, alpha=0.95,
            )
            axes[1].add_patch(circle)
        else:
            # Cyan square around resolved features
            r = max(10, np.sqrt(area / np.pi) * 2.5)
            sq = Rectangle(
                (cx - r, cy - r), 2 * r, 2 * r,
                fill=False, edgecolor=color, linewidth=1.4, alpha=0.95,
            )
            axes[1].add_patch(sq)
    axes[1].set_title(
        "Shadow-blob classification\n"
        f"red = sub-pixel ({counts['subpixel']}, area 2-5 px)   "
        f"yellow = marginal ({counts['marginal']}, 6-20 px)   "
        f"cyan = DEM-resolved ({counts['resolved']}, >20 px)",
        fontsize=11, fontweight="bold",
    )
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # Scale bar (50 m bar)
    bar_m = 50.0
    bar_px = bar_m / PIXEL_SCALE_M
    bar_x0 = crop.shape[1] - bar_px - 30
    bar_y0 = crop.shape[0] - 30
    for ax in axes:
        ax.plot([bar_x0, bar_x0 + bar_px], [bar_y0, bar_y0], color="white",
                linewidth=4, solid_capstyle="butt")
        ax.text(
            bar_x0 + bar_px / 2, bar_y0 - 14, f"{bar_m:.0f} m",
            color="white", ha="center", fontsize=10,
            bbox=dict(facecolor="black", alpha=0.6, pad=2, edgecolor="none"),
        )

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"Loading {FRAME}")
    arr = load_nac()

    # Pick two zoomed crops to demonstrate different terrain:
    #   (a) valley-floor mare with many small boulders / craters
    #   (b) slope with obvious large boulders

    # The image is line-scan, sun ~upper-left -> shadows toward lower-right.
    # The Apollo 17 area is in the middle of the image. The valley floor
    # is around line 30000-35000 in this frame.

    crops = [
        (26000, 2400, 400, "valley_floor",
         "Apollo 17 valley floor — sub-pixel boulders show as tiny dark "
         "spots tracking the anti-sun direction"),
        (40000, 800, 400, "massif_slope",
         "South Massif foothills — larger resolved boulders cast clear "
         "elongated shadows"),
    ]

    for line, sample, size, tag, subtitle in crops:
        crop = arr[line:line + size, sample:sample + size].copy()
        if crop.shape != (size, size):
            print(f"   skipping {tag}: out-of-bounds (got {crop.shape})")
            continue
        print(f"\n{tag}: crop {crop.shape} at line={line} sample={sample}")
        labelled, blobs = find_shadow_blobs(crop, threshold=28)
        sizes = {"subpixel": 0, "marginal": 0, "resolved": 0}
        for _, area, _, _ in blobs:
            sizes[classify(area)] += 1
        print(f"   detected: {sizes}")
        out_path = OUT / f"{FRAME}_{tag}_annotated.png"
        render(crop, blobs, out_path,
               title=f"{FRAME} — {subtitle}")
        print(f"   -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
