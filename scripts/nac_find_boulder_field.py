"""Sweep the NAC frame for the boulder-field crop with the strongest
visual story: many resolved boulder shadows plus a rich sub-pixel
population.

For each candidate crop position, count shadow blobs by size class.
Pick the crops where:
  - resolved (>20 px) count is high  -> obvious large boulders for email
  - sub-pixel (2-5 px) count is also high  -> rich sub-Nyquist signal
Render the top 3 as annotated visualisations.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
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
PIXEL_SCALE_M = 0.808
CROP = 500          # px
THRESHOLD = 28


def load_nac() -> np.ndarray:
    with IMG.open("rb") as f:
        f.seek(HEADER_OFFSET)
        return np.frombuffer(f.read(LINES * SAMPLES), dtype=np.uint8).reshape(
            LINES, SAMPLES
        )


def find_blobs(crop: np.ndarray):
    shadow_mask = crop < THRESHOLD
    shadow_mask = ndi.binary_closing(shadow_mask, iterations=1)
    labelled, n = ndi.label(shadow_mask)
    sub = mar = res = 0
    blobs = []
    for label_id in range(1, n + 1):
        mask = labelled == label_id
        area = int(mask.sum())
        if area < 2:
            continue
        ys, xs = np.where(mask)
        cy, cx = float(ys.mean()), float(xs.mean())
        blobs.append((area, cy, cx))
        if area <= 5:
            sub += 1
        elif area <= 20:
            mar += 1
        else:
            res += 1
    return blobs, sub, mar, res


def score(sub: int, mar: int, res: int) -> float:
    """Weight 'lots of resolved boulders' most heavily, then sub-pixel.
    Penalise crops where everything is resolved (no sub-pixel signal)
    and crops that are entirely sub-pixel noise."""
    if res == 0 and mar == 0:
        return 0.0
    return res * 3 + mar * 1.5 + sub * 0.4


def render(crop: np.ndarray, blobs, out_path: Path, title: str,
           sub: int, mar: int, res: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    p1, p99 = np.percentile(crop, [0.5, 99])
    for ax in axes:
        ax.imshow(crop, cmap="gray", vmin=p1, vmax=p99, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].set_title(
        f"NAC raw, contrast-stretched\n"
        f"({crop.shape[1] * PIXEL_SCALE_M:.0f} m x "
        f"{crop.shape[0] * PIXEL_SCALE_M:.0f} m, native 0.81 m/px)",
        fontsize=11, fontweight="bold",
    )

    for area, cy, cx in blobs:
        if area <= 5:
            color = "#ff3838"
            patch = Circle((cx, cy), radius=7, fill=False,
                           edgecolor=color, linewidth=1.2, alpha=0.9)
        elif area <= 20:
            r = max(9, np.sqrt(area / np.pi) * 2.5)
            color = "#ffcc00"
            patch = Circle((cx, cy), radius=r, fill=False,
                           edgecolor=color, linewidth=1.4, alpha=0.95)
        else:
            r = max(12, np.sqrt(area / np.pi) * 2.5)
            color = "#00bfff"
            patch = Rectangle((cx - r, cy - r), 2 * r, 2 * r,
                              fill=False, edgecolor=color, linewidth=1.4,
                              alpha=0.95)
        axes[1].add_patch(patch)

    axes[1].set_title(
        f"Shadow detections, sized by area\n"
        f"red = sub-pixel ({sub})  yellow = marginal ({mar})  "
        f"cyan = resolved ({res})",
        fontsize=11, fontweight="bold",
    )

    # Scale bar
    bar_m = 50.0
    bar_px = bar_m / PIXEL_SCALE_M
    bar_x0 = crop.shape[1] - bar_px - 30
    bar_y0 = crop.shape[0] - 30
    for ax in axes:
        ax.plot([bar_x0, bar_x0 + bar_px], [bar_y0, bar_y0],
                color="white", linewidth=4, solid_capstyle="butt")
        ax.text(bar_x0 + bar_px / 2, bar_y0 - 14, f"{bar_m:.0f} m",
                color="white", ha="center", fontsize=10,
                bbox=dict(facecolor="black", alpha=0.6, pad=2,
                          edgecolor="none"))

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"Loading {FRAME}")
    arr = load_nac()

    # Sweep along the long axis. Cross-track, pick three sample positions.
    line_positions = list(range(2000, LINES - CROP, 1500))
    sample_positions = [600, (SAMPLES - CROP) // 2, SAMPLES - CROP - 600]
    print(f"Scanning {len(line_positions) * len(sample_positions)} candidate crops...")

    results = []
    for line in line_positions:
        for sample in sample_positions:
            crop = arr[line:line + CROP, sample:sample + CROP]
            if crop.shape != (CROP, CROP):
                continue
            blobs, sub, mar, res = find_blobs(crop)
            s = score(sub, mar, res)
            results.append((s, line, sample, sub, mar, res, blobs, crop))

    results.sort(key=lambda r: -r[0])
    print(f"\nTop 5 by score (3*res + 1.5*mar + 0.4*sub):")
    for i, (s, line, sample, sub, mar, res, _, _) in enumerate(results[:5]):
        print(f"   [{i+1}] line={line:5d} sample={sample:4d}  "
              f"score={s:6.1f}  sub={sub:3d}  mar={mar:3d}  res={res:3d}")

    # Render top 3
    for i, (s, line, sample, sub, mar, res, blobs, crop) in enumerate(results[:3]):
        tag = f"top{i+1}_line{line}_sample{sample}"
        out_path = OUT / f"{FRAME}_{tag}.png"
        subtitle = (
            f"Apollo 17 site, line {line}, sample {sample}  "
            f"(score={s:.1f}: {res} resolved + {mar} marginal + {sub} sub-pixel)"
        )
        render(crop, blobs, out_path,
               title=f"{FRAME}  -  {subtitle}",
               sub=sub, mar=mar, res=res)
        print(f"   -> {out_path}")


if __name__ == "__main__":
    main()
