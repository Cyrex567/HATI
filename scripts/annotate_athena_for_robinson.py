"""Annotate the LROC team's published M1495922531 oblique image of the
IM-2 Athena landing site, marking the lander/crater for the reply to
Mark Robinson.

Source attribution: LROC team's featured image post #1408,
https://lroc.im-ldi.com/images/1408, oblique NAC M1495922531LR.

The lander is the bright spot near image centre. The 20 m crater it
tipped into is co-located with the lander. We add a single arrow at
the lander, a separate inset zoom showing the crater rim, a coordinates
box, and a clean caption identifying what's annotated.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "lroc_published" / "athena_lroc_hero_1300p.png"
OUT = ROOT / "output" / "athena_for_robinson.png"

# Centre the annotation on the LROC team's own white-pointer arrow
# (baked into the published mosaic). That's the authoritative marker;
# trying to pick a single lander pixel separately is overreach.
LROC_ARROW_X = 540
LROC_ARROW_Y = 555


def main() -> None:
    img = np.array(Image.open(SRC).convert("L"))
    h, w = img.shape
    print(f"Image: {w} x {h}")

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1])
    ax_main = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])

    # --- Main panel: full LROC image
    ax_main.imshow(img, cmap="gray", vmin=np.percentile(img, 1),
                   vmax=np.percentile(img, 99))
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    # Red circle around the LROC team's own arrow marker -- the
    # authoritative pointer at the lander.
    circle_main = Circle(
        (LROC_ARROW_X, LROC_ARROW_Y), radius=35,
        fill=False, edgecolor="#ff3838", linewidth=2.5, alpha=0.95,
    )
    ax_main.add_patch(circle_main)
    ax_main.text(
        LROC_ARROW_X + 50, LROC_ARROW_Y - 50,
        "IM-2 Athena lander\n(LROC marker, ~20 m crater)",
        color="#ff3838", fontsize=10, fontweight="bold",
        bbox=dict(facecolor="black", alpha=0.75, pad=4, edgecolor="#ff3838"),
    )

    # Coordinates box (lower-left)
    coord_text = (
        "Touchdown location\n"
        "  84.7906 deg S, 29.1957 deg E\n"
        "  Mons Mouton, lunar south pole\n"
        "Crater diameter\n"
        "  ~20 m  (LROC M1495922531LR)\n"
        "Observation\n"
        "  2025-03-07 16:54:21 UTC\n"
        "  ~23.5 h post-touchdown"
    )
    ax_main.text(
        12, h - 12, coord_text,
        color="white", fontsize=9, fontfamily="monospace",
        va="bottom", ha="left",
        bbox=dict(facecolor="black", alpha=0.75, pad=6,
                  edgecolor="white", linewidth=0.5),
    )

    ax_main.set_title(
        "IM-2 Athena landing site, LROC NAC oblique view "
        "(M1495922531LR, 2025-03-07)",
        fontsize=11, fontweight="bold",
    )

    # --- Zoom panel: 200x200 px crop centred on the LROC arrow
    half = 100
    x0 = max(0, LROC_ARROW_X - half)
    y0 = max(0, LROC_ARROW_Y - half)
    x1 = min(w, LROC_ARROW_X + half)
    y1 = min(h, LROC_ARROW_Y + half)
    zoom_crop = img[y0:y1, x0:x1]
    ax_zoom.imshow(zoom_crop, cmap="gray",
                   vmin=np.percentile(zoom_crop, 1),
                   vmax=np.percentile(zoom_crop, 99),
                   extent=[x0, x1, y1, y0])
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
    ax_zoom.set_title(
        "Zoom: ~200 px around LROC arrow\n"
        "(white arrow is the LROC team's marker)",
        fontsize=10, fontweight="bold",
    )

    # Red circle around the LROC arrow in the zoom too
    circ_zoom = Circle(
        (LROC_ARROW_X, LROC_ARROW_Y), radius=25,
        fill=False, edgecolor="#ff3838", linewidth=2, alpha=0.9,
    )
    ax_zoom.add_patch(circ_zoom)

    # Draw the zoom-region rectangle on the main panel (centred on arrow)
    rect_x = [x0, x1, x1, x0, x0]
    rect_y = [y0, y0, y1, y1, y0]
    ax_main.plot(rect_x, rect_y, color="#ffcc00", linewidth=1.2, alpha=0.8)
    ax_main.text(x0, y0 - 8, "zoom region", color="#ffcc00",
                 fontsize=8, fontweight="bold")

    fig.suptitle(
        "IM-2 Athena: annotated landing site for the small crater of interest",
        fontsize=13, fontweight="bold", y=1.005,
    )

    # Source attribution
    fig.text(
        0.5, -0.01,
        "Base image: NASA / Goddard Space Flight Center / LROC team, "
        "Arizona State University, post #1408 (lroc.im-ldi.com/images/1408). "
        "Annotation: G. Csaba Morvai, HATI v2.0 forensic reconstruction.",
        ha="center", fontsize=8, style="italic", color="#444444",
    )

    fig.tight_layout()
    fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
