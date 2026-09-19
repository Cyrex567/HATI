"""Figures for the saturation analysis report.

These are drawn from the run's own audit_statistics.json where possible, and from
closed-form arithmetic where the point is the arithmetic itself. Nothing here
reads a cube or invokes ISIS; it is safe to run on a laptop.

    python make_figures.py
"""
from __future__ import annotations

import math
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "fig"
FIG.mkdir(exist_ok=True)

NAVY = "#0D1B2A"
BLUE = "#1F3A5F"
RED = "#C1121F"
SILVER = "#8B97A5"
TEAL = "#1B7A6E"
AMBER = "#B8860B"
GRID = "#D8DEE7"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Georgia", "DejaVu Serif"],
    "font.size": 9,
    "axes.edgecolor": SILVER,
    "axes.labelcolor": NAVY,
    "text.color": NAVY,
    "xtick.color": BLUE,
    "ytick.color": BLUE,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "figure.dpi": 200,
})

# Values verified against run.json, candidates.csv and coreg_report.csv.
# Eligibility counts are retained in the snapshot; these are selected-root
# summaries, not independent frame measurements or a causal regression.
VERIFIED = json.loads((HERE / "verified_run_data.json").read_text(encoding="utf-8"))
FRAMES = [(p['pid'].removeprefix('nac.'), p['az_map'], p['elev'], p['ncc'],
           p['residual_px'], p['median_delta'], p['positive_fraction'])
          for p in VERIFIED['frames']]


def fig_geometry() -> None:
    """The shadow law and the sweep, drawn rather than described."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))

    # --- left: side view, h and L
    ax1.set_xlim(-1, 13)
    ax1.set_ylim(-1.2, 3.2)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.plot([-0.5, 12.5], [0, 0], color=SILVER, lw=1.4)
    ax1.add_patch(plt.Rectangle((10.0, 0), 0.32, 0.5, color=NAVY, zorder=3))
    ax1.fill([10.0, 10.0, 0.46], [0, 0.5, 0], color=RED, alpha=0.16, zorder=1)
    ax1.plot([10.0, 0.46], [0.5, 0], color=RED, lw=1.3, zorder=2)
    ax1.annotate("", xy=(10.6, 0.60), xytext=(12.4, 1.5),
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0))
    ax1.text(12.5, 1.62, "Sun, e $\\approx$ 3$\\degree$", color=BLUE, fontsize=8, ha="right")
    ax1.annotate("", xy=(10.55, 0), xytext=(10.55, 0.5),
                 arrowprops=dict(arrowstyle="<->", color=NAVY, lw=0.9))
    ax1.text(10.85, 0.24, "h", color=NAVY, fontsize=9, style="italic")
    ax1.annotate("", xy=(0.46, -0.42), xytext=(10.0, -0.42),
                 arrowprops=dict(arrowstyle="<->", color=NAVY, lw=0.9))
    ax1.text(6.3, -0.85, "L = h cot e $\\approx$ 19h", color=NAVY, fontsize=8.5, ha="center")
    ax1.text(6.0, 2.55, "Illustrative height and shadow geometry\nCaster width is a separate parameter",
             color=BLUE, fontsize=8, ha="center")

    # --- right: top view, four azimuths converging on one base
    ax2.set_xlim(-2.1, 2.1)
    ax2.set_ylim(-2.1, 2.1)
    ax2.set_aspect("equal")
    ax2.axis("off")
    circ = plt.Circle((0, 0), 1.65, fill=False, color=SILVER, ls=(0, (3, 3)), lw=0.8)
    ax2.add_patch(circ)
    shades = [RED, "#D2434E", "#DE6A73", "#E89096"]
    for ang, col in zip([28, 92, 158, 236], shades):
        r = math.radians(ang)
        ax2.plot([0, 1.55 * math.cos(r)], [0, 1.55 * math.sin(r)], color=col, lw=2.4,
                 solid_capstyle="round")
    ax2.plot(0, 0, "o", color=NAVY, ms=5, zorder=5)
    ax2.text(0, -1.95, "Four frames, four shadow directions, one base.\n"
                       "A stain painted on the ground gives no convergence.",
             color=BLUE, fontsize=8, ha="center")

    fig.tight_layout()
    fig.savefig(FIG / "fig_geometry.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_occupancy() -> None:
    """Sensitivity to fixed diameter; no inferred rock abundance or geology band."""
    cell_area = (4 * .9)**2
    f = np.linspace(0, .25, 600)
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    for diameter, color in ((.3, TEAL), (.6, BLUE), (1.2, AMBER)):
        area = math.pi*diameter**2/4
        occupancy = 1-np.exp(-f*cell_area/area)
        ax.plot(f*100, occupancy*100, color=color, lw=1.8, label=f"d = {diameter} m")
        target = -math.log(1-.891)*area/cell_area*100
        ax.plot(target, 89.1, 'o', color=color, ms=4)
        ax.annotate(f"{target:.2f}%", (target,89.1), xytext=(4,-17),
                    textcoords='offset points', fontsize=8, color=color)
    ax.axhline(89.1, color=RED, lw=1, ls='--')
    ax.text(24,94,'89.1% used as hypothetical occupancy',ha='right',color=RED,fontsize=8)
    ax.set_xlabel('Summed circular planform area / ground area (%)')
    ax.set_ylabel('Cells containing at least one centre (%)')
    ax.set_xlim(0,25);ax.set_ylim(0,100)
    ax.set_title('Poisson sensitivity example: diameter is not height',fontsize=10)
    ax.legend(frameon=False,loc='lower right',fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'fig_occupancy.png',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def fig_frame_evidence() -> None:
    """Descriptive selected-root summaries; geometry and frame are confounded."""
    cot = np.array([1 / math.tan(math.radians(f[2])) for f in FRAMES])
    d = np.array([f[5] for f in FRAMES])       # median delta chi^2, not closure
    ncc = np.array([f[3] for f in FRAMES])
    names = [f[0] for f in FRAMES]
    is_re = np.array([n.endswith("re") for n in names])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.4))

    # --- left: evidence versus measured geometry; no universal expected slope.
    for i in range(len(FRAMES)):
        col = AMBER if is_re[i] else BLUE
        ax1.plot(cot[i], d[i], "o", color=col, ms=7, zorder=4)
        ax1.annotate(str(i), (cot[i], d[i]), textcoords="offset points",
                     xytext=(0, 9), fontsize=6.2, ha="center", color=SILVER)
    ax1.set_yscale("log")
    ax1.set_xlabel("cot e  (shadow length per unit height)")
    ax1.set_ylabel("median $\\Delta\\chi^2$ per root")
    ax1.set_title("Evidence against shadow length", color=NAVY, fontsize=9.5, pad=6)
    ax1.set_xlim(12, 19)

    ax1.set_ylim(0.8, 600)
    ax1.text(.04,.08,"Signed joint-fit contributions; different eligible root sets.\n"
                         "No isolated elevation effect is inferred.",transform=ax1.transAxes,
             fontsize=7,color=BLUE)

    # --- right: evidence vs aligned NCC
    for i in range(len(FRAMES)):
        col = AMBER if is_re[i] else BLUE
        ax2.plot(ncc[i], d[i], "o", color=col, ms=7, zorder=4)
        ax2.annotate(str(i), (ncc[i], d[i]), textcoords="offset points",
                     xytext=(0, 9), fontsize=6.2, ha="center", color=SILVER)
    ax2.set_yscale("log")
    ax2.set_ylim(0.8, 600)
    ax2.set_xlabel("aligned NCC against the reference")
    ax2.set_title("Evidence against reference NCC", color=NAVY, fontsize=9.5, pad=6)
    ax2.set_xlim(0.2, 1.0)

    handles = [plt.Line2D([], [], marker="o", ls="", color=BLUE, label="LE channel"),
               plt.Line2D([], [], marker="o", ls="", color=AMBER, label="RE channel")]
    ax2.legend(handles=handles, fontsize=7.5, frameon=False, loc="lower left")

    fig.tight_layout()
    fig.savefig(FIG / "fig_frame_evidence.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_template_bank() -> None:
    """Template selection counts, not an inferred physical size distribution."""
    heights = {k+" m":v for k,v in VERIFIED["heights"].items()}
    widths = {k+" m":v for k,v in VERIFIED["widths"].items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    ax1.bar(list(heights), list(heights.values()), color=TEAL, width=0.55)
    ax1.set_title("Selected height under the template bank\nNot a measured size-frequency distribution",
                  color=NAVY, fontsize=9, pad=6)
    ax1.set_ylabel("retained roots")
    ax1.grid(axis="x", visible=False)

    bars = ax2.bar(list(widths), list(widths.values()), color=[SILVER, RED], width=0.42)
    ax2.set_title("Fitted width: 93 per cent pin at the\nwidest template the bank offers",
                  color=NAVY, fontsize=9, pad=6)
    ax2.grid(axis="x", visible=False)
    ax2.annotate("bank maximum", xy=(1, 6052), xytext=(0.55, 4300),
                 fontsize=7.6, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))

    fig.tight_layout()
    fig.savefig(FIG / "fig_template_bank.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_geometry()
    fig_occupancy()
    fig_frame_evidence()
    fig_template_bank()
    for p in sorted(FIG.glob("*.png")):
        print("wrote", p.name, f"{p.stat().st_size/1024:.0f} KB")
