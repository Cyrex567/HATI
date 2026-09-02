"""Figure: which roughness channels survived the cross-site control.

Plots the measured cross-site AUC of every Tier-1 channel from mare_control_v3,
sorted, against the two lines that decide the verdict: 0.50 (coin flip) and 0.70
(the bar for keeping a channel). Numbers are the run of 2026-06-15.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "output" / "athena"

NAVY, SILVER, RED, TEAL = "#0d1b2a", "#8b97a5", "#c1121f", "#1b7a6e"

# channel, config A (posting-matched only), config C (fully matched)
ROWS = [
    ("RMS slope",   0.789, 0.826),
    ("TRI",         0.776, 0.808),
    ("Slope IQR",   0.434, 0.432),
    ("|TPI|",       0.420, 0.420),
    ("MDS @40 m",   0.376, 0.375),
    ("Curvature IQR", 0.215, 0.369),
    ("MDS @24 m",   0.357, 0.367),
    ("MDS @64 m",   0.365, 0.362),
    ("MDS @16 m",   0.317, 0.351),
    ("Planar dev",  0.305, 0.342),
]
ROWS.sort(key=lambda r: r[2])

fig, ax = plt.subplots(figsize=(9.6, 5.6))
names = [r[0] for r in ROWS]
a = [r[1] for r in ROWS]
c = [r[2] for r in ROWS]
y = range(len(ROWS))

for i, (nm, av, cv) in enumerate(ROWS):
    keep = cv >= 0.70
    ax.barh(i, cv, height=0.62, color=(TEAL if keep else SILVER),
            edgecolor="none", zorder=3)
    ax.plot([av], [i], marker="|", ms=13, mew=2, color=NAVY, zorder=4)
    ax.text(cv + 0.012, i, f"{cv:.3f}", va="center", fontsize=9.5,
            color=(NAVY if keep else "#5c6c80"), fontweight=("bold" if keep else "normal"),
            zorder=5)
    ax.text(0.018, i, "KEEP" if keep else "cut", va="center", fontsize=8.5,
            color=("white" if keep else "#41506380"), fontweight="bold", zorder=5)

ax.axvline(0.5, color=NAVY, ls=":", lw=1.4, zorder=2)
ax.axvline(0.7, color=RED, ls="--", lw=1.6, zorder=2)
ax.text(0.5, len(ROWS) - 0.3, "  0.50 coin flip", color=NAVY, fontsize=9, va="center")
ax.text(0.7, len(ROWS) - 0.3, "  0.70 keep bar", color=RED, fontsize=9, va="center")

ax.set_yticks(list(y)); ax.set_yticklabels(names, fontsize=10.5)
ax.set_xlim(0, 0.95); ax.set_xlabel(
    "cross-site AUC: how often the channel ranks rough massif ground above smooth mare ground")
ax.set_title("Which roughness measurements survive a change of site\n"
             "bars = fully matched test  |  navy tick = same test before resolution matching",
             fontsize=11.5, fontweight="bold", color=NAVY, loc="left")
ax.grid(axis="x", alpha=0.25, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "channel_verdict.png", dpi=150, bbox_inches="tight", facecolor="white")
print(f"  -> {OUT/'channel_verdict.png'}")
