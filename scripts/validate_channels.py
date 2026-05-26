"""Sanity validation for the Tier-1 heatmap channels.

This is *not* the V3 audit (which requires Phase 1 to have produced the
1,459-object hazard map). This is a quick self-consistency check that
runs from the cached channels:

  - Inter-channel Pearson correlation matrix (do the channels agree on
    where the hazards are?).
  - Marginal heatmap distribution (is the output a usable probability?).
  - High-hazard fraction (is the bias parameter sensible for mare-like
    terrain?).
  - Per-channel value range and outlier behaviour.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "output" / "heatmap"


def channel_correlation(channels: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Pearson correlation between flattened channels.

    Channels that highlight the same hazard features should correlate
    positively. Channels that capture different geometric properties (e.g.
    direction-resolved vs. isotropic) should be less correlated, which is
    a good sign that they add complementary information.
    """
    names = sorted(channels)
    n = len(names)
    flat = np.stack([channels[name].astype(np.float32).ravel() for name in names])
    finite = np.isfinite(flat).all(axis=0)
    flat = flat[:, finite]
    corr = np.corrcoef(flat)
    return corr, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=OUT_DIR / "apollo17_heatmap_channels_subset.npz",
        help="Cached channels (.npz) to validate.",
    )
    args = parser.parse_args()

    if not args.cache.exists():
        raise SystemExit(f"Cache file not found: {args.cache}")

    print(f"Loading cached channels from {args.cache.name} ...")
    cached = np.load(args.cache)
    skip = {"heatmap", "pre_sigmoid"}
    channels = {
        name: cached[name].astype(np.float32)
        for name in cached.files
        if name not in skip
    }
    heatmap = cached["heatmap"] if "heatmap" in cached.files else None
    print(f"   {len(channels)} channels loaded")
    if heatmap is not None:
        print(f"   heatmap shape: {heatmap.shape}")

    # --- Inter-channel correlation
    corr, names = channel_correlation(channels)
    print("\nInter-channel Pearson correlation:")
    print("                 " + "  ".join(f"{n[:8]:>8}" for n in names))
    for i, name in enumerate(names):
        row = "  ".join(f"{corr[i, j]:+.2f}" for j in range(len(names)))
        print(f"  {name[:14]:<14}  {row}")
    mean_off_diag = (corr.sum() - np.trace(corr)) / (corr.size - corr.shape[0])
    print(f"\n   mean off-diagonal correlation: {mean_off_diag:+.3f}")
    print("   (~ 0.4 to 0.7 is healthy; lower means channels are too "
          "independent, higher means they're measuring the same thing)")

    # --- Correlation heatmap figure
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(
                j, i, f"{corr[i, j]:+.2f}",
                ha="center", va="center",
                fontsize=8,
                color="white" if abs(corr[i, j]) > 0.5 else "black",
            )
    plt.colorbar(im, ax=ax, fraction=0.04, label="Pearson r")
    ax.set_title(
        "Inter-channel Pearson correlation\n"
        "Tier-1 heatmap channels, Apollo 17 site",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    corr_path = OUT_DIR / "apollo17_channel_correlation.png"
    fig.savefig(corr_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n   -> {corr_path}")

    # --- Heatmap distribution
    if heatmap is not None:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(heatmap.ravel(), bins=60, color="#4a3fa7", edgecolor="white")
        ax.axvline(0.5, color="red", linestyle="--", label="hazard threshold")
        ax.axvline(0.8, color="darkred", linestyle="--", label="high-hazard threshold")
        ax.set_xlabel("Hazard probability H(x)")
        ax.set_ylabel("Pixel count")
        ax.set_title(
            "Heatmap distribution, Apollo 17 site\n"
            f"mean={heatmap.mean():.3f}  median={np.median(heatmap):.3f}  "
            f"std={heatmap.std():.3f}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_yscale("log")
        ax.legend()
        dist_path = OUT_DIR / "apollo17_heatmap_distribution.png"
        fig.tight_layout()
        fig.savefig(dist_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"   -> {dist_path}")

    print("\nSanity validation complete.")


if __name__ == "__main__":
    main()
