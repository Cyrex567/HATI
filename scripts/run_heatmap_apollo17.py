"""Run the HATI v2.0 Tier-1 heatmap on the Apollo 17 DEM.

Usage:
    python scripts/run_heatmap_apollo17.py            # full DEM (slow)
    python scripts/run_heatmap_apollo17.py --subset   # 2000x2000 centre crop

Outputs land in output/heatmap/:
    apollo17_heatmap.tiff                aligned to the input DEM
    apollo17_channels.npz                all individual channels + z-scores
    apollo17_heatmap_overview.png        composite visualisation
    apollo17_heatmap_stats.txt           plain-text summary statistics
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.heatmap import dem_features, fusion  # noqa: E402

DEM_PATH = ROOT / "data" / "APOLLO17_DTM_150CM.tiff"
OUT_DIR = ROOT / "output" / "heatmap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PIXEL_SCALE_M = 1.5  # LRO SLDEM2015 GLD100 at this site


def hillshade(dem: np.ndarray, az_deg: float = 315.0, alt_deg: float = 45.0) -> np.ndarray:
    """Classic Burrough hillshade for visualisation only."""
    az = np.radians(az_deg)
    alt = np.radians(alt_deg)
    gy, gx = np.gradient(dem.astype(np.float32))
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip((shaded + 1) / 2, 0, 1)


def load_dem(subset: bool) -> tuple[np.ndarray, dict]:
    with rasterio.open(DEM_PATH) as src:
        full_w, full_h = src.width, src.height
        if subset:
            crop = 2000
            x0 = (full_w - crop) // 2
            y0 = (full_h - crop) // 2
            window = Window(x0, y0, crop, crop)
            dem = src.read(1, window=window).astype(np.float32)
            meta = {
                "shape": dem.shape,
                "x0": x0,
                "y0": y0,
                "width": crop,
                "height": crop,
                "scale_m": PIXEL_SCALE_M,
            }
        else:
            dem = src.read(1).astype(np.float32)
            meta = {
                "shape": dem.shape,
                "x0": 0,
                "y0": 0,
                "width": full_w,
                "height": full_h,
                "scale_m": PIXEL_SCALE_M,
            }
    return dem, meta


def render_money_shot(
    dem: np.ndarray,
    hs: np.ndarray,
    result,
    out_path: Path,
    region_label: str,
) -> None:
    """Single-panel overlay of the heatmap on the hillshade. This is the
    figure to put in outreach emails and on the website."""
    h, w = hs.shape
    aspect = w / h
    fig_w = 12
    fig_h = fig_w / aspect
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1)
    im = ax.imshow(
        result.heatmap, cmap="magma", alpha=0.6, vmin=0, vmax=1, interpolation="nearest"
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Hazard probability H(x)", fontsize=11)
    ax.set_title(
        f"HATI v2.0 sub-resolution hazard heatmap  ({region_label})",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        0.01,
        0.01,
        "Tier-1 channels: MDS (L=3,5,10,20 px), RMS slope, IQR slope, "
        "IQR curvature, RMS planar dev., |TPI|, TRI\n"
        "Fusion: literature-anchored weights, robust z-score "
        "with log1p + clip, normalised, logistic sigmoid",
        transform=ax.transAxes,
        fontsize=8,
        color="white",
        bbox=dict(facecolor="black", alpha=0.65, pad=4, edgecolor="none"),
        verticalalignment="bottom",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_overview(dem: np.ndarray, hs: np.ndarray, result, channels: dict, out_path: Path) -> None:
    """Multi-panel composite for technical review.

    Layout:
        Row 1: DEM, hillshade, heatmap, pre-sigmoid
        Row 2+: per-channel z-score panels (4 per row)
    The overlay version lives in a separate "money shot" figure produced by
    render_money_shot.
    """
    chan_names = list(channels.keys())
    n_channels = len(chan_names)
    cols = 4
    chan_rows = (n_channels + cols - 1) // cols
    n_rows = 1 + chan_rows

    fig = plt.figure(figsize=(cols * 4, n_rows * 4))
    gs = fig.add_gridspec(n_rows, cols)

    # --- Top row: DEM, hillshade, heatmap, pre-sigmoid
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(dem, cmap="viridis")
    ax.set_title("DEM (elevation, m)", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Hillshade (az=315, alt=45)", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(result.heatmap, cmap="magma", vmin=0, vmax=1)
    ax.set_title("Heatmap H(x) ∈ [0,1]", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 3])
    im = ax.imshow(result.pre_sigmoid, cmap="RdBu_r")
    ax.set_title("Pre-sigmoid (weighted z sum + bias)", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Channel rows
    for i, name in enumerate(chan_names):
        r = 1 + i // cols
        c = i % cols
        ax = fig.add_subplot(gs[r, c])
        z = result.channel_zscores.get(name)
        if z is None:
            ax.imshow(channels[name], cmap="cividis")
            ax.set_title(f"{name} (raw)", fontsize=9)
        else:
            ax.imshow(z, cmap="RdBu_r", vmin=-3, vmax=3)
            ax.set_title(f"{name}  w={result.weights.get(name, 0):.2f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "HATI v2.0 - Tier-1 heatmap, Apollo 17 site (LRO SLDEM2015 GLD100)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_stats(result, channels: dict, meta: dict, out_path: Path, elapsed: float) -> None:
    h = result.heatmap
    flat = h.flatten()
    pcts = np.percentile(flat, [1, 5, 25, 50, 75, 95, 99])
    lines = [
        "HATI v2.0 Tier-1 heatmap statistics",
        "=" * 60,
        f"DEM region:      x0={meta['x0']}, y0={meta['y0']}, "
        f"shape={meta['shape'][0]} x {meta['shape'][1]} px, "
        f"scale={meta['scale_m']} m/px",
        f"Elapsed:         {elapsed:.1f} s",
        f"Heatmap range:   [{h.min():.4f}, {h.max():.4f}]",
        f"Heatmap mean:    {h.mean():.4f}",
        f"Heatmap std:     {h.std():.4f}",
        f"Percentiles:     p01={pcts[0]:.3f}, p05={pcts[1]:.3f}, "
        f"p25={pcts[2]:.3f}, p50={pcts[3]:.3f}, "
        f"p75={pcts[4]:.3f}, p95={pcts[5]:.3f}, p99={pcts[6]:.3f}",
        f"High-hazard:     {(h > 0.5).mean()*100:.2f}% of pixels above 0.5",
        f"                 {(h > 0.8).mean()*100:.2f}% of pixels above 0.8",
        "",
        "Channel mean |w * z| contributions (relative importance):",
    ]
    sorted_contribs = sorted(result.contributions.items(), key=lambda kv: -kv[1])
    for name, c in sorted_contribs:
        w = result.weights.get(name, 0.0)
        lines.append(f"  {name:<18}  w={w:.2f}  mean|wz|={c:.3f}")
    lines.append("")
    lines.append("Channel raw ranges:")
    for name in sorted(channels):
        arr = channels[name]
        lines.append(f"  {name:<18}  min={arr.min():.4e}  max={arr.max():.4e}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        action="store_true",
        help="Run on a 2000x2000 centre crop (fast). Default: full DEM.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=11,
        help="Sliding window size for Tier-1 channels.",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Skip channel computation; load channels from the cached "
        ".npz so fusion tuning can iterate fast.",
    )
    args = parser.parse_args()

    print(f"Loading DEM ({'subset' if args.subset else 'full'})...")
    dem, meta = load_dem(args.subset)
    print(f"   shape = {dem.shape},  range = [{dem.min():.1f}, {dem.max():.1f}] m")

    suffix = "_subset" if args.subset else ""
    cache_path = OUT_DIR / f"apollo17_heatmap_channels{suffix}.npz"

    t0 = time.perf_counter()
    if args.from_cache and cache_path.exists():
        print(f"Loading channels from cache: {cache_path.name}")
        cached = np.load(cache_path)
        # The cached npz stores both the channels and the fused outputs
        # (heatmap, pre_sigmoid). Filter to just channel arrays.
        skip = {"heatmap", "pre_sigmoid"}
        channels = {
            name: cached[name].astype(np.float32)
            for name in cached.files
            if name not in skip
        }
        print(f"   loaded {len(channels)} channels  "
              f"({time.perf_counter() - t0:.1f} s)")
    else:
        print("Computing Tier-1 channels...")
        channels = dem_features.compute_tier1_stack(
            dem,
            scale_m=PIXEL_SCALE_M,
            window_px=args.window,
            verbose=True,
        )
        print(f"   ({time.perf_counter() - t0:.1f} s)")

    print("Fusing channels...")
    t1 = time.perf_counter()
    result = fusion.fuse(channels)
    print(f"   ({time.perf_counter() - t1:.1f} s)")

    print("Rendering visualisation...")
    hs = hillshade(dem)
    suffix_local = "_subset" if args.subset else ""

    money_path = OUT_DIR / f"apollo17_heatmap_main{suffix_local}.png"
    region_label = (
        f"Apollo 17 site  ({meta['shape'][1]} x {meta['shape'][0]} px, "
        f"{meta['scale_m']:.1f} m/px)"
    )
    render_money_shot(dem, hs, result, money_path, region_label)
    print(f"   -> {money_path}")

    overview_path = OUT_DIR / f"apollo17_heatmap_overview{suffix_local}.png"
    render_overview(dem, hs, result, channels, overview_path)
    print(f"   -> {overview_path}")

    stats_path = OUT_DIR / (
        "apollo17_heatmap_stats"
        + ("_subset" if args.subset else "")
        + ".txt"
    )
    elapsed = time.perf_counter() - t0
    write_stats(result, channels, meta, stats_path, elapsed)
    print(f"   -> {stats_path}")

    npz_path = OUT_DIR / (
        "apollo17_heatmap_channels"
        + ("_subset" if args.subset else "")
        + ".npz"
    )
    np.savez_compressed(
        npz_path,
        heatmap=result.heatmap,
        pre_sigmoid=result.pre_sigmoid,
        **channels,
    )
    print(f"   -> {npz_path}")

    print("")
    print(f"DONE in {time.perf_counter() - t0:.1f} s.")


if __name__ == "__main__":
    main()
