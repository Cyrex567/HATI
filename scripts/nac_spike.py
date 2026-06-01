"""Stage A spike: read one NAC EDR frame and check that shadow features are
visible. Saves a downsampled visualisation plus a small crop for
inspection.

NAC EDR is uncalibrated raw counts (uint8). For shadow detection we don't
need radiometric calibration -- shadows are simply low-DN regions. This
spike validates the read pipeline; orthorectification comes in Stage B.

Frame: M1412631647RE
  Centre:           20.02 deg N, 30.74 deg E (Apollo 17 site)
  Map resolution:   0.808 m/px
  Incidence angle:  66.92 deg  (-> sun elevation 23.08 deg)
  Observation:      2022-07-16T17:26:33 UTC
  Shape:            52224 lines x 5064 samples (uint8)
  Header offset:    5064 bytes
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "nac"
OUT = ROOT / "output" / "nac"
OUT.mkdir(parents=True, exist_ok=True)

FRAME = "M1412631647RE"
IMG = DATA / f"{FRAME}.IMG"

LINES = 52224
SAMPLES = 5064
HEADER_OFFSET = 5064  # bytes


def main() -> None:
    if not IMG.exists():
        raise SystemExit(f"NAC IMG not found: {IMG}")

    print(f"Reading {IMG.name} ({IMG.stat().st_size / 1e6:.1f} MB)")
    with IMG.open("rb") as f:
        f.seek(HEADER_OFFSET)
        raw = np.frombuffer(f.read(LINES * SAMPLES), dtype=np.uint8)
    arr = raw.reshape(LINES, SAMPLES)
    print(f"   array shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"   value range: [{arr.min()}, {arr.max()}], mean: {arr.mean():.1f}")

    # NAC images are unprojected line-scan images. The "lines" axis is
    # the time-of-flight direction (very long), the "samples" axis is the
    # cross-track direction. To get a reasonable preview, downsample
    # heavily along the long axis.
    downsample_factor = 32
    preview = arr[::downsample_factor, :].copy()
    print(f"   downsampled preview shape: {preview.shape}")

    # --- Save the full-frame downsampled preview
    fig, ax = plt.subplots(figsize=(7, 22))
    ax.imshow(preview, cmap="gray", vmin=np.percentile(preview, 1),
              vmax=np.percentile(preview, 99))
    ax.set_title(
        f"{FRAME} (NAC EDR, downsampled {downsample_factor}x along lines)\n"
        f"Apollo 17 site, 0.808 m/px native, sun elev 23 deg",
        fontsize=10, fontweight="bold",
    )
    ax.set_xlabel("Sample (cross-track, ~0.8 m/px)")
    ax.set_ylabel(f"Line (along-track, every {downsample_factor}-th line)")
    plt.tight_layout()
    preview_path = OUT / f"{FRAME}_full_preview.png"
    fig.savefig(preview_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> {preview_path}")

    # --- Save a native-resolution centre crop, ~2000x2000 px
    # The image centre is the Apollo 17 area at full NAC resolution.
    cl, cs = LINES // 2, SAMPLES // 2
    half = 1000
    crop = arr[cl - half:cl + half, cs - half:min(cs + half, SAMPLES)]
    print(f"   centre crop: {crop.shape}")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(crop, cmap="gray", vmin=np.percentile(crop, 1),
              vmax=np.percentile(crop, 99))
    ax.set_title(
        f"{FRAME} centre crop  (native 0.808 m/px)\n"
        "Shadows visible as dark crater interiors; sun illuminates from above-left",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Sample (~0.8 m/px)")
    ax.set_ylabel("Line (~0.8 m/px)")
    plt.tight_layout()
    crop_path = OUT / f"{FRAME}_center_crop.png"
    fig.savefig(crop_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> {crop_path}")

    # --- Histogram of pixel values
    fig, ax = plt.subplots(figsize=(9, 4))
    sample = arr[::8, ::4]  # subsample for speed
    ax.hist(sample.ravel(), bins=256, color="#3a4a8c", edgecolor="none")
    ax.axvline(np.percentile(sample, 5), color="darkred", linestyle="--",
               label=f"p5 = {np.percentile(sample, 5):.0f}  (potential shadow threshold)")
    ax.set_xlabel("Raw DN (8-bit)")
    ax.set_ylabel("Pixel count")
    ax.set_title(f"{FRAME} pixel-value histogram", fontsize=11)
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    hist_path = OUT / f"{FRAME}_histogram.png"
    fig.savefig(hist_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> {hist_path}")

    print("\nStage A spike complete.")
    print("\nNext: Stage B will project this NAC frame into the DEM frame so")
    print("we can stack multiple frames at the same coordinates and compute")
    print("shadow statistics per pixel.")


if __name__ == "__main__":
    main()
