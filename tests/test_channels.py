"""Fast invariant + regression tests for the HATI channel/fusion core.

Tiny synthetic arrays only -- runs in seconds on a laptop, no project data needed.
    python tests/test_channels.py

Covers the audit findings fixed on 2026-06-15:
  A0.1  channels defined in pixels, not metres  -> physical mode + transferability
  A0.2  per-array normalisation is crop-dependent -> fixed baseline
  A0.3  nodata fill contaminates the statistics   -> validity mask
  A0.4  discrete Laplacian amplifies noise        -> curvature smoothing
  A0.7  contributions were a magnitude, not the signed attribution
  A0.7  fuse() silently dropped weighted channels -> strict mode
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.heatmap import dem_features as df, fusion  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def plane(shape=(120, 120), slope_deg=12.0, scale_m=2.0) -> np.ndarray:
    """Perfectly planar surface tilted by slope_deg along +x."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    return xx * scale_m * np.tan(np.radians(slope_deg))


def bumpy(shape=(160, 160), scale_m=2.0, seed=3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = np.zeros(shape, np.float32)
    for _ in range(40):
        r, c = rng.integers(10, shape[0] - 10), rng.integers(10, shape[1] - 10)
        z[r - 2:r + 3, c - 2:c + 3] += rng.uniform(0.5, 3.0)
    return z


# ---------------------------------------------------------------------------
print("\n-- geometry invariants -------------------------------------------------")

SC, SLOPE = 2.0, 12.0
p = plane(scale_m=SC, slope_deg=SLOPE)
i = slice(20, -20)

s_leg = np.degrees(df._slope_magnitude(p, SC))[i, i]
check("legacy slope recovers a known plane", abs(s_leg.mean() - SLOPE) < 0.1,
      f"{s_leg.mean():.3f} deg vs {SLOPE} truth")

s_phys = np.degrees(df.slope_at_baseline(p, SC, baseline_m=16.0))[i, i]
check("physical-baseline slope recovers the same plane", abs(s_phys.mean() - SLOPE) < 0.1,
      f"{s_phys.mean():.3f} deg at a 16 m baseline")

# A0.7: rms_planar_dev is tilt-invariant (the old docstring claimed otherwise)
rpd = df.rms_planar_deviation(p, 11)[i, i]
check("rms_planar_dev is exactly zero on a tilted plane", float(np.abs(rpd).max()) < 1e-3,
      f"max {float(np.abs(rpd).max()):.2e} m on a {SLOPE} deg plane")
tpi = df.topographic_position_index(p, 11)[i, i]
check("TPI is zero on a tilted plane", float(np.abs(tpi).max()) < 1e-3)

# ---------------------------------------------------------------------------
print("\n-- A0.1  transferability: same physical baseline across postings -------")

fine = bumpy((240, 240), scale_m=1.0)
coarse = fine[::2, ::2].copy()          # same terrain, 2 m posting

leg_f = df.rms_slope(fine, 1.0, window_px=11)[30:-30, 30:-30].mean()
leg_c = df.rms_slope(coarse, 2.0, window_px=11)[15:-15, 15:-15].mean()
leg_gap = abs(leg_f - leg_c) / max(leg_f, 1e-9)

phy_f = df.rms_slope(fine, 1.0, df.px_for(22, 1.0), slope_baseline_m=8.0)[30:-30, 30:-30].mean()
phy_c = df.rms_slope(coarse, 2.0, df.px_for(22, 2.0), slope_baseline_m=8.0)[15:-15, 15:-15].mean()
phy_gap = abs(phy_f - phy_c) / max(phy_f, 1e-9)

check("physical mode agrees across postings better than legacy", phy_gap < leg_gap,
      f"legacy mismatch {100*leg_gap:.1f}%  ->  physical {100*phy_gap:.1f}%")

# ---------------------------------------------------------------------------
print("\n-- A0.1  the guard: refuse a baseline the grid cannot represent --------")

try:
    df.slope_at_baseline(fine, 4.0, baseline_m=2.0, strict=True)
    check("strict mode refuses a sub-Nyquist baseline", False, "no error raised")
except ValueError:
    check("strict mode refuses a sub-Nyquist baseline", True)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    df.slope_at_baseline(fine, 4.0, baseline_m=2.0, strict=False)
    check("non-strict mode warns and clamps", len(w) == 1)

# ---------------------------------------------------------------------------
print("\n-- A0.3  nodata must not bias the normalisation ------------------------")

_rng = np.random.default_rng(11)
real = _rng.normal(1.0, 0.30, 6000).astype(np.float32)      # genuine terrain values
x = np.concatenate([real, np.full(4000, 500.0, np.float32)])  # 40% gap-fill sentinel
valid = np.concatenate([np.ones(6000, bool), np.zeros(4000, bool)])
# A robust z-score should give the real data unit spread. Fill in the statistics
# inflates the IQR and crushes that spread toward zero.
sd_bad = float(fusion.robust_zscore(x, clip=None)[:6000].std())
sd_good = float(fusion.robust_zscore(x, clip=None, valid=valid)[:6000].std())
check("validity mask keeps invented fill out of the statistics",
      sd_bad < 0.5 < sd_good,
      f"spread of z on real data {sd_bad:.3f} (contaminated) -> {sd_good:.3f} (masked)")

# ---------------------------------------------------------------------------
print("\n-- A0.2  fixed baseline makes scores crop-independent ------------------")

# Half-smooth, half-rough scene: a crop of the smooth half has very different
# statistics from the whole tile, which is exactly when per-array normalisation
# moves a fixed pixel's score.
_r2 = np.random.default_rng(5)
mixed = np.empty((200, 200), np.float32)
mixed[:100, :] = _r2.normal(0, 0.05, (100, 200))    # quiet terrain
mixed[100:, :] = _r2.normal(0, 1.00, (100, 200))    # rough terrain
whole = df.rms_slope(mixed, 2.0, window_px=7)
crop = whole[:60, :60]                                  # the smooth corner
z_whole_free = fusion.robust_zscore(whole, clip=None)[:60, :60]
z_crop_free = fusion.robust_zscore(crop, clip=None)
drift_free = float(np.abs(z_whole_free - z_crop_free).mean())

base = fusion.fit_baseline({"rms_slope": whole})["rms_slope"]
z_whole_fx = fusion.robust_zscore(whole, clip=None, stats=base)[:60, :60]
z_crop_fx = fusion.robust_zscore(crop, clip=None, stats=base)
drift_fixed = float(np.abs(z_whole_fx - z_crop_fx).mean())

check("fixed baseline removes crop dependence", drift_fixed < 1e-6 < drift_free,
      f"same pixels shift by {drift_free:.3f} z free -> {drift_fixed:.2e} fixed")

# ---------------------------------------------------------------------------
print("\n-- A0.7  attribution is signed and sums to the score -------------------")

dem = bumpy((140, 140), 2.0)
stack = df.compute_tier1_stack(dem, scale_m=2.0, window_px=7,
                               mds_scales_px=(3, 5), verbose=False)
w = {k: v for k, v in fusion.DEFAULT_WEIGHTS.items() if k in stack}
res = fusion.fuse(stack, w)
tot = sum(res.attribution.values())
check("sum of attributions == pre_sigmoid - bias",
      float(np.abs(tot - (res.pre_sigmoid - res.bias)).max()) < 1e-4,
      f"max residual {float(np.abs(tot - (res.pre_sigmoid - res.bias)).max()):.2e}")
check("heatmap stays inside [0, 1]",
      float(res.heatmap.min()) >= 0.0 and float(res.heatmap.max()) <= 1.0)

# ---------------------------------------------------------------------------
print("\n-- A0.7  strict mode catches a silently-shrunken model -----------------")

try:
    fusion.fuse(stack, fusion.DEFAULT_WEIGHTS, strict=True)   # wants mds_L10/L20
    check("strict fuse raises on missing weighted channels", False, "no error raised")
except KeyError:
    check("strict fuse raises on missing weighted channels", True)

# ---------------------------------------------------------------------------
print("\n-- A0.4  curvature smoothing suppresses noise amplification ------------")

rng = np.random.default_rng(0)
noisy = bumpy((160, 160), 2.0) + rng.normal(0, 0.05, (160, 160)).astype(np.float32)
raw = df.iqr_curvature(noisy, 2.0, 7, smooth_m=0.0)[20:-20, 20:-20].mean()
sm = df.iqr_curvature(noisy, 2.0, 7, smooth_m=8.0)[20:-20, 20:-20].mean()
check("smoothed curvature is less noise-dominated", sm < raw,
      f"IQR(curv) {raw:.4f} -> {sm:.4f} 1/m")

# ---------------------------------------------------------------------------
print("\n-- physical stack names are portable across sites ----------------------")

a = df.compute_tier1_stack(fine, 1.0, window_m=22.0, mds_baselines_m=(8.0, 16.0),
                           slope_baseline_m=8.0, verbose=False)
b = df.compute_tier1_stack(coarse, 2.0, window_m=22.0, mds_baselines_m=(8.0, 16.0),
                           slope_baseline_m=8.0, verbose=False)
check("identical channel names on DEMs of different posting", set(a) == set(b),
      f"{sorted(set(a) - set(b)) or 'identical'}")
wp = fusion.build_weights_physical((8.0, 16.0))
check("portable weight set covers the physical stack", set(a) <= set(wp),
      f"uncovered: {sorted(set(a) - set(wp)) or 'none'}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
print("FAILURES:" if FAILS else "ALL CHECKS PASSED")
for f in FAILS:
    print("  -", f)
sys.exit(1 if FAILS else 0)
