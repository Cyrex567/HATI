"""Literature-anchored channel fusion for the HATI heatmap.

Each Tier-1 channel is robustly normalised, linearly combined with
literature-anchored weights, and squashed by a logistic sigmoid into [0, 1].

The weights are *not* tuned to maximise any HATI-internal performance metric.
They reflect the literature's documented relevance of each channel to landing
safety. See heatmap_explained.md, Part 3.

Three things to know about the normalisation
--------------------------------------------
1. **It is per-array by default.** The median/IQR come from whatever array is
   handed in, so the score of a fixed pixel depends on what else is in the tile:
   crop the tile differently and the value moves. That makes the default output a
   *within-scene relative ranking*, not an absolute measurement.
2. **Pass ``baseline`` to make it absolute.** Supplying a fixed
   {channel: (median, scale)} dictionary -- measured once on a reference scene --
   makes values comparable across tiles and across sites. ``fit_baseline()``
   builds one.
3. **Pass ``valid`` so nodata cannot bias it.** Feature computation needs a
   gap-filled DEM, but invented values must never enter the statistics that every
   channel is measured against.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Literature-anchored weights
# ---------------------------------------------------------------------------
#
# Sign convention: positive weight means high channel value contributes to
# higher hazard. All Tier-1 channels are constructed so "high" means "rougher",
# so all weights are positive.
#
# Magnitudes are anchored as follows:
#   - rms_slope, iqr_slope: Rosenburg 2011 establishes these as the primary
#     first-derivative roughness signatures. Weight 1.0 (reference).
#   - mds_*: Kreslavsky & Head 2000 / Rosenburg 2011 establish differential
#     slope as a feature-size-resolved curvature signature. The smaller scales
#     sit closest to the Nyquist limit, where unresolved hazards imprint most
#     strongly, so weights drop with scale.
#   - iqr_curvature: Kreslavsky 2013, profile-curvature spread as a crater-bowl
#     signature. Weight 1.0.
#   - rms_planar_dev: Wang 2024 residual-from-planar-fit. Weight 0.7.
#   - tpi_abs, tri: standard GIS hazard metrics, Wang 2024. Weight 0.7 -- below
#     the scale-bearing channels because they are more general-purpose.
DEFAULT_WEIGHTS: dict[str, float] = {
    "rms_slope": 1.0,
    "iqr_slope": 1.0,
    "iqr_curvature": 1.0,
    "rms_planar_dev": 0.7,
    "tpi_abs": 0.7,
    "tri": 0.7,
    "mds_L3": 1.0,
    "mds_L5": 0.9,
    "mds_L10": 0.8,
    "mds_L20": 0.6,
}

DEFAULT_BIAS: float = -1.0

# ---------------------------------------------------------------------------
# v2.5 surviving channel set
# ---------------------------------------------------------------------------
#
# The cross-site control (scripts/mare_control_v3.py, 2026-06-15) scored every
# channel by how well it ranks rough massif terrain above smooth mare terrain,
# with both scenes matched for posting AND effective resolution. Only two cleared
# the 0.70 bar. The rest are kept in the code for reproducibility but excluded
# from the operational model.
#
#   rms_slope    0.826   keep
#   tri          0.808   keep
#   iqr_slope    0.432   cut  (below chance: prefers the SMOOTH site)
#   tpi_abs      0.420   cut
#   mds_*        0.35 to 0.38  cut
#   iqr_curv     0.369   cut
#   planar_dev   0.342   cut
V25_WEIGHTS: dict[str, float] = {"rms_slope": 1.0, "tri": 0.7}

#: Measured cross-site AUC per channel, for provenance and for figures.
V3_CROSS_SITE_AUC: dict[str, float] = {
    "rms_slope": 0.826, "tri": 0.808, "iqr_slope": 0.432, "tpi_abs": 0.420,
    "mds_B40m": 0.375, "iqr_curvature": 0.369, "mds_B24m": 0.367,
    "mds_B64m": 0.362, "mds_B16m": 0.351, "rms_planar_dev": 0.342,
}


def v25_weights(include_iqr_slope: bool = False) -> dict[str, float]:
    """Operational v2.5 weights: the channels that transfer between sites.

    ``include_iqr_slope`` restores slope IQR despite its 0.432 cross-site score.
    It exists so the choice is explicit and visible in the call, rather than a
    silent edit to a dictionary.
    """
    w = dict(V25_WEIGHTS)
    if include_iqr_slope:
        w["iqr_slope"] = 1.0
    return w

#: Weights for physical-mode MDS channels, keyed by baseline in metres.
MDS_WEIGHT_BY_BASELINE: dict[float, float] = {8.0: 1.0, 16.0: 0.9, 32.0: 0.8, 64.0: 0.6}


def build_weights_physical(mds_baselines_m: tuple[float, ...]) -> dict[str, float]:
    """Weight dict for a physical-mode stack -- portable between sites.

    Non-MDS weights are copied unchanged; each ``mds_B{B}m`` channel takes the
    weight of the nearest tabulated baseline, so a weight set defined once
    applies to any DEM regardless of its posting.
    """
    w = {k: v for k, v in DEFAULT_WEIGHTS.items() if not k.startswith("mds_")}
    for B in mds_baselines_m:
        nearest = min(MDS_WEIGHT_BY_BASELINE, key=lambda b: abs(b - B))
        w[f"mds_B{B:g}m"] = MDS_WEIGHT_BY_BASELINE[nearest]
    return w


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def _robust_stats(x: np.ndarray, valid: np.ndarray | None,
                  log_transform: bool) -> tuple[float, float, bool]:
    """Median and IQR-derived scale over valid, finite samples."""
    finite = np.isfinite(x)
    if valid is not None:
        finite &= valid
    if not finite.any():
        return 0.0, 1.0, False
    sample = x[finite]
    logged = bool(log_transform and (sample >= 0).all())
    if logged:
        sample = np.log1p(sample)
    med = float(np.median(sample))
    q25, q75 = np.percentile(sample, [25, 75])
    return med, max(float(q75 - q25), 1e-12) / 1.349, logged


def robust_zscore(x: np.ndarray, log_transform: bool = True, clip: float | None = 4.0,
                  valid: np.ndarray | None = None,
                  stats: tuple[float, float] | None = None) -> np.ndarray:
    """Z-score using median and an IQR-derived scale.

    A ``log1p`` transform compresses the long tail of the non-negative,
    heavily-skewed Tier-1 channels so a few extreme pixels cannot set the scale.
    Clipping bounds how much any single channel can contribute: channels still
    discriminate ordinally, but none can monopolise the sigmoid through one
    outlier.

    ``valid`` restricts the statistics to real data (gap-filled pixels must not
    define the baseline). ``stats`` supplies a fixed (median, scale) instead of
    measuring it here, which is what makes cross-tile comparison possible.
    """
    x_f = x.astype(np.float32)
    if stats is not None:
        med, scale = stats
        if log_transform:
            finite = np.isfinite(x_f)
            if (x_f[finite] >= 0).all() if finite.any() else False:
                x_f = np.log1p(x_f)
    else:
        med, scale, logged = _robust_stats(x_f, valid, log_transform)
        if logged:
            x_f = np.log1p(x_f)
    z = (x_f - med) / scale
    if clip is not None:
        z = np.clip(z, -clip, clip)
    return z.astype(np.float32)


def fit_baseline(channels: dict[str, np.ndarray], valid: np.ndarray | None = None,
                 log_transform: bool = True) -> dict[str, tuple[float, float]]:
    """Measure {channel: (median, scale)} on a reference scene.

    Feed the result to ``fuse(..., baseline=...)`` to score any other tile on the
    same absolute footing -- the difference between a relative ranker and a
    measurement.
    """
    return {name: _robust_stats(arr, valid, log_transform)[:2]
            for name, arr in channels.items()}


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


@dataclass
class FusionResult:
    heatmap: np.ndarray                       # [0, 1] bounded
    pre_sigmoid: np.ndarray                   # the linear combination, for debugging
    channel_zscores: dict[str, np.ndarray]
    weights: dict[str, float]
    bias: float
    #: Signed per-pixel attribution phi_k = w_k * z_k / sum|w|. These sum exactly
    #: to (pre_sigmoid - bias), so each pixel's score decomposes without residual.
    attribution: dict[str, np.ndarray] = field(default_factory=dict)
    #: Mean |attribution| per channel -- a magnitude summary, NOT the attribution.
    contributions: dict[str, float] = field(default_factory=dict)
    valid: np.ndarray | None = None


def fuse(
    channels: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
    bias: float | None = None,
    *,
    valid: np.ndarray | None = None,
    baseline: dict[str, tuple[float, float]] | None = None,
    clip: float | None = 4.0,
    strict: bool = False,
    keep_attribution: bool = True,
) -> FusionResult:
    """Combine Tier-1 channels into a [0, 1] heatmap.

    ``valid`` keeps gap-filled pixels out of the normalisation statistics.
    ``baseline`` (from ``fit_baseline``) fixes the normalisation so the output is
    comparable across tiles and sites. ``strict=True`` raises when a weighted
    channel is missing, instead of silently fusing a smaller model -- worth
    enabling whenever the MDS set differs from the default.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if bias is None:
        bias = DEFAULT_BIAS

    used_names = sorted(set(weights) & set(channels))
    unused_channels = sorted(set(channels) - set(weights))
    missing_weights = sorted(set(weights) - set(channels))
    if unused_channels:
        print(f"  fusion: ignoring channels with no weight: {unused_channels}")
    if missing_weights:
        msg = (f"weighted channels absent from the stack: {missing_weights} -- "
               f"fusing {len(used_names)} of {len(weights)} channels")
        if strict:
            raise KeyError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        print(f"  fusion: {msg}")
    if not used_names:
        raise ValueError("No channels to fuse.")

    total_w = sum(abs(weights[n]) for n in used_names)
    zscores: dict[str, np.ndarray] = {}
    attribution: dict[str, np.ndarray] = {}
    contribs: dict[str, float] = {}
    accum: np.ndarray | None = None
    for name in used_names:
        z = robust_zscore(channels[name], clip=clip, valid=valid,
                          stats=(baseline or {}).get(name))
        zscores[name] = z
        phi = (weights[name] * z) / max(total_w, 1e-12)
        if keep_attribution:
            attribution[name] = phi
        contribs[name] = float(np.nanmean(np.abs(phi)))
        accum = phi if accum is None else accum + phi

    pre_sigmoid = accum + bias
    heatmap = 1.0 / (1.0 + np.exp(-np.clip(pre_sigmoid, -30.0, 30.0)))
    return FusionResult(
        heatmap=heatmap.astype(np.float32),
        pre_sigmoid=pre_sigmoid.astype(np.float32),
        channel_zscores=zscores,
        weights={k: weights[k] for k in used_names},
        bias=bias,
        attribution=attribution,
        contributions=contribs,
        valid=valid,
    )
