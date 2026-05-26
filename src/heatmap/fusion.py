"""Literature-anchored channel fusion for the HATI v2.0 heatmap.

Each Tier-1 channel is z-normalised against its own distribution on the
processed tile (acting as a regional baseline), then linearly combined with
literature-anchored weights, then passed through a logistic sigmoid to bound
the output in [0, 1].

The weights are *not* tuned to maximise any HATI-internal performance metric.
They reflect the literature's documented relevance of each channel to landing
safety. See heatmap_explained.md, Part 3.

A future paper may replace these with learned weights from a held-out subset
of the v1.5 forensic audit, but that work is deferred to keep the v2.0
contribution methodologically defensible from first principles.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Literature-anchored weights
# ---------------------------------------------------------------------------
#
# Sign convention: positive weight means high channel value contributes to
# higher hazard probability. All Tier-1 channels we use are constructed so
# that "high" means "rougher / more hazardous", so all weights are positive.
#
# Magnitudes are anchored as follows:
#   - rms_slope, iqr_slope: Rosenburg 2011 establishes these as the primary
#     first-derivative roughness signatures. Weight 1.0 (reference).
#   - mds_L3 .. mds_L20: Kreslavsky & Head 2000 / Rosenburg 2011 establish
#     differential slope as a feature-size-resolved curvature signature.
#     The smaller scales (L3, L5) sit closest to the Nyquist limit and are
#     where unresolved hazards imprint most strongly. Weights drop with L.
#   - iqr_curvature: Kreslavsky 2013 establishes profile-curvature spread
#     as a crater-bowl signature. Weight 1.0.
#   - rms_planar_dev: Wang 2024 includes residual-from-planar-fit; weight
#     0.7 to acknowledge the planar-fit approximation we currently use.
#   - tpi_abs, tri: standard GIS hazard metrics, Wang 2024. Weights 0.7,
#     0.7 -- a little below the named-scale-bearing channels because they
#     are more general-purpose.
#
# A bias term shifts the sigmoid midpoint. We choose -2.0 so that a tile
# of average roughness (all z-scores near 0) lands well below 0.5.
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
# Normalisation
# ---------------------------------------------------------------------------


def robust_zscore(
    x: np.ndarray,
    log_transform: bool = True,
    clip: float = 4.0,
) -> np.ndarray:
    """Z-score using median and inter-quartile-based scale, with optional
    log1p transform for heavily-skewed non-negative channels and clipping
    to prevent outliers from dominating the fusion.

    All Tier-1 channels we use are non-negative (slope, |TPI|, IQR of slope
    and curvature, RMS deviations, MDS, TRI). Most have a long-tailed
    distribution where flat terrain dominates the lower bulk and rough
    terrain extends a thin tail. A log1p transform compresses that tail
    so a few extreme pixels don't dominate the regional z-score scale.

    The clip at +/- 4 standard deviations bounds the dynamic range each
    channel can contribute. Channels still discriminate ordinally, but no
    single channel can monopolise the sigmoid output through one extreme
    pixel.
    """
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)

    x_f = x.astype(np.float32)
    if log_transform and (x_f[finite] >= 0).all():
        x_f = np.log1p(x_f)

    med = np.median(x_f[finite])
    q25, q75 = np.percentile(x_f[finite], [25, 75])
    iqr = max(q75 - q25, 1e-12)
    scale = iqr / 1.349
    z = (x_f - med) / scale
    if clip is not None:
        z = np.clip(z, -clip, clip)
    return z.astype(np.float32)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


@dataclass
class FusionResult:
    heatmap: np.ndarray              # [0, 1] bounded
    pre_sigmoid: np.ndarray          # the linear combination, for debugging
    channel_zscores: dict[str, np.ndarray]
    weights: dict[str, float]
    bias: float
    contributions: dict[str, float] = field(default_factory=dict)


def fuse(
    channels: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
    bias: float | None = None,
) -> FusionResult:
    """Combine Tier-1 channels into a [0, 1] heatmap.

    Channels not present in `weights` are silently dropped (with a printed
    note). Weights named but not in `channels` are also dropped.
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
        print(f"  fusion: weights with no matching channel: {missing_weights}")

    zscores: dict[str, np.ndarray] = {}
    accum: np.ndarray | None = None
    contribs: dict[str, float] = {}
    total_w = 0.0
    for name in used_names:
        z = robust_zscore(channels[name])
        zscores[name] = z
        w = weights[name]
        total_w += abs(w)
        contribs[name] = float(np.nanmean(np.abs(w * z)))
        if accum is None:
            accum = w * z
        else:
            accum = accum + w * z
    if accum is None:
        raise ValueError("No channels to fuse.")

    # Normalise by total weight magnitude. After this, accum has the same
    # scale as a single z-scored channel (roughly +/- 4 given the clip).
    # This decouples the channel count from the sigmoid saturation point,
    # so adding or removing a channel doesn't silently change the heatmap
    # midpoint.
    accum = accum / max(total_w, 1e-12)
    pre_sigmoid = accum + bias
    # Clip before sigmoid to avoid overflow on extreme negative values
    pre_sigmoid_clipped = np.clip(pre_sigmoid, -30.0, 30.0)
    heatmap = 1.0 / (1.0 + np.exp(-pre_sigmoid_clipped))
    return FusionResult(
        heatmap=heatmap.astype(np.float32),
        pre_sigmoid=pre_sigmoid.astype(np.float32),
        channel_zscores=zscores,
        weights={k: weights[k] for k in used_names},
        bias=bias,
        contributions=contribs,
    )
