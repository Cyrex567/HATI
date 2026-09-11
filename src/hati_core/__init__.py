"""hati_core: the portable detection engine.

One import surface for everything HATI actually decides with, separated from the
analysis scripts around it. This is the piece that ports to another domain (see
the Poseidon maritime work): the scripts, the papers and the dashboard are
site-specific, the core is not.

The engine is deliberately small, because the cross-site control removed most of
what used to be in it:

    channels  physical-baseline terrain descriptors (metres, degrees)
    gate      a deterministic lander-limit decision, in physical units
    fuse      fixed literature-anchored weights, signed attribution, no fitting
    freeze    a config hash, so a result can be tied to the exact settings

Typical use::

    from hati_core import Config, channels, gate, fuse

    cfg = Config(scale_m=8.0, window_m=56.0, slope_baseline_m=16.0)
    ch  = channels(dem, cfg, valid=mask)
    g   = gate(ch, theta_max_deg=8.0)
    h   = fuse(ch, valid=mask)
    print(cfg.hash())          # tie the numbers to the settings that made them
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field

import numpy as np
from scipy import ndimage as ndi

from src.heatmap import dem_features as _df
from src.heatmap import fusion as _fusion

__all__ = ["Config", "channels", "gate", "fuse", "fit_baseline",
           "SURVIVING_CHANNELS", "CROSS_SITE_AUC", "__version__"]

__version__ = "2.5.3"

#: The channels that survived the cross-site control (mare_control_v3, 2026-06-15).
#: Everything else scored below chance between sites and was removed.
SURVIVING_CHANNELS: tuple[str, ...] = ("rms_slope", "tri")

#: Measured cross-site AUC, kept next to the code that uses it.
CROSS_SITE_AUC = dict(_fusion.V3_CROSS_SITE_AUC)


@dataclass(frozen=True)
class Config:
    """Every setting that changes a number, in physical units.

    ``hash()`` gives a short digest of the whole configuration. Quote it beside
    any published result and the result becomes reproducible without trusting a
    changelog. Freeze this before a held-out validation run.
    """
    scale_m: float                       # DEM posting, metres per pixel
    window_m: float = 56.0               # analysis window, metres
    slope_baseline_m: float = 16.0       # baseline the slope is measured over
    curvature_smooth_m: float = 0.0      # 0 disables curvature pre-smoothing
    theta_max_deg: float = 8.0           # lander slope limit
    clip: float | None = 4.0             # z-score clip
    bias: float = _fusion.DEFAULT_BIAS
    include_iqr_slope: bool = False      # explicit opt-in for a rejected channel
    mds_baselines_m: tuple[float, ...] = field(default_factory=tuple)
    tri_baseline_m: float | None = None # explicit opt-in; refit baseline before comparing scores

    def weights(self) -> dict[str, float]:
        return _fusion.v25_weights(include_iqr_slope=self.include_iqr_slope)

    def hash(self) -> str:
        payload = json.dumps({**asdict(self), "version": __version__,
                              "weights": self.weights()}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def channels(dem: np.ndarray, cfg: Config, valid: np.ndarray | None = None,
             all_channels: bool = False) -> dict[str, np.ndarray]:
    """Terrain descriptors at explicit physical baselines.

    Returns only the surviving channels unless ``all_channels`` is set, which is
    for diagnostics and for re-running the cross-site control, not for operations.
    """
    observed = np.isfinite(dem)
    if valid is not None:
        observed &= np.asarray(valid, bool)
    if not observed.any():
        raise ValueError("DEM contains no valid terrain")
    filled = np.asarray(dem)
    if not observed.all():
        idx = ndi.distance_transform_edt(~observed, return_distances=False, return_indices=True)
        filled = filled[tuple(idx)]
    stack = _df.compute_tier1_stack(
        filled, scale_m=cfg.scale_m, window_m=cfg.window_m,
        mds_baselines_m=cfg.mds_baselines_m or None,
        slope_baseline_m=cfg.slope_baseline_m,
        curvature_smooth_m=cfg.curvature_smooth_m, strict=True, verbose=False)
    if cfg.tri_baseline_m is not None:
        stack["tri"] = _df.tri_at_baseline(filled, cfg.scale_m, cfg.tri_baseline_m)
    # Mask the support of every requested filter, not just the missing centre.
    win = _df.px_for(cfg.window_m, cfg.scale_m) // 2
    reach = max(1, win + int(np.ceil(4 * cfg.slope_baseline_m / cfg.scale_m / 2.3548)) + 1)
    if cfg.tri_baseline_m is not None:
        reach = max(reach, int(np.ceil(cfg.tri_baseline_m / (2*cfg.scale_m)))+1)
    if all_channels:
        reach = max(reach, 2 * win,
                    win + int(np.ceil(4 * cfg.curvature_smooth_m / cfg.scale_m / 2.3548)) + 1,
                    win + int(np.ceil(8 * max(cfg.mds_baselines_m or (0,)) / cfg.scale_m / 3)) + 1)
    unsupported = ndi.maximum_filter((~observed).astype(np.uint8), size=2 * reach + 1) > 0
    stack = {k: np.where(unsupported, np.nan, v) for k, v in stack.items()}
    if all_channels:
        return stack
    keep = set(cfg.weights())
    return {k: v for k, v in stack.items() if k in keep}


def gate(dem_or_channels, cfg: Config | None = None, *, scale_m: float | None = None,
         theta_max_deg: float | None = None) -> np.ndarray:
    """Deterministic lander gate: True where footprint slope exceeds the limit.

    Accepts a DEM (with ``cfg`` or ``scale_m``) or a computed channel dict. The
    decision is a comparison of two physical quantities in degrees, which is why
    it transfers between sites and why a reviewer can replay it by hand.
    """
    theta = theta_max_deg if theta_max_deg is not None else (cfg.theta_max_deg if cfg else 8.0)
    if isinstance(dem_or_channels, dict):
        slope_rad = dem_or_channels["rms_slope"]
    else:
        s = scale_m if scale_m is not None else (cfg.scale_m if cfg else None)
        if s is None:
            raise ValueError("gate() on a DEM needs cfg or scale_m")
        effective = cfg or Config(scale_m=s, slope_baseline_m=2.0 * s)
        slope_rad = channels(dem_or_channels, effective)["rms_slope"]
    # True means excluded from clearance; unknown terrain must not appear safe.
    return ~np.isfinite(slope_rad) | (np.degrees(slope_rad) > theta)


def fuse(ch: dict[str, np.ndarray], cfg: Config | None = None, *,
         valid: np.ndarray | None = None,
         baseline: dict[str, tuple[float, float]] | None = None):
    """Fixed-weight fusion with signed per-channel attribution."""
    cfg = cfg or Config(scale_m=8.0)
    return _fusion.fuse(ch, cfg.weights(), cfg.bias, valid=valid,
                        baseline=baseline, clip=cfg.clip, strict=True)


def fit_baseline(ch: dict[str, np.ndarray], valid: np.ndarray | None = None):
    """Measure a fixed normalisation on a reference scene, for cross-site scoring."""
    return _fusion.fit_baseline(ch, valid=valid)
