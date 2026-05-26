"""Tier-1 DEM-derived feature channels for the HATI v2.0 heatmap.

All channels operate on a 2D DEM array and a uniform pixel scale (m/px).
Each function returns a 2D array of the same shape, ready to be stacked
and fused into the heatmap.

The implementations favour vectorised scipy.ndimage operations over
per-window Python loops. A 5000 x 5000 tile completes all seven channels
in a few minutes on a modern laptop CPU.

References (linked to each channel by its docstring):
  Kreslavsky & Head 2000, JGR 105:26695
  Rosenburg et al. 2011, JGR 116:E02001
  Kreslavsky et al. 2013, Icarus 226:52
  Cai & Fa 2020, JGR Planets, doi:10.1029/2020JE006429
  Lemelin et al. 2020, JGR Planets, doi:10.1029/2019JE006105
  Wang et al. 2024, Remote Sensing 16:3632
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


# ---------------------------------------------------------------------------
# Slope and curvature primitives
# ---------------------------------------------------------------------------


def _slope_magnitude(dem: np.ndarray, scale_m: float) -> np.ndarray:
    """Slope magnitude in radians.

    np.gradient returns gradients along the two image axes in pixel units.
    We convert to physical units by dividing by the pixel scale, take the
    magnitude, and return arctangent so the result is in radians.
    """
    gy, gx = np.gradient(dem.astype(np.float32))
    gx /= scale_m
    gy /= scale_m
    return np.arctan(np.hypot(gx, gy))


def _laplacian(dem: np.ndarray, scale_m: float) -> np.ndarray:
    """Profile curvature proxy (Laplacian of elevation), in 1/m."""
    return ndi.laplace(dem.astype(np.float32)) / (scale_m * scale_m)


# ---------------------------------------------------------------------------
# Tier-1 channels
# ---------------------------------------------------------------------------


def rms_slope(dem: np.ndarray, scale_m: float, window_px: int = 11) -> np.ndarray:
    """Root-mean-square slope in a sliding window.

    Rosenburg 2011 / Cai & Fa 2020: first-derivative roughness. Steeper
    terrain produces higher RMS slope. A field of unresolved sub-pixel
    features inflates RMS slope even when no individual feature is
    detectable.
    """
    s = _slope_magnitude(dem, scale_m)
    mean_sq = ndi.uniform_filter(s * s, size=window_px)
    return np.sqrt(np.maximum(mean_sq, 0.0))


def iqr_slope(dem: np.ndarray, scale_m: float, window_px: int = 11) -> np.ndarray:
    """Interquartile range of slope.

    Kreslavsky 2013 / Wang 2024. Robust spread of slope orientations.
    Boulder fields and crater fields produce wider slope distributions
    than smooth maria.
    """
    s = _slope_magnitude(dem, scale_m)
    q75 = ndi.percentile_filter(s, 75, size=window_px)
    q25 = ndi.percentile_filter(s, 25, size=window_px)
    return q75 - q25


def iqr_curvature(dem: np.ndarray, scale_m: float, window_px: int = 11) -> np.ndarray:
    """Interquartile range of profile curvature.

    Kreslavsky 2013. A crater bowl has a characteristic curvature signature
    (rim positive, floor negative); the IQR of curvature captures this even
    when individual craters are below detection.
    """
    lap = _laplacian(dem, scale_m)
    q75 = ndi.percentile_filter(lap, 75, size=window_px)
    q25 = ndi.percentile_filter(lap, 25, size=window_px)
    return q75 - q25


def rms_planar_deviation(dem: np.ndarray, window_px: int = 11) -> np.ndarray:
    """RMS deviation from a local mean (planar fit proxy).

    Wang 2024 uses a full local planar fit; here we approximate by
    subtracting the local mean and taking RMS of the residuals. For nearly-
    flat regional terrain this is close to a full plane fit. For tilted
    regional terrain a true plane fit is required and is a v2.1 task.
    """
    dem_f = dem.astype(np.float32)
    local_mean = ndi.uniform_filter(dem_f, size=window_px)
    residual = dem_f - local_mean
    mean_sq = ndi.uniform_filter(residual * residual, size=window_px)
    return np.sqrt(np.maximum(mean_sq, 0.0))


def median_differential_slope(
    dem: np.ndarray,
    scale_m: float,
    L_px: int,
    window_px: int = 11,
) -> np.ndarray:
    """Median Differential Slope at characteristic scale L_px.

    Kreslavsky & Head 2000 / Rosenburg 2011. Slope is computed at scale L
    (Gaussian-smoothed gradient with sigma ~ L/3) and at scale 2L. The
    difference of magnitudes is taken in a local median filter to retain
    only the curvature contribution from features near L_px in size.
    """
    sig_L = max(0.5, L_px / 3.0)
    sig_2L = max(0.5, (2 * L_px) / 3.0)
    smoothed_L = ndi.gaussian_filter(dem.astype(np.float32), sig_L)
    smoothed_2L = ndi.gaussian_filter(dem.astype(np.float32), sig_2L)
    slope_L = _slope_magnitude(smoothed_L, scale_m)
    slope_2L = _slope_magnitude(smoothed_2L, scale_m)
    diff = np.abs(slope_L - slope_2L)
    return ndi.median_filter(diff, size=window_px)


def topographic_position_index(
    dem: np.ndarray, window_px: int = 11
) -> np.ndarray:
    """TPI: signed difference between each pixel and its local mean.

    Standard GIS hazard metric. Positive => ridge / peak. Negative =>
    valley / crater floor. The magnitude is what we care about for hazard
    ranking, so callers may want to take absolute value before fusion.
    """
    dem_f = dem.astype(np.float32)
    local_mean = ndi.uniform_filter(dem_f, size=window_px)
    return dem_f - local_mean


def terrain_ruggedness_index(
    dem: np.ndarray, window_px: int = 3
) -> np.ndarray:
    """TRI: mean absolute elevation change between each pixel and its
    immediate neighbours.

    Standard GIS hazard metric (Riley et al. 1999 in the geomorphology
    literature; standardised in Wang 2024 for lunar applications). A
    3 x 3 window is the canonical TRI; larger windows can be used for
    coarser ruggedness.
    """
    dem_f = dem.astype(np.float32)
    abs_diffs = np.zeros_like(dem_f)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = ndi.shift(dem_f, (dy, dx), order=0, mode="nearest")
            abs_diffs += np.abs(dem_f - shifted)
    return abs_diffs / 8.0


# ---------------------------------------------------------------------------
# Convenience: compute the full Tier-1 stack
# ---------------------------------------------------------------------------


def compute_tier1_stack(
    dem: np.ndarray,
    scale_m: float = 1.5,
    window_px: int = 11,
    mds_scales_px: tuple[int, ...] = (3, 5, 10, 20),
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Compute all Tier-1 channels on a single DEM array.

    Returns a dict mapping channel name to 2D array. The names match the
    keys used by `heatmap.fusion.fuse`.
    """
    out: dict[str, np.ndarray] = {}

    def step(name: str, fn) -> None:
        if verbose:
            print(f"   ... {name}")
        out[name] = fn()

    step("rms_slope", lambda: rms_slope(dem, scale_m, window_px))
    step("iqr_slope", lambda: iqr_slope(dem, scale_m, window_px))
    step("iqr_curvature", lambda: iqr_curvature(dem, scale_m, window_px))
    step(
        "rms_planar_dev",
        lambda: rms_planar_deviation(dem, window_px),
    )
    step("tpi_abs", lambda: np.abs(topographic_position_index(dem, window_px)))
    step("tri", lambda: terrain_ruggedness_index(dem, window_px=3))

    for L in mds_scales_px:
        step(
            f"mds_L{L}",
            lambda L=L: median_differential_slope(dem, scale_m, L, window_px),
        )

    return out
