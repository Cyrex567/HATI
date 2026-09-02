"""Tier-1 DEM-derived feature channels for the HATI heatmap.

All channels operate on a 2D DEM array and a uniform pixel scale (m/px).
Each function returns a 2D array of the same shape, ready to be stacked
and fused into the heatmap.

TWO MODES
---------
*Legacy (pixel) mode* -- windows and scales given in PIXELS. Reproduces every
number published before 2026-06-15 bit-for-bit. Kept because existing scripts and
results depend on it, but it is **not transferable between DEMs of different
posting**: an 11 px window is 44 m at 4 m/px and 16.5 m at 1.5 m/px, and the
legacy slope uses ``np.gradient`` (a central difference), which fixes the slope
baseline at 2 px no matter what ``scale_m`` says.

*Physical mode* -- windows and baselines given in METRES (``window_m``,
``mds_baselines_m``, ``slope_baseline_m``). Pixel windows are derived per-DEM, and
slope is measured at an explicit physical baseline by Gaussian pre-smoothing. This
is the mode to use for any cross-site comparison, because two DEMs of different
posting then measure the *same physical quantity*. Channel names carry the baseline
in metres (``mds_B16m``) so a weight dictionary is portable between sites.

Physical mode refuses a baseline finer than ``MIN_BASELINE_FACTOR`` x the pixel
scale rather than silently returning noise (Shannon 1949: a sampled grid cannot
represent detail below its own limit).

References (linked to each channel by its docstring):
  Kreslavsky & Head 2000, JGR 105:26695
  Rosenburg et al. 2011, JGR 116:E02001
  Kreslavsky et al. 2013, Icarus 226:52
  Cai & Fa 2020, JGR Planets, doi:10.1029/2020JE006429
  Lemelin et al. 2020, JGR Planets, doi:10.1029/2019JE006105
  Wang et al. 2024, Remote Sensing 16:3632
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage as ndi

#: A requested physical baseline must be at least this many pixels wide.
MIN_BASELINE_FACTOR = 2.0

#: Gaussian FWHM -> sigma. A baseline of B metres is realised as a Gaussian of
#: FWHM = B, i.e. sigma = B / 2.355, applied before a 2-px central difference.
_FWHM_TO_SIGMA = 1.0 / 2.3548200450309493


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def px_for(length_m: float, scale_m: float, *, odd: bool = True, minimum: int = 3) -> int:
    """Pixel window realising a physical length, clamped to a sane minimum."""
    n = int(round(length_m / float(scale_m)))
    if odd and n % 2 == 0:
        n += 1
    return max(n, minimum)


def _check_baseline(baseline_m: float, scale_m: float, what: str, strict: bool) -> float:
    """Refuse (or warn about) a baseline finer than the grid can represent."""
    floor = MIN_BASELINE_FACTOR * scale_m
    if baseline_m >= floor:
        return baseline_m
    msg = (f"{what} baseline {baseline_m:g} m is below the {floor:g} m floor for a "
           f"{scale_m:g} m/px grid ({MIN_BASELINE_FACTOR:g}x pixel scale); "
           f"the result would be sampling noise, not terrain.")
    if strict:
        raise ValueError(msg)
    warnings.warn(msg + " Clamping to the floor.", RuntimeWarning, stacklevel=3)
    return floor


# ---------------------------------------------------------------------------
# Slope and curvature primitives
# ---------------------------------------------------------------------------


def _slope_magnitude(dem: np.ndarray, scale_m: float) -> np.ndarray:
    """LEGACY slope magnitude in radians, at an implicit 2-pixel baseline.

    ``np.gradient`` returns central differences in pixel units; dividing by the
    pixel scale converts to physical units. The baseline is therefore always 2 px
    -- resolution-dependent, and the reason legacy channels do not transfer
    between DEMs of different posting. Retained for reproducibility.
    """
    gy, gx = np.gradient(dem.astype(np.float32))
    gx /= scale_m
    gy /= scale_m
    return np.arctan(np.hypot(gx, gy))


def slope_at_baseline(dem: np.ndarray, scale_m: float, baseline_m: float,
                      *, strict: bool = True) -> np.ndarray:
    """Slope magnitude (radians) measured at an explicit physical baseline.

    The DEM is Gaussian-smoothed to FWHM = ``baseline_m`` before differencing, so
    the returned slope describes the terrain at that scale rather than at the
    grid's Nyquist limit. This is the transferable slope: the same
    ``baseline_m`` on two different DEMs measures the same physical quantity.
    """
    baseline_m = _check_baseline(baseline_m, scale_m, "slope", strict)
    sigma_px = (baseline_m / scale_m) * _FWHM_TO_SIGMA
    z = ndi.gaussian_filter(dem.astype(np.float32), sigma_px) if sigma_px > 0.3 \
        else dem.astype(np.float32)
    gy, gx = np.gradient(z)
    return np.arctan(np.hypot(gx / scale_m, gy / scale_m))


def _laplacian(dem: np.ndarray, scale_m: float, smooth_m: float = 0.0) -> np.ndarray:
    """Profile curvature proxy (Laplacian of elevation), in 1/m.

    A discrete Laplacian amplifies high-frequency content; on a stereo DEM that
    content is largely matching noise. ``smooth_m`` > 0 first smooths to an
    explicit scale so the channel measures curvature of *terrain* rather than of
    noise. ``smooth_m = 0`` reproduces the legacy behaviour.
    """
    z = dem.astype(np.float32)
    if smooth_m > 0:
        z = ndi.gaussian_filter(z, (smooth_m / scale_m) * _FWHM_TO_SIGMA)
    return ndi.laplace(z) / (scale_m * scale_m)


# ---------------------------------------------------------------------------
# Tier-1 channels
# ---------------------------------------------------------------------------


def rms_slope(dem: np.ndarray, scale_m: float, window_px: int = 11,
              slope_baseline_m: float | None = None, *, strict: bool = True) -> np.ndarray:
    """Root-mean-square slope in a sliding window.

    Rosenburg 2011 / Cai & Fa 2020: first-derivative roughness. Steeper terrain
    produces higher RMS slope. A field of unresolved sub-pixel features inflates
    RMS slope even when no individual feature is detectable.
    """
    s = (_slope_magnitude(dem, scale_m) if slope_baseline_m is None
         else slope_at_baseline(dem, scale_m, slope_baseline_m, strict=strict))
    return np.sqrt(np.maximum(ndi.uniform_filter(s * s, size=window_px), 0.0))


def iqr_slope(dem: np.ndarray, scale_m: float, window_px: int = 11,
              slope_baseline_m: float | None = None, *, strict: bool = True) -> np.ndarray:
    """Interquartile range of slope.

    Kreslavsky 2013 / Wang 2024. Robust spread of slope orientations. Boulder
    fields and crater fields produce wider slope distributions than smooth maria.
    """
    s = (_slope_magnitude(dem, scale_m) if slope_baseline_m is None
         else slope_at_baseline(dem, scale_m, slope_baseline_m, strict=strict))
    return (ndi.percentile_filter(s, 75, size=window_px)
            - ndi.percentile_filter(s, 25, size=window_px))


def iqr_curvature(dem: np.ndarray, scale_m: float, window_px: int = 11,
                  smooth_m: float = 0.0) -> np.ndarray:
    """Interquartile range of profile curvature.

    Kreslavsky 2013. A crater bowl has a characteristic curvature signature (rim
    positive, floor negative); the IQR of curvature captures this even when
    individual craters are below detection. Pass ``smooth_m`` to suppress the
    noise amplification inherent to the discrete Laplacian.
    """
    lap = _laplacian(dem, scale_m, smooth_m)
    return (ndi.percentile_filter(lap, 75, size=window_px)
            - ndi.percentile_filter(lap, 25, size=window_px))


def rms_planar_deviation(dem: np.ndarray, window_px: int = 11) -> np.ndarray:
    """RMS deviation from the local mean.

    Wang 2024 uses a full local planar fit. Subtracting the local mean is
    equivalent for the *linear* term: over a symmetric window the mean of a plane
    equals its centre value, so a perfectly planar surface returns exactly zero
    regardless of how steeply it is tilted. The channel is therefore
    tilt-invariant and measures curvature and above, which is what we want.
    """
    dem_f = dem.astype(np.float32)
    residual = dem_f - ndi.uniform_filter(dem_f, size=window_px)
    return np.sqrt(np.maximum(ndi.uniform_filter(residual * residual, size=window_px), 0.0))


def median_differential_slope(dem: np.ndarray, scale_m: float, L_px: int,
                              window_px: int = 11) -> np.ndarray:
    """Median Differential Slope at characteristic scale ``L_px``.

    Kreslavsky & Head 2000 / Rosenburg 2011. Slope is computed at scale L
    (Gaussian-smoothed gradient with sigma ~ L/3) and at 2L; the difference of
    magnitudes retains the curvature contribution from features near L in size.
    """
    smoothed_L = ndi.gaussian_filter(dem.astype(np.float32), max(0.5, L_px / 3.0))
    smoothed_2L = ndi.gaussian_filter(dem.astype(np.float32), max(0.5, (2 * L_px) / 3.0))
    diff = np.abs(_slope_magnitude(smoothed_L, scale_m)
                  - _slope_magnitude(smoothed_2L, scale_m))
    return ndi.median_filter(diff, size=window_px)


def topographic_position_index(dem: np.ndarray, window_px: int = 11) -> np.ndarray:
    """TPI: signed difference between each pixel and its local mean.

    Standard GIS hazard metric. Positive => ridge / peak, negative => valley or
    crater floor. Tilt-invariant for the same reason as ``rms_planar_deviation``.
    Callers generally take the absolute value before fusion.
    """
    dem_f = dem.astype(np.float32)
    return dem_f - ndi.uniform_filter(dem_f, size=window_px)


def terrain_ruggedness_index(dem: np.ndarray, window_px: int = 3) -> np.ndarray:
    """TRI: mean absolute elevation change to the 8 immediate neighbours.

    Standard GIS hazard metric (Riley et al. 1999; standardised for lunar use in
    Wang 2024). Implemented by edge-padding and slicing rather than eight
    ``ndi.shift`` calls -- same result, a fraction of the cost.
    """
    dem_f = dem.astype(np.float32)
    p = np.pad(dem_f, 1, mode="edge")
    acc = np.zeros_like(dem_f)
    h, w = dem_f.shape
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy == 1 and dx == 1:
                continue
            acc += np.abs(dem_f - p[dy:dy + h, dx:dx + w])
    return acc / 8.0


# ---------------------------------------------------------------------------
# Convenience: compute the full Tier-1 stack
# ---------------------------------------------------------------------------


def compute_tier1_stack(
    dem: np.ndarray,
    scale_m: float = 1.5,
    window_px: int = 11,
    mds_scales_px: tuple[int, ...] = (3, 5, 10, 20),
    *,
    window_m: float | None = None,
    mds_baselines_m: tuple[float, ...] | None = None,
    slope_baseline_m: float | None = None,
    curvature_smooth_m: float = 0.0,
    strict: bool = True,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Compute all Tier-1 channels on a single DEM array.

    Legacy (pixel) mode -- the default -- uses ``window_px`` / ``mds_scales_px``
    and emits ``mds_L{px}`` names, reproducing pre-2026-06-15 results exactly.

    Physical mode activates when ``window_m`` or ``mds_baselines_m`` is given:
    pixel windows are derived from ``scale_m``, slope is measured at
    ``slope_baseline_m`` (defaulting to twice the pixel scale), and MDS channels
    are named ``mds_B{metres}m`` so weights are portable across sites.

    Returns a dict mapping channel name to a 2D array. Names match the keys used
    by ``heatmap.fusion.fuse``.
    """
    physical = window_m is not None or mds_baselines_m is not None
    out: dict[str, np.ndarray] = {}

    def step(name: str, fn) -> None:
        if verbose:
            print(f"   ... {name}")
        out[name] = fn()

    if physical:
        if window_m is None:
            window_m = window_px * scale_m
        win = px_for(_check_baseline(window_m, scale_m, "window", strict), scale_m)
        if slope_baseline_m is None:
            slope_baseline_m = MIN_BASELINE_FACTOR * scale_m
        slope_baseline_m = _check_baseline(slope_baseline_m, scale_m, "slope", strict)
        if verbose:
            print(f"   [physical] window {window_m:g} m -> {win} px | "
                  f"slope baseline {slope_baseline_m:g} m | scale {scale_m:g} m/px")
        step("rms_slope", lambda: rms_slope(dem, scale_m, win, slope_baseline_m, strict=strict))
        step("iqr_slope", lambda: iqr_slope(dem, scale_m, win, slope_baseline_m, strict=strict))
        step("iqr_curvature", lambda: iqr_curvature(dem, scale_m, win, curvature_smooth_m))
        step("rms_planar_dev", lambda: rms_planar_deviation(dem, win))
        step("tpi_abs", lambda: np.abs(topographic_position_index(dem, win)))
        step("tri", lambda: terrain_ruggedness_index(dem, window_px=3))
        for B in (mds_baselines_m or ()):
            Bc = _check_baseline(B, scale_m, "MDS", strict)
            L = max(1, int(round(Bc / scale_m)))
            step(f"mds_B{B:g}m",
                 lambda L=L: median_differential_slope(dem, scale_m, L, win))
        return out

    step("rms_slope", lambda: rms_slope(dem, scale_m, window_px))
    step("iqr_slope", lambda: iqr_slope(dem, scale_m, window_px))
    step("iqr_curvature", lambda: iqr_curvature(dem, scale_m, window_px))
    step("rms_planar_dev", lambda: rms_planar_deviation(dem, window_px))
    step("tpi_abs", lambda: np.abs(topographic_position_index(dem, window_px)))
    step("tri", lambda: terrain_ruggedness_index(dem, window_px=3))
    for L in mds_scales_px:
        step(f"mds_L{L}", lambda L=L: median_differential_slope(dem, scale_m, L, window_px))
    return out
