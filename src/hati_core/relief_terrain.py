"""Relief correction for the production maps.

Shape from shading on the image window supplies metre-scale relief that the 4 m
DEM cannot resolve. This module keeps the two sources apart where each is
reliable: the merged surface is the DEM below the DEM's own resolution plus the
SfS detail finer than it (a complementary Gaussian split). On that surface it
traces self and cast shadows with the same horizon tracer the DEM uses, and it
marks where a slope facing away from the Sun is steeper than the Sun: there SfS
sees only shadow, so its slope is a lower bound.
"""
import numpy as np
from scipy import ndimage as ndi

from .dem_shadow import predict_visibility
from .sfs import solve_sfs, solve_sfs_nonlinear

MODELS = ('none', 'linear', 'nonlinear')


def solve_relief(stack, valid, azimuths, elevations, pixel_m, model, options=None, progress=None):
    """Shape from shading with the selected model; returns the solution and the lit mask.

    lit marks the pixels the model corrected. Both models leave pixels outside
    their common support unchanged, and the nonlinear model also those it predicts
    in terrain shadow; they keep their relief shading, so they must not enter a
    rock search on the corrected images.
    """
    options = dict(options or {})
    if model == 'linear':
        solved = solve_sfs(stack, valid, azimuths, elevations, pixel_m, progress=progress, **options)
        lit = np.broadcast_to(solved['common'], np.shape(stack)).copy()
    elif model == 'nonlinear':
        solved = solve_sfs_nonlinear(stack, valid, azimuths, elevations, pixel_m, **options)
        lit = np.asarray(solved['lit'], bool)
    else:
        raise ValueError(f'relief model must be one of {MODELS[1:]}')
    return solved, lit


def _lowpass(a, sigma_px):
    """Gaussian mean over finite samples only; NaN where no sample contributes."""
    finite = np.isfinite(a)
    weight = ndi.gaussian_filter(finite.astype(float), sigma_px, mode='nearest')
    total = ndi.gaussian_filter(np.where(finite, a, 0.), sigma_px, mode='nearest')
    return np.where(weight > 1e-3, total/np.maximum(weight, 1e-12), np.nan)


def merged_surface(broad_m, detail_m, broad_pixel_m, pixel_m):
    """DEM broad shape plus SfS detail finer than the DEM, on the image grid.

    broad_m: DEM heights resampled (bilinearly) to the image grid; NaN = unknown.
    detail_m: SfS relative heights on the same grid; NaN outside its support,
    where the merged surface is the DEM alone. The split scale is the DEM
    posting in image pixels, so each source contributes where it resolves.
    """
    broad_m, detail_m = np.asarray(broad_m, float), np.asarray(detail_m, float)
    if broad_m.shape != detail_m.shape or broad_pixel_m <= 0 or pixel_m <= 0:
        raise ValueError('surfaces must share a grid and postings must be positive')
    sigma = max(broad_pixel_m/pixel_m, 1.)
    fine = detail_m-_lowpass(detail_m, sigma)
    fine = np.where(np.isfinite(fine), fine, 0.)
    return _lowpass(broad_m, sigma)+fine, fine


def relief_visibility(surface_m, pixel_m, azimuths, elevations, max_distance_m, *, solar_radius_deg=.266):
    """Visible solar-disc fraction per frame on the merged surface (NaN = not established).

    Uses the DEM horizon tracer unchanged. Metre-scale relief casts shadows over
    tens of metres at grazing Sun, so a short horizon on a grid padded by at
    least that distance suffices; the DEM's own long-range horizon stays separate.
    """
    return np.asarray([predict_visibility(surface_m, pixel_m, az, el, max_distance_m, vertical_sigma_m=0.,
                                          solar_radius_deg=solar_radius_deg)['visible']
                       for az, el in zip(azimuths, elevations)])


def slope_lower_bound(surface_m, pixel_m, azimuths, elevations):
    """True where, in some frame, the surface rises toward the Sun more steeply than the Sun stands.

    Azimuths are clockwise from image up, as for the DEM tracer. Such a slope is
    in its own shadow, so shading constrains only its minimum steepness.
    """
    gr, gc = np.gradient(np.asarray(surface_m, float), pixel_m)
    known = np.isfinite(gr) & np.isfinite(gc)
    mask = np.zeros(gr.shape, bool)
    for az, el in zip(azimuths, elevations):
        a = np.radians(az)
        rise = gr*(-np.cos(a))+gc*np.sin(a)        # height gain per metre toward the Sun
        mask |= known & (rise > np.tan(np.radians(el)))
    return mask


def combine_visibility(dem_visible, relief_visible, lit=None):
    """Pixel-wise minimum of DEM and relief visibility; unknown in either stays unknown.

    lit, if given, removes pixels the nonlinear model left uncorrected.
    """
    a, b = np.asarray(dem_visible, float), np.asarray(relief_visible, float)
    if a.shape != b.shape:
        raise ValueError('visibility stacks must share a grid')
    combined = np.where(np.isfinite(a) & np.isfinite(b), np.minimum(a, b), np.nan)
    if lit is not None:
        combined = np.where(np.asarray(lit, bool), combined, 0.)
    return combined


def lower_bound_terrain(score, lower_bound, pixel_m, footprint_diameter_m):
    """Terrain score that is unknown wherever a footprint touches a lower-bound slope.

    A lower bound can still establish a hazard: a score at or above the 0.5
    limit stays, since the true slope is at least as steep. Below the limit the
    footprint is unqualified (NaN), never counted as safe.
    """
    from .landing_terrain import disk
    touched = ndi.binary_dilation(np.asarray(lower_bound, bool), structure=disk(footprint_diameter_m/(2*pixel_m)))
    score = np.asarray(score, float)
    return np.where(touched & ~(score >= .5), np.nan, score), touched
