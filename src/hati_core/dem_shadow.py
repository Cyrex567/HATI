"""Finite-range lunar DEM horizon prediction, independent of ISIS execution."""
import numpy as np
from scipy import ndimage as ndi


def predict_visibility(dem,pixel_m,azimuth_deg,elevation_deg,max_distance_m,
                       *, vertical_sigma_m=0.,sigma_multiplier=2.,solar_radius_deg=.266,
                       moon_radius_m=1737400.):
    """Trace terrain up-Sun at one-posting intervals with bilinear heights.

    Angles use the raster convention: clockwise from image up. Returns central
    and conservative solar-disc visibility on the MODELLED finite horizon.
    A ray leaving the DEM or crossing nodata cannot establish illumination.
    An observed blocker can establish full shadow despite unknown farther rays.
    Vertical uncertainty is a bounded +/- k*sigma height envelope, not a claimed
    calibrated confidence interval. Terrain beyond max_distance remains outside
    the model. Compute on a halo, then crop to the science area.
    """
    z = np.asarray(dem,float)
    numbers = (pixel_m,azimuth_deg,elevation_deg,max_distance_m,vertical_sigma_m,
               sigma_multiplier,solar_radius_deg,moon_radius_m)
    if z.ndim != 2 or not np.isfinite(numbers).all() or pixel_m <= 0 or max_distance_m < pixel_m:
        raise ValueError('invalid DEM or horizon geometry')
    if not 0 < elevation_deg < 90 or min(vertical_sigma_m,sigma_multiplier,solar_radius_deg)<0 or moon_radius_m<=0:
        raise ValueError('invalid illumination or uncertainty')
    valid = np.isfinite(z)
    fill = np.where(valid,z,0.)
    az = np.radians(azimuth_deg)
    dr,dc = -np.cos(az),np.sin(az)
    horizon = np.full(z.shape,-np.inf)
    upper,lower = horizon.copy(),horizon.copy()
    complete = valid.copy()
    # Sum of endpoint height envelopes is conservative without assuming
    # independent DEM errors. It is not sqrt(2)*sigma unless that independence
    # can actually be established.
    envelope = 2*sigma_multiplier*vertical_sigma_m
    for distance in np.arange(pixel_m,max_distance_m+pixel_m*.001,pixel_m):
        shift = (-dr*distance/pixel_m,-dc*distance/pixel_m)
        neighbour = ndi.shift(fill,shift,order=1,mode='constant',cval=0.,prefilter=False)
        coverage = ndi.shift(valid.astype(float),shift,order=1,mode='constant',cval=0.,prefilter=False)
        keep = valid & (coverage>=1-1e-8)
        complete &= keep
        dz = neighbour-z-distance**2/(2*moon_radius_m)
        for out,delta in ((horizon,0.),(upper,envelope),(lower,-envelope)):
            angle = np.degrees(np.arctan2(dz+delta,distance))
            np.maximum(out,np.where(keep,angle,-np.inf),out=out)
    gr,gc = np.gradient(fill,pixel_m)
    normal_ok = ndi.minimum_filter(valid.astype('uint8'),size=3,mode='constant',cval=0)>0
    self_horizon = np.degrees(np.arctan(gr*dr+gc*dc))
    horizon = np.maximum(horizon,self_horizon)
    # The slope envelope uses the minimum central-difference baseline (2 px).
    slope_unc = envelope/(2*pixel_m)
    upper = np.maximum(upper,np.degrees(np.arctan(gr*dr+gc*dc+slope_unc)))
    lower = np.maximum(lower,np.degrees(np.arctan(gr*dr+gc*dc-slope_unc)))
    complete &= normal_ok
    offsets = np.linspace(-.8,.8,5)
    weights = np.sqrt(1-offsets**2); weights/=weights.sum()
    def visibility(h):
        fraction = sum(w*(elevation_deg+o*solar_radius_deg>h) for o,w in zip(offsets,weights))
        return np.where(valid & normal_ok & (complete | (h>=elevation_deg+solar_radius_deg)),fraction,np.nan)
    incidence = np.maximum(0.,np.sin(np.radians(elevation_deg))
                           -(gr*dr+gc*dc)*np.cos(np.radians(elevation_deg))) / np.sqrt(1+gr*gr+gc*gc)
    return dict(visible=visibility(horizon),visible_conservative=visibility(upper),
                visible_optimistic=visibility(lower),horizon_deg=np.where(complete,horizon,np.nan),
                ray_complete=complete,incidence_factor=np.where(normal_ok,incidence,np.nan),
                max_distance_m=float(max_distance_m))
