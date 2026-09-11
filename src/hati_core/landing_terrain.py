"""Scale-explicit local surface measurements and conservative landing indices.

Indices are deterministic, configured rankings, never probabilities of loss.
Measurements are made on the native DEM grid. Upsampling does not add support.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class LandingConfig:
    label: str = 'illustrative_limits_not_Athena_specifications'
    baselines_m: tuple = (16., 32., 64.)
    footprint_diameter_m: float = 8.
    navigation_margin_m: float = 5.
    slope_limit_deg: float = 8.
    rms_limit_m: float = .5
    relief_limit_m: float = 1.
    shadow_score_scale: float = 8.
    horizon_distance_m: float = 400.
    dem_vertical_sigma_m: float = .5
    horizon_sigma_multiplier: float = 2.

    def __post_init__(self):
        positive = (*self.baselines_m, self.footprint_diameter_m, self.slope_limit_deg,
                    self.rms_limit_m, self.relief_limit_m, self.shadow_score_scale,
                    self.horizon_distance_m)
        if not self.baselines_m or any(not np.isfinite(v) or v <= 0 for v in positive):
            raise ValueError('baselines, limits and distances must be finite and positive')
        if self.slope_limit_deg >= 90:
            raise ValueError('slope limit must be below 90 degrees')
        if any(not np.isfinite(v) or v < 0 for v in (
                self.navigation_margin_m,self.dem_vertical_sigma_m,self.horizon_sigma_multiplier)):
            raise ValueError('uncertainties and margins must be finite and nonnegative')

    def hash(self):
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()[:16]


def disk(radius_px):
    reach = int(np.ceil(radius_px))
    yy,xx = np.mgrid[-reach:reach+1,-reach:reach+1]
    return xx*xx+yy*yy <= radius_px*radius_px+1e-10


def plane_metrics(dem, pixel_m, diameter_m, *, valid=None, quantile_block=64):
    """Fit z = a*r + b*c + intercept in a circular physical window.

    Every sample must be observed; neither border padding nor nodata filling is
    accepted as terrain. Returns vertical height residuals (metres), slope and
    signed derivatives (m/m). Percentiles do not replace the separate extrema.
    Constant support allows exact convolutional least squares. Subtract a global
    datum before computing moments to avoid cancellation on absolute heights.
    """
    z = np.asarray(dem,float)
    if z.ndim != 2 or not np.isfinite(pixel_m) or pixel_m <= 0:
        raise ValueError('need a 2D DEM and positive posting')
    if not np.isfinite(diameter_m) or diameter_m < 4*pixel_m:
        raise ValueError('plane diameter must span at least four native DEM postings')
    observed = np.isfinite(z)
    if valid is not None:
        if np.shape(valid) != z.shape:
            raise ValueError('valid mask shape differs from DEM')
        observed &= np.asarray(valid,bool)
    kernel = disk(diameter_m/(2*pixel_m))
    radius = kernel.shape[0]//2
    yy,xx = np.mgrid[-radius:radius+1,-radius:radius+1]*pixel_m
    count = int(kernel.sum())
    support = ndi.correlate(observed.astype(float),kernel.astype(float),mode='constant',cval=0) >= count-.01
    fill = np.where(observed,z-(np.median(z[observed]) if observed.any() else 0.),0.)
    mean = ndi.correlate(fill,kernel.astype(float)/count,mode='constant')
    moment = np.sum((yy*kernel)**2)
    a = ndi.correlate(fill,yy*kernel,mode='constant')/moment
    b = ndi.correlate(fill,xx*kernel,mode='constant')/moment
    second = ndi.correlate(fill*fill,kernel.astype(float)/count,mode='constant')
    rms = np.sqrt(np.maximum(0.,second-mean*mean-(a*a+b*b)*moment/count))
    hi,lo = np.full(z.shape,-np.inf),np.full(z.shape,np.inf)
    padded = np.pad(fill,radius)
    offsets = [(int(r-radius),int(c-radius)) for r,c in np.argwhere(kernel)]
    h,w = z.shape
    for dr,dc in offsets:
        residual = padded[radius+dr:radius+dr+h,radius+dc:radius+dc+w]-mean-a*dr*pixel_m-b*dc*pixel_m
        hi = np.maximum(hi,residual)
        lo = np.minimum(lo,residual)
    p95 = np.full(z.shape,np.nan)
    for r in range(0,h,quantile_block):
        for c in range(0,w,quantile_block):
            rh,cw = min(quantile_block,h-r),min(quantile_block,w-c)
            sl = np.s_[r:r+rh,c:c+cw]
            if not support[sl].any():
                continue
            residuals = np.stack([
                padded[radius+r+dr:radius+r+dr+rh,radius+c+dc:radius+c+dc+cw]
                -mean[sl]-a[sl]*dr*pixel_m-b[sl]*dc*pixel_m for dr,dc in offsets])
            p95[sl] = np.percentile(abs(residuals),95,axis=0)
    arrays = dict(slope_deg=np.degrees(np.arctan(np.hypot(a,b))),
                  rms_height_m=rms, positive_relief_m=np.maximum(hi,0),
                  negative_relief_m=np.maximum(-lo,0), relief_p95_m=p95,
                  slope_row=a,slope_col=b)
    return {**{k:np.where(support,v,np.nan) for k,v in arrays.items()},
            'support':support,'diameter_m':float(diameter_m),'sample_count':count}


def terrain_assessment(dem,pixel_m,cfg):
    # Unsupported requested baselines are replaced explicitly, never treated as
    # native measurements at a finer scale than the raster can represent.
    diameters = sorted(set(max(float(v),4*pixel_m) for v in
                           (*cfg.baselines_m,cfg.footprint_diameter_m)))
    measurements = {str(d):plane_metrics(dem,pixel_m,d) for d in diameters}
    primary = max(cfg.footprint_diameter_m,4*pixel_m)
    ch = measurements[str(float(primary))]
    components = np.stack([ch['slope_deg']/cfg.slope_limit_deg,
                           ch['rms_height_m']/cfg.rms_limit_m,
                           ch['positive_relief_m']/cfg.relief_limit_m,
                           ch['negative_relief_m']/cfg.relief_limit_m])
    ratio = np.max(components,axis=0)
    score = ratio/(1+ratio)  # configured limit is 0.5; no probability claim
    return dict(measurements=measurements,primary=ch,score=score,
                exceeds_limits=np.isfinite(ratio)&(ratio>=1),
                footprint_resolved=cfg.footprint_diameter_m >= 4*pixel_m,
                effective_diameter_m=primary)


def buffer_evidence(score,pixel_m,radius_m):
    """Worst score in a physical disk; incomplete low evidence stays unknown.

    Known high evidence can still exclude a position with unknown neighbours.
    Return coverage separately even for these one-sided exclusion conclusions.
    """
    a = np.asarray(score,float)
    if pixel_m <= 0 or radius_m < 0:
        raise ValueError('invalid buffer dimensions')
    kernel = disk(radius_m/pixel_m)
    finite = np.isfinite(a)
    complete = ndi.minimum_filter(finite.astype('uint8'),footprint=kernel,mode='constant',cval=0)>0
    peak = ndi.maximum_filter(np.where(finite,a,-np.inf),footprint=kernel,mode='constant',cval=-np.inf)
    return np.where(complete | (peak>=.5),peak,np.nan),complete


def fuse_landing(terrain,shadow):
    """Monotone maximum of configured indices, without independence assumptions.

    Missing low evidence cannot become clearance. A known high hazard remains
    visible even when the other module is unobservable; status records this.
    Status: 0 unknown; 1 both available; 2 high evidence but incomplete.
    """
    a,b = np.asarray(terrain,float),np.asarray(shadow,float)
    if a.shape != b.shape:
        raise ValueError('maps must share one grid')
    both = np.isfinite(a)&np.isfinite(b)
    peak = np.fmax(a,b)
    high = np.isfinite(peak)&(peak>=.5)
    fused = np.where(both|high,peak,np.nan)
    status = np.where(both,1,np.where(high,2,0)).astype('uint8')
    return fused,status
