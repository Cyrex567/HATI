"""Observable image/DEM discrepancies and local apparent registration offsets.

These diagnostics do not identify craters, calibrate false alarms, or correct
images. All thresholds are explicit research settings, not fitted to touchdown.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class SceneConfig:
    dark_fraction: float = .2
    min_core_diameter_px: float = 4.
    min_lit_frames: int = 3
    min_discrepant_frames: int = 2
    tile_px: int = 96
    max_shift_px: int = 6
    max_azimuth_difference_deg: float = 8.
    max_elevation_difference_deg: float = 1.
    highpass_sigma_px: float = 2.
    min_common_fraction: float = .8

    def __post_init__(self):
        vals = list(asdict(self).values())
        if not np.isfinite(vals).all() or any(v <= 0 for v in vals):
            raise ValueError('scene diagnostic settings must be finite and positive')
        if not 0 < self.dark_fraction < 1 or not 0 < self.min_common_fraction <= 1:
            raise ValueError('invalid fraction')
        if any(not isinstance(v, int) for v in (self.min_lit_frames, self.min_discrepant_frames, self.tile_px, self.max_shift_px)):
            raise ValueError('frame counts, tile size and shift bound must be integers')
        if self.tile_px <= 2*self.max_shift_px+4:
            raise ValueError('tile too small for the registration search')


def broad_dark_discrepancy(stack, visibility, cfg=None):
    """Find broad dark components despite nominal DEM illumination.

    Darkness is relative to each frame's finite positive median. Components
    qualify only if their fully observed interior contains an inscribed core
    of the configured diameter. Report the entire component, not just its core.
    Albedo, missing relief and registration can all produce this discrepancy.
    Unknown pixels and image edges never extend an observed dark component.
    """
    cfg = cfg or SceneConfig()
    a, v = np.asarray(stack, float), np.asarray(visibility, float)
    if a.ndim != 3 or v.shape != a.shape:
        raise ValueError('image and visibility stacks must share a grid')
    lit = np.isfinite(a) & np.isfinite(v) & (v >= .99)
    counts = np.zeros(a.shape[1:], dtype=np.int32)
    scales = []
    usable = lit.copy()
    for i, frame in enumerate(a):
        observed = np.isfinite(frame) & (frame > 0)
        scale = float(np.median(frame[observed])) if observed.any() else np.nan
        scales.append(scale)
        if not np.isfinite(scale) or scale <= 0:
            usable[i] = False
            continue
        dark = np.isfinite(frame) & (frame <= cfg.dark_fraction*scale)
        labels, _ = ndi.label(dark)
        # Padding forces outside-image pixels to be unknown, not dark support.
        distance = ndi.distance_transform_edt(np.pad(dark, 1))[1:-1, 1:-1]
        # Subtract half a pixel: EDT measures centre-to-centre, not edge distance.
        cores = dark & (2*np.maximum(distance-.5, 0) >= cfg.min_core_diameter_px)
        wide = np.isin(labels, np.unique(labels[cores])) if cores.any() else np.zeros_like(dark)
        counts += wide & usable[i]
    lit_count = usable.sum(axis=0)
    assessed = lit_count >= cfg.min_lit_frames
    flag = assessed & (counts >= cfg.min_discrepant_frames)
    fraction = np.divide(counts, lit_count, out=np.full(counts.shape, np.nan), where=lit_count > 0)
    return dict(flag=flag, assessed=assessed, discrepant_frames=counts,
                lit_frames=lit_count, fraction=fraction, normalization_medians=scales)


def local_registration(stack, azimuths, elevations, cfg=None):
    """Fixed-tile integer NCC search for similar-illumination frame pairs.

    Compare A[r,c] with B[r+dr,c+dc]. Returned offsets describe sampling B,
    not a correction to apply. Use one eroded mask for every tested shift.
    Flat or unsupported patches produce no displacement. Peak separation and
    boundary flags accompany offsets; these are not measured error sigmas.
    """
    cfg = cfg or SceneConfig()
    a = np.asarray(stack, float)
    az, el = np.asarray(azimuths, float), np.asarray(elevations, float)
    if a.ndim != 3 or az.shape != (len(a),) or el.shape != az.shape or not np.isfinite([az, el]).all():
        raise ValueError('stack and finite frame geometry required')
    n, h, w = a.shape
    reach = cfg.max_shift_px
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            arc = abs((az[i]-az[j]+180) % 360-180)
            if arc > cfg.max_azimuth_difference_deg or abs(el[i]-el[j]) > cfg.max_elevation_difference_deg:
                continue
            rows = []
            valid = np.isfinite(a[[i,j]])
            filtered = []
            # Exclude the full Gaussian support near holes; filling affects no
            # retained sample. The search mask also covers every B displacement.
            halo = int(np.ceil(4*cfg.highpass_sigma_px))
            for f in range(2):
                z = np.where(valid[f], a[[i,j][f]], 0.)
                filtered.append(z-ndi.gaussian_filter(z, cfg.highpass_sigma_px))
            va = ndi.minimum_filter(valid[0].astype('uint8'), size=2*halo+1, mode='constant', cval=0)>0
            vb = ndi.minimum_filter(valid[1].astype('uint8'), size=2*(halo+reach)+1, mode='constant', cval=0)>0
            for r in range(0, h-cfg.tile_px+1, cfg.tile_px):
                for c in range(0, w-cfg.tile_px+1, cfg.tile_px):
                    yy, xx = np.mgrid[r+reach:r+cfg.tile_px-reach, c+reach:c+cfg.tile_px-reach]
                    keep = (va & vb)[yy, xx]
                    entry = dict(row_px=r, col_px=c, common_fraction=float(keep.mean()), status='insufficient_support')
                    rows.append(entry)
                    if keep.mean() < cfg.min_common_fraction:
                        continue
                    yy, xx = yy[keep], xx[keep]
                    first = filtered[0][yy, xx]; first = first-first.mean()
                    norm = np.linalg.norm(first)
                    if norm <= 1e-10:
                        entry['status'] = 'uninformative'; continue
                    scores = []
                    for dr in range(-reach, reach+1):
                        for dc in range(-reach, reach+1):
                            second = filtered[1][yy+dr, xx+dc]; second = second-second.mean()
                            denom = norm*np.linalg.norm(second)
                            if denom > 1e-10:
                                scores.append((float(first@second/denom), dr, dc))
                    if not scores:
                        entry['status'] = 'uninformative'; continue
                    scores.sort(key=lambda p: (-p[0], p[1]*p[1]+p[2]*p[2], p[1], p[2]))
                    peak, dr, dc = scores[0]
                    competitors = [s for s, rr, cc in scores if abs(rr-dr)>1 or abs(cc-dc)>1]
                    entry.update(status='measured_apparent_offset', row_offset_px=dr, col_offset_px=dc,
                                 ncc=peak, peak_gap=peak-max(competitors) if competitors else None,
                                 search_boundary=abs(dr)==reach or abs(dc)==reach)
            pairs.append(dict(frames=[i,j], azimuth_difference_deg=float(arc), tiles=rows))
    return dict(configuration=asdict(cfg), pairs=pairs,
                meaning='Apparent integer sampling offsets of the second frame; no corrections or covariance updates applied.',
                limitations=['Illumination, relief/parallax and albedo may confound NCC.',
                             'No selected pair means no registration conclusion.',
                             'Offsets at the search boundary or with ambiguous peaks are unresolved.'])
