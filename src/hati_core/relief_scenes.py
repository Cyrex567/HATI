"""Seeded relief scenes: small mounds, bowl craters and elephant-hide ripples, with rocks.

An independent forward model for campaign tests T12-T16. It shares no code with
the detector's rectangular shadow templates, its linearised relief hypothesis or
its shape-from-shading null:

- a supersampled height field, each feature scaled to a declared maximum slope;
- Lunar-Lambert photometry from the local surface normal, relative to flat ground;
- cast shadows from a horizon sweep toward the Sun over five solar-disc strips;
- rock bodies (make_rock meshes) whose upward facets are lit or self-shadowed as
  seen from above, and whose shadows fall on the local terrain plane at the root;
- anisotropic optical blur, pixel integration, a per-frame brightness plane and noise.

Photometry is illustrative, not calibrated NAC radiance. Shadowed ground keeps
28% of flat-ground brightness, the shadow contrast the other generators use.
"""
import numpy as np
from scipy import ndimage as ndi

from .rock_scenes import _above_ground, _raster_triangle

RELIEF_KINDS = ('mound', 'bowl', 'ripples')


def relief_feature(kind, size_m, max_slope_deg, *, centre_px=None, seed=0):
    """size_m is a mound or bowl diameter, or a ripple wavelength."""
    if kind not in RELIEF_KINDS:
        raise ValueError(f'relief kind must be one of {RELIEF_KINDS}')
    if not np.isfinite([size_m, max_slope_deg]).all() or size_m <= 0 or not 0 < max_slope_deg < 45:
        raise ValueError('relief size must be positive and the slope within (0, 45) degrees')
    return dict(kind=kind, size_m=float(size_m), max_slope_deg=float(max_slope_deg),
                centre_px=None if centre_px is None else [float(v) for v in centre_px], seed=int(seed))


def boulder_field(abundance, *, exponent=3., d_min_m=.2, d_max_m=2., height_ratio=.5, seed=0):
    """A population of small rocks stamped into the height field as steep domes.

    Diameters follow a truncated power law, cumulative number proportional to
    D**-exponent between d_min_m and d_max_m; rocks are added until they cover
    the fraction `abundance` of the scene. Height is height_ratio times the
    diameter. Positions are uniform and may overlap.
    """
    if not 0 < abundance < .5 or not exponent > 0 or not 0 < d_min_m < d_max_m or not 0 < height_ratio <= 1:
        raise ValueError('abundance in (0, 0.5), positive exponent, 0 < d_min < d_max and height ratio in (0, 1] required')
    return dict(kind='boulders', abundance=float(abundance), exponent=float(exponent), d_min_m=float(d_min_m),
                d_max_m=float(d_max_m), height_ratio=float(height_ratio), seed=int(seed), centre_px=None)


def boulder_layer(feature, rows_m, cols_m, spacing_m):
    """Heights and footprint mask of a boulder field on the supersampled grid."""
    rng = np.random.default_rng(feature['seed'])
    area = (len(rows_m)*spacing_m)*(len(cols_m)*spacing_m)
    lo, hi, b = feature['d_min_m']**-feature['exponent'], feature['d_max_m']**-feature['exponent'], feature['exponent']
    diameters, covered = [], 0.
    while covered < feature['abundance']*area:
        for d in (lo-rng.uniform(size=4096)*(lo-hi))**(-1/b):
            diameters.append(float(d)); covered += np.pi*d*d/4
            if covered >= feature['abundance']*area:
                break
    layer = np.zeros((len(rows_m), len(cols_m)))
    centres = np.column_stack([rng.uniform(rows_m[0], rows_m[-1], len(diameters)), rng.uniform(cols_m[0], cols_m[-1], len(diameters))])
    for (r, c), d in zip(centres, diameters):
        radius = d/2
        i0, i1 = np.searchsorted(rows_m, [r-radius, r+radius]); j0, j1 = np.searchsorted(cols_m, [c-radius, c+radius])
        if i1 <= i0 or j1 <= j0:
            continue
        dr, dc = rows_m[i0:i1, None]-r, cols_m[None, j0:j1]-c
        dome = feature['height_ratio']*d*np.sqrt(np.clip(1-(dr*dr+dc*dc)/(radius*radius), 0, None))
        layer[i0:i1, j0:j1] = np.maximum(layer[i0:i1, j0:j1], dome)
    d = np.asarray(diameters)
    return layer, layer > 0, dict(count=len(d), area_fraction=float(covered/area), exponent=feature['exponent'],
                                  diameter_m=dict(zip(('min', 'median', 'p90', 'max'), np.percentile(d, [0, 50, 90, 100]).tolist())),
                                  taller_than_0_3m=int(np.sum(feature['height_ratio']*d >= .3)))


def _feature_shape(feature, rows_m, cols_m, centre_m):
    """Unit-amplitude shape on the supersampled grid, in metres."""
    kind, size = feature['kind'], feature['size_m']
    dr, dc = rows_m[:, None]-centre_m[0], cols_m[None, :]-centre_m[1]
    r = np.hypot(dr, dc)
    if kind == 'mound':
        s = size/4
        return np.exp(-r**2/(2*s*s))
    if kind == 'bowl':
        radius = size/2
        # A depression with a low raised rim: bright and dark swap sides relative to a mound.
        return -np.exp(-r**2/(2*(radius/2)**2))+.25*np.exp(-(r-radius)**2/(2*(radius/4)**2))
    rng = np.random.default_rng(feature['seed'])
    field = np.zeros((len(rows_m), len(cols_m)))
    for angle, scale, phase in zip(rng.uniform(0, np.pi, 3), rng.uniform(.8, 1.25, 3), rng.uniform(0, 2*np.pi, 3)):
        k = 2*np.pi/(size*scale)
        field += np.cos(k*(dr*np.cos(angle)+dc*np.sin(angle))+phase)
    return field


def height_field(rows_m, cols_m, features, centres_m, spacing_m):
    h = np.zeros((len(rows_m), len(cols_m)))
    for feature, centre in zip(features, centres_m):
        if feature['kind'] == 'boulders':
            continue                                   # stamped separately by boulder_layer
        shape = _feature_shape(feature, rows_m, cols_m, centre)
        gr, gc = np.gradient(shape, spacing_m)
        steepest = float(np.max(np.hypot(gr, gc)))
        if steepest > 0:
            h += shape*np.tan(np.radians(feature['max_slope_deg']))/steepest
    return h


def lit_fraction(h, spacing_m, azimuth_deg, elevation_deg, solar_radius_deg=.266):
    """Visible fraction of the solar disc at each grid point, from cast shadows only.

    Resample onto a lattice aligned with the down-Sun direction; a point is in
    shadow when any sunward point rises above its Sun line, i.e. when the running
    maximum of h + distance * tan(elevation) exceeds its own value. Terrain beyond
    the grid is taken as flat.
    """
    a = np.radians(azimuth_deg)
    down = np.array([np.cos(a), -np.sin(a)])       # down-Sun, (row, col); azimuth clockwise from map up
    across = np.array([np.sin(a), np.cos(a)])
    H, W = h.shape
    corners = np.array([[0, 0], [0, W-1], [H-1, 0], [H-1, W-1]], float)
    u0, v0 = (corners@down).min()-1, (corners@across).min()-1
    nu = int(np.ceil(np.ptp(corners@down)))+3
    nv = int(np.ceil(np.ptp(corners@across)))+3
    ui, vi = u0+np.arange(nu), v0+np.arange(nv)
    rows = ui[:, None]*down[0]+vi[None, :]*across[0]
    cols = ui[:, None]*down[1]+vi[None, :]*across[1]
    rotated = ndi.map_coordinates(h, [rows, cols], order=1, mode='constant', cval=np.nan)
    base = np.where(np.isfinite(rotated), rotated, 0.)
    distance = (ui*spacing_m)[:, None]
    offsets = np.linspace(-.8, .8, 5); weights = np.sqrt(1-offsets**2); weights /= weights.sum()
    lit = np.zeros_like(base)
    for off, weight in zip(offsets, weights):
        g = base+distance*np.tan(np.radians(elevation_deg+off*solar_radius_deg))
        prior = np.maximum.accumulate(np.vstack([np.full((1, nv), -np.inf), g[:-1]]), axis=0)
        lit += weight*(prior <= g+1e-9)
    R, C = np.indices(h.shape, dtype=float)
    return ndi.map_coordinates(lit, [R*down[0]+C*down[1]-u0, R*across[0]+C*across[1]-v0], order=1, mode='nearest')


def _lunar_lambert(mu0, mu, L):
    mu0 = np.clip(mu0, 0, None)
    return 2*L*mu0/(mu0+mu)+(1-L)*mu0


def _sun(azimuth_deg, elevation_deg):
    """Unit vector toward the Sun in (row, col, up)."""
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    return np.array([-np.cos(a)*np.cos(e), np.sin(a)*np.cos(e), np.sin(e)])


def shading(h, spacing_m, azimuth_deg, elevation_deg, L=.5):
    """Lunar-Lambert brightness relative to flat ground, nadir view, no cast shadows."""
    gr, gc = np.gradient(h, spacing_m)
    sun = _sun(azimuth_deg, elevation_deg)
    norm = np.sqrt(1+gr**2+gc**2)
    flat = _lunar_lambert(np.sin(np.radians(elevation_deg)), 1., L)
    return _lunar_lambert((-gr*sun[0]-gc*sun[1]+sun[2])/norm, 1/norm, L)/flat


def _rock_layers(rocks, canvas, origin_pad, pixel_m, ss, h, spacing_m, azimuth_deg, elevation_deg,
                 solar_radius_deg, L, rock_albedo, floor):
    """Rock shadow cover, and brightness of rock facets visible from above."""
    cover = np.zeros(canvas)
    face_value = np.zeros(canvas); face_mask = np.zeros(canvas, bool)
    if not rocks:
        return cover, face_value, face_mask
    gr, gc = np.gradient(h, spacing_m)
    disc = [(0., 0.)]+[(.75*np.cos(t), .75*np.sin(t)) for t in np.linspace(0, 2*np.pi, 6, endpoint=False)]
    sun = _sun(azimuth_deg, elevation_deg)
    flat = _lunar_lambert(np.sin(np.radians(elevation_deg)), 1., L)
    for rock in rocks:
        v = rock['vertices']; origin = (rock['root']+origin_pad)*pixel_m
        idx = np.clip(np.rint(origin/pixel_m*ss+(ss-1)/2).astype(int), 0, np.array(canvas)-1)
        slope_rc = np.array([gr[tuple(idx)], gc[tuple(idx)]])     # receiving plane at the root
        for da, de in disc:
            angle = np.radians(azimuth_deg+da*solar_radius_deg/max(np.cos(np.radians(elevation_deg)), .01))
            direction = np.array([np.cos(angle), -np.sin(angle)])
            denom = np.tan(np.radians(elevation_deg+de*solar_radius_deg))+slope_rc@direction
            if denom <= 0:
                continue                                           # the local plane faces away; already dark
            cast = np.zeros(canvas, bool)
            for face in rock['faces']:
                polygon = _above_ground(v[face])
                if len(polygon) < 3:
                    continue
                p = np.asarray(polygon)
                projected = p[:, :2]+origin+p[:, 2, None]/denom*direction
                for k in range(1, len(projected)-1):
                    _raster_triangle(cast, projected[[0, k, k+1]], pixel_m, ss)
            cover += cast/len(disc)
        centre = v.mean(axis=0)
        for face in rock['faces']:
            tri = v[face]
            normal = np.cross(tri[1]-tri[0], tri[2]-tri[0])
            length = np.linalg.norm(normal)
            if length < 1e-12:
                continue
            normal /= length
            if normal@(tri.mean(axis=0)-centre) < 0:
                normal = -normal                                   # outward for a convex body
            if normal[2] <= 1e-6:
                continue                                           # not visible from above
            polygon = _above_ground(tri)
            if len(polygon) < 3:
                continue
            value = rock_albedo*max(float(_lunar_lambert(normal@sun, normal[2], L)/flat), floor)
            p = np.asarray(polygon)[:, :2]+origin
            mask = np.zeros(canvas, bool)
            for k in range(1, len(p)-1):
                _raster_triangle(mask, p[[0, k, k+1]], pixel_m, ss)
            face_value[mask] = value; face_mask |= mask
    return np.clip(cover, 0, 1), face_value, face_mask


def render_relief(shape, azimuths, elevations, *, pixel_m, seed, noise=.015, features=(), rocks=(),
                  supersample=4, solar_radius_deg=.266, lunar_lambert_l=.5, rock_albedo=1.3,
                  shadow_floor=.28, texture=.04, stain=.2, frame_plane=.0003,
                  registration_sigma_px=0., structured_null=False, pad_px=8, plane_slope_rc=(0., 0.)):
    """Render an (n, rows, cols) stack of relief, rocks, albedo and noise.

    Features without centre_px sit at the scene centre. plane_slope_rc tilts the
    whole receiving surface (dz per metre along rows and columns), which changes
    its brightness and every shadow length. Returns the stack, the true height and
    maximum slope per pixel, and the generator truth.
    """
    if len(plane_slope_rc) != 2 or not np.isfinite(plane_slope_rc).all():
        raise ValueError('plane slopes must be two finite values')
    azimuths = np.asarray(azimuths, float); elevations = np.asarray(elevations, float)
    if len(azimuths) != len(elevations) or len(azimuths) < 1 or not np.isfinite([*azimuths, *elevations]).all():
        raise ValueError('finite geometry and equal illumination lengths required')
    if any(e <= solar_radius_deg or e >= 90 for e in elevations):
        raise ValueError('Sun must be above the horizontal and below zenith')
    if pixel_m <= 0 or noise < 0 or registration_sigma_px < 0 or type(supersample) is not int or supersample < 1:
        raise ValueError('invalid scene scales')
    rng = np.random.default_rng(seed); ss = supersample; pad = pad_px
    padded = (shape[0]+2*pad, shape[1]+2*pad)
    canvas = (padded[0]*ss, padded[1]*ss)
    spacing = pixel_m/ss
    # Supersample centre i lies at (i-(ss-1)/2)*spacing metres; padded pixel P centre at P*pixel_m.
    rows_m = (np.arange(canvas[0])-(ss-1)/2)*spacing
    cols_m = (np.arange(canvas[1])-(ss-1)/2)*spacing
    centres = [((np.asarray(f['centre_px']) if f['centre_px'] is not None else (np.asarray(shape)-1)/2)+pad)*pixel_m
               for f in features]
    h = height_field(rows_m, cols_m, list(features), centres, spacing)
    h = h+plane_slope_rc[0]*(rows_m-rows_m.mean())[:, None]+plane_slope_rc[1]*(cols_m-cols_m.mean())[None, :]
    stones = np.zeros(h.shape, bool); populations = []
    for feature in features:
        if feature['kind'] == 'boulders':
            layer, footprint, info = boulder_layer(feature, rows_m, cols_m, spacing)
            h += layer; stones |= footprint; populations.append(info)
    rows, cols = np.indices(shape)
    albedo = 1+texture*ndi.gaussian_filter(rng.normal(size=shape), 1.2)-stain*np.exp(
        -((rows-shape[0]*.3)**2+(cols-shape[1]*.7)**2)/8)
    albedo_ss = np.kron(np.pad(albedo, pad, mode='edge'), np.ones((ss, ss)))
    albedo_ss = np.where(stones, rock_albedo*albedo_ss, albedo_ss)
    frames, shifts = [], []
    for i, (az, el) in enumerate(zip(azimuths, elevations)):
        relative = shading(h, spacing, az, el, lunar_lambert_l)
        lit = lit_fraction(h, spacing, az, el, solar_radius_deg)
        cover, face_value, face_mask = _rock_layers(rocks, canvas, pad, pixel_m, ss, h, spacing, az, el,
                                                    solar_radius_deg, lunar_lambert_l, rock_albedo, shadow_floor)
        lit = lit*(1-cover)
        radiance = albedo_ss*(lit*np.maximum(relative, shadow_floor)+(1-lit)*shadow_floor)
        radiance = np.where(face_mask, face_value, radiance)
        radiance = ndi.gaussian_filter(radiance, (.52*ss, .68*ss), mode='nearest')
        frame = radiance.reshape(padded[0], ss, padded[1], ss).mean(axis=(1, 3))[pad:pad+shape[0], pad:pad+shape[1]]
        frame = frame+frame_plane*i*(rows-cols)
        if structured_null:
            frame = frame+.12*np.sin((rows+.7*cols)/3+.8*i)
        shift = rng.normal(0, registration_sigma_px, 2)
        if registration_sigma_px:
            frame = ndi.shift(frame, shift, order=1, mode='nearest')
        white = rng.normal(size=shape); colored = ndi.gaussian_filter(rng.normal(size=shape), .8)
        frames.append(frame+noise*(.8*white+.6*colored/max(colored.std(), 1e-12)))
        shifts.append(shift.tolist())
    gr, gc = np.gradient(h, spacing)
    slope = np.degrees(np.arctan(np.hypot(gr, gc)))
    height_px = h.reshape(padded[0], ss, padded[1], ss).mean(axis=(1, 3))[pad:pad+shape[0], pad:pad+shape[1]]
    slope_px = slope.reshape(padded[0], ss, padded[1], ss).max(axis=(1, 3))[pad:pad+shape[0], pad:pad+shape[1]]
    return dict(stack=np.asarray(frames), height_m=height_px, slope_deg=slope_px,
                truth=dict(seed=seed, features=list(features), rocks=[r['truth'] for r in rocks], boulder_fields=populations,
                           shape=list(shape),
                           pixel_m=pixel_m, azimuths=azimuths.tolist(), elevations=elevations.tolist(),
                           noise_sigma=noise, supersample=ss, solar_radius_deg=solar_radius_deg,
                           lunar_lambert_l=lunar_lambert_l, rock_albedo=rock_albedo, shadow_floor=shadow_floor,
                           registration_shifts_px=shifts, structured_null=structured_null,
                           plane_slope_rc=[float(v) for v in plane_slope_rc],
                           renderer='relief_heightfield_horizon_lunar_lambert_v1',
                           limitations='Illustrative photometry; rock shadows fall on the local plane at the root; '
                                       'terrain beyond the padded scene is flat; no multiple scattering.'))
