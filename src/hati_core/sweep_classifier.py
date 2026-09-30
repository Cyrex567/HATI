"""Sweep morphology classifier (HATI 2.6): boulder, hummock, crater or something else, from the solar sweep.

Each cell is explained by competing generative hypotheses. All of them pass through
the detector's null (a static image and a brightness surface per frame, the same
RegistrationProjector), so none can win by describing albedo or a brightness drift:

    boulder    the detector's cast-shadow rectangle from a fixed root, plus a lit face at
               the root (a blurred point displaced toward the Sun). Bright up-Sun, then a
               long dark streak down-Sun whose length grows as cot(elevation).
    hummock    a paraboloid dome: Lambert shading of its flanks and the shadow it casts
               once they are steeper than the Sun. Bright up-Sun, dark down-Sun, confined
               to about its own footprint until the flanks cast.
    crater     a paraboloid bowl: Lambert shading of its walls and the shadow its up-Sun
               rim casts inside it. Dark up-Sun, bright down-Sun, never longer than the bowl.
    extended   a free slope field: at every pixel its own slope, shaded to first order by
               each frame's Sun (brightness change -cot(e_k) grad(h).s_k, the rank-2 model of
               noise_scale.relief_consistency). It describes any gentle relief, compact or
               not, but spends two parameters per pixel; a compact hypothesis wins only
               when its extra structure (one centre, cast shadows) predicts better.

The dome and bowl have closed-form shadow boundaries (derived in the docstrings of
_bowl_shadow and _dome_shadow) and Lambert photometry. They share no code with the
independent relief generator used to test them, which renders Gaussian mounds and
rimmed Gaussian bowls with Lunar-Lambert photometry and a numerical horizon sweep.

The three compact hypotheses each have two nonnegative amplitudes: one for the
shading or lit face, one for the shadow. The order of bright and dark along the Sun
is what separates protrusions from depressions, and the growth of the dark streak
with cot(elevation) and its fixed pivot is what separates a steep compact caster
from gentle relief.

Hypotheses are compared by withheld-frame prediction (each frame predicted from a
fit to the others), never by a fitted score alone. A class is called only when its
gain beats every rival by a calibrated margin and the same model with the Sun
directions reassigned among frames predicts worse by a calibrated margin. Otherwise
the answer is 'ambiguous', with the two leading hypotheses, or 'no_signal' or
'non_solar_change'. Margins come from calibrate_margins on scenes with known truth
and must be checked on separate scenes (confusion_with_intervals).

Nothing here is a validated lunar classification. The calibration scenes are
synthetic; field error rates need independently labelled terrain.
"""
from dataclasses import asdict, dataclass, replace
import hashlib
import json

import numpy as np
from scipy.stats import beta as beta_distribution

from .noise_scale import sun_loadings
from .shadow_likelihood import RegistrationProjector, shadow_template

CLASSES = ('boulder', 'hummock', 'crater', 'extended')
LABELS = CLASSES+('ambiguous', 'no_signal', 'non_solar_change')
TRUTH_TO_CLASS = {'rock': 'boulder', 'mound': 'hummock', 'bowl': 'crater', 'ripples': 'extended',
                  'none': 'no_signal', 'stripes': 'non_solar_change'}


@dataclass(frozen=True)
class SweepClassifierConfig:
    radius_px: int = 16                 # patch half-size; the patch is (2r+1) square
    support_px: float = 12.             # fitting disc radius
    rock_heights_m: tuple = (.1, .2, .3, .6, 1.2, 2.4)
    rock_widths_m: tuple = (.3, .6, 1.2)
    rock_offsets_px: tuple = (-1.5, -.5, .5, 1.5)
    relief_offsets_px: tuple = (-1., 1.)
    crater_diameters_m: tuple = (1.8, 2.7, 3.6, 5.4, 7.2, 10.8)
    crater_depth_ratios: tuple = (.005, .01, .02, .05, .1, .2)
    hummock_diameters_m: tuple = (1.8, 2.7, 3.6, 5.4, 7.2, 10.8)
    hummock_height_ratios: tuple = (.005, .01, .02, .05, .1, .2)
    lit_face: bool = True
    max_shadow_contrast: float = 1.
    max_shading_gain: float = 4.
    min_frame_fraction: float = .85
    min_common_fraction: float = .8
    supersample: int = 4

    def __post_init__(self):
        if type(self.radius_px) is not int or self.radius_px < 8:
            raise ValueError('radius_px must be an integer of at least 8')
        if not 2 <= self.support_px <= self.radius_px:
            raise ValueError('support must lie between 2 px and the patch radius')
        for grid in (self.rock_heights_m, self.rock_widths_m, self.crater_diameters_m, self.crater_depth_ratios,
                     self.hummock_diameters_m, self.hummock_height_ratios):
            if not grid or any(not np.isfinite(v) or v <= 0 for v in grid):
                raise ValueError('every template grid must be nonempty, finite and positive')
        if not self.rock_offsets_px or not self.relief_offsets_px:
            raise ValueError('offset grids must be nonempty')
        if not 0 < self.min_frame_fraction <= 1 or not 0 < self.min_common_fraction <= 1:
            raise ValueError('coverage fractions must lie in (0, 1]')
        if self.max_shadow_contrast <= 0 or self.max_shading_gain <= 0:
            raise ValueError('amplitude bounds must be positive')
        if type(self.supersample) is not int or self.supersample < 1:
            raise ValueError('supersample must be a positive integer')

    def hash(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- geometry

def _toward_sun(azimuth_deg):
    """Unit (row, col) vector toward the Sun; azimuth clockwise from map up, as sun_loadings."""
    a = np.radians(azimuth_deg)
    return np.array([-np.cos(a), np.sin(a)])


def _bowl_shadow(u, v2, radius_m, depth_m, tan_e):
    """Cast shadow inside a paraboloid bowl z = d(r^2/R^2 - 1), rim at z = 0.

    u is the offset toward the Sun, v2 the squared offset across it. Along the Sun
    line through a point P the surface minus the ray from P is
    (t)(d/R^2 (t + 2u) - tan e) for a distance t toward the Sun, so the wall rises
    above the ray once t > R^2 tan(e)/d - 2u. The ray leaves the bowl at
    t_exit = -u + sqrt(R^2 - v^2); beyond the rim the ground is level and lower.
    P is in shadow when t_exit exceeds that distance:
        sqrt(R^2 - v^2) + u > R^2 tan(e) / d.
    No shadow falls once tan(e) exceeds the steepest wall, 2d/R.
    """
    inside = u*u+v2 < radius_m**2
    return inside & (np.sqrt(np.maximum(radius_m**2-v2, 0.))+u > radius_m**2*tan_e/depth_m)


def _dome_shadow(u, v2, radius_m, height_m, tan_e):
    """Attached and cast shadow of a paraboloid dome z = H(1 - r^2/R^2) on level ground.

    On the dome, a point faces away from the Sun more steeply than the Sun stands
    where u < -R^2 tan(e) / (2H) (the terminator). On the ground outside, the ray
    t tan(e) meets the dome when a*t^2 + b*t + c < 0 has a positive root, with
    a = H/R^2, b = 2Hu/R^2 + tan(e), c = H(r^2/R^2 - 1) > 0: that needs b < 0 and
    b^2 > 4ac. The shadow tip on the Sun line through the centre is at
    u = -R^2 tan(e)/(4H) - H/tan(e).
    """
    r2 = u*u+v2
    inside = r2 < radius_m**2
    a = height_m/radius_m**2
    b = 2*height_m*u/radius_m**2+tan_e
    c = height_m*(r2/radius_m**2-1)
    attached = inside & (u < -radius_m**2*tan_e/(2*height_m))
    cast = ~inside & (b < 0) & (b*b > 4*a*c)
    return attached | cast


def relief_components(shape, centre, diameter_m, relief_ratio, kind, azimuths, elevations, pixel_m,
                      psf_sigma_px, solar_radius_deg=.266, supersample=4):
    """Shading change and shadow cover of a paraboloid dome ('hummock') or bowl ('crater').

    Returns two (frames, rows, cols) arrays on the pixel grid: the Lambert shading
    change relative to level ground on lit parts of the feature, and the fraction
    of each pixel in shadow. Both are integrated over pixels and blurred by the PSF.
    Five equal-area solar-disc strips set the penumbra, as for the rock templates.
    """
    from scipy import ndimage as ndi
    if kind not in ('hummock', 'crater'):
        raise ValueError("kind must be 'hummock' or 'crater'")
    radius = diameter_m/2
    relief = relief_ratio*diameter_m
    ss = supersample
    rows = ((np.arange(shape[0]*ss)+.5)/ss-.5-centre[0])*pixel_m
    cols = ((np.arange(shape[1]*ss)+.5)/ss-.5-centre[1])*pixel_m
    R, C = np.meshgrid(rows, cols, indexing='ij')
    r2 = R*R+C*C
    inside = r2 < radius**2
    k = 2*relief/radius**2                       # |grad z| = k r on the paraboloid
    sign = 1. if kind == 'crater' else -1.       # bowl: grad z = +k P; dome: grad z = -k P
    norm = np.sqrt(1+(k*k)*r2*inside)
    offsets = np.linspace(-.8, .8, 5)
    weights = np.sqrt(1-offsets**2); weights /= weights.sum()
    shading, cover = [], []
    for az, el in zip(azimuths, elevations):
        s = _toward_sun(az)
        u = R*s[0]+C*s[1]
        v2 = np.maximum(r2-u*u, 0.)
        grad_s = sign*k*u*inside                  # height gain per metre toward the Sun
        shade = np.zeros_like(R); shadow = np.zeros_like(R)
        for off, w in zip(offsets, weights):
            e = np.radians(el+off*solar_radius_deg)
            dark = (_bowl_shadow(u, v2, radius, relief, np.tan(e)) if kind == 'crater'
                    else _dome_shadow(u, v2, radius, relief, np.tan(e)))
            cos_i = (np.sin(e)-grad_s*np.cos(e))/norm
            lit_change = np.where(inside & ~dark, np.maximum(cos_i, 0.)/np.sin(e)-1., 0.)
            shade += w*lit_change
            shadow += w*dark
        sigma = psf_sigma_px*ss
        if sigma > 0:
            shade = ndi.gaussian_filter(shade, sigma, mode='constant')
            shadow = ndi.gaussian_filter(shadow, sigma, mode='constant')
        shading.append(shade.reshape(shape[0], ss, shape[1], ss).mean(axis=(1, 3)))
        cover.append(shadow.reshape(shape[0], ss, shape[1], ss).mean(axis=(1, 3)))
    return np.asarray(shading), np.asarray(cover)


def lit_face(shape, root, width_m, azimuths, pixel_m, psf_sigma_px):
    """A rock's sunlit face: a unit point displaced half a width toward the Sun, blurred by the PSF."""
    rows, cols = np.indices(shape, dtype=float)
    sigma = max(psf_sigma_px, .35)
    out = []
    for az in azimuths:
        centre = np.asarray(root, float)+.5*width_m/pixel_m*_toward_sun(az)
        g = np.exp(-((rows-centre[0])**2+(cols-centre[1])**2)/(2*sigma*sigma))
        out.append(g/g.sum())
    return np.asarray(out)


# ---------------------------------------------------------------- banks

def build_banks(shape, azimuths, elevations, sc, cfg, slopes=(0., 0.)):
    """Template banks per compact class: components (templates, 2, frames, rows, cols) and parameters.

    Component 0 is the brightening part (shading change or lit face), scaled by a
    nonnegative gain; component 1 is shadow cover, entering as a darkening with a
    nonnegative contrast. The boulder bank uses the detector's own shadow renderer.
    """
    azimuths = np.asarray(azimuths, float); elevations = np.asarray(elevations, float)
    centre = (np.asarray(shape)-1)/2
    banks = {}
    comps, params = [], []
    for dy in cfg.rock_offsets_px:
        for dx in cfg.rock_offsets_px:
            root = centre+[dy, dx]
            for h in cfg.rock_heights_m:
                for w in cfg.rock_widths_m:
                    cover = shadow_template(shape, root, azimuths, elevations, h, w, sc, slopes)[0]
                    face = lit_face(shape, root, w, azimuths, sc.pixel_m, sc.psf_sigma_px) if cfg.lit_face else np.zeros_like(cover)
                    comps.append((face, cover)); params.append(dict(offset_px=[dy, dx], height_m=h, width_m=w))
    banks['boulder'] = (np.asarray(comps), params)
    for kind, diameters, ratios in (('hummock', cfg.hummock_diameters_m, cfg.hummock_height_ratios),
                                    ('crater', cfg.crater_diameters_m, cfg.crater_depth_ratios)):
        comps, params = [], []
        for dy in cfg.relief_offsets_px:
            for dx in cfg.relief_offsets_px:
                for d in diameters:
                    for ratio in ratios:
                        shade, cover = relief_components(shape, centre+[dy, dx], d, ratio, kind, azimuths, elevations,
                                                         sc.pixel_m, sc.psf_sigma_px, sc.solar_radius_deg, cfg.supersample)
                        comps.append((shade, cover))
                        params.append(dict(offset_px=[dy, dx], diameter_m=d, relief_ratio=ratio, relief_m=ratio*d,
                                           max_slope_deg=float(np.degrees(np.arctan(4*ratio)))))
        banks[kind] = (np.asarray(comps), params)
    return banks


# ---------------------------------------------------------------- fitting

def fit_two(x1, x2, y, upper1, upper2):
    """Box-constrained least squares y ~ g*x1 - c*x2 with 0 <= g <= upper1, 0 <= c <= upper2, per template.

    x1, x2: (templates, n) projected components; y: (n,). The optimum of a convex
    quadratic over a box is its interior minimum when feasible, otherwise it lies
    on an edge, where it is the clipped one-dimensional minimum; every candidate is
    evaluated and the best feasible one kept. Returns gains, contrasts and the
    improvement over the zero model, all per template.
    """
    x2 = -x2
    a11 = np.einsum('ij,ij->i', x1, x1); a22 = np.einsum('ij,ij->i', x2, x2); a12 = np.einsum('ij,ij->i', x1, x2)
    b1 = x1@y; b2 = x2@y
    def objective(g, c):
        return 2*g*b1+2*c*b2-g*g*a11-2*g*c*a12-c*c*a22    # improvement, larger is better
    det = a11*a22-a12*a12
    with np.errstate(divide='ignore', invalid='ignore'):
        g0 = np.where(det > 1e-12*np.maximum(a11*a22, 1e-300), (b1*a22-b2*a12)/det, -1.)
        c0 = np.where(det > 1e-12*np.maximum(a11*a22, 1e-300), (b2*a11-b1*a12)/det, -1.)
        def clip1(num, den, upper):
            return np.clip(np.where(den > 1e-300, num/np.maximum(den, 1e-300), 0.), 0., upper)
        candidates = [(np.zeros_like(b1), np.zeros_like(b1))]
        candidates.append((clip1(b1, a11, upper1), np.zeros_like(b1)))                       # c = 0
        candidates.append((np.zeros_like(b1), clip1(b2, a22, upper2)))                       # g = 0
        candidates.append((np.full_like(b1, upper1), clip1(b2-upper1*a12, a22, upper2)))     # g at bound
        candidates.append((clip1(b1-upper2*a12, a11, upper1), np.full_like(b1, upper2)))     # c at bound
    feasible = (g0 >= 0) & (g0 <= upper1) & (c0 >= 0) & (c0 <= upper2)
    candidates.append((np.where(feasible, g0, 0.), np.where(feasible, c0, 0.)))
    values = np.stack([objective(g, c) for g, c in candidates])
    best = np.argmax(values, axis=0)
    idx = np.arange(len(b1))
    g = np.stack([g for g, _ in candidates])[best, idx]
    c = np.stack([c for _, c in candidates])[best, idx]
    return g, c, np.maximum(values[best, idx], 0.)


def _project_bank(projector, comps):
    """Apply the nuisance projection to both components of every template: (templates, 2, n)."""
    flat = comps.reshape(-1, *comps.shape[2:])
    out = projector.apply(flat)
    return out.reshape(comps.shape[0], 2, -1)


def _best(projector, residual, comps, cfg):
    p = _project_bank(projector, comps)
    g, c, improvement = fit_two(p[:, 0], p[:, 1], residual.ravel(), cfg.max_shading_gain, cfg.max_shadow_contrast)
    j = int(np.argmax(improvement))
    return j, float(g[j]), float(c[j]), float(improvement[j]), improvement


# ---------------------------------------------------------------- one cell

def usable_frames(patch, valid, cfg):
    """Indices of the frames evaluate_cell will use for this patch; render banks for exactly these."""
    common, status, frames = _usable(np.asarray(patch, float), valid, None, cfg)
    return frames if common is not None else np.array([], int)


def _usable(patch, valid, sc, cfg):
    radius = cfg.radius_px
    y, x = np.indices(patch.shape[1:])
    support = np.hypot(y-radius, x-radius) <= cfg.support_px
    usable = np.asarray(valid, bool) & np.isfinite(patch)
    frames = np.flatnonzero(usable[:, support].mean(axis=1) >= cfg.min_frame_fraction)
    if len(frames) < 4:
        return None, 'insufficient_frames', frames
    common = support & usable[frames].all(axis=0)
    if common.sum() < 24 or common.sum()/support.sum() < cfg.min_common_fraction:
        return None, 'insufficient_common_support', frames
    return common, 'ok', frames


def evaluate_cell(patch, valid, azimuths, elevations, sigma, sc, cfg=None, *, slopes=(0., 0.),
                  banks=None, shuffled_banks=None, shifts=None, record_folds=False):
    """Withheld-frame prediction errors of every hypothesis for one cell, with Sun checks.

    patch and valid are (frames, 2r+1, 2r+1) around the cell centre. For each frame
    k, every hypothesis is fitted to the other frames only (static image, nuisance
    projection and amplitudes all from them) and predicts frame k. Errors are mean
    squared per pixel in noise units, averaged over withheld frames. The Sun check
    repeats the same prediction with the measured (azimuth, elevation) pairs
    cyclically reassigned among frames; banks for those geometries can be passed in
    (shuffled_banks, keyed by shift) so a campaign renders them once. With
    record_folds, the template and amplitudes each compact hypothesis chose in each
    fold are returned, so a test can confirm the withheld frame never chose them.
    """
    cfg = cfg or SweepClassifierConfig()
    sc = replace(sc, radius_px=cfg.radius_px, root_support_px=cfg.support_px)
    patch = np.asarray(patch, float)
    if patch.shape[1:] != (2*cfg.radius_px+1,)*2:
        raise ValueError('patch must match the classifier radius')
    common, status, frames = _usable(patch, valid, sc, cfg)
    if common is None:
        return dict(status=status, frames=frames.tolist())
    data = patch[frames]
    az = np.asarray(azimuths, float)[frames]; el = np.asarray(elevations, float)[frames]
    m = len(frames)
    shape = patch.shape[1:]
    if banks is not None and any(b[0].shape[2] != m for b in banks.values()):
        # Banks rendered for other frames would pair each frame with another frame's Sun.
        raise ValueError('banks must be rendered for exactly the frames this cell uses; see usable_frames')
    if banks is None:
        banks = build_banks(shape, az, el, sc, cfg, slopes)
    shifts = shifts if shifts is not None else sorted({1, m//2, m-1})
    if shuffled_banks is None:
        shuffled_banks = {k: build_banks(shape, np.roll(az, k), np.roll(el, k), sc, cfg, slopes) for k in shifts}
    loadings = sun_loadings(az, el)

    def static(sel):
        # Pixels outside the common support may be NaN; they never enter a fit, but must not poison the median.
        reference = np.median(np.where(np.isfinite(data[sel]), data[sel], 0.), axis=0)
        return reference, np.where(common, reference/np.mean(reference[common]), 1.)

    reference, albedo = static(np.arange(m))
    full = RegistrationProjector(common, np.full(m, sigma), reference, sc.registration_sigma_px, spatial_degree=2)
    residual = full.apply(data)
    fits = {}
    for name in ('boulder', 'hummock', 'crater'):
        comps, params = banks[name]
        j, g, c, improvement, surface = _best(full, residual, comps, cfg)
        order = np.argsort(surface)[::-1]
        compatible = [params[i] for i in order if surface[j]-surface[i] <= 4.]
        keys = ('height_m', 'width_m') if name == 'boulder' else ('diameter_m', 'relief_m', 'max_slope_deg')
        # Over every compatible template, not a truncated list: a truncated minimum overstates a lower end.
        # Descriptive ranges (chi-square within 4 of the best), not confidence intervals.
        ranges = {k: [float(min(q[k] for q in compatible)), float(max(q[k] for q in compatible))] for k in keys}
        fits[name] = dict(parameters=params[j], gain=g, contrast=c, score=float(np.sqrt(improvement)),
                          compatible=compatible[:50], compatible_count=len(compatible), compatible_ranges=ranges)
    centred = loadings-loadings.mean(axis=0)
    slope_field = np.linalg.pinv(centred)@residual        # (2, pixels): a slope per pixel, up to albedo
    left = residual-centred@slope_field
    fits['extended'] = dict(improvement=float(np.sum(residual**2)-np.sum(left**2)),
                            indicative_slope_deg=float(np.degrees(np.arctan(np.median(np.hypot(*slope_field))*sigma))))

    names = ('null',)+CLASSES
    errors = {n: [] for n in names}
    folds = []
    sun = {n: {k: [] for k in shifts} for n in CLASSES}
    for k in range(m):
        train = np.array([i for i in range(m) if i != k])
        reference, albedo = static(train)
        p = RegistrationProjector(common, np.full(len(train), sigma), reference, sc.registration_sigma_px, spatial_degree=2)

        def spatial(a):
            v = np.asarray(a)[..., common]/sigma
            v = v-(v@p.q)@p.q.T
            return v-((v@p.modes)*p.attenuation)@p.modes.T

        observed = spatial(data)
        train_residual = p.apply(data[train])
        errors['null'].append(float(np.mean((observed[k]-observed[train].mean(axis=0))**2)))

        def predicted_error(model):
            return float(np.mean((observed[k]-model[k]-(observed[train]-model[train]).mean(axis=0))**2))

        chosen = {}

        def compact_error(comps, name=None):
            j, g, c, _, _ = _best(p, train_residual, comps[:, :, train], cfg)
            if name:
                chosen[name] = (j, g, c)
            model = g*spatial(comps[j, 0])-c*spatial(comps[j, 1])
            return predicted_error(model)

        def extended_error(frame_loadings):
            # A slope per pixel from the training frames' Sun loadings; the albedo scale cancels.
            centred_train = frame_loadings[train]-frame_loadings[train].mean(axis=0)
            y = observed[train]-observed[train].mean(axis=0)
            field = np.linalg.pinv(centred_train)@y
            held = (frame_loadings[k]-frame_loadings[train].mean(axis=0))@field
            return float(np.mean((observed[k]-observed[train].mean(axis=0)-held)**2))

        for name in ('boulder', 'hummock', 'crater'):
            errors[name].append(compact_error(banks[name][0], name))
            for shift in shifts:
                sun[name][shift].append(compact_error(shuffled_banks[shift][name][0]))
        errors['extended'].append(extended_error(loadings))
        for shift in shifts:
            sun['extended'][shift].append(extended_error(np.roll(loadings, shift, axis=0)))
        folds.append(dict(withheld=int(frames[k]), chosen=chosen))
    e = {n: float(np.mean(v)) for n, v in errors.items()}
    gains = {n: e['null']-e[n] for n in CLASSES}
    sun_gains = {n: max(e['null']-float(np.mean(v)) for v in sun[n].values()) for n in CLASSES}
    result = dict(status='assessed', frames=frames.tolist(), common_pixels=int(common.sum()),
                  error_null=e['null'], errors={n: e[n] for n in CLASSES}, gains=gains,
                  sun_gains=sun_gains, sun_margins={n: gains[n]-sun_gains[n] for n in CLASSES},
                  frame_errors={n: [float(v) for v in errors[n]] for n in names},
                  fits=fits, shifts=list(shifts))
    if record_folds:
        result['folds'] = folds
    return result


# ---------------------------------------------------------------- decision

COMPACT = ('boulder', 'hummock', 'crater')


def candidate(row, compactness):
    """The leading explanation, its closest rival, the lead and the compactness ratio.

    The compactness ratio is the gain of the best single compact feature over the
    gain of the free slope field: the share of the Sun-consistent signal that one
    boulder, hummock or crater explains. Below the calibrated ratio the relief is
    called extended; otherwise the compact classes compete among themselves.
    """
    g = row['gains']
    ranked = sorted(COMPACT, key=lambda c: -g[c])
    ratio = g[ranked[0]]/g['extended'] if g['extended'] > 0 else float('inf')
    if ratio < compactness:
        return 'extended', ranked[0], g['extended']-g[ranked[0]], ratio
    return ranked[0], ranked[1], g[ranked[0]]-g[ranked[1]], ratio


def decide(row, margins):
    """Label one assessed cell from its withheld-frame gains and calibrated margins.

    no_signal         no hypothesis predicts withheld frames better than the null by margins['none'];
    non_solar_change  the leading hypothesis fails the Sun check (margin at most margins['sun']) and
                      reassigned geometry explains nearly as much as the measured geometry: its Sun
                      margin is at most margins['sun_ratio'] of its gain;
    ambiguous         the Sun check fails on a change too weak to tell whether it follows the Sun,
                      or the leading hypothesis does not beat its rival by margins['pair'][class];
    <class>           otherwise.

    A weak change that cannot pass the Sun check is not evidence that the change ignores
    the Sun; calling it non_solar_change would turn missing evidence into a wrong label.
    """
    if row.get('status') != 'assessed':
        return None
    gains = row['gains']
    best, rival, lead, ratio = candidate(row, margins['compactness'])
    sun = row['sun_margins'][best]
    detail = dict(best=best, runner_up=rival, lead=lead, gain=gains[best], compactness=ratio, sun_margin=sun)
    if max(gains.values()) <= margins['none']:
        return dict(detail, label='no_signal')
    if sun <= margins['sun']:
        share = sun/gains[best] if gains[best] > 0 else 0.
        return dict(detail, label='non_solar_change' if share <= margins.get('sun_ratio', float('inf')) else 'ambiguous')
    if lead <= margins['pair'][best]:
        return dict(detail, label='ambiguous')
    return dict(detail, label=best)


def _conformal_quantile(values, target):
    """The ceil((n+1)(1-target))-th smallest value: the split-conformal margin for a false-call rate <= target.

    With exchangeable calibration and test scenes, a new scene exceeds this margin
    with probability at most target. With too few values to reach that rank the
    margin is infinite: nothing is called rather than deciding on too little.
    """
    values = np.sort(np.asarray(values, float))
    n = len(values)
    if n == 0:
        return 0.
    rank = int(np.ceil((n+1)*(1-target)))
    return float(values[rank-1]) if rank <= n else float('inf')


def calibrate_margins(rows, targets):
    """Margins that keep each declared error below its target on the calibration scenes.

    rows carry truth in TRUTH_TO_CLASS keys and the output of evaluate_cell.
    targets (fractions): none, a blank scene called signal; sun, a stripes or blank
    scene passing the Sun check; compactness, an extended-relief scene called
    compact; pair[class], a scene of another physical truth leading as this class
    by more than the margin. Margins are split-conformal quantiles: on test scenes
    exchangeable with these, each declared error rate holds in expectation. They
    say nothing about lunar terrain unlike the calibration scenes.
    """
    ok = [r for r in rows if r.get('status') == 'assessed']
    blank = [r for r in ok if r['truth'] == 'none']
    stripes = [r for r in ok if r['truth'] == 'stripes']
    none = _conformal_quantile([max(r['gains'].values()) for r in blank], targets['none'])
    sun = max(_conformal_quantile([max(r['sun_margins'][c] for c in CLASSES) for r in stripes], targets['sun']),
              _conformal_quantile([max(r['sun_margins'][c] for c in CLASSES) for r in blank], targets['sun']))
    ripples = [r for r in ok if r['truth'] == 'ripples' and r['gains']['extended'] > 0]
    ratios = [max(r['gains'][c] for c in COMPACT)/r['gains']['extended'] for r in ripples]
    compactness = _conformal_quantile(ratios, targets['compactness']) if ratios else 1.
    # Share of the leading gain that survives Sun reassignment, on scenes whose change ignores the Sun:
    # at most this share (for all but the target fraction of stripes) is called non-solar change.
    shares = []
    for r in stripes:
        best = candidate(r, compactness)[0]
        if r['gains'][best] > 0:
            shares.append(max(r['sun_margins'][best], 0.)/r['gains'][best])
    sun_ratio = _conformal_quantile(shares, targets.get('sun_ratio', targets['sun'])) if shares else 0.
    pair = {}
    physical = [r for r in ok if TRUTH_TO_CLASS.get(r['truth']) in CLASSES]
    for c in COMPACT:
        rivals = [r for r in physical if TRUTH_TO_CLASS[r['truth']] in COMPACT and TRUTH_TO_CLASS[r['truth']] != c]
        leads = [r['gains'][c]-max(r['gains'][o] for o in COMPACT if o != c) for r in rivals]
        pair[c] = max(0., _conformal_quantile(leads, targets['pair'].get(c, .05)))
    compact_truth = [r for r in physical if TRUTH_TO_CLASS[r['truth']] in COMPACT]
    leads = [r['gains']['extended']-max(r['gains'][o] for o in COMPACT) for r in compact_truth]
    pair['extended'] = max(0., _conformal_quantile(leads, targets['pair'].get('extended', .05)))
    return dict(none=max(0., none), sun=max(0., sun), sun_ratio=float(min(sun_ratio, 1.)), compactness=float(compactness), pair=pair,
                calibration_counts={t: sum(r['truth'] == t for r in ok) for t in TRUTH_TO_CLASS},
                targets=targets, method='split-conformal quantiles, rank ceil((n+1)(1-target))')


def clopper_pearson(k, n, level=.95):
    """Exact binomial interval for k successes in n trials."""
    if n == 0:
        return (None, None)
    a = (1-level)/2
    low = 0. if k == 0 else float(beta_distribution.ppf(a, k, n-k+1))
    high = 1. if k == n else float(beta_distribution.ppf(1-a, k+1, n-k))
    return (low, high)


def confusion_with_intervals(rows, margins):
    """Confusion table on test scenes, with per-class call rates and exact 95% intervals."""
    table = {}
    for r in rows:
        decision = decide(r, margins)
        if decision is None:
            continue
        table.setdefault(r['truth'], {label: 0 for label in LABELS})[decision['label']] += 1
    summary = {}
    for truth, counts in table.items():
        n = sum(counts.values())
        target = TRUTH_TO_CLASS[truth]
        correct = counts.get(target, 0)
        wrong = n-correct-counts['ambiguous']
        summary[truth] = dict(scenes=n, correct=correct, abstained=counts['ambiguous'], wrong_call=wrong,
                              correct_rate=correct/n, correct_rate_95=clopper_pearson(correct, n),
                              wrong_call_rate=wrong/n, wrong_call_rate_95=clopper_pearson(wrong, n))
    called = {}
    for label in CLASSES:
        column = {t: c[label] for t, c in table.items()}
        n = sum(column.values())
        right = sum(v for t, v in column.items() if TRUTH_TO_CLASS[t] == label)
        called[label] = dict(calls=n, precision=right/n if n else None, precision_95=clopper_pearson(right, n))
    return dict(table=table, per_truth=summary, per_call=called)
