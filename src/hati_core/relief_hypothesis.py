"""Compact caster versus relief, compared by withheld-frame prediction (campaign T13).

H0        static image and a brightness plane per frame (the detector's null);
H_rock    the regional detector's rock-shadow bank with nonnegative contrast;
H_relief  shading of a small height field shared by every frame, linearised
          about flat ground: brightness changes by -cot(e_k) grad(h).s_k times the
          local relative albedo, with s_k the horizontal direction toward the Sun.
          The basis is a Gaussian bump and its two first moments at each scale,
          so mounds and bowls are the same model with opposite sign and the order
          of bright and dark along the Sun follows from the fit.

Each model is fitted to the training frames only and then predicts the
withheld frame through the same spatial projection; the comparison uses those
prediction errors, never a fitted score alone. Linear shading holds below the
Sun elevation. Steeper slopes self-shadow and are only partly described, so
relief slopes estimated here are lower bounds.
"""
import numpy as np

from .noise_scale import sun_loadings
from .shadow_likelihood import RegistrationProjector, shadow_template

CLASSES = ('rock_like', 'relief_like', 'ambiguous', 'none')


def relief_basis(shape, pixel_m, scales_m):
    """Height basis (metres per unit coefficient), shape (basis, rows, cols)."""
    rows, cols = np.indices(shape, dtype=float)
    centre = (np.asarray(shape)-1)/2
    dr, dc = (rows-centre[0])*pixel_m, (cols-centre[1])*pixel_m
    out = []
    for s in scales_m:
        g = np.exp(-(dr**2+dc**2)/(2*s*s))
        out.extend((g, dr/s*g, dc/s*g))
    return np.asarray(out)


def relief_design(basis, pixel_m, azimuths, elevations, albedo):
    """Brightness change of each basis height under each frame: (basis, frames, rows, cols)."""
    loadings = sun_loadings(azimuths, elevations)
    out = []
    for b in basis:
        gr, gc = np.gradient(b, pixel_m)
        out.append(-albedo[None]*(loadings[:, 0, None, None]*gr[None]+loadings[:, 1, None, None]*gc[None]))
    return np.asarray(out)


def rock_bank(shape, radius, azimuths, elevations, sc, rc, slopes):
    """The regional detector's templates for one patch: (templates, frames, rows, cols), parameters."""
    offsets = np.arange(rc.cell_px)-(rc.cell_px-1)/2
    pad = rc.cell_px; base = float(offsets[0])
    rendered = {(ht, width): shadow_template((shape[0]+2*pad, shape[1]+2*pad), (radius+pad+base,)*2,
                                             azimuths, elevations, ht, width, sc, slopes)[0]
                for ht in rc.heights_m for width in rc.widths_m}
    templates, parameters = [], []
    for dy in offsets:
        for dx in offsets:
            for ht in rc.heights_m:
                for width in rc.widths_m:
                    r0, c0 = int(pad+base-dy), int(pad+base-dx)
                    templates.append(rendered[ht, width][:, r0:r0+shape[0], c0:c0+shape[1]])
                    parameters.append((float(dy), float(dx), float(ht), float(width)))
    return np.asarray(templates), parameters


def _fit_rock(projector, residual, templates, sc):
    rt = projector.apply(templates)
    energy = np.sum(rt*rt, axis=(1, 2))
    raw = np.sum((templates[..., projector.common]/projector.sigma[:, None])**2, axis=(1, 2))
    eligible = (energy > 1e-12) & (energy/np.maximum(raw, 1e-30) >= sc.min_identifiability)
    if not eligible.any():
        return None
    inner = np.sum(rt*residual, axis=(1, 2))
    contrast = np.clip(-inner/np.maximum(energy, 1e-30), 0, sc.max_contrast)
    improvement = np.where(eligible, np.maximum(0, -2*contrast*inner-contrast**2*energy), -np.inf)
    best = int(np.argmax(improvement))
    return dict(index=best, contrast=float(contrast[best]), improvement=float(improvement[best]))


def _fit_relief(projector, residual, design):
    x = projector.apply(design).reshape(len(design), -1).T
    coefficients, *_ = np.linalg.lstsq(x, residual.ravel(), rcond=None)
    fitted = x@coefficients
    return dict(coefficients=coefficients, improvement=float(fitted@fitted))


def relief_slope_deg(basis, coefficients, pixel_m, support):
    h = np.tensordot(coefficients, basis, axes=1)
    gr, gc = np.gradient(h, pixel_m)
    return float(np.degrees(np.arctan(np.max(np.hypot(gr, gc)[support]))))


def compare_models(patch, valid, azimuths, elevations, sigma, sc, rc, scales_m, slopes=(0., 0.)):
    """Withheld-frame prediction errors of H0, H_rock and H_relief for one cell.

    patch and valid are (frames, 2r+1, 2r+1) around the cell centre. Errors are
    mean squared per pixel in noise units, averaged over every withheld frame.
    Also returns full-data fits: the rock score as the detector computes it, the
    relief improvement and a lower-bound relief slope.
    """
    patch = np.asarray(patch, float); valid = np.asarray(valid, bool)
    radius = sc.radius_px; shape = patch.shape[1:]
    if shape != (2*radius+1,)*2:
        raise ValueError('patch must match the detector radius')
    y, x = np.indices(shape)
    support = np.hypot(y-radius, x-radius) <= sc.root_support_px
    usable = valid & np.isfinite(patch)
    frames = np.flatnonzero(usable[:, support].mean(axis=1) >= rc.min_frame_fraction)
    if len(frames) < 4:
        return dict(status='insufficient_frames', frames=frames.tolist())
    common = support & usable[frames].all(axis=0)
    if common.sum() < 12 or common.sum()/support.sum() < rc.min_common_fraction:
        return dict(status='insufficient_common_support', frames=frames.tolist())
    data = patch[frames]
    az = np.asarray(azimuths, float)[frames]; el = np.asarray(elevations, float)[frames]
    def static(frames_used):
        reference = np.median(np.where(np.isfinite(data[frames_used]), data[frames_used], 0.), axis=0)
        return reference, np.where(common, reference/np.mean(reference[common]), 1.)
    try:
        templates, parameters = rock_bank(shape, radius, az, el, sc, rc, slopes)
    except ValueError:
        return dict(status='invalid_geometry', frames=frames.tolist())
    basis = relief_basis(shape, sc.pixel_m, scales_m)
    m = len(frames)
    reference, albedo = static(np.arange(m))
    full = RegistrationProjector(common, np.full(m, sigma), reference, sc.registration_sigma_px)
    residual = full.apply(data)
    rock = _fit_rock(full, residual, templates, sc)
    relief = _fit_relief(full, residual, relief_design(basis, sc.pixel_m, az, el, albedo))
    errors = dict(null=[], rock=[], relief=[])
    for k in range(m):
        train = np.array([i for i in range(m) if i != k])
        # Withheld intensities never enter the static image, the albedo or any fit.
        reference, albedo = static(train)
        design = relief_design(basis, sc.pixel_m, az, el, albedo)
        p = RegistrationProjector(common, np.full(len(train), sigma), reference, sc.registration_sigma_px)
        def spatial(a):
            v = np.asarray(a)[..., common]/sigma
            v = v-(v@p.q)@p.q.T
            return v-((v@p.modes)*p.attenuation)@p.modes.T
        observed = spatial(data)
        errors['null'].append(float(np.mean((observed[k]-observed[train].mean(axis=0))**2)))
        train_residual = p.apply(data[train])
        fit = _fit_rock(p, train_residual, templates[:, train], sc)
        if fit is None:
            errors['rock'].append(errors['null'][-1])
        else:
            t = fit['contrast']*spatial(templates[fit['index']])
            errors['rock'].append(float(np.mean((observed[k]+t[k]-(observed[train]+t[train]).mean(axis=0))**2)))
        coefficients = _fit_relief(p, train_residual, design[:, train])['coefficients']
        model = np.tensordot(coefficients, spatial(design), axes=1)
        errors['relief'].append(float(np.mean((observed[k]-model[k]-(observed[train]-model[train]).mean(axis=0))**2)))
    e = {k: float(np.mean(v)) for k, v in errors.items()}
    best_rock = parameters[rock['index']] if rock else None
    return dict(status='assessed', frames=frames.tolist(), common_pixels=int(common.sum()),
                error_null=e['null'], error_rock=e['rock'], error_relief=e['relief'],
                gain_rock=e['null']-e['rock'], gain_relief=e['null']-e['relief'],
                rock_score=float(np.sqrt(rock['improvement'])) if rock else None,
                rock_parameters=dict(root_offset=list(best_rock[:2]), height_m=best_rock[2], width_m=best_rock[3],
                                     contrast=rock['contrast']) if rock else None,
                relief_improvement=relief['improvement'],
                relief_slope_lower_bound_deg=relief_slope_deg(basis, relief['coefficients'], sc.pixel_m, support),
                relief_centre_height_m=float(np.tensordot(relief['coefficients'], basis, axes=1)[radius, radius]))


def classify(row, margins):
    """rock_like / relief_like / ambiguous / none from withheld-frame gains and calibrated margins."""
    if row.get('status') != 'assessed':
        return None
    g_rock, g_relief = row['gain_rock'], row['gain_relief']
    if max(g_rock, g_relief) <= margins['none']:
        return 'none'
    if g_rock-g_relief > margins['rock']:
        return 'rock_like'
    if g_relief-g_rock > margins['relief']:
        return 'relief_like'
    return 'ambiguous'


def calibrate_margins(rows, targets):
    """Smallest margins meeting the declared error targets on calibration scenes.

    rows carry truth in ('rock', 'relief', 'none') and withheld-frame gains.
    Calling a rock relief-like hides a rock inside the terrain module, so that
    error gets the tighter target. Neither margin falls below the difference
    between the two gains that noise alone produces on blank scenes, so a cell
    the data cannot separate is called ambiguous rather than decided by noise.
    """
    def quantile(values, target):
        values = np.asarray(values, float)
        return max(0., float(np.quantile(values, 1-target))) if len(values) else 0.
    rocks = [r for r in rows if r['truth'] == 'rock' and r.get('status') == 'assessed']
    reliefs = [r for r in rows if r['truth'] == 'relief' and r.get('status') == 'assessed']
    blanks = [r for r in rows if r['truth'] == 'none' and r.get('status') == 'assessed']
    floor = quantile([abs(r['gain_rock']-r['gain_relief']) for r in blanks], targets['blank_called_signal'])
    return dict(relief=max(floor, quantile([r['gain_relief']-r['gain_rock'] for r in rocks], targets['rock_called_relief'])),
                rock=max(floor, quantile([r['gain_rock']-r['gain_relief'] for r in reliefs], targets['relief_called_rock'])),
                none=quantile([max(r['gain_rock'], r['gain_relief']) for r in blanks], targets['blank_called_signal']),
                ambiguity_floor=floor,
                calibration_counts=dict(rock=len(rocks), relief=len(reliefs), none=len(blanks)))


def confusion(rows, margins):
    table = {}
    for r in rows:
        label = classify(r, margins)
        if label is None:
            continue
        table.setdefault(r['truth'], {c: 0 for c in CLASSES})[label] += 1
    return table
