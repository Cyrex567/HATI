"""Experimental adaptive shadow context and joint dimension profiles.

This module does not replace the baseline hazard maps. Every scale fits H0/H1
on identical observations, but scores at different scales are NOT comparable
significances. The last attempted scale controls the status; a failed expansion
cannot fall back to an earlier apparent success. No operational safety labels.
"""
from dataclasses import asdict, dataclass, replace
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing

import numpy as np
from scipy import ndimage as ndi

from .shadow_likelihood import RegistrationProjector, shadow_template, endpoint_support


@dataclass(frozen=True)
class AdaptiveConfig:
    scale_factors: tuple = (1, 2, 4)
    heights_m: tuple = (.1, .3, .6, 1.2, 2.4)
    widths_m: tuple = (.2, .6, 1.2, 2.4)
    fine_step_m: float = .1
    delta_chi2: float = 4.
    warning_score: float = 8.
    region_cutoff_fraction: float = .25
    max_plane_departure_m: float = .15
    stability_m: float = .2
    max_compatibility_span_m: float = .4
    spatial_degree: int = 2
    max_cells: int = 0  # zero means the entire queue; any cap is reported
    workers: int = 1

    def __post_init__(self):
        if not self.scale_factors or self.scale_factors[0] != 1 or any(
                type(v) is not int or v < 1 for v in self.scale_factors) or any(
                b != 2*a for a, b in zip(self.scale_factors, self.scale_factors[1:])):
            raise ValueError('scales must start at one and double')
        for grid in (self.heights_m, self.widths_m):
            if len(grid) < 2 or any(not np.isfinite(v) or v <= 0 for v in grid) or any(
                    b <= a for a, b in zip(grid, grid[1:])):
                raise ValueError('dimension knots must be positive and strictly increasing')
        for v in (self.fine_step_m, self.delta_chi2, self.warning_score,
                  self.max_plane_departure_m, self.stability_m, self.max_compatibility_span_m):
            if not np.isfinite(v) or v <= 0:
                raise ValueError('adaptive scales must be finite and positive')
        if not 0 < self.region_cutoff_fraction <= 1 or self.spatial_degree not in (1, 2):
            raise ValueError('invalid cutoff fraction or spatial degree')
        if type(self.max_cells) is not int or self.max_cells < 0:
            raise ValueError('max_cells must be a nonnegative integer')
        if type(self.workers) is not int or not 1 <= self.workers <= 16:
            raise ValueError('workers must be an integer between one and sixteen')

    def hash(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def cell_table(shape, shadow_cfg, regional_cfg):
    """Same cell ownership and root centres as the baseline, including edges."""
    h, w = shape; radius = shadow_cfg.radius_px; step = regional_cfg.cell_px
    return np.asarray([(r, min(r+step, h-radius), c, min(c+step, w-radius),
                        min(r+step//2, h-radius-1), min(c+step//2, w-radius-1))
                       for r in range(radius, h-radius, step)
                       for c in range(radius, w-radius, step)], int).reshape(-1, 6)


def plan_regions(baseline, shadow_cfg, regional_cfg, cfg):
    """Spatial queue; fraction triggers context for a whole processing tile.

    Individual warning or cutoff cells qualify independently. Neighbouring
    requests are merged for scheduling/display, never pooled into one object.
    """
    shape = baseline['status'].shape
    table = cell_table(shape, shadow_cfg, regional_cfg)
    requested = np.zeros(shape, bool); reasons = {}; tiles = {}
    missing = baseline.get('endpoint_missing_count')
    for i, (r0, r1, c0, c1, r, c) in enumerate(table):
        if baseline['status'][r0, c0] != 1:
            continue
        reason = []
        if baseline['score'][r0, c0] >= cfg.warning_score:
            reason.append('baseline_warning')
        if baseline['endpoint_censored'][r0, c0] > 0:
            reason.append('predicted_cutoff')
        if missing is not None and missing[r0, c0] > 0:
            reason.append('predicted_endpoint_missing')
        key = ((r0-shadow_cfg.radius_px)//regional_cfg.tile_px,
               (c0-shadow_cfg.radius_px)//regional_cfg.tile_px)
        tiles.setdefault(key, []).append(i)
        reasons[i] = reason
    for indices in tiles.values():
        fraction = np.mean(['predicted_cutoff' in reasons[i] for i in indices])
        if fraction >= cfg.region_cutoff_fraction:
            for i in indices:
                reasons[i].append('regional_cutoff_fraction')
    for i, reason in reasons.items():
        if reason:
            r0, r1, c0, c1, *_ = table[i]
            requested[r0:r1, c0:c1] = True
    labels, count = ndi.label(requested, structure=np.ones((3, 3), int))
    regions = []
    for label, box in enumerate(ndi.find_objects(labels), 1):
        regions.append(dict(region=label, bounds=[box[0].start, box[0].stop, box[1].start, box[1].stop]))
    queue = []
    for i, row in enumerate(table):
        if reasons.get(i):
            queue.append(dict(cell_id=int(i), bounds=row[:4].tolist(), centre=row[4:].tolist(),
                              region=int(labels[row[0], row[2]]), reasons=reasons[i]))
    # Deterministic spatial order; a score cannot change who receives a cap.
    queue.sort(key=lambda x: (x['region'], *x['centre']))
    return dict(queue=queue, regions=regions, region_map=labels, requested=requested,
                cells_total=len(table), regions_total=count)


def _refined_axis(knots, values, step):
    """Fill both neighbouring coarse intervals for every compatible knot."""
    knots = np.asarray(knots)
    grid = set(float(v) for v in knots)
    for value in values:
        i = int(np.argmin(abs(knots-value)))
        low, high = knots[max(0, i-1)], knots[min(len(knots)-1, i+1)]
        ticks = np.arange(int(np.ceil((low-1e-9)/step)), int(np.floor((high+1e-9)/step))+1)
        grid.update(float(np.round(t*step, 10)) for t in ticks)
    return sorted(grid)


def fit_patch(patch, visibility, azimuths, elevations, sigma, sc, rc, cfg,
              slopes=(0., 0.), *, selected_frames=None, display=False, terrain=None):
    """Profile every original root, using bounded memory and exact pixel crops.

    terrain, if given, is the relative receiving surface in metres on the patch
    grid widened by rc.cell_px on every side; shadows are cast onto it instead of
    onto the plane given by slopes.
    """
    patch = np.asarray(patch, float); azimuths = np.asarray(azimuths); elevations = np.asarray(elevations)
    if patch.ndim != 3 or patch.shape[1:] != (2*sc.radius_px+1,)*2 or np.shape(visibility) != patch.shape:
        raise ValueError('patch and visibility must match the declared extraction radius')
    if not np.isfinite(sigma) or sigma <= 0 or len(azimuths) != len(patch) or len(elevations) != len(patch):
        raise ValueError('invalid noise or illumination dimensions')
    y, x = np.indices(patch.shape[1:]); radius = sc.radius_px
    support = np.hypot(y-radius, x-radius) <= sc.root_support_px
    valid = np.isfinite(patch) & np.isfinite(visibility) & (np.asarray(visibility) >= .99)
    available = np.flatnonzero(valid[:, support].mean(axis=1) >= rc.min_frame_fraction)
    selected = available if selected_frames is None else np.asarray(selected_frames, int)
    if selected.ndim != 1 or len(np.unique(selected)) != len(selected) or np.any((selected < 0) | (selected >= len(patch))):
        raise ValueError('selected frames must be unique valid indices')
    if len(selected) < 3 or not np.isin(selected, available).all():
        return dict(status='insufficient_frames', frames=selected.tolist())
    common = support & valid[selected].all(axis=0)
    fraction = float(common.sum()/support.sum())
    if common.sum() < 12 or fraction < rc.min_common_fraction:
        return dict(status='insufficient_common_support', frames=selected.tolist(), common_fraction=fraction)
    if not np.isfinite(slopes).all():
        return dict(status='missing_terrain', frames=selected.tolist())
    data = patch[selected]
    reference = np.median(np.where(np.isfinite(data), data, 0.), axis=0)
    p = RegistrationProjector(common, np.full(len(selected), sigma), reference,
                              sc.registration_sigma_px, spatial_degree=cfg.spatial_degree)
    residual = p.apply(data)
    offsets = np.arange(rc.cell_px)-(rc.cell_px-1)/2
    pad = rc.cell_px; base = float(offsets[0]); shape = common.shape
    if terrain is not None and np.shape(terrain) != (shape[0]+2*pad, shape[1]+2*pad):
        raise ValueError('terrain must cover the patch widened by the cell size on every side')
    inner = None if terrain is None else np.asarray(terrain, float)[pad:-pad, pad:-pad]
    evaluated = {}
    def evaluate(heights, widths):
        for ht in heights:
            for width in widths:
                if (ht, width) in evaluated:
                    continue
                try:
                    canvas = shadow_template((shape[0]+2*pad, shape[1]+2*pad),
                        (radius+pad+base,)*2, azimuths[selected], elevations[selected], ht, width, sc, slopes,
                        terrain=terrain)[0]
                except ValueError:
                    return False
                best = None
                for dy in offsets:
                    for dx in offsets:
                        r0, c0 = int(pad+base-dy), int(pad+base-dx)
                        t = canvas[:, r0:r0+shape[0], c0:c0+shape[1]]
                        rt = p.apply(t); energy = float(np.sum(rt*rt))
                        raw = float(np.sum((t[:, common]/sigma)**2))
                        ident = energy/max(raw, 1e-30)
                        if energy <= 1e-12 or ident < sc.min_identifiability:
                            continue
                        inner = float(np.sum(rt*residual))
                        amplitude = float(np.clip(-inner/energy, 0, sc.max_contrast))
                        improvement = max(0., -2*amplitude*inner-amplitude**2*energy)
                        fit = dict(height_m=float(ht), width_m=float(width), root_offset=[float(dy), float(dx)],
                                   improvement=improvement, score=float(np.sqrt(improvement)),
                                   contrast=amplitude, identifiability=ident)
                        if best is None or improvement > best['improvement']:
                            best = fit
                evaluated[ht, width] = best
        return True
    if not evaluate(cfg.heights_m, cfg.widths_m):
        return dict(status='invalid_geometry', frames=selected.tolist())
    coarse = [v for v in evaluated.values() if v]
    if not coarse:
        return dict(status='nonidentifiable', frames=selected.tolist())
    best = max(coarse, key=lambda p: p['improvement'])
    if best['score'] >= cfg.warning_score:
        compatible = [v for v in coarse if best['improvement']-v['improvement'] <= cfg.delta_chi2]
        evaluate(_refined_axis(cfg.heights_m, [v['height_m'] for v in compatible], cfg.fine_step_m),
                 _refined_axis(cfg.widths_m, [v['width_m'] for v in compatible], cfg.fine_step_m))
    fitted = [evaluated[k] for k in sorted(evaluated) if evaluated[k]]
    best = max(fitted, key=lambda p: p['improvement'])
    compatible = [v for v in fitted if best['improvement']-v['improvement'] <= cfg.delta_chi2]
    hr = [min(v['height_m'] for v in compatible), max(v['height_m'] for v in compatible)]
    wr = [min(v['width_m'] for v in compatible), max(v['width_m'] for v in compatible)]
    boundary = any(np.isclose(v['height_m'], (cfg.heights_m[0], cfg.heights_m[-1])).any() or
                   np.isclose(v['width_m'], (cfg.widths_m[0], cfg.widths_m[-1])).any() for v in compatible)
    endpoints = endpoint_support(shape, np.array([radius, radius])+best['root_offset'],
        azimuths[selected], elevations[selected], best['height_m'], sc, slopes, valid=valid[selected], common=common,
        terrain=inner)
    compatible_context = True
    for candidate in compatible:
        checks = endpoint_support(shape, np.array([radius, radius])+candidate['root_offset'],
            azimuths[selected], elevations[selected], candidate['height_m'], sc, slopes, valid=valid[selected], common=common,
            terrain=inner)
        compatible_context &= all(v['endpoint_supported'] and v['background_supported'] for v in checks)
    for item, frame in zip(endpoints, selected):
        item['frame'] = int(frame)
    template = shadow_template(shape, np.array([radius, radius])+best['root_offset'], azimuths[selected],
                               elevations[selected], best['height_m'], best['width_m'], sc, slopes, terrain=inner)[0]
    rt = p.apply(template); after = residual+best['contrast']*rt
    best['frame_delta_chi2'] = np.sum(residual**2-after**2, axis=1).tolist()
    result = dict(status='assessed', best=best, frames=selected.tolist(), common_fraction=fraction,
                  common_pixels=int(common.sum()), endpoints=endpoints,
                  endpoint_censored=any(v['censored'] for v in endpoints),
                  endpoint_context_supported=all(v['endpoint_supported'] and v['background_supported'] for v in endpoints),
                  compatible_context_supported=bool(compatible_context),
                  height_range_m=hr, width_range_m=wr, dimension_at_boundary=bool(boundary),
                  surface=[[v['height_m'], v['width_m'], v['score']] for v in fitted],
                  compatible_pairs=[[v['height_m'], v['width_m']] for v in compatible],
                  hypotheses_evaluated=len(evaluated)*rc.cell_px**2, spatial_degree=cfg.spatial_degree,
                  receiving_surface='terrain' if terrain is not None else 'plane',
                  null_energy=float(np.sum(residual**2)), fitted_energy=float(np.sum(after**2)),
                  uncertainty='Descriptive grid compatibility, no calibrated confidence level; height/width are equivalent rectangular-shadow parameters.')
    if display:
        result['_display'] = dict(patch=data, template=template, common=common, residual_null=residual,
                                  projected_template=rt, frames=selected, slope_rc=slopes)
    return result


def _receiving_surface(window, slopes, pixel_m, centre, support_px, max_missing=.25):
    """DEM tilt at the cell centre plus metre-scale relief from a surface model.

    The surface model's own plane is removed inside the window: shape from
    shading does not observe planes shared by every frame, the DEM does. Holes
    in the model (pixels shadowed in some frame) carry no relief beyond the
    plane; None when more than max_missing of the fitting support is a hole.
    """
    window = np.asarray(window, float)
    yy, xx = (np.indices(window.shape)-centre)*pixel_m
    finite = np.isfinite(window)
    if finite.sum() < 12 or 1-finite[np.hypot(yy, xx) <= support_px*pixel_m].mean() > max_missing:
        return None
    design = np.column_stack([np.ones(finite.sum()), yy[finite], xx[finite]])
    plane = np.linalg.lstsq(design, window[finite], rcond=None)[0]
    relief = np.where(finite, window-(plane[0]+plane[1]*yy+plane[2]*xx), 0.)
    return relief+slopes[0]*yy+slopes[1]*xx


def refine_cell(stack, visibility, azimuths, elevations, sigma, sc, rc, cfg, centre,
                slope_row, slope_col, *, observer=None, terrain=None):
    """Expand real context; freeze initial eligible frames throughout this cell.

    terrain, an optional full-image surface model (relative heights in metres),
    replaces the planar receiving surface: shadows are cast onto the DEM tilt plus
    the model's metre-scale relief, and the plane-departure gate no longer applies
    because the departure is modelled rather than assumed away.
    """
    r, c = map(int, centre); h, w = stack.shape[1:]
    history = []; frozen = None; previous = None; state = 'unresolved_scale_limit'
    for scale in cfg.scale_factors:
        current = replace(sc, radius_px=sc.radius_px*scale, root_support_px=sc.root_support_px*scale)
        radius = current.radius_px
        entry = dict(scale=scale, radius_px=radius, support_px=current.root_support_px)
        if r-radius < 0 or c-radius < 0 or r+radius >= h or c+radius >= w:
            entry.update(status='image_edge'); history.append(entry); state='unresolved_image_edge'; break
        sl = np.s_[r-radius:r+radius+1, c-radius:c+radius+1]
        y, x = np.indices((2*radius+1,)*2)
        support = np.hypot(y-radius, x-radius) <= current.root_support_px
        sr, scol = np.asarray(slope_row)[sl], np.asarray(slope_col)[sl]
        slopes = (float(slope_row[r, c]), float(slope_col[r, c]))
        if not np.isfinite(sr[support]).all() or not np.isfinite(scol[support]).all() or not np.isfinite(slopes).all():
            entry.update(status='missing_terrain'); history.append(entry); state='unresolved_terrain'; break
        # Conservative proxy from gradient variation, not DEM-error validation.
        departure = float(np.max(np.hypot(sr[support]-slopes[0], scol[support]-slopes[1]))*
                          current.root_support_px*sc.pixel_m)
        entry['plane_departure_proxy_m'] = departure
        surface = None
        if terrain is None:
            if departure > cfg.max_plane_departure_m:
                entry.update(status='terrain_plane_limit'); history.append(entry); state='unresolved_terrain'; break
        else:
            pad = rc.cell_px
            if r-radius-pad < 0 or c-radius-pad < 0 or r+radius+pad >= h or c+radius+pad >= w:
                entry.update(status='image_edge'); history.append(entry); state='unresolved_image_edge'; break
            surface = _receiving_surface(np.asarray(terrain)[r-radius-pad:r+radius+pad+1, c-radius-pad:c+radius+pad+1],
                                         slopes, sc.pixel_m, radius+pad, current.root_support_px)
            if surface is None:
                entry.update(status='missing_terrain'); history.append(entry); state='unresolved_terrain'; break
            entry['receiving_surface'] = 'terrain'
        fit = fit_patch(stack[(slice(None), *sl)], visibility[(slice(None), *sl)], azimuths, elevations,
                        sigma, current, rc, cfg, slopes, selected_frames=frozen, display=observer is not None,
                        terrain=surface)
        sample = fit.pop('_display', None)
        entry.update(fit); history.append(entry)
        if observer is not None:
            observer(dict(centre=[r, c], pass_result=entry, sample=sample))
        if fit['status'] != 'assessed':
            state = 'unresolved_'+fit['status']; break
        if frozen is None:
            frozen = fit['frames']
        if fit['best']['score'] < cfg.warning_score:
            state='low_evidence_unqualified'; break
        dimensions = np.array([fit['best']['height_m'], fit['best']['width_m']])
        stable = previous is not None and np.max(abs(dimensions-previous)) <= cfg.stability_m
        narrow = max(np.ptp(fit['height_range_m']), np.ptp(fit['width_range_m'])) <= cfg.max_compatibility_span_m
        if fit['compatible_context_supported'] and not fit['dimension_at_boundary'] and stable and narrow:
            state='context_supported_unvalidated'; break
        previous = dimensions
    return dict(centre=[r, c], status=state, history=history,
                final=history[-1] if history[-1]['status'] == 'assessed' else None)


_WORKER_INPUTS = None


def _init_worker(inputs):
    global _WORKER_INPUTS
    _WORKER_INPUTS = inputs


def _worker_cell(q):
    stack, visibility, az, el, sigma, sc, rc, cfg, sr, scol, display, terrain = _WORKER_INPUTS
    events = []
    result = refine_cell(stack, visibility, az, el, sigma, sc, rc, cfg, q['centre'], sr, scol,
                         observer=events.append if display else None, terrain=terrain)
    return result, events


def _ordered_cells(queue, inputs, observer, load_record):
    """Bound outstanding work; retain queue order independent of CPU scheduling."""
    cfg = inputs[7]
    if cfg.workers == 1:
        for q in queue:
            cached = load_record(q) if load_record else None
            if cached is not None:
                yield q, cached
            else:
                result = refine_cell(*inputs[:8], q['centre'], *inputs[8:10], observer=observer, terrain=inputs[11])
                yield q, result
        return
    with ProcessPoolExecutor(max_workers=cfg.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=_init_worker, initargs=(inputs,)) as pool:
        pending = []; source = iter(queue)
        def enqueue():
            try:
                q = next(source)
            except StopIteration:
                return False
            cached = load_record(q) if load_record else None
            pending.append((q, cached, None if cached is not None else pool.submit(_worker_cell, q)))
            return True
        for _ in range(cfg.workers*2):
            if not enqueue():
                break
        while pending:
            q, cached, future = pending.pop(0)
            if cached is not None:
                result = cached
            else:
                result, events = future.result()
                if observer:
                    for event in events:
                        observer(event)
            yield q, result
            enqueue()


def refine_regions(stack, visibility, azimuths, elevations, sigma, sc, rc, cfg, baseline,
                   slope_row, slope_col, *, on_record=None, observer=None, progress=None, load_record=None, terrain=None):
    """Run the queue without an implicit top-score cap; stream cell histories.

    terrain: optional full-image surface model passed to every refine_cell.
    """
    if np.shape(stack) != np.shape(visibility) or np.shape(stack)[1:] != baseline['status'].shape:
        raise ValueError('adaptive inputs must share the original image grid')
    if terrain is not None and np.shape(terrain) != np.shape(stack)[1:]:
        raise ValueError('terrain must share the original image grid')
    plan = plan_regions(baseline, sc, rc, cfg); shape = stack.shape[1:]
    keys = ('score', 'height_m', 'width_m', 'height_low_m', 'height_high_m', 'width_low_m', 'width_high_m',
            'pass_scale', 'endpoint_censored_count', 'endpoint_missing_count')
    maps = {k: np.full(shape, np.nan) for k in keys}
    status_map = np.zeros(shape, np.uint8); status_map[plan['requested']] = 1
    counts = {}; processed = 0
    queue = plan['queue'][:cfg.max_cells] if cfg.max_cells else plan['queue']
    inputs = (stack, visibility, azimuths, elevations, sigma, sc, rc, cfg, slope_row, slope_col, observer is not None, terrain)
    for q, result in _ordered_cells(queue, inputs, observer, load_record):
        record = {**q, **{k: v for k, v in result.items() if k != 'centre'}}
        if on_record:
            on_record(record)
        processed += 1; state = result['status']; counts[state] = counts.get(state, 0)+1
        r0, r1, c0, c1 = q['bounds']; out = np.s_[r0:r1, c0:c1]
        status_map[out] = 2 if state == 'context_supported_unvalidated' else 3 if state == 'low_evidence_unqualified' else 4
        maps['pass_scale'][out] = result['history'][-1]['scale']
        final = result['final']
        if final:
            maps['score'][out] = final['best']['score']
            # A weak noise fit still has a numerical argmax. Keep it in the
            # audit history, never paint it as an estimated object's dimensions.
            if final['best']['score'] >= cfg.warning_score:
                for key in ('height_m', 'width_m'):
                    maps[key][out] = final['best'][key]
                for dim in ('height', 'width'):
                    maps[dim+'_low_m'][out], maps[dim+'_high_m'][out] = final[dim+'_range_m']
            maps['endpoint_censored_count'][out] = sum(p['censored'] for p in final['endpoints'])
            maps['endpoint_missing_count'][out] = sum(not p['endpoint_supported'] for p in final['endpoints'])
        if progress:
            progress(dict(processed=processed, requested=len(plan['queue']), last=record,
                          status=status_map, score=maps['score']))
    return dict(**maps, status=status_map, region=plan['region_map'], regions=plan['regions'],
                requested=len(plan['queue']), processed=processed, unprocessed=len(plan['queue'])-processed,
                state_counts=counts, configuration=asdict(cfg), config_hash=cfg.hash(),
                status_legend={'0':'not_requested_or_baseline_unavailable', '1':'queued_unprocessed',
                               '2':'context_supported_unvalidated', '3':'low_evidence_unqualified', '4':'unresolved'},
                interpretation='Experimental dimension diagnostics. No score maximization across scales, no calibrated detection probability, no change to baseline fusion.')


def held_out_prediction(patch, visibility, azimuths, elevations, sigma, sc, rc, cfg,
                        slopes=(0., 0.), *, held_frame=-1, wrong_azimuth_deg=90.):
    """Fit only training intensities; freeze object before withheld prediction.

    Common support may use withheld validity, never withheld intensities. The
    held frame gets the same spatial nuisance projection for all predictions.
    Predictive errors are descriptive, not independent chi-square statistics.
    """
    n = len(patch)
    if n < 4:
        return dict(status='insufficient_frames')
    held_frame %= n
    train = np.array([i for i in range(n) if i != held_frame])
    valid = np.isfinite(patch) & np.isfinite(visibility) & (visibility >= .99)
    # Freeze a shared mask before training, without admitting held-out values.
    common_valid = valid.all(axis=0)
    vis = np.broadcast_to(common_valid, np.shape(patch)).astype(float)
    fit = fit_patch(patch, vis, azimuths, elevations, sigma, sc, rc, cfg,
                    slopes, selected_frames=train)
    if fit['status'] != 'assessed':
        return dict(status=fit['status'], held_frame=held_frame, training=fit)
    best = fit['best']; shape = patch.shape[1:]; centre = (np.array(shape)-1)/2
    y, x = np.indices(shape); common = common_valid & (np.hypot(y-centre[0], x-centre[1]) <= sc.root_support_px)
    reference = np.median(np.where(np.isfinite(patch[train]), patch[train], 0.), axis=0)
    projector = RegistrationProjector(common, np.full(len(train), sigma), reference,
                                      sc.registration_sigma_px, spatial_degree=cfg.spatial_degree)
    def spatial(a):
        v = np.asarray(a)[..., common]/sigma
        v = v-(v@projector.q)@projector.q.T
        return v-((v@projector.modes)*projector.attenuation)@projector.modes.T
    az = np.asarray(azimuths, float); el = np.asarray(elevations, float)
    root = centre+best['root_offset']
    try:
        t = shadow_template(shape, root, az, el, best['height_m'], best['width_m'], sc, slopes)[0]
    except ValueError:
        return dict(status='invalid_held_geometry', training=fit, held_frame=int(held_frame))
    observed = spatial(patch); predicted = spatial(t)
    static = observed[train].mean(axis=0)
    corrected_static = (observed[train]+best['contrast']*predicted[train]).mean(axis=0)
    errors = dict(static=float(np.mean((observed[held_frame]-static)**2)),
                  correct=float(np.mean((observed[held_frame]+best['contrast']*predicted[held_frame]-corrected_static)**2)))
    try:
        wrong = shadow_template(shape, root, [az[held_frame]+wrong_azimuth_deg], [el[held_frame]],
                                best['height_m'], best['width_m'], sc, slopes)[0][0]
        errors['wrong_direction'] = float(np.mean((observed[held_frame]+best['contrast']*spatial(wrong)-corrected_static)**2))
    except ValueError:
        errors['wrong_direction'] = None
    return dict(status='assessed', held_frame=int(held_frame), training_frames=train.tolist(),
                training=fit, error_per_pixel=errors, common_pixels=int(common.sum()),
                correct_advantage_over_static=errors['static']-errors['correct'],
                correct_advantage_over_wrong=(errors['wrong_direction']-errors['correct']) if errors['wrong_direction'] is not None else None,
                wrong_azimuth_offset_deg=wrong_azimuth_deg,
                interpretation='Fixed-window conditional prediction. Withheld intensities did not select scale, root, dimensions, contrast, or covariance. No calibrated confidence or lunar validation.')
