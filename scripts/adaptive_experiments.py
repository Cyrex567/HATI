"""T9-T11 workers: adaptive context, independent rocks, withheld illumination."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from landing_maps import clean_json, write_tif
from src.hati_core.adaptive_shadow import AdaptiveConfig, refine_regions, held_out_prediction, cell_table
from src.hati_core.regional_shadow import assess_regions
from src.hati_core.rock_scenes import load_catalog, make_rock, render_rocks
from src.hati_core.campaign_controls import render_control


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(clean_json(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def adaptive_config(ex):
    return AdaptiveConfig(**ex.cfg.get('adaptive', {}))


def plot_layers(path, result, *, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout='constrained')
    fields = [('status', 'Refinement state'), ('pass_scale', 'Last attempted scale'),
              ('score', 'Last-scale score (uncalibrated)'), ('height_m', 'Equivalent height (m)'),
              ('width_m', 'Equivalent width (m)'), ('endpoint_censored_count', 'Predicted endpoint cutoffs')]
    for ax, (key, label) in zip(axes.flat, fields):
        kwargs = dict(interpolation='nearest')
        if key == 'status':
            kwargs.update(cmap=ListedColormap(['#7e8b98', '#e9b554', '#62bfa7', '#7c81b7', '#c84662']), vmin=-.5, vmax=4.5)
        else:
            kwargs['cmap'] = plt.get_cmap('magma').with_extremes(bad='#7e8b98')
        if key == 'pass_scale':
            kwargs.update(vmin=1, vmax=max(result['configuration']['scale_factors']) or 1)
        im = ax.imshow(result[key], **kwargs); ax.set_title(label); ax.set_xlabel('Column'); ax.set_ylabel('Row')
        bar = fig.colorbar(im, ax=ax, shrink=.8)
        if key == 'status':
            bar.set_ticks(range(5)); bar.set_ticklabels(['Not requested', 'Queued', 'Context supported*', 'Low evidence*', 'Unresolved'])
        elif key == 'pass_scale':
            bar.set_ticks(result['configuration']['scale_factors'])
            bar.set_ticklabels([str(v)+'x' for v in result['configuration']['scale_factors']])
    fig.suptitle(title+'\n*Unvalidated; no safety or dimension-accuracy claim')
    fig.savefig(path, dpi=140); plt.close(fig)


def t9(ex):
    cfg = adaptive_config(ex); d = ex.data; base = ex.baseline()
    cells = ex.out/'cells'; cells.mkdir(exist_ok=True)
    signature = fingerprint(dict(config=asdict(cfg), shadow=asdict(ex.sc), regional=asdict(ex.rc),
                                 noise=ex.noise, input=ex.proof,
                                 baseline=hashlib.sha256((ex.baseline_dir/'regional.npz').read_bytes()).hexdigest()))
    # Outer campaign resume verifies source/input hashes. Each completed cell
    # also has a signature and checksum so interruption need not redo the queue.
    def record(row):
        save(cells/f'{row["cell_id"]:06d}.json', dict(signature=signature, sha256=fingerprint(row), record=row))
    def cached(q):
        path = cells/f'{q["cell_id"]:06d}.json'
        if not path.exists():
            return None
        item = json.loads(path.read_text(encoding='utf-8')); row = item.get('record')
        if item.get('signature') != signature or item.get('sha256') != fingerprint(row) or any(row.get(k) != q[k] for k in q):
            raise ValueError('adaptive cell checkpoint provenance/checksum mismatch: '+path.name)
        return row
    last = [0.]
    def progress(info):
        if time.monotonic()-last[0] > 15 or info['processed'] == info['requested']:
            print(f'Adaptive context: {info["processed"]}/{info["requested"]} requested cells; '
                  f'last={info["last"]["status"]}', flush=True); last[0] = time.monotonic()
        ex.live.adaptive_progress(info)
    result = refine_regions(d['stack'], d['visibility'], d['azimuths'], d['elevations'], ex.noise,
                            ex.sc, ex.rc, cfg, base, d['slope_row'], d['slope_col'],
                            on_record=record, load_record=cached, observer=ex.live.adaptive, progress=progress)
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(ex.out/'adaptive.npz', **arrays)
    for key, array in arrays.items():
        write_tif(ex.out/(key+'.tif'), array, ex.transform, ex.crs,
                  'Experimental adaptive diagnostic; no calibrated probability, confidence interval or landing clearance')
    plot_layers(ex.out/'adaptive_context.png', result,
                title='HATI | adaptive context | '+('SYNTHETIC DEMONSTRATION' if ex.run.get('demo') else 'saved lunar observations'))
    details = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
    details['adaptive_configuration'] = details.pop('configuration')
    return ex.result('PARTIAL', 'Adaptive cell histories and dimension compatibility exported; calibration and independent field validation remain open.', **details)


def _control_trial(ex, cfg, scene, slopes, folder, truth, *, noise=None, predict=True, save_arrays=True):
    """One synthetic scene through the baseline and the full adaptive procedure.

    noise is the sigma the detector assumes (default: the source run's value).
    predict=False skips the withheld-frame predictions, which T16 does not use.
    save_arrays=False keeps the per-scene arrays out of the results archive.
    """
    noise = ex.noise if noise is None else noise
    shape = scene.shape[1:]; sr = np.full(shape, slopes[0]); scol = np.full(shape, slopes[1])
    visibility = np.ones_like(scene)
    kind = truth['kind']
    if kind == 'missing_context':
        r = shape[0]//2; visibility[:, r+8:r+12, :] = 0
    if kind == 'terrain_break':
        sr[:, shape[1]//2+8:] += .05
    baseline = assess_regions(scene, ex.data['azimuths'], ex.data['elevations'], noise, ex.sc, ex.rc,
                              visible=visibility, slope_row=sr, slope_col=scol)
    original_assessed = baseline['status'] == 1
    original_scores = baseline['score'][original_assessed]
    # Predeclared central ROI, independent of scene intensities and fit results.
    # Every cell inside it goes through the normal adaptive request decision.
    table = cell_table(shape, ex.sc, ex.rc); middle = np.array(shape)//2
    central_coordinate = min(table[:, 4:], key=lambda v: np.linalg.norm(v-middle))
    half_span = ex.rc.cell_px*(ex.cfg.get('rock_roi_cells', 3)-1)/2
    scope = np.zeros(shape, bool); roi_count = 0
    for r0, r1, c0, c1, r, c in table:
        if abs(r-central_coordinate[0]) <= half_span and abs(c-central_coordinate[1]) <= half_span:
            scope[r0:r1, c0:c1] = True; roi_count += 1
    baseline['status'] = np.where(scope, baseline['status'], 0)
    records = []
    result = refine_regions(scene, visibility, ex.data['azimuths'], ex.data['elevations'], noise,
                            ex.sc, ex.rc, replace(cfg, max_cells=0, workers=1), baseline, sr, scol, on_record=records.append)
    save(folder/'cell_histories.json', records)
    if save_arrays:
        np.savez_compressed(folder/'scene.npz', stack=scene, visibility=visibility, slope_row=sr, slope_col=scol,
                            azimuths=ex.data['azimuths'], elevations=ex.data['elevations'])
        np.savez_compressed(folder/'adaptive.npz', **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    root = np.asarray(truth.get('root_px', middle), float)
    warnings = []
    for row in records:
        fit = row['final']
        if fit and fit['best']['score'] >= cfg.warning_score:
            pos = np.asarray(row['centre'])+fit['best']['root_offset']
            warnings.append((float(np.linalg.norm(pos-root)), row))
    recovered = [row for distance, row in warnings if distance <= 2]
    # Evaluate the predeclared cell nearest the planted root, never the best
    # measured height/error across the ROI. Unrequested/failed cells are missing.
    nearest_centre = min([row[4:] for row in table if scope[row[0], row[2]]], key=lambda v: np.linalg.norm(v-root))
    central = next((row for row in records if np.array_equal(row['centre'], nearest_centre)), None)
    fitted = central['final'] if central else None
    if fitted and fitted['best']['score'] < cfg.warning_score:
        fitted = None
    ht = truth.get('height_m')
    row = dict(**truth, roi_cells=roi_count, roi_assessed_cells=int(sum(baseline['status'][r[0], r[2]] == 1 for r in table)),
               requested=result['requested'], processed=result['processed'], state_counts=result['state_counts'],
               baseline_frame_warning_fraction=float(np.mean(original_scores >= ex.rc.score_scale)) if len(original_scores) else None,
               adaptive_warning_cells=len(warnings), context_supported_cells=result['state_counts'].get('context_supported_unvalidated', 0),
               cells_with_final_fit=sum(r['final'] is not None for r in records),
               unresolved_cells=sum(r['status'].startswith('unresolved') for r in records),
               recovered_within_2px=bool(recovered) if ht is not None else None,
               central_status=central['status'] if central else 'not_requested_or_unavailable',
               central_best=fitted['best'] if fitted else None,
               height_error_m=fitted['best']['height_m']-ht if ht is not None and fitted else None,
               height_in_compatibility=bool(any(np.isclose(p[0], ht) for p in fitted['compatible_pairs'])) if ht is not None and fitted else None,
               width_error_m=fitted['best']['width_m']-truth['width_m'] if ht is not None and fitted else None,
               dimension_interpretation='Template-equivalent dimensions versus imposed body dimensions; shape mismatch is part of this test.')
    # Same predeclared central target at the largest window, even if adaptive
    # selection did not request it. Held intensities cannot select this window.
    predictions = []
    rad = ex.sc.radius_px*max(cfg.scale_factors); r, c = map(int, nearest_centre)
    if predict and r-rad >= 0 and c-rad >= 0 and r+rad < shape[0] and c+rad < shape[1] and kind != 'terrain_break':
        sc = replace(ex.sc, radius_px=rad, root_support_px=ex.sc.root_support_px*max(cfg.scale_factors))
        sl = np.s_[:, r-rad:r+rad+1, c-rad:c+rad+1]
        for degree in (1, 2):
            prediction = held_out_prediction(scene[sl], visibility[sl], ex.data['azimuths'], ex.data['elevations'],
                noise, sc, ex.rc, replace(cfg, spatial_degree=degree), slopes)
            predictions.append(dict(spatial_degree=degree, **prediction))
    save(folder/'withheld_predictions.json', predictions)
    row['withheld_predictions'] = [{k: v for k, v in p.items() if k != 'training'} for p in predictions]
    save(folder/'summary.json', row)
    return row, result


def t10(ex):
    cfg = adaptive_config(ex); rows = []; examples = set()
    catalog = getattr(ex.args, 'rock_catalog', None)
    meshes = load_catalog(catalog) if catalog else []
    if catalog and not meshes:
        raise ValueError('rock catalog has no evaluation meshes')
    seeds = ex.cfg.get('rock_seeds', 2); heights = ex.cfg.get('rock_heights_m', [.2, .3, .4, .5, .6, .8, 1.2])
    # Enough true image context for every configured scale at the central ROI.
    size = 2*ex.sc.radius_px*max(cfg.scale_factors)+4*ex.rc.cell_px+1
    root = [size//2+.3, size//2+.2]
    scenarios = [(kind, None) for kind in ('static', 'structured_null', 'resolved_ridge')]
    scenarios += [(kind, ht) for kind in ('procedural', 'apollo_proxy', 'overlap') for ht in heights
                  if kind != 'apollo_proxy' or meshes]
    scenarios += [(kind, .6) for kind in ('missing_context', 'registration_stress', 'terrain_break', 'sloping_plane')]
    for case, (kind, height) in enumerate(scenarios):
        for trial in range(seeds):
            seed = ex.cfg['seed']+100000+100*case+trial
            folder = ex.out/f'scenes/{case:03d}_{kind}_{trial:02d}'; folder.mkdir(parents=True, exist_ok=True)
            slopes = (.01, -.01) if kind == 'sloping_plane' else (0., 0.)
            truth = dict(kind=kind, seed=seed, height_m=height, width_m=.6 if height is not None else None,
                         root_px=root, slope_rc=slopes, scene_scope='predeclared central ROI; full-frame baseline also recorded')
            if kind == 'resolved_ridge':
                scene = render_control((size, size), ex.data['azimuths'], ex.data['elevations'], pixel_m=ex.sc.pixel_m,
                    seed=seed, noise=ex.noise, kind=kind, height=.6, slope_rc=slopes, root=root)
                truth['generator'] = 'independent_rounded_ridge_control'
            else:
                rocks = []
                if height is not None:
                    mesh = meshes[trial % len(meshes)] if kind == 'apollo_proxy' else None
                    rocks.append(make_rock(seed, root, height, .6, aspect=1.35, mesh=mesh))
                    if kind == 'overlap':
                        rocks.append(make_rock(seed+1, [root[0]+1.7, root[1]+2.1], height*.7, .5))
                generated = render_rocks((size, size), ex.data['azimuths'], ex.data['elevations'], rocks,
                    pixel_m=ex.sc.pixel_m, seed=seed, noise=ex.noise, slope_rc=slopes,
                    supersample=ex.cfg.get('rock_supersample', 6),
                    registration_sigma_px=.5 if kind == 'registration_stress' else 0., structured_null=kind == 'structured_null')
                scene = generated['stack']; truth['generator_truth'] = generated['truth']
            row, result = _control_trial(ex, cfg, scene, slopes, folder, truth)
            rows.append(row)
            if kind not in examples:
                plot_layers(ex.out/(kind+'_adaptive.png'), result, title='HATI | synthetic '+kind)
                ex.plot_grid(kind+'_scene.png', [scene[0], scene[-1]], ['Synthetic first frame', 'Synthetic last frame'])
                examples.add(kind)
            # Synthetic local coordinates never overlay the real Athena input.
            ex.live.update(force=True, kind='controls', fit=None, message='Independent 3D rock controls',
                control=dict(kind=kind, height_m=height, seed=seed, status=row['central_status'],
                             recovered=row['recovered_within_2px'], maximum_score=None))
            print(f'3D control {len(rows)}/{len(scenarios)*seeds}: {kind} h={height}; {row["central_status"]}', flush=True)
    save(ex.out/'controls.json', rows)
    summaries = []
    for kind in sorted({r['kind'] for r in rows}):
        sample = [r for r in rows if r['kind'] == kind]
        errors = [r['height_error_m'] for r in sample if r['height_error_m'] is not None]
        summaries.append(dict(kind=kind, trials=len(sample), trials_with_warning=sum(r['adaptive_warning_cells'] > 0 for r in sample),
            trials_with_context_supported=sum(r['context_supported_cells'] > 0 for r in sample),
            context_supported_cells=sum(r['context_supported_cells'] for r in sample),
            trials_with_final_fit=sum(r['cells_with_final_fit'] > 0 for r in sample),
            trials_with_unresolved_cells=sum(r['unresolved_cells'] > 0 for r in sample),
            trials_without_requests=sum(r['requested'] == 0 for r in sample),
            recovered=sum(r['recovered_within_2px'] is True for r in sample),
            requested_cells=sum(r['requested'] for r in sample), processed_cells=sum(r['processed'] for r in sample),
            dimension_trials=len(errors), median_height_bias_m=float(np.median(errors)) if errors else None,
            median_absolute_height_error_m=float(np.median(np.abs(errors))) if errors else None))
    return ex.result('PARTIAL', 'Independent 3D scenes and full adaptive decisions within fixed synthetic ROIs recorded; field false-alarm and dimension calibration remain open.',
        controls=summaries, trials=len(rows), evaluation_meshes=len(meshes), catalog_supplied=bool(catalog),
        missing_sources=[] if meshes else ['External lunar shape catalog not supplied; procedural controls only'],
        limitations=['Synthetic ROI trials are not a full-image false-alarm calibration.',
                     'Ridge controls contain real relief; a warning there is not automatically a false alarm.',
                     'Returned Apollo samples, convex reduction and rescaling do not establish polar morphology priors.',
                     'Independent rendered shadows use simplified photometry and planar receiving terrain.'])


def t11(ex):
    cfg = adaptive_config(ex); d = ex.data; rows = []
    step = ex.cfg.get('prediction_step_px', 128)
    largest = ex.sc.radius_px*max(cfg.scale_factors)
    # Fixed lattice, every held frame, every declared scale and both nuisance
    # families. Neither T9's adaptive outcomes nor held-out intensities select it.
    for r in range(largest, d['stack'].shape[1]-largest, step):
        for c in range(largest, d['stack'].shape[2]-largest, step):
            for scale in cfg.scale_factors:
                sc = replace(ex.sc, radius_px=ex.sc.radius_px*scale, root_support_px=ex.sc.root_support_px*scale)
                rad = sc.radius_px; sl = np.s_[:, r-rad:r+rad+1, c-rad:c+rad+1]
                slopes = (d['slope_row'][r, c], d['slope_col'][r, c])
                sr = d['slope_row'][r-rad:r+rad+1, c-rad:c+rad+1]
                scol = d['slope_col'][r-rad:r+rad+1, c-rad:c+rad+1]
                y, x = np.indices(sr.shape); support = np.hypot(y-rad, x-rad) <= sc.root_support_px
                departure = np.max(np.hypot(sr[support]-slopes[0], scol[support]-slopes[1]))*sc.root_support_px*sc.pixel_m
                for degree in (1, 2):
                    for held in range(len(d['stack'])):
                        identity = dict(row_px=r, col_px=c, scale=scale, spatial_degree=degree, held_frame=held)
                        if not np.isfinite(departure) or departure > cfg.max_plane_departure_m:
                            fit = dict(status='terrain_plane_limit')
                        else:
                            fit = held_out_prediction(d['stack'][sl], d['visibility'][sl], d['azimuths'], d['elevations'],
                                ex.noise, sc, ex.rc, replace(cfg, spatial_degree=degree), slopes, held_frame=held)
                        rows.append(dict(**identity, **{k: v for k, v in fit.items() if k != 'held_frame'}))
            ex.live.update(force=True, kind='prediction', fit=None, message='Predicting withheld illumination',
                           prediction=dict(row_px=r, col_px=c, trials=len(rows), last_status=rows[-1]['status']))
            print(f'Withheld illumination: {len(rows)} declared position/scale/frame/null trials', flush=True)
    save(ex.out/'predictions.json', rows)
    summary = []
    for degree in (1, 2):
        for scale in cfg.scale_factors:
            subset = [r for r in rows if r['spatial_degree'] == degree and r['scale'] == scale]
            assessed = [r for r in subset if r['status'] == 'assessed']
            wrong = [r for r in assessed if r['correct_advantage_over_wrong'] is not None]
            summary.append(dict(spatial_degree=degree, scale=scale, trials=len(subset), assessed=len(assessed),
                median_correct_advantage_over_static=float(np.median([r['correct_advantage_over_static'] for r in assessed])) if assessed else None,
                median_correct_advantage_over_wrong=float(np.median([r['correct_advantage_over_wrong'] for r in wrong])) if wrong else None,
                correct_better_than_static=sum(r['correct_advantage_over_static'] > 0 for r in assessed),
                correct_better_than_wrong=sum(r['correct_advantage_over_wrong'] > 0 for r in wrong), wrong_assessed=len(wrong)))
    return ex.result('PARTIAL', 'Fixed-window withheld-frame prediction measured without held-intensity leakage; these correlated within-scene tests are not independent annotated-scene validation.',
                     predictions=summary, trials=len(rows), selection='fixed lattice; all frames/scales/null families declared in advance',
                     limitations=['Common validity masks may use all frames; intensities, covariance reference and object fit use training frames only.',
                                  'Withheld spatial illumination coefficients are nuisance-fitted equally for every competing prediction.',
                                  'Wrong direction rotates only the withheld prediction by 90 degrees with the training object frozen.',
                                  'No best-scale selection or fitted success threshold is made from these results.'])


# ---------------------------------------------------------------------- T16
# Worker state for forked processes; set only for the duration of t16().
_T16 = {}


def _t16_run(job):
    from src.hati_core.relief_scenes import render_relief, relief_feature
    ex, cfg, size, root = _T16['ex'], _T16['cfg'], _T16['size'], _T16['root']
    rocks = [make_rock(job['seed'], root, job['height_m'], .6, aspect=1.35)] if job['height_m'] is not None else []
    relief = job.get('relief')
    features = [relief_feature(relief[0], relief[1], relief[2], seed=job['seed'])] if relief else []
    generated = render_relief((size, size), ex.data['azimuths'], ex.data['elevations'], pixel_m=ex.sc.pixel_m,
                              seed=job['seed'], noise=job['render_noise'], features=features, rocks=rocks,
                              supersample=ex.cfg.get('relief_supersample', 4), structured_null=job['kind'] == 'structured_null')
    folder = ex.out/job['folder']; folder.mkdir(parents=True, exist_ok=True)
    truth = dict(kind=job['kind'], seed=job['seed'], height_m=job['height_m'], relief=relief,
                 width_m=.6 if job['height_m'] is not None else None, root_px=root, slope_rc=(0., 0.),
                 noise_pass=job['noise_pass'], render_noise=job['render_noise'], model_noise=float(ex.noise),
                 generator_truth=generated['truth'], scene_scope='predeclared central ROI; full-frame baseline also recorded')
    row, _ = _control_trial(ex, cfg, generated['stack'], (0., 0.), folder, truth, predict=False, save_arrays=job['example'])
    return row


def summarise_null_trials(rows, gate):
    """Per noise pass and scenario: adaptive warnings and context-supported outcomes."""
    groups = {}
    for r in rows:
        groups.setdefault((r['noise_pass'], r['kind'], r['height_m']), []).append(r)
    out = []
    for (name, kind, height), sample in groups.items():
        supported = sum(r['context_supported_cells'] > 0 for r in sample)
        entry = dict(noise_pass=name, kind=kind, height_m=height, trials=len(sample),
                     render_noise=sample[0]['render_noise'], model_noise=sample[0]['model_noise'],
                     trials_with_adaptive_warning=sum(r['adaptive_warning_cells'] > 0 for r in sample),
                     trials_with_context_supported=supported,
                     context_supported_cells=sum(r['context_supported_cells'] for r in sample),
                     roi_assessed_cells=sum(r['roi_assessed_cells'] for r in sample),
                     requested_cells=sum(r['requested'] for r in sample),
                     unresolved_cells=sum(r['unresolved_cells'] for r in sample),
                     recovered_within_2px=sum(r['recovered_within_2px'] is True for r in sample) if height is not None else None)
        if height is None:
            fraction = supported/len(sample) if sample else None
            entry.update(context_supported_trial_fraction=fraction, declared_gate_max_fraction=gate,
                         within_declared_gate=bool(fraction is not None and fraction <= gate))
        out.append(entry)
    return out


def _plot_t16(ex, summaries, gate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys(s['noise_pass'] for s in summaries))
    fig, axes = plt.subplots(1, len(names), figsize=(6.5*len(names), 4.2), squeeze=False, layout='constrained')
    for ax, name in zip(axes[0], names):
        rows = [s for s in summaries if s['noise_pass'] == name]
        labels = [s['kind'] if s['height_m'] is None else f'{s["kind"]} {s["height_m"]} m' for s in rows]
        values = [s['trials_with_context_supported']/s['trials'] for s in rows]
        ax.set_axisbelow(True)
        ax.barh(range(len(rows)), values, color='#1f3a5f', height=.6)
        for i, s in enumerate(rows):
            ax.text(min(values[i]+.02, .9), i, f'{s["trials_with_context_supported"]}/{s["trials"]}', va='center', fontsize=9, color='#13283b')
        ax.axvline(gate, color='#c1121f', ls='--', lw=1.2, label=f'declared gate for no-caster scenes ({gate:g})')
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.set_xlim(0, 1)
        ax.set_xlabel('trials with a context-supported cell')
        ax.set_title(f'{name.replace("_", " ")}: rendered sigma {rows[0]["render_noise"]:.4f}, model sigma {rows[0]["model_noise"]:.4f}', fontsize=10)
        ax.legend(frameon=False, loc='lower right', fontsize=8)
    fig.suptitle('HATI T16 | no-caster scenes through the full adaptive procedure | synthetic ROI diagnostic')
    fig.savefig(ex.out/'adaptive_nulls.png', dpi=140); plt.close(fig)


def t16(ex):
    """No-caster scenes through the complete adaptive procedure, with enough seeds for a rate."""
    import os
    cfg = adaptive_config(ex)
    seeds = ex.cfg.get('null_seeds', 12)
    heights = ex.cfg.get('null_caster_heights_m', [.3, .6])
    gate = ex.cfg.get('null_gate_max_fraction', .1)
    mode = ex.cfg.get('null_render_noise', 'measured')
    if mode == 'measured':
        t12 = ex.args.campaign/'stages/T12/result.json'
        measured = json.loads(t12.read_text(encoding='utf-8')) if t12.exists() else {}
        # Relief is rendered explicitly below, so the noise to draw is what is left once
        # Sun-consistent shading is removed; older T12 results fall back to the raw residual.
        key = 'relief_corrected_sigma' if measured.get('relief_corrected_sigma') else \
            'measured_pooled_sigma_quadratic' if cfg.spatial_degree == 2 else 'measured_pooled_sigma'
        render = measured.get(key) or measured.get('measured_pooled_sigma')
        if not render:
            return ex.result('BLOCKED', 'null_render_noise is "measured" but this campaign has no usable T12 residual scale. '
                             'Run T12 first, or set null_render_noise to "assumed" or a number.')
        source = f'T12 {key}'
    elif mode == 'assumed':
        render, source = float(ex.noise), 'assumed model sigma'
    else:
        render, source = float(mode), 'configured value'
    passes = [('render_assumed', float(ex.noise))]
    if not np.isclose(render, ex.noise):
        passes.append(('render_measured', float(render)))
    size = 2*ex.sc.radius_px*max(cfg.scale_factors)+4*ex.rc.cell_px+1
    root = [size//2+.3, size//2+.2]
    # Sun-consistent relief with no caster: the null the athena residual points to (T12, T13).
    relief_scenes = [tuple(s) for s in ex.cfg.get('null_relief_scenes', [['ripples', 6., 2.], ['mound', 6., 4.]])]
    scenarios = [('static', None, None), ('structured_null', None, None)]
    scenarios += [(f'{k}_{slope:g}deg', None, (k, float(size_m), float(slope))) for k, size_m, slope in relief_scenes]
    scenarios += [('procedural', float(h), None) for h in heights]
    jobs = []
    for name, noise in passes:
        for case, (kind, height, relief) in enumerate(scenarios):
            for trial in range(seeds):
                # Disjoint from T10's seeds, and shared across noise passes, so
                # both passes see the same scenes with the same draws, rescaled.
                jobs.append(dict(noise_pass=name, render_noise=noise, kind=kind, height_m=height, relief=relief,
                                 seed=ex.cfg['seed']+200000+100*case+trial, example=trial == 0,
                                 folder=f'{name}/{case:03d}_{kind}_{trial:02d}'))
    workers = cfg.workers if os.name != 'nt' else 1
    rows = []
    def report(row, done):
        ex.live.update(force=True, kind='controls', fit=None, message='T16: no-caster scenes through the adaptive procedure',
                       control=dict(kind=row['kind'], height_m=row['height_m'], seed=row['seed'], status=row['central_status'],
                                    recovered=row['recovered_within_2px'], maximum_score=None))
        print(f'T16 {done}/{len(jobs)}: {row["noise_pass"]} {row["kind"]} h={row["height_m"]}; '
              f'context-supported cells {row["context_supported_cells"]}, adaptive warnings {row["adaptive_warning_cells"]}', flush=True)
    _T16.update(ex=ex, cfg=cfg, size=size, root=root)
    try:
        if workers > 1:
            import multiprocessing as mp
            with mp.get_context('fork').Pool(workers) as pool:
                for done, row in enumerate(pool.imap(_t16_run, jobs), 1):
                    rows.append(row); report(row, done)
        else:
            for done, job in enumerate(jobs, 1):
                row = _t16_run(job); rows.append(row); report(row, done)
    finally:
        _T16.clear()
    save(ex.out/'null_trials.json', rows)
    summaries = summarise_null_trials(rows, gate)
    save(ex.out/'summary.json', summaries)
    _plot_t16(ex, summaries, gate)
    nulls = [s for s in summaries if s['height_m'] is None]
    return ex.result('PARTIAL', 'No-caster scenes run through the complete adaptive request, expansion and stopping procedure; '
                     'context-supported rates are reported against a declared research gate, not a calibrated false-alarm rate.',
                     render_noise_source=source, model_sigma=float(ex.noise),
                     noise_passes=[dict(name=n, render_noise=v, model_noise=float(ex.noise)) for n, v in passes],
                     seeds_per_scenario=seeds, summaries=summaries, declared_gate_max_fraction=gate,
                     null_scenarios=len(nulls), null_scenarios_within_gate=sum(s['within_declared_gate'] for s in nulls),
                     relief_scenes=[dict(kind=k, size_m=s, max_slope_deg=sl) for k, s, sl in relief_scenes],
                     generator='relief_heightfield_horizon_lunar_lambert_v1',
                     limitations=['Synthetic 3x3-cell ROIs, not full-image false-alarm calibration.',
                                  'The changing background is a drifting stripe pattern unrelated to Sun geometry; the relief '
                                  'scenes are Sun-consistent mounds and ripples from the independent generator.',
                                  'Relief slopes and sizes are declared, not measured lunar distributions.'])
