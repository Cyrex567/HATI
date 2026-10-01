"""T13 and T14 workers: relief against compact casters, and shape from shading as a null.

T13 renders mounds, bowls and elephant-hide ripples with the independent relief
generator, runs the unchanged regional detector on them, then compares rock and
relief hypotheses by withheld-frame prediction with margins calibrated on
simulated scenes, and applies the calibrated rule to a declared sample of
Athena cells. T14 fits a linearised multi-image shape-from-shading surface to
the whole stack, reruns the unchanged detector on the relief-corrected stack and
measures, by injecting rendered rocks into the real images, how much caster
signal the correction removes.
"""
import json
from pathlib import Path
import time

import numpy as np

from landing_maps import clean_json, write_tif
from src.hati_core.adaptive_shadow import cell_table
from src.hati_core.regional_shadow import assess_regions
from src.hati_core.relief_hypothesis import (CLASSES, SIGNS, calibrate_margins, classify, compare_models, confusion,
                                             relief_sign, sign_confusion)
from src.hati_core.relief_scenes import render_relief, relief_feature
from src.hati_core.rock_population import describe as describe_population, prior_from_config, sample_population
from src.hati_core.rock_scenes import load_catalog, make_rock

CLASS_COLOURS = {'rock_like': '#b64262', 'relief_like': '#64b9a5', 'ambiguous': '#e2a441', 'none': '#80909d'}


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def _t12(ex):
    path = ex.args.campaign/'stages/T12/result.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def _noise_passes(ex):
    """The assumed sigma, plus T12's relief-corrected scale when this campaign measured it."""
    t12 = _t12(ex)
    passes = [('assumed', float(ex.noise))]
    relief = t12.get('relief_corrected_sigma')
    if relief and not np.isclose(relief, ex.noise):
        passes.append(('relief_corrected', float(relief)))
    return passes, t12


def _slope_limit(ex):
    return float(ex.run.get('landing', {}).get('slope_limit_deg', ex.cfg.get('relief_slope_limit_deg', 8.)))


def _solver_progress(ex, what, every=2.):
    """HATI Watch message for a running shape-from-shading solve; display only."""
    last = [-float('inf')]
    def report(info):
        if time.monotonic()-last[0] < every:
            return
        last[0] = time.monotonic()
        ex.live.update(force=True, kind='stage', solver=info,
                       message=f'T14: {what} · pass {info["pass_index"]} of {info["passes"]} · '
                               f'iteration {info["iteration"]:,} of at most {info["iterations"]:,}')
    return report


def _progress(ex, message, done, total, started, last):
    if time.monotonic()-last[0] > 15 or done == total:
        rate = (time.monotonic()-started)/max(done, 1)
        print(f'{message}: {done}/{total} ({rate*(total-done)/60:.1f} min left)', flush=True)
        ex.live.update(force=True, kind='stage', message=f'{message}: {done}/{total}')
        last[0] = time.monotonic()


# ---------------------------------------------------------------------- T13

def _detector_on_relief(ex, passes):
    """Part A: the unchanged detector on relief-only scenes, with rocks as positives."""
    cfg, d = ex.cfg, ex.data
    size = cfg.get('relief_scene_px', 64); ss = cfg.get('relief_supersample', 4)
    centre = ((size-1)/2+.3, (size-1)/2+.2)
    scenes = [(k, float(s), float(sl), None) for k in cfg.get('relief_kinds', ['mound', 'bowl', 'ripples'])
              for s in cfg.get('relief_sizes_m', [3., 6., 12.]) for sl in cfg.get('relief_slopes_deg', [1., 2., 5., 10.])]
    scenes += [('rock', None, None, float(h)) for h in cfg.get('relief_rock_heights_m', [.3, .6, 1.2])]
    seeds = cfg.get('relief_seeds', 3)
    rows, total, started, last = [], len(passes)*len(scenes)*seeds, time.monotonic(), [0.]
    for name, sigma in passes:
        for case, (kind, size_m, slope, height) in enumerate(scenes):
            for trial in range(seeds):
                # Shared across noise passes: the same scenes with the same draws, rescaled.
                seed = cfg['seed']+300000+100*case+trial
                if kind == 'rock':
                    out = render_relief((size, size), d['azimuths'], d['elevations'], pixel_m=ex.sc.pixel_m, seed=seed,
                                        noise=sigma, rocks=[make_rock(seed, centre, height, .6, aspect=1.35)], supersample=ss)
                else:
                    out = render_relief((size, size), d['azimuths'], d['elevations'], pixel_m=ex.sc.pixel_m, seed=seed,
                                        noise=sigma, features=[relief_feature(kind, size_m, slope, seed=seed)], supersample=ss)
                # The detector assumes the noise actually rendered, so any response is the relief's.
                fit = assess_regions(out['stack'], d['azimuths'], d['elevations'], sigma, ex.sc, ex.rc)
                ok = fit['status'] == 1; score = fit['score'][ok]
                distance = [np.hypot(r['row_px']-centre[0], r['col_px']-centre[1]) for r in fit['candidates']]
                reach = size_m/2/ex.sc.pixel_m+3 if kind in ('mound', 'bowl') else np.inf
                rows.append(dict(noise_pass=name, sigma=sigma, kind=kind, size_m=size_m, max_slope_deg=slope,
                                 height_m=height, seed=seed, assessed_pixels=int(ok.sum()),
                                 exceedance_fraction=float(np.mean(score >= ex.rc.score_scale)) if score.size else None,
                                 maximum_score=float(score.max()) if score.size else None,
                                 warning_roots=len(distance), warning_roots_on_feature=int(sum(v <= reach for v in distance)),
                                 recovered_within_2px=bool(any(v <= 2 for v in distance)) if kind == 'rock' else None,
                                 true_max_slope_deg=float(out['slope_deg'].max())))
                _progress(ex, 'T13 detector on relief scenes', len(rows), total, started, last)
    return rows


def _summarise_detector(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['noise_pass'], r['kind'], r['size_m'], r['max_slope_deg'], r['height_m']), []).append(r)
    out = []
    for (name, kind, size_m, slope, height), sample in groups.items():
        scores = [r['maximum_score'] for r in sample if r['maximum_score'] is not None]
        exceed = [r['exceedance_fraction'] for r in sample if r['exceedance_fraction'] is not None]
        out.append(dict(noise_pass=name, sigma=sample[0]['sigma'], kind=kind, size_m=size_m, max_slope_deg=slope, height_m=height,
                        trials=len(sample), trials_with_warning=sum(r['warning_roots'] > 0 for r in sample),
                        recovered_within_2px=sum(bool(r['recovered_within_2px']) for r in sample) if kind == 'rock' else None,
                        median_maximum_score=float(np.median(scores)) if scores else None,
                        median_exceedance_fraction=float(np.median(exceed)) if exceed else None))
    return out


def _competition_scenes(ex, sigma, seed_offset, seeds, scales):
    """Part B scenes at the detector's patch size: rocks on flat and rippled ground, relief, blanks, stripes."""
    cfg, d = ex.cfg, ex.data
    r = ex.sc.radius_px; size = 2*r+1; centre = (r+.3, r+.2); ss = cfg.get('relief_supersample', 4)
    background = cfg.get('relief_background_slope_deg') or _t12(ex).get('relief_background_slope_deg') or 2.
    wavelength = cfg.get('relief_background_wavelength_m', 6.)
    variants = [('rock', dict(height_m=float(h), background_slope_deg=b))
                for h in cfg.get('relief_rock_heights_m', [.3, .6, 1.2]) for b in (None, float(background))]
    variants += [('relief', dict(kind=k, size_m=float(s), max_slope_deg=float(sl)))
                 for k in cfg.get('relief_kinds', ['mound', 'bowl', 'ripples'])
                 for s in cfg.get('relief_competition_sizes_m', [3., 6., 12.])
                 for sl in cfg.get('relief_competition_slopes_deg', [1., 2., 5.])]
    variants += [('none', {}), ('stripes', {})]
    # Blank scenes set the noise floor of the margins and stripes set the Sun margin,
    # so both get their own, larger count.
    blanks = cfg.get('relief_blank_scenes', 24)
    rows = []
    for case, (truth, v) in enumerate(variants):
        for trial in range(blanks if truth in ('none', 'stripes') else seeds):
            seed = cfg['seed']+seed_offset+100*case+trial
            features, rocks = [], []
            if truth == 'rock':
                rocks = [make_rock(seed, centre, v['height_m'], .6, aspect=1.35)]
                if v['background_slope_deg']:
                    features = [relief_feature('ripples', wavelength, v['background_slope_deg'], seed=seed)]
            elif truth == 'relief':
                features = [relief_feature(v['kind'], v['size_m'], v['max_slope_deg'], seed=seed)]
            out = render_relief((size, size), d['azimuths'], d['elevations'], pixel_m=ex.sc.pixel_m, seed=seed, noise=sigma,
                                features=features, rocks=rocks, supersample=ss, structured_null=truth == 'stripes')
            result = compare_models(out['stack'], np.ones_like(out['stack'], bool), d['azimuths'], d['elevations'],
                                    sigma, ex.sc, ex.rc, scales)
            rows.append(dict(truth=truth, seed=seed, **v, **result))
    return rows


def _athena_cells(ex, sigma, margins, scales):
    """Part C: the calibrated rule on declared Athena cells from the T1 baseline."""
    cfg, d = ex.cfg, ex.data
    b = ex.baseline()
    radius = ex.sc.radius_px
    touchdown = np.array([ex.run['counterfactual']['row_px'], ex.run['counterfactual']['col_px']], float)
    near_px = cfg.get('relief_touchdown_radius_px', 24)
    assessed = [row for row in b['cell_table'] if b['status'][row[4], row[5]] == 1]
    near = [row for row in assessed if np.hypot(row[4]-touchdown[0], row[5]-touchdown[1]) <= near_px]
    others = [row for row in assessed if np.hypot(row[4]-touchdown[0], row[5]-touchdown[1]) > near_px]
    count = cfg.get('relief_athena_cells', 2000)
    rng = np.random.default_rng(cfg['seed']+600000)
    if count and count < len(others):
        others = [others[i] for i in sorted(rng.choice(len(others), count, replace=False))]
    valid = (np.nan_to_num(np.asarray(d['visibility'], float), nan=0.) >= .99) & np.isfinite(d['stack'])
    limit = _slope_limit(ex)
    cells = [(row, True) for row in near]+[(row, False) for row in others]
    rows, started, last = [], time.monotonic(), [0.]
    for (r0, r1, c0, c1, cr, cc), is_near in cells:
        sl = np.s_[:, cr-radius:cr+radius+1, cc-radius:cc+radius+1]
        slopes = (float(d['slope_row'][cr, cc]), float(d['slope_col'][cr, cc]))
        entry = dict(row_px=int(cr), col_px=int(cc), near_touchdown=is_near, baseline_score=float(b['score'][cr, cc]),
                     distance_to_touchdown_px=float(np.hypot(cr-touchdown[0], cc-touchdown[1])))
        if not np.isfinite(slopes).all():
            entry.update(status='missing_terrain', label=None)
        else:
            result = compare_models(d['stack'][sl], valid[sl], d['azimuths'], d['elevations'], sigma, ex.sc, ex.rc, scales, slopes)
            entry.update(result)
            entry['label'] = classify(result, margins)
            entry['relief_sign'] = relief_sign(result, margins) if entry['label'] == 'relief_like' else None
            if entry['label'] == 'relief_like':
                ratio = result['relief_slope_lower_bound_deg']/limit
                entry['relief_slope_index_lower_bound'] = ratio/(1+ratio)
        entry.pop('frames', None)
        rows.append(entry)
        _progress(ex, 'T13 Athena cells', len(rows), len(cells), started, last)
    return rows, touchdown


def _class_fractions(rows):
    labelled = [r['label'] for r in rows if r.get('label')]
    signs = [r['relief_sign'] for r in rows if r.get('label') == 'relief_like' and r.get('relief_sign')]
    return dict(cells=len(labelled), **{c: (labelled.count(c)/len(labelled) if labelled else None) for c in CLASSES},
                relief_signs={s: (signs.count(s)/len(signs) if signs else None) for s in SIGNS})


def _plot_t13(ex, detector, matrix, athena, touchdown, margins, signs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    passes = list(dict.fromkeys(s['noise_pass'] for s in detector))
    fig = plt.figure(figsize=(7.2*len(passes)+14, 10), layout='constrained')
    grid = fig.add_gridspec(2, len(passes)+2)
    for i, name in enumerate(passes):
        ax = fig.add_subplot(grid[:, i])
        relief = [s for s in detector if s['noise_pass'] == name and s['kind'] != 'rock']
        labels = list(dict.fromkeys(f'{s["kind"]} {s["size_m"]:g} m' for s in relief))
        slopes = sorted({s['max_slope_deg'] for s in relief})
        value = np.full((len(labels), len(slopes)), np.nan); text = {}
        for s in relief:
            i_row, i_col = labels.index(f'{s["kind"]} {s["size_m"]:g} m'), slopes.index(s['max_slope_deg'])
            value[i_row, i_col] = s['trials_with_warning']/s['trials']
            text[i_row, i_col] = f'{s["trials_with_warning"]}/{s["trials"]}\n{s["median_maximum_score"]:.0f}' if s['median_maximum_score'] is not None else ''
        im = ax.imshow(value, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        for (a, b), t in text.items():
            ax.text(b, a, t, ha='center', va='center', fontsize=9, color='#13283b' if value[a, b] < .6 else 'white')
        ax.set_xticks(range(len(slopes))); ax.set_xticklabels([f'{v:g} deg' for v in slopes])
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        rocks = [s for s in detector if s['noise_pass'] == name and s['kind'] == 'rock']
        rock_text = ', '.join(f'{s["height_m"]:g} m {s["recovered_within_2px"]}/{s["trials"]}' for s in rocks)
        ax.set_title(f'Unchanged detector on relief, sigma {relief[0]["sigma"]:.4f}\n'
                     f'cell: scenes with warnings / median max score\nrocks recovered: {rock_text}', fontsize=9)
        ax.set_xlabel('maximum slope of the feature')
        fig.colorbar(im, ax=ax, shrink=.6, label='share of scenes with warning roots')
    ax = fig.add_subplot(grid[0, len(passes)])
    truths = [t for t in ('rock', 'relief', 'none', 'stripes') if t in matrix]
    table = np.array([[matrix[t][c] for c in CLASSES] for t in truths], float)
    share = table/np.maximum(table.sum(axis=1, keepdims=True), 1)
    ax.imshow(share, cmap='Blues', vmin=0, vmax=1)
    for a in range(len(truths)):
        for b in range(len(CLASSES)):
            ax.text(b, a, f'{int(table[a, b])}', ha='center', va='center', fontsize=9, color='#13283b' if share[a, b] < .6 else 'white')
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels([c.replace('_', ' ') for c in CLASSES])
    ax.set_yticks(range(len(truths))); ax.set_yticklabels([f'true {t}' for t in truths])
    sign_text = '; '.join(f'{kind}s: ' + ', '.join(f'{v} {s}' for s, v in row.items()) for kind, row in signs.items())
    ax.set_title('Held-out simulated scenes, calibrated margins' + (f'\nrelief-like {sign_text}' if sign_text else ''), fontsize=9)
    ax = fig.add_subplot(grid[1, len(passes)])
    groups = [('all sampled', athena), ('baseline warning', [r for r in athena if r['baseline_score'] >= ex.rc.score_scale]),
              ('near touchdown', [r for r in athena if r['near_touchdown']])]
    left = np.zeros(len(groups))
    for c in CLASSES:
        widths = np.array([_class_fractions(g)[c] or 0 for _, g in groups])
        ax.barh(range(len(groups)), widths, left=left, color=CLASS_COLOURS[c], label=c.replace('_', ' '),
                edgecolor='white', linewidth=2, height=.6)
        left += widths
    ax.set_yticks(range(len(groups))); ax.set_yticklabels([f'{g} ({_class_fractions(r)["cells"]})' for g, r in groups])
    ax.set_xlim(0, 1); ax.invert_yaxis(); ax.set_xlabel('share of Athena cells'); ax.legend(frameon=False, ncols=2, fontsize=8)
    shares = _class_fractions(athena)['relief_signs']
    ax.set_title('Athena cells by class' + ('' if shares['protrusion'] is None else
                 f'\nrelief-like cells: {shares["protrusion"]:.0%} protrusion, {shares["depression"]:.0%} depression, '
                 f'{shares["undetermined"]:.0%} undetermined'), fontsize=10)
    ax = fig.add_subplot(grid[:, len(passes)+1])
    mean = np.nanmean(ex.data['stack'], axis=0)
    ax.imshow(mean, cmap='gray', vmin=np.nanpercentile(mean, 1), vmax=np.nanpercentile(mean, 99))
    for c in CLASSES:
        pts = np.array([[r['col_px'], r['row_px']] for r in athena if r.get('label') == c]).reshape(-1, 2)
        ax.scatter(pts[:, 0], pts[:, 1], s=16, color=CLASS_COLOURS[c], label=c.replace('_', ' '),
                   edgecolors='white', linewidths=.3)
    if touchdown is not None:
        ax.plot(touchdown[1], touchdown[0], marker='+', color='white', ms=18, mew=2.5)
    ax.set_title('Sampled Athena cells (+ touchdown)' if athena else 'Athena cells not classified (no T1 baseline)', fontsize=10)
    if athena:
        # Below the image: labels drawn over the lunar surface are unreadable on dark craters.
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.06), ncols=4, frameon=False, fontsize=9, markerscale=1.6)
    fig.suptitle('HATI T13 | compact, extended and depression models compete | research diagnostic, not a hazard map')
    fig.savefig(ex.out/'relief_competition.png', dpi=130); plt.close(fig)


def t13(ex):
    """Compete compact, extended and depression models (report section 13)."""
    cfg = ex.cfg
    scales = tuple(cfg.get('relief_scales_m', [.9, 1.8, 3.6]))
    passes, t12 = _noise_passes(ex)
    sigma = passes[-1][1]
    source = 'T12 relief-corrected residual scale' if passes[-1][0] == 'relief_corrected' else 'assumed model sigma (no T12 relief result)'
    ex.live.update(force=True, kind='stage', message='T13: the unchanged detector on relief scenes')
    detector = _detector_on_relief(ex, passes)
    save(ex.out/'detector_on_relief.json', detector)
    detector_summary = _summarise_detector(detector)
    seeds = cfg.get('relief_calibration_seeds', 6)
    ex.live.update(force=True, kind='stage', message='T13: calibrating rock versus relief margins')
    calibration = _competition_scenes(ex, sigma, 400000, seeds, scales)
    evaluation = _competition_scenes(ex, sigma, 500000, seeds, scales)
    targets = dict(rock_called_relief=cfg.get('relief_target_rock_called_relief', .05),
                   relief_called_rock=cfg.get('relief_target_relief_called_rock', .10),
                   blank_called_signal=cfg.get('relief_target_blank_called_signal', .10),
                   stripes_called_relief=cfg.get('relief_target_stripes_called_relief', .10),
                   sign_error=cfg.get('relief_target_sign_error', .10))
    margins = calibrate_margins(calibration, targets)
    matrix = confusion(evaluation, margins)
    signs = sign_confusion(evaluation, margins)
    save(ex.out/'competition_scenes.json', dict(calibration=calibration, evaluation=evaluation))
    athena, touchdown, blocked = [], None, None
    if (ex.baseline_dir/'regional.npz').exists():
        ex.live.update(force=True, kind='stage', message='T13: classifying Athena cells')
        athena, touchdown = _athena_cells(ex, sigma, margins, scales)
        save(ex.out/'athena_cells.json', athena)
    else:
        blocked = 'T1 baseline unavailable in this campaign; Athena cells were not classified.'
    summary = dict(all_sampled=_class_fractions(athena),
                   baseline_warning=_class_fractions([r for r in athena if r['baseline_score'] >= ex.rc.score_scale]),
                   near_touchdown=_class_fractions([r for r in athena if r['near_touchdown']]))
    nearest = min(athena, key=lambda r: r['distance_to_touchdown_px']) if athena else None
    _plot_t13(ex, detector_summary, matrix, athena, touchdown, margins, signs)
    relief_rows = [s for s in detector_summary if s['kind'] != 'rock']
    return ex.result('PARTIAL', 'Relief scenes through the unchanged detector; rock and relief hypotheses compared by withheld-frame '
                     'prediction with margins calibrated on simulated scenes. Classes are research labels, not hazard decisions.',
                     noise_passes=[dict(name=n, sigma=s) for n, s in passes], competition_sigma=sigma, competition_sigma_source=source,
                     detector_on_relief=detector_summary,
                     relief_scenes_with_warnings=dict(scenes=sum(s['trials'] for s in relief_rows),
                                                      with_warnings=sum(s['trials_with_warning'] for s in relief_rows)),
                     relief_scales_m=list(scales), targets=targets, margins=margins,
                     evaluation_confusion=matrix, evaluation_sign_confusion=signs, athena_summary=summary,
                     touchdown_cell=nearest and {k: nearest.get(k) for k in ('row_px', 'col_px', 'label', 'relief_sign', 'baseline_score',
                                                                             'gain_rock', 'gain_relief', 'gain_mound', 'gain_bowl',
                                                                             'sun_margin', 'relief_slope_lower_bound_deg',
                                                                             'distance_to_touchdown_px')},
                     slope_limit_deg=_slope_limit(ex), athena_blocked=blocked,
                     limitations=['Relief slopes come from linearised shading and are lower bounds above the Sun elevation.',
                                  'Margins are calibrated on generated scenes at one noise level; they are not lunar error rates.',
                                  'A relief-like cell is handed to the terrain module as a slope hazard, never cleared.',
                                  'The sample of Athena cells is declared by seed, not chosen from scores.'])


# ---------------------------------------------------------------------- T14

def _injection_sites(common, count, spacing, margin, touchdown, seed, offset=0):
    """Sites at least spacing apart on a half-spacing grid; offset shifts the grid for another round."""
    rng = np.random.default_rng(seed)
    H, W = common.shape
    candidates = [(r, c) for r in range(margin+offset, H-margin, spacing//2) for c in range(margin+offset, W-margin, spacing//2)
                  if common[r-8:r+9, c-8:c+9].all() and np.hypot(r-touchdown[0], c-touchdown[1]) > spacing]
    rng.shuffle(candidates)
    chosen = []
    for r, c in candidates:
        if all(np.hypot(r-a, c-b) >= spacing for a, b in chosen):
            chosen.append((r, c))
        if len(chosen) == count:
            break
    return chosen


def _t13_rule(ex):
    """T13's calibrated margins and the sigma they were calibrated at, if this campaign ran T13."""
    path = ex.args.campaign/'stages/T13/result.json'
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding='utf-8'))
    return (d['margins'], d['competition_sigma'], tuple(d.get('relief_scales_m', [.9, 1.8, 3.6]))) if d.get('margins') else None


def _size_casters(ex, stack, cells, sigma, terrain=None, phase=None, relief=None, touchdown=None):
    """Adaptive height and width refinement (T9's machinery) at the given cells of a stack.

    terrain, the shape-from-shading surface, replaces the planar receiving surface
    so larger windows need not be refused for relief. Returns one record per
    cell. A shadow that runs past the fitting window only
    bounds the height from below. Where even the shortest compatible fit of the
    widest warning pass is censored, the bound is geometric: the height whose
    shadow at the lowest Sun elevation just reaches the window edge. Otherwise it
    is the lower end of the compatible height range, which injected rocks show can
    overshoot. Deciding censoring from the best fit alone let a template that was
    too long turn the window size into a lower bound the data never established. A height estimate is
    reported only when the context expansion reached stable, endpoint-supported
    dimensions. With a phase name, every pass and finished cell also goes to HATI
    Watch; relief maps cell centres to T13 labels for that display.
    """
    from dataclasses import replace
    from src.hati_core.adaptive_shadow import AdaptiveConfig, refine_regions
    # sfs_sizing_adaptive overrides the shared adaptive settings for T14's sizing only (T9 keeps its own).
    cfg = AdaptiveConfig(**{**ex.cfg.get('adaptive', {}), **ex.cfg.get('sfs_sizing_adaptive', {})})
    shape = stack.shape[1:]
    chosen = np.zeros(shape, bool)
    for r0, r1, c0, c1, *_ in cells:
        chosen[r0:r1, c0:c1] = True
    # A queue of exactly these cells: plan_regions requests baseline warnings only.
    baseline = dict(status=chosen.astype(np.uint8), score=np.where(chosen, cfg.warning_score+1., 0.),
                    endpoint_censored=np.zeros(shape), endpoint_missing_count=np.zeros(shape))
    clearance = ex.cfg.get('sfs_clearance_m', .3)
    records, watch = [], {}
    if phase is not None:
        # HATI Watch follows every pass and every finished cell; display only.
        label = f'T14 sizing {phase}'
        ex.live.sizing_start(phase, len(cells), image_shape=shape, touchdown=touchdown, pixel_m=ex.sc.pixel_m,
                             clearance_m=clearance)
        watch = dict(observer=lambda info: ex.live.adaptive(info, subrun=label),
                     progress=lambda info: ex.live.adaptive_progress(info, label=label))
    def collect(record):
        records.append(record)
        if phase is not None:
            row = _caster_row(ex, cfg, record, clearance)
            ex.live.sizing_row(row, relief=(relief or {}).get((row['row_px'], row['col_px'])))
    refine_regions(stack, ex.data['visibility'], ex.data['azimuths'], ex.data['elevations'], sigma, ex.sc, ex.rc,
                   replace(cfg, max_cells=0), baseline, ex.data['slope_row'], ex.data['slope_col'], on_record=collect,
                   terrain=terrain, **watch)
    return [_caster_row(ex, cfg, record, clearance) for record in records]


def _caster_row(ex, cfg, record, clearance):
    """T14's record for one sized cell: warning scales, height lower bound and, where supported, a height.

    A window the context guard rejected (context_conflict) does not count: its larger
    context contradicted where a smaller window saw the shadow end.
    """
    warned = [h for h in record['history'] if h.get('status') == 'assessed' and h['best']['score'] >= cfg.warning_score
              and not h.get('context_conflict')]
    row = dict(row_px=int(record['centre'][0]), col_px=int(record['centre'][1]), state=record['status'],
               scales_warning=[h['scale'] for h in warned], height_lower_bound_m=None, height_m=None)
    if warned:
        last = warned[-1]
        lowest = last.get('lowest_compatible')
        censored = bool(last['lowest_compatible_censored'] if 'lowest_compatible_censored' in last else last['endpoint_censored'])
        offset = (lowest or last['best'])['root_offset']
        reach_m = (last['support_px']-float(np.hypot(*offset)))*ex.sc.pixel_m
        geometric = reach_m*float(np.tan(np.radians(np.min(np.asarray(ex.data['elevations'])[last['frames']]))))
        bound = geometric if censored else float(last['height_range_m'][0])
        row.update(score=float(last['best']['score']), width_m=float(last['best']['width_m']), censored=censored,
                   height_lower_bound_m=bound, compatible_range_m=[float(v) for v in last['height_range_m']],
                   exceeds_clearance=bool(bound >= clearance))
        final = record.get('final')
        if record['status'] == 'context_supported_unvalidated' and final:
            row.update(height_m=float(final['best']['height_m']), height_range_m=[float(v) for v in final['height_range_m']])
    return row


def _cell_containing(table, r, c):
    for row in table:
        if row[0] <= r < row[1] and row[2] <= c < row[3]:
            return row
    return None


def _root_score(fit, root, radius=2.):
    evidence = fit['root_evidence']
    if not len(evidence):
        return None
    near = evidence[np.hypot(evidence[:, 0]-root[0], evidence[:, 1]-root[1]) <= radius]
    return float(near[:, 2].max()) if len(near) else None


def _height_groups(cfg, heights):
    """(label, low, high) for each group the planted rocks are summarised in.

    'fixed' planting groups by its configured heights (low == high); 'population'
    planting draws continuous heights and groups them by planted_height_bins_m.
    """
    if cfg.get('planted_geometry', 'population') == 'fixed':
        return [(f'{h:g} m', float(h), float(h)) for h in heights]
    bins = [float(b) for b in cfg.get('planted_height_bins_m', [.15, .3, .6, 1.2, 2.])]
    return [(f'{lo:g} to {hi:g} m', lo, hi) for lo, hi in zip(bins[:-1], bins[1:])]


def _group_of(height, groups):
    """The group a planted height belongs to; the last bin includes its upper edge."""
    for j, (label, lo, hi) in enumerate(groups):
        if (lo == hi and abs(height-lo) < 1e-9) or lo <= height < hi or (j == len(groups)-1 and lo < hi and height == hi):
            return label
    return None


def _calibration_for(height, groups, recovery):
    """How often planted rocks of this height were found in this run (corrected stack, measured noise).

    Fixed planting uses the tallest planted height not above it (a smaller rock, so the
    share is conservative); population planting uses the bin holding it, and the top bin
    for anything taller. Heights below the planted range have no calibration.
    """
    if height is None or not groups:
        return None
    if groups[0][1] == groups[0][2]:
        below = [g for g in groups if g[1] <= height+1e-9]
        label = below[-1][0] if below else None
    else:
        label = _group_of(height, groups) or (groups[-1][0] if height >= groups[-1][2] else None)
    found = recovery.get(label, {}).get('corrected_measured') if label else None
    return dict(group=label, found=found['recovered'], of=found['quiet_sites']) if found else None


def _conformal_margin(bounds, truths, coverage=.9, kind='offset'):
    """Split-conformal correction that makes height lower bounds hold for rocks like the planted ones.

    kind 'offset' takes the excess bound - true height (metres) and the result is subtracted from a
    bound; 'ratio' takes bound / true height and a bound is divided by it. It is the
    ceil((n+1) * coverage)-th smallest excess over n planted rocks, never below the identity
    (0 m or 1), so for a new rock exchangeable with them the corrected bound holds with probability
    at least coverage. None when n is too small for that coverage.
    """
    if kind not in ('offset', 'ratio'):
        raise ValueError('kind must be offset or ratio')
    excess = np.asarray(bounds, float)-np.asarray(truths, float) if kind == 'offset' else np.asarray(bounds, float)/np.asarray(truths, float)
    n = len(excess); k = int(np.ceil((n+1)*coverage))
    if not n or k > n:
        return None
    return max(0. if kind == 'offset' else 1., float(np.sort(excess)[k-1]))


def _corrected_bound(bound, margin, kind='offset'):
    if bound is None or margin is None:
        return None
    return max(0., bound-margin) if kind == 'offset' else bound/margin


def _exact_interval(k, n, confidence=.95):
    """Clopper-Pearson interval for k successes in n trials; None for no trials."""
    from scipy.stats import beta
    if not n:
        return None
    a = 1-confidence
    return [0. if k == 0 else float(beta.ppf(a/2, k, n-k+1)), 1. if k == n else float(beta.ppf(1-a/2, k+1, n-k))]


def _measurable(groups, injection, detection_target=.9, coverage_target=.9, confidence=.95):
    """The planted heights this run can vouch for, with exact intervals.

    For each height group: the share of planted rocks found (relief-corrected stack, measured
    noise, quiet sites) and the share of height lower bounds that hold (bound at most the rock's
    own height plus 5 cm), each with a Clopper-Pearson interval. A group is established when the
    lower end of both intervals reaches its target. measurable_from_m is the lower edge of the
    lowest group from which every taller group is established; None when even the tallest is not.
    Where T14 calibrated the bounds, the calibrated (cross-fitted) ones are judged; the fitted
    ones are counted alongside.
    """
    calibrated = any(r.get('sized_height_lower_bound_calibrated_m') is not None for r in injection)
    key = 'sized_height_lower_bound_calibrated_m' if calibrated else 'sized_height_lower_bound_m'
    out, ordered = {}, []
    for label, lo, hi in groups:
        rows = [r for r in injection if _group_of(r['height_m'], groups) == label]
        found = [r['recovered_corrected_measured'] for r in rows if r.get('recovered_corrected_measured') is not None]
        bounded = [r for r in rows if r.get(key) is not None]
        holds = sum(r[key] <= r['height_m']+.05 for r in bounded)
        fitted = [r for r in rows if r.get('sized_height_lower_bound_m') is not None]
        fitted_holds = sum(r['sized_height_lower_bound_m'] <= r['height_m']+.05 for r in fitted)
        errors = [r['sized_height_m']-r['height_m'] for r in rows if r.get('sized_height_m') is not None]
        found_interval, holds_interval = _exact_interval(sum(found), len(found), confidence), _exact_interval(holds, len(bounded), confidence)
        established = bool(found_interval and holds_interval and found_interval[0] >= detection_target
                           and holds_interval[0] >= coverage_target)
        out[label] = dict(height_range_m=[lo, hi], planted=len(rows), found=int(sum(found)), quiet_sites=len(found),
                          found_interval=found_interval, bounded=len(bounded), bound_holds=int(holds), holds_interval=holds_interval,
                          fitted_bounded=len(fitted), fitted_bound_holds=int(fitted_holds),
                          estimates=len(errors), median_error_m=float(np.median(errors)) if errors else None,
                          max_abs_error_m=float(np.max(np.abs(errors))) if errors else None, established=established)
        ordered.append((lo, established))
    measurable_from = None
    for lo, established in sorted(ordered, reverse=True):
        if not established:
            break
        measurable_from = lo
    return dict(confidence=confidence, detection_target=detection_target, coverage_target=coverage_target,
                bounds_judged='calibrated, cross-fitted' if calibrated else 'as fitted',
                groups=out, measurable_from_m=measurable_from,
                note='A height group is established when, at this confidence, at least the target share of its planted rocks '
                     'is found and at least the target share of their height lower bounds hold. Planted rocks only: the '
                     'statement covers rocks like them on this stack, not independent truth.')


def _group_detections(casters, cell_px, groups, recovery):
    """Real detections as candidate objects: sized warning cells that touch, merged.

    One rock can set off neighbouring cells, so cells whose centres lie within one cell
    step of each other (8-connected on the cell grid) form one candidate object, which may
    still hold several rocks. Its height bound and estimate are the largest of its cells.
    Its label is rock-like if any cell is, else ambiguous; unchecked where T13 did not run.
    Each object carries the planted-rock calibration for its height from this run.
    """
    if not casters:
        return []
    index = {(round(c['row_px']/cell_px), round(c['col_px']/cell_px)): i for i, c in enumerate(casters)}
    parent = list(range(len(casters)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for (gr, gc), i in index.items():
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                j = index.get((gr+dr, gc+dc))
                if j is not None:
                    parent[find(i)] = find(j)
    clusters = {}
    for i, c in enumerate(casters):
        clusters.setdefault(find(i), []).append(c)
    objects = []
    for cells in clusters.values():
        bounds = [c['height_lower_bound_m'] for c in cells if c.get('height_lower_bound_m') is not None]
        estimates = [c['height_m'] for c in cells if c.get('height_m') is not None]
        labels = [c.get('relief_check') for c in cells]
        peak = max(cells, key=lambda c: c.get('score') or 0.)
        bound, estimate = (max(bounds) if bounds else None), (max(estimates) if estimates else None)
        corrected = [c['height_lower_bound_calibrated_m'] for c in cells if c.get('height_lower_bound_calibrated_m') is not None]
        objects.append(dict(
            height_lower_bound_calibrated_m=max(corrected) if corrected else None,
            exceeds_clearance_calibrated=any(bool(c.get('exceeds_clearance_calibrated')) for c in cells),
            cells=len(cells), row_px=float(np.mean([c['row_px'] for c in cells])), col_px=float(np.mean([c['col_px'] for c in cells])),
            peak_row_px=int(peak['row_px']), peak_col_px=int(peak['col_px']), peak_score=peak.get('score'),
            height_lower_bound_m=bound, height_m=estimate,
            label='rock_like' if 'rock_like' in labels else 'ambiguous' if 'ambiguous' in labels else 'unchecked',
            distance_to_touchdown_m=float(min(c['distance_to_touchdown_m'] for c in cells)),
            exceeds_clearance=any(bool(c.get('exceeds_clearance')) for c in cells),
            calibration=_calibration_for(estimate if estimate is not None else bound, groups, recovery)))
    objects.sort(key=lambda o: (o['height_lower_bound_m'] is None, -(o['height_lower_bound_m'] or 0.), o['distance_to_touchdown_m']))
    for k, o in enumerate(objects):
        o['object'] = k
    return objects


PLANTED_GEOMETRY = ('width_m', 'length_m', 'height_over_diameter', 'burial', 'yaw_deg', 'shape')

RECOVERY = {'original_assumed': ('original stack, assumed sigma', '#80909d'),
            'corrected_assumed': ('relief-corrected, assumed sigma', '#5b7fa6'),
            'corrected_measured': ('relief-corrected, sigma measured after correction', '#1f3a5f')}


def _plot_t14(ex, solved, before, after, injection, sigma_after, touchdown, groups=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    fig, axes = plt.subplots(2, 3, figsize=(19, 11), layout='constrained')
    h = solved['height_m']
    ax = axes[0, 0]
    shade = LightSource(azdeg=315, altdeg=35).hillshade(np.nan_to_num(h), vert_exag=40, dx=ex.sc.pixel_m, dy=ex.sc.pixel_m)
    ax.imshow(np.where(solved['common'], shade, np.nan), cmap='gray')
    ax.set_title('Shape-from-shading surface, hillshade (x40 vertical)')
    ax = axes[0, 1]
    im = ax.imshow(solved['slope_deg'], cmap='magma', vmin=0, vmax=np.nanpercentile(solved['slope_deg'], 99))
    fig.colorbar(im, ax=ax, shrink=.75, label='slope (degrees), lower bound above the Sun elevation')
    ax.set_title('Metre-scale slope from shading')
    ax = axes[0, 2]
    values = solved['slope_deg'][np.isfinite(solved['slope_deg'])]
    ax.set_axisbelow(True)
    ax.hist(values, bins=60, color='#1f3a5f')
    ax.axvline(_slope_limit(ex), color='#c1121f', ls='--', lw=1.2, label=f'terrain slope limit {_slope_limit(ex):g} deg')
    ax.set_xlabel('slope (degrees)'); ax.set_ylabel('pixels'); ax.legend(frameon=False)
    ax.set_title(f'Slope distribution (median {np.median(values):.1f}, p90 {np.percentile(values, 90):.1f} deg)')
    score_max = max(np.nanpercentile(before['score'], 99) if before is not None else 0, np.nanpercentile(after['score'], 99))
    for ax, fit, title in ((axes[1, 0], before, 'T1 baseline score (assumed sigma)'),
                           (axes[1, 1], after, 'Score after removing relief shading (assumed sigma)')):
        if fit is None:
            ax.axis('off'); continue
        s = np.where(fit['status'] == 1, fit['score'], np.nan)
        im = ax.imshow(s, cmap=plt.get_cmap('magma').with_extremes(bad='#80909d'), vmin=0, vmax=score_max)
        ax.plot(touchdown[1], touchdown[0], marker='+', color='white', ms=16, mew=2.5)
        exceed = float(np.mean(s[np.isfinite(s)] >= ex.rc.score_scale))
        ax.set_title(f'{title}\n{100*exceed:.1f}% of assessed pixels at or above {ex.rc.score_scale:g} (+ touchdown)')
        fig.colorbar(im, ax=ax, shrink=.75, label='score')
    ax = axes[1, 2]
    groups = groups or [(f'{h:g} m', h, h) for h in sorted({r['height_m'] for r in injection})]
    x = np.arange(len(groups)); width = .27
    ax.set_axisbelow(True)
    for j, (key, (label, colour)) in enumerate(RECOVERY.items()):
        clean = [[r['recovered_'+key] for r in injection if _group_of(r['height_m'], groups) == g[0] and r['recovered_'+key] is not None]
                 for g in groups]
        share = [np.mean(c) if c else 0. for c in clean]
        ax.bar(x+(j-1)*width, share, width=width-.03, color=colour, label=label)
        for xi, c, v in zip(x+(j-1)*width, clean, share):
            ax.text(xi, v+.02, f'{sum(c)}/{len(c)}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f'{g[0]} rock' for g in groups], fontsize=8); ax.set_ylim(0, 1.45)
    ax.set_ylabel('injected rocks recovered within 2 px'); ax.legend(frameon=False, loc='upper left', fontsize=8)
    ax.set_title(f'Rocks injected into the real images\n(quiet sites only; measured sigma {sigma_after:.3f})')
    fig.suptitle(f'HATI T14 | shape from shading as a structural null | residual scale after relief {sigma_after:.4f} | research diagnostic')
    fig.savefig(ex.out/'shape_from_shading.png', dpi=130); plt.close(fig)


def t14(ex):
    """Shape-from-shading as a structural null (report section 13)."""
    from rasterio.crs import CRS as RasterCRS
    from src.hati_core.noise_scale import NoiseScaleConfig, measure_residual_scale
    from src.hati_core.sfs import rock_factor, shadow_reach_px, solve_sfs, solve_sfs_nonlinear
    cfg, d = ex.cfg, ex.data
    valid = (np.nan_to_num(np.asarray(d['visibility'], float), nan=0.) >= .99) & np.isfinite(d['stack'])
    options = dict(grid_px=cfg.get('sfs_grid_px', 2), smoothness=cfg.get('sfs_smoothness', 3.),
                   dark_ratio=cfg.get('sfs_dark_ratio', .5), shadow_sigma=cfg.get('sfs_shadow_sigma', 3.))
    if cfg.get('sfs_model', 'linear') == 'nonlinear':
        solver = solve_sfs_nonlinear
        options.update(iterations=cfg.get('sfs_iterations_per_step', 800), gauss_newton=cfg.get('sfs_gauss_newton', 6))
    else:
        solver = solve_sfs
        options.update(iterations=cfg.get('sfs_iterations', 1500), passes=cfg.get('sfs_passes', 1))
    ex.live.update(force=True, kind='stage', message='T14: solving shape from shading')
    def watch(what):
        return dict(progress=_solver_progress(ex, what)) if solver is solve_sfs else {}
    started = time.monotonic()
    solved = solver(d['stack'], valid, d['azimuths'], d['elevations'], ex.sc.pixel_m, **options,
                    **watch('solving shape from shading'))
    print(f'T14 shape from shading: explained {solved["explained_fraction"]:.3f} of the frame-to-frame ratio variance '
          f'in {time.monotonic()-started:.0f}s ({solved["lsqr_iterations"]} iterations)', flush=True)
    crs = RasterCRS.from_wkt(ex.crs.to_wkt())
    write_tif(ex.out/'sfs_height_m.tif', solved['height_m'], ex.transform, crs,
              'Relative height from linearised multi-image shape from shading; mean and shared planes unobserved')
    write_tif(ex.out/'sfs_slope_deg.tif', solved['slope_deg'], ex.transform, crs,
              'Slope from shape from shading; a lower bound where slopes exceed the Sun elevation')
    np.savez_compressed(ex.out/'sfs.npz', height_m=solved['height_m'].astype('float32'),
                        slope_deg=solved['slope_deg'].astype('float32'), common=solved['common'])
    ex.live.field('Shape-from-shading slope (degrees)', solved['slope_deg'], explained=round(solved['explained_fraction'], 3))
    # The DEM receiving slopes are an independent, coarser measurement of the same surface.
    from scipy import ndimage as ndi
    smooth = [ndi.gaussian_filter(np.nan_to_num(np.gradient(np.nan_to_num(solved['height_m']), ex.sc.pixel_m)[i]), 2.) for i in (0, 1)]
    check = solved['common'] & np.isfinite(d['slope_row']) & np.isfinite(d['slope_col'])
    dem = dict(pixels=int(check.sum()))
    for name, mine, theirs in (('row', smooth[0], d['slope_row']), ('col', smooth[1], d['slope_col'])):
        a, b = mine[check], np.asarray(theirs)[check]
        a, b = a-a.mean(), b-b.mean()
        denominator = np.sqrt((a*a).sum()*(b*b).sum())
        dem[f'correlation_{name}'] = float((a*b).sum()/denominator) if check.sum() > 10 and denominator > 0 else None
    noise_cfg = NoiseScaleConfig(**cfg.get('noise_scale', {}))
    after_scale = measure_residual_scale(solved['corrected'], d['visibility'], d['slope_row'], d['slope_col'], noise_cfg)
    sigma_after = after_scale['pooled_sigma'] or float(ex.noise)
    ex.live.update(force=True, kind='stage', message='T14: detector on the relief-corrected stack')
    corrected = ex.regional('relief_corrected', data=dict(d, stack=solved['corrected']),
                            input_proof=dict(source=ex.proof, relief_correction=dict(stage='T14', **solved['configuration'])))
    before = ex.baseline() if (ex.baseline_dir/'regional.npz').exists() else None
    ok = corrected['status'] == 1; s = corrected['score'][ok]
    rescaled = s*float(ex.noise)/sigma_after
    touchdown = (int(ex.run['counterfactual']['row_px']), int(ex.run['counterfactual']['col_px']))
    exceedance = dict(after_assumed_sigma=float(np.mean(s >= ex.rc.score_scale)) if s.size else None,
                      after_measured_sigma=float(np.mean(rescaled >= ex.rc.score_scale)) if s.size else None,
                      median_score_after=float(np.median(s)) if s.size else None,
                      touchdown_score_after=float(corrected['score'][touchdown]) if ok[touchdown] else None)
    if before is not None:
        b_ok = before['status'] == 1
        exceedance.update(before_assumed_sigma=float(np.mean(before['score'][b_ok] >= ex.rc.score_scale)),
                          median_score_before=float(np.median(before['score'][b_ok])),
                          touchdown_score_before=float(before['score'][touchdown]) if b_ok[touchdown] else None)
    # Injection: rendered rocks multiplied into the real images, then the same correction and detector.
    ex.live.update(force=True, kind='stage', message='T14: injecting rendered rocks into the real images')
    heights = cfg.get('sfs_injection_heights_m', [.3, .6, 1.2])
    rounds = cfg.get('sfs_injection_rounds', 1)
    if type(rounds) is not int or rounds < 1:
        raise ValueError('sfs_injection_rounds must be a positive integer')
    # 'population' gives every planted rock its own height, proportions, burial, yaw and body (NASA Apollo
    # meshes from the catalog's development split, or procedural); 'fixed' plants the 2.5 bodies at fixed heights.
    if cfg.get('planted_geometry', 'population') == 'population':
        prior = prior_from_config(cfg)
        catalog = getattr(ex.args, 'rock_catalog', None)
        meshes = load_catalog(catalog, split='development') if catalog else []
        tallest = prior.height_m[1]
    else:
        prior, meshes, tallest = None, [], max(heights)
    groups = _height_groups(cfg, heights)
    requested, spacing = cfg.get('sfs_injection_sites', 24), cfg.get('sfs_injection_spacing_px', 40)
    if cfg.get('sfs_injection_window', 'auto') == 'auto':
        # Drawn in full, the tallest rock's shadow at the lowest Sun must not reach the next site's scoring window.
        # Level ground sets the spacing; sites where the ground falls along the shadow are checked one by one below.
        # Both use the Sun's centre (the umbra): the fainter penumbra beyond it stays inside the 12 px margin.
        reach_px = shadow_reach_px(tallest, (0., 0.), d['azimuths'], d['elevations'], ex.sc.pixel_m, solar_radius_deg=0.)
        spacing = max(spacing, int(np.ceil(reach_px))+ex.sc.radius_px+12)
    injection, round_log = [], []
    table = cell_table(d['stack'].shape[1:], ex.sc, ex.rc)
    half = ex.sc.radius_px+12
    for k in range(rounds):
        # Each round shifts the site grid by a fraction of its step, so the rounds sample different ground;
        # the first round is the single-round grid.
        offset = int(round(k*spacing/(2*rounds)))
        sites = _injection_sites(solved['common'], requested, spacing, max(ex.sc.radius_px+20, 36), touchdown,
                                 cfg['seed']+700000+1000*k, offset=offset)
        specs = sample_population(len(sites), cfg['seed']+720000+1000*k, prior, meshes) if prior else None
        placed = [(r+.3, c+.2, specs[i]['height_m'] if specs else heights[i % len(heights)]) for i, (r, c) in enumerate(sites)]
        round_log.append(dict(round=k, grid_offset_px=offset, spacing_px=spacing, requested=requested, placed=len(placed)))
        if not placed:
            continue
        tag = f' (round {k+1} of {rounds})' if rounds > 1 else ''
        ex.live.update(force=True, kind='stage', message=f'T14: rendering {len(placed)} planted rocks into the real images{tag}')
        # Each rock stands on the DEM plane the sizing assumes at its site, and its shadow is drawn in full;
        # 'flat' and a fixed window reproduce the 2.5 planting, whose rocks read too tall on rising ground.
        plane, window_setting = cfg.get('sfs_injection_receiving_plane', 'dem'), cfg.get('sfs_injection_window', 'auto')
        slopes = [tuple(float(np.nan_to_num(f[int(r), int(c)])) for f in (d['slope_row'], d['slope_col'])) for r, c, _ in placed] \
            if plane == 'dem' else None
        round_log[-1].update(receiving_plane=plane, window_px=window_setting)
        if slopes is not None and window_setting == 'auto':
            # Ground falling along a shadow lengthens it past the level-ground reach the spacing allows for;
            # such a rock could darken its neighbour's scoring window, so its site is left empty.
            allowance = spacing-ex.sc.radius_px-12
            keep = [j for j, (_, _, h) in enumerate(placed)
                    if shadow_reach_px(h, slopes[j], d['azimuths'], d['elevations'], ex.sc.pixel_m, solar_radius_deg=0.) <= allowance]
            round_log[-1].update(dropped_long_shadow=len(placed)-len(keep), placed=len(keep))
            placed, slopes = [placed[j] for j in keep], [slopes[j] for j in keep]
            specs = [specs[j] for j in keep] if specs else None
            if not placed:
                continue
        windows = []
        factor = rock_factor(d['stack'].shape[1:], placed, d['azimuths'], d['elevations'], ex.sc.pixel_m,
                             seed=cfg['seed']+710000+1000*k, supersample=cfg.get('relief_supersample', 4),
                             window_px=window_setting, slopes=slopes, info=windows, rocks=specs, meshes=meshes)
        round_log[-1].update(max_shadow_reach_px=max(w['reach_px'] for w in windows),
                             shadows_clipped=sum(w['shadow_clipped'] for w in windows))
        injected = d['stack']*factor
        ex.live.update(force=True, kind='stage', message=f'T14: solving shape from shading with the planted rocks{tag}')
        solved_injected = solver(injected, valid, d['azimuths'], d['elevations'], ex.sc.pixel_m, **options,
                                 **watch(f'solving shape from shading with the planted rocks{tag}'))
        # The rock alone: images with it minus images without it, before and after the correction.
        # Background removal cancels in both, so their ratio is the rock signal the correction keeps.
        rock_before, rock_after = injected-d['stack'], solved_injected['corrected']-solved['corrected']
        last = [0.]; first = len(injection)
        for i, (r, c, height) in enumerate(placed):
            window = np.s_[int(r)-half:int(r)+half+1, int(c)-half:int(c)+half+1]
            root = (r-(int(r)-half), c-(int(c)-half))
            def score(stack):
                fit = assess_regions(stack[(slice(None), *window)], d['azimuths'], d['elevations'], ex.noise, ex.sc, ex.rc,
                                     visible=np.asarray(d['visibility'])[(slice(None), *window)],
                                     slope_row=d['slope_row'][window], slope_col=d['slope_col'][window])
                return _root_score(fit, root)
            scores = dict(original=score(d['stack']), injected=score(injected),
                          corrected=score(solved['corrected']), corrected_injected=score(solved_injected['corrected']),
                          rock_only_before=score(rock_before), rock_only_after=score(rock_after))
            def recovered(base, test, factor=1.):
                # Scores scale exactly as 1/sigma, so another noise level is a rescaling.
                if base is None or test is None or base*factor >= ex.rc.score_scale:
                    return None                          # background already warning: recovery is undefined
                return bool(test*factor >= ex.rc.score_scale)
            def energy(a):
                a = a[(slice(None), *window)]
                return float(np.sum(np.where(np.isfinite(a), a, 0.)**2))
            e_before, e_after = energy(rock_before), energy(rock_after)
            change = np.abs(solved_injected['height_m']-solved['height_m'])[window]
            injection.append(dict(round=k, site=len(injection), row_px=r, col_px=c, height_m=height,
                                  **({key: specs[i][key] for key in PLANTED_GEOMETRY} if specs else {}),
                                  **{f'score_{key}': v for key, v in scores.items()},
                                  rock_signal_kept=(scores['rock_only_after']/scores['rock_only_before']
                                                    if scores['rock_only_after'] is not None and scores['rock_only_before'] else None),
                                  rock_amplitude_kept=float(np.sqrt(e_after/e_before)) if e_before > 0 else None,
                                  recovered_original_assumed=recovered(scores['original'], scores['injected']),
                                  recovered_corrected_assumed=recovered(scores['corrected'], scores['corrected_injected']),
                                  recovered_corrected_measured=recovered(scores['corrected'], scores['corrected_injected'],
                                                                         float(ex.noise)/sigma_after),
                                  surface_change_near_rock_m=float(np.nanmax(change[half-4:half+5, half-4:half+5]))))
            _progress(ex, f'T14 injected sites{tag}', i+1, len(placed), started, last)
        # The same sizing on rocks of known height, in the relief-corrected injected images.
        ex.live.update(force=True, kind='stage', message=f'T14: sizing the injected rocks{tag}')
        site_cells = [_cell_containing(table, int(r), int(c)) for r, c, _ in placed]
        by_cell = {(s['row_px'], s['col_px']): s for s in
                   _size_casters(ex, solved_injected['corrected'], [row for row in site_cells if row is not None], sigma_after,
                                 terrain=solved_injected['height_m'], phase=f'planted rocks{tag}', touchdown=touchdown)}
        for row, cell in zip(injection[first:], site_cells):
            size = by_cell.get((int(cell[4]), int(cell[5]))) if cell is not None else None
            row.update(sized_state=size and size['state'], sized_height_m=size and size['height_m'],
                       sized_height_lower_bound_m=size and size['height_lower_bound_m'])
    # Calibrated lower bounds (split conformal): the planted rocks' excess of bound over true height sets a margin
    # that makes bounds hold for at least the target share of rocks like them. Planted rocks are cross-fitted, each
    # half corrected with the other half's margin, so their coverage is measured out of sample; real casters get the
    # margin of all planted rocks.
    kind, coverage_target = cfg.get('bound_calibration', 'offset'), cfg.get('measurable_coverage_target', .9)
    bounded_planted = [r for r in injection if r.get('sized_height_lower_bound_m') is not None]
    def margin_of(rows):
        return _conformal_margin([r['sized_height_lower_bound_m'] for r in rows], [r['height_m'] for r in rows], coverage_target, kind)
    bound_margin, cross = None, []
    if kind != 'none':
        halves = (bounded_planted[0::2], bounded_planted[1::2])
        for mine, other in ((halves[0], halves[1]), (halves[1], halves[0])):
            m = margin_of(other); cross.append(m)
            for r in mine:
                r['sized_height_lower_bound_calibrated_m'] = _corrected_bound(r['sized_height_lower_bound_m'], m, kind)
        bound_margin = margin_of(bounded_planted)
    calibration = dict(kind=kind, coverage=coverage_target, planted_bounds=len(bounded_planted), margin=bound_margin,
                       cross_fitted_margins=cross,
                       unit='m subtracted from a bound' if kind == 'offset' else 'factor dividing a bound' if kind == 'ratio' else None,
                       note='Split conformal: for a rock exchangeable with the planted ones on this stack, a corrected bound holds '
                            'with at least the stated probability. Real rocks differ from planted ones in ways this cannot check.')
    save(ex.out/'injection.json', injection)
    # Summaries per height group: the planted heights themselves ('fixed') or height bins ('population').
    members = {label: [r for r in injection if _group_of(r['height_m'], groups) == label] for label, _, _ in groups}
    absorption, recovery, sizing_check = {}, {}, {}
    for label, lo, hi in groups:
        rows = members[label]
        kept = [r['rock_signal_kept'] for r in rows if r['rock_signal_kept'] is not None]
        amplitude = [r['rock_amplitude_kept'] for r in rows if r['rock_amplitude_kept'] is not None]
        absorption[label] = dict(
            height_range_m=[lo, hi], sites=len(kept), median_signal_kept=float(np.median(kept)) if kept else None,
            p25_p75_signal_kept=[float(v) for v in np.percentile(kept, [25, 75])] if kept else None,
            share_under_half=float(np.mean(np.asarray(kept) < .5)) if kept else None,
            median_amplitude_kept=float(np.median(amplitude)) if amplitude else None)
        for key in RECOVERY:
            clean = [r['recovered_'+key] for r in rows if r['recovered_'+key] is not None]
            recovery.setdefault(label, {})[key] = dict(recovered=int(sum(clean)), quiet_sites=len(clean))
        bounded = [r for r in rows if r.get('sized_height_lower_bound_m') is not None]
        estimated = [r for r in rows if r.get('sized_height_m') is not None]
        # Each rock is checked against its own true height, so mixed heights in one bin stay exact.
        sizing_check[label] = dict(
            height_range_m=[lo, hi], sites=len(rows), with_warning_evidence=len(bounded), sized=len(estimated),
            lower_bound_holds=sum(r['sized_height_lower_bound_m'] <= r['height_m']+.05 for r in bounded),
            lower_bound_median_m=float(np.median([r['sized_height_lower_bound_m'] for r in bounded])) if bounded else None,
            estimate_median_m=float(np.median([r['sized_height_m'] for r in estimated])) if estimated else None,
            estimate_median_error_m=float(np.median([r['sized_height_m']-r['height_m'] for r in estimated])) if estimated else None)
    measurable = _measurable(groups, injection, cfg.get('measurable_detection_target', .9), cfg.get('measurable_coverage_target', .9))
    small = [v['median_signal_kept'] for v in absorption.values() if v['height_range_m'][1] <= .6 and v['median_signal_kept'] is not None]
    absorption_verdict = dict(
        small_rock_signal_mostly_absorbed=bool(small) and max(small) < .5,
        rule='If under half of the rock-only score survives for 0.3-0.6 m rocks, stop subtracting relief from the '
             'data alone; model it as a nuisance both hypotheses see (the RegistrationProjector rule), so the '
             'sensitivity layers count how much of each rock looks like relief.')
    # Sub-pixel casters: warning cells that survive the relief correction at the residual scale
    # measured after it, checked against relief with T13's rule where available, then sized.
    ex.live.update(force=True, kind='stage', message='T14: sizing sub-pixel casters')
    scale = float(ex.noise)/sigma_after
    candidates = [row for row in corrected['cell_table'] if corrected['status'][row[4], row[5]] == 1
                  and corrected['score'][row[4], row[5]]*scale >= ex.rc.score_scale]
    examined = candidates
    cap = cfg.get('sfs_sizing_cells', 1500)
    if cap and len(candidates) > cap:
        rng = np.random.default_rng(cfg['seed']+800000)
        examined = [candidates[i] for i in sorted(rng.choice(len(candidates), cap, replace=False))]
    rule = _t13_rule(ex); labels = {}
    if rule:
        margins, sigma13, scales = rule; radius = ex.sc.radius_px; last = [0.]; started = time.monotonic()
        ex.live.sizing_start('relief check', len(examined), image_shape=d['stack'].shape[1:], touchdown=touchdown,
                             pixel_m=ex.sc.pixel_m, clearance_m=cfg.get('sfs_clearance_m', .3))
        for i, (r0, r1, c0, c1, cr, cc) in enumerate(examined):
            sl = np.s_[:, cr-radius:cr+radius+1, cc-radius:cc+radius+1]
            slopes_rc = (float(d['slope_row'][cr, cc]), float(d['slope_col'][cr, cc]))
            if np.isfinite(slopes_rc).all():
                labels[(int(cr), int(cc))] = classify(compare_models(solved['corrected'][sl], valid[sl], d['azimuths'], d['elevations'],
                                                                     sigma13, ex.sc, ex.rc, scales, slopes_rc), margins)
            ex.live.sizing_relief(cr, cc, labels.get((int(cr), int(cc))))
            _progress(ex, 'T14 relief check on caster candidates', i+1, len(examined), started, last)
    # Relief-like cells go to the terrain module; 'none' means no model predicts the withheld frames.
    keep = [row for row in examined if not rule or labels.get((int(row[4]), int(row[5]))) in ('rock_like', 'ambiguous')]
    casters = _size_casters(ex, solved['corrected'], keep, sigma_after, terrain=solved['height_m'], phase='detections',
                            relief=labels, touchdown=touchdown) if keep else []
    for row in casters:
        row['relief_check'] = labels.get((row['row_px'], row['col_px']))
        row['distance_to_touchdown_m'] = float(np.hypot(row['row_px']-touchdown[0], row['col_px']-touchdown[1])*ex.sc.pixel_m)
        calibrated = _corrected_bound(row['height_lower_bound_m'], bound_margin, kind)
        row.update(height_lower_bound_calibrated_m=calibrated,
                   exceeds_clearance_calibrated=bool(calibrated is not None and calibrated >= cfg.get('sfs_clearance_m', .3)))
    save(ex.out/'subpixel_casters.json', casters)
    clearance = cfg.get('sfs_clearance_m', .3)
    area_ha = float((corrected['status'] == 1).sum())*ex.sc.pixel_m**2/1e4
    exceeding = [c for c in casters if c.get('exceeds_clearance')]
    bounds = [c['height_lower_bound_m'] for c in casters if c['height_lower_bound_m'] is not None]
    estimates = [c['height_m'] for c in casters if c['height_m'] is not None]
    nearest = min(exceeding, key=lambda c: c['distance_to_touchdown_m']) if exceeding else None
    subpixel = dict(candidate_cells=len(candidates), examined_cells=len(examined), sigma=sigma_after,
                    relief_check='T13 calibrated margins on the relief-corrected stack' if rule else None,
                    relief_check_classes={c: sum(v == c for v in labels.values()) for c in CLASSES} if rule else None,
                    sized_cells=len(casters), with_warning_evidence=len(bounds), context_supported=len(estimates),
                    clearance_m=clearance, exceeding_clearance=len(exceeding),
                    exceeding_clearance_cells_per_ha=(len(exceeding)*len(candidates)/max(len(examined), 1)/area_ha) if area_ha else None,
                    height_lower_bound_m=dict(zip(('p10', 'median', 'p90'), np.percentile(bounds, [10, 50, 90]).tolist())) if bounds else None,
                    height_estimate_m=dict(zip(('p10', 'median', 'p90'), np.percentile(estimates, [10, 50, 90]).tolist())) if estimates else None,
                    nearest_exceeding_clearance=nearest and {k: nearest.get(k) for k in ('row_px', 'col_px', 'distance_to_touchdown_m',
                                                                                          'height_lower_bound_m', 'height_m', 'state')},
                    injected_rock_sizing=sizing_check)
    # The real rocks HATI found: touching sized cells merged into candidate objects, each with the
    # planted-rock calibration for its height from this run.
    objects = _group_detections(casters, ex.rc.cell_px, groups, recovery)
    save(ex.out/'real_rocks.json', objects)
    def height_of(o):
        return o['height_m'] if o['height_m'] is not None else o['height_lower_bound_m']
    real = dict(objects=len(objects), cells=len(casters), examined_share=len(examined)/max(len(candidates), 1),
                pixel_m=float(ex.sc.pixel_m), touchdown_px=list(touchdown), image_shape=list(d['stack'].shape[1:]),
                by_label={k: sum(o['label'] == k for o in objects) for k in ('rock_like', 'ambiguous', 'unchecked')},
                with_height=sum(height_of(o) is not None for o in objects), with_estimate=sum(o['height_m'] is not None for o in objects),
                at_or_above_clearance=sum(o['exceeds_clearance'] for o in objects),
                at_or_above_clearance_calibrated=sum(o['exceeds_clearance_calibrated'] for o in objects),
                by_height_group={label: sum(_calibration_for(height_of(o), groups, recovery) is not None
                                            and _calibration_for(height_of(o), groups, recovery)['group'] == label for o in objects)
                                 for label, _, _ in groups},
                below_calibrated_range=sum(height_of(o) is not None and _calibration_for(height_of(o), groups, recovery) is None
                                           for o in objects),
                nearest_at_or_above_clearance=min((o for o in objects if o['exceeds_clearance']),
                                                  key=lambda o: o['distance_to_touchdown_m'], default=None),
                note='Candidate objects are touching sized warning cells, merged; one may hold several rocks. Heights are '
                     'context-supported estimates where available, otherwise lower bounds. The calibration is the share of '
                     'planted rocks of that height found in this run, after the relief correction at the measured noise.')
    population = dict(geometry='population', planted=len(injection), **describe_population(prior, meshes),
                      shapes={s: sum(r.get('shape') == s for r in injection) for s in sorted({r.get('shape') for r in injection})},
                      height_over_diameter_median=float(np.median([r['height_over_diameter'] for r in injection])) if injection else None) \
        if prior else dict(geometry='fixed', heights_m=list(heights), planted=len(injection),
                           body='0.6 m wide procedural body, aspect 1.35, burial 0.1')
    _plot_t14(ex, solved, before, corrected, injection, sigma_after, touchdown, groups)
    _plot_subpixel(ex, casters, injection, touchdown, clearance, sigma_after)
    slopes = solved['slope_deg'][np.isfinite(solved['slope_deg'])]
    limit = _slope_limit(ex)
    return ex.result('PARTIAL', 'Linearised shape-from-shading surface used as a structural null; the unchanged detector reruns on the '
                     'relief-corrected stack and injected rocks measure what the correction removes. Not a validated DEM.',
                     sfs=dict(explained_fraction=solved['explained_fraction'], ratio_rms_before=solved['ratio_rms_before'],
                              ratio_rms_after=solved['ratio_rms_after'], used_fraction_per_frame=solved['used_fraction_per_frame'],
                              lsqr_iterations=solved['lsqr_iterations'], lsqr_stop=solved['lsqr_stop'], configuration=solved['configuration']),
                     slope_deg=dict(median=float(np.median(slopes)), p90=float(np.percentile(slopes, 90)),
                                    p99=float(np.percentile(slopes, 99)), fraction_above_limit=float(np.mean(slopes > limit)),
                                    limit=limit, at_touchdown=float(solved['slope_deg'][touchdown]) if np.isfinite(solved['slope_deg'][touchdown]) else None),
                     dem_slope_agreement=dem, residual_scale_after=dict(pooled_sigma=after_scale['pooled_sigma'],
                                                                         per_frame_sigma=after_scale['per_frame_sigma'],
                                                                         structure=after_scale['structure'], patches=after_scale['patches']),
                     exceedance=exceedance, injection_recovery=recovery, injected_sites=len(injection), injection_rounds=round_log,
                     relief_absorption=absorption, relief_absorption_verdict=absorption_verdict, subpixel_casters=subpixel,
                     planted_population=population, real_rocks=real, measurable=measurable, bound_calibration=calibration,
                     limitations=['First-order shading: slopes above the Sun elevation are underestimated and cast shadows are excluded, not modelled.',
                                  'The surface is relative; its mean and planes shared by every frame are unobserved.',
                                  'Recovery is counted only where the corrected background was quiet at the site.',
                                  'Injected rocks are rendered on the DEM plane at their site (or flat, if so configured) and '
                                  'multiplied into the real images.',
                                  'Planted shapes are rescaled NASA Apollo samples and procedural bodies with sourced mean '
                                  'proportions; the spreads and the log-uniform heights are calibration choices, not a measured '
                                  'polar rock population.',
                                  'Caster heights are equivalent rectangular-shadow heights; a shadow that leaves the fitting window gives '
                                  'only a lower bound. Warning cells are not object counts: one rock can set off neighbouring cells.',
                                  'The clearance value is illustrative, not a verified lander limit.'])


def _plot_subpixel(ex, casters, injection, touchdown, clearance, sigma):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.8), layout='constrained')
    ax = axes[0]
    mean = np.nanmean(ex.data['stack'], axis=0)
    ax.imshow(mean, cmap='gray', vmin=np.nanpercentile(mean, 1), vmax=np.nanpercentile(mean, 99))
    bounded = [c for c in casters if c['height_lower_bound_m'] is not None]
    if bounded:
        pts = np.array([[c['col_px'], c['row_px'], c['height_lower_bound_m']] for c in bounded])
        im = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], s=14, cmap='viridis', vmin=0, vmax=max(1.2, pts[:, 2].max()),
                        edgecolors='white', linewidths=.3)
        fig.colorbar(im, ax=ax, shrink=.75, label='height lower bound (m)')
    ax.plot(touchdown[1], touchdown[0], marker='+', color='white', ms=18, mew=2.5)
    ax.set_title(f'Sub-pixel caster cells after relief correction (sigma {sigma:.3f}; + touchdown)', fontsize=10)
    ax = axes[1]
    ax.set_axisbelow(True)
    values = [c['height_lower_bound_m'] for c in bounded]
    if values:
        ax.hist(values, bins=np.arange(0, max(values)+.2, .1), color='#1f3a5f')
    ax.axvline(clearance, color='#c1121f', ls='--', lw=1.2, label=f'illustrative clearance {clearance:g} m')
    ax.set_xlabel('height lower bound (m)'); ax.set_ylabel('cells'); ax.legend(frameon=False)
    ax.set_title(f'{sum(v >= clearance for v in values)} of {len(values)} cells bounded at or above clearance', fontsize=10)
    ax = axes[2]
    ax.set_axisbelow(True)
    for key, marker, colour, label in (('sized_height_lower_bound_m', 'v', '#5b7fa6', 'lower bound'),
                                       ('sized_height_m', 'o', '#1f3a5f', 'context-supported estimate')):
        pts = np.array([[r['height_m'], r[key]] for r in injection if r.get(key) is not None]).reshape(-1, 2)
        jitter = np.random.default_rng(0).uniform(-.03, .03, len(pts))
        ax.scatter(pts[:, 0]+jitter, pts[:, 1], marker=marker, color=colour, s=36, label=label)
    top = max([1.4]+[r[k] for r in injection for k in ('sized_height_lower_bound_m', 'sized_height_m') if r.get(k) is not None])
    ax.plot([0, top], [0, top], color='#80909d', lw=1, ls=':')
    ax.set_xlim(0, top); ax.set_ylim(0, top)
    ax.set_xlabel('true height of the injected rock (m)'); ax.set_ylabel('recovered height (m)')
    ax.legend(frameon=False, loc='upper left'); ax.set_title('Rocks injected into the real images, sized the same way', fontsize=10)
    fig.suptitle('HATI T14 | sub-pixel casters after relief correction | research diagnostic, not a hazard map')
    fig.savefig(ex.out/'subpixel_casters.png', dpi=130); plt.close(fig)
