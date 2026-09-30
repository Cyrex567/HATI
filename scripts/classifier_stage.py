"""T18 worker: the sweep morphology classifier on generated scenes, on features planted into the real images, and on Athena cells.

Three parts, in the order they depend on each other:

1. Margins. Generated scenes (independent relief generator) under the stack's own
   measured geometry, at T12's relief-corrected residual scale when this campaign
   measured it, else at the assumed sigma. Margins are calibrated on one set of
   seeds against the declared targets and frozen; a second set of seeds gives the
   held-out confusion table with exact 95% intervals.
2. Planted features. Rocks, bowls and mounds rendered on level ground are
   multiplied into the real images at declared sites. Each site is classified
   before and after planting. Recovery counts only sites whose own background was
   called no_signal, so a busy background cannot be mistaken for a success.
3. Athena cells. Every assessed cell within the touchdown radius plus a random
   sample declared by seed, never chosen by score. Labels are research labels:
   a boulder call carries its fitted height and compatible range, a crater call
   its diameter and depth, a hummock call its slope; none of them clears ground.

Config keys (all optional): classifier (SweepClassifierConfig fields),
classifier_seeds, classifier_blanks, classifier_targets, classifier_workers,
classifier_planted_sites, classifier_planted_kinds, classifier_athena_cells,
classifier_slope_bin.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import time

import numpy as np

from classifier_experiments import banks_for, classify_scene, variants
from relief_experiments import _injection_sites, _noise_passes, _slope_limit, save
from src.hati_core.relief_scenes import relief_feature, render_relief
from src.hati_core.rock_scenes import make_rock
from src.hati_core.sweep_classifier import (CLASSES, LABELS, TRUTH_TO_CLASS, SweepClassifierConfig, calibrate_margins,
                                            confusion_with_intervals, decide, evaluate_cell, usable_frames)

DEFAULT_TARGETS = dict(none=.05, sun=.05, compactness=.1, pair=dict(boulder=.05, hummock=.1, crater=.05, extended=.1))
PLANTED = [('rock', dict(height_m=.3)), ('rock', dict(height_m=.6)), ('rock', dict(height_m=1.2)),
           ('bowl', dict(size_m=3., max_slope_deg=10.)), ('bowl', dict(size_m=6., max_slope_deg=5.)),
           ('mound', dict(size_m=6., max_slope_deg=5.))]


def _geometry(ex, ccfg):
    return dict(azimuths=np.asarray(ex.data['azimuths'], float).tolist(),
                elevations=np.asarray(ex.data['elevations'], float).tolist(),
                pixel_m=float(ex.sc.pixel_m), registration_sigma_px=float(ex.sc.registration_sigma_px),
                psf_sigma_px=float(ex.sc.psf_sigma_px), classifier=asdict(ccfg))


def _map(fn, jobs, workers):
    if workers <= 1:
        return [fn(j) for j in jobs]
    import multiprocessing
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        return list(pool.map(fn, jobs, chunksize=2))


def _feature_factor(shape, site, kind, spec, azimuths, elevations, pixel_m, seed, window_px):
    """Multiplicative brightness of one feature rendered on level, uniform ground; 1 outside its window."""
    r, c = site
    half = window_px//2
    r0, c0 = int(r)-half, int(c)-half
    if r0 < 0 or c0 < 0 or r0+window_px > shape[0] or c0+window_px > shape[1]:
        raise ValueError('planting window must lie inside the image')
    local = (r-r0, c-c0)
    rocks, features = [], []
    if kind == 'rock':
        rocks = [make_rock(seed, local, spec['height_m'], .6, aspect=1.35)]
    else:
        features = [relief_feature(kind, spec['size_m'], spec['max_slope_deg'], centre_px=local, seed=seed)]
    out = render_relief((window_px, window_px), azimuths, elevations, pixel_m=pixel_m, seed=seed, noise=0.,
                        rocks=rocks, features=features, supersample=4, texture=0., stain=0., frame_plane=0.)
    factor = np.ones((len(azimuths), *shape))
    factor[:, r0:r0+window_px, c0:c0+window_px] = out['stack']
    return factor


def _cell_job(job):
    """Classify one real patch; banks are cached per process by rounded receiving slope."""
    patch, valid, sigma, slopes, geometry, meta = job
    ccfg = SweepClassifierConfig(**geometry['classifier'])
    from src.hati_core.shadow_likelihood import ShadowConfig
    sc = replace(ShadowConfig(pixel_m=geometry['pixel_m'], registration_sigma_px=geometry['registration_sigma_px'],
                              psf_sigma_px=geometry['psf_sigma_px']), radius_px=ccfg.radius_px, root_support_px=ccfg.support_px)
    az, el = np.asarray(geometry['azimuths'], float), np.asarray(geometry['elevations'], float)
    # The same frame rule evaluate_cell applies, so the banks pair every frame with its own Sun.
    frames = usable_frames(patch, valid, ccfg)
    if len(frames) < 4:
        return dict(meta, status='insufficient_frames_or_support', frames=frames.tolist())
    banks = banks_for(ccfg, sc, az[frames], el[frames], slopes)
    row = evaluate_cell(patch[frames], valid[frames], az[frames], el[frames], sigma, sc, ccfg, slopes=slopes,
                        banks=banks['measured'], shuffled_banks=banks['shuffled'], shifts=banks['shifts'])
    row['frames'] = frames[row['frames']].tolist() if row.get('frames') else frames.tolist()
    fits = row.pop('fits', {})
    row['best_parameters'] = {k: v.get('parameters') for k, v in fits.items() if 'parameters' in v}
    row['compatible_ranges'] = {k: v['compatible_ranges'] for k, v in fits.items() if 'compatible_ranges' in v}
    return dict(meta, **row)


def _patch(ex, stack, valid, r, c, radius, bin_):
    sl = np.s_[:, r-radius:r+radius+1, c-radius:c+radius+1]
    slopes = tuple(float(np.round(v/bin_)*bin_) for v in (ex.data['slope_row'][r, c], ex.data['slope_col'][r, c]))
    return stack[sl], valid[sl], slopes


def _hazard(label, row, clearance, slope_limit):
    """The physical quantity a call carries into the hazard maps; never a clearance."""
    p = (row.get('best_parameters') or {}).get(label) or {}
    r = (row.get('compatible_ranges') or {}).get(label) or {}
    note = 'compatible ranges are descriptive (chi-square within 4 of the best), not confidence intervals'
    if label == 'boulder':
        low = r.get('height_m', [p.get('height_m')])[0]
        return dict(height_m=p.get('height_m'), height_range_m=r.get('height_m'), exceeds_clearance=bool(low >= clearance),
                    note=note+'; the clearance test uses the lower end')
    if label in ('crater', 'hummock'):
        slope_low = r.get('max_slope_deg', [p.get('max_slope_deg') or 0])[0]
        names = ('depth_m', 'wall_slope_deg') if label == 'crater' else ('height_m', 'flank_slope_deg')
        return {'diameter_m': p.get('diameter_m'), 'diameter_range_m': r.get('diameter_m'), names[0]: p.get('relief_m'),
                names[0].replace('_m', '_range_m'): r.get('relief_m'), names[1]: p.get('max_slope_deg'),
                names[1].replace('_deg', '_range_deg'): r.get('max_slope_deg'),
                'exceeds_slope_limit': bool(slope_low >= slope_limit), 'note': note+'; the slope test uses the lower end'}
    return {}


def _plot(ex, synthetic, planted, athena, touchdown):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colours = dict(boulder='#b64262', hummock='#e2a441', crater='#1f3a5f', extended='#64b9a5', ambiguous='#9c7ab8',
                   no_signal='#c9d2da', non_solar_change='#5b5b5b')
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), layout='constrained')
    ax = axes[0]
    truths = list(TRUTH_TO_CLASS)
    table = synthetic['table']
    matrix = np.array([[table.get(t, {}).get(label, 0) for label in LABELS] for t in truths], float)
    share = matrix/np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    ax.imshow(share, cmap='Blues', vmin=0, vmax=1)
    for a in range(len(truths)):
        for b in range(len(LABELS)):
            if matrix[a, b]:
                ax.text(b, a, f'{int(matrix[a, b])}', ha='center', va='center', fontsize=9, color='white' if share[a, b] > .6 else '#13283b')
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=35, ha='right'); ax.set_yticks(range(len(truths)), truths)
    ax.set_title('Generated scenes, held-out seeds, frozen margins')
    ax = axes[1]
    kinds = list(dict.fromkeys(p['kind'] for p in planted))
    for k, kind in enumerate(kinds):
        rows = [p for p in planted if p['kind'] == kind and p['background_label'] == 'no_signal']
        right = sum(p['planted_label'] == TRUTH_TO_CLASS[p['truth']] for p in rows)
        ax.barh(k, right/max(len(rows), 1), color='#1f3a5f')
        ax.text(min(right/max(len(rows), 1), .9)+.02, k, f'{right}/{len(rows)} quiet sites', va='center', fontsize=9)
    ax.set_yticks(range(len(kinds)), kinds); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel('planted features given their own class (quiet backgrounds only)')
    ax.set_title('Features planted into the real images')
    ax = axes[2]
    frame = np.nanmedian(ex.data['stack'], axis=0)
    ax.imshow(frame, cmap='gray', vmin=np.nanpercentile(frame, 1), vmax=np.nanpercentile(frame, 99))
    for label in LABELS:
        pts = [(r['col_px'], r['row_px']) for r in athena if r.get('label') == label]
        if pts:
            x, y = zip(*pts)
            ax.scatter(x, y, s=14, color=colours[label], label=f'{label} ({len(pts)})')
    if touchdown is not None:
        ax.plot(touchdown[1], touchdown[0], '+', color='#db334a', ms=16, mew=2)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.04), ncols=3, frameon=False, fontsize=8)
    ax.set_title('Declared Athena cells (+ touchdown)'); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('HATI T18 | sweep morphology classifier | research labels, not hazard decisions')
    fig.savefig(ex.out/'classifier.png', dpi=130); plt.close(fig)


def t18(ex):
    """Sweep morphology classifier: boulder, hummock, crater or something else (HATI 2.6)."""
    cfg, d = ex.cfg, ex.data
    ccfg = SweepClassifierConfig(**cfg.get('classifier', {}))
    passes, _ = _noise_passes(ex)
    sigma = passes[-1][1]
    sigma_source = 'T12 relief-corrected residual scale' if passes[-1][0] == 'relief_corrected' else 'assumed model sigma'
    geometry = _geometry(ex, ccfg)
    workers = int(cfg.get('classifier_workers', 1))
    targets = cfg.get('classifier_targets', DEFAULT_TARGETS)
    started = time.monotonic()
    # 1. Margins from generated scenes under this stack's geometry.
    ex.live.update(force=True, kind='stage', message='T18: generated scenes for the classifier margins')
    seeds, blanks = int(cfg.get('classifier_seeds', 2)), int(cfg.get('classifier_blanks', 20))
    jobs = []
    for split, offset in (('calibration', 0), ('test', 50000)):
        for case, (truth, spec) in enumerate(variants()):
            jobs += [(split, truth, spec, cfg['seed']+900000+offset+100*case+s, sigma, False, geometry) for s in range(seeds)]
        for truth in ('none', 'stripes'):
            jobs += [(split, truth, {}, cfg['seed']+990000+offset+(1000 if truth == 'stripes' else 0)+s, sigma, False, geometry)
                     for s in range(blanks)]
    scenes = _map(classify_scene, jobs, workers)
    margins = calibrate_margins([r for r in scenes if r['split'] == 'calibration'], targets)
    synthetic = confusion_with_intervals([r for r in scenes if r['split'] == 'test'], margins)
    save(ex.out/'generated_scenes.json', scenes)
    print(f'T18 margins from {len(scenes)} generated scenes in {time.monotonic()-started:.0f} s', flush=True)
    # 2. Features planted into the real images at declared sites.
    ex.live.update(force=True, kind='stage', message='T18: planting rocks, bowls and mounds into the real images')
    valid = (np.nan_to_num(np.asarray(d['visibility'], float), nan=0.) >= .99) & np.isfinite(d['stack'])
    stack = np.where(np.isfinite(d['stack']), d['stack'], np.nan)
    common = valid.all(axis=0)
    touchdown = (int(ex.run['counterfactual']['row_px']), int(ex.run['counterfactual']['col_px']))
    radius, bin_ = ccfg.radius_px, float(cfg.get('classifier_slope_bin', .01))
    per_kind = int(cfg.get('classifier_planted_sites', 8))
    kinds = [k for k in PLANTED if f'{k[0]} {list(k[1].values())[0]:g}' in cfg.get('classifier_planted_kinds',
             [f'{k[0]} {list(k[1].values())[0]:g}' for k in PLANTED])]
    sites = _injection_sites(common, per_kind*len(kinds), 2*radius+8, radius+24, touchdown, cfg['seed']+910000)
    planted_jobs, planted_meta = [], []
    window = 2*radius+1
    for i, (r, c) in enumerate(sites):
        truth, spec = kinds[i % len(kinds)]
        if not (np.isfinite(d['slope_row'][r, c]) and np.isfinite(d['slope_col'][r, c])):
            continue
        factor = _feature_factor(stack.shape[1:], (r+.3, c+.2), truth, spec, d['azimuths'], d['elevations'], ex.sc.pixel_m,
                                 cfg['seed']+920000+i, window+8)
        before, v, slopes = _patch(ex, stack, valid, r, c, radius, bin_)
        after = _patch(ex, stack*factor, valid, r, c, radius, bin_)[0]
        meta = dict(site=i, row_px=int(r), col_px=int(c), truth=truth, kind=f'{truth} ' + ' '.join(f'{k}={v:g}' for k, v in spec.items()))
        planted_jobs += [(before, v, sigma, slopes, geometry, dict(meta, phase='background')),
                         (after, v, sigma, slopes, geometry, dict(meta, phase='planted'))]
    planted_rows = _map(_cell_job, planted_jobs, workers)
    planted = []
    for background, after in zip(planted_rows[0::2], planted_rows[1::2]):
        b, a = decide(background, margins), decide(after, margins)
        planted.append(dict(site=after['site'], row_px=after['row_px'], col_px=after['col_px'], truth=after['truth'], kind=after['kind'],
                            background_label=b and b['label'], planted_label=a and a['label'],
                            planted_lead=a and a['lead'], planted_parameters=after.get('best_parameters')))
    save(ex.out/'planted.json', dict(rows=planted_rows, summary=planted))
    recovery = {}
    for kind in dict.fromkeys(p['kind'] for p in planted):
        rows = [p for p in planted if p['kind'] == kind]
        quiet = [p for p in rows if p['background_label'] == 'no_signal']
        recovery[kind] = dict(sites=len(rows), quiet_sites=len(quiet),
                              own_class=sum(p['planted_label'] == TRUTH_TO_CLASS[p['truth']] for p in quiet),
                              labels={label: sum(p['planted_label'] == label for p in quiet) for label in LABELS})
    # 3. Declared Athena cells.
    athena = []
    if (ex.baseline_dir/'regional.npz').exists():
        ex.live.update(force=True, kind='stage', message='T18: classifying declared Athena cells')
        b = ex.baseline()
        assessed = [row for row in b['cell_table'] if b['status'][row[4], row[5]] == 1
                    and radius <= row[4] < stack.shape[1]-radius and radius <= row[5] < stack.shape[2]-radius]
        near_px = cfg.get('relief_touchdown_radius_px', 24)
        near = [row for row in assessed if np.hypot(row[4]-touchdown[0], row[5]-touchdown[1]) <= near_px]
        others = [row for row in assessed if np.hypot(row[4]-touchdown[0], row[5]-touchdown[1]) > near_px]
        count = int(cfg.get('classifier_athena_cells', 200))
        rng = np.random.default_rng(cfg['seed']+930000)
        if count < len(others):
            others = [others[i] for i in sorted(rng.choice(len(others), count, replace=False))]
        jobs = []
        for row, is_near in [(row, True) for row in near]+[(row, False) for row in others]:
            cr, cc = int(row[4]), int(row[5])
            if not (np.isfinite(d['slope_row'][cr, cc]) and np.isfinite(d['slope_col'][cr, cc])):
                athena.append(dict(row_px=cr, col_px=cc, near_touchdown=is_near, status='missing_terrain', label=None))
                continue
            patch, v, slopes = _patch(ex, stack, valid, cr, cc, radius, bin_)
            jobs.append((patch, v, sigma, slopes, geometry, dict(row_px=cr, col_px=cc, near_touchdown=is_near,
                                                                 baseline_score=float(b['score'][cr, cc]),
                                                                 distance_to_touchdown_m=float(np.hypot(cr-touchdown[0], cc-touchdown[1])*ex.sc.pixel_m))))
        clearance, limit = cfg.get('sfs_clearance_m', .3), _slope_limit(ex)
        for row in _map(_cell_job, jobs, workers):
            decision = decide(row, margins)
            row['label'] = decision and decision['label']
            row['decision'] = decision
            row['hazard'] = _hazard(row['label'], row, clearance, limit) if row['label'] in CLASSES else {}
            athena.append(row)
        save(ex.out/'athena_cells.json', athena)
    counts = {label: sum(r.get('label') == label for r in athena) for label in LABELS}
    near = [r for r in athena if r.get('near_touchdown')]
    _plot(ex, synthetic, planted, athena, touchdown)
    return ex.result('PARTIAL', 'Boulder, hummock, crater, extended relief or no call, by withheld-frame prediction with '
                     'split-conformal margins from generated scenes; checked on held-out scenes and on features planted into the '
                     'real images. Research labels, not hazard decisions.',
                     sigma=sigma, sigma_source=sigma_source, classifier=asdict(ccfg), classifier_hash=ccfg.hash(),
                     targets=targets, margins=margins, generated_test=synthetic, planted_recovery=recovery,
                     athena_counts=counts, athena_cells=len(athena),
                     near_touchdown={label: sum(r.get('label') == label for r in near) for label in LABELS},
                     elapsed_seconds=round(time.monotonic()-started, 1),
                     limitations=['Margins come from generated scenes; they hold for scenes like those, not for lunar terrain in general.',
                                  'Planted features are rendered on level ground and multiplied into the real images.',
                                  'Dome and bowl hypotheses are paraboloids with Lambert photometry; real shapes differ.',
                                  'A call is evidence for a morphology at this cell, not a count of objects or a clearance.'])
