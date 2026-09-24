"""Workers for T1-T11; all scientific inputs are local cached data."""
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
import zipfile

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from review_shadow_bundle import read_bundle
from landing_maps import clean_json, write_tif
from live_feedback import LiveFeedback
from src.hati_core.regional_shadow import RegionalConfig, assess_regions
from src.hati_core.shadow_likelihood import ShadowConfig, RegistrationProjector, shadow_template
from src.hati_core.scene_diagnostics import SceneConfig, local_registration
from src.hati_core.campaign_controls import render_control
from adaptive_experiments import t9, t10, t11, t16


def save_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('no_rows\n', encoding='utf-8'); return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(clean_json(v)) if isinstance(v, (list, dict, np.ndarray)) else v for k, v in row.items()})


def load_inputs(bundle):
    data, run, transform, crs, proof = read_bundle(bundle)
    with zipfile.ZipFile(bundle) as z:
        with np.load(io.BytesIO(z.read('aligned_stack.npz')), allow_pickle=False) as n:
            for key in ('slope_row', 'slope_col'):
                if key not in n or n[key].shape != data['stack'].shape[1:]:
                    raise ValueError('campaign requires the original saved receiving-slope fields')
                data[key] = n[key]
    return data, run, transform, crs, proof


def validate_config(cfg):
    if cfg['schema_version'] != 1:
        raise ValueError('unsupported campaign configuration version')
    for k in ('seed', 'synthetic_seeds', 'profile_step_px'):
        if type(cfg[k]) is not int or cfg[k] < (0 if k == 'seed' else 1):
            raise ValueError(f'invalid {k}')
    for key in ('synthetic_locations', 'registration_planted_shifts_px'):
        if not cfg[key] or any(len(x) != 2 for x in cfg[key]):
            raise ValueError(f'invalid {key}')
    if any(not 0 < v < 1 for pos in cfg['synthetic_locations'] for v in pos):
        raise ValueError('synthetic locations must lie inside the window')
    if not cfg['geometry_shifts'] or any(type(s) is not int or s < 1 for s in cfg['geometry_shifts']):
        raise ValueError('positive cyclic shifts required')
    if not cfg['profile_heights_m'] or any(not np.isfinite(v) or v <= 0 for v in cfg['profile_heights_m']):
        raise ValueError('positive finite profile heights required')
    for key in ('expanded_width_m', 'larger_support_px', 'profile_delta_chi2'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'invalid {key}')
    for key in ('thermal_min_coverage', 'held_out_min_recall', 'held_out_max_false_alarm_fraction'):
        if not 0 < cfg[key] <= 1:
            raise ValueError(f'invalid {key}')
    for name, frames in cfg['drop_frame_groups'].items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or not frames or any(not isinstance(p, str) for p in frames):
            raise ValueError('group names must be plain directory names with a nonempty frame list')
    from src.hati_core.adaptive_shadow import AdaptiveConfig
    AdaptiveConfig(**cfg.get('adaptive', {}))
    for key in ('rock_seeds', 'rock_supersample', 'rock_roi_cells', 'prediction_step_px'):
        if key in cfg and (type(cfg[key]) is not int or cfg[key] < 1):
            raise ValueError('positive integer required: '+key)
    if cfg.get('rock_roi_cells', 3) % 2 != 1:
        raise ValueError('rock_roi_cells must be odd')
    if 'rock_heights_m' in cfg and (not cfg['rock_heights_m'] or any(not np.isfinite(v) or v <= 0 for v in cfg['rock_heights_m'])):
        raise ValueError('rock heights must be finite and positive')
    # T12 and T16 keys are optional; defaults live in the workers.
    from src.hati_core.noise_scale import NoiseScaleConfig
    NoiseScaleConfig(**cfg.get('noise_scale', {}))
    for key in ('noise_control_seeds', 'null_seeds'):
        if key in cfg and (type(cfg[key]) is not int or cfg[key] < 1):
            raise ValueError('positive integer required: '+key)
    if 'null_caster_heights_m' in cfg and any(not np.isfinite(v) or v <= 0 for v in cfg['null_caster_heights_m']):
        raise ValueError('null caster heights must be finite and positive')
    mode = cfg.get('null_render_noise', 'measured')
    if mode not in ('assumed', 'measured') and not (isinstance(mode, (int, float)) and np.isfinite(mode) and mode > 0):
        raise ValueError('null_render_noise must be "assumed", "measured" or a positive number')
    if 'null_gate_max_fraction' in cfg and not 0 < cfg['null_gate_max_fraction'] <= 1:
        raise ValueError('null_gate_max_fraction must lie in (0, 1]')


class Experiment:
    def __init__(self, args):
        self.args = args
        self.out = args.output
        self.out.mkdir(parents=True, exist_ok=True)
        self.cfg = json.loads(args.config.read_text(encoding='utf-8'))
        validate_config(self.cfg)
        self.data, self.run, self.transform, self.crs, self.proof = load_inputs(args.bundle)
        source = self.run['shadow_configuration']
        self.sc = ShadowConfig(**source['shadow'])
        self.rc = RegionalConfig(**source['regional'])
        self.noise = source['noise_sigma']
        self.baseline_dir = args.campaign/'stages/T1/baseline'
        self.live = LiveFeedback(args.campaign, args.stage)
        self.live.inputs(self.data, self.run)
        self.live.update(force=True, kind='stage', message='Preparing '+args.stage, subrun=None)

    def result(self, status, reason, **details):
        result = dict(test=self.args.stage, status=status, reason=reason, provenance=self.proof, input_demo=bool(self.run.get('demo', False)),
                      configuration=self.cfg, **details)
        save_json(self.out/'result.json', result)
        self.live.update(force=True, message=reason, result_status=status)
        print(f'{self.args.stage}: {status}: {reason}', flush=True)
        return result

    def regional(self, name, *, indices=None, order=None, rc=None, sc=None, data=None, profiles=False, input_proof=None):
        d = data or self.data; rc = rc or self.rc; sc = sc or self.sc
        # Held-out scenes use a different grid and must never be overlaid on the development image.
        watch = self.live if data is None else None
        if watch:
            watch.last_region = -float('inf')
            watch.update(force=True, subrun=name, kind='stage', message='Preparing regional search', fit=None, score=None)
        folder = self.out/name; folder.mkdir(parents=True, exist_ok=True)
        geometry = np.arange(len(d['stack'])) if order is None else np.asarray(order)
        if watch:
            watch.update(force=True, model_azimuths=d['azimuths'][geometry].tolist(),
                         model_elevations=d['elevations'][geometry].tolist(), geometry_order=geometry.tolist())
        setup = dict(shadow=asdict(sc), regional=asdict(rc), frame_indices=indices,
                     geometry_order=geometry.tolist(), noise_sigma=self.noise,
                     input_provenance=input_proof or self.proof, profiles=profiles)
        # Subruns can be long; only reuse complete, hash-verified results.
        stamp = folder/'complete.json'
        signature = hashlib.sha256(json.dumps(clean_json(setup), sort_keys=True).encode()).hexdigest()
        if stamp.exists():
            old = json.loads(stamp.read_text())
            if old['signature'] == signature and all((folder/p).is_file() and
                    hashlib.sha256((folder/p).read_bytes()).hexdigest() == h for p, h in old['files'].items()):
                print(f'Reusing {name}', flush=True)
                if watch:
                    watch.update(force=True, kind='stage', message='Reusing verified regional results', subrun=name)
                with np.load(folder/'regional.npz', allow_pickle=False) as n:
                    return {k: n[k] for k in n.files}
        profile_rows, profile_scores, profile_ident = [], [], []
        def audit(cell):
            shape = (rc.cell_px**2, len(rc.heights_m), len(rc.widths_m))
            profile_rows.append([cell['row_px'], cell['col_px']])
            profile_scores.append(cell['scores'].reshape(shape).max(axis=0))
            profile_ident.append(cell['identifiability'].reshape(shape).max(axis=0))
        last = [0.]
        def progress(info):
            if time.monotonic()-last[0] > 20:
                print(f'{name}: {info["cells_assessed"]}/{info["cells_visited"]} assessed/visited', flush=True)
                last[0] = time.monotonic()
        result = assess_regions(d['stack'], d['azimuths'][geometry], d['elevations'][geometry],
                                self.noise, sc, rc, visible=d['visibility'],
                                slope_row=d['slope_row'], slope_col=d['slope_col'],
                                frame_indices=indices, progress=progress,
                                audit_callback=audit if profiles else None,
                                observer=(lambda info: watch.regional(info, name)) if watch else None)
        arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(folder/'regional.npz', **arrays)
        save_json(folder/'candidates.json', result['candidates'])
        write_csv(folder/'candidates.csv', result['candidates'])
        assessed = result['status'] == 1
        score = result['score'][assessed]
        summary = dict(setup=setup, cells_visited=result['cells_visited'], cells_assessed=result['cells_assessed'],
                       assessed_pixel_fraction=float(assessed.mean()),
                       exceedance_fraction_assessed=float(np.mean(score >= rc.score_scale)) if len(score) else None,
                       median_score=float(np.median(score)) if len(score) else None,
                       score_p95=float(np.percentile(score, 95)) if len(score) else None,
                       candidates=len(result['candidates']))
        save_json(folder/'summary.json', summary)
        if profiles:
            np.savez_compressed(folder/'profiles.npz', coordinates=np.asarray(profile_rows).reshape(-1, 2),
                                scores=np.asarray(profile_scores).reshape(-1, len(rc.heights_m), len(rc.widths_m)),
                                max_identifiability=np.asarray(profile_ident).reshape(-1, len(rc.heights_m), len(rc.widths_m)),
                                heights_m=rc.heights_m, widths_m=rc.widths_m)
        files = [folder/'regional.npz', folder/'summary.json', folder/'candidates.json', folder/'candidates.csv']
        if profiles:
            files.append(folder/'profiles.npz')
        save_json(stamp, dict(signature=signature, files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
        return arrays

    def baseline(self):
        with np.load(self.baseline_dir/'regional.npz', allow_pickle=False) as n:
            return {k: n[k] for k in n.files}

    def compare(self, first, second):
        # Root positions are identical for full-grid ablations/stress/width tests.
        a, b = first['root_evidence'], second['root_evidence']
        lookup = {(r, c): s for r, c, s in b}
        matched = [(s, lookup[r, c]) for r, c, s in a if (r, c) in lookup]
        x = np.asarray(matched).reshape(-1, 2)
        return dict(first_roots=len(a), second_roots=len(b), matched_roots=len(x),
                    median_paired_score_change=float(np.median(x[:, 1]-x[:, 0])) if len(x) else None,
                    fraction_second_lower=float(np.mean(x[:, 1] < x[:, 0])) if len(x) else None,
                    first_exceedance_on_matched=float(np.mean(x[:, 0] >= self.rc.score_scale)) if len(x) else None,
                    second_exceedance_on_matched=float(np.mean(x[:, 1] >= self.rc.score_scale)) if len(x) else None,
                    interpretation='Descriptive paired root scores; neighbouring roots are spatially dependent.')

    def plot_grid(self, filename, arrays, titles):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        fig, axes = plt.subplots(1, len(arrays), figsize=(5*len(arrays), 5), squeeze=False, layout='constrained')
        score_values = [a[np.isfinite(a)] for a, t in zip(arrays, titles) if 'score' in t.lower()]
        score_max = max([float(a.max()) for a in score_values if a.size] or [1.])
        for ax, array, title in zip(axes[0], arrays, titles):
            categorical = 'status' in title.lower()
            cmap = ListedColormap(['#80909d', '#64b9a5', '#e2a441', '#b64262']) if categorical else plt.get_cmap('magma').with_extremes(bad='#80909d')
            limits = (-.5, 3.5) if categorical else (0, 1) if 'fraction' in title.lower() else (0, score_max)
            im = ax.imshow(array, cmap=cmap, vmin=limits[0], vmax=limits[1], interpolation='nearest')
            ax.set_title(title); ax.set_xlabel('Column (pixels)'); ax.set_ylabel('Row (pixels)')
            bar = fig.colorbar(im, ax=ax, shrink=.75)
            if categorical:
                bar.set_ticks([0, 1, 2, 3]); bar.set_ticklabels(['Border', 'Assessed', 'Unavailable', 'Nonidentifiable'])
        label = 'synthetic demonstration' if self.run.get('demo', False) else 'research evidence'
        fig.suptitle('HATI saturation diagnostics | '+label)
        fig.savefig(self.out/filename, dpi=140); plt.close(fig)

    def synthetic(self, *, indices=None, order=None, rc=None, kinds=None,
                  render_noise=None, model_noise=None, seeds=None):
        """Same seeds, geometry, masks, slopes and noise for matched variants.

        render_noise sets the noise drawn into the scene; model_noise is the sigma
        the detector assumes. Both default to the source run's assumed sigma, so
        existing stages are unchanged. T12 separates them.
        """
        render = self.noise if render_noise is None else float(render_noise)
        model = self.noise if model_noise is None else float(model_noise)
        trials = self.cfg['synthetic_seeds'] if seeds is None else int(seeds)
        rows = []
        d = self.data; radius = self.sc.radius_px
        size = 2*radius+16; half = size//2
        h, w = d['stack'].shape[1:]
        for location, (fy, fx) in enumerate(self.cfg['synthetic_locations']):
            if min(h, w) < size:
                rows.append(dict(location=location, status='window_too_small')); continue
            y, x = int(np.clip(round(fy*h), half, h-half)), int(np.clip(round(fx*w), half, w-half))
            sl = np.s_[:, y-half:y-half+size, x-half:x-half+size]
            slopes = (float(d['slope_row'][y, x]), float(d['slope_col'][y, x]))
            if not np.isfinite(slopes).all():
                rows.append(dict(location=location, status='missing_slope')); continue
            mask = np.isfinite(d['stack'][sl])
            scenarios = kinds or [('static', .3), ('structured_null', .3), ('resolved_ridge', .3),
                                  ('caster', .3), ('caster', .6), ('caster', 1.2)]
            for kind, height in scenarios:
                for trial in range(trials):
                    seed = self.cfg['seed']+100*location+trial
                    entry = dict(location=location, row_px=y, col_px=x, seed=seed, kind=kind,
                                 height_m=height if kind in ('caster', 'overlap') else None,
                                 render_noise=render, model_noise=model)
                    try:
                        stack = render_control((size, size), d['azimuths'], d['elevations'],
                                               pixel_m=self.sc.pixel_m, seed=seed, noise=render,
                                               kind=kind, height=height, slope_rc=slopes)
                        stack[~mask] = np.nan
                        geometry = np.arange(len(stack)) if order is None else np.asarray(order)
                        fit = assess_regions(stack, d['azimuths'][geometry], d['elevations'][geometry], model,
                                             self.sc, rc or self.rc, visible=d['visibility'][sl], frame_indices=indices,
                                             slope_row=d['slope_row'][y-half:y-half+size, x-half:x-half+size],
                                             slope_col=d['slope_col'][y-half:y-half+size, x-half:x-half+size])
                        root = ((size-1)/2+.3, (size-1)/2+.2)
                        distances = np.asarray([np.hypot(p['row_px']-root[0], p['col_px']-root[1]) for p in fit['candidates']])
                        score = fit['score'][np.isfinite(fit['score'])]
                        entry.update(status='assessed' if len(score) else 'unassessed', cells_assessed=fit['cells_assessed'],
                                     maximum_score=float(score.max()) if len(score) else None,
                                     recovered_within_2px=bool(np.any(distances <= 2)) if kind in ('caster', 'overlap') and len(score) else None,
                                     warning_roots=len(distances),
                                     far_warning_roots=int(np.sum(distances > 3)) if kind == 'caster' else None)
                    except ValueError as exc:
                        entry.update(status='invalid_geometry', reason=str(exc))
                    rows.append(entry)
                    self.live.update(kind='controls', message='Testing independent synthetic scenes',
                                     control=dict(location=location, kind=kind, height_m=height, seed=seed,
                                                  status=entry['status'], maximum_score=entry.get('maximum_score'),
                                                  recovered=entry.get('recovered_within_2px')))
            print(f'Independent synthetic location {location+1}/{len(self.cfg["synthetic_locations"])} done', flush=True)
        return rows


def maps(ex):
    if ex.args.dem is None or not ex.args.dem.exists():
        return ex.result('BLOCKED', 'Native DEM unavailable; supply --dem to regenerate the three maps.')
    from sweep_products import load_dem_context
    from landing_maps import run
    from src.hati_core.landing_terrain import LandingConfig
    from rasterio.crs import CRS as RasterCRS
    cfg = LandingConfig(**ex.run['landing'])
    halo = cfg.horizon_distance_m+max(*cfg.baselines_m, cfg.footprint_diameter_m)/2+10
    reference_crs = RasterCRS.from_wkt(ex.crs.to_wkt())
    context = load_dem_context(ex.args.dem, ex.transform, reference_crs, ex.data['stack'].shape[1:], halo)
    sweep = dict(stack=ex.data['stack'], azimuths=ex.data['azimuths'], elevations=ex.data['elevations'],
                 transform=ex.transform, crs=ex.crs, pixel_m=ex.sc.pixel_m, frames=ex.run['frames'],
                 site_lat=ex.run.get('site_lat'), site_lon=ex.run.get('site_lon'),
                 test_pixel=(ex.run['counterfactual']['row_px'], ex.run['counterfactual']['col_px']))
    report = run(sweep, context, ex.out/'maps', cfg, ex.sc, ex.rc, ex.noise,
                 provenance=dict(**ex.proof, purpose='cached campaign replay; no ISIS'), is_demo=bool(ex.run.get('demo', False)), live=ex.live)
    return ex.result('COMPLETE', 'Separate terrain, shadow and fused maps regenerated with observability and touchdown attribution.',
                     counterfactual=report['counterfactual'], qualification=report['fusion_status_fractions'])


def t1(ex):
    baseline = ex.regional('baseline', profiles=True)
    candidates = json.loads((ex.baseline_dir/'candidates.json').read_text())
    frames = []
    for i, (az, el, frame) in enumerate(zip(ex.data['azimuths'], ex.data['elevations'], ex.run['frames'])):
        contributions = [p['frame_delta_chi2'][p['frames'].index(i)] for p in candidates if i in p['frames']]
        frames.append(dict(frame=i, pid=frame['pid'], azimuth_deg=az, elevation_deg=el,
                           cot_e=1/np.tan(np.radians(el)), eligible_selected_roots=len(contributions),
                           signed_median=np.median(contributions) if contributions else None,
                           negative_count=int(np.sum(np.asarray(contributions) < 0))))
    write_csv(ex.out/'frame_contributions.csv', frames)
    controls = ex.synthetic(); write_csv(ex.out/'independent_controls.csv', controls)
    ex.plot_grid('baseline.png', [baseline['score'], baseline['common_fraction'], baseline['status']],
                 ['Regional score', 'Common support fraction', 'Assessment status'])
    return ex.result('COMPLETE' if np.any(baseline['status'] == 1) and any(r['status'] == 'assessed' for r in controls) else 'PARTIAL',
                     'Signed contributions and independent geometry/mask controls recorded; no log-slope or occupancy inference.',
                     frames=frames, controls=controls,
                     synthetic_scope='Finite-Sun detector tested against a point-Sun rounded-caster control with anisotropic PSF and correlated noise.')


def t2(ex):
    base = ex.baseline(); n = len(ex.data['stack'])
    groups = [(f'without_frame_{i:02d}', [j for j in range(n) if j != i]) for i in range(n)]
    missing = []
    pids = [p['pid'].lower().split('.')[-1] for p in ex.run['frames']]
    for name, drops in ex.cfg['drop_frame_groups'].items():
        absent = [p for p in drops if p.lower() not in pids]
        if absent:
            missing.append(dict(group=name, reason='Declared frame IDs missing', frames=absent)); continue
        groups.append((name, [i for i, p in enumerate(pids) if p not in [d.lower() for d in drops]]))
    rows = []
    for name, indices in groups:
        if len(indices) < 3:
            missing.append(dict(group=name, reason='Fewer than three frames remain')); continue
        result = ex.regional(name, indices=indices)
        az = np.sort(ex.data['azimuths'][indices] % 360)
        comparison = ex.compare(base, result)
        rows.append(dict(group=name, retained_frames=indices, sun_arc_deg=360-float(np.diff(np.r_[az, az[0]+360]).max()), **comparison))
        controls = ex.synthetic(indices=indices, kinds=[('static', .3), ('caster', .3)])
        write_csv(ex.out/name/'matched_injections.csv', controls)
        planted = [p for p in controls if p.get('kind') == 'caster' and p['status'] == 'assessed']
        nulls = [p for p in controls if p.get('kind') == 'static' and p['status'] == 'assessed']
        rows[-1]['assessed_caster_trials'] = len(planted)
        rows[-1]['recovered_caster_trials'] = sum(bool(p['recovered_within_2px']) for p in planted)
        rows[-1]['assessed_null_trials'] = len(nulls)
        rows[-1]['null_trials_with_warnings'] = sum(p['warning_roots'] > 0 for p in nulls)
    write_csv(ex.out/'paired_ablations.csv', rows)
    return ex.result('PARTIAL' if missing or not rows or any(r['matched_roots'] == 0 for r in rows) else 'COMPLETE',
                     'Every leave-one-out and declared group evaluated on frozen parent eligibility/common pixels with matched injections.',
                     comparisons=rows, unavailable_groups=missing,
                     support='Parent per-cell eligibility and common pixels remain fixed; retained-frame count and identifiability may fall. No frames are removed from production.')


def t3(ex):
    base = ex.baseline(); n = len(ex.data['stack']); rows = []
    shifts = list(dict.fromkeys(s % n for s in ex.cfg['geometry_shifts'] if s % n))
    for shift in shifts:
        order = np.roll(np.arange(n), shift)
        result = ex.regional(f'cyclic_{shift}', order=order)
        rows.append(dict(shift=shift, permutation=order.tolist(), **ex.compare(base, result)))
        write_csv(ex.out/f'cyclic_{shift}'/'independent_controls.csv', ex.synthetic(order=order))
    write_csv(ex.out/'geometry_comparison.csv', rows)
    return ex.result(('PARTIAL' if any(r['matched_roots'] == 0 for r in rows) else 'COMPLETE') if rows else 'BLOCKED',
                     'Full regional cyclic geometry stress completed with fixed observed masks and receiving slopes; these are not permutation p-values.',
                     comparisons=rows, geometry_policy='Azimuth/elevation pairs reassigned together; DEM visibility masks deliberately fixed.')


def t4(ex):
    path = ex.args.thermal
    if path is None:
        return ex.result('BLOCKED', 'No independent thermal footprint CSV supplied. Requires valid polar coverage, source/uncertainty and effective footprints; use --thermal.',
                         required_columns=['source', 'footprint_id', 'row_start', 'row_stop', 'col_start', 'col_stop', 'rock_abundance', 'uncertainty'])
    base = ex.baseline(); a = base['index']; rows = []
    seen = set()
    with path.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            r0, r1, c0, c1 = [int(row[k]) for k in ('row_start', 'row_stop', 'col_start', 'col_stop')]
            value, uncertainty = float(row['rock_abundance']), float(row['uncertainty'])
            key = (row['source'], row['footprint_id'])
            if key in seen or not all(key) or not np.isfinite([value, uncertainty]).all() or not 0 <= value <= 1 or uncertainty < 0:
                raise ValueError('invalid/duplicate thermal footprint, source, fraction or uncertainty')
            seen.add(key)
            if r1 <= r0 or c1 <= c0:
                raise ValueError('thermal footprints require increasing half-open pixel bounds')
            region = a[max(0, r0):max(0, min(len(a), r1)), max(0, c0):max(0, min(a.shape[1], c1))]
            coverage = np.isfinite(region).sum()/((r1-r0)*(c1-c0))
            rows.append(dict(**row, observed_fraction=coverage,
                             mean_shadow_index=float(np.nanmean(region)) if coverage >= ex.cfg['thermal_min_coverage'] else None))
    valid = [r for r in rows if r['mean_shadow_index'] is not None]
    x = [float(r['rock_abundance']) for r in valid]; y = [r['mean_shadow_index'] for r in valid]
    correlation = float(spearmanr(x, y).statistic) if len(valid) >= 3 and len(set(x)) > 1 and len(set(y)) > 1 else None
    write_csv(ex.out/'thermal_aggregates.csv', rows)
    return ex.result('COMPLETE' if correlation is not None else 'PARTIAL',
                     'Thermal footprints aggregated at supplied effective support; descriptive association only, with no size-cut conversion.',
                     footprints=len(rows), supported_footprints=len(valid), descriptive_spearman=correlation,
                     limitations=['Overlapping footprints are not independent samples; no significance p-value is reported.',
                                  'This window may contain too few independent thermal footprints. Broader coverage requires a broader matched HATI run.',
                                  'Source validity, polar calibration and footprint definitions need external review.'])


def t5(ex):
    if ex.cfg['expanded_width_m'] <= max(ex.rc.widths_m):
        raise ValueError('expanded width must extend the saved bank')
    expanded = replace(ex.rc, widths_m=(*ex.rc.widths_m, ex.cfg['expanded_width_m']))
    out = ex.regional('expanded_width', rc=expanded, profiles=True)
    candidates = json.loads((ex.out/'expanded_width/candidates.json').read_text())
    fraction = float(np.mean([p['template_width_m'] == expanded.widths_m[-1] for p in candidates])) if candidates else None
    write_csv(ex.out/'matched_controls.csv', ex.synthetic(rc=expanded))
    comparison = ex.compare(ex.baseline(), out)
    ex.plot_grid('expanded_width.png', [ex.baseline()['score'], out['score']], ['Original width bank score', 'Expanded width bank score'])
    return ex.result('COMPLETE' if comparison['matched_roots'] else 'PARTIAL', 'Expanded width bank, full cell height/width score profiles and matched controls recorded.',
                     comparison=comparison, widest_selected_fraction=fraction,
                     interpretation='profiles.npz maximizes each height/width score over all roots in a cell. Bank-edge migration is a scale/model diagnostic, not an object classification.')


def fit_profile(patch, visibility, azimuths, elevations, slopes, cfg, heights, widths, sigma, delta):
    """Profile root, width and contrast over declared grid; never a confidence interval."""
    yy, xx = np.indices(patch.shape[1:])
    centre = (np.array(patch.shape[1:])-1)/2
    support = np.hypot(yy-centre[0], xx-centre[1]) <= cfg.root_support_px
    valid = np.isfinite(patch) & np.isfinite(visibility) & (visibility >= .99)
    selected = np.flatnonzero(valid[:, support].mean(axis=1) >= .85)
    if len(selected) < 3:
        return dict(status='insufficient_frames', frames=selected.tolist())
    common = support & valid[selected].all(axis=0)
    if common.sum()/support.sum() < .8 or not np.isfinite(slopes).all():
        return dict(status='insufficient_common_support', frames=selected.tolist(), common_fraction=float(common.sum()/support.sum()))
    data = patch[selected]
    ref = np.median(np.where(np.isfinite(data), data, 0.), axis=0)
    p = RegistrationProjector(common, np.full(len(selected), sigma), ref, cfg.registration_sigma_px)
    residual = p.apply(data)
    profiles = []
    for height in heights:
        best = None
        for width in widths:
            for dy in (-.5, .5):
                for dx in (-.5, .5):
                    root = centre+[dy, dx]
                    try:
                        template, censored = shadow_template(support.shape, root, azimuths[selected], elevations[selected], height, width, cfg, slopes)
                    except ValueError:
                        continue
                    projected = p.apply(template)
                    energy = float(np.sum(projected**2))
                    raw = float(np.sum((template[:, common]/sigma)**2))
                    ident = energy/max(raw, 1e-30)
                    if energy < 1e-12 or ident < cfg.min_identifiability:
                        continue
                    inner = float(np.sum(projected*residual))
                    amplitude = float(np.clip(-inner/energy, 0, cfg.max_contrast))
                    improvement = max(0., -2*amplitude*inner-amplitude**2*energy)
                    # Record valid samples at and beyond each nominal endpoint.
                    # Darkness there is not proof of a continuing isolated shadow.
                    endpoints = []
                    for f, az, el in zip(selected, azimuths[selected], elevations[selected]):
                        dr, dc = np.cos(np.radians(az)), -np.sin(np.radians(az))
                        denom = np.tan(np.radians(el))+slopes[0]*dr+slopes[1]*dc
                        length = height/denom/cfg.pixel_m
                        end = root+np.array([dr, dc])*length
                        beyond = end+np.array([dr, dc])*2
                        def observed(pos):
                            r, c = np.rint(pos).astype(int)
                            return bool(0 <= r < support.shape[0] and 0 <= c < support.shape[1] and common[r, c])
                        def intensity(pos):
                            r, c = np.rint(pos).astype(int)
                            return float(patch[f, r, c]) if observed(pos) else None
                        endpoints.append(dict(frame=int(f), endpoint_row_px=end[0], endpoint_col_px=end[1],
                                              endpoint_in_common_support=observed(end), beyond_in_common_support=observed(beyond),
                                              endpoint_intensity=intensity(end), beyond_intensity=intensity(beyond)))
                    candidate = dict(height_m=height, width_m=width, root_offset=[dy, dx], improvement=improvement,
                                     score=np.sqrt(improvement), contrast=amplitude, identifiability=ident,
                                     predicted_endpoint_censored=censored, endpoints=endpoints)
                    if best is None or candidate['improvement'] > best['improvement']:
                        best = candidate
        profiles.append(best or dict(height_m=height, improvement=None, score=None))
    fitted = [p for p in profiles if p['improvement'] is not None]
    if not fitted:
        return dict(status='nonidentifiable', profiles=profiles, frames=selected.tolist())
    best = max(fitted, key=lambda p: p['improvement'])
    compatible = [p['height_m'] for p in fitted if best['improvement']-p['improvement'] <= delta]
    return dict(status='assessed', frames=selected.tolist(), common_pixels=int(common.sum()), best=best, profiles=profiles,
                descriptive_delta_set_m=compatible, delta_chi2=delta,
                uncertainty='Grid compatibility set only. No calibrated confidence level or height bound.')


def t6(ex):
    if not ex.sc.root_support_px < ex.cfg['larger_support_px'] <= ex.sc.radius_px:
        raise ValueError('larger support must exceed original and fit within saved patch radius')
    larger = replace(ex.sc, root_support_px=ex.cfg['larger_support_px'])
    d = ex.data; radius = ex.sc.radius_px; shape = d['stack'].shape[1:]
    profiles = []
    for r in range(radius, shape[0]-radius, ex.cfg['profile_step_px']):
        for c in range(radius, shape[1]-radius, ex.cfg['profile_step_px']):
            sl = np.s_[:, r-radius:r+radius+1, c-radius:c+radius+1]
            for cfg in (ex.sc, larger):
                fit = fit_profile(d['stack'][sl], d['visibility'][sl], d['azimuths'], d['elevations'],
                                  (d['slope_row'][r, c], d['slope_col'][r, c]), cfg,
                                  ex.cfg['profile_heights_m'], ex.rc.widths_m, ex.noise, ex.cfg['profile_delta_chi2'])
                profiles.append(dict(row_px=r, col_px=c, support_px=cfg.root_support_px, **fit))
                ex.live.update(kind='height', message='Comparing height hypotheses',
                               height_profile=dict(row_px=r, col_px=c, support_px=cfg.root_support_px,
                                   status=fit['status'], heights_m=ex.cfg['profile_heights_m'],
                                   scores=[p.get('score') for p in fit.get('profiles', [])],
                                   delta_set_m=fit.get('descriptive_delta_set_m', [])))
            if len(profiles) % 10 == 0:
                print(f'Height profiles: {len(profiles)} support/position combinations', flush=True)
    save_json(ex.out/'real_height_profiles.json', profiles)
    controls = []
    # Known heights, overlapping casters, both fitting supports, real masks and slopes.
    for location, (fy, fx) in enumerate(ex.cfg['synthetic_locations']):
        r = int(np.clip(round(fy*shape[0]), radius, shape[0]-radius-1))
        c = int(np.clip(round(fx*shape[1]), radius, shape[1]-radius-1))
        sl = np.s_[:, r-radius:r+radius+1, c-radius:c+radius+1]
        slopes = (d['slope_row'][r, c], d['slope_col'][r, c])
        if not np.isfinite(slopes).all():
            controls.append(dict(location=location, status='missing_slope')); continue
        for kind in ('caster', 'overlap'):
            for height in (.3, .6, 1.2):
                for trial in range(ex.cfg['synthetic_seeds']):
                    seed = ex.cfg['seed']+100*location+trial
                    try:
                        patch = render_control((2*radius+1,)*2, d['azimuths'], d['elevations'], pixel_m=ex.sc.pixel_m,
                                               seed=seed, noise=ex.noise, kind=kind, height=height, slope_rc=slopes)
                    except ValueError:
                        controls.append(dict(location=location, status='invalid_geometry', height_m=height)); continue
                    patch[~np.isfinite(d['stack'][sl])] = np.nan
                    for cfg in (ex.sc, larger):
                        result = fit_profile(patch, d['visibility'][sl], d['azimuths'], d['elevations'], slopes, cfg,
                                             ex.cfg['profile_heights_m'], ex.rc.widths_m, ex.noise, ex.cfg['profile_delta_chi2'])
                        controls.append(dict(location=location, seed=seed, kind=kind, truth_height_m=height,
                                             support_px=cfg.root_support_px, truth_in_delta_set=(height in result['descriptive_delta_set_m']) if result['status'] == 'assessed' else None,
                                             height_error_m=result['best']['height_m']-height if result['status'] == 'assessed' else None, **result))
        print(f'Height control location {location+1} complete', flush=True)
    save_json(ex.out/'height_controls.json', controls)
    write_csv(ex.out/'height_control_summary.csv', [{k: r.get(k) for k in ('location', 'seed', 'kind', 'truth_height_m',
              'support_px', 'status', 'truth_in_delta_set', 'height_error_m')} for r in controls])
    return ex.result('PARTIAL', 'Height profiles, predicted endpoint support and known-height/overlap recovery measured. Calibrated uncertainty and observed-continuation validation remain open.',
                     real_assessed=sum(p['status'] == 'assessed' for p in profiles), real_profiles=len(profiles),
                     controls=len(controls), limitation='Finite grid delta sets are not confidence intervals. Larger support does not establish identifiability, handle arbitrary overlap or prove an observed shadow continues.')


def t7(ex):
    real, controls = {}, []
    d = ex.data
    for tile in ex.cfg['registration_tiles_px']:
        cfg = SceneConfig(tile_px=tile)
        real[str(tile)] = local_registration(d['stack'], d['azimuths'], d['elevations'], cfg)
        ex.live.update(force=True, kind='registration', message='Checking local registration',
                       registration=dict(tile_px=tile, pairs=len(real[str(tile)]['pairs']),
                                         measured_tiles=sum(r['status'] == 'measured_apparent_offset' for p in real[str(tile)]['pairs'] for r in p['tiles'])))
        for pair in real[str(tile)]['pairs']:
            i, j = pair['frames']
            for shift in ex.cfg['registration_planted_shifts_px']:
                # Copy a common texture, keep each frame's actual holes; apply a
                # known shift before restoring holes. Add a brightness gradient.
                first = d['stack'][i]
                valid = np.isfinite(first)
                texture = np.where(valid, first, 0.)
                second = ndi.shift(texture, shift, order=0, mode='constant', cval=0.)
                shifted_valid = ndi.shift(valid.astype(float), shift, order=0, mode='constant', cval=0.) > .99
                yy, xx = np.indices(first.shape)
                second += .0002*(yy-xx)
                second[~shifted_valid | ~np.isfinite(d['stack'][j])] = np.nan
                measured = local_registration(np.stack([first, second]), d['azimuths'][[i, j]], d['elevations'][[i, j]], cfg)
                for p in measured['pairs']:
                    for row in p['tiles']:
                        entry = dict(tile_px=tile, frames=[i, j], planted_shift_px=shift, **row)
                        if row['status'] == 'measured_apparent_offset':
                            entry['recovery_error_px'] = float(np.hypot(row['row_offset_px']-shift[0], row['col_offset_px']-shift[1]))
                        controls.append(entry)
        print(f'Registration tile {tile} complete', flush=True)
    save_json(ex.out/'real_registration.json', real)
    write_csv(ex.out/'planted_shift_controls.csv', controls)
    measured = [r for p in real.values() for pair in p['pairs'] for r in pair['tiles'] if r['status'] == 'measured_apparent_offset']
    return ex.result('COMPLETE' if measured else 'PARTIAL',
                     'Declared tile sizes evaluated with the same strict mask rule; planted offsets tested with real holes. No image correction or registration sigma inferred.',
                     measured_real_tiles=len(measured), planted_measured=sum(r['status'] == 'measured_apparent_offset' for r in controls),
                     support_diagnosis='Each tile records common fraction after filter and shift erosion. Unsupported tiles remain explicitly unavailable.')


def t8(ex):
    manifest_path = ex.args.held_out
    if manifest_path is None:
        return ex.result('BLOCKED', 'Independent held-out scene bundles and complete hazard/nonhazard annotation masks were not supplied; use --held-out. No field performance verdict is possible.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('split') != 'held_out' or not manifest.get('annotation_source') or not manifest.get('independent_of_development'):
        raise ValueError('held-out manifest must declare split, annotation source and independence')
    rows, seen = [], {ex.proof['stack_sha256']}
    for i, scene in enumerate(manifest['scenes']):
        bundle = manifest_path.parent/scene['bundle']; labels = manifest_path.parent/scene['labels']
        data, run, transform, crs, proof = load_inputs(bundle)
        if proof['stack_sha256'] in seen:
            raise ValueError('development stack or repeated scene cannot be a held-out scene')
        seen.add(proof['stack_sha256'])
        if not np.isclose(run['image_posting_m'], ex.sc.pixel_m):
            raise ValueError('held-out posting differs; declare a separate calibrated configuration')
        with np.load(labels, allow_pickle=False) as n:
            hazard, complete = n['hazard'], n['complete']
            if str(n['stack_sha256']) != proof['stack_sha256']:
                raise ValueError('annotation mask must be bound to held-out stack hash')
        if hazard.dtype != bool or complete.dtype != bool or hazard.shape != data['stack'].shape[1:] or complete.shape != hazard.shape:
            raise ValueError('hazard/complete annotations must be boolean masks on the exact scene grid')
        result = ex.regional(f'scene_{i:02d}', data=data, input_proof=proof)
        # Count cells once, not every replicated score pixel. Entire cell must
        # be annotated; any labelled hazard makes the cell positive.
        tp = fp = tn = fn = unknown_positive = unknown_negative = incomplete = 0
        radius, cell = ex.sc.radius_px, ex.rc.cell_px
        for r in range(radius, hazard.shape[0]-radius, cell):
            for c in range(radius, hazard.shape[1]-radius, cell):
                sl = np.s_[r:min(r+cell, hazard.shape[0]-radius), c:min(c+cell, hazard.shape[1]-radius)]
                if not complete[sl].all():
                    incomplete += 1; continue
                positive = bool(hazard[sl].any())
                score = result['score'][r, c]
                if not np.isfinite(score):
                    unknown_positive += int(positive); unknown_negative += int(not positive); continue
                warning = score >= ex.rc.score_scale
                tp += int(positive and warning); fn += int(positive and not warning)
                fp += int(not positive and warning); tn += int(not positive and not warning)
        recall = tp/(tp+fn+unknown_positive) if tp+fn+unknown_positive else None
        far = fp/(fp+tn) if fp+tn else None
        rows.append(dict(scene=i, annotation_source=manifest['annotation_source'], source_stack=proof['stack_sha256'],
                         true_positive=tp, false_positive=fp, true_negative=tn, false_negative=fn,
                         unknown_positive=unknown_positive, unknown_negative=unknown_negative, incomplete_cells=incomplete,
                         recall_counting_unknown_as_missed=recall, false_alarm_fraction_assessed_negative=far,
                         meets_declared_targets=bool(recall is not None and far is not None and
                               recall >= ex.cfg['held_out_min_recall'] and far <= ex.cfg['held_out_max_false_alarm_fraction'])))
    write_csv(ex.out/'held_out_results.csv', rows)
    return ex.result('COMPLETE' if rows else 'BLOCKED', 'Held-out cell-level performance recorded at the unchanged score threshold; targets are declared research criteria, not mission tolerances.',
                     scenes=rows, all_scenes_meet_targets=bool(rows) and all(r['meets_declared_targets'] for r in rows),
                     limitations=['Neighbouring cells are dependent; no binomial independence or universal mission-loss bound is claimed.',
                                  'Complete negative annotations must include the intended unresolved hazard class.',
                                  'Cell hazard detection does not validate individual object counts, heights or footpad stability.'])


NOISE_CONTROL_KINDS = [('static', .3), ('structured_null', .3), ('caster', .3), ('caster', .6), ('caster', 1.2)]


def summarise_controls(rows):
    """Warnings and recovery per scenario, counted over assessed trials only."""
    groups = {}
    for r in rows:
        groups.setdefault((r['kind'], r.get('height_m')), []).append(r)
    out = []
    for (kind, height), sample in groups.items():
        assessed = [r for r in sample if r['status'] == 'assessed']
        scores = [r['maximum_score'] for r in assessed if r.get('maximum_score') is not None]
        out.append(dict(kind=kind, height_m=height, trials=len(sample), assessed=len(assessed),
                        trials_with_warning_roots=sum(r['warning_roots'] > 0 for r in assessed),
                        recovered_within_2px=sum(bool(r.get('recovered_within_2px')) for r in assessed) if kind in ('caster', 'overlap') else None,
                        median_maximum_score=float(np.median(scores)) if scores else None))
    return out


def t12(ex):
    """Residual scale under the null model, then the baseline controls at that scale."""
    from src.hati_core.noise_scale import NoiseScaleConfig, measure_residual_scale, patch_map
    d = ex.data; assumed = float(ex.noise)
    base_cfg = NoiseScaleConfig(**ex.cfg.get('noise_scale', {}))
    ex.live.update(force=True, kind='stage', message='Measuring the residual scale of the null model')
    measured = {}
    for degree in (1, 2):
        measured[degree] = measure_residual_scale(d['stack'], d['visibility'], d['slope_row'], d['slope_col'],
                                                  replace(base_cfg, spatial_degree=degree))
    linear, quadratic = measured[1], measured[2]
    save_json(ex.out/'residual_scale.json', dict(linear=linear, quadratic=quadratic, assumed_sigma=assumed))
    if not linear['patches']:
        return ex.result('BLOCKED', 'No patch met the support, illumination and slope rules, so no residual scale was measured.',
                         rejected_patches=linear['rejected_patches'], configuration_noise_scale=asdict(base_cfg))
    sigma = linear['pooled_sigma']
    ex.live.field('Residual scale per patch, plane null (normalised intensity)', patch_map(linear, d['stack'].shape[1:]),
                  pooled=round(sigma, 5), assumed=assumed, ratio=round(sigma/assumed, 3), patches=linear['patches'])
    # Reference structure: the same estimator on rendered noise alone. The
    # structure metrics are scale-free, so one rendering at the measured scale
    # tells us what independent (slightly coloured) noise looks like here.
    size = 4*base_cfg.patch_px
    ref_stack = render_control((size, size), d['azimuths'], d['elevations'], pixel_m=ex.sc.pixel_m,
                               seed=ex.cfg['seed']+900000, noise=sigma, kind='static')
    reference = measure_residual_scale(ref_stack, np.ones_like(ref_stack), np.zeros((size, size)), np.zeros((size, size)), base_cfg)
    rows = []
    frames = [p['pid'] for p in ex.run['frames']]
    for i, pid in enumerate(frames):
        rows.append(dict(frame=i, pid=pid, azimuth_deg=float(d['azimuths'][i]), elevation_deg=float(d['elevations'][i]),
                         sigma_plane=linear['per_frame_sigma'][i], sigma_quadratic=quadratic['per_frame_sigma'][i] if quadratic['patches'] else None,
                         ratio_to_assumed=linear['per_frame_sigma'][i]/assumed))
    # If T1 ran in this campaign, relate per-frame scale to per-frame evidence.
    comparison = None
    t1_result = ex.args.campaign/'stages/T1/result.json'
    if t1_result.exists():
        contributions = {f['frame']: f.get('signed_median') for f in json.loads(t1_result.read_text(encoding='utf-8')).get('frames', [])}
        pairs = [(r['sigma_plane'], contributions.get(r['frame'])) for r in rows if contributions.get(r['frame']) is not None]
        for r in rows:
            r['t1_signed_median_contribution'] = contributions.get(r['frame'])
        if len(pairs) >= 3:
            rho = spearmanr([a for a, _ in pairs], [b for _, b in pairs]).correlation
            comparison = dict(frames=len(pairs), spearman_sigma_vs_signed_median=rho,
                              interpretation='Descriptive, one value per frame. A positive coefficient is what equal noise weights '
                                             'would produce if noisier frames dominated the evidence; it is not a test.')
    write_csv(ex.out/'per_frame_scale.csv', rows)
    # "If all excess were noise": scores scale as 1/sigma. Needs T1's baseline.
    rescaled = None
    if (ex.baseline_dir/'regional.npz').exists():
        b = ex.baseline(); ok = b['status'] == 1; s = b['score'][ok]
        if s.size:
            rescaled = dict(assessed_pixels=int(ok.sum()), threshold=ex.rc.score_scale,
                            fraction_above_threshold_assumed=float(np.mean(s >= ex.rc.score_scale)),
                            fraction_above_threshold_rescaled=float(np.mean(s*assumed/sigma >= ex.rc.score_scale)),
                            median_score_assumed=float(np.median(s)), median_score_rescaled=float(np.median(s)*assumed/sigma),
                            interpretation='Upper-bound arithmetic: treats the whole residual excess as independent noise.')
    passes = []
    seeds = ex.cfg.get('noise_control_seeds', ex.cfg['synthetic_seeds'])
    for name, render, model in (('render_assumed_model_assumed', assumed, assumed),
                                ('render_measured_model_assumed', sigma, assumed),
                                ('render_measured_model_measured', sigma, sigma)):
        ex.live.update(force=True, kind='stage', message=f'Controls: {name.replace("_", " ")}')
        controls = ex.synthetic(kinds=NOISE_CONTROL_KINDS, render_noise=render, model_noise=model, seeds=seeds)
        write_csv(ex.out/f'controls_{name}.csv', controls)
        passes.append(dict(name=name, render_noise=render, model_noise=model, scenarios=summarise_controls(controls)))
        print(f'T12 control pass {name} complete', flush=True)
    _plot_t12(ex, linear, rows, passes, assumed)
    return ex.result('PARTIAL', 'Residual scale measured on patches chosen by support, illumination and slope; controls rerun at that scale. '
                     'The residual includes unmodelled structure, so it bounds independent noise from above.',
                     assumed_sigma=assumed, measured_pooled_sigma=sigma, measured_pooled_sigma_quadratic=quadratic['pooled_sigma'],
                     ratio_to_assumed=sigma/assumed, median_patch_sigma=linear['median_patch_sigma'],
                     patches=linear['patches'], rejected_patches=linear['rejected_patches'],
                     negative_variance_frames=linear['negative_variance_frames'],
                     structure=linear['structure'], reference_noise_structure=reference['structure'],
                     per_frame=rows, frame_scale_vs_evidence=comparison, rescaled_baseline=rescaled,
                     control_passes=passes, configuration_noise_scale=asdict(base_cfg))


def _plot_t12(ex, linear, rows, passes, assumed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from src.hati_core.noise_scale import patch_map
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    ax = axes[0, 0]
    sig = [r['sigma_plane'] for r in rows]
    ax.set_axisbelow(True)
    ax.bar(range(len(sig)), sig, color='#1f3a5f')
    ax.axhline(assumed, color='#c1121f', ls='--', lw=1.2, label=f'assumed {assumed:g}')
    ax.axhline(linear['pooled_sigma'], color='#13283b', lw=1, label=f'pooled {linear["pooled_sigma"]:.4f}')
    # Headroom above the tallest bar keeps the legend off the bars.
    ax.set_ylim(0, 1.3*max(*sig, assumed, linear['pooled_sigma']))
    ax.set_xticks(range(len(sig))); ax.set_xticklabels([r['pid'].split('.')[-1] for r in rows], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('residual scale (normalised intensity)'); ax.set_title('Per-frame residual scale, plane null')
    ax.legend(frameon=False, loc='upper left', ncols=2)
    ax = axes[0, 1]
    ax.hist([q['pooled_sigma'] for q in linear['per_patch']], bins=30, color='#1f3a5f')
    ax.axvline(assumed, color='#c1121f', ls='--', lw=1.2)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('patch residual scale'); ax.set_ylabel('patches'); ax.set_title(f'{linear["patches"]} patches chosen without scores')
    ax = axes[1, 0]
    im = ax.imshow(patch_map(linear, ex.data['stack'].shape[1:]), cmap=plt.get_cmap('magma').with_extremes(bad='#80909d'), interpolation='nearest')
    fig.colorbar(im, ax=ax, shrink=.8, label='patch residual scale'); ax.set_title('Where the residual is large (grey: not selected)')
    ax = axes[1, 1]; ax.axis('off')
    header = ['scenario'] + [p['name'].replace('render_', 'rendered ').replace('_model_', '\nmodel ') for p in passes]
    cells = []
    for i, s in enumerate(passes[0]['scenarios']):
        label = s['kind'] + (f' {s["height_m"]} m' if s['kind'] == 'caster' else '')
        cells.append([label] + [f'{p["scenarios"][i]["trials_with_warning_roots"]}/{p["scenarios"][i]["assessed"]}' for p in passes])
    table = ax.table(cellText=cells, colLabels=header, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.6)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_height(2*cell.get_height())  # two-line headers
    ax.set_title('Trials with warning roots / assessed trials')
    fig.suptitle('HATI T12 | residual scale of the null model | research diagnostic')
    fig.savefig(ex.out/'residual_scale.png', dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', choices=['maps', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T16'], required=True)
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--campaign', type=Path, required=True)
    ap.add_argument('--dem', type=Path)
    ap.add_argument('--thermal', type=Path)
    ap.add_argument('--held-out', type=Path)
    ap.add_argument('--rock-catalog', type=Path)
    args = ap.parse_args()
    ex = Experiment(args)
    globals()['maps' if args.stage == 'maps' else args.stage.lower()](ex)


if __name__ == '__main__':
    main()
