"""Calibrate and test the sweep morphology classifier (HATI 2.6) on independent synthetic scenes.

Scenes come from the independent relief generator (src/hati_core/relief_scenes.py):
3D rocks (boulder), Gaussian mounds (hummock), rimmed Gaussian bowls (crater),
elephant-hide ripples (extended), blank ground with albedo texture (no_signal) and
stripes that change between frames without following the Sun (non_solar_change),
all under the measured Athena sweep. The classifier's own dome and bowl models are
paraboloids with Lambert photometry, so it never sees the generator's shapes.

Margins are calibrated on one set of seeds against declared error targets and
then applied, frozen, to scenes with different seeds. The test confusion table
carries exact 95% binomial intervals. A second test at twice the noise shows what
happens when the test scenes are no longer like the calibration scenes.

    python scripts/classifier_experiments.py --output output/sweep_classifier --workers 4
    python scripts/classifier_experiments.py --quick

Synthetic, conditional evidence. It shows what the sweep can separate in principle
at this geometry and noise; field error rates need independently labelled terrain.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT)]

from src.hati_core.relief_scenes import relief_feature, render_relief  # noqa: E402
from src.hati_core.rock_scenes import make_rock  # noqa: E402
from src.hati_core.shadow_likelihood import ShadowConfig  # noqa: E402
from src.hati_core.sweep_classifier import (CLASSES, LABELS, TRUTH_TO_CLASS, SweepClassifierConfig, build_banks,  # noqa: E402
                                            calibrate_margins, confusion_with_intervals, decide, evaluate_cell)

AZIMUTHS = np.array([43.13, 43.54, 68.97, 73.66, 326.66, 354.63, 5.97, 24.94])
ELEVATIONS = np.array([4.35, 3.59, 3.47, 3.38, 3.58, 3.63, 3.28, 3.69])
PIXEL_M = .9
SIGMA = .03
TARGETS = dict(none=.05, sun=.05, compactness=.1,
               pair=dict(boulder=.05, hummock=.1, crater=.05, extended=.1))


def variants():
    out = [('rock', dict(height_m=h, background=b)) for h in (.15, .3, .6, 1.2) for b in ('flat', 'ripples')]
    out += [(kind, dict(size_m=s, max_slope_deg=sl)) for kind in ('mound', 'bowl')
            for s in (3., 6., 12.) for sl in (2., 5., 10.)]
    out += [('ripples', dict(size_m=w, max_slope_deg=sl)) for w in (3., 6., 12.) for sl in (1., 2., 5.)]
    return out


def render(truth, spec, seed, sigma, cfg, azimuths=AZIMUTHS, elevations=ELEVATIONS, pixel_m=PIXEL_M):
    """Noisy (frames, 2r+1, 2r+1) stack of one scene of the given truth at the classifier's patch size."""
    size = 2*cfg.radius_px+1
    centre = (cfg.radius_px+.3, cfg.radius_px+.2)
    features, rocks, structured = [], [], False
    if truth == 'rock':
        rocks = [make_rock(seed, centre, spec['height_m'], .6, aspect=1.35)]
        if spec['background'] == 'ripples':
            features = [relief_feature('ripples', 6., 1., seed=seed)]
    elif truth in ('mound', 'bowl', 'ripples'):
        features = [relief_feature(truth, spec['size_m'], spec['max_slope_deg'], seed=seed)]
    elif truth == 'stripes':
        structured = True
    return render_relief((size, size), azimuths, elevations, pixel_m=pixel_m, seed=seed, noise=sigma,
                         features=features, rocks=rocks, supersample=4, structured_null=structured)['stack']


_BANKS = {}


def banks_for(cfg, sc, azimuths, elevations, slopes=(0., 0.)):
    """Measured and Sun-reassigned banks for one geometry and receiving slope, built once per process."""
    azimuths, elevations = np.asarray(azimuths, float), np.asarray(elevations, float)
    key = (cfg.hash(), sc.pixel_m, sc.psf_sigma_px, tuple(azimuths), tuple(elevations), tuple(slopes))
    if key not in _BANKS:
        if len(_BANKS) >= 8:
            _BANKS.pop(next(iter(_BANKS)))
        shape = (2*cfg.radius_px+1,)*2
        shifts = sorted({1, len(azimuths)//2, len(azimuths)-1})
        _BANKS[key] = dict(shifts=shifts, measured=build_banks(shape, azimuths, elevations, sc, cfg, slopes),
                           shuffled={k: build_banks(shape, np.roll(azimuths, k), np.roll(elevations, k), sc, cfg, slopes)
                                     for k in shifts})
    return _BANKS[key]


def quick_config(cfg):
    return replace(cfg, rock_offsets_px=(-.5, .5), rock_heights_m=(.15, .3, .6, 1.2), crater_diameters_m=(2.7, 5.4, 10.8),
                   hummock_diameters_m=(2.7, 5.4, 10.8))


def classify_scene(job):
    """One generated scene: (split, truth, spec, seed, sigma, quick[, geometry]) -> evaluate_cell row with its truth."""
    split, truth, spec, seed, sigma, quick, *rest = job
    geometry = rest[0] if rest else dict(azimuths=AZIMUTHS, elevations=ELEVATIONS, pixel_m=PIXEL_M,
                                         registration_sigma_px=.5, psf_sigma_px=.6, classifier={})
    cfg = SweepClassifierConfig(**geometry.get('classifier', {}))
    if quick:
        cfg = quick_config(cfg)
    sc = replace(ShadowConfig(pixel_m=geometry['pixel_m'], registration_sigma_px=geometry['registration_sigma_px'],
                              psf_sigma_px=geometry['psf_sigma_px']),
                 radius_px=cfg.radius_px, root_support_px=cfg.support_px)
    az, el = np.asarray(geometry['azimuths'], float), np.asarray(geometry['elevations'], float)
    banks = banks_for(cfg, sc, az, el)
    stack = render(truth, spec, seed, sigma, cfg, az, el, geometry['pixel_m'])
    row = evaluate_cell(stack, np.ones_like(stack, bool), az, el, sigma, sc, cfg,
                        banks=banks['measured'], shuffled_banks=banks['shuffled'], shifts=banks['shifts'])
    fits = row.pop('fits', {})
    row.update(split=split, truth=truth, spec=spec, seed=seed, sigma=sigma,
               best_parameters={k: v.get('parameters') for k, v in fits.items() if 'parameters' in v},
               config_hash=cfg.hash())
    return row


def plot(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    truths = [t for t in TRUTH_TO_CLASS]
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.6), layout='constrained')
    for ax, key, title in ((axes[0], 'test', 'Held-out seeds, frozen margins'),
                           (axes[1], 'test_double_noise', 'Held-out seeds at twice the noise (not exchangeable)')):
        table = result[key]['table']
        matrix = np.array([[table.get(t, {}).get(label, 0) for label in LABELS] for t in truths], float)
        share = matrix/np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        ax.imshow(share, cmap='Blues', vmin=0, vmax=1)
        for a in range(len(truths)):
            for b in range(len(LABELS)):
                if matrix[a, b]:
                    ax.text(b, a, f'{int(matrix[a, b])}', ha='center', va='center', fontsize=9,
                            color='white' if share[a, b] > .6 else '#13283b')
        ax.set_xticks(range(len(LABELS)), LABELS, rotation=35, ha='right')
        ax.set_yticks(range(len(truths)), [f'{t} ({TRUTH_TO_CLASS[t]})' for t in truths])
        ax.set_title(title)
    ax = axes[2]
    rows = [r for r in result['per_variant'] if r['split'] == 'test']
    labels = [r['name'] for r in rows]
    ax.barh(range(len(rows)), [r['correct'] / r['scenes'] for r in rows], color='#1f3a5f', label='correct')
    ax.barh(range(len(rows)), [r['abstained'] / r['scenes'] for r in rows],
            left=[r['correct'] / r['scenes'] for r in rows], color='#e2a441', label='ambiguous')
    ax.set_yticks(range(len(rows)), labels, fontsize=7)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel('share of held-out scenes')
    ax.set_title('Per scene type (held-out seeds)'); ax.legend(frameon=False, loc='lower right')
    fig.savefig(path, dpi=130)


def per_variant(rows, margins):
    groups = {}
    for r in rows:
        name = f"{r['truth']} " + ' '.join(f'{k}={v}' for k, v in r['spec'].items())
        groups.setdefault((r['split'], name, r['truth']), []).append(r)
    out = []
    for (split, name, truth), sample in sorted(groups.items()):
        labels = [decide(r, margins)['label'] for r in sample if r['status'] == 'assessed']
        out.append(dict(split=split, name=name, truth=truth, scenes=len(labels),
                        correct=sum(label == TRUTH_TO_CLASS[truth] for label in labels),
                        abstained=sum(label == 'ambiguous' for label in labels),
                        labels={label: labels.count(label) for label in set(labels)}))
    return out


def main(argv=None):
    ap = ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--output', type=Path, default=ROOT/'output/sweep_classifier')
    ap.add_argument('--workers', type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument('--seeds', type=int, default=3, help='seeds per physical scene type and split')
    ap.add_argument('--blanks', type=int, default=30, help='blank and stripes scenes per split')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--reuse-calibration', type=Path,
                    help='classifier_rows.json of an earlier run: reuse its calibration scenes and render only new test scenes')
    ap.add_argument('--test-seed-offset', type=int, default=0,
                    help='added to every test seed, so a revised rule can be checked on scenes nobody has looked at')
    args = ap.parse_args(argv)
    kinds = variants()
    seeds, blanks = args.seeds, args.blanks
    if args.quick:
        kinds = [k for i, k in enumerate(kinds) if i % 4 == 0]
        seeds, blanks = 1, 4
    jobs, reused = [], []
    if args.reuse_calibration:
        reused = [r for r in json.loads(args.reuse_calibration.read_text(encoding='utf-8')) if r['split'] == 'calibration']
    splits = [('test', 50000+args.test_seed_offset, SIGMA), ('test_double_noise', 90000+args.test_seed_offset, 2*SIGMA)]
    if not reused:
        splits.insert(0, ('calibration', 0, SIGMA))
    for split, offset, sigma in splits:
        for case, (truth, spec) in enumerate(kinds):
            for s in range(seeds):
                jobs.append((split, truth, spec, 7000+offset+100*case+s, sigma, args.quick))
        for truth in ('none', 'stripes'):
            for s in range(blanks):
                jobs.append((split, truth, {}, 7000+offset+80000+(1000 if truth == 'stripes' else 0)+s, sigma, args.quick))
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows = list(reused)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(classify_scene, jobs, chunksize=2):
            rows.append(row)
            done = len(rows)-len(reused)
            if done % 20 == 0 or done == len(jobs):
                print(f'{done}/{len(jobs)} scenes, {time.monotonic()-started:.0f} s', flush=True)
    calibration = [r for r in rows if r['split'] == 'calibration']
    margins = calibrate_margins(calibration, TARGETS)
    result = dict(margins=margins, targets=TARGETS, sigma=SIGMA, azimuths=AZIMUTHS.tolist(), elevations=ELEVATIONS.tolist(),
                  pixel_m=PIXEL_M, classifier=asdict(SweepClassifierConfig()),
                  calibration_in_sample=confusion_with_intervals(calibration, margins),
                  test=confusion_with_intervals([r for r in rows if r['split'] == 'test'], margins),
                  test_double_noise=confusion_with_intervals([r for r in rows if r['split'] == 'test_double_noise'], margins),
                  per_variant=per_variant(rows, margins), elapsed_seconds=round(time.monotonic()-started, 1),
                  calibration_reused_from=str(args.reuse_calibration) if reused else None,
                  test_seed_offset=args.test_seed_offset,
                  interpretation='Synthetic scenes from the independent relief generator under the Athena sweep. '
                                 'The test split uses new seeds with frozen margins; the double-noise split is '
                                 'deliberately unlike the calibration scenes. Not a lunar error rate.')
    (args.output/'classifier_result.json').write_text(json.dumps(result, indent=2, default=float), encoding='utf-8')
    (args.output/'classifier_rows.json').write_text(json.dumps(rows, indent=1, default=float), encoding='utf-8')
    plot(result, args.output/'classifier_confusion.png')
    print(json.dumps(dict(margins=margins, test=result['test']['per_truth'], per_call=result['test']['per_call']),
                     indent=1, default=float))


if __name__ == '__main__':
    main()
