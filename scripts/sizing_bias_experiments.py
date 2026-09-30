"""Where caster heights go wrong: one factor at a time, sized the way T14 sizes.

Every trial renders one rock of known exposed height with the independent relief
generator (src/hati_core/relief_scenes.py), adds noise, and sizes the centre cell
with the adaptive refinement T14 uses (refine_cell), then applies T14's own
lower-bound rule (_caster_row). Conditions change one thing against the clean
scene: a wider optical blur than the detector assumes, registration jitter larger
than it models, small-scale ground relief the DEM cannot see, a tilted receiving
plane the DEM reports as flat, a second rock down-Sun, a boulder field, the
shape-from-shading correction T14 applies first, and noise near the detection
floor. Noise draws are shared between conditions for the same seed, so the
comparisons are paired.

Output: bias of the best height, width and T14 lower bound per condition and
height, how often the lower bound exceeds the truth, and the selection effect
among detected rocks near the floor.

    python scripts/sizing_bias_experiments.py --output output/sizing_bias --workers 4
    python scripts/sizing_bias_experiments.py --quick        # small smoke run

These are synthetic, conditional trials. They explain mechanisms; they do not
measure the bias on the real Athena stack.
"""
from argparse import ArgumentParser, Namespace
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]

from src.hati_core.adaptive_shadow import AdaptiveConfig, refine_cell  # noqa: E402
from src.hati_core.regional_shadow import RegionalConfig  # noqa: E402
from src.hati_core.relief_scenes import boulder_field, relief_feature, render_relief  # noqa: E402
from src.hati_core.rock_scenes import make_rock  # noqa: E402
from src.hati_core.sfs import solve_sfs  # noqa: E402
from src.hati_core.shadow_likelihood import ShadowConfig  # noqa: E402

# The Athena sweep as measured: map azimuths clockwise from up, elevations in degrees.
AZIMUTHS = [43.13, 43.54, 68.97, 73.66, 326.66, 354.63, 5.97, 24.94]
ELEVATIONS = [4.35, 3.59, 3.47, 3.38, 3.58, 3.63, 3.28, 3.69]
PIXEL_M = .9
SIGMA = .03              # the production assumed noise (landing_maps --noise-sigma default)
# The 24 post-landing frames of configs/post_landing_all_frames.csv: ground azimuth plus the 29.2 degree
# map rotation the pre-landing frames show (map minus ground azimuth), and Sun elevation.
_POST_GROUND = [60.7, 56.7, 52.8, 48.8, 44.9, 40.9, 37.0, 33.0, 29.0, 21.1, 5.2, 1.2, 353.3, 349.3, 325.5, 317.6,
                313.6, 305.7, 301.7, 297.8, 293.9, 289.9, 286.0, 39.4]
GEOMETRIES = {
    'athena8': (AZIMUTHS, ELEVATIONS),
    'post_landing24': ([round((a+29.2) % 360, 2) for a in _POST_GROUND],
                       [2.88, 3.24, 3.5, 3.73, 4.0, 4.24, 4.38, 4.64, 4.77, 5.11, 5.42, 5.41, 5.36, 5.3, 4.41, 3.97,
                        3.69, 3.11, 2.87, 2.46, 2.16, 1.9, 1.56, 4.04]),
}

CONDITIONS = {
    'clean': 'flat ground, production blur and registration',
    'wider_blur': 'extra optical/resampling blur of 0.8 px the detector does not model',
    'registration_1px': 'frame-to-frame registration jitter of 1 px (detector assumes 0.5 px)',
    'ripples_1deg': 'elephant-hide ripples, 6 m wavelength, 1 deg maximum slope, DEM reports flat',
    'ripples_3deg': 'the same ripples at 3 deg maximum slope',
    'tilt_away_1deg': 'ground falls 1 deg along the mean down-Sun direction, DEM reports flat',
    'tilt_toward_1deg': 'ground rises 1 deg along the mean down-Sun direction, DEM reports flat',
    'second_rock_downsun': 'a second rock of the same height 2.5 m down-Sun of the target',
    'boulder_field': 'the target inside a field of 0.2-1.0 m rocks covering 3% of the ground',
    'sfs_corrected_ripples': 'ripples at 3 deg, then the T14 linear shape-from-shading correction',
    'sfs_corrected_flat': 'flat ground, then the T14 linear shape-from-shading correction',
    'sfs_image_only_ripples': 'ripples at 3 deg, the corrected images sized on the DEM plane (no SfS surface)',
    'sfs_surface_only_ripples': 'ripples at 3 deg, the original images sized on the SfS surface',
    'no_rock_ripples_3deg': 'ripples at 3 deg and no rock: anything sized is not a caster',
    'no_rock_boulder_field': 'the boulder field without the target rock at the cell centre',
}


def mean_down_sun(azimuths=AZIMUTHS):
    """Unit (row, col) vector along the mean down-Sun direction of the sweep."""
    a = np.radians(azimuths)
    v = np.array([np.cos(a).mean(), -np.sin(a).mean()])
    return v/np.linalg.norm(v)


def add_noise(stack, sigma, rng):
    """The relief generator's own noise mix: 0.8 white plus 0.6 blurred, unit variance overall."""
    out = []
    for frame in stack:
        white = rng.normal(size=frame.shape)
        colored = ndi.gaussian_filter(rng.normal(size=frame.shape), .8)
        out.append(frame+sigma*(.8*white+.6*colored/max(colored.std(), 1e-12)))
    return np.asarray(out)


def true_plane(condition, azimuths=AZIMUTHS):
    """The ground tilt (row, column gradient) the renderer applies and the detector is not told about."""
    if condition not in ('tilt_away_1deg', 'tilt_toward_1deg'):
        return (0., 0.)
    sign = -1. if condition == 'tilt_away_1deg' else 1.
    return tuple(float(v) for v in sign*np.tan(np.radians(1.))*mean_down_sun(azimuths))


def scene(condition, height, seed, size, azimuths=AZIMUTHS, elevations=ELEVATIONS):
    """Noise-free stack for one trial, plus what the detector is told about the ground."""
    centre = ((size-1)/2+.3, (size-1)/2+.2)
    rocks, features = ([] if condition.startswith('no_rock') else [make_rock(seed, centre, height, .6, aspect=1.35)]), []
    plane = true_plane(condition, azimuths)
    registration = 0.
    down = mean_down_sun(azimuths)
    if condition in ('ripples_1deg', 'ripples_3deg', 'sfs_corrected_ripples', 'sfs_image_only_ripples',
                     'sfs_surface_only_ripples', 'no_rock_ripples_3deg'):
        features = [relief_feature('ripples', 6., 1. if condition == 'ripples_1deg' else 3., seed=seed)]
    elif condition == 'second_rock_downsun':
        offset = 2.5/PIXEL_M*down
        rocks.append(make_rock(seed+50000, (centre[0]+offset[0], centre[1]+offset[1]), height, .6, aspect=1.35))
    elif condition in ('boulder_field', 'no_rock_boulder_field'):
        features = [boulder_field(.03, d_min_m=.2, d_max_m=1., seed=seed)]
    elif condition == 'registration_1px':
        registration = 1.
    out = render_relief((size, size), azimuths, elevations, pixel_m=PIXEL_M, seed=seed, noise=0., rocks=rocks,
                        features=features, supersample=4, plane_slope_rc=plane, registration_sigma_px=registration)
    stack = out['stack']
    if condition == 'wider_blur':
        stack = np.asarray([ndi.gaussian_filter(f, .8, mode='nearest') for f in stack])
    return stack, centre


def size_one(job):
    """Render, add noise, optionally correct relief, size the centre cell; one record."""
    condition, height, seed, sigma, scales, *rest = job
    slope_search = tuple(rest[0]) if rest else ()
    geometry = rest[1] if len(rest) > 1 else 'athena8'
    az, el = GEOMETRIES[geometry]
    sc = ShadowConfig(pixel_m=PIXEL_M, registration_sigma_px=.5)
    rc = RegionalConfig()
    cfg = AdaptiveConfig(scale_factors=tuple(scales), spatial_degree=2, slope_search_deg=slope_search)
    size = 2*sc.radius_px*max(scales)+1+2*rc.cell_px+4
    stack, centre = scene(condition, height, seed, size, az, el)
    rng = np.random.default_rng(seed+900000)       # shared across conditions for this seed: paired noise
    stack = add_noise(stack, sigma, rng)
    visibility = np.ones_like(stack)
    zeros = np.zeros(stack.shape[1:])
    terrain = None
    if condition.startswith('sfs_'):
        # The T14 workstation settings: 1 px grid, smoothness 1, two passes, 3000 iterations.
        solved = solve_sfs(stack, np.ones_like(stack, bool), az, el, PIXEL_M, grid_px=1,
                           smoothness=1., dark_ratio=.5, passes=2, shadow_sigma=3., iterations=3000)
        if condition != 'sfs_surface_only_ripples':
            stack = solved['corrected']
        if condition != 'sfs_image_only_ripples':
            terrain = np.nan_to_num(solved['height_m'])
    cell = (int(round(centre[0])), int(round(centre[1])))
    result = refine_cell(stack, visibility, np.asarray(az), np.asarray(el), sigma, sc, rc, cfg,
                         cell, zeros, zeros, terrain=terrain)
    from relief_experiments import _caster_row
    ex = Namespace(sc=sc, data=dict(elevations=np.asarray(el)))
    row = _caster_row(ex, cfg, dict(result, centre=list(cell)), .3)
    assessed = [h for h in result['history'] if h.get('status') == 'assessed']
    last = assessed[-1] if assessed else None
    first = assessed[0] if assessed else None
    suffix = ('_slope_fitted' if slope_search else '')+('' if geometry == 'athena8' else '_'+geometry)
    record = dict(condition=condition+suffix, true_height_m=height, seed=seed, sigma=sigma, geometry=geometry,
                  state=result['status'], slope_search_deg=list(slope_search),
                  true_plane_slope_rc=list(true_plane(condition, az)),
                  lower_bound_m=row['height_lower_bound_m'], censored=row.get('censored'),
                  context_height_m=row['height_m'])
    for name, h in (('last', last), ('first', first)):
        if h is None:
            record.update({f'{name}_{k}': None for k in ('score', 'height_m', 'width_m', 'height_low_m', 'height_high_m', 'scale',
                                                          'receiving_slope_rc')})
            continue
        # With the slope search on, the fitted plane gradient (None when the plane was fixed at the DEM slope).
        record[f'{name}_receiving_slope_rc'] = h.get('receiving_slope_rc')
        record.update({f'{name}_score': float(h['best']['score']), f'{name}_height_m': float(h['best']['height_m']),
                       f'{name}_width_m': float(h['best']['width_m']), f'{name}_height_low_m': float(h['height_range_m'][0]),
                       f'{name}_height_high_m': float(h['height_range_m'][1]), f'{name}_scale': int(h['scale'])})
    return record


FLOOR_HEIGHTS_M = tuple(float(v) for v in np.round(np.arange(.05, 1.201, .05), 3))


def floor_one(job):
    """One clean rock near the detection floor, fitted on a fixed fine grid at the first scale.

    Refinement is switched off (warning score out of reach) so every fit, detected
    or not, chooses from the same heights; detection is then score >= 8.
    """
    from src.hati_core.adaptive_shadow import fit_patch
    height, seed, sigma = job
    sc = ShadowConfig(pixel_m=PIXEL_M, registration_sigma_px=.5)
    rc = RegionalConfig()
    cfg = AdaptiveConfig(heights_m=FLOOR_HEIGHTS_M, widths_m=(.3, .6, .9, 1.2), warning_score=1e9, spatial_degree=2)
    size = 2*sc.radius_px+1+2*rc.cell_px+4
    stack, centre = scene('clean', height, seed, size)
    stack = add_noise(stack, sigma, np.random.default_rng(seed+900000))
    r, c, radius = int(round(centre[0])), int(round(centre[1])), sc.radius_px
    patch = stack[:, r-radius:r+radius+1, c-radius:c+radius+1]
    fit = fit_patch(patch, np.ones_like(patch), np.asarray(AZIMUTHS), np.asarray(ELEVATIONS), sigma, sc, rc, cfg)
    best = fit.get('best') or {}
    return dict(condition='floor', true_height_m=height, seed=seed, sigma=sigma, state=fit['status'],
                score=best.get('score'), height_m=best.get('height_m'), width_m=best.get('width_m'),
                contrast=best.get('contrast'))


def summarise(records, key='last_height_m'):
    groups = {}
    for r in records:
        groups.setdefault((r['condition'], r['true_height_m'], r['sigma']), []).append(r)
    out = []
    for (condition, height, sigma), rows in sorted(groups.items()):
        warned = [r for r in rows if r['last_score'] is not None and r['last_score'] >= 8.]
        est = np.array([r[key] for r in warned], float)
        truth = max(height, 1e-9)
        widths = np.array([r['last_width_m'] for r in warned], float)
        bounds = np.array([r['lower_bound_m'] for r in rows if r['lower_bound_m'] is not None], float)
        out.append(dict(condition=condition, true_height_m=height, sigma=sigma, trials=len(rows), warned=len(warned),
                        median_height_m=float(np.median(est)) if est.size else None,
                        median_height_error_m=float(np.median(est-height)) if est.size else None,
                        mean_height_ratio=float(np.mean(est/truth)) if est.size and height > 0 else None,
                        median_width_m=float(np.median(widths)) if widths.size else None,
                        lower_bound_median_m=float(np.median(bounds)) if bounds.size else None,
                        lower_bound_above_truth=int(np.sum(bounds > height+.05)), lower_bound_count=int(bounds.size),
                        interval_covers_truth=int(sum(r['last_height_low_m'] <= height <= r['last_height_high_m'] for r in warned)),
                        context_supported=sum(r['state'] == 'context_supported_unvalidated' for r in rows)))
    return out


def selection_effect(records):
    """Near the floor: all fits against the fits that crossed the warning score."""
    out = []
    groups = {}
    for r in records:
        if r['condition'] == 'floor':
            groups.setdefault(r['true_height_m'], []).append(r)
    for height, rows in sorted(groups.items()):
        fitted = [r for r in rows if r['height_m'] is not None]
        warned = [r for r in fitted if r['score'] >= 8.]
        def stats(sample):
            v = np.array([r['height_m'] for r in sample], float)
            return dict(n=len(v), mean_m=float(v.mean()) if v.size else None, median_m=float(np.median(v)) if v.size else None)
        out.append(dict(true_height_m=height, sigma=rows[0]['sigma'], all_fits=stats(fitted), detected=stats(warned),
                        detection_fraction=len(warned)/max(len(fitted), 1)))
    return out


def plot(summary, records, selection, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rock = [s for s in summary if s['true_height_m'] > 0]
    conditions = list(dict.fromkeys(s['condition'] for s in rock))
    heights = sorted({s['true_height_m'] for s in rock})
    fig, axes = plt.subplots(1, 4, figsize=(28, 7), layout='constrained')
    colours = ['#1f3a5f', '#5b7fa6', '#b64262']
    for ax, key, xlabel, title, origin in (
            (axes[0], 'mean_height_ratio', 'mean fitted height / true height (warned fits, final scale)',
             'Height: one factor at a time against the clean scene', 1.),
            (axes[1], 'median_width_m', 'median fitted shadow width (m); rocks are 0.6 m wide, 0.81 m long',
             'Width: the sub-pixel dimension absorbs blur and misregistration', .6)):
        for k, height in enumerate(heights):
            values = [next((s[key] for s in rock if s['condition'] == c and s['true_height_m'] == height), None)
                      for c in conditions]
            y = np.arange(len(conditions))+(k-(len(heights)-1)/2)*.25
            ax.barh(y, [(v if v is not None else origin)-origin for v in values], height=.24, left=origin,
                    color=colours[k % 3], label=f'{height:g} m rock')
        ax.axvline(origin, color='k', lw=.8)
        ax.set_yticks(np.arange(len(conditions)), conditions)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.legend(frameon=False, loc='lower right')
    ax = axes[2]
    empty = [r for r in records if r['condition'].startswith('no_rock')]
    names = list(dict.fromkeys(r['condition'] for r in empty))
    for k, name in enumerate(names):
        rows = [r for r in empty if r['condition'] == name]
        fitted = [r['last_height_m'] for r in rows if r['last_score'] is not None and r['last_score'] >= 8.]
        bounds = [r['lower_bound_m'] for r in rows if r['lower_bound_m'] is not None]
        ax.scatter(np.full(len(fitted), k-.1), fitted, color='#1f3a5f', label='fitted height' if k == 0 else None)
        ax.scatter(np.full(len(bounds), k+.1), bounds, marker='v', color='#b64262', label='T14 lower bound' if k == 0 else None)
        ax.text(k, -.08, f'{len(fitted)}/{len(rows)} warned', ha='center', fontsize=8)
    ax.axhline(.3, color='#80909d', ls='--', lw=.8, label='0.3 m clearance')
    ax.set_xticks(range(len(names)), names)
    ax.set_ylim(-.15, None)
    ax.set_ylabel('height reported at a cell with no rock (m)')
    ax.set_title('What the sizer reports where there is no caster')
    ax.legend(frameon=False)
    ax = axes[3]
    if selection:
        h = [s['true_height_m'] for s in selection]
        ax.plot(h, h, color='k', lw=.8, label='truth')
        ax.plot(h, [s['all_fits']['mean_m'] for s in selection], 'o-', color='#5b7fa6', label='all fits')
        ax.plot(h, [s['detected']['mean_m'] for s in selection], 's-', color='#b64262', label='fits that crossed score 8')
        for s in selection:
            ax.annotate(f"{s['detection_fraction']:.0%} detected", (s['true_height_m'], s['detected']['mean_m'] or 0),
                        textcoords='offset points', xytext=(4, 6), fontsize=8)
        ax.set_xlabel('true height (m)'); ax.set_ylabel('mean fitted height (m), fixed 5 cm grid, first scale')
        ax.set_title(f'Near the floor: all fits against detected fits (noise sigma {selection[0]["sigma"]})')
        ax.legend(frameon=False)
    fig.savefig(path, dpi=120)


def merge(paths, output):
    """Combine the records of several runs into one result and figure, recomputing every summary."""
    parts = [json.loads(Path(p).read_text(encoding='utf-8')) for p in paths]
    records = [r for part in parts for r in part['records']]
    conditions = {}
    for part in parts:
        conditions.update(part['conditions'])
    summary = summarise([r for r in records if r['condition'] != 'floor'])
    selection = selection_effect(records)
    # Parts may come from different sweeps; every record names its own, and the header lists them all.
    geometries = sorted({r.get('geometry', 'athena8') for r in records})
    payload = dict(parts[0], conditions=conditions, summary=summary, selection_near_floor=selection, records=records,
                   geometry=geometries[0] if len(geometries) == 1 else geometries,
                   azimuths=GEOMETRIES[geometries[0]][0] if len(geometries) == 1 else {g: GEOMETRIES[g][0] for g in geometries},
                   elevations=GEOMETRIES[geometries[0]][1] if len(geometries) == 1 else {g: GEOMETRIES[g][1] for g in geometries},
                   merged_from=[str(p) for p in paths],
                   elapsed_seconds=round(sum(part.get('elapsed_seconds', 0) for part in parts), 1))
    output.mkdir(parents=True, exist_ok=True)
    (output/'sizing_bias.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    plot(summary, records, selection, output/'sizing_bias.png')
    return payload


def main(argv=None):
    ap = ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--output', type=Path, default=ROOT/'output/sizing_bias')
    ap.add_argument('--workers', type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--floor-seeds', type=int, default=40)
    ap.add_argument('--quick', action='store_true', help='two seeds, two conditions: a smoke run')
    ap.add_argument('--conditions', help='comma-separated subset of conditions; the floor trials run only without it')
    ap.add_argument('--heights', help='comma-separated true heights in metres (default 0.3,0.6,1.2)')
    ap.add_argument('--merge', nargs='+', type=Path, help='combine earlier sizing_bias.json files into --output and stop')
    ap.add_argument('--scales', default='1,2,4', help='adaptive scale factors (default 1,2,4)')
    ap.add_argument('--slope-search', help='receiving-slope offsets in degrees profiled with height, e.g. -1,-0.5,0,0.5,1')
    ap.add_argument('--geometry', default='athena8', choices=sorted(GEOMETRIES),
                    help='athena8: the pre-landing sweep; post_landing24: the 24 post-landing frames (wider azimuth and elevation)')
    args = ap.parse_args(argv)
    if args.merge:
        merge(args.merge, args.output)
        return
    conditions = list(CONDITIONS)
    heights = [.3, .6, 1.2]
    seeds = args.seeds
    floor_heights, floor_seeds, floor_sigma = [.15, .2, .25, .3, .4], args.floor_seeds, .1
    if args.quick:
        conditions, heights, seeds, floor_heights, floor_seeds = ['clean', 'tilt_away_1deg', 'no_rock_ripples_3deg'], [.6], 2, [.2], 6
    if args.conditions:
        conditions = [c.strip() for c in args.conditions.split(',')]
        unknown = [c for c in conditions if c not in CONDITIONS]
        if unknown:
            ap.error(f'unknown conditions {unknown}')
        floor_heights = []
    if args.heights:
        heights = [float(v) for v in args.heights.split(',')]
    scales = tuple(int(v) for v in args.scales.split(','))
    search = tuple(float(v) for v in args.slope_search.split(',')) if args.slope_search else ()
    jobs = [(c, 0. if c.startswith('no_rock') else h, 1000+s, SIGMA, scales, search, args.geometry)
            for c in conditions for h in (heights[:1] if c.startswith('no_rock') else heights) for s in range(seeds)]
    # Near the floor the selection effect matters; a single scale and a fixed grid keep this cheap and fair.
    floor_jobs = [(h, 5000+s, floor_sigma) for h in floor_heights for s in range(floor_seeds)]
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    records = []
    total = len(jobs)+len(floor_jobs)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(size_one, jobs, chunksize=1):
            records.append(record)
            if len(records) % 10 == 0 or len(records) == len(jobs):
                print(f'{len(records)}/{total} trials, {time.monotonic()-started:.0f} s', flush=True)
        for record in pool.map(floor_one, floor_jobs, chunksize=4):
            records.append(record)
            if len(records) % 50 == 0 or len(records) == total:
                print(f'{len(records)}/{total} trials, {time.monotonic()-started:.0f} s', flush=True)
    summary = summarise([r for r in records if r['condition'] != 'floor'])
    selection = selection_effect(records)
    az, el = GEOMETRIES[args.geometry]
    payload = dict(conditions={c: CONDITIONS[c] for c in conditions}, geometry=args.geometry, azimuths=az, elevations=el,
                   pixel_m=PIXEL_M, sigma=SIGMA, detector=dict(psf_sigma_px=.6, registration_sigma_px=.5, spatial_degree=2),
                   summary=summary, selection_near_floor=selection, records=records,
                   elapsed_seconds=round(time.monotonic()-started, 1),
                   interpretation='Synthetic paired trials; mechanisms, not a measured bias on real images. '
                                  'Heights are equivalent rectangular-shadow heights from the final adaptive pass.')
    (args.output/'sizing_bias.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    plot(summary, records, selection, args.output/"sizing_bias.png")
    print(json.dumps(dict(summary=summary, selection_near_floor=selection), indent=1))


if __name__ == '__main__':
    main()
