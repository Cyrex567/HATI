"""Compare HATI on the held-out post-landing stack with the pre-landing campaign.

Both inputs are campaign folders that ran T1, T12, T13 and T14 on the same grid:
the pre-landing campaign, and one on frames acquired after the landing
(ingest_sweep.py --after, landing_maps.py --after). Rocks do not move, so whatever
is physical should repeat between the two independent image sets: the measured
casters, the warnings after the relief correction, the shape-from-shading detail
and the rock-or-relief labels. Each comparison is set against shifted copies of the
same maps, which keep their clustering, so repetition by chance is measured rather
than assumed. The lander's neighbourhood is described on its own and left out of
the comparisons, because the landing changed the surface there.

The plan and its thresholds are fixed in configs/post_landing_check.json, written
before any post-landing frame was analysed; this script only applies them.

Run:  python scripts/post_landing_check.py --pre <campaign> --post <campaign> --output <folder>
"""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
LABELS = ('rock_like', 'relief_like', 'ambiguous', 'none')


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_campaign(folder):
    """The arrays and records one campaign folder contributes to the comparison."""
    folder = Path(folder); st = folder/'stages'
    with zipfile.ZipFile(folder/'inputs/hati_diagnostic_bundle.zip') as z:
        with np.load(io.BytesIO(z.read('aligned_stack.npz')), allow_pickle=False) as n:
            stack = n['stack'].astype(float)
            transform = n['transform'].astype(float)
            frames = [str(v) for v in n['frame_ids']]
            azimuths, elevations = n['azimuths'].astype(float), n['elevations'].astype(float)
        run = json.loads(z.read('source_run.json'))
    t14 = _json(st/'T14/result.json')
    corrected, baseline = np.load(st/'T14/relief_corrected/regional.npz'), np.load(st/'T1/baseline/regional.npz')
    sfs = np.load(st/'T14/sfs.npz')
    assumed = float(run['shadow_configuration']['noise_sigma'])
    measured = float((t14.get('subpixel_casters') or {}).get('sigma') or t14['residual_scale_after']['pooled_sigma'])
    table = corrected['cell_table']
    centre = (table[:, 4], table[:, 5])
    cells = dict(row=table[:, 4], col=table[:, 5],
                 status=corrected['status'][centre], score=corrected['score'][centre]*assumed/measured,
                 base_status=baseline['status'][centre], base_score=baseline['score'][centre])
    height = np.where(sfs['common'], sfs['height_m'], np.nan).astype(float)
    return dict(folder=str(folder), frames=frames, azimuths=azimuths, elevations=elevations,
                transform=transform, shape=stack.shape[1:], mean=np.nanmean(stack, axis=0),
                pixel_m=float(run['image_posting_m']),
                touchdown=(float(run['counterfactual']['row_px']), float(run['counterfactual']['col_px'])),
                sigma_assumed=assumed, sigma_measured=measured, cells=cells,
                corrected_status=corrected['status'], sfs_height=height, sfs_slope=np.where(sfs['common'], sfs['slope_deg'], np.nan),
                casters=_json(st/'T14/subpixel_casters.json'), labels=_json(st/'T13/athena_cells.json'),
                t12=_json(st/'T12/result.json'), t14=t14)


def _shifts(rng, cfg, pixel_m):
    c = cfg['chance']
    angle = rng.uniform(0, 2*np.pi, c['trials'])
    length = rng.uniform(c['shift_min_m'], c['shift_max_m'], c['trials'])/pixel_m
    return np.column_stack([length*np.sin(angle), length*np.cos(angle)])


def _summary(observed, chance, n):
    chance = np.asarray([v for v in chance if v is not None and np.isfinite(v)])
    sd = float(chance.std(ddof=1)) if len(chance) > 1 else float('nan')
    z = (observed-chance.mean())/sd if len(chance) > 1 and sd > 0 and observed is not None else None
    return dict(n=int(n), observed=observed, chance_mean=float(chance.mean()) if len(chance) else None,
                chance_sd=sd if np.isfinite(sd) else None, z=None if z is None else float(z),
                chance_at_or_above=float(np.mean(chance >= observed)) if len(chance) and observed is not None else None)


def point_repeat(a, b, usable_b, keep, radius_px, shifts, shape):
    """Share of the points a that have a point b within radius, against shifted copies of a.

    A point of a counts only where b's run could have found it: inside the window,
    outside the disturbed disc and on a cell b's run assessed."""
    a = np.asarray(a, float).reshape(-1, 2); b = np.asarray(b, float).reshape(-1, 2)
    def eligible(p):
        r, c = np.round(p[:, 0]).astype(int), np.round(p[:, 1]).astype(int)
        inside = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
        ok = np.zeros(len(p), bool)
        ok[inside] = usable_b[r[inside], c[inside]] & keep[r[inside], c[inside]]
        return p[ok]
    tree = cKDTree(b) if len(b) else None
    def share(p):
        p = eligible(p)
        if not len(p) or tree is None:
            return None, len(p)
        d, _ = tree.query(p, distance_upper_bound=radius_px)
        return float(np.mean(np.isfinite(d))), len(p)
    observed, n = share(a)
    return _summary(observed, [share(a+s)[0] for s in shifts], n)


def height_agreement(pre, post, radius_px):
    """Lower bounds of casters found in both stacks, matched to the nearest partner."""
    a = [c for c in pre if c['height_lower_bound_m'] is not None]
    b = [c for c in post if c['height_lower_bound_m'] is not None]
    if not a or not b:
        return dict(pairs=0)
    tree = cKDTree([[c['row_px'], c['col_px']] for c in b])
    d, k = tree.query([[c['row_px'], c['col_px']] for c in a], distance_upper_bound=radius_px)
    pairs = [(a[i]['height_lower_bound_m'], b[j]['height_lower_bound_m']) for i, j in enumerate(k) if np.isfinite(d[i])]
    if not pairs:
        return dict(pairs=0)
    x, y = np.asarray(pairs).T
    rho = spearmanr(x, y).correlation if len(pairs) > 2 else None
    return dict(pairs=len(pairs), median_absolute_difference_m=float(np.median(np.abs(x-y))),
                median_difference_post_minus_pre_m=float(np.median(y-x)),
                spearman=None if rho is None or not np.isfinite(rho) else float(rho), values=np.asarray(pairs).tolist())


def cell_repeat(pre, post, keep_cells, shifts_px, cell_px=4):
    """P(post warns | pre warns) on cells both runs assessed, against shifted pre maps."""
    rows, cols = pre['cells']['row'], pre['cells']['col']
    r0, c0 = rows.min(), cols.min()
    grid = lambda v, fill: _grid(v, rows, cols, r0, c0, cell_px, fill)
    pre_ok, post_ok = grid(pre['cells']['status'] == 1, False), grid(post['cells']['status'] == 1, False)
    pre_warn, post_warn = grid(pre['cells']['score'] >= 8, False), grid(post['cells']['score'] >= 8, False)
    keep = grid(keep_cells, False)
    def conditional(di, dj):
        both = _window(pre_ok & keep & pre_warn, di, dj) & post_ok & keep
        return (float(post_warn[both].mean()) if both.any() else None), int(both.sum())
    observed, n = conditional(0, 0)
    chance = [conditional(int(round(s[0]/cell_px)), int(round(s[1]/cell_px)))[0] for s in shifts_px]
    base = post_ok & keep & pre_ok
    result = _summary(observed, chance, n)
    result['post_warning_rate'] = float(post_warn[base].mean()) if base.any() else None
    result['pre_warning_rate'] = float(pre_warn[base].mean()) if base.any() else None
    return result


def _grid(values, rows, cols, r0, c0, cell_px, fill):
    i, j = (rows-r0)//cell_px, (cols-c0)//cell_px
    out = np.full((i.max()+1, j.max()+1), fill, dtype=np.asarray(values).dtype)
    out[i, j] = values
    return out


def _window(a, di, dj):
    """a shifted by (di, dj) cells onto its own grid; uncovered cells are False."""
    out = np.zeros_like(a)
    h, w = a.shape
    rs, re_ = max(0, di), min(h, h+di); cs, ce = max(0, dj), min(w, w+dj)
    out[rs:re_, cs:ce] = a[rs-di:re_-di, cs-dj:ce-dj]
    return out


def _lowpass(a, sigma):
    m = np.isfinite(a)
    num, den = ndi.gaussian_filter(np.where(m, a, 0.), sigma), ndi.gaussian_filter(m.astype(float), sigma)
    out = num/np.maximum(den, 1e-12)
    out[den < 1e-3] = np.nan
    return out


def field_correlation(a, b, keep, shifts, step=2):
    """Pearson correlation of two fields on shared finite pixels, against shifted copies of b."""
    def corr(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        return (float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 100 else None), int(m.sum())
    a = np.where(keep, a, np.nan)[::step, ::step]
    observed, n = corr(a, np.where(keep, b, np.nan)[::step, ::step])
    chance = [corr(a, ndi.shift(np.where(keep, b, np.nan), s, order=0, mode='constant', cval=np.nan)[::step, ::step])[0]
              for s in shifts]
    return _summary(observed, chance, n)


def label_agreement(pre, post, keep):
    """Rock-or-relief labels on sampled cells present in both runs, with Cohen's kappa."""
    def index(cells):
        return {(c['row_px'], c['col_px']): c['label'] for c in cells if c.get('label') in LABELS
                and keep[int(c['row_px']), int(c['col_px'])]}
    a, b = index(pre), index(post)
    shared = sorted(set(a) & set(b))
    if not shared:
        return dict(cells=0)
    x = np.array([LABELS.index(a[k]) for k in shared]); y = np.array([LABELS.index(b[k]) for k in shared])
    table = np.zeros((4, 4), int)
    np.add.at(table, (x, y), 1)
    agree = float(np.trace(table)/table.sum())
    expected = float((table.sum(1) @ table.sum(0))/table.sum()**2)
    kappa = (agree-expected)/(1-expected) if expected < 1 else None
    return dict(cells=len(shared), agreement=agree, chance_agreement=expected, kappa=kappa,
                table={LABELS[i]: dict(zip(LABELS, map(int, table[i]))) for i in range(4)})


def neighbourhood(c, radii_m):
    """What one run reports around the lander: assessment, warnings, casters, labels."""
    tr, tc = c['touchdown']; px = c['pixel_m']
    dist = np.hypot(c['cells']['row']-tr, c['cells']['col']-tc)*px
    out = {}
    bands = [('lander', 0, radii_m['lander']), ('neighbourhood', 0, radii_m['neighbourhood']),
             ('annulus', radii_m['neighbourhood'], radii_m['disturbed'])]
    for name, lo, hi in bands:
        s = (dist >= lo) & (dist <= hi)
        assessed = s & (c['cells']['status'] == 1)
        base = s & (c['cells']['base_status'] == 1)
        casters = [dict(distance_m=round(k['distance_to_touchdown_m'], 1), bound_m=k['height_lower_bound_m'],
                        estimate_m=k['height_m'], state=k['state'], label=k.get('relief_check'))
                   for k in c['casters'] if lo <= k['distance_to_touchdown_m'] <= hi]
        labels = [k['label'] for k in c['labels']
                  if k.get('label') and lo <= np.hypot(k['row_px']-tr, k['col_px']-tc)*px <= hi]
        out[name] = dict(
            cells=int(s.sum()), baseline_assessed=int(base.sum()),
            baseline_warning=int((base & (c['cells']['base_score'] >= 8)).sum()),
            baseline_max_score=float(c['cells']['base_score'][base].max()) if base.any() else None,
            corrected_assessed=int(assessed.sum()),
            corrected_warning=int((assessed & (c['cells']['score'] >= 8)).sum()),
            corrected_quiet=int((assessed & (c['cells']['score'] < 8)).sum()),
            casters=sorted(casters, key=lambda k: k['distance_m']),
            labels={k: labels.count(k) for k in LABELS})
    return out


def _keep(shape, touchdown, pixel_m, radius_m):
    rr, cc = np.indices(shape)
    return np.hypot(rr-touchdown[0], cc-touchdown[1])*pixel_m > radius_m


def compare(pre, post, cfg):
    if pre['shape'] != post['shape'] or not np.allclose(pre['transform'], post['transform'], rtol=0, atol=1e-6):
        raise ValueError('the two campaigns must share one grid')
    if set(pre['frames']) & set(post['frames']):
        raise ValueError('the two stacks share frames; the comparison needs independent image sets')
    rng = np.random.default_rng(cfg['chance']['seed'])
    px = pre['pixel_m']; shape = pre['shape']
    keep = _keep(shape, pre['touchdown'], px, cfg['radii_m']['disturbed'])
    shifts = _shifts(rng, cfg, px)
    radius = cfg['match_radius_m']/px
    def points(c, minimum=None):
        return [[k['row_px'], k['col_px']] for k in c['casters'] if k['height_lower_bound_m'] is not None
                and (minimum is None or k['height_lower_bound_m'] >= minimum-1e-9)]
    usable = {k: c['corrected_status'] == 1 for k, c in (('pre', pre), ('post', post))}
    casters = {}
    for label, minimum in (('bounded', None), ('above_clearance', cfg['clearance_m'])):
        casters[label] = dict(pre_to_post=point_repeat(points(pre, minimum), points(post, minimum), usable['post'], keep, radius, shifts, shape),
                              post_to_pre=point_repeat(points(post, minimum), points(pre, minimum), usable['pre'], keep, radius, shifts, shape))
    in_keep = lambda c: [k for k in c['casters'] if keep[int(k['row_px']), int(k['col_px'])]]
    heights = height_agreement(in_keep(pre), in_keep(post), radius)
    keep_cells = keep[pre['cells']['row'], pre['cells']['col']]
    warnings = cell_repeat(pre, post, keep_cells, shifts)
    sigma = cfg['detail_sigma_px']
    detail = field_correlation(pre['sfs_height']-_lowpass(pre['sfs_height'], sigma),
                               post['sfs_height']-_lowpass(post['sfs_height'], sigma), keep, shifts)
    slope = field_correlation(pre['sfs_slope'], post['sfs_slope'], keep, shifts)
    labels = label_agreement(pre['labels'], post['labels'], keep)
    th = cfg['thresholds']
    ok = lambda s: s['z'] is not None and s['z'] >= th['repeat_z_min']
    lander = neighbourhood(post, cfg['radii_m'])
    verdicts = {k: bool(val) for k, val in {
        'E1_casters_repeat': ok(casters['bounded']['pre_to_post']) and ok(casters['bounded']['post_to_pre']),
        'E2_heights_agree': heights.get('pairs', 0) > 0 and heights['median_absolute_difference_m'] <= th['height_agreement_median_m'],
        'E3_relief_detail_repeats': detail['observed'] is not None and detail['observed'] >= th['sfs_detail_correlation_min'] and ok(detail),
        'E4_warnings_repeat': ok(warnings),
        'E5_no_quiet_reading_at_lander': lander['lander']['corrected_quiet'] == 0}.items()}
    return dict(casters=casters, heights=heights, warnings=warnings, sfs_detail=detail, sfs_slope=slope,
                labels=labels, verdicts=verdicts,
                lander=dict(pre=neighbourhood(pre, cfg['radii_m']), post=lander),
                stacks={k: dict(frames=c['frames'], azimuths_map=c['azimuths'].tolist(), elevations=c['elevations'].tolist(),
                                sigma_assumed=c['sigma_assumed'], sigma_after_correction=c['sigma_measured'],
                                t12_measured_sigma=c['t12'].get('measured_pooled_sigma'),
                                t12_sun_explained=(c['t12'].get('relief_consistency') or {}).get('explained_true_geometry'),
                                sfs_explained=c['t14']['sfs']['explained_fraction'],
                                warnings_after_correction=c['t14']['exceedance']['after_measured_sigma'],
                                casters=(c['t14'].get('subpixel_casters') or {}).get('with_warning_evidence'))
                        for k, c in (('pre', pre), ('post', post))})


def figure(pre, post, result, cfg, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    tr, tc = pre['touchdown']; px = pre['pixel_m']; half = int(round(72/px))
    crop = np.s_[int(tr)-half:int(tr)+half, int(tc)-half:int(tc)+half]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
    ratio = np.log(np.clip(post['mean'], 1e-3, None)/np.clip(pre['mean'], 1e-3, None))
    for ax, img, title, kw in ((axes[0, 0], pre['mean'], 'Mean image before the landing', dict(cmap='gray')),
                               (axes[0, 1], post['mean'], 'Mean image after the landing', dict(cmap='gray')),
                               (axes[0, 2], ratio, 'log(after / before)', dict(cmap='RdBu_r', vmin=-.6, vmax=.6))):
        a = img[crop]
        if 'vmin' not in kw:
            kw = dict(kw, vmin=np.nanpercentile(a, 1), vmax=np.nanpercentile(a, 99))
        ax.imshow(a, **kw)
        for r in (cfg['radii_m']['lander'], cfg['radii_m']['neighbourhood']):
            ax.add_patch(Circle((half, half), r/px, fill=False, ec='#C1121F', lw=1, ls='--'))
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    ax = axes[1, 0]
    ax.imshow(np.nanmean([pre['mean'], post['mean']], axis=0), cmap='gray', alpha=.5)
    for c, colour, marker, name in ((pre, '#2E6DB4', 'o', 'before'), (post, '#C1121F', 'x', 'after')):
        p = np.array([[k['col_px'], k['row_px']] for k in c['casters'] if k['height_lower_bound_m'] is not None]).reshape(-1, 2)
        ax.scatter(p[:, 0], p[:, 1], s=6, c=colour, marker=marker, lw=.6, label=f'casters {name} ({len(p)})')
    ax.add_patch(Circle((tc, tr), cfg['radii_m']['disturbed']/px, fill=False, ec='k', lw=1, ls='--'))
    ax.set_title('Measured casters in both stacks', fontsize=10); ax.legend(fontsize=7, loc='lower left')
    ax.set_xticks([]); ax.set_yticks([])
    ax = axes[1, 1]
    values = np.asarray(result['heights'].get('values') or []).reshape(-1, 2)
    if len(values):
        ax.scatter(values[:, 0], values[:, 1], s=10, c='#12937D', alpha=.7)
        top = max(1.5, values.max()+.1); ax.plot([0, top], [0, top], color='k', lw=.8)
        ax.set_xlim(0, top); ax.set_ylim(0, top)
    ax.set_xlabel('height lower bound before (m)'); ax.set_ylabel('after (m)')
    ax.set_title(f"Matched casters: {result['heights'].get('pairs', 0)}", fontsize=10)
    ax = axes[1, 2]
    rows = [('casters, before in after', result['casters']['bounded']['pre_to_post']),
            ('casters, after in before', result['casters']['bounded']['post_to_pre']),
            ('warnings after correction', result['warnings']), ('relief detail', result['sfs_detail'])]
    for i, (name, s) in enumerate(rows):
        if s.get('chance_mean') is not None:
            ax.errorbar(s['chance_mean'], i, xerr=2*(s['chance_sd'] or 0), fmt='o', color='#8B97A5', capsize=3)
        if s.get('observed') is not None:
            ax.plot(s['observed'], i, 'D', color='#C1121F')
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8); ax.invert_yaxis()
    ax.set_title('Observed (red) against shifted copies (grey, mean and 2 sd)', fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def report(result, cfg, path):
    v = result['verdicts']; s = result['stacks']
    lines = ['# Post-landing check', '',
             f"Plan fixed on {cfg['declared'].split(',')[0]}; window {cfg['window']['after']} to {cfg['window']['before']}.",
             '', '## Expectations', '']
    for text, key in zip(cfg['expectations'], v):
        lines.append(f"- {'MET' if v[key] else 'NOT MET'}: {text}")
    c = result['casters']['bounded']
    lines += ['', '## Numbers', '',
              f"Frames: {len(s['pre']['frames'])} before, {len(s['post']['frames'])} after. Noise after correction: "
              f"{s['pre']['sigma_after_correction']:.4f} before, {s['post']['sigma_after_correction']:.4f} after.",
              f"Casters found again: {c['pre_to_post']['observed']} of the pre-landing set "
              f"(shifted copies {c['pre_to_post']['chance_mean']}, z {c['pre_to_post']['z']}); "
              f"{c['post_to_pre']['observed']} of the post-landing set (z {c['post_to_pre']['z']}).",
              f"Heights of {result['heights'].get('pairs', 0)} matched casters: median absolute difference "
              f"{result['heights'].get('median_absolute_difference_m')} m.",
              f"Warnings after correction: P(after | before) {result['warnings']['observed']}, base rate "
              f"{result['warnings']['post_warning_rate']}, z {result['warnings']['z']}.",
              f"Relief detail correlation {result['sfs_detail']['observed']} (z {result['sfs_detail']['z']}); "
              f"slope correlation {result['sfs_slope']['observed']}.",
              f"Rock-or-relief labels on {result['labels'].get('cells', 0)} shared cells: agreement "
              f"{result['labels'].get('agreement')}, kappa {result['labels'].get('kappa')}.",
              '', '## At the lander (after the landing)', '']
    for band, d in result['lander']['post'].items():
        lines.append(f"- {band}: {d['cells']} cells, {d['corrected_assessed']} assessed after correction, "
                     f"{d['corrected_warning']} warn, {d['corrected_quiet']} quiet; {len(d['casters'])} measured casters.")
    lines += ['', 'Counts of cells are not counts of rocks. Nothing here validates a landing decision.', '']
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pre', type=Path, required=True, help='pre-landing campaign folder')
    ap.add_argument('--post', type=Path, required=True, help='post-landing campaign folder')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--config', type=Path, default=ROOT/'configs/post_landing_check.json')
    args = ap.parse_args()
    cfg = _json(args.config)
    pre, post = load_campaign(args.pre), load_campaign(args.post)
    result = compare(pre, post, cfg)
    result['provenance'] = dict(config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                pre=str(args.pre), post=str(args.post))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'post_landing_check.json').write_text(json.dumps(result, indent=2, default=float), encoding='utf-8')
    report(result, cfg, args.output/'REPORT.md')
    figure(pre, post, result, cfg, args.output/'post_landing_check.png')
    print((args.output/'REPORT.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
