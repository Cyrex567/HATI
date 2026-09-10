"""Reproducible, independent-renderer stress checks; not a lunar validation."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
from test_shadow_likelihood import AZ, EL, independent_scene, small_config
from src.hati_core.shadow_likelihood import search_stack, gaussian_search_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'output/validation/v25_synthetic.json')
    parser.add_argument('--seeds', type=int, default=12)
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error('--seeds must be positive')
    cfg = small_config()
    scenarios = [
        ('static_albedo', dict(static=True)),
        ('subpixel_caster', dict(height=.3,width=.6)),
        ('metre_caster', dict(height=1.,width=1.8)),
        ('static_albedo_jitter', dict(static=True,jitter=.25)),
        ('subpixel_caster_jitter', dict(height=.3,width=.6,jitter=.25)),
        ('subpixel_caster_high_noise', dict(height=.3,width=.6,noise=.04)),
    ]
    output = dict(purpose='Synthetic stress characterization only; no external ground truth',
                  seeds=list(range(900,900+args.seeds)), config_hash=cfg.hash(),
                  assumptions='56x56 pixels at 0.9 m; 7 azimuths spanning 310 degrees; independent tapered caster; anisotropic PSF; partly correlated noise',
                  matching='Highest ranked root within 2 pixels of (27.3,28.2); no score cutoff',
                  scenarios={})
    for name, settings in scenarios:
        rows = []
        for seed in output['seeds']:
            stack = independent_scene(seed,**settings)
            result = search_stack(stack,AZ,EL,settings.get('noise',.012),cfg)
            best = result['candidates'][0] if result['candidates'] else None
            distance = float(np.hypot(best['row_px']-27.3,best['col_px']-28.2)) if best else None
            rows.append(dict(seed=seed, top_score=best['score'] if best else 0.,
                             localization_error_px=distance if not settings.get('static') else None,
                             root_localized=bool(distance is not None and distance <= 2) if not settings.get('static') else None,
                             search_truncated=result['search_truncated']))
        output['scenarios'][name] = dict(settings=settings, trials=rows,
            median_top_score=float(np.median([r['top_score'] for r in rows])),
            localized_count=sum(bool(r['root_localized']) for r in rows) if not settings.get('static') else None)
        print(name, 'median score', round(output['scenarios'][name]['median_top_score'],2),
              'localized', output['scenarios'][name]['localized_count'], flush=True)
    # Full-search calibration on a separate pure Gaussian no-shadow draw. The
    # simulation grid is deliberately small; reproducibility check, not power.
    rng = np.random.default_rng(12001)
    null = rng.normal(0,.02,(7,40,40))
    cal_cfg = small_config(root_offsets_px=(0.,))
    result = search_stack(null,AZ,EL,.02,cal_cfg)
    output['gaussian_null_smoke'] = gaussian_search_calibration(
        result,null,AZ,EL,.02,cal_cfg,trials=19,seed=12002)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
