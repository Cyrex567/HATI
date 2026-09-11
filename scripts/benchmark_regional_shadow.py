"""Independent-renderer stress benchmark of regional evidence, not lunar validation."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from test_shadow_likelihood import independent_scene,AZ,EL
from src.hati_core.shadow_likelihood import ShadowConfig
from src.hati_core.regional_shadow import RegionalConfig,assess_regions


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds',type=int,default=8)
    ap.add_argument('--output',type=Path,default=ROOT/'output/validation/v253_regional.json')
    args=ap.parse_args()
    if args.seeds<1: ap.error('positive seed count required')
    cfg=RegionalConfig()
    scenarios=[('static',dict(static=True),7),('small_caster',dict(height=.3,width=.6),7),
        ('static_jitter',dict(static=True,jitter=.25),7),
        ('small_caster_jitter',dict(height=.3,width=.6,jitter=.25),7),
        ('small_caster_narrow_sweep',dict(height=.3,width=.6),4),
        ('small_caster_high_noise',dict(height=.3,width=.6,noise=.04),7)]
    result=dict(purpose='Independent tapered-caster synthetic stress test; not field calibration',
        seeds=list(range(1400,1400+args.seeds)),shape=[40,40],root=[20.3,20.2],regional=asdict(cfg),
        matching='Any retained score >=8 root within 2 px; false roots beyond 3 px of the injected root',
        registration_assumption='Template blur and first-order albedo covariance vary together; not an isolated covariance ablation',
        dem_scope='Flat receiving plane and fully observed visibility; DEM model tested separately',
        renderer_sha256=hashlib.sha256((ROOT/'tests/test_shadow_likelihood.py').read_bytes()).hexdigest(),scenarios={})
    for name,settings,n in scenarios:
        result['scenarios'][name]={}
        for reg in (0.,.25):
            rows=[]; sc=ShadowConfig(registration_sigma_px=reg)
            for seed in result['seeds']:
                stack=independent_scene(seed,shape=(40,40),root=(20.3,20.2),**settings)[:n]
                out=assess_regions(stack,AZ[:n],EL[:n],settings.get('noise',.012),sc,cfg)
                distances=[float(np.hypot(p['row_px']-20.3,p['col_px']-20.2)) for p in out['candidates']]
                static=settings.get('static',False)
                rows.append(dict(seed=seed,maximum_score=float(np.nanmax(out['score'])),
                    detected=None if static else any(d<=2 for d in distances),
                    false_roots=len(distances) if static else sum(d>3 for d in distances),
                    cells_visited=out['cells_visited'],cells_assessed=out['cells_assessed']))
            summary=dict(settings=settings,frames=n,azimuth_span_deg=float(AZ[n-1]-AZ[0]),
                         shadow_config_hash=sc.hash(),trials=rows,
                         median_maximum_score=float(np.median([r['maximum_score'] for r in rows])),
                         detection_count=None if settings.get('static') else sum(r['detected'] for r in rows),
                         false_root_total=sum(r['false_roots'] for r in rows))
            result['scenarios'][name][str(reg)]=summary
            print(name,reg,'median max',round(summary['median_maximum_score'],2),
                  'detected',summary['detection_count'],'false roots',summary['false_root_total'],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__': main()
