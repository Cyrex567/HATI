"""Explain an existing run and export/test cached intensities, without ISIS."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import rasterio
from pyproj import CRS
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from landing_maps import clean_json
from sweep_products import load_sweep,on_reference
from src.hati_core.warning_attribution import describe_warning
from src.hati_core.shadow_diagnostics import diagnose_stack
from src.hati_core.shadow_likelihood import ShadowConfig


def inspect_run(folder):
    folder=Path(folder); run=json.loads((folder/'run.json').read_text())
    with rasterio.open(folder/'shadow_score.tif') as src:
        grid=(src.shape,src.transform,src.crs)
    def read(name):
        with rasterio.open(folder/(name+'.tif')) as src:
            if (src.shape,src.transform,src.crs)!=grid: raise ValueError('diagnostic rasters must share a grid')
            return src.read(1).astype(float)
    maps={k:read(k+'_hazard') for k in ('terrain','shadow','fused')}
    raw=read('shadow_score'); status=read('shadow_status'); common=read('shadow_common_fraction')
    with (folder/'candidates.csv').open() as f: candidates=list(csv.DictReader(f))
    roots={k:read('shadow_best_root_'+k+'_px') if (folder/('shadow_best_root_'+k+'_px.tif')).exists() else None for k in ('row','col')}
    cf=run['counterfactual']; cfg=run['landing']
    report=describe_warning(maps,raw,status,common,cf['row_px'],cf['col_px'],run['image_posting_m'],
        cfg['footprint_diameter_m']/2+cfg['navigation_margin_m'],threshold=cfg['shadow_score_scale'],
        candidates=candidates,root_row=roots['row'],root_col=roots['col'])
    report['qualification']=dict(footprint_resolved=run['footprint_resolved'],
        shadow_envelope_qualified_fraction=float(np.mean(read('shadow_envelope_ok')>=1)),
        both_modules_qualified_fraction=float(np.mean(read('fusion_status')==1)))
    report['source_run_sha256']=hashlib.sha256((folder/'run.json').read_bytes()).hexdigest()
    return run,grid,report


def write_summary(output,report):
    raw=report['raw_cell'];source=report['source'];scope=report['maps']['shadow']['threshold_exceedance_fraction_available']
    text=['# HATI warning attribution','',f"Origin: **{report['warning_origin']}**.",'',
        f"Direct cell assessed: **{raw['assessed']}**; common support: {raw['common_fraction']}; required: {raw['required_common_fraction']}.",
        f"Buffered shadow exceedance among available pixels: {scope:.2%}." if scope is not None else 'Buffered shadow map unavailable.',
        '',report['interpretation'],'']
    if source:
        text += [f"Source resolution: {source['root_resolution']}; raw regional maximum: {source['score']:.4f}."]
        if source['root']:
            text += [f"Winning root distance from fixed location: **{source['root']['distance_from_test_location_m']:.3f} m**.",
                     f"Within nominal radius: {source['root']['within_nominal_radius']}."]
    text += ['',report['buffer']['meaning'], '',
        'The diagnostic brightness model is evaluated separately. Its results do not replace production maps, calibrate thresholds or establish landing clearance.']
    (output/'SUMMARY.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--manifest',type=Path,default=ROOT/'data/sweep/manifest.json')
    ap.add_argument('--report-only',action='store_true',help='attribute existing maps without reading image cubes')
    ap.add_argument('--step-px',type=int,default=64,help='fixed diagnostic patch-centre spacing; no image downsampling')
    args=ap.parse_args(); output=args.output or args.run_dir/'diagnostics_v254';output.mkdir(parents=True,exist_ok=True)
    run,grid,report=inspect_run(args.run_dir)
    (output/'warning_attribution.json').write_text(json.dumps(clean_json(report),indent=2,allow_nan=False),encoding='utf-8')
    write_summary(output,report)
    print(f"Warning origin: {report['warning_origin']}",flush=True)
    if args.report_only:
        print(f'Existing products assessed: {output}'); return
    expected=run['provenance']['manifest_sha256']
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest()!=expected:
        raise ValueError('manifest differs from the analysed run; use its saved manifest')
    half=int(run['provenance']['arguments']['half'])
    sweep=load_sweep(args.manifest,half,run['provenance']['arguments']['before'])
    same_crs=CRS.from_user_input(sweep['crs']).equals(CRS.from_user_input(grid[2]),ignore_axis_order=True)
    if sweep['stack'].shape[1:]!=grid[0] or not same_crs or not np.allclose(tuple(sweep['transform']),tuple(grid[1]),rtol=0,atol=1e-8):
        raise ValueError('cached sweep does not match output grid')
    if [p['pid'] for p in sweep['frames']]!=[p['pid'] for p in run['frames']]:
        raise ValueError('cached frame identities/order differ')
    if not np.allclose(sweep['azimuths'],run['azimuths_map'],atol=1e-7,rtol=0) or not np.allclose(sweep['elevations'],run['elevations'],atol=1e-7,rtol=0):
        raise ValueError('cached geometry differs from saved run')
    visibility=[]
    for i in range(len(sweep['stack'])):
        with rasterio.open(args.run_dir/f'dem_frame_{i:02d}_visible.tif') as src:
            if (src.shape,src.transform,src.crs)!=grid: raise ValueError('visibility grid differs')
            visibility.append(src.read(1))
    slopes=[]
    diameter=float(run['effective_terrain_diameter_m'])
    for axis in ('row','col'):
        with rasterio.open(args.run_dir/f'dem_{diameter}m_slope_{axis}.tif') as src:
            context=dict(transform=src.transform,crs=src.crs)
            slopes.append(on_reference(src.read(1),context,grid[1],grid[2],grid[0],nearest=True))
    stackfile=output/'aligned_stack.npz'
    np.savez_compressed(stackfile,stack=sweep['stack'],azimuths=sweep['azimuths'],elevations=sweep['elevations'],
        visibility=np.array(visibility),slope_row=slopes[0],slope_col=slopes[1],
        transform=np.asarray(tuple(grid[1])),crs=np.asarray(grid[2].to_wkt()),
        frame_ids=np.asarray([p['pid'] for p in sweep['frames']]))
    print('Exported aligned intensities and model inputs; evaluating fixed-grid controls',flush=True)
    cfg=ShadowConfig(**run['shadow_configuration']['shadow'])
    diagnostics=diagnose_stack(sweep['stack'],sweep['azimuths'],sweep['elevations'],cfg,
        sigma=run['shadow_configuration']['noise_sigma'],step_px=args.step_px,visibility=np.array(visibility),
        slope_row=slopes[0],slope_col=slopes[1],progress=lambda n:print(f'Assessed diagnostic grid position {n}',flush=True))
    diagnostics['provenance']=dict(manifest_sha256=expected,
        stack_sha256=hashlib.sha256(stackfile.read_bytes()).hexdigest(),
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            'scripts/diagnose_landing_run.py','src/hati_core/shadow_diagnostics.py','src/hati_core/shadow_likelihood.py')})
    (output/'model_diagnostics.json').write_text(json.dumps(clean_json(diagnostics),indent=2,allow_nan=False),encoding='utf-8')
    (output/'source_run.json').write_bytes((args.run_dir/'run.json').read_bytes())
    bundle=output/'hati_diagnostic_bundle.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ('aligned_stack.npz','warning_attribution.json','SUMMARY.md','model_diagnostics.json','source_run.json'):
            z.write(output/name,arcname=name)
    print(json.dumps(diagnostics['summary'],indent=2),flush=True)
    print(f'Send this diagnostic bundle: {bundle}',flush=True)


if __name__=='__main__':main()
