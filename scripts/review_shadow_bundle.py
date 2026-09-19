"""Replay scene diagnostics from a portable ZIP. Never opens cubes or runs ISIS."""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from affine import Affine
from pyproj import CRS

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from src.hati_core import __version__
from src.hati_core.scene_diagnostics import SceneConfig,broad_dark_discrepancy,local_registration
from landing_maps import clean_json,write_tif


def read_bundle(path):
    # Read named payloads, never extract archive paths or execute bundle text.
    with zipfile.ZipFile(path) as z:
        required=('aligned_stack.npz','model_diagnostics.json','source_run.json')
        if any(z.namelist().count(k)!=1 for k in required):
            raise ValueError('bundle requires exactly one of each named payload')
        payload=z.read('aligned_stack.npz')
        previous=json.loads(z.read('model_diagnostics.json'))
        runbytes=z.read('source_run.json');run=json.loads(runbytes)
    digest=hashlib.sha256(payload).hexdigest()
    if digest!=previous['provenance']['stack_sha256']:
        raise ValueError('aligned stack hash does not match provenance')
    with np.load(io.BytesIO(payload),allow_pickle=False) as n:
        data={k:n[k] for k in ('stack','visibility','azimuths','elevations','frame_ids','transform','crs')}
    a=data['stack'];transform=Affine(*data['transform'][:6]);crs=CRS.from_user_input(str(data['crs']))
    if a.ndim!=3 or len(a)<3 or data['visibility'].shape!=a.shape:
        raise ValueError('invalid image/visibility stack')
    if list(data['frame_ids'])!=[p['pid'] for p in run['frames']]:
        raise ValueError('frame identity/order mismatch')
    for key,runkey in (('azimuths','azimuths_map'),('elevations','elevations')):
        if data[key].shape!=(len(a),) or not np.isfinite(data[key]).all() or not np.allclose(data[key],run[runkey],atol=1e-7,rtol=0):
            raise ValueError('geometry mismatch')
    if not np.allclose(tuple(transform),run['transform'],atol=1e-8,rtol=0) or not crs.equals(CRS.from_user_input(run['crs']),ignore_axis_order=True):
        raise ValueError('raster grid mismatch')
    pixel=run['image_posting_m']
    if not np.isfinite(pixel) or pixel<=0 or not np.allclose([transform.a,transform.b,transform.d,transform.e],[pixel,0,0,-pixel],atol=1e-8,rtol=0):
        raise ValueError('diagnostics require square north-up map pixels')
    return data,run,transform,crs,dict(stack_sha256=digest,source_run_sha256=hashlib.sha256(runbytes).hexdigest())


def preview(output,data,run,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    h,w=data['stack'].shape[1:];pixel=run['image_posting_m']
    cf=run['counterfactual'];row,col=cf['row_px'],cf['col_px']
    extent=(-col*pixel,(w-col)*pixel,(row-h)*pixel,row*pixel)
    status=np.where(result['flag'],2,np.where(result['assessed'],1,0))
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    finite=np.isfinite(data['stack']);count=finite.sum(axis=0)
    mean=np.divide(np.where(finite,data['stack'],0).sum(axis=0),count,out=np.full((h,w),np.nan),where=count>0)
    images=[mean,result['fraction'],status]
    titles=['Mean aligned intensity','Broad dark / nominally lit fraction','Discrepancy assessment']
    for ax,a,title,cmap in zip(axes,images,titles,['gray','magma',ListedColormap(['#65717f','#55afb2','#e79c3a'])]):
        cmap=plt.get_cmap(cmap).with_extremes(bad='#65717f')
        im=ax.imshow(a,extent=extent,cmap=cmap,vmin=0,vmax=2 if title!=titles[1] else 1,interpolation='nearest')
        ax.scatter([0],[0],marker='+',c='red');ax.set_title(title)
        ax.set_xlabel('Map east offset (m)');ax.set_ylabel('Map north offset (m)')
        bar=fig.colorbar(im,ax=ax,shrink=.7)
        if title==titles[2]:
            bar.set_ticks([0,1,2]);bar.set_ticklabels(['Unknown','No flag','Discrepancy'])
    fig.suptitle(f'HATI {__version__} | Scene diagnostics — no object identification or landing clearance')
    fig.savefig(output/'scene_diagnostics.png',dpi=150);plt.close(fig)


def review(bundle,output,cfg=None):
    cfg=cfg or SceneConfig();output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('choose a new output folder to preserve the previous review')
    data,run,transform,crs,provenance=read_bundle(bundle)
    row,col=(float(run['counterfactual'][k]) for k in ('row_px','col_px'))
    if not np.isfinite([row,col]).all() or not 0<=row<data['stack'].shape[1] or not 0<=col<data['stack'].shape[2]:
        raise ValueError('test coordinate outside grid')
    row,col=int(row),int(col)
    output.mkdir(parents=True,exist_ok=True)
    result=broad_dark_discrepancy(data['stack'],data['visibility'],cfg)
    registration=local_registration(data['stack'],data['azimuths'],data['elevations'],cfg)
    for key in ('flag','assessed','fraction','discrepant_frames','lit_frames'):
        write_tif(output/f'scene_dark_{key}.tif',result[key],transform,crs,
                  'Broad dark structure despite nominal DEM illumination; albedo/terrain/registration unresolved')
    summary=dict(version=__version__,configuration=asdict(cfg),
        assessed_fraction=float(result['assessed'].mean()),flag_fraction=float(result['flag'].mean()),
        test_location={k:result[k][row,col] for k in ('flag','assessed','fraction','lit_frames','discrepant_frames')},
        normalization_medians=result['normalization_medians'],registration=registration,
        limitations=['Broad dark structure is not an identified crater or boulder.',
            'No flag is not validation of the DEM, registration or subpixel model.',
            'Thresholds are declared research settings, not fitted to this site or calibrated operating limits.',
            'Original terrain, shadow and fused maps are not recomputed by this portable review.'],provenance=provenance)
    summary['provenance']['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
        'scripts/review_shadow_bundle.py','src/hati_core/scene_diagnostics.py')}
    summary=clean_json(summary)
    (output/'scene_diagnostics.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    preview(output,data,run,result)
    lines=[f'# HATI {__version__} portable scene review','',
        f"Assessed area: {summary['assessed_fraction']:.2%}. Broad-dark discrepancy: {summary['flag_fraction']:.2%} of the full map.",
        f"Fixed test location: {json.dumps(summary['test_location'])}.",'',
        '## Similar-illumination registration checks','',
        '| Frames | Measured tiles | Median apparent displacement (px) | Boundary peaks |',
        '|---|---:|---:|---:|']
    for pair in registration['pairs']:
        rows=[p for p in pair['tiles'] if p['status']=='measured_apparent_offset']
        med=float(np.median([np.hypot(p['row_offset_px'],p['col_offset_px']) for p in rows])) if rows else None
        lines.append(f"| {pair['frames']} | {len(rows)} | {med if med is not None else 'unavailable'} | {sum(p['search_boundary'] for p in rows)} |")
    lines+=['','Offsets compare A[r,c] with B[r+dr,c+dc]. They are illumination-confounded diagnostics, not corrections or uncertainty estimates.',
            '',*summary['limitations'],'','![Scene diagnostics](scene_diagnostics.png)']
    (output/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('assessed_fraction','flag_fraction','test_location')},indent=2),flush=True)
    print(f'Saved portable review: {output}',flush=True)
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--config',type=Path,help='JSON SceneConfig research settings')
    args=ap.parse_args()
    review(args.bundle,args.output,SceneConfig(**json.loads(args.config.read_text())) if args.config else None)


if __name__=='__main__':main()
