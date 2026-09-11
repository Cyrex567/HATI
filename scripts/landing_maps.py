"""HATI three-map postprocessing from saved products; no ISIS or downloads."""
from pathlib import Path
from dataclasses import asdict
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from src.hati_core.landing_terrain import LandingConfig,terrain_assessment,buffer_evidence,fuse_landing
from src.hati_core.dem_shadow import predict_visibility
from src.hati_core.shadow_likelihood import ShadowConfig
from src.hati_core.regional_shadow import RegionalConfig,assess_regions
from sweep_products import load_sweep,load_dem_context,on_reference
from sweep_contract import DEFAULT_BEFORE


def clean_json(value):
    if isinstance(value,dict): return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [clean_json(v) for v in value]
    if isinstance(value,np.ndarray): return clean_json(value.tolist())
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value,(bool,np.bool_)): return bool(value)
    if isinstance(value,np.integer): return int(value)
    if isinstance(value,Path): return str(value)
    return value


def write_tif(path,array,transform,crs,meaning):
    import rasterio
    a=np.asarray(array,dtype='float32')
    with rasterio.open(path,'w',driver='GTiff',height=a.shape[0],width=a.shape[1],count=1,
                       dtype='float32',crs=crs,transform=transform,nodata=np.nan,
                       compress='deflate') as dst:
        dst.write(a,1)
        dst.update_tags(meaning=meaning,software='HATI 2.5.3',index_is_probability='false')


def counterfactual(maps,status,row,col,*,is_demo=False):
    """A fixed-location descriptive test; no tuning or guaranteed positive label."""
    common=(status==1)
    result=dict(site='synthetic demonstration' if is_demo else 'IM-2 Athena prelanding counterfactual',
                row_px=float(row),col_px=float(col),eligible_common_fraction=float(common.mean()),
                ranking_scope='Comparable fully qualified pixels inside this analysed window only',maps={})
    rr,cc=int(np.floor(row)),int(np.floor(col))
    for name,a in maps.items():
        score=float(a[rr,cc]); values=a[common&np.isfinite(a)]
        observed=a[np.isfinite(a)]
        descriptive_percentile=100*float(((observed<score).sum()+.5*(observed==score).sum())/len(observed)) \
            if np.isfinite(score) and len(observed) else None
        percentile=None
        if np.isfinite(score) and len(values) and common[rr,cc]:
            percentile=100*float(((values<score).sum()+.5*(values==score).sum())/len(values))
        result['maps'][name]=dict(index=score,at_or_above_configured_threshold=bool(score>=.5)
            if np.isfinite(score) else None,hazard_percentile_in_common_region=percentile,
            descriptive_percentile_in_available_module_area=descriptive_percentile)
    result['assessment']='flagged_by_configured_index' if maps['fused'][rr,cc]>=.5 else (
        'lower_index_under_model' if common[rr,cc] else 'insufficient_support')
    result['qualified_at_touchdown']=bool(common[rr,cc])
    result['limitation']='A terrain flag cannot prove avoidance of mission loss or explain descent dynamics.'
    return clean_json(result)


def rank_centres(maps,status,pixel,transform,cfg):
    """Regular alternative centres; descriptive ordering is separate from qualification."""
    step=max(1,int(np.ceil(cfg.footprint_diameter_m/pixel)))
    rows=[]
    for r in range(step//2,status.shape[0],step):
        for c in range(step//2,status.shape[1],step):
            a,b=maps['terrain'][r,c],maps['shadow'][r,c]
            if not np.isfinite([a,b]).all(): continue
            x,y=transform*(c+.5,r+.5)
            rows.append(dict(row_px=r,col_px=c,x_m=x,y_m=y,terrain_index=a,shadow_index=b,
                nominal_maximum_index=max(a,b),fused_index=maps['fused'][r,c],
                both_modules_qualified=bool(status[r,c]==1)))
    rows.sort(key=lambda p:(p['nominal_maximum_index'],p['row_px'],p['col_px']))
    return clean_json(rows)


def make_previews(output,maps,status,pixel,row,col,label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.facecolor':'#101a27',
                         'axes.facecolor':'#101a27','text.color':'#edf3f8','axes.labelcolor':'#edf3f8',
                         'xtick.color':'#c2cedb','ytick.color':'#c2cedb','axes.edgecolor':'#526274'})
    h,w=next(iter(maps.values())).shape
    extent=(-col*pixel,(w-col)*pixel,(row-h)*pixel,row*pixel)
    cmap=plt.get_cmap('YlOrRd').copy(); cmap.set_bad('#455567')
    titles={'terrain':'Terrain / fitted plane','shadow':'Shadow sweep / regional evidence','fused':'Fused / qualified evidence'}
    def panel(ax,key,a):
        im=ax.imshow(a,vmin=0,vmax=1,cmap=cmap,extent=extent,interpolation='nearest')
        ax.scatter([0],[0],marker='+',s=160,c='white',linewidths=2)
        ax.set_title(titles[key],color='#edf3f8'); ax.set_xlabel('Map east offset (m)')
        ax.set_ylabel('Map north offset (m)')
        return im
    for key,a in maps.items():
        fig,ax=plt.subplots(figsize=(7,6))
        im=panel(ax,key,a); fig.colorbar(im,ax=ax,label='Configured index; 0.5 = threshold')
        fig.suptitle('HATI 2.5.3 | '+label,fontsize=13)
        fig.text(.05,.015,'Grey = unavailable. Low index is not a landing clearance.',fontsize=9)
        fig.tight_layout(rect=(0,.04,1,.94)); fig.savefig(output/(key+'_hazard.png'),dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(17,6),layout='constrained')
    for ax,(key,a) in zip(axes,maps.items()): im=panel(ax,key,a)
    fig.colorbar(im,ax=axes,shrink=.72,label='Configured index (not probability)')
    fig.suptitle('HATI 2.5.3  |  '+label+'\nSeparate modules + conservative fusion; white cross = fixed test location',fontsize=16)
    fig.savefig(output/'three_maps.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6))
    im=ax.imshow(status,vmin=0,vmax=2,cmap=ListedColormap(['#455567','#4cadb8','#e79536']),extent=extent)
    ax.set_title('Fusion observability'); ax.set_xlabel('Map east offset (m)'); ax.set_ylabel('Map north offset (m)')
    bar=fig.colorbar(im,ax=ax,ticks=[0,1,2]); bar.ax.set_yticklabels(['Unknown','Both qualified','High, incomplete'])
    fig.tight_layout(); fig.savefig(output/'observability.png',dpi=160); plt.close(fig)


def run(sweep,context,output,cfg,shadow_cfg,regional_cfg,noise_sigma,*,is_demo=False,provenance=None):
    started=time.monotonic(); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    transform,crs=sweep['transform'],sweep['crs']; shape=sweep['stack'].shape[1:]; pixel=sweep['pixel_m']
    def project(a): return on_reference(a,context,transform,crs,shape,nearest=True)
    print('Measuring native-posting DEM planes and relief',flush=True)
    terrain=terrain_assessment(context['dem'],context['pixel_m'],cfg)
    primary=terrain['primary']
    for diameter,metrics in terrain['measurements'].items():
        for key,a in metrics.items():
            if isinstance(a,np.ndarray):
                write_tif(output/f'dem_{diameter}m_{key}.tif',a,context['transform'],crs,
                          f'Native posting {context["pixel_m"]} m; physical diameter {diameter} m; {key}')
    nominal=[]; conservative=[]
    for i,(az,el) in enumerate(zip(sweep['azimuths'],sweep['elevations'])):
        print(f'Predicting DEM shadow: frame {i+1}/{len(sweep["stack"])}',flush=True)
        prediction=predict_visibility(context['dem'],context['pixel_m'],az,el,cfg.horizon_distance_m,
                vertical_sigma_m=cfg.dem_vertical_sigma_m,sigma_multiplier=cfg.horizon_sigma_multiplier,
                solar_radius_deg=shadow_cfg.solar_radius_deg)
        nominal.append(project(prediction['visible'])); conservative.append(project(prediction['visible_conservative']))
        for key,a in prediction.items():
            if isinstance(a,np.ndarray):
                write_tif(output/f'dem_frame_{i:02d}_{key}.tif',project(a),transform,crs,
                          f'{key}; finite {cfg.horizon_distance_m} m horizon; height-envelope scenario, not calibrated confidence')
    print('Systematic regional search; every interior analysis cell is visited',flush=True)
    last=[0.]
    def progress(info):
        if time.monotonic()-last[0]>20:
            print(f'Cells visited: {info["cells_visited"]}; assessed: {info["cells_assessed"]}',flush=True)
            last[0]=time.monotonic()
    regional=assess_regions(sweep['stack'],sweep['azimuths'],sweep['elevations'],noise_sigma,
            shadow_cfg,regional_cfg,visible=np.asarray(nominal),conservative_visible=np.asarray(conservative),
            slope_row=project(primary['slope_row']),slope_col=project(primary['slope_col']),progress=progress)
    terrain_index,terrain_complete=buffer_evidence(project(terrain['score']),pixel,cfg.navigation_margin_m)
    shadow_radius=cfg.footprint_diameter_m/2+cfg.navigation_margin_m
    shadow_index,shadow_complete=buffer_evidence(regional['index'],pixel,shadow_radius)
    # Separate maps retain measured evidence. Only qualified low evidence can
    # enter fusion. Strong evidence remains an exclusion even with missing data.
    tq=terrain_complete & terrain['footprint_resolved']
    sq=regional['sensitivity_ok']&regional['envelope_ok']&(regional['status']==1)
    qualified_shadow=np.where(sq|(regional['index']>=.5),regional['index'],np.nan)
    qualified_shadow,_=buffer_evidence(qualified_shadow,pixel,shadow_radius)
    _,sq_buffer=buffer_evidence(np.where(sq,0.,np.nan),pixel,shadow_radius)
    qualified_terrain=np.where(tq|(terrain_index>=.5),terrain_index,np.nan)
    fused,status=fuse_landing(qualified_terrain,qualified_shadow)
    # An available high value does not itself establish complete assessment.
    both_qualified=tq & sq_buffer
    status=np.where(both_qualified&np.isfinite(fused),1,np.where(fused>=.5,2,0)).astype('uint8')
    maps=dict(terrain=terrain_index,shadow=shadow_index,fused=fused)
    for name,a in maps.items():
        write_tif(output/(name+'_hazard.tif'),a,transform,crs,
                  'Configured regional hazard index, not probability; 0.5 threshold; use observability layers')
    write_tif(output/'fusion_status.tif',status,transform,crs,'0 unknown; 1 both qualified; 2 high evidence incomplete')
    for key in ('score','required_contrast','common_fraction','status','frame_count','envelope_ok','sensitivity_ok'):
        write_tif(output/('shadow_'+key+'.tif'),regional[key],transform,crs,key+'; conditional template/noise model')
    write_tif(output/'terrain_qualified.tif',tq,transform,crs,'Native footprint resolution and navigation support')
    write_tif(output/'shadow_qualified.tif',sq_buffer,transform,crs,'Buffered model sensitivity and DEM envelope support')
    candidates=regional['candidates']
    for p in candidates:
        p['x_m'],p['y_m']=transform*(p['col_px']+.5,p['row_px']+.5)
    fields=['row_px','col_px','x_m','y_m','score','contrast','identifiability','template_height_m',
            'template_width_m','endpoint_censored','frames','frame_delta_chi2']
    with (output/'candidates.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(candidates)
    row,col=sweep.get('test_pixel',(shape[0]/2,shape[1]/2))
    cf=counterfactual(maps,status,row,col,is_demo=is_demo)
    ranking=rank_centres(maps,status,pixel,transform,cfg)
    with (output/'site_ranking.csv').open('w',newline='',encoding='utf-8') as f:
        names=['row_px','col_px','x_m','y_m','terrain_index','shadow_index','nominal_maximum_index',
               'fused_index','both_modules_qualified']
        writer=csv.DictWriter(f,fieldnames=names); writer.writeheader(); writer.writerows(ranking)
    label='SYNTHETIC DEMONSTRATION' if is_demo else 'ATHENA / PRELANDING RESEARCH'
    make_previews(output,maps,status,pixel,row,col,label)
    report=dict(version='2.5.3',status='experimental_unvalidated',demo=is_demo,landing=asdict(cfg),
        landing_config_hash=cfg.hash(),shadow_configuration=regional['configuration'],
        shadow_config_hash=regional['config_hash'],frames=sweep.get('frames',[]),
        azimuths_map=sweep['azimuths'],elevations=sweep['elevations'],crs=crs.to_wkt(),transform=list(transform),
        site_lat=sweep.get('site_lat'),site_lon=sweep.get('site_lon'),
        image_posting_m=pixel,dem_posting_m=context['pixel_m'],dem_source=context['source'],
        dem_halo_m=context['halo_m'],effective_terrain_diameter_m=terrain['effective_diameter_m'],
        footprint_resolved=terrain['footprint_resolved'],cells_visited=regional['cells_visited'],
        cells_assessed=regional['cells_assessed'],candidate_count=len(candidates),search_truncated=False,
        coverage={k:float(np.isfinite(a).mean()) for k,a in maps.items()},
        fusion_status_fractions={str(i):float((status==i).mean()) for i in range(3)},counterfactual=cf,
        provenance=provenance or {},elapsed_seconds=time.monotonic()-started,
        ranking=dict(centres=len(ranking),ordering='ascending nominal maximum of separate buffered indices',
                     meaning='Descriptive alternatives; qualification column is separate. No reachability or clearance guarantee.'),
        limitations=[
            'Illustrative limits unless replaced by verified vehicle configuration; no probability of LOM.',
            'Terrain uses least-squares surfaces, not a solved footpad contact/stability model.',
            'Native posting is sampling, not a DEM accuracy or effective spatial-resolution guarantee.',
            'Finite DEM horizon; terrain beyond the recorded distance is not modelled.',
            'DEM uncertainty is a bounded assumed envelope; at grazing Sun it can leave most low scores unqualified.',
            'Frame-centre Sun geometry is held constant in this window; local slopes are binned in m/m.',
            'A fixed cell bank produces regional evidence maxima, not object counts or calibrated significance.',
            'Sensitivity is expected template signal under assumed noise, not injection-recovery completeness.',
            'First-order albedo-gradient covariance does not repair incorrect or spatially varying registration.',
            'Footprint and navigation buffers use map metres and discrete pixel-centre disks.',
            'No global multiple-search false-alarm or empirical detection calibration is claimed.',
            'Low module indices are descriptive; fused low indices require the recorded model qualifications.'])
    (output/'run.json').write_text(json.dumps(clean_json(report),indent=2,allow_nan=False),encoding='utf-8')
    (output/'counterfactual.json').write_text(json.dumps(cf,indent=2,allow_nan=False),encoding='utf-8')
    print(f'Saved three separate heatmaps and fusion to {output}',flush=True)
    print(f'Fixed-location result: {cf["assessment"]}; common qualified area {cf["eligible_common_fraction"]:.1%}',flush=True)
    return report


def demo_inputs():
    from affine import Affine
    from rasterio.crs import CRS
    sys.path.insert(0,str(ROOT/'tests'))
    from test_shadow_likelihood import independent_scene,AZ,EL
    from scipy.ndimage import gaussian_filter
    shape=(80,80); pixel=.9; halo=40
    transform=Affine(pixel,0,-36,0,-pixel,36)
    crs=CRS.from_string('+proj=stere +lat_0=-90 +lon_0=0 +R=1737400 +units=m')
    yy,xx=np.indices((160,160)); rng=np.random.default_rng(49)
    z=1.4*np.exp(-((yy-65)**2+(xx-62)**2)/16)+.08*gaussian_filter(rng.normal(size=(160,160)),1)
    stack=independent_scene(shape=shape,root=(40.3,40.2),height=.5,width=.9)
    return dict(stack=stack,azimuths=AZ,elevations=EL,transform=transform,crs=crs,pixel_m=pixel),dict(
        dem=z,pixel_m=pixel,transform=transform*Affine.translation(-halo,-halo),crs=crs,
        source='synthetic DEM; independent tapered shadow renderer',halo_m=halo*pixel)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,default=ROOT/'data/sweep/manifest.json')
    ap.add_argument('--output',type=Path,default=ROOT/'output/athena/landing_maps_v253')
    ap.add_argument('--before',default=DEFAULT_BEFORE)
    ap.add_argument('--half',type=int,default=256,help='half-width within already ingested window; systematic within this region')
    ap.add_argument('--config',type=Path,help='JSON object of LandingConfig fields')
    ap.add_argument('--registration-sigma-px',type=float,required=True)
    ap.add_argument('--noise-sigma',type=float,default=.03)
    ap.add_argument('--demo',action='store_true',help='synthetic offline exercise; no real-site inference')
    args=ap.parse_args()
    cfg=LandingConfig(**json.loads(args.config.read_text())) if args.config else LandingConfig()
    if args.demo:
        if args.config is None:
            cfg=LandingConfig(baselines_m=(4.,8.,16.),horizon_distance_m=20.,dem_vertical_sigma_m=0.,
                              navigation_margin_m=1.,label='synthetic_illustrative_limits')
        sweep,context=demo_inputs()
    else:
        sweep=load_sweep(args.manifest,args.half,args.before)
        halo=cfg.horizon_distance_m+max(*cfg.baselines_m,cfg.footprint_diameter_m)/2+10
        context=load_dem_context(sweep['dem_path'],sweep['transform'],sweep['crs'],sweep['stack'].shape[1:],halo)
    sc=ShadowConfig(pixel_m=sweep['pixel_m'],registration_sigma_px=args.registration_sigma_px)
    rc=RegionalConfig(score_scale=cfg.shadow_score_scale)
    rev=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    from importlib.metadata import version
    source_files=['scripts/landing_maps.py','scripts/sweep_products.py','scripts/shadow_kinematics_real.py',
                  'src/hati_core/landing_terrain.py','src/hati_core/dem_shadow.py',
                  'src/hati_core/regional_shadow.py','src/hati_core/shadow_likelihood.py']
    provenance=dict(revision=rev.stdout.strip(),arguments=vars(args),
                    source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files},
                    python_version=sys.version,packages={p:version(p) for p in ('numpy','scipy','rasterio','pyproj','matplotlib')},
                    manifest_sha256=None if args.demo else hashlib.sha256(args.manifest.read_bytes()).hexdigest())
    run(sweep,context,args.output,cfg,sc,rc,args.noise_sigma,is_demo=args.demo,provenance=provenance)


if __name__=='__main__': main()
