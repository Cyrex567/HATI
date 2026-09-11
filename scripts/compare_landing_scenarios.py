"""Compare spatial evidence across assumptions; equal candidate counts are not stability."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import rasterio
from scipy.spatial import cKDTree
from landing_maps import write_tif,clean_json


def compare(folders,output):
    folders=[Path(p) for p in folders]; output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if len(folders)<2: raise ValueError('at least two scenarios required')
    arrays=[]; roots=[]; runs=[]; grid=None
    for folder in folders:
        run=json.loads((folder/'run.json').read_text()); runs.append(run)
        with rasterio.open(folder/'shadow_hazard.tif') as src:
            key=(src.shape,src.transform,src.crs)
            if grid is not None and key!=grid: raise ValueError('scenario grids differ')
            grid=key; arrays.append(src.read(1))
        with (folder/'candidates.csv').open() as f:
            roots.append(np.array([[float(r['row_px']),float(r['col_px'])] for r in csv.DictReader(f)]).reshape(-1,2))
    # Geometry, radiometry and masks must represent the same input window.
    if any(r.get('provenance',{}).get('manifest_sha256')!=runs[0].get('provenance',{}).get('manifest_sha256')
           or r.get('landing_config_hash')!=runs[0].get('landing_config_hash') for r in runs[1:]):
        raise ValueError('comparison requires matching manifest and landing configuration')
    stack=np.array(arrays); common=np.isfinite(stack).all(axis=0)
    span=np.where(common,np.max(stack,axis=0)-np.min(stack,axis=0),np.nan)
    disagreement=np.where(common,(np.any(stack>=.5,axis=0)&~np.all(stack>=.5,axis=0)).astype(float),np.nan)
    for name,a in [('shadow_index_range',span),('shadow_threshold_disagreement',disagreement)]:
        write_tif(output/(name+'.tif'),a,grid[1],grid[2],'Cross-scenario '+name+'; not a confidence interval')
    comparisons=[]
    for i in range(len(folders)):
        for j in range(i+1,len(folders)):
            def matched(a,b):
                if len(a)==0: return None
                if len(b)==0: return 0.
                return float(np.mean(cKDTree(b).query(a)[0]<=3))
            comparisons.append(dict(first=folders[i].name,second=folders[j].name,
                first_count=len(roots[i]),second_count=len(roots[j]),
                first_with_neighbour_in_second=matched(roots[i],roots[j]),
                second_with_neighbour_in_first=matched(roots[j],roots[i])))
    report=dict(scenarios=[str(p) for p in folders],common_shadow_fraction=float(common.mean()),
        median_index_range=float(np.median(span[common])) if common.any() else None,
        threshold_disagreement_fraction=float(disagreement[common].mean()) if common.any() else None,
        candidate_spatial_comparisons=comparisons,
        matching='Directional nearest-neighbour support within 3 pixels; many-to-one, not object association.',
        limitation='Stability across chosen assumptions is not truth, calibration, or a mission-loss bound.')
    (output/'comparison.json').write_text(json.dumps(clean_json(report),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('folders',type=Path,nargs='+')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); compare(args.folders,args.output)


if __name__=='__main__': main()
