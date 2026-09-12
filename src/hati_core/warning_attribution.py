"""Explain regional warnings without equating them with direct detections."""
import numpy as np


def describe_warning(maps,raw_score,raw_status,common_fraction,row,col,pixel_m,radius_m,
                     *,threshold=8.,required_common=.8,candidates=(),root_row=None,root_col=None,
                     cell_px=4,patch_radius=12):
    shape=raw_score.shape
    rr,cc=int(np.floor(row)),int(np.floor(col))
    if not (0<=rr<shape[0] and 0<=cc<shape[1]):
        raise ValueError('test location is outside the map')
    def number(v): return float(v) if np.isfinite(v) else None
    result=dict(raw_cell=dict(status=int(raw_status[rr,cc]),score=number(raw_score[rr,cc]),
                common_fraction=number(common_fraction[rr,cc]),required_common_fraction=required_common,
                assessed=bool(raw_status[rr,cc]==1)),maps={},source=None)
    for name,a in maps.items():
        valid=np.isfinite(a)
        result['maps'][name]=dict(available_fraction=float(valid.mean()),
            threshold_exceedance_fraction_available=float(np.mean(a[valid]>=.5)) if valid.any() else None,
            threshold_exceedance_fraction_window=float(np.mean(a>=.5)))
    valid=np.isfinite(raw_score)
    result['raw_threshold_exceedance_fraction_assessed']=float(np.mean(raw_score[valid]>=threshold)) if valid.any() else None
    reach=int(np.ceil(radius_m/pixel_m))
    r0,r1=max(0,rr-reach),min(shape[0],rr+reach+1)
    c0,c1=max(0,cc-reach),min(shape[1],cc+reach+1)
    yy,xx=np.mgrid[r0:r1,c0:c1]
    eligible=((yy-rr)**2+(xx-cc)**2 <= (radius_m/pixel_m)**2+1e-10)&np.isfinite(raw_score[r0:r1,c0:c1])
    if eligible.any():
        values=np.where(eligible,raw_score[r0:r1,c0:c1],-np.inf)
        maximum=float(values.max()); positions=np.argwhere(values==maximum)+[r0,c0]
        sr,sc=positions[0]
        source=dict(score=maximum,regional_index=maximum/(maximum+threshold),
                    representative_source_pixel=[int(sr),int(sc)],tied_source_pixels=len(positions),
                    root=None,root_resolution='not recorded')
        if root_row is not None and root_col is not None and np.isfinite([root_row[sr,sc],root_col[sr,sc]]).all():
            source['root']=dict(row_px=float(root_row[sr,sc]),col_px=float(root_col[sr,sc]))
            source['root_resolution']='recorded cell winner'
        else:
            # Historical candidate tables are deduplicated; only claim a
            # recovered winner if its score AND owned cell agree uniquely.
            cells={((int(r)-patch_radius)//cell_px,(int(c)-patch_radius)//cell_px) for r,c in positions}
            matches=[]
            for p in candidates:
                pr,pc=float(p['row_px']),float(p['col_px'])
                cell=(int(np.floor((pr-patch_radius)/cell_px)),int(np.floor((pc-patch_radius)/cell_px)))
                if cell in cells and np.isclose(float(p['score']),maximum,rtol=1e-6,atol=1e-6):
                    matches.append(dict(row_px=pr,col_px=pc,contrast=float(p['contrast']),
                        endpoint_censored=str(p['endpoint_censored']).lower()=='true'))
            if len(matches)==1:
                source['root']=matches[0]; source['root_resolution']='unique saved candidate matching score and cell'
        if source['root'] is not None:
            p=source['root']
            p['distance_from_test_location_m']=float(np.hypot(p['row_px']+.5-row,p['col_px']+.5-col)*pixel_m)
            p['within_nominal_radius']=bool(p['distance_from_test_location_m']<=radius_m)
        result['source']=source
    result['buffer']=dict(nominal_radius_m=radius_m,
        meaning='Disk centred on sampled map pixel, applied to four-pixel regional maxima; not an exact root-distance test',
        cell_diagonal_m=float(np.sqrt(2)*cell_px*pixel_m))
    result['warning_origin']='unavailable'
    if maps['fused'][rr,cc]>=.5:
        result['warning_origin']='regional_warning_direct_cell_unassessed' if raw_status[rr,cc]!=1 and maps['shadow'][rr,cc]>=.5 else 'configured_regional_warning'
    elif np.isfinite(maps['fused'][rr,cc]):
        result['warning_origin']='lower_index_under_recorded_model'
    result['interpretation']='A regional threshold warning is not an independently validated object detection or successful mission-loss counterfactual.'
    return result
