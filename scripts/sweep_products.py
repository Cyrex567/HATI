"""Read audited sweep products. No ISIS programs and no network operations."""
from pathlib import Path
import json
import numpy as np
import rasterio
from sweep_contract import PROCESSING_VERSION,predates,DEFAULT_BEFORE


def load_sweep(manifest,half,before=DEFAULT_BEFORE):
    import athena_counterfactual as ac
    from shadow_kinematics_real import frame_window
    from src.hati_core.shadow_likelihood import map_sun_azimuth
    manifest=Path(manifest)
    entries=json.loads(manifest.read_text())
    if not isinstance(entries,list) or len(entries)<3 or any(
            e.get('processing_version')!=PROCESSING_VERSION or not e.get('gate_pass')
            or not predates(e.get('utc',''),before) for e in entries):
        raise ValueError('need >=3 audited, pre-cutoff, gate-passing manifest frames')
    if half<16 or any(half>int(e.get('half_px',0)) for e in entries):
        raise ValueError('science window must fit inside every ingested registration window')
    with rasterio.open(ac.ORTHO_IMG) as src:
        row,col=ac.ortho_pixel()
        transform=src.window_transform(rasterio.windows.Window(col-half,row-half,2*half,2*half))
        crs=src.crs
        sx,sy=np.hypot(transform.a,transform.d),np.hypot(transform.b,transform.e)
        if not np.isclose(sx,sy) or abs(transform.a*transform.b+transform.d*transform.e)>1e-8:
            raise ValueError('square orthogonal reference pixels are required')
    frames,azimuths,elevations,provenance=[],[],[],[]
    for entry in entries:
        pid=entry['pid']; base=pid.split('.')[-1].upper()
        side=manifest.parent/(base+'.geom.json')
        # Do not call geometry_for: even rebuild=False can call campt when a
        # surviving level-1 cube exists. This reader consumes sidecars only.
        g=json.loads(side.read_text())
        if g.get('site_lat')!=ac.TD_LAT or g.get('site_lon')!=ac.TD_LON or not np.isfinite(
                [g.get('az',np.nan),g.get('elev',np.nan)]).all():
            raise ValueError(f'{pid}: missing/invalid measured site geometry')
        shift=entry.get('shift_px')
        if np.shape(shift)!=(2,) or not np.isfinite(shift).all():
            raise ValueError(f'{pid}: missing/invalid registration shift')
        path=Path(entry['lev2'])
        if not path.exists():
            path=manifest.parent/path.name
        with rasterio.open(path) as projected:
            axes=lambda t:np.array([t.a,t.b,t.d,t.e])
            if projected.crs!=crs or not np.allclose(axes(projected.transform),axes(transform),atol=1e-8,rtol=0):
                raise ValueError(f'{pid}: projected CRS/pixel axes differ from the audited reference')
        original_half=int(entry['half_px'])
        frame=frame_window(path,shift,original_half,ac)
        if frame is None:
            raise ValueError(f'{pid}: projected image is unreadable')
        frame=frame[original_half-half:original_half+half,original_half-half:original_half+half]
        valid=np.isfinite(frame)&(frame>0)
        if valid.mean()<.8:
            raise ValueError(f'{pid}: less than 80% positive image coverage; no silent frame dropping')
        scale=float(np.median(frame[valid]))
        frames.append(np.where(valid,frame/scale,np.nan))
        azimuths.append(map_sun_azimuth(crs,transform,ac.TD_LAT,ac.TD_LON,g['az']))
        elevations.append(g['elev'])
        provenance.append(dict(pid=pid,utc=entry['utc'],lev2=str(path),geometry=g,
                               normalization_median=scale,shift_px=shift,
                               residual_px=entry.get('residual_px'),closure_px=entry.get('closure_px')))
    from pyproj import CRS,Transformer
    target=CRS.from_user_input(crs)
    x,y=Transformer.from_crs(target.geodetic_crs,target,always_xy=True).transform(ac.TD_LON,ac.TD_LAT)
    test_col,test_row=(~transform)*(x,y)
    return dict(stack=np.stack(frames),azimuths=np.array(azimuths),elevations=np.array(elevations),
                transform=transform,crs=crs,pixel_m=float(sx),frames=provenance,
                site_lat=ac.TD_LAT,site_lon=ac.TD_LON,dem_path=ac.DTM_PATH,test_pixel=(test_row,test_col))


def load_dem_context(path,reference_transform,reference_crs,shape,halo_m):
    """Read original DEM samples with a horizon/filter halo; never upsample first.

    The native grid must share reference CRS and axis orientation so directional
    slopes and Sun bearings remain comparable. Raster origins may differ.
    """
    with rasterio.open(path) as src:
        if src.crs!=reference_crs:
            raise ValueError('DEM and reference CRS differ; supply a verified DEM in the reference CRS')
        native=np.hypot(src.transform.a,src.transform.d)
        if not np.isclose(native,np.hypot(src.transform.b,src.transform.e)):
            raise ValueError('DEM must have square pixels')
        ref=np.hypot(reference_transform.a,reference_transform.d)
        axes=lambda t:np.array([[t.a,t.b],[t.d,t.e]])
        if not np.allclose(axes(src.transform)/native,axes(reference_transform)/ref,atol=1e-8):
            raise ValueError('DEM and image axes differ; vector-gradient transport must be specified')
        corners=[(~src.transform)*(reference_transform*(c,r))
                 for r in (0,shape[0]) for c in (0,shape[1])]
        padding=int(np.ceil(halo_m/native))
        col0=int(np.floor(min(p[0] for p in corners)))-padding
        row0=int(np.floor(min(p[1] for p in corners)))-padding
        col1=int(np.ceil(max(p[0] for p in corners)))+padding
        row1=int(np.ceil(max(p[1] for p in corners)))+padding
        window=rasterio.windows.Window(col0,row0,col1-col0,row1-row0)
        transform=src.window_transform(window)
        dem=src.read(1,window=window,boundless=True,masked=True,out_dtype='float64').filled(np.nan)
        dem[dem<=-1e30]=np.nan
    return dict(dem=dem,transform=transform,crs=reference_crs,pixel_m=float(native),
                source=str(path),halo_m=float(halo_m))


def on_reference(array,context,transform,crs,shape,*,nearest=False):
    from rasterio.warp import reproject,Resampling
    result=np.full(shape,np.nan,dtype='float64')
    reproject(np.asarray(array,float),result,src_transform=context['transform'],src_crs=context['crs'],
              dst_transform=transform,dst_crs=crs,src_nodata=np.nan,dst_nodata=np.nan,
              resampling=Resampling.nearest if nearest else Resampling.bilinear)
    coverage=np.zeros(shape,dtype='float64')
    reproject(np.isfinite(array).astype(float),coverage,
              src_transform=context['transform'],src_crs=context['crs'],
              dst_transform=transform,dst_crs=crs,
              resampling=Resampling.nearest if nearest else Resampling.bilinear)
    result[coverage<1-1e-8]=np.nan
    return result
