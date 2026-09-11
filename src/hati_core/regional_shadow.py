"""Systematic spatial shadow assessment without an image-wide candidate cap.

Every analysis cell is visited. Cells contain a regular bank of root positions;
their maximum score is a regional index, not a count or density of objects.
Masks, informative-frame counts and template sensitivity accompany every cell.
"""
from collections import OrderedDict
from dataclasses import dataclass,asdict
import hashlib
import json
import numpy as np
from .shadow_likelihood import ShadowConfig,RegistrationProjector,shadow_template


@dataclass(frozen=True)
class RegionalConfig:
    cell_px: int = 4
    tile_px: int = 64
    heights_m: tuple = (.3,.6,1.2)
    widths_m: tuple = (.6,1.2)
    root_contrast: float = .6
    score_scale: float = 8.
    min_frame_fraction: float = .85
    min_common_fraction: float = .8
    slope_bin: float = .005
    cache_banks: int = 8

    def __post_init__(self):
        if any(not isinstance(v,int) or v<1 for v in (self.cell_px,self.tile_px,self.cache_banks)):
            raise ValueError('cell, tile and cache counts must be positive integers')
        if self.cell_px>4 or self.tile_px%self.cell_px:
            raise ValueError('cell spacing must be <=4 and divide tile size')
        if not self.heights_m or not self.widths_m or any(not np.isfinite(v) or v<=0 for v in
                (*self.heights_m,*self.widths_m,self.root_contrast,self.score_scale,self.slope_bin)):
            raise ValueError('template dimensions and scales must be positive')
        if not 0<self.min_frame_fraction<=1 or not 0<self.min_common_fraction<=1:
            raise ValueError('coverage fractions must lie in (0,1]')


def assess_regions(stack,azimuths,elevations,sigma,shadow_cfg,regional_cfg=None,*,
                   visible=None,conservative_visible=None,slope_row=None,slope_col=None,
                   progress=None):
    cfg = regional_cfg or RegionalConfig()
    stack = np.asarray(stack,float)
    azimuths,elevations = np.asarray(azimuths,float),np.asarray(elevations,float)
    if stack.ndim!=3 or len(stack)<3 or azimuths.shape!=(len(stack),) or elevations.shape!=(len(stack),):
        raise ValueError('need an aligned stack and one illumination per frame')
    if not np.isfinite(sigma) or sigma<=0 or not np.isfinite(azimuths).all() or not np.isfinite(elevations).all():
        raise ValueError('invalid noise or geometry')
    n,h,w = stack.shape
    radius = shadow_cfg.radius_px
    if min(h,w)<2*radius+cfg.cell_px:
        raise ValueError('window too small for root support')
    if visible is None:
        visible = np.ones_like(stack)
    if np.shape(visible)!=stack.shape:
        raise ValueError('DEM visibility must share the image stack grid')
    if conservative_visible is None:
        conservative_visible = visible
    if np.shape(conservative_visible)!=stack.shape:
        raise ValueError('visibility envelope must match stack')
    slope_row = np.zeros((h,w)) if slope_row is None else np.asarray(slope_row,float)
    slope_col = np.zeros((h,w)) if slope_col is None else np.asarray(slope_col,float)
    if slope_row.shape!=(h,w) or slope_col.shape!=(h,w):
        raise ValueError('receiving gradients must share the image grid')
    usable = np.isfinite(stack)&np.isfinite(visible)&(np.asarray(visible)>=.99)
    strong_support = usable&np.isfinite(conservative_visible)&(np.asarray(conservative_visible)>=.99)
    shape = (2*radius+1,)*2
    yy,xx = np.indices(shape)
    support = np.hypot(yy-radius,xx-radius)<=shadow_cfg.root_support_px
    offsets = np.arange(cfg.cell_px)-(cfg.cell_px-1)/2
    bank_cache = OrderedDict()
    fields = {key:np.full((h,w),np.nan) for key in (
        'score','index','required_contrast','common_fraction','slope_row','slope_col')}
    status = np.zeros((h,w),dtype='uint8')
    # 0 unvisited/border, 1 assessed, 2 unavailable, 3 nonidentifiable
    frames_map = np.zeros((h,w),dtype='uint8')
    envelope_ok = np.zeros((h,w),bool)
    sensitivity_ok = np.zeros((h,w),bool)
    roots=[]
    cell_count=assessed=0
    def bank(selected,slopes):
        key=(tuple(selected),*slopes)
        if key in bank_cache:
            bank_cache.move_to_end(key)
            return bank_cache[key]
        templates,parameters=[],[]
        # All root offsets differ by whole pixels. Render each shape once on
        # a larger canvas, then take exact translated crops (no interpolation).
        pad=cfg.cell_px
        base_offset=float(offsets[0])
        rendered={}
        for ht in cfg.heights_m:
            for width in cfg.widths_m:
                rendered[ht,width]=shadow_template((shape[0]+2*pad,shape[1]+2*pad),
                    (radius+pad+base_offset,)*2,azimuths[selected],elevations[selected],
                    ht,width,shadow_cfg,slopes)[0]
        for dy in offsets:
            for dx in offsets:
                for ht in cfg.heights_m:
                    for width in cfg.widths_m:
                        r0,c0=int(pad+base_offset-dy),int(pad+base_offset-dx)
                        template=rendered[ht,width][:,r0:r0+shape[0],c0:c0+shape[1]]
                        az=np.radians(azimuths[selected]); dr,dc=np.cos(az),-np.sin(az)
                        length=ht/(np.tan(np.radians(elevations[selected]-.8*shadow_cfg.solar_radius_deg))
                                   +slopes[0]*dr+slopes[1]*dc)/shadow_cfg.pixel_m
                        censored=bool(np.any(np.hypot(dy+dr*length,dx+dc*length)>
                                             shadow_cfg.root_support_px-.75))
                        templates.append(template)
                        parameters.append((dy,dx,ht,width,censored))
        result=(np.asarray(templates),parameters)
        bank_cache[key]=result
        if len(bank_cache)>cfg.cache_banks:
            bank_cache.popitem(last=False)
        return result
    for tr in range(radius,h-radius,cfg.tile_px):
        for tc in range(radius,w-radius,cfg.tile_px):
            for row in range(tr,min(tr+cfg.tile_px,h-radius),cfg.cell_px):
                for col in range(tc,min(tc+cfg.tile_px,w-radius),cfg.cell_px):
                    # Each cell owns its pixels once, including a short final cell.
                    endr,endc=min(row+cfg.cell_px,h-radius),min(col+cfg.cell_px,w-radius)
                    out=np.s_[row:endr,col:endc]
                    # Use the centre of this cell; offsets cover all owned pixel
                    # roots at subpixel half-grid locations. Full patches retain
                    # a radius halo even at tile seams.
                    cr=min(row+cfg.cell_px//2,h-radius-1)
                    cc=min(col+cfg.cell_px//2,w-radius-1)
                    patch_sl=np.s_[:,cr-radius:cr+radius+1,cc-radius:cc+radius+1]
                    status[out]=2; cell_count+=1
                    local_usable=usable[patch_sl]
                    selected=np.flatnonzero(local_usable[:,support].mean(axis=1)>=cfg.min_frame_fraction)
                    frames_map[out]=len(selected)
                    if len(selected)<3:
                        continue
                    common=local_usable[selected].all(axis=0)&support
                    fraction=common.sum()/support.sum()
                    fields['common_fraction'][out]=fraction
                    if fraction<cfg.min_common_fraction:
                        continue
                    sr,sc=slope_row[cr,cc],slope_col[cr,cc]
                    if not np.isfinite([sr,sc]).all():
                        continue
                    slopes=tuple(float(np.round(v/cfg.slope_bin)*cfg.slope_bin) for v in (sr,sc))
                    try:
                        templates,parameters=bank(selected,slopes)
                    except ValueError:
                        continue
                    patch=stack[patch_sl][selected]
                    reference=np.median(np.where(np.isfinite(patch),patch,0.),axis=0)
                    projector=RegistrationProjector(common,np.full(len(selected),sigma),reference,
                                                    shadow_cfg.registration_sigma_px)
                    residual=projector.apply(patch)
                    rt=projector.apply(templates)
                    energy=np.sum(rt*rt,axis=(1,2))
                    raw=np.sum((templates[...,common]/sigma)**2,axis=(1,2))
                    ident=energy/np.maximum(raw,1e-30)
                    eligible=(energy>1e-12)&(ident>=shadow_cfg.min_identifiability)
                    if not eligible.any():
                        status[out]=3
                        continue
                    inner=np.sum(rt*residual,axis=(1,2))
                    contrast=np.clip(-inner/np.maximum(energy,1e-30),0,shadow_cfg.max_contrast)
                    improvement=np.maximum(0,-2*contrast*inner-contrast**2*energy)
                    scores=np.where(eligible,np.sqrt(improvement),-np.inf)
                    best=int(np.argmax(scores))
                    score=float(scores[best]); param=parameters[best]
                    # Model sensitivity for the smallest template, worst sampled
                    # root position. This is an expected signal calculation,
                    # not a recovered-object completeness estimate.
                    target=np.array([p[2]==min(cfg.heights_m) and p[3]==min(cfg.widths_m) for p in parameters])
                    target_energy=np.where(eligible[target],energy[target],0.)
                    worst=float(np.min(target_energy))
                    required=cfg.score_scale/np.sqrt(worst) if worst>1e-12 else np.inf
                    fields['required_contrast'][out]=required
                    sensitivity_ok[out]=required<=cfg.root_contrast
                    fields['score'][out]=score
                    fields['index'][out]=score/(cfg.score_scale+score)
                    fields['slope_row'][out],fields['slope_col'][out]=slopes
                    envelope_ok[out]=bool(strong_support[patch_sl][selected][:,common].all())
                    status[out]=1; assessed+=1
                    if score>=cfg.score_scale:
                        amplitude=float(contrast[best])
                        frame_delta=np.sum(residual**2-(residual+amplitude*rt[best])**2,axis=1)
                        roots.append(dict(row_px=float(cr+param[0]),col_px=float(cc+param[1]),
                                          score=score,contrast=amplitude,identifiability=float(ident[best]),
                                          template_height_m=param[2],template_width_m=param[3],
                                          endpoint_censored=bool(param[4]),frames=selected.tolist(),
                                          frame_delta_chi2=frame_delta.tolist()))
            if progress:
                progress(dict(tile_row=tr,tile_col=tc,cells_visited=cell_count,cells_assessed=assessed))
    roots.sort(key=lambda p:(-p['score'],p['row_px'],p['col_px']))
    distinct=[]; buckets={}; sep=shadow_cfg.min_separation_px
    for root in roots:
        key=(int(root['row_px']//sep),int(root['col_px']//sep))
        neighbours=[r for dr in (-1,0,1) for dc in (-1,0,1)
                    for r in buckets.get((key[0]+dr,key[1]+dc),[])]
        if all(np.hypot(root['row_px']-r['row_px'],root['col_px']-r['col_px'])>=sep for r in neighbours):
            distinct.append(root)
            buckets.setdefault(key,[]).append(root)
    payload=dict(regional=asdict(cfg),shadow=asdict(shadow_cfg),noise_sigma=float(sigma))
    return dict(**fields,status=status,frame_count=frames_map,envelope_ok=envelope_ok,
                sensitivity_ok=sensitivity_ok,candidates=distinct,cells_visited=cell_count,
                cells_assessed=assessed,search_truncated=False,configuration=payload,
                config_hash=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:16])
