"""Fixed-grid shadow model diagnostics; no threshold fitting or safety decisions."""
from dataclasses import asdict
import numpy as np
from .shadow_likelihood import RegistrationProjector,shadow_template


def best_fit(projector,data,bank,min_identifiability=.02):
    residual=projector.apply(data); t=projector.apply(bank)
    energy=np.sum(t*t,axis=(1,2)); inner=np.sum(t*residual,axis=(1,2))
    raw=np.sum((bank[...,projector.common]/projector.sigma[:,None])**2,axis=(1,2))
    eligible=(energy>1e-12)&(energy/np.maximum(raw,1e-30)>=min_identifiability)
    contrast=np.clip(-inner/np.maximum(energy,1e-30),0,1)
    improvement=np.maximum(0,-2*contrast*inner-contrast**2*energy)
    improvement=np.where(eligible,improvement,0.)
    k=int(np.argmax(improvement))
    dof=(len(data)-1)*(projector.common.sum()-projector.q.shape[1])
    return dict(score=float(np.sqrt(improvement[k])),winner=k,contrast=float(contrast[k]),
        null_energy_per_dof=float(np.sum(residual**2)/max(dof,1)),
        improvement_fraction=float(improvement[k]/max(np.sum(residual**2),1e-30)),
        any_identifiable=bool(eligible.any()))


def diagnose_stack(stack,azimuths,elevations,cfg,*,sigma=.03,step_px=64,visibility=None,
                   slope_row=None,slope_col=None,progress=None):
    a=np.asarray(stack,float); az=np.asarray(azimuths,float); el=np.asarray(elevations,float)
    if a.ndim!=3 or len(a)<3 or az.shape!=(len(a),) or el.shape!=az.shape:
        raise ValueError('aligned frames and one measured geometry per frame required')
    if not np.isfinite(sigma) or sigma<=0 or step_px<1:
        raise ValueError('positive noise and grid spacing required')
    if not np.isfinite(az).all() or not np.isfinite(el).all():
        raise ValueError('finite measured illumination required')
    if visibility is not None and np.shape(visibility)!=a.shape:
        raise ValueError('visibility grid differs')
    h,w=a.shape[1:]; radius=cfg.radius_px
    if any(v is not None and np.shape(v)!=(h,w) for v in (slope_row,slope_col)):
        raise ValueError('slope fields differ from the image grid')
    yy,xx=np.indices((2*radius+1,)*2); support=np.hypot(yy-radius,xx-radius)<=cfg.root_support_px
    results=[]
    for r in range(radius,h-radius,step_px):
        for c in range(radius,w-radius,step_px):
            sl=np.s_[:,r-radius:r+radius+1,c-radius:c+radius+1]
            patch=a[sl]; valid=np.isfinite(patch)
            if visibility is not None: valid&=np.asarray(visibility)[sl]>=.99
            selected=np.flatnonzero(valid[:,support].mean(axis=1)>=.85)
            entry=dict(row_px=r,col_px=c,frames=selected.tolist(),status='insufficient_support')
            results.append(entry)
            if len(selected)<3: continue
            common=valid[selected].all(axis=0)&support
            entry['common_fraction']=float(common.sum()/support.sum())
            if entry['common_fraction']<.8: continue
            slopes=(float(slope_row[r,c]) if slope_row is not None else 0.,
                    float(slope_col[r,c]) if slope_col is not None else 0.)
            if not np.isfinite(slopes).all(): continue
            parameters=[]; templates=[]
            try:
                for dy in (-.5,.5):
                    for dx in (-.5,.5):
                        for ht in (.3,.6,1.2):
                            for width in (.6,1.2):
                                t,_=shadow_template(support.shape,(radius+dy,radius+dx),az[selected],el[selected],ht,width,cfg,slopes)
                                templates.append(t); parameters.append([dy,dx,ht,width])
            except ValueError:
                entry['status']='invalid_receiving_plane'; continue
            bank=np.asarray(templates); data=patch[selected]
            reference=np.median(np.where(np.isfinite(data),data,0),axis=0)
            entry['models']={}; entry['status']='assessed'
            for gain,name in ((False,'baseline'),(True,'per_frame_albedo_gain')):
                p=RegistrationProjector(common,[sigma]*len(selected),reference,cfg.registration_sigma_px,albedo_gain=gain)
                fitted=best_fit(p,data,bank,cfg.min_identifiability)
                fitted['selected_template']=parameters[fitted['winner']]
                fitted['geometry_stress_scores']=[best_fit(p,data,np.roll(bank,k,axis=1),cfg.min_identifiability)['score']
                    for k in range(1,min(4,len(selected)))]
                # Selection uses train intensities only. Held-out nuisance
                # fitting is allowed, but root/template/contrast stay fixed.
                if len(selected)>=6:
                    train=np.arange(len(selected))%2==0; test=~train
                    train_ref=np.median(np.where(np.isfinite(data[train]),data[train],0),axis=0)
                    pt=RegistrationProjector(common,[sigma]*int(train.sum()),train_ref,cfg.registration_sigma_px,albedo_gain=gain)
                    ph=RegistrationProjector(common,[sigma]*int(test.sum()),train_ref,cfg.registration_sigma_px,albedo_gain=gain)
                    fit=best_fit(pt,data[train],bank[:,train],cfg.min_identifiability)
                    d=ph.apply(data[test]); t=ph.apply(bank[fit['winner'],test]); amp=fit['contrast']
                    delta=np.sum(d*d-(d+amp*t)**2)
                    fitted['held_out']=dict(train_frames=selected[train].tolist(),test_frames=selected[test].tolist(),
                        training_score=fit['score'],fixed_contrast=amp,delta_chi2=float(delta),
                        template=parameters[fit['winner']],training_identifiable=fit['any_identifiable'])
                entry['models'][name]=fitted
            if progress: progress(len(results))
    assessed=[p for p in results if p['status']=='assessed']
    summary={}
    for name in ('baseline','per_frame_albedo_gain'):
        rows=[p['models'][name] for p in assessed]
        held=[p['held_out'] for p in rows if 'held_out' in p and p['held_out']['training_identifiable'] and p['held_out']['training_score']>=8]
        summary[name]=dict(assessed=len(rows),fraction_score_ge_8=float(np.mean([p['score']>=8 for p in rows])) if rows else None,
            median_null_energy_per_dof=float(np.median([p['null_energy_per_dof'] for p in rows])) if rows else None,
            median_score=float(np.median([p['score'] for p in rows])) if rows else None,
            fraction_true_geometry_beats_all_stress_scores=float(np.mean([p['score']>max(p['geometry_stress_scores']) for p in rows])) if rows else None,
            selected_training_warnings=len(held),held_out_positive_fraction=float(np.mean([p['delta_chi2']>0 for p in held])) if held else None)
    return dict(configuration=asdict(cfg),noise_sigma=sigma,step_px=step_px,patches=results,summary=summary,
        scope='Fixed-grid patch-centre template search, not the full regional search or object completeness.',
        limitations=['Diagnostic gain nuisance is opt-in here; production maps and thresholds are unchanged.',
            'Median texture is data-derived; reduced energy is a conditional diagnostic, not a calibrated chi-square test.',
            'Cyclic template-frame reassignment is a stress control with fixed observed masks, not an exchangeable null or p-value.',
            'Held-out split alternates manifest frames, may have similar illumination, and refits nuisance terms only.',
            'Lower scores can reflect lost signal; do not select a model merely because it produces fewer warnings.'])
