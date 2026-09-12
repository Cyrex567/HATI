"""Attribution and diagnostic-model regressions, without ISIS."""
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from src.hati_core.warning_attribution import describe_warning
from src.hati_core.shadow_likelihood import RegistrationProjector,ShadowConfig,shadow_template
from src.hati_core.shadow_diagnostics import diagnose_stack,best_fit


class DiagnosticTests(unittest.TestCase):
    def test_unknown_direct_cell_is_not_reported_as_detection(self):
        a=np.full((40,40),20.);a[20:24,20:24]=np.nan
        a[24:28,12:16]=120
        maps=dict(terrain=np.full_like(a,.3),shadow=np.full_like(a,.94),fused=np.full_like(a,.94))
        status=np.where(np.isfinite(a),1,2);common=np.where(status==1,1.,.76)
        candidates=[dict(row_px=25.5,col_px=12.5,score=120.,contrast=1.,endpoint_censored=True)]
        result=describe_warning(maps,a,status,common,20.5,20.5,1,7,candidates=candidates)
        self.assertEqual(result['warning_origin'],'regional_warning_direct_cell_unassessed')
        self.assertIsNone(result['raw_cell']['score'])
        self.assertEqual(result['maps']['shadow']['threshold_exceedance_fraction_available'],1)
        self.assertFalse(result['source']['root']['within_nominal_radius'])

    def test_missing_deduplicated_source_stays_unresolved(self):
        a=np.full((32,32),10.); maps={k:a/18 for k in ('terrain','shadow','fused')}
        r=describe_warning(maps,a,np.ones_like(a),np.ones_like(a),16,16,1,2)
        self.assertIsNone(r['source']['root'])
        self.assertEqual(r['source']['root_resolution'],'not recorded')

    def test_albedo_gain_projection_matches_dense_nuisance_least_squares(self):
        rng=np.random.default_rng(56);static=rng.normal(size=(7,7));common=np.ones((7,7),bool)
        yy,xx=np.indices(common.shape)
        spatial=np.column_stack([np.ones(49),yy.ravel(),xx.ravel(),static.ravel()])
        design=np.column_stack([np.tile(np.eye(49),(3,1)),np.kron(np.eye(3),spatial)])
        stack=rng.normal(size=(3,7,7));y=stack.ravel()/.03
        expected=y-design@np.linalg.lstsq(design,y,rcond=None)[0]
        p=RegistrationProjector(common,[.03]*3,static,0,albedo_gain=True)
        np.testing.assert_allclose(p.apply(stack).ravel(),expected,atol=1e-10)

    def test_gain_only_texture_is_removed_but_moving_shadow_survives(self):
        from scipy.ndimage import gaussian_filter
        rng=np.random.default_rng(8);static=1+gaussian_filter(rng.normal(size=(25,25)),1)*.2
        common=np.ones((25,25),bool);az=[0,50,100,170,250,310];el=[4]*6
        gain=np.array([.8,1.1,1.2,.9,1,.85])
        data=gain[:,None,None]*static
        base=RegistrationProjector(common,[.03]*6,static,0)
        gainp=RegistrationProjector(common,[.03]*6,static,0,albedo_gain=True)
        self.assertGreater(np.linalg.norm(base.apply(data)),1)
        np.testing.assert_allclose(gainp.apply(data),0,atol=1e-12)
        cfg=ShadowConfig();t,_=shadow_template((25,25),(12,12),az,el,.6,1.2,cfg)
        fit=best_fit(gainp,data-.6*t,np.array([t]))
        self.assertGreater(fit['score'],8)
        self.assertAlmostEqual(fit['contrast'],.6,places=10)

    def test_fixed_grid_controls_repeat_and_do_not_relabel_maps(self):
        rng=np.random.default_rng(2);a=1+rng.normal(0,.03,(6,32,32))
        args=(a,[0,50,100,170,250,310],[4]*6,ShadowConfig(supersample=2))
        first=diagnose_stack(*args);second=diagnose_stack(*args)
        self.assertEqual(first,second)
        self.assertEqual(len(first['patches']),1)
        self.assertIn('held_out',first['patches'][0]['models']['baseline'])
        self.assertEqual(len(first['patches'][0]['models']['baseline']['geometry_stress_scores']),3)
        self.assertIn('production maps',first['limitations'][0])

    def test_bundle_exports_intensities_without_external_executables(self):
        import hashlib,json,tempfile,zipfile
        from unittest.mock import patch
        import landing_maps as lm
        import diagnose_landing_run as cli
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot  # initialize host fonts before the executable guard
        from src.hati_core.landing_terrain import LandingConfig
        from src.hati_core.regional_shadow import RegionalConfig
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);manifest=folder/'manifest.json';manifest.write_text('[]')
            sweep,context=lm.demo_inputs()
            sweep['frames']=[dict(pid=f'TEST{i}') for i in range(len(sweep['stack']))]
            cfg=LandingConfig(baselines_m=(4.,),horizon_distance_m=10.,dem_vertical_sigma_m=0.)
            provenance=dict(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                            arguments=dict(half=40,before='2025-03-06T00:00:00Z'))
            with patch('subprocess.run',side_effect=AssertionError('External executable forbidden')):
                lm.run(sweep,context,folder/'run',cfg,ShadowConfig(),RegionalConfig(heights_m=(.3,),widths_m=(.6,)),.03,
                       is_demo=True,provenance=provenance)
                argv=['diagnose_landing_run.py','--run-dir',str(folder/'run'),'--manifest',str(manifest),'--step-px','128']
                with patch.object(sys,'argv',argv),patch.object(cli,'load_sweep',return_value=sweep):cli.main()
            output=folder/'run/diagnostics_v254'
            with np.load(output/'aligned_stack.npz',allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['stack'],sweep['stack'])
            with zipfile.ZipFile(output/'hati_diagnostic_bundle.zip') as archive:
                self.assertIn('model_diagnostics.json',archive.namelist())
                self.assertIn('aligned_stack.npz',archive.namelist())
            self.assertEqual(json.loads((output/'model_diagnostics.json').read_text())['step_px'],128)

    def test_held_out_frames_cannot_change_selected_training_template(self):
        az=[0,50,100,170,250,310];el=[4]*6;cfg=ShadowConfig(supersample=2)
        t,_=shadow_template((32,32),(12.5,12.5),az,el,.6,1.2,cfg)
        a=1-.6*t;b=a.copy();b[1::2]=1+.6*t[1::2]
        first=diagnose_stack(a,az,el,cfg)['patches'][0]['models']['baseline']['held_out']
        second=diagnose_stack(b,az,el,cfg)['patches'][0]['models']['baseline']['held_out']
        self.assertEqual(first['template'],second['template'])
        self.assertEqual(first['fixed_contrast'],second['fixed_contrast'])
        self.assertGreater(first['delta_chi2'],0)
        self.assertLess(second['delta_chi2'],0)


if __name__=='__main__':unittest.main()
