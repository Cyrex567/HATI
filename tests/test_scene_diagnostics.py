"""Independent geometry, morphology, registration and portable replay checks."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy import ndimage as ndi
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from src.hati_core.root_footprint import buffer_roots
from src.hati_core.scene_diagnostics import SceneConfig,broad_dark_discrepancy,local_registration
from src.hati_core.stereo_confidence import category_counts,sample_categories
from src.hati_core.warning_attribution import describe_warning


class SceneTests(unittest.TestCase):
    def test_exact_disks_match_independent_pairwise_distances(self):
        rng=np.random.default_rng(15)
        roots=np.column_stack([rng.uniform(0,12,60),rng.uniform(0,10,60),rng.uniform(0,20,60)])
        roots=np.vstack([roots,[6.5,5.5,80],[6.5,4.5,80]])
        a=buffer_roots(roots,np.ones((13,11),bool),.9,2.1)
        for r in range(13):
            for c in range(11):
                keep=np.hypot(roots[:,0]-r,roots[:,1]-c)*.9 <= 2.1
                expected=max(roots[keep,2]) if keep.any() else np.nan
                np.testing.assert_allclose(a['score'][r,c],expected,equal_nan=True)
                if keep.any():
                    winner=sorted(roots[keep].tolist(),key=lambda p:(-p[2],p[0],p[1]))[0]
                    self.assertEqual([a['root_row'][r,c],a['root_col'][r,c]],winner[:2])

    def test_weaker_inside_root_survives_stronger_outside_cell_winner(self):
        roots=np.array([[10,13,120],[10,11,16]])
        a=buffer_roots(roots,np.ones((24,24),bool),1,2)
        self.assertEqual(a['score'][10,10],16)
        self.assertEqual(a['root_col'][10,10],11)
        unknown=buffer_roots(roots,np.zeros((24,24),bool),1,2)
        self.assertGreater(unknown['index'][10,10],.5)
        self.assertTrue(np.isnan(unknown['index'][0,0]))
        empty=buffer_roots([],np.ones((24,24),bool),1,2)
        self.assertTrue(np.isnan(empty['index']).all())

    def test_exact_attribution_matches_map_and_distinguishes_test_coordinate(self):
        a=buffer_roots([[10,12,16]],np.ones((24,24),bool),1,2)
        maps={k:a['index'] for k in ('terrain','shadow','fused')}
        raw=np.zeros((24,24));status=np.ones_like(raw)
        result=describe_warning(maps,raw,status,status,10.1,10.1,1,2,buffered_source=a)
        self.assertTrue(result['source']['root']['within_map_disk'])
        self.assertFalse(result['source']['root']['within_nominal_radius'])
        self.assertAlmostEqual(result['source']['regional_index'],maps['shadow'][10,10])

    def test_broad_darkness_distinguishes_narrow_structure_and_nominal_shadow(self):
        a=np.ones((4,64,64));a[:,20:35,20:35]=.05;a[:,45,10:50]=.05
        v=np.ones_like(a)
        r=broad_dark_discrepancy(a,v)
        self.assertTrue(r['flag'][27,27]);self.assertFalse(r['flag'][45,27])
        self.assertEqual(r['discrepant_frames'][27,27],4)
        v[:,20:35,20:35]=0
        shadow=broad_dark_discrepancy(a,v)
        self.assertFalse(shadow['flag'][27,27]);self.assertFalse(shadow['assessed'][27,27])

    def test_holes_and_single_frame_do_not_supply_repeated_discrepancy(self):
        a=np.ones((4,32,32));a[0,10:20,10:20]=.01
        out=broad_dark_discrepancy(a,np.ones_like(a))
        self.assertFalse(out['flag'].any())
        a[:]=np.nan
        out=broad_dark_discrepancy(a,np.ones_like(a))
        self.assertFalse(out['assessed'].any());self.assertFalse(out['flag'].any())
        self.assertTrue(np.isnan(out['fraction']).all())
        # A one-pixel-wide component against an image edge has no broad core.
        a=np.ones((4,32,32));a[:,:,0]=0
        self.assertFalse(broad_dark_discrepancy(a,np.ones_like(a))['flag'].any())

    def test_local_offset_sign_gain_invariance_and_wraparound_pair_selection(self):
        rng=np.random.default_rng(28);a=1+ndi.gaussian_filter(rng.normal(size=(192,192)),.6)
        b=np.roll(a,(2,-3),axis=(0,1))*1.4+.3
        cfg=SceneConfig(tile_px=64,max_shift_px=4,highpass_sigma_px=1,min_common_fraction=.5)
        result=local_registration(np.array([a,b]),[358,2],[4,4],cfg)
        rows=[r for r in result['pairs'][0]['tiles'] if r['status']=='measured_apparent_offset']
        self.assertGreater(len(rows),0)
        for row in rows:
            self.assertEqual((row['row_offset_px'],row['col_offset_px']),(2,-3))
            self.assertGreater(row['ncc'],.999)
            self.assertFalse(row['search_boundary'])
        self.assertEqual(local_registration(np.array([a,b]),[0,90],[4,4],cfg)['pairs'],[])

    def test_flat_or_missing_registration_does_not_report_zero_offset(self):
        cfg=SceneConfig(tile_px=64,max_shift_px=4,highpass_sigma_px=1,min_common_fraction=.5)
        for value in (1.,np.nan):
            result=local_registration(np.full((2,192,192),value),[0,0],[4,4],cfg)
            self.assertTrue(all(p['status']!='measured_apparent_offset' for p in result['pairs'][0]['tiles']))

    def test_confidence_categories_do_not_rank_manual_edits_above_correlation(self):
        a=np.array([[14.,15.],[4.,99.]])
        rows={p['code']:p for p in category_counts(a)}
        self.assertEqual(rows[14]['description'],'Successful correlation')
        self.assertEqual(rows[15]['description'],'Manually edited')
        self.assertEqual(rows[99]['description'],'Undocumented code')
        self.assertEqual(sample_categories(a,0,0)['window_shape'],[2,2])
        with self.assertRaises(ValueError):sample_categories(a,-1,0)

    def test_discrepancy_blocks_low_fusion_but_preserves_high_evidence(self):
        import tempfile,rasterio
        from unittest.mock import patch
        import landing_maps as lm
        from src.hati_core.landing_terrain import LandingConfig
        from src.hati_core.shadow_likelihood import ShadowConfig
        from src.hati_core.regional_shadow import RegionalConfig
        sweep,context=lm.demo_inputs();sweep['stack'][:]=1.;context['dem'][:]=0.
        sweep['stack'][:,35:50,35:50]=.05
        shape=(80,80);yy,xx=np.indices(shape)
        scores=np.ones(shape);scores[40,40]=16
        regional={k:np.ones(shape) for k in ('required_contrast','common_fraction','status','frame_count',
                  'envelope_ok','sensitivity_ok','null_energy_per_dof','best_contrast','endpoint_censored',
                  'best_height_m','best_width_m','dimension_at_boundary','endpoint_censored_count','endpoint_missing_count')}
        regional['envelope_ok']=np.ones(shape,bool);regional['sensitivity_ok']=np.ones(shape,bool)
        regional.update(score=scores,index=scores/(scores+8),best_root_row_px=yy,best_root_col_px=xx,
                        root_evidence=np.column_stack([yy.ravel(),xx.ravel(),scores.ravel()]),candidates=[],
                        configuration={},config_hash='test',cells_visited=6400,cells_assessed=6400)
        cfg=LandingConfig(baselines_m=(8.,),horizon_distance_m=10.,dem_vertical_sigma_m=0.)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with patch.object(lm,'assess_regions',return_value=regional),patch.object(lm,'make_previews'):
                lm.run(sweep,context,out,cfg,ShadowConfig(),RegionalConfig(),.03,is_demo=True)
            def read(name):
                with rasterio.open(out/(name+'.tif')) as src:return src.read(1)
            fused,status=read('fused_hazard'),read('fusion_status')
            self.assertGreater(fused[40,40],.5);self.assertEqual(status[40,40],2)
            # This centre includes the discrepant feature but not the high root.
            self.assertTrue(np.isnan(fused[40,52]));self.assertEqual(status[40,52],0)
            self.assertLess(fused[18,18],.5);self.assertEqual(status[18,18],1)

    def test_portable_review_checks_hash_and_runs_without_external_executables(self):
        import hashlib,io,json,tempfile,zipfile
        from unittest.mock import patch
        from affine import Affine
        from pyproj import CRS
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot  # host font initialization before guard
        from review_shadow_bundle import review,read_bundle
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);buf=io.BytesIO();a=np.ones((3,32,32))
            crs=CRS.from_proj4('+proj=stere +lat_0=-90 +R=1737400 +units=m')
            transform=Affine(.9,0,0,0,-.9,0);ids=['a','b','c']
            np.savez_compressed(buf,stack=a,visibility=a,frame_ids=ids,azimuths=[0,120,240],elevations=[4,4,4],
                                transform=tuple(transform),crs=crs.to_wkt())
            payload=buf.getvalue()
            run=dict(frames=[dict(pid=p) for p in ids],azimuths_map=[0,120,240],elevations=[4,4,4],
                     transform=tuple(transform),crs=crs.to_wkt(),image_posting_m=.9,
                     counterfactual=dict(row_px=16.5,col_px=16.5))
            def bundle(path,digest):
                with zipfile.ZipFile(path,'w') as z:
                    z.writestr('aligned_stack.npz',payload)
                    z.writestr('source_run.json',json.dumps(run))
                    z.writestr('model_diagnostics.json',json.dumps(dict(provenance=dict(stack_sha256=digest))))
            bundle(folder/'good.zip',hashlib.sha256(payload).hexdigest())
            with patch('subprocess.run',side_effect=AssertionError('external executable forbidden')):
                result=review(folder/'good.zip',folder/'out')
            self.assertEqual(result['flag_fraction'],0)
            self.assertTrue((folder/'out/scene_dark_flag.tif').exists())
            with self.assertRaises(FileExistsError):review(folder/'good.zip',folder/'out')
            bundle(folder/'bad.zip','wrong')
            with self.assertRaises(ValueError):read_bundle(folder/'bad.zip')


if __name__=='__main__':unittest.main()
