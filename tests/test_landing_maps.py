"""Offline physics and product regressions; never requires ISIS."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy import ndimage as ndi
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from src.hati_core.landing_terrain import plane_metrics,LandingConfig,terrain_assessment,buffer_evidence,fuse_landing
from src.hati_core.dem_shadow import predict_visibility
from src.hati_core.shadow_likelihood import NuisanceProjector,RegistrationProjector,ShadowConfig,shadow_template
from src.hati_core.regional_shadow import RegionalConfig,assess_regions
from test_shadow_likelihood import independent_scene,AZ,EL


class LandingTests(unittest.TestCase):
    def test_plane_and_curved_surface_against_direct_least_squares(self):
        yy,xx=np.indices((31,31)); z=1200+.2*yy*4-.1*xx*4+.007*(yy**2+xx**2)
        result=plane_metrics(z,4,24)
        keep=(yy-15)**2+(xx-15)**2<=3**2
        design=np.column_stack([np.ones(keep.sum()),(yy[keep]-15)*4,(xx[keep]-15)*4])
        fit=np.linalg.lstsq(design,z[keep],rcond=None)[0]
        residual=z[keep]-design@fit
        self.assertAlmostEqual(result['rms_height_m'][15,15],np.sqrt(np.mean(residual**2)),places=9)
        self.assertAlmostEqual(result['slope_row'][15,15],fit[1],places=10)
        self.assertAlmostEqual(result['positive_relief_m'][15,15],residual.max(),places=9)
        self.assertAlmostEqual(result['negative_relief_m'][15,15],-residual.min(),places=9)
        self.assertAlmostEqual(result['relief_p95_m'][15,15],np.percentile(abs(residual),95),places=9)
        planar=plane_metrics(1200+.2*yy*4-.1*xx*4,4,24)
        self.assertLess(planar['rms_height_m'][15,15],1e-5)

    def test_nodata_edges_and_unsupported_footprint(self):
        dem=np.zeros((40,40)); dem[20,20]=np.nan
        out=terrain_assessment(dem,4,LandingConfig(baselines_m=(16.,)))
        self.assertFalse(out['footprint_resolved'])
        self.assertEqual(out['effective_diameter_m'],16)
        self.assertTrue(np.isnan(out['score'][20,21]))
        self.assertTrue(np.isnan(out['score'][0,0]))
        self.assertEqual(out['score'][10,10],0)

    def test_buffers_and_fusion_never_average_away_high_or_fill_unknown(self):
        a=np.zeros((15,15)); a[7,7]=.9; a[7,8]=np.nan
        buffered,complete=buffer_evidence(a,1,2)
        self.assertEqual(buffered[7,8],.9); self.assertFalse(complete[7,8])
        self.assertTrue(np.isnan(buffered[0,0]))
        fused,status=fuse_landing(np.array([.9,.1,np.nan,.2]),np.array([.1,np.nan,.8,.3]))
        np.testing.assert_allclose(fused,[.9,np.nan,.8,.3],equal_nan=True)
        np.testing.assert_array_equal(status,[1,0,2,1])

    def test_flat_dem_and_analytic_ridge_shadow_both_directions(self):
        z=np.zeros((81,81)); z[30,:]=4
        p=predict_visibility(z,1,0,10,20)
        self.assertEqual(p['visible'][40,40],0)
        self.assertAlmostEqual(p['visible'][55,40],1)
        self.assertTrue(np.isnan(p['visible'][5,40]))
        reverse=predict_visibility(z[::-1],1,180,10,20)
        self.assertEqual(reverse['visible'][40,40],0)
        self.assertAlmostEqual(reverse['visible'][25,40],1)

    def test_height_envelope_and_holes_do_not_invent_illumination(self):
        z=np.zeros((61,61)); z[15,30]=np.nan
        out=predict_visibility(z,1,0,4,20,vertical_sigma_m=.2)
        self.assertTrue(np.isnan(out['visible'][25,30]))
        self.assertAlmostEqual(out['visible'][35,40],1)
        self.assertEqual(out['visible_conservative'][35,40],0)

    def test_batched_projectors_match_single_and_covariance_inverse(self):
        rng=np.random.default_rng(32); common=np.ones((9,9),bool)
        static=rng.normal(size=common.shape); sigma=.03; reg=.3
        p=RegistrationProjector(common,[sigma]*4,static,reg)
        bank=rng.normal(size=(5,4,9,9))
        np.testing.assert_allclose(p.apply(bank),np.stack([p.apply(a) for a in bank]),atol=1e-12)
        base=NuisanceProjector(common,[sigma]*4)
        gr,gc=np.gradient(ndi.gaussian_filter(static,.6))
        g=np.column_stack([gr.ravel(),gc.ravel()])*reg/sigma
        g-=base.q@(base.q.T@g)
        covariance=np.eye(common.sum())+g@g.T
        residual=base.apply(bank[0])
        expected=np.sum(residual*np.linalg.solve(covariance,residual.T).T)
        self.assertAlmostEqual(np.sum(p.apply(bank[0])**2),expected,places=8)

    def test_regional_search_is_uncapped_repeatable_and_tile_invariant(self):
        scene=independent_scene(shape=(40,40),root=(20.3,20.2),height=.3,width=.6)
        sc=ShadowConfig(registration_sigma_px=.25,max_candidates=1)
        cfg=dict(heights_m=(.3,),widths_m=(.6,))
        a=assess_regions(scene,AZ,EL,.012,sc,RegionalConfig(tile_px=4,**cfg))
        b=assess_regions(scene,AZ,EL,.012,sc,RegionalConfig(tile_px=16,**cfg))
        self.assertEqual(a['cells_visited'],16); self.assertEqual(a['cells_assessed'],16)
        self.assertFalse(a['search_truncated'])
        np.testing.assert_array_equal(a['score'],b['score'])
        self.assertGreater(np.nanmax(a['score']),8)
        root=a['candidates'][0]
        self.assertLess(np.hypot(root['row_px']-20.3,root['col_px']-20.2),2)

    def test_regional_no_geometry_diversity_and_no_visibility(self):
        stack=np.ones((3,32,32)); sc=ShadowConfig()
        rc=RegionalConfig(heights_m=(.3,),widths_m=(.6,))
        flat=assess_regions(stack,[40]*3,[4]*3,.03,sc,rc)
        self.assertEqual(flat['cells_assessed'],0)
        self.assertTrue((flat['status'][12:20,12:20]==3).all())
        dark=assess_regions(stack,[0,120,240],[4]*3,.03,sc,rc,visible=np.zeros_like(stack))
        self.assertTrue((dark['status'][12:20,12:20]==2).all())

    def test_translated_template_bank_matches_direct_renderer(self):
        scene=independent_scene(shape=(32,32),root=(16.3,16.2))
        sc=ShadowConfig(registration_sigma_px=.25)
        cfg=RegionalConfig(heights_m=(.3,),widths_m=(.6,))
        out=assess_regions(scene,AZ,EL,.012,sc,cfg)
        cr=cc=14; radius=12; shape=(25,25)
        yy,xx=np.indices(shape); common=np.hypot(yy-radius,xx-radius)<=6
        patch_data=scene[:,cr-radius:cr+radius+1,cc-radius:cc+radius+1]
        p=RegistrationProjector(common,[.012]*7,np.median(patch_data,axis=0),.25)
        data=p.apply(patch_data); scores=[]
        for dy in (-1.5,-.5,.5,1.5):
            for dx in (-1.5,-.5,.5,1.5):
                template,_=shadow_template(shape,(radius+dy,radius+dx),AZ,EL,.3,.6,sc)
                t=p.apply(template); energy=np.sum(t*t); inner=np.sum(t*data)
                contrast=np.clip(-inner/energy,0,1)
                scores.append(np.sqrt(max(0,-2*contrast*inner-contrast**2*energy)))
        self.assertAlmostEqual(out['score'][12,12],max(scores),places=6)

    def test_counterfactual_keeps_ties_and_unknown(self):
        from landing_maps import counterfactual
        maps={key:np.full((8,8),.2) for key in ('terrain','shadow','fused')}
        cf=counterfactual(maps,np.ones((8,8)),4,4)
        self.assertEqual(cf['maps']['fused']['hazard_percentile_in_common_region'],50)
        maps['fused'][4,4]=np.nan
        cf=counterfactual(maps,np.zeros((8,8)),4,4)
        self.assertEqual(cf['assessment'],'insufficient_support')
        self.assertIsNone(cf['maps']['fused']['index'])

    def test_scenario_comparison_detects_moved_roots_despite_equal_counts(self):
        import csv
        from affine import Affine
        from rasterio.crs import CRS
        from landing_maps import write_tif
        from compare_landing_scenarios import compare
        crs=CRS.from_string('+proj=stere +lat_0=-90 +lon_0=0 +R=1737400 +units=m')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); folders=[]
            for i in range(2):
                folder=root/str(i); folder.mkdir(); folders.append(folder)
                (folder/'run.json').write_text(json.dumps(dict(landing_config_hash='same',
                                                          provenance=dict(manifest_sha256='same'))))
                a=np.full((12,12),.2+.5*i); a[0,0]=np.nan
                write_tif(folder/'shadow_hazard.tif',a,Affine(.9,0,0,0,-.9,0),crs,'test')
                with (folder/'candidates.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=['row_px','col_px']); writer.writeheader()
                    writer.writerow(dict(row_px=2+i*8,col_px=2))
            report=compare(folders,root/'comparison')
            self.assertEqual(report['threshold_disagreement_fraction'],1)
            self.assertEqual(report['candidate_spatial_comparisons'][0]['first_with_neighbour_in_second'],0)

    def test_translated_radiance_does_not_accept_nodata_blends(self):
        import rasterio
        from rasterio.transform import from_origin
        import athena_counterfactual as ac
        from shadow_kinematics_real import frame_window
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'frame.tif'; x,y=ac.touchdown_xy()
            z=np.ones((64,64),dtype='float32'); z[32,32]=np.nan
            with rasterio.open(path,'w',driver='GTiff',height=64,width=64,count=1,dtype='float32',
                crs='+proj=stere +lat_0=-90 +lon_0=0 +R=1737400 +units=m',
                transform=from_origin(x-32*.9,y+32*.9,.9,.9),nodata=np.nan) as dst: dst.write(z,1)
            original=frame_window(path,[0,0],32,ac)
            # The map inverse can place the exact-boundary site one pixel to
            # either side through floating-point projection; locate the hole
            # in the actual site-centred read before testing its four blends.
            r,c=np.argwhere(np.isnan(original[10:-10,10:-10]))[0]+10
            a=frame_window(path,[.25,.25],32,ac)
            self.assertTrue(np.isnan(a[r:r+2,c:c+2]).all())
            self.assertEqual(a[20,20],1)

    def test_saved_reader_missing_sidecar_never_calls_isis(self):
        import rasterio
        from rasterio.transform import from_origin
        import athena_counterfactual as ac
        import sweep_products
        import shadow_kinematics_real as kin
        from sweep_contract import PROCESSING_VERSION
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); path=folder/'frame.tif'; x,y=ac.touchdown_xy()
            with rasterio.open(path,'w',driver='GTiff',height=64,width=64,count=1,dtype='float32',
                  crs='+proj=stere +lat_0=-90 +lon_0=0 +R=1737400 +units=m',
                  transform=from_origin(x-32*.9,y+32*.9,.9,.9)) as dst:
                dst.write(np.ones((64,64),dtype='float32'),1)
            entries=[dict(pid=f'MTEST{i}LE',lev2=str(path),shift_px=[0,0],half_px=32,gate_pass=True,
                          utc='2024-01-01T00:00:00Z',processing_version=PROCESSING_VERSION) for i in range(3)]
            manifest=folder/'manifest.json'; manifest.write_text(json.dumps(entries))
            with patch.object(ac,'ORTHO_IMG',path),patch.object(kin,'geometry_for',side_effect=AssertionError('ISIS path')):
                with self.assertRaises(FileNotFoundError): sweep_products.load_sweep(manifest,20)

    def test_real_product_adapter_exports_all_maps_and_retains_native_dem_samples(self):
        import rasterio
        from rasterio.transform import from_origin
        import athena_counterfactual as ac
        import shadow_kinematics_real as kin
        from sweep_products import load_sweep,load_dem_context
        from sweep_contract import PROCESSING_VERSION
        from landing_maps import run
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); x,y=ac.touchdown_xy()
            transform=from_origin(x-32*.9,y+32*.9,.9,.9)
            crs='+proj=stere +lat_0=-90 +lon_0=0 +R=1737400 +units=m'
            entries=[]
            for i in range(3):
                pid=f'MTEST{i}LE'; path=folder/(pid+'.tif')
                with rasterio.open(path,'w',driver='GTiff',height=64,width=64,count=1,dtype='float32',
                                   transform=transform,crs=crs) as dst:
                    dst.write(np.ones((64,64),dtype='float32'),1)
                (folder/(pid+'.geom.json')).write_text(json.dumps(dict(
                    site_lat=ac.TD_LAT,site_lon=ac.TD_LON,az=i*120,elev=4)))
                entries.append(dict(pid=pid,lev2=str(path),shift_px=[0,0],half_px=32,gate_pass=True,
                    utc='2024-01-01T00:00:00Z',processing_version=PROCESSING_VERSION))
            dem=folder/'dem.tif'; dt=from_origin(x-90+.17,y+90+.31,1.8,1.8)
            zz=np.zeros((100,100),dtype='float32'); zz[50,50]=1
            with rasterio.open(dem,'w',driver='GTiff',height=100,width=100,count=1,dtype='float32',
                               transform=dt,crs=crs) as dst: dst.write(zz,1)
            manifest=folder/'manifest.json'; manifest.write_text(json.dumps(entries))
            with patch.object(ac,'ORTHO_IMG',folder/'MTEST0LE.tif'),patch.object(ac,'DTM_PATH',dem), \
                 patch.object(kin,'geometry_for',side_effect=AssertionError('ISIS called')), \
                 patch('subprocess.run',side_effect=AssertionError('External executable called')):
                sweep=load_sweep(manifest,20)
                context=load_dem_context(dem,sweep['transform'],sweep['crs'],(40,40),20)
                self.assertEqual(np.nanmax(context['dem']),1)  # no resampling of impulse
                self.assertEqual(np.count_nonzero(context['dem']==1),1)
                cfg=LandingConfig(baselines_m=(8.,16.),horizon_distance_m=10.,dem_vertical_sigma_m=0.,
                                  navigation_margin_m=0.)
                report=run(sweep,context,folder/'out',cfg,ShadowConfig(registration_sigma_px=.25),
                           RegionalConfig(heights_m=(.3,),widths_m=(.6,)),.03)
            self.assertFalse(report['demo']); self.assertEqual(report['cells_visited'],16)
            for name in ('terrain','shadow','fused'):
                with rasterio.open(folder/'out'/(name+'_hazard.tif')) as src:
                    self.assertEqual(src.shape,(40,40)); self.assertEqual(src.transform,sweep['transform'])
                    self.assertEqual(src.crs,sweep['crs']); self.assertTrue(np.isnan(src.nodata))
                self.assertTrue((folder/'out'/(name+'_hazard.png')).exists())
            self.assertTrue((folder/'out/three_maps.png').exists())
            self.assertTrue((folder/'out/site_ranking.csv').exists())


if __name__=='__main__': unittest.main()
