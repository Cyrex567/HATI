"""Offline v2.5 checks. Independent tapered caster renderer; no ISIS calls."""
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.hati_core.shadow_likelihood import (
    NuisanceProjector, ShadowConfig, fit_template, map_sun_azimuth,
    search_stack, shadow_template)


AZ = np.array([0, 50, 100, 160, 210, 260, 310.])
EL = np.array([3, 4, 6, 4, 3, 5, 4.])


def independent_scene(seed=731, *, height=0.5, width=0.9, static=False,
                      noise=0.012, jitter=0., shape=(56, 56), root=(27.3, 28.2)):
    """Tapered shadow from a rounded cross-section; 12x supersampling.

    This renderer does not import or call the detector's shadow renderer. Blur
    is anisotropic; optional frame jitter and correlated noise stress mismatch.
    It remains a synthetic model, not independent lunar ground truth.
    """
    rng = np.random.default_rng(seed)
    ss, pixel = 12, 0.9
    yy, xx = np.indices((shape[0]*ss, shape[1]*ss), dtype=float)
    yy = ((yy+.5)/ss-.5-root[0])*pixel
    xx = ((xx+.5)/ss-.5-root[1])*pixel
    rows, cols = np.indices(shape, dtype=float)
    texture = 0.04*ndi.gaussian_filter(rng.normal(size=shape), 1.2)
    stain = ((abs(cols-17) < 2) & (rows > 12) & (rows < 43))*0.55
    result = []
    for i, (az, el) in enumerate(zip(AZ, EL)):
        dr, dc = np.cos(np.radians(az)), -np.sin(np.radians(az))
        along, across = yy*dr+xx*dc, -yy*dc+xx*dr
        local_height = height*np.sqrt(np.maximum(0, 1-(across/(width/2))**2))
        length = local_height/np.tan(np.radians(el))
        cast = ((along >= 0) & (along < length) & (abs(across) < width/2)).astype(float)
        cast = ndi.gaussian_filter(cast, (0.52*ss, 0.68*ss))
        cast = cast.reshape(shape[0], ss, shape[1], ss).mean(axis=(1, 3))
        frame = 1 + texture - stain + .0003*i*(rows-cols)
        if not static:
            frame = frame - .72*cast
        if jitter:
            frame = ndi.shift(frame, rng.normal(0, jitter, 2), order=1, mode="nearest")
        white = rng.normal(size=shape)
        colored = ndi.gaussian_filter(rng.normal(size=shape), .8)
        frame += noise*(.8*white + .6*colored / colored.std())
        result.append(frame)
    return np.asarray(result)


def small_config(**kwargs):
    return ShadowConfig(heights_m=(.3,.5,1.), widths_m=(.4,.9,1.8),
                        max_candidates=8, supersample=4, **kwargs)


class ShadowLikelihoodTests(unittest.TestCase):
    def test_projector_removes_static_texture_and_frame_planes(self):
        rng = np.random.default_rng(4)
        common = rng.random((21,21)) > .1
        yy, xx = np.indices(common.shape)
        albedo = rng.normal(size=common.shape)
        stack = np.array([albedo + .03*i*yy + .05*(i*i)*xx + i for i in range(5)])
        projector = NuisanceProjector(common, [.01,.04,.02,.06,.03])
        np.testing.assert_allclose(projector.apply(stack), 0, atol=1e-10)

    def test_projection_matches_explicit_least_squares(self):
        rng = np.random.default_rng(71)
        common = np.ones((5,5), bool)
        projector = NuisanceProjector(common, [.1,.2,.3])
        yy, xx = np.indices(common.shape)
        spatial = np.stack([np.ones(25), yy.ravel(), xx.ravel()], axis=1)
        design = np.column_stack([np.tile(np.eye(25), (3,1)), np.kron(np.eye(3), spatial)])
        weights = np.repeat(1/projector.sigma, 25)
        design *= weights[:,None]
        stack = rng.normal(size=(3,5,5))
        y = stack.ravel()*weights
        expected = y-design@np.linalg.lstsq(design, y, rcond=None)[0]
        np.testing.assert_allclose(projector.apply(stack).ravel(), expected, atol=1e-11)

    def test_stationary_stripe_has_no_shadow_likelihood(self):
        scene = independent_scene(static=True, noise=0)
        result = search_stack(scene, AZ, EL, .02, small_config(root_offsets_px=(0.,)))
        self.assertLess(max([c['score'] for c in result['candidates']] or [0]), 1e-8)

    def test_no_diversity_is_unidentifiable(self):
        cfg = small_config(root_offsets_px=(0.,))
        t, _ = shadow_template((25,25),(12,12),[40]*7,[4]*7,.5,.9,cfg)
        p = NuisanceProjector(np.ones((25,25), bool), np.ones(7)*.02)
        self.assertIsNone(fit_template(p, p.apply(1-.7*t), t, cfg))

    def test_independent_subpixel_caster_is_localized(self):
        result = search_stack(independent_scene(height=.3,width=.6), AZ, EL, .012, small_config())
        self.assertTrue(result['candidates'])
        best = result['candidates'][0]
        self.assertLess(np.hypot(best['row_px']-27.3,best['col_px']-28.2), 2.)
        self.assertGreater(best['score'], 8.)

    def test_contradictory_frames_reduce_shared_fit(self):
        cfg = small_config(root_offsets_px=(0.,))
        t, _ = shadow_template((25,25),(12,12),AZ,EL,.5,.9,cfg)
        p = NuisanceProjector(np.ones((25,25),bool), np.ones(7)*.02)
        clean = 1-.5*t
        mixed = clean.copy()
        mixed[::2] = 1+.5*t[::2]
        self.assertLess(fit_template(p,p.apply(mixed),t,cfg)['score'],
                        fit_template(p,p.apply(clean),t,cfg)['score']*.5)

    def test_width_is_fractional_and_independent_of_height(self):
        cfg = small_config(psf_sigma_px=0, solar_radius_deg=0, root_offsets_px=(0.,))
        a,_ = shadow_template((41,41),(20,20),[0]*3,[4]*3,.3,.4,cfg)
        b,_ = shadow_template((41,41),(20,20),[0]*3,[4]*3,.3,.9,cfg)
        self.assertGreater(b.sum(), a.sum()*1.5)

    def test_receiving_slope_shortens_shadow_and_censors_endpoint(self):
        cfg = small_config(root_offsets_px=(0.,))
        a,censored = shadow_template((25,25),(12,12),[0]*3,[4]*3,2.,.9,cfg)
        b,_ = shadow_template((25,25),(12,12),[0]*3,[4]*3,2.,.9,cfg,(.3,0.))
        self.assertTrue(censored)
        self.assertLess(b.sum(),a.sum())
        with self.assertRaises(ValueError):
            shadow_template((25,25),(12,12),[0]*3,[4]*3,.5,.9,cfg,(-.1,0.))

    def test_both_poles_and_rotated_raster(self):
        from affine import Affine
        cfg = '+proj=stere +lat_0={pole} +lon_0=0 +R=1737400 +units=m'
        for pole,lat,expected in [(-90,-80,29),(90,80,331)]:
            angle = map_sun_azimuth(cfg.format(pole=pole),Affine(.9,0,0,0,-.9,0),lat,29,0)
            self.assertLess(abs(angle-expected), .01)
        angle = map_sun_azimuth(cfg.format(pole=-90),Affine(0,-.9,0,-.9,0,0),-80,29,0)
        self.assertLess(abs(angle-299), .01)

    def test_missing_data_and_invalid_noise_are_explicit(self):
        scene = independent_scene()
        scene[0,:,:30] = np.nan
        result = search_stack(scene,AZ,EL,.02,small_config(root_offsets_px=(0.,)))
        self.assertLess(result['common_fraction'], .5)
        self.assertEqual(result['status'], 'experimental_unvalidated')
        with self.assertRaises(ValueError):
            search_stack(scene,AZ,EL,0,small_config())

    def test_scale_explicit_tri_agrees_on_a_plane(self):
        from src.heatmap.dem_features import tri_at_baseline
        values = []
        for scale in (1.,2.,4.):
            yy,xx = np.indices((80,80))
            z = .15*xx*scale + .06*yy*scale
            values.append(tri_at_baseline(z,scale,16.)[20:-20,20:-20].mean())
        np.testing.assert_allclose(values, values[0], atol=1e-6)

    def test_auc_ties_are_half_credit_and_order_invariant(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        from mare_control_v3 import auc
        self.assertEqual(auc(np.ones(8),[0,0,0,0,1,1,1,1]), .5)
        self.assertEqual(auc([0,0,1,1],[0,1,0,1]), .5)

    def test_catalog_bearing_uses_correct_hemisphere(self):
        from unittest.mock import patch
        sys.path.insert(0,str(ROOT/'scripts'))
        import solar_sweep_query as query
        with patch.object(query,'subsolar_longitude',return_value=29.):
            self.assertAlmostEqual(query.sun_azimuth_at(None,29.,-70.),0.)
            self.assertAlmostEqual(query.sun_azimuth_at(None,29.,70.),180.)

    def test_real_adapter_writes_georeferenced_products_without_isis(self):
        import json
        import tempfile
        from unittest.mock import patch
        import rasterio
        from rasterio.transform import from_origin
        sys.path.insert(0,str(ROOT/'scripts'))
        import athena_counterfactual as ac
        import shadow_roots_real as runner
        import shadow_kinematics_real as kin
        from sweep_contract import PROCESSING_VERSION
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            x,y = ac.touchdown_xy()
            transform = from_origin(x-32*.9,y+32*.9,.9,.9)
            crs = '+proj=stere +lat_0=-90 +lon_0=0 +R=1737400 +units=m'
            manifest = []
            for i in range(3):
                pid = f'MTEST{i}LE'
                cube = work/(pid+'.tif')
                with rasterio.open(cube,'w',driver='GTiff',height=64,width=64,count=1,
                                   dtype='float32',transform=transform,crs=crs) as dst:
                    dst.write(np.full((64,64),1000,dtype='float32'),1)
                (work/(pid+'.geom.json')).write_text(json.dumps(dict(
                    site_lat=ac.TD_LAT,site_lon=ac.TD_LON,az=i*100,elev=4)))
                manifest.append(dict(pid=pid,lev2=str(cube),shift_px=[0,0],half_px=32,
                                     utc='2024-01-01T00:00:00Z',processing_version=PROCESSING_VERSION,
                                     gate_pass=True))
            path = work/'manifest.json'
            path.write_text(json.dumps(manifest))
            argv = ['shadow_roots_real.py','--manifest',str(path),'--output',str(work/'result'),
                    '--half','20','--registration-sigma-px','.25','--max-candidates','2']
            with patch.object(ac,'ORTHO_IMG',work/'MTEST0LE.tif'), patch.object(kin,'SWEEP_DIR',work), \
                    patch.object(sys,'argv',argv):
                runner.main()
            result = json.loads((work/'result/run.json').read_text())
            self.assertEqual(result['candidates'],[])
            self.assertEqual(len(result['frames']),3)
            with rasterio.open(work/'result/common_support.tif') as support:
                self.assertEqual(support.shape,(40,40))
                self.assertTrue((support.read(1)==1).all())


if __name__ == '__main__':
    unittest.main()
