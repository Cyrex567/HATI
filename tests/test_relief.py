"""Relief generator, rock-versus-relief competition and shape from shading (campaign T12-T16)."""
from argparse import Namespace
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'tests')]
from src.hati_core.adaptive_shadow import AdaptiveConfig
from src.hati_core.landing_terrain import LandingConfig
from src.hati_core.noise_scale import NoiseScaleConfig, relief_consistency
from src.hati_core.regional_shadow import RegionalConfig
from src.hati_core.relief_hypothesis import calibrate_margins, compare_models, confusion
from src.hati_core.relief_scenes import lit_fraction, relief_feature, render_relief
from src.hati_core.rock_scenes import make_rock
from src.hati_core.sfs import rock_factor, solve_sfs
from src.hati_core.shadow_likelihood import ShadowConfig

# The Athena sweep's measured geometry: azimuths clockwise from map up, elevations in degrees.
AZ = [43.1, 43.5, 69., 73.7, 326.7, 354.6, 6., 24.9]
EL = [4.35, 3.59, 3.47, 3.38, 3.58, 3.63, 3.28, 3.69]
QUIET = dict(pixel_m=.9, seed=1, noise=0., texture=0., stain=0., frame_plane=0.)


def detrended_correlation(a, b):
    rows, cols = np.indices(a.shape)
    design = np.column_stack([np.ones(a.size), rows.ravel(), cols.ravel()])
    def detrend(x):
        return x.ravel()-design@np.linalg.lstsq(design, x.ravel(), rcond=None)[0]
    x, y = detrend(a), detrend(b)
    return float(x@y/np.sqrt((x@x)*(y@y)))


class GeneratorTests(unittest.TestCase):
    def test_flat_ground_renders_at_one(self):
        np.testing.assert_allclose(render_relief((24, 24), [90.], [3.5], **QUIET)['stack'], 1, atol=1e-9)

    def test_mounds_and_bowls_swap_bright_and_dark_along_the_sun(self):
        # Sun in the east: columns increase toward it.
        for kind, sign in (('mound', 1), ('bowl', -1)):
            f = render_relief((48, 48), [90.], [3.5], features=[relief_feature(kind, 8., 2.)], **QUIET)['stack'][0]
            self.assertGreater(sign*(f[24, 27]-f[24, 20]), .3)

    def test_declared_slope_and_cast_shadow_reach(self):
        out = render_relief((64, 64), [90.], [3.5], features=[relief_feature('mound', 8., 10.)], **QUIET)
        self.assertAlmostEqual(out['slope_deg'].max(), 10., delta=.05)
        dark = np.flatnonzero(out['stack'][0][32] < .5)
        reach = out['height_m'].max()/np.tan(np.radians(3.5))/.9
        self.assertLess(dark.max(), 32)                    # on the far side from the Sun
        self.assertGreater(32-dark.min(), .5*reach)
        self.assertLess(32-dark.min(), 1.2*reach+4)

    def test_horizon_sweep_matches_a_wall(self):
        h = np.zeros((40, 40)); h[:, 30:] = 1.            # 1 m step, Sun from the east
        lit = lit_fraction(h, .5, 90., 5., solar_radius_deg=0.)
        shadow = np.flatnonzero(lit[20] < .5)
        self.assertAlmostEqual(30-shadow.min(), 1/np.tan(np.radians(5))/.5, delta=1.5)

    def test_rock_has_a_bright_face_and_a_long_thin_shadow(self):
        rock = make_rock(3, (32.3, 40.2), .6, .6, aspect=1.35, yaw_deg=0)
        f = render_relief((64, 64), [90.], [3.5], rocks=[rock], **QUIET)['stack'][0]
        self.assertGreater(f[32, 40], 1.2)
        self.assertGreater(len(np.flatnonzero(f[32, :40] < .9)), 6)   # about 11 px expected

    def test_injected_rock_factor_is_one_away_from_sites(self):
        factor = rock_factor((64, 64), [(32.3, 32.2, .6)], [90.], [3.5], .9, seed=2, window_px=32)
        self.assertEqual(factor.shape, (1, 64, 64))
        np.testing.assert_allclose(factor[0, :10], 1)
        self.assertLess(factor[0, 32, 20:32].min(), .8)


class CompetitionTests(unittest.TestCase):
    sc = ShadowConfig(radius_px=12, root_support_px=6, supersample=4)
    rc = RegionalConfig()

    def compare(self, **scene):
        out = render_relief((25, 25), AZ, EL, pixel_m=.9, seed=11, noise=.01, **scene)
        return compare_models(out['stack'], np.ones_like(out['stack'], bool), AZ, EL, .01, self.sc, self.rc, (.9, 1.8, 3.6))

    def test_withheld_frames_prefer_relief_for_a_mound_and_rock_for_a_rock(self):
        mound = self.compare(features=[relief_feature('mound', 6., 2.)])
        rock = self.compare(rocks=[make_rock(5, (12.3, 12.2), .6, .6, aspect=1.35)])
        self.assertGreater(mound['rock_score'], 8)          # the unchanged detector fires on a 2-degree mound
        self.assertGreater(mound['gain_relief'], 2*mound['gain_rock'])
        self.assertGreater(rock['gain_rock'], 2*rock['gain_relief'])

    def test_margins_meet_their_declared_targets(self):
        rng = np.random.default_rng(0)
        rows = [dict(truth='rock', status='assessed', gain_rock=2+rng.normal(), gain_relief=1+rng.normal()) for _ in range(200)]
        rows += [dict(truth='relief', status='assessed', gain_rock=1+rng.normal(), gain_relief=3+rng.normal()) for _ in range(200)]
        rows += [dict(truth='none', status='assessed', gain_rock=abs(rng.normal(0, .1)), gain_relief=abs(rng.normal(0, .1)))
                 for _ in range(200)]
        margins = calibrate_margins(rows, dict(rock_called_relief=.05, relief_called_rock=.1, blank_called_signal=.1))
        table = confusion(rows, margins)
        self.assertLessEqual(table['rock']['relief_like'], 10)
        self.assertLessEqual(table['relief']['rock_like'], 20)
        self.assertGreaterEqual(table['none']['none'], 180)


class ShapeFromShadingTests(unittest.TestCase):
    def test_planted_relief_is_recovered_and_rock_shadows_survive(self):
        features = [relief_feature('mound', 8., 2., centre_px=(30, 30)), relief_feature('bowl', 10., 3., centre_px=(66, 70)),
                    relief_feature('ripples', 6., 1., seed=2)]
        rock = make_rock(9, (40.3, 72.2), .6, .6, aspect=1.35)
        out = render_relief((96, 96), AZ, EL, pixel_m=.9, seed=5, noise=.01, features=features, rocks=[rock])
        solved = solve_sfs(out['stack'], np.ones_like(out['stack'], bool), AZ, EL, .9, grid_px=2, smoothness=3.)
        inner = np.s_[8:-8, 8:-8]
        self.assertGreater(solved['explained_fraction'], .85)
        self.assertGreater(detrended_correlation(solved['height_m'][inner], out['height_m'][inner]), .85)
        window = np.s_[6, 36:46, 60:73]
        self.assertAlmostEqual(out['stack'][window].min(), solved['corrected'][window].min(), delta=.05)

    def test_residual_follows_the_sun_for_relief_but_not_for_noise(self):
        out = render_relief((96, 96), AZ, EL, pixel_m=.9, seed=5, noise=.01,
                            features=[relief_feature('ripples', 6., 2., seed=4)], texture=0., stain=0., frame_plane=0.)
        ones = np.ones_like(out['stack']); flat = np.zeros((96, 96)); cfg = NoiseScaleConfig(patch_px=24)
        relief = relief_consistency(out['stack'], ones, flat, flat, AZ, EL, cfg)
        self.assertEqual(relief['fraction_shuffled_at_or_above_true'] <= .01, True)
        self.assertGreater(relief['explained_true_geometry'], .7)
        noise = 1+.02*np.random.default_rng(8).normal(size=out['stack'].shape)
        null = relief_consistency(noise, ones, flat, flat, AZ, EL, cfg)
        self.assertGreater(null['fraction_shuffled_at_or_above_true'], .01)
        self.assertLess(null['explained_true_geometry'], .5)


def relief_bundle(path, *, size=96):
    """A small eight-frame bundle with relief and one rock, on the Athena geometry."""
    from pyproj import CRS
    from affine import Affine
    features = [relief_feature('ripples', 6., 1.5, seed=3), relief_feature('mound', 8., 2., centre_px=(64, 30))]
    rock = make_rock(4, (60.3, 66.2), .6, .6, aspect=1.35)
    stack = render_relief((size, size), AZ, EL, pixel_m=.9, seed=21, noise=.015, features=features, rocks=[rock])['stack']
    sc = ShadowConfig(radius_px=6, root_support_px=3, supersample=2)
    rc = RegionalConfig(heights_m=(.3, .6), widths_m=(.6,), tile_px=16)
    crs = CRS.from_proj4('+proj=stere +lat_0=-90 +lat_ts=-85 +lon_0=0 +R=1737400 +units=m')
    transform = Affine(.9, 0, 0, 0, -.9, 0); ids = [f'relief{i}' for i in range(len(AZ))]
    run = dict(frames=[dict(pid=p) for p in ids], azimuths_map=AZ, elevations=EL, transform=tuple(transform), crs=crs.to_wkt(),
               image_posting_m=.9, counterfactual=dict(row_px=20, col_px=20), demo=True,
               landing=asdict(LandingConfig(baselines_m=(4.,), footprint_diameter_m=4., navigation_margin_m=1.,
                                            horizon_distance_m=8., dem_vertical_sigma_m=0.)),
               shadow_configuration=dict(shadow=asdict(sc), regional=asdict(rc), noise_sigma=.015))
    buf = io.BytesIO()
    np.savez_compressed(buf, stack=stack, visibility=np.ones_like(stack), frame_ids=ids, azimuths=np.array(AZ), elevations=np.array(EL),
                        slope_row=np.zeros((size, size)), slope_col=np.zeros((size, size)), transform=tuple(transform), crs=crs.to_wkt())
    payload = buf.getvalue()
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('aligned_stack.npz', payload)
        z.writestr('source_run.json', json.dumps(run))
        z.writestr('model_diagnostics.json', json.dumps(dict(provenance=dict(stack_sha256=hashlib.sha256(payload).hexdigest()))))


class ReliefCampaignTests(unittest.TestCase):
    def test_t12_to_t16_on_a_relief_bundle_without_external_calls(self):
        from saturation_experiments import Experiment, t1, t12, t13, t14, t16
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); bundle = out/'input.zip'; relief_bundle(bundle)
            cfg = json.loads((ROOT/'configs/saturation_campaign.json').read_text())
            # Two workers on Linux exercise T16's forked pool in the software checks.
            cfg.update(synthetic_seeds=1, synthetic_locations=[[.5, .5]],
                       adaptive=asdict(AdaptiveConfig(scale_factors=(1, 2), heights_m=(.2, .4, .6), widths_m=(.3, .6, .9),
                                                      max_cells=2, workers=1 if os.name == 'nt' else 2)),
                       rock_roi_cells=1, noise_scale=dict(patch_px=24, max_slope=.05), noise_control_seeds=1, null_seeds=1,
                       null_caster_heights_m=[.6], null_render_noise='measured', null_relief_scenes=[['ripples', 6., 2.]],
                       relief_supersample=2, relief_kinds=['mound'], relief_sizes_m=[4.], relief_slopes_deg=[2.], relief_seeds=1,
                       relief_rock_heights_m=[.6], relief_scene_px=32, relief_competition_sizes_m=[4.],
                       relief_competition_slopes_deg=[2.], relief_calibration_seeds=2, relief_athena_cells=4,
                       relief_touchdown_radius_px=4, sfs_injection_sites=2, sfs_injection_spacing_px=20)
            config = out/'config.json'; config.write_text(json.dumps(cfg))
            def run(stage, fn):
                args = Namespace(stage=stage, bundle=bundle, config=config, output=out/'stages'/stage,
                                 campaign=out, dem=None, thermal=None, held_out=None, rock_catalog=None)
                with patch('subprocess.run', side_effect=AssertionError('external ingestion forbidden')):
                    return fn(Experiment(args))
            run('T1', t1)
            noise = run('T12', t12)
            self.assertTrue(noise['relief_consistency']['available'])
            self.assertEqual(noise['control_generator'], 'relief')
            self.assertEqual(len(noise['control_passes']), 4)
            self.assertIn('relief', [s['kind'] for s in noise['control_passes'][0]['scenarios']])
            compete = run('T13', t13)
            self.assertEqual(compete['status'], 'PARTIAL')
            self.assertEqual(compete['competition_sigma_source'], 'T12 relief-corrected residual scale')
            self.assertIn('rock', compete['evaluation_confusion'])
            self.assertTrue(compete['athena_summary']['all_sampled']['cells'] > 0)
            self.assertTrue((out/'stages/T13/relief_competition.png').exists())
            sfs = run('T14', t14)
            self.assertEqual(sfs['status'], 'PARTIAL')
            self.assertGreater(sfs['sfs']['explained_fraction'], .5)
            self.assertGreater(sfs['injected_sites'], 0)
            self.assertIsNotNone(sfs['exceedance']['after_assumed_sigma'])
            for name in ('shape_from_shading.png', 'sfs_height_m.tif', 'sfs_slope_deg.tif', 'injection.json'):
                self.assertTrue((out/'stages/T14'/name).exists(), name)
            nulls = run('T16', t16)
            self.assertIn('relief_corrected_sigma', nulls['render_noise_source'])
            self.assertIn('ripples_2deg', {s['kind'] for s in nulls['summaries']})


if __name__ == '__main__':
    unittest.main()
