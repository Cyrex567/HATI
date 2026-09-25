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
from src.hati_core.relief_hypothesis import calibrate_margins, compare_models, confusion, sign_confusion
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

    def test_ground_tilted_toward_the_sun_casts_no_shadow(self):
        # Terrain beyond the grid continues at its edge height. A zero fill stood as a wall above
        # Sun-facing ground and cast false shadow strips into tilted scenes (6-21% of a 90 m canvas).
        rows, cols = np.indices((120, 120), dtype=float)
        for az, el in ((43.1, 4.35), (6., 3.28), (326.7, 3.58)):
            toward = np.array([-np.cos(np.radians(az)), np.sin(np.radians(az))])
            h = -.03*(toward[0]*(rows-60)+toward[1]*(cols-60))*.5
            self.assertEqual(float((lit_fraction(h, .5, az, el) < .5).mean()), 0.)
        # An Athena-like tilt of 2 deg toward the Sun: uniformly brighter, never shadowed.
        out = render_relief((96, 96), [43.1], [4.35], plane_slope_rc=(.03, -.02), **QUIET)
        self.assertGreater(out['stack'].min(), 1.)

    def test_rock_has_a_bright_face_and_a_long_thin_shadow(self):
        rock = make_rock(3, (32.3, 40.2), .6, .6, aspect=1.35, yaw_deg=0)
        f = render_relief((64, 64), [90.], [3.5], rocks=[rock], **QUIET)['stack'][0]
        self.assertGreater(f[32, 40], 1.2)
        self.assertGreater(len(np.flatnonzero(f[32, :40] < .9)), 6)   # about 11 px expected

    def test_boulder_fields_meet_their_cover_and_cast_shadows(self):
        from src.hati_core.relief_scenes import boulder_field
        out = render_relief((32, 32), [90.], [3.5], supersample=12, features=[boulder_field(.04, d_min_m=.1, seed=2)], **QUIET)
        info = out['truth']['boulder_fields'][0]
        self.assertAlmostEqual(info['area_fraction'], .04, delta=.002)
        self.assertGreaterEqual(info['diameter_m']['min'], .1)
        self.assertLessEqual(info['diameter_m']['max'], 2.)
        frame = out['stack'][0]
        self.assertLess(frame.min(), .8)            # shadows
        self.assertGreater(frame.max(), 1.05)       # lit rock faces, brighter rock albedo
        with self.assertRaises(ValueError):
            boulder_field(.6)

    def test_injected_rock_factor_is_one_away_from_sites(self):
        factor = rock_factor((64, 64), [(32.3, 32.2, .6)], [90.], [3.5], .9, seed=2, window_px=32)
        self.assertEqual(factor.shape, (1, 64, 64))
        np.testing.assert_allclose(factor[0, :10], 1)
        self.assertLess(factor[0, 32, 20:32].min(), .8)


class TerrainTemplateTests(unittest.TestCase):
    sc = ShadowConfig(radius_px=12, root_support_px=6, supersample=4)

    def test_a_tilted_plane_as_terrain_reproduces_the_planar_template(self):
        from src.hati_core.shadow_likelihood import shadow_template
        rows, cols = np.indices((25, 25), dtype=float); root = (12.3, 12.2); slope = (.01, -.02)
        plane = (slope[0]*(rows-root[0])+slope[1]*(cols-root[1]))*.9
        planar = shadow_template((25, 25), root, [90.], [3.5], .3, .6, self.sc, slope)[0]
        terrain = shadow_template((25, 25), root, [90.], [3.5], .3, .6, self.sc, terrain=plane)[0]
        np.testing.assert_allclose(planar, terrain, atol=.01)

    def test_rising_ground_shortens_the_shadow_and_a_dip_lengthens_it(self):
        from src.hati_core.shadow_likelihood import shadow_template
        cols = np.indices((25, 25), dtype=float)[1]
        def coverage(terrain):
            return shadow_template((25, 25), (12.3, 12.2), [90.], [3.5], .3, .6, self.sc, terrain=terrain)[0].sum()
        flat = coverage(np.zeros((25, 25)))
        self.assertLess(coverage(np.where(cols < 10, .12, 0.)), .8*flat)
        self.assertGreater(coverage(np.where(cols < 10, -.12, 0.)), 1.2*flat)


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

    def test_mound_and_bowl_signs_and_the_sun_check(self):
        mound = self.compare(features=[relief_feature('mound', 6., 2.)])
        bowl = self.compare(features=[relief_feature('bowl', 6., 2.)])
        stripes = self.compare(structured_null=True)
        self.assertGreater(mound['gain_mound'], 5*max(mound['gain_bowl'], .1))
        self.assertGreater(bowl['gain_bowl'], 5*max(bowl['gain_mound'], .1))
        # Relief follows the Sun; stripes that merely change between frames do not.
        self.assertGreater(mound['sun_margin'], 1.)
        self.assertLess(stripes['sun_margin'], .5)

    def test_margins_meet_their_declared_targets(self):
        rng = np.random.default_rng(0)
        def row(truth, rock, mound, bowl, sun, kind=None):
            return dict(truth=truth, kind=kind, status='assessed', gain_rock=rock, gain_mound=mound, gain_bowl=bowl,
                        gain_relief=max(mound, bowl), sun_margin=sun)
        rows = [row('rock', 2+rng.normal(), 1+rng.normal(), 1+rng.normal(), rng.normal(0, .3)) for _ in range(200)]
        rows += [row('relief', 1+rng.normal(), 3+rng.normal(), rng.normal(0, .3), 2+rng.normal(), 'mound') for _ in range(100)]
        rows += [row('relief', 1+rng.normal(), rng.normal(0, .3), 3+rng.normal(), 2+rng.normal(), 'bowl') for _ in range(100)]
        rows += [row('none', *np.abs(rng.normal(0, .1, 3)), rng.normal(0, .1)) for _ in range(200)]
        rows += [row('stripes', 1+rng.normal(0, .3), 1.5+rng.normal(0, .3), 1.5+rng.normal(0, .3), -.5+rng.normal(0, .3))
                 for _ in range(200)]
        margins = calibrate_margins(rows, dict(rock_called_relief=.05, relief_called_rock=.1, blank_called_signal=.1,
                                               stripes_called_relief=.1, sign_error=.1))
        table, signs = confusion(rows, margins), sign_confusion(rows, margins)
        self.assertLessEqual(table['rock']['relief_like'], 10)
        self.assertLessEqual(table['relief']['rock_like'], 20)
        self.assertGreaterEqual(table['none']['none'], 180)
        self.assertLessEqual(table['stripes']['relief_like'], 20)
        self.assertLessEqual(signs['mound']['depression'], 10)
        self.assertLessEqual(signs['bowl']['protrusion'], 10)


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

    def test_a_second_pass_leaves_cast_shadows_out_and_keeps_the_rock(self):
        # Same seed with and without the rock: identical noise and relief, so the
        # difference of the corrected stacks is the rock signal the correction kept.
        features = [relief_feature('ripples', 6., 1.5, seed=2)]
        rock = make_rock(9, (40.3, 72.2), .6, .6, aspect=1.35)
        scene = dict(pixel_m=.9, seed=5, noise=.01, features=features)
        with_rock = render_relief((96, 96), AZ, EL, rocks=[rock], **scene)['stack']
        without = render_relief((96, 96), AZ, EL, **scene)['stack']
        ones = np.ones_like(with_rock, bool)
        one = solve_sfs(with_rock, ones, AZ, EL, .9, grid_px=1, smoothness=1.)
        two = solve_sfs(with_rock, ones, AZ, EL, .9, grid_px=1, smoothness=1., passes=2)
        base = solve_sfs(without, ones, AZ, EL, .9, grid_px=1, smoothness=1., passes=2)
        self.assertEqual(one['shadow_excluded_fraction'], 0.)
        self.assertGreater(two['shadow_excluded_fraction'], 0.)
        self.assertLess(two['shadow_excluded_fraction'], .05)
        near = np.s_[:, 28:54, 50:80]
        signal = with_rock[near]-without[near]
        kept = two['corrected'][near]-base['corrected'][near]
        self.assertGreater(float(np.sum(kept*signal)/np.sum(signal*signal)), .8)

    def test_gauss_newton_photometry_and_recovery(self):
        from src.hati_core.sfs import _sun_vector, lunar_lambert, solve_sfs_nonlinear
        from src.hati_core.relief_scenes import shading
        rng = np.random.default_rng(0)
        p, q = rng.normal(0, .03, 50), rng.normal(0, .03, 50); sun = _sun_vector(40., 3.6); e = 1e-6
        R, dp, dq = lunar_lambert(p, q, sun); lit = R > 0
        np.testing.assert_allclose(dp[lit], ((lunar_lambert(p+e, q, sun)[0]-R)/e)[lit], atol=1e-4)
        np.testing.assert_allclose(dq[lit], ((lunar_lambert(p, q+e, sun)[0]-R)/e)[lit], atol=1e-4)
        h = np.cumsum(np.cumsum(rng.normal(0, .002, (40, 40)), 0), 1); gr, gc = np.gradient(h, .9)
        np.testing.assert_allclose(lunar_lambert(gr, gc, sun)[0], shading(h, .9, 40., 3.6), atol=1e-12)
        out = render_relief((72, 72), AZ, EL, pixel_m=.9, seed=5, noise=.01,
                            features=[relief_feature('ripples', 5., 2.5, seed=2), relief_feature('mound', 10., 3.)])
        solved = solve_sfs_nonlinear(out['stack'], np.ones_like(out['stack'], bool), AZ, EL, .9, grid_px=1, smoothness=1.)
        self.assertGreater(solved['explained_fraction'], .9)
        self.assertTrue(all(g['relative_change'] >= 0 for g in solved['gauss_newton'] if 'relative_change' in g))

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
            for name in ('shape_from_shading.png', 'sfs_height_m.tif', 'sfs_slope_deg.tif', 'injection.json',
                         'subpixel_casters.json', 'subpixel_casters.png'):
                self.assertTrue((out/'stages/T14'/name).exists(), name)
            casters = sfs['subpixel_casters']
            self.assertEqual(casters['relief_check'], 'T13 calibrated margins on the relief-corrected stack')
            self.assertEqual(set(casters['injected_rock_sizing']), {'0.3 m', '0.6 m', '1.2 m'})
            injected = json.loads((out/'stages/T14/injection.json').read_text())
            self.assertTrue(all('sized_state' in r for r in injected))
            nulls = run('T16', t16)
            self.assertIn('relief_corrected_sigma', nulls['render_noise_source'])
            self.assertIn('ripples_2deg', {s['kind'] for s in nulls['summaries']})


if __name__ == '__main__':
    unittest.main()
