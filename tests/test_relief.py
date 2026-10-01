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

    def test_flat_planting_is_unchanged(self):
        from src.hati_core.rock_scenes import make_rock
        old = np.ones((2, 64, 64))
        rock = make_rock(2, (32.3-16, 32.2-16), .6, .6, aspect=1.35)
        old[:, 16:48, 16:48] = render_relief((32, 32), [90., 30.], [3.5, 4.], pixel_m=.9, seed=2, noise=0., rocks=[rock],
                                             supersample=4, texture=0., stain=0., frame_plane=0.)['stack']
        new = rock_factor((64, 64), [(32.3, 32.2, .6)], [90., 30.], [3.5, 4.], .9, seed=2, window_px=32)
        np.testing.assert_array_equal(new, old)
        level = rock_factor((64, 64), [(32.3, 32.2, .6)], [90., 30.], [3.5, 4.], .9, seed=2, window_px=32, slopes=[(0., 0.)])
        np.testing.assert_allclose(level, old, atol=1e-9)

    @staticmethod
    def _shadow_px(factor, row=32, root=32):
        """Distance from the rock to the far end of its shadow along its row (Sun from the east), in pixels.

        The run of darkened pixels is the one that reaches within four pixels of the
        rock; the rock's own soft edge may leave the pixel next to it undarkened.
        """
        dark = factor[0, row, :root] < .9
        near = np.flatnonzero(dark[root-4:])
        if not len(near):
            return 0
        start = root-4+int(near[-1])
        while start > 0 and dark[start-1]:
            start -= 1
        return root-start

    def test_tilted_planting_follows_the_ground_under_the_shadow(self):
        site = [(64.3, 64.2, .6)]
        args = ((128, 128), site, [90.], [3.5], .9)
        flat = rock_factor(*args, seed=2, window_px='auto')
        # Sun from the east, shadow to the west: ground rising westward is a negative slope along columns.
        rising = rock_factor(*args, seed=2, window_px='auto', slopes=[(0., -.02)])
        falling = rock_factor(*args, seed=2, window_px='auto', slopes=[(0., .02)])
        lengths = [self._shadow_px(f, 64, 64) for f in (rising, flat, falling)]
        self.assertLess(lengths[0], lengths[1]); self.assertLess(lengths[1], lengths[2])
        # On the tilted plane only the rock changes the image: far from it the factor is one.
        np.testing.assert_allclose(rising[0, :20], 1, atol=1e-6)
        np.testing.assert_allclose(rising[0, 64, 100:], 1, atol=1e-6)

    def test_auto_window_keeps_the_whole_shadow(self):
        # A 1.2 m rock at 3.5 degrees casts about 22 px of shadow; a 32 px window cuts it at 16.
        args = ((128, 128), [(64.3, 64.2, 1.2)], [90.], [3.5], .9)
        cut = rock_factor(*args, seed=2, window_px=32)
        full = rock_factor(*args, seed=2, window_px='auto')
        self.assertLessEqual(self._shadow_px(cut, 64, 64), 16)
        self.assertGreater(self._shadow_px(full, 64, 64), 18)
        # Windows are clipped at the image edge instead of refusing the site.
        edge = rock_factor((64, 64), [(10.3, 10.2, 1.2)], [90.], [3.5], .9, seed=2, window_px='auto')
        self.assertEqual(edge.shape, (1, 64, 64))

    def test_auto_window_follows_ground_falling_along_the_shadow(self):
        # 1.2 m at 3.5 degrees on ground falling 0.02 along the shadow: about 32 px of shadow, past the
        # 28 px half-window that the level-ground length would give.
        args = ((160, 160), [(80.3, 80.2, 1.2)], [90.], [3.5], .9)
        info = []
        falling = rock_factor(*args, seed=2, window_px='auto', slopes=[(0., .02)], info=info)
        expected = 1.2/(np.tan(np.radians(3.5))-.02)/.9
        self.assertGreater(self._shadow_px(falling, 80, 80), expected-3)
        self.assertFalse(info[0]['shadow_clipped'])
        self.assertGreater(info[0]['window_px']//2, expected)

    def test_shadow_reach_follows_the_generator_geometry(self):
        from src.hati_core.sfs import shadow_reach_px
        # Sun from the east: shadows run west, along -columns; the lower solar limb sets the reach.
        tan_e = np.tan(np.radians(3.5-.266))
        self.assertAlmostEqual(shadow_reach_px(.6, (0., 0.), [90.], [3.5], .9), .6/tan_e/.9, places=6)
        self.assertAlmostEqual(shadow_reach_px(.6, (0., .01), [90.], [3.5], .9), .6/(tan_e-.01)/.9, places=6)
        self.assertAlmostEqual(shadow_reach_px(.6, (0., -.01), [90.], [3.5], .9), .6/(tan_e+.01)/.9, places=6)
        # A plane facing away from the Sun casts no shadow in that frame; the longest of the others counts.
        self.assertEqual(shadow_reach_px(.6, (0., .5), [90.], [3.5], .9), 0.)
        self.assertAlmostEqual(shadow_reach_px(.6, (0., 0.), [90., 0.], [3.5, 2.], .9),
                               .6/np.tan(np.radians(2.-.266))/.9, places=6)

    def test_capped_reach_is_reported_as_clipped(self):
        info = []
        rock_factor((96, 96), [(48.3, 48.2, 1.2)], [90.], [1.], .9, seed=2, window_px='auto', max_reach_px=20, info=info)
        self.assertTrue(info[0]['shadow_clipped'])
        self.assertEqual(info[0]['window_px'], 52)


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
    def test_solver_progress_is_display_only(self):
        stack = render_relief((40, 40), AZ, EL, pixel_m=.9, seed=4, noise=.01,
                              features=[relief_feature('mound', 8., 3., centre_px=(20, 20))])['stack']
        valid = np.ones_like(stack, bool)
        plain = solve_sfs(stack, valid, AZ, EL, .9, grid_px=1, iterations=300, passes=2)
        seen = []
        watched = solve_sfs(stack, valid, AZ, EL, .9, grid_px=1, iterations=300, passes=2, progress=seen.append)
        for key in ('height_m', 'corrected', 'predicted_ratio', 'planes'):
            np.testing.assert_array_equal(plain[key], watched[key], err_msg=key)
        self.assertEqual(plain['lsqr_iterations'], watched['lsqr_iterations'])
        self.assertEqual({s['pass_index'] for s in seen}, {1, 2})
        self.assertLessEqual(max(s['iteration'] for s in seen), 302)

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


class PlantedSummaryTests(unittest.TestCase):
    """Height groups, calibration lookup and the merging of real detections into candidate objects."""

    def test_groups_follow_the_planting_mode(self):
        from relief_experiments import _group_of, _height_groups
        fixed = _height_groups(dict(planted_geometry='fixed'), [.3, .6, 1.2])
        self.assertEqual([g[0] for g in fixed], ['0.3 m', '0.6 m', '1.2 m'])
        self.assertEqual(_group_of(.6, fixed), '0.6 m'); self.assertIsNone(_group_of(.5, fixed))
        bins = _height_groups(dict(planted_height_bins_m=[.15, .3, .6]), [.3])
        self.assertEqual([g[0] for g in bins], ['0.15 to 0.3 m', '0.3 to 0.6 m'])
        self.assertEqual(_group_of(.3, bins), '0.3 to 0.6 m')        # lower edges belong to their bin
        self.assertEqual(_group_of(.6, bins), '0.3 to 0.6 m')        # the top edge closes the last bin
        self.assertIsNone(_group_of(.1, bins)); self.assertIsNone(_group_of(.7, bins))

    def test_calibration_uses_the_height_a_candidate_reached(self):
        from relief_experiments import _calibration_for, _height_groups
        recovery = {g: dict(corrected_measured=dict(recovered=i, quiet_sites=10)) for i, g in
                    enumerate(['0.15 to 0.3 m', '0.3 to 0.6 m', '0.3 m', '0.6 m', '1.2 m'])}
        bins = _height_groups({}, [])
        self.assertEqual(_calibration_for(.4, bins, recovery), dict(group='0.3 to 0.6 m', found=1, of=10))
        self.assertIsNone(_calibration_for(.1, bins, recovery))                       # below the planted range
        self.assertEqual(_calibration_for(3., bins, {'1.2 to 2 m': dict(corrected_measured=dict(recovered=7, quiet_sites=9))})['group'],
                         '1.2 to 2 m')                                                 # taller: the top bin
        fixed = _height_groups(dict(planted_geometry='fixed'), [.3, .6, 1.2])
        self.assertEqual(_calibration_for(.9, fixed, recovery)['group'], '0.6 m')      # the tallest planted not above it
        self.assertIsNone(_calibration_for(.2, fixed, recovery))
        self.assertIsNone(_calibration_for(None, bins, recovery))

    def test_planting_and_sizing_settings_are_checked(self):
        from saturation_experiments import validate_config
        base = json.loads((ROOT/'configs/saturation_campaign.json').read_text())
        validate_config(dict(base))
        validate_config(dict(base, sfs_sizing_adaptive=dict(context_guard=True, scale_factors=[1, 2, 4, 8])))
        validate_config(dict(base, sfs_sizing_adaptive=dict(caster_profile='dome'), estimate_calibration='none'))
        for bad in (dict(planted_geometry='random'), dict(planted_height_bins_m=[.3, .6, 1.2]),   # bins miss 0.15 to 0.3 m
                    dict(planted_height_bins_m=[.6, .3]), dict(planted_rock_prior=dict(burial=[0, 1.5])),
                    dict(sfs_sizing_adaptive=dict(scale_factors=[1, 3])), dict(sfs_sizing_adaptive=dict(guard=True)),
                    dict(measurable_detection_target=1.2), dict(sfs_sizing_images='sideways'), dict(bound_calibration='median'),
                    dict(estimate_calibration='mean'), dict(sfs_sizing_adaptive=dict(caster_profile='cone')),
                    dict(bound_calibration_coverage=1.), dict(estimate_calibration_coverage=0.), dict(sfs_sizing_subgrid='yes')):
            with self.assertRaises((ValueError, TypeError), msg=str(bad)):
                validate_config(dict(base, **bad))

    def test_conformal_margin_makes_bounds_hold(self):
        from relief_experiments import _conformal_margin, _corrected_bound
        rng = np.random.default_rng(0)
        truth = rng.uniform(.3, 2., 2000)
        bound = truth+rng.normal(.08, .06, truth.size)                       # biased high by about 8 cm
        cal, test = slice(0, 1000), slice(1000, None)
        m = _conformal_margin(bound[cal], truth[cal], .9)
        held = [_corrected_bound(b, m) <= t for b, t in zip(bound[test], truth[test])]
        self.assertTrue(.88 <= np.mean(held) <= .93)                         # about 90% out of sample
        f = _conformal_margin(bound[cal], truth[cal], .9, kind='ratio')
        self.assertGreater(f, 1.)
        self.assertGreaterEqual(np.mean([_corrected_bound(b, f, 'ratio') <= t for b, t in zip(bound[test], truth[test])]), .88)
        # Conservative bounds need no correction, and too few rocks cannot promise 90%.
        self.assertEqual(_conformal_margin(truth[:50]-.2, truth[:50], .9), 0.)
        self.assertIsNone(_conformal_margin(bound[:8], truth[:8], .9))       # ceil(9 * 0.9) = 9 > 8
        self.assertIsNotNone(_conformal_margin(bound[:9], truth[:9], .9))
        self.assertIsNone(_corrected_bound(None, m))

    def test_estimate_intervals_hold_and_follow_the_bias(self):
        from relief_experiments import _estimate_factors, _estimate_interval
        rng = np.random.default_rng(1)
        truth = rng.uniform(.3, 2., 2000)
        estimate = truth*np.exp(rng.normal(-.1, .07, truth.size))           # reads about 10% short
        cal, test = slice(0, 1000), slice(1000, None)
        low, high = _estimate_factors(estimate[cal], truth[cal], .9)
        self.assertGreater(np.sqrt(low*high), 1.08)                          # centred about 10% above a short reading
        inside = [lo <= t <= hi for (lo, hi), t in zip((_estimate_interval(e, (low, high)) for e in estimate[test]), truth[test])]
        self.assertTrue(.88 <= np.mean(inside) <= .93)                       # about 90% out of sample
        # Each tail needs enough rocks: floor(20 * 0.05) = 1 is the smallest usable rank.
        self.assertIsNone(_estimate_factors(estimate[:18], truth[:18], .9))
        self.assertIsNotNone(_estimate_factors(estimate[:19], truth[:19], .9))
        self.assertIsNone(_estimate_interval(None, (low, high)))
        self.assertIsNone(_estimate_interval(1., None))

    def test_caster_rows_read_heights_between_grid_steps(self):
        from types import SimpleNamespace
        from relief_experiments import _caster_row
        # One assessed pass whose profile peaks at 0.6 m but leans towards 0.7 m; the shadow end is inside the window.
        surface = [[.5, .6, 8.], [.6, .6, 10.], [.7, .6, 9.8], [.6, 1.2, 9.]]
        fit = dict(status='assessed', scale=2, support_px=12., frames=[0, 1, 2], best=dict(height_m=.6, width_m=.6, score=10.,
                   root_offset=[0., 0.]), height_range_m=[.6, .7], lowest_compatible=dict(root_offset=[0., 0.]),
                   lowest_compatible_censored=False, surface=surface)
        record = dict(centre=[40, 40], status='context_supported_unvalidated', history=[fit], final=fit)
        def row(subgrid):
            ex = SimpleNamespace(sc=SimpleNamespace(pixel_m=.9), data=dict(elevations=[3.3, 3.5, 4.]),
                                 cfg=dict(sfs_sizing_subgrid=subgrid))
            return _caster_row(ex, AdaptiveConfig(), record, .3)
        grid, sub = row(False), row(True)
        self.assertEqual((grid['height_m'], grid['height_lower_bound_m']), (.6, .6))
        self.assertTrue(.6 < sub['height_m'] < .65)                  # the vertex leans towards 0.7 m
        self.assertTrue(.5 < sub['height_lower_bound_m'] <= .6)      # never above the grid's bound
        self.assertEqual(sub['compatible_range_m'], [.6, .7])        # the grid range is still reported

    def test_exact_intervals(self):
        from relief_experiments import _exact_interval
        self.assertAlmostEqual(_exact_interval(5, 5)[0], .025**(1/5), places=6)        # 5 of 5 found: at least 48%
        self.assertAlmostEqual(_exact_interval(45, 45)[0], .025**(1/45), places=6)      # 45 of 45: at least 92%
        self.assertAlmostEqual(_exact_interval(0, 10)[1], 1-.025**(1/10), places=6)
        lo, hi = _exact_interval(7, 10)
        self.assertTrue(lo < .7 < hi)
        self.assertIsNone(_exact_interval(0, 0))

    def test_measurable_from_needs_every_taller_group(self):
        from relief_experiments import _height_groups, _measurable
        def rock(h, found, bound):
            return dict(height_m=h, recovered_corrected_measured=found, sized_height_lower_bound_m=bound, sized_height_m=None)
        injection = ([rock(1.5, True, 1.2)]*45+[rock(.9, True, .8)]*45                          # tall rocks: all found, bounds hold
                     +[rock(.45, True, .4)]*30+[rock(.45, True, 1.1)]*15                         # 0.3 to 0.6 m: a third overshoot
                     +[rock(.2, False, None)]*20)                                                # small: never found
        m = _measurable(_height_groups({}, []), injection)
        self.assertEqual(m['measurable_from_m'], .6)
        g = m['groups']
        self.assertTrue(g['1.2 to 2 m']['established'] and g['0.6 to 1.2 m']['established'])
        self.assertFalse(g['0.3 to 0.6 m']['established'])
        self.assertEqual((g['0.3 to 0.6 m']['bound_holds'], g['0.3 to 0.6 m']['bounded']), (30, 45))
        self.assertEqual((g['0.15 to 0.3 m']['found'], g['0.15 to 0.3 m']['quiet_sites']), (0, 20))
        # Too few rocks cannot establish anything, however well they do.
        self.assertIsNone(_measurable(_height_groups({}, []), [rock(1.5, True, 1.2)]*5)['measurable_from_m'])

    def test_busy_sites_do_not_count_against_sizing(self):
        from relief_experiments import _height_groups, _measurable
        # Where the background already warned, the sizing measures that feature too: counted apart, not judged.
        quiet = [dict(height_m=1.5, recovered_corrected_measured=True, sized_height_lower_bound_m=1.4, sized_height_m=None)]*45
        busy = [dict(height_m=1.5, recovered_corrected_measured=None, sized_height_lower_bound_m=2.4, sized_height_m=None)]*5
        g = _measurable(_height_groups({}, []), quiet+busy)['groups']['1.2 to 2 m']
        self.assertEqual((g['planted'], g['quiet_sites'], g['busy_sites']), (50, 45, 5))
        self.assertEqual((g['bound_holds'], g['bounded'], g['busy_fitted_bound_holds']), (45, 45, [0, 5]))
        self.assertTrue(g['established'])

    def test_touching_cells_merge_into_one_candidate(self):
        from relief_experiments import _group_detections, _height_groups
        def cell(r, c, bound, label='rock_like', est=None, score=10.):
            return dict(row_px=r, col_px=c, height_lower_bound_m=bound, height_m=est, relief_check=label, score=score,
                        distance_to_touchdown_m=float(np.hypot(r, c)), exceeds_clearance=bound is not None and bound >= .3)
        casters = [cell(40, 40, .4, 'ambiguous', est=.5), cell(44, 44, .9, score=30., est=1.), cell(48, 40, None, 'ambiguous'),
                   cell(100, 100, .2, 'ambiguous')]
        casters[0]['height_interval_m'], casters[1]['height_interval_m'] = [.45, .6], [.9, 1.2]
        recovery = {'0.6 to 1.2 m': dict(corrected_measured=dict(recovered=5, quiet_sites=6))}
        objects = _group_detections(casters, 4, _height_groups({}, []), recovery)
        self.assertEqual([o['cells'] for o in objects], [3, 1])
        big = objects[0]
        self.assertEqual((big['height_lower_bound_m'], big['label'], big['peak_row_px']), (.9, 'rock_like', 44))
        self.assertEqual((big['height_m'], big['height_interval_m']), (1., [.9, 1.2]))     # the interval of the tallest estimate
        self.assertIsNone(objects[1]['height_interval_m'])
        self.assertEqual(big['calibration'], dict(group='0.6 to 1.2 m', found=5, of=6))
        self.assertTrue(big['exceeds_clearance'])
        self.assertEqual((objects[1]['label'], objects[1]['object']), ('ambiguous', 1))
        self.assertEqual(_group_detections([], 4, [], {}), [])


class ReliefCampaignTests(unittest.TestCase):
    def test_t12_to_t16_on_a_relief_bundle_without_external_calls(self):
        from saturation_experiments import Experiment, t1, t12, t13, t14, t16
        with tempfile.TemporaryDirectory() as tmp:
            # 112 px leaves room for both injection rounds at the spacing a 1.2 m rock's full shadow needs.
            out = Path(tmp); bundle = out/'input.zip'; relief_bundle(bundle, size=112)
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
                       relief_touchdown_radius_px=4, sfs_injection_sites=2, sfs_injection_spacing_px=20,
                       sfs_injection_rounds=2, planted_rock_prior=dict(height_m=[.2, 1.2]),
                       planted_height_bins_m=[.2, .6, 1.2])
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
            # Planted rocks are drawn from the lunar shape population and summarised by height bin.
            self.assertEqual(set(casters['injected_rock_sizing']), {'0.2 to 0.6 m', '0.6 to 1.2 m'})
            self.assertEqual(sfs['planted_population']['geometry'], 'population')
            self.assertEqual(len(sfs['planted_population']['sources']), 2)
            injected = json.loads((out/'stages/T14/injection.json').read_text())
            self.assertTrue(all('sized_state' in r for r in injected))
            self.assertTrue(all(.2 <= r['height_m'] <= 1.2 and r['shape'] == 'procedural' and 0 < r['height_over_diameter'] <= 1
                                for r in injected))
            self.assertEqual(len({round(r['height_m'], 6) for r in injected}), len(injected))   # no two alike
            # The real rocks it found: touching cells merged, each with this run's calibration for its height.
            real = json.loads((out/'stages/T14/real_rocks.json').read_text())
            self.assertEqual(sfs['real_rocks']['objects'], len(real))
            self.assertEqual(sum(o['cells'] for o in real), sfs['real_rocks']['cells'])
            self.assertIn('pixel_m', sfs['real_rocks'])
            # Two rounds on shifted grids; every site scores the rock alone before and after correction.
            self.assertEqual([r['round'] for r in sfs['injection_rounds']], [0, 1])
            self.assertEqual(sfs['injected_sites'], sum(r['placed'] for r in sfs['injection_rounds']))
            self.assertEqual({r['round'] for r in injected}, {0, 1})
            kept = [r['rock_signal_kept'] for r in injected if r['rock_signal_kept'] is not None]
            self.assertTrue(kept and all(0 <= v < 2 for v in kept))
            self.assertTrue(all(r['score_rock_only_before'] is not None for r in injected))
            self.assertIn('small_rock_signal_mostly_absorbed', sfs['relief_absorption_verdict'])
            self.assertEqual(set(sfs['relief_absorption']), {'0.2 to 0.6 m', '0.6 to 1.2 m'})
            nulls = run('T16', t16)
            self.assertIn('relief_corrected_sigma', nulls['render_noise_source'])
            self.assertIn('ripples_2deg', {s['kind'] for s in nulls['summaries']})


class OriginalImageSizingTests(unittest.TestCase):
    def test_t14_sizes_on_the_original_images(self):
        from saturation_experiments import Experiment, t1, t14
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); bundle = out/'input.zip'; relief_bundle(bundle, size=112)
            cfg = json.loads((ROOT/'configs/saturation_campaign.json').read_text())
            cfg.update(adaptive=asdict(AdaptiveConfig(scale_factors=(1, 2), heights_m=(.2, .4, .6), widths_m=(.3, .6, .9),
                                                      max_cells=2, workers=1)),
                       noise_scale=dict(patch_px=24, max_slope=.05), relief_supersample=2, sfs_injection_sites=2,
                       sfs_injection_spacing_px=20, planted_rock_prior=dict(height_m=[.2, 1.2]), planted_height_bins_m=[.2, .6, 1.2],
                       sfs_sizing_images='original', sfs_sizing_adaptive=dict(context_guard=True, pad_edges=True))
            config = out/'config.json'; config.write_text(json.dumps(cfg))
            def run(stage, fn):
                args = Namespace(stage=stage, bundle=bundle, config=config, output=out/'stages'/stage,
                                 campaign=out, dem=None, thermal=None, held_out=None, rock_catalog=None)
                with patch('subprocess.run', side_effect=AssertionError('external ingestion forbidden')):
                    return fn(Experiment(args))
            run('T1', t1)
            sfs = run('T14', t14)
            self.assertEqual(sfs['status'], 'PARTIAL')
            self.assertGreater(sfs['injected_sites'], 0)
            self.assertIn('measurable', sfs)
            self.assertEqual(sfs['bound_calibration']['kind'], 'offset')
            self.assertEqual(sfs['estimate_calibration']['kind'], 'conformal')
            self.assertIn('estimate_interval_holds', next(iter(sfs['measurable']['groups'].values())))


if __name__ == '__main__':
    unittest.main()
