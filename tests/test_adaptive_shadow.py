"""Adaptive context, missing-data, selection and prediction leakage checks."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.hati_core.adaptive_shadow import (AdaptiveConfig, fit_patch, refine_cell,
    refine_regions, plan_regions, held_out_prediction, cell_table)
from src.hati_core.shadow_likelihood import ShadowConfig, shadow_template, endpoint_support, RegistrationProjector
from src.hati_core.regional_shadow import RegionalConfig


AZ = np.array([0., 60., 125., 195., 265.])
EL = np.array([4., 5., 6., 5., 4.])
SC = ShadowConfig(radius_px=6, root_support_px=3, supersample=2, solar_radius_deg=0.)
RC = RegionalConfig(cell_px=1, heights_m=(.2, .6), widths_m=(.4, .8))
AC = AdaptiveConfig(heights_m=(.2, .4, .6), widths_m=(.3, .6, .9), fine_step_m=.1)


def matched(shape=(49, 49), height=.4, width=.6, seed=3):
    # Matched renderer tests algebra/selection, never scientific transfer.
    root = (np.array(shape)-1)/2
    t = shadow_template(shape, root, AZ, EL, height, width, SC)[0]
    return 1-.7*t+np.random.default_rng(seed).normal(0, .005, t.shape)


class AdaptiveTests(unittest.TestCase):
    def test_endpoint_reasons_distinguish_crop_support_and_missing_pixels(self):
        cfg = replace(SC, radius_px=12, root_support_px=10)
        args = ((25, 25), (12, 12), [0.], [4.], .3, cfg)
        result = endpoint_support(*args)[0]
        self.assertEqual(result['reason'], 'supported_prediction')
        mask = np.ones((1, 25, 25), bool)
        mask[0, round(result['longest_endpoint_row_px']), round(result['longest_endpoint_col_px'])] = False
        self.assertEqual(endpoint_support(*args, valid=mask)[0]['reason'], 'endpoint_missing')
        self.assertEqual(endpoint_support(*args[:5], replace(cfg, root_support_px=3))[0]['reason'], 'support_cutoff')
        self.assertEqual(endpoint_support((25, 25), (12, 12), [0.], [4.], 2., cfg)[0]['reason'], 'patch_cutoff')

    def test_expansion_reads_new_pixels_and_preserves_frame_set(self):
        stack = matched((65, 65)); observed = []; original = fit_patch
        def recording(data, *args, **kw):
            observed.append((data.copy(), kw.get('selected_frames')))
            return original(data, *args, **kw)
        with patch('src.hati_core.adaptive_shadow.fit_patch', side_effect=recording):
            result = refine_cell(stack, np.ones_like(stack), AZ, EL, .005, SC, RC, AC,
                                 (32, 32), np.zeros((65, 65)), np.zeros((65, 65)))
        self.assertGreaterEqual(len(observed), 2)
        self.assertEqual(observed[0][0].shape, (5, 13, 13))
        self.assertEqual(observed[1][0].shape, (5, 25, 25))
        np.testing.assert_array_equal(observed[1][0], stack[:, 20:45, 20:45])
        self.assertEqual(observed[1][1], result['history'][0]['frames'])
        self.assertEqual([p['scale'] for p in result['history']], [1, 2, 4])

    def test_failed_expansion_does_not_fall_back_to_small_window_dimensions(self):
        stack = matched((17, 17)); slopes = np.zeros((17, 17))
        result = refine_cell(stack, np.ones_like(stack), AZ, EL, .005, SC, RC, AC, (8, 8), slopes, slopes)
        self.assertEqual(result['status'], 'unresolved_image_edge')
        self.assertIsNone(result['final'])
        self.assertEqual(result['history'][0]['status'], 'assessed')

    def test_larger_window_cannot_silently_drop_a_contradictory_frame(self):
        stack = matched((65, 65)); vis = np.ones_like(stack)
        y, x = np.indices((65, 65)); vis[4, np.hypot(y-32, x-32) > 4] = 0
        result = refine_cell(stack, vis, AZ, EL, .005, SC, RC, AC, (32, 32),
                             np.zeros((65, 65)), np.zeros((65, 65)))
        self.assertEqual(result['history'][0]['frames'], list(range(5)))
        self.assertEqual(result['status'], 'unresolved_insufficient_frames')
        self.assertIsNone(result['final'])

    def test_terrain_variation_blocks_planar_extrapolation(self):
        stack = matched((65, 65)); slopes = np.zeros((65, 65)); slopes[32, 37] = .3
        result = refine_cell(stack, np.ones_like(stack), AZ, EL, .005, SC, RC, AC,
                             (32, 32), slopes, np.zeros_like(slopes))
        self.assertEqual(result['status'], 'unresolved_terrain')
        self.assertEqual(result['history'][-1]['status'], 'terrain_plane_limit')

    def test_dimension_surface_includes_ten_centimetre_hypotheses(self):
        sc = replace(SC, radius_px=12, root_support_px=10)
        stack = matched((25, 25), height=.3)
        fit = fit_patch(stack, np.ones_like(stack), AZ, EL, .005, sc, RC, AC)
        self.assertEqual(fit['status'], 'assessed')
        self.assertIn(.3, [v[0] for v in fit['surface']])
        self.assertLessEqual(abs(fit['best']['height_m']-.3), .1)
        self.assertIn('no calibrated', fit['uncertainty'])

    def test_quadratic_null_projects_data_and_template_through_same_operator(self):
        y, x = np.indices((17, 17)); texture = np.sin(y/2)+np.cos(x/3)
        stack = np.array([texture+i*.01*y*x+i*.02*x*x for i in range(4)])
        projector = RegistrationProjector(np.ones((17, 17), bool), np.ones(4), texture, .2, spatial_degree=2)
        self.assertLess(np.linalg.norm(projector.apply(stack)), 1e-9)

    def test_withheld_values_cannot_change_training_parameters_or_profile(self):
        sc = replace(SC, radius_px=12, root_support_px=10)
        a = matched((25, 25)); b = a.copy(); b[-1] += np.random.default_rng(14).normal(0, 3, (25, 25))
        left = held_out_prediction(a, np.ones_like(a), AZ, EL, .005, sc, RC, AC)
        right = held_out_prediction(b, np.ones_like(b), AZ, EL, .005, sc, RC, AC)
        self.assertEqual(left['training'], right['training'])
        self.assertNotEqual(left['error_per_pixel'], right['error_per_pixel'])
        self.assertGreater(left['correct_advantage_over_wrong'], 0)

    def test_region_fraction_and_isolated_candidate_both_trigger(self):
        shape = (24, 24); baseline = {k: np.zeros(shape) for k in ('status', 'score', 'endpoint_censored')}
        rc = replace(RC, cell_px=4, tile_px=8)
        table = cell_table(shape, SC, rc)
        for r0, r1, c0, c1, *_ in table:
            baseline['status'][r0:r1, c0:c1] = 1
        baseline['score'][14:18, 14:18] = 10
        baseline['endpoint_censored'][6:10, 6:10] = 1
        plan = plan_regions(baseline, SC, rc, AC)
        self.assertTrue(any(q['reasons'] == ['baseline_warning'] for q in plan['queue']))
        self.assertTrue(any(q['reasons'] == ['regional_cutoff_fraction'] for q in plan['queue']))

    def test_budget_is_explicit_and_unknown_cells_have_no_dimensions(self):
        stack = matched((25, 25)); shape = (25, 25)
        base = dict(status=np.ones(shape), score=np.full(shape, 12.), endpoint_censored=np.ones(shape))
        records = []
        result = refine_regions(stack, np.ones_like(stack), AZ, EL, .005, SC, RC,
            replace(AC, max_cells=1), base, np.zeros(shape), np.zeros(shape), on_record=records.append)
        self.assertEqual(result['processed'], 1)
        self.assertGreater(result['unprocessed'], 0)
        self.assertTrue(np.isnan(result['height_m'][result['status'] == 1]).all())
        self.assertEqual(len(records), 1)

    def test_invalid_configuration_rejected(self):
        for kw in ({'fine_step_m': 0}, {'scale_factors': (1, 3)}, {'max_cells': -1}, {'heights_m': (.6, .3)}):
            with self.assertRaises(ValueError):
                AdaptiveConfig(**kw)


class SlopeSearchTests(unittest.TestCase):
    # Athena-like sweep: every shadow points roughly the same way, so a tilt the DEM missed biases height.
    AZ = np.array([43.1, 43.5, 69., 73.7, 326.7, 354.6, 6., 24.9])
    EL = np.array([4.35, 3.59, 3.47, 3.38, 3.58, 3.63, 3.28, 3.69])
    SC = ShadowConfig(radius_px=12, root_support_px=9, supersample=2, solar_radius_deg=0.)
    AC = AdaptiveConfig(heights_m=(.1, .3, .5), widths_m=(.3, .6, .9), fine_step_m=.1)

    def test_the_fitted_plane_recovers_a_tilt_the_dem_missed(self):
        # Matched renderer: algebra and selection only. The ground falls 2 degrees along +rows; the DEM says flat.
        tilt = (-np.tan(np.radians(2.)), 0.)
        truth = shadow_template((25, 25), (12., 12.), self.AZ, self.EL, .3, .6, self.SC, tilt)[0]
        patch = 1-.7*truth+np.random.default_rng(5).normal(0, .005, truth.shape)
        visible = np.ones_like(patch)
        fixed = fit_patch(patch, visible, self.AZ, self.EL, .005, self.SC, RC, self.AC)
        fitted = fit_patch(patch, visible, self.AZ, self.EL, .005, self.SC, RC, replace(self.AC, slope_search_deg=(-2., 0., 2.)))
        self.assertEqual(fixed['receiving_surface'], 'plane')
        self.assertEqual(fitted['receiving_surface'], 'fitted_plane')
        np.testing.assert_allclose(fitted['receiving_slope_rc'], tilt, atol=1e-12)
        self.assertAlmostEqual(fitted['best']['height_m'], .3, places=9)
        self.assertGreater(fixed['best']['height_m'], .3)                 # downhill shadows read as a taller rock
        self.assertGreater(fitted['best']['improvement'], fixed['best']['improvement'])

    def test_configurations_without_a_search_keep_their_hash(self):
        import hashlib
        import json
        from dataclasses import asdict
        cfg = AdaptiveConfig()
        # The hash of the fields that existed before the opt-in options (slope search, context guard, edge
        # padding, caster profile): each is left out when off, so older configurations keep their hashes and
        # cached results.
        payload = {k: v for k, v in asdict(cfg).items()
                   if k not in ('slope_search_deg', 'context_guard', 'pad_edges', 'caster_profile')}
        self.assertEqual(cfg.hash(), 'bbc58da3a19b2025')
        self.assertEqual(cfg.hash(), hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16])
        self.assertNotEqual(cfg.hash(), replace(cfg, slope_search_deg=(-1., 0., 1.)).hash())
        with self.assertRaises(ValueError):
            AdaptiveConfig(slope_search_deg=(-1., 1.))        # the DEM slope itself must stay a candidate


class EdgePaddingTests(unittest.TestCase):
    """Windows may run past the image edge as missing data instead of stopping the cell."""

    def test_window_fills_outside_pixels(self):
        from src.hati_core.adaptive_shadow import _window
        a = np.arange(25.).reshape(5, 5)
        w = _window(a, 0, 4, 1, np.nan)
        np.testing.assert_array_equal(w[1:, :2], a[:2, 3:])
        self.assertTrue(np.isnan(w[0]).all() and np.isnan(w[:, 2]).all())
        stack = np.stack([a, a+1])
        np.testing.assert_array_equal(_window(stack, 2, 2, 2, np.nan), stack)
        self.assertEqual(_window(np.ones((5, 5), bool), 0, 0, 1, False).sum(), 4)

    @staticmethod
    def scene(root, shape=(49, 49), height=.4):
        t = shadow_template(shape, np.asarray(root, float), AZ, EL, height, .6, SC)[0]
        return 1-.7*t+np.random.default_rng(5).normal(0, .005, t.shape)

    def summary(self, record):
        return [(h['scale'], h['status'], h.get('best', {}).get('height_m'), h.get('height_range_m')) for h in record['history']]

    def test_padding_changes_nothing_inside(self):
        image = self.scene((24, 24)); vis = np.ones_like(image); zero = np.zeros(image.shape[1:])
        cfg = replace(AC, scale_factors=(1, 2))
        plain = refine_cell(image, vis, AZ, EL, .005, SC, RC, cfg, (24, 24), zero, zero)
        padded = refine_cell(image, vis, AZ, EL, .005, SC, RC, replace(cfg, pad_edges=True), (24, 24), zero, zero)
        self.assertEqual(self.summary(plain), self.summary(padded))
        self.assertEqual(plain['status'], padded['status'])

    def test_padding_reaches_past_the_edge(self):
        # A rock 8 px from the top: the scale-2 window (radius 12) runs past the edge, its support (radius 6) does not.
        image = self.scene((8, 24)); vis = np.ones_like(image); zero = np.zeros(image.shape[1:])
        cfg = replace(AC, scale_factors=(1, 2))
        plain = refine_cell(image, vis, AZ, EL, .005, SC, RC, cfg, (8, 24), zero, zero)
        self.assertEqual(plain['history'][1]['status'], 'image_edge')
        padded = refine_cell(image, vis, AZ, EL, .005, SC, RC, replace(cfg, pad_edges=True), (8, 24), zero, zero)
        self.assertEqual(padded['history'][1]['status'], 'assessed')
        self.assertEqual(padded['history'][1]['outside_support_fraction'], 0.)
        self.assertNotEqual(padded['status'], 'unresolved_image_edge')
        # A centre so close to the edge that most of the support is outside still fails, through the fit's own minimum.
        edge = refine_cell(image, vis, AZ, EL, .005, SC, RC, replace(cfg, pad_edges=True), (1, 24), zero, zero)
        self.assertTrue(all(h['status'] != 'assessed' or h['scale'] == 1 for h in edge['history']))


class ContextGuardTests(unittest.TestCase):
    """A larger window that contradicts where a smaller one saw the shadow end is not believed."""

    @staticmethod
    def scripted(by_scale):
        def fake(patch, visibility, azimuths, elevations, sigma, sc, rc, cfg, slopes=(0., 0.), *, selected_frames=None,
                 display=False, terrain=None):
            scale = sc.radius_px//SC.radius_px
            h, lo, hi, seen = by_scale[scale]
            best = dict(height_m=h, width_m=.6, root_offset=[0., 0.], score=12., improvement=144., contrast=.5)
            return dict(status='assessed', best=best, frames=[0, 1, 2, 3, 4], height_range_m=[lo, hi], width_range_m=[.6, .6],
                        endpoint_context_supported=seen, lowest_compatible_censored=not seen, endpoint_censored=not seen,
                        compatible_context_supported=False, dimension_at_boundary=False, support_px=sc.root_support_px,
                        lowest_compatible=dict(height_m=lo, width_m=.6, root_offset=[0., 0.]))
        return fake

    def run_cell(self, cfg, by_scale):
        stack = np.ones((5, 80, 80)); vis = np.ones_like(stack); zero = np.zeros((80, 80))
        with patch('src.hati_core.adaptive_shadow.fit_patch', side_effect=self.scripted(by_scale)):
            return refine_cell(stack, vis, AZ, EL, .01, SC, RC, cfg, (40, 40), zero, zero)

    def test_guard_keeps_the_window_that_saw_the_shadow_end(self):
        # Scale 1 sees a 0.35 m rock's shadow end; scale 2 claims 1.1 m from unrelated dark ground further on.
        jump = {1: (.35, .3, .4, True), 2: (1.1, 1.0, 1.2, False), 4: (1.3, 1.1, 1.5, False)}
        free = self.run_cell(AC, jump)
        self.assertEqual(free['status'], 'unresolved_scale_limit')
        self.assertEqual(free['final']['best']['height_m'], 1.3)
        guarded = self.run_cell(replace(AC, context_guard=True), jump)
        self.assertEqual(guarded['status'], 'context_conflict')
        self.assertTrue(guarded['history'][-1]['context_conflict'])
        self.assertEqual(guarded['final']['best']['height_m'], .35)
        self.assertEqual(len(guarded['history']), 2)

    def test_guard_lets_a_cut_shadow_grow(self):
        # Scale 1's shadow ran past its window (censored): a taller fit in a larger window is expected, not a conflict.
        growth = {1: (.3, .3, .4, False), 2: (.9, .8, 1.0, True), 4: (.95, .85, 1.05, True)}
        guarded = self.run_cell(replace(AC, context_guard=True), growth)
        self.assertNotEqual(guarded['status'], 'context_conflict')
        self.assertEqual(guarded['final']['best']['height_m'], .95)

    def test_guard_leaves_config_hashes_alone(self):
        self.assertEqual(AC.hash(), replace(AC, context_guard=False).hash())
        self.assertNotEqual(AC.hash(), replace(AC, context_guard=True).hash())


class CasterProfileTests(unittest.TestCase):
    """A dome caster's shadow tapers to the tip a plate caster's reaches; each template reads its own caster."""

    def test_dome_tapers_to_the_same_tip(self):
        cfg = replace(SC, radius_px=20, root_support_px=18, psf_sigma_px=0., registration_sigma_px=0.)
        args = ((41, 41), (20., 20.), [0.], [4.], .4, 3.6, cfg)     # Sun from the top: the shadow runs down the rows
        plate, plate_cut = shadow_template(*args)
        dome, dome_cut = shadow_template(*args, profile='dome')
        self.assertEqual(plate_cut, dome_cut)
        tip = lambda t, col: np.flatnonzero(t[0, :, col] > .5).max()
        self.assertEqual(tip(plate, 20), tip(dome, 20))              # same tip down the middle
        self.assertLess(tip(dome, 21), tip(plate, 21))               # shorter towards the sides
        self.assertAlmostEqual(dome.sum()/plate.sum(), np.pi/4, delta=.03)
        with self.assertRaises(ValueError):
            shadow_template(*args, profile='cone')

    def test_each_template_reads_its_own_caster(self):
        # A rounded rock's tapered shadow read with rectangles comes out a quarter short, its true
        # height outside the compatible range; read with the dome template it comes out right.
        sc = replace(SC, radius_px=12, root_support_px=10)
        def scene(profile):
            t = shadow_template((25, 25), (12., 12.), AZ, EL, .4, .6, SC, profile=profile)[0]
            return 1-.7*t+np.random.default_rng(3).normal(0, .005, t.shape)
        def read(image, profile):
            fit = fit_patch(image, np.ones_like(image), AZ, EL, .005, sc, RC, replace(AC, caster_profile=profile), (0., 0.))
            return fit['best']['height_m'], fit['height_range_m']
        self.assertEqual(read(scene('dome'), 'dome'), (.4, [.4, .4]))
        self.assertEqual(read(scene('dome'), 'plate'), (.3, [.3, .3]))
        self.assertEqual(read(scene('plate'), 'plate'), (.4, [.4, .4]))

    def test_plate_keeps_config_hashes(self):
        self.assertEqual(AC.hash(), replace(AC, caster_profile='plate').hash())
        self.assertNotEqual(AC.hash(), replace(AC, caster_profile='dome').hash())
        with self.assertRaises(ValueError):
            AdaptiveConfig(caster_profile='cone')


if __name__ == '__main__':
    unittest.main()
