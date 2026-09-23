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


if __name__ == '__main__':
    unittest.main()
