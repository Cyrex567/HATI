"""T12 residual scale and T16 adaptive nulls: estimator truth, selection rules, workers."""
from argparse import Namespace
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'tests')]
from src.hati_core.noise_scale import (NoiseScaleConfig, measure_residual_scale, per_frame_sigma,
                                       select_patches, structure)
from src.hati_core.adaptive_shadow import AdaptiveConfig


def planted(n=5, size=120, sigma=(.01, .02, .03, .05, .08), quadratic=False, seed=1):
    rng = np.random.default_rng(seed)
    rr, cc = np.indices((size, size), float)
    texture = .2*rng.normal(size=(size, size))   # static albedo; the null must remove it exactly
    frames = []
    for k in range(n):
        frame = 1+texture+.001*k*rr-.0007*k*cc+sigma[k]*rng.normal(size=(size, size))
        if quadratic:
            # Strong enough that, inside a 24 px patch, the part a plane cannot
            # absorb (~0.17 rms at the outer frames) exceeds the planted noise.
            frame += 2e-3*(k+1)*(rr-size/2)**2
        frames.append(frame)
    return np.stack(frames), np.asarray(sigma)


class ResidualScaleTests(unittest.TestCase):
    def measure(self, stack, cfg=NoiseScaleConfig(patch_px=24)):
        ones = np.ones_like(stack); flat = np.zeros(stack.shape[1:])
        return measure_residual_scale(stack, ones, flat, flat, cfg)

    def test_planted_per_frame_noise_is_recovered(self):
        stack, sigma = planted()
        m = self.measure(stack)
        np.testing.assert_allclose(m['per_frame_sigma'], sigma, rtol=.08)
        self.assertAlmostEqual(m['pooled_sigma'], float(np.sqrt(np.mean(sigma**2))), delta=.03*m['pooled_sigma'])
        self.assertEqual(m['negative_variance_frames'], [])

    def test_a_quadratic_background_needs_the_quadratic_null(self):
        stack, sigma = planted(quadratic=True)
        plane = self.measure(stack, NoiseScaleConfig(patch_px=24, spatial_degree=1))
        quad = self.measure(stack, NoiseScaleConfig(patch_px=24, spatial_degree=2))
        np.testing.assert_allclose(quad['per_frame_sigma'], sigma, rtol=.08)
        self.assertGreater(plane['per_frame_sigma'][-1], 1.2*sigma[-1])

    def test_patches_are_chosen_by_support_illumination_and_slope_only(self):
        stack, _ = planted(n=4, size=48, sigma=(.01,)*4)
        visibility = np.ones_like(stack); slope_row = np.zeros((48, 48)); slope_col = np.zeros((48, 48))
        stack[2, 3, 3] = np.nan                    # patch (0, 0): missing data
        visibility[1, 30, 5] = .5                  # patch (24, 0): not fully lit in one frame
        slope_row[5, 30] = .2                      # patch (0, 24): steeper than the limit
        accepted, rejected = select_patches(stack, visibility, slope_row, slope_col, NoiseScaleConfig(patch_px=24))
        self.assertEqual(accepted, [(24, 24)])
        self.assertEqual(rejected, dict(data=1, illumination=1, slope=1))

    def test_structure_separates_independent_noise_from_a_drifting_pattern(self):
        rng = np.random.default_rng(4)
        noise = rng.normal(size=(40, 24, 24))
        rows, cols = np.indices((24, 24))
        drifting = np.stack([np.sin((rows+.7*cols)/3+.8*i) for i in range(40)]) + .1*noise
        white, stripes = structure(noise, (2, 4)), structure(drifting, (2, 4))
        self.assertLess(abs(white['lag1_correlation']), .05)
        self.assertLess(abs(white['block_variance_ratio']['4']-1), .15)
        self.assertGreater(stripes['lag1_correlation'], .5)
        self.assertGreater(stripes['block_variance_ratio']['4'], 2)

    def test_negative_per_frame_variance_is_clipped_and_reported(self):
        sigma, pooled, negative = per_frame_sigma([1e-8, .01, .01, .01])
        self.assertEqual(sigma[0], 0.)
        self.assertEqual(negative, [0])
        self.assertGreater(pooled, 0)

    def test_stage_selection_keeps_declared_order_and_rejects_unknown_stages(self):
        from run_saturation_campaign import select_stages
        self.assertEqual(select_stages('T16,T1,T12'), ['T1', 'T12', 'T16'])
        self.assertEqual(select_stages(None)[0], 'maps')
        with self.assertRaises(ValueError):
            select_stages('T1,T99')

    def test_null_summary_counts_context_supported_trials_against_the_gate(self):
        from adaptive_experiments import summarise_null_trials
        row = dict(noise_pass='render_measured', kind='static', height_m=None, render_noise=.1, model_noise=.03,
                   adaptive_warning_cells=0, context_supported_cells=0, roi_assessed_cells=9, requested=0,
                   unresolved_cells=0, recovered_within_2px=None)
        rows = [row, dict(row, context_supported_cells=2, adaptive_warning_cells=3)]
        summary = summarise_null_trials(rows, .1)[0]
        self.assertEqual(summary['trials_with_context_supported'], 1)
        self.assertEqual(summary['context_supported_cells'], 2)
        self.assertEqual(summary['context_supported_trial_fraction'], .5)
        self.assertFalse(summary['within_declared_gate'])


class NoiseWorkerTests(unittest.TestCase):
    def test_t12_then_t16_render_at_the_measured_scale_without_external_calls(self):
        from test_saturation_campaign import tiny_bundle
        from saturation_experiments import Experiment, t1, t12, t16
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); bundle = out/'input.zip'; tiny_bundle(bundle)
            cfg = json.loads((ROOT/'configs/saturation_campaign.json').read_text())
            # Two workers on Linux run T16's forked pool here, in the software
            # checks, rather than for the first time after the long stages.
            cfg.update(synthetic_seeds=1, synthetic_locations=[[.5, .5]],
                       adaptive=asdict(AdaptiveConfig(scale_factors=(1, 2), heights_m=(.2, .4, .6), widths_m=(.3, .6, .9),
                                                      max_cells=2, workers=1 if os.name == 'nt' else 2)),
                       rock_supersample=2, rock_roi_cells=1, noise_scale=dict(patch_px=8, max_slope=.05),
                       noise_control_seeds=1, null_seeds=1, null_caster_heights_m=[.3], null_render_noise='measured')
            config = out/'config.json'; config.write_text(json.dumps(cfg))
            def run(stage, fn, campaign):
                args = Namespace(stage=stage, bundle=bundle, config=config, output=campaign/'stages'/stage,
                                 campaign=campaign, dem=None, thermal=None, held_out=None, rock_catalog=None)
                with patch('subprocess.run', side_effect=AssertionError('external ingestion forbidden')):
                    return fn(Experiment(args))
            # Without T12 in the campaign, T16 must refuse rather than guess a noise level.
            blocked = run('T16', t16, out/'empty')
            self.assertEqual(blocked['status'], 'BLOCKED')
            run('T1', t1, out)
            noise = run('T12', t12, out)
            self.assertEqual(noise['status'], 'PARTIAL')
            self.assertGreater(noise['measured_pooled_sigma'], 0)
            self.assertEqual(len(noise['control_passes']), 3)
            self.assertIsNotNone(noise['rescaled_baseline'])
            self.assertTrue((out/'stages/T12/residual_scale.png').exists())
            nulls = run('T16', t16, out)
            self.assertEqual(nulls['status'], 'PARTIAL')
            self.assertIn('T12', nulls['render_noise_source'])
            names = [p['name'] for p in nulls['noise_passes']]
            self.assertEqual(names[0], 'render_assumed')
            kinds = {(s['noise_pass'], s['kind']) for s in nulls['summaries']}
            self.assertIn(('render_assumed', 'structured_null'), kinds)
            for s in nulls['summaries']:
                if s['height_m'] is None:
                    self.assertIn('within_declared_gate', s)
            self.assertTrue((out/'stages/T16/adaptive_nulls.png').exists())


if __name__ == '__main__':
    unittest.main()
