"""Small offline end-to-end workers, replay checkpoints and live observations."""
from argparse import Namespace
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'tests')]
from test_saturation_campaign import tiny_bundle
from saturation_experiments import Experiment, t1, t9, t10, t11
from src.hati_core.adaptive_shadow import AdaptiveConfig, refine_regions
from test_adaptive_shadow import AZ, EL, SC, RC, AC, matched
from live_feedback import LiveFeedback


class AdaptiveCampaignTests(unittest.TestCase):
    def test_workers_export_replayable_arrays_histories_and_prediction_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); bundle = out/'input.zip'; tiny_bundle(bundle)
            cfg = json.loads((ROOT/'configs/saturation_campaign.json').read_text())
            cfg.update(synthetic_seeds=1, synthetic_locations=[[.5, .5]],
                adaptive=asdict(AdaptiveConfig(scale_factors=(1, 2), heights_m=(.2, .4, .6), widths_m=(.3, .6, .9), max_cells=2)),
                rock_seeds=1, rock_heights_m=[.3], rock_roi_cells=1, rock_supersample=2, prediction_step_px=128)
            config = out/'config.json'; config.write_text(json.dumps(cfg))
            for stage, fn in [('T1', t1), ('T9', t9), ('T10', t10), ('T11', t11)]:
                args = Namespace(stage=stage, bundle=bundle, config=config, output=out/'stages'/stage,
                                 campaign=out, dem=None, thermal=None, held_out=None,
                                 rock_catalog=ROOT/'data/rock_shapes/apollo_proxy_v1/catalog.json')
                with patch('subprocess.run', side_effect=AssertionError('external ingestion forbidden')):
                    result = fn(Experiment(args))
                self.assertIn(result['status'], ('COMPLETE', 'PARTIAL'))
                if stage == 'T9':
                    self.assertEqual(result['processed'], 2)
                    with patch('src.hati_core.adaptive_shadow.refine_cell', side_effect=AssertionError('checkpoint not reused')):
                        replay = t9(Experiment(args))
                    self.assertEqual(replay['state_counts'], result['state_counts'])
            for path in ('T9/adaptive.npz', 'T9/cells/000000.json', 'T10/controls.json', 'T11/predictions.json'):
                if path.endswith('000000.json'):
                    self.assertTrue(list((out/'stages/T9/cells').glob('*.json')))
                else:
                    self.assertTrue((out/'stages'/path).exists(), path)
            controls = json.loads((out/'stages/T10/controls.json').read_text())
            self.assertIn('apollo_proxy', [r['kind'] for r in controls])
            self.assertTrue(all(r['requested'] == r['processed'] for r in controls))
            self.assertTrue(all(r['roi_cells'] == 1 for r in controls))

    def test_serial_parallel_and_observer_preserve_scientific_results(self):
        stack = matched((25, 25)); shape = stack.shape[1:]
        baseline = dict(status=np.ones(shape), score=np.full(shape, 12.), endpoint_censored=np.ones(shape))
        cfg = replace(AC, max_cells=2)
        args = (stack, np.ones_like(stack), AZ, EL, .005, SC, RC)
        sr = np.zeros(shape); records_a = []; records_b = []
        a = refine_regions(*args, cfg, baseline, sr, sr, on_record=records_a.append)
        with tempfile.TemporaryDirectory() as tmp:
            live = LiveFeedback(tmp, 'T9', interval=0)
            b = refine_regions(*args, replace(cfg, workers=2), baseline, sr, sr,
                               on_record=records_b.append, observer=live.adaptive, progress=live.adaptive_progress)
            self.assertTrue((Path(tmp)/'live/T9.json').exists())
        for key, value in a.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, b[key], err_msg=key)
        self.assertEqual(records_a, records_b)
        self.assertTrue(np.any(a['status'] == 3))
        self.assertTrue(np.isnan(a['height_m'][a['status'] == 3]).all())
        self.assertTrue(np.isnan(a['width_m'][a['status'] == 3]).all())


if __name__ == '__main__':
    unittest.main()
