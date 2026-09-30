"""Post-landing comparison: repeat measures against shifted copies, and a run on two tiny campaigns.

Run: python -m unittest discover -s tests -p test_post_landing_check.py
"""
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import post_landing_check as plc

CFG = json.loads((ROOT/'configs/post_landing_check.json').read_text(encoding='utf-8'))
SIZE, PIXEL = 96, .9


def fake_campaign(folder, frames, casters, seed, touchdown=(48.3, 48.2)):
    """A campaign folder with the files the comparison reads, on a 96 px grid."""
    rng = np.random.default_rng(seed)
    folder = Path(folder); st = folder/'stages'
    for sub in ('T1/baseline', 'T12', 'T13', 'T14/relief_corrected'):
        (st/sub).mkdir(parents=True, exist_ok=True)
    (folder/'inputs').mkdir(parents=True, exist_ok=True)
    stack = 1+.05*rng.normal(size=(len(frames), SIZE, SIZE))
    buf = io.BytesIO()
    np.savez(buf, stack=stack, transform=np.array([PIXEL, 0, 0, 0, -PIXEL, 0.]), frame_ids=np.array(frames),
             azimuths=np.linspace(0, 90, len(frames)), elevations=np.full(len(frames), 4.))
    run = dict(shadow_configuration=dict(noise_sigma=.03), image_posting_m=PIXEL,
               counterfactual=dict(row_px=touchdown[0], col_px=touchdown[1]))
    with zipfile.ZipFile(folder/'inputs/hati_diagnostic_bundle.zip', 'w') as z:
        z.writestr('aligned_stack.npz', buf.getvalue()); z.writestr('source_run.json', json.dumps(run))
    cells = [(r, r+4, c, c+4, r+2, c+2) for r in range(8, SIZE-8, 4) for c in range(8, SIZE-8, 4)]
    status = np.zeros((SIZE, SIZE), int); score = np.zeros((SIZE, SIZE))
    warm = np.random.default_rng(1).random((SIZE, SIZE)) < .3       # shared warning pattern
    for r, _, c, _, _, _ in cells:
        status[r:r+4, c:c+4] = 1
        score[r:r+4, c:c+4] = 30. if warm[r, c] else 2.
    np.savez(st/'T14/relief_corrected/regional.npz', status=status, score=score, cell_table=np.array(cells))
    np.savez(st/'T1/baseline/regional.npz', status=status, score=score*3)
    yy, xx = np.indices((SIZE, SIZE))
    height = np.sin(yy/5.)*np.cos(xx/7.)+.05*rng.normal(size=(SIZE, SIZE))
    np.savez(st/'T14/sfs.npz', height_m=height.astype('float32'), slope_deg=np.abs(height).astype('float32'),
             common=np.ones((SIZE, SIZE), bool))
    records = [dict(row_px=r, col_px=c, height_lower_bound_m=b, height_m=None, state='unresolved_scale_limit',
                    relief_check='rock_like', distance_to_touchdown_m=float(np.hypot(r-touchdown[0], c-touchdown[1])*PIXEL))
               for r, c, b in casters]
    (st/'T14/subpixel_casters.json').write_text(json.dumps(records))
    (st/'T14/result.json').write_text(json.dumps(dict(
        subpixel_casters=dict(sigma=.09, with_warning_evidence=len(records)), residual_scale_after=dict(pooled_sigma=.09),
        sfs=dict(explained_fraction=.8), exceedance=dict(after_measured_sigma=.2))))
    labels = [dict(row_px=r, col_px=c, label=('rock_like', 'relief_like')[(r//4+c//4) % 2]) for r, _, c, _, _, _ in cells]
    (st/'T13/athena_cells.json').write_text(json.dumps(labels))
    (st/'T12/result.json').write_text(json.dumps(dict(measured_pooled_sigma=.15,
                                                      relief_consistency=dict(explained_true_geometry=.8))))
    return folder


class RepeatMeasures(unittest.TestCase):
    def test_point_sets_repeat_only_where_they_coincide(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(10, 86, size=(40, 2)); usable = np.ones((SIZE, SIZE), bool)
        shifts = plc._shifts(rng, dict(chance=dict(trials=50, shift_min_m=10, shift_max_m=30)), PIXEL)
        same = plc.point_repeat(a, a+.5, usable, usable, 2, shifts, (SIZE, SIZE))
        self.assertEqual(same['observed'], 1.)
        self.assertGreater(same['z'], 3)
        other = plc.point_repeat(a, rng.uniform(10, 86, size=(3, 2)), usable, usable, 2, shifts, (SIZE, SIZE))
        self.assertLess(other['observed'], .5)
        # Points the other run could not have assessed do not count.
        blind = usable.copy(); blind[:, :48] = False
        self.assertEqual(plc.point_repeat(a, a, blind, usable, 2, shifts, (SIZE, SIZE))['n'],
                         int(np.sum(np.round(a[:, 1]) >= 48)))

    def test_fields_and_labels(self):
        rng = np.random.default_rng(2)
        f = rng.normal(size=(64, 64)); keep = np.ones_like(f, bool)
        shifts = np.array([[12., 5.], [-9., 14.], [20., -3.]])
        self.assertAlmostEqual(plc.field_correlation(f, f, keep, shifts, step=1)['observed'], 1.)
        labels = [dict(row_px=r, col_px=c, label=plc.LABELS[(r+c) % 4]) for r in range(4) for c in range(4)]
        agree = plc.label_agreement(labels, labels, np.ones((8, 8), bool))
        self.assertEqual(agree['agreement'], 1.); self.assertAlmostEqual(agree['kappa'], 1.)
        np.testing.assert_array_equal(plc._window(np.eye(4, dtype=bool), 1, 0)[1:], np.eye(4, dtype=bool)[:-1])


class Campaigns(unittest.TestCase):
    def test_two_campaigns_compare_and_refuse_shared_frames(self):
        rng = np.random.default_rng(3)
        rocks = [(float(r), float(c), round(float(b), 1)) for r, c, b in
                 zip(rng.uniform(10, 86, 30), rng.uniform(10, 86, 30), rng.uniform(.2, 1., 30))]
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(CFG, radii_m=dict(lander=4., neighbourhood=8., disturbed=10.),
                       chance=dict(CFG['chance'], trials=40, shift_min_m=15., shift_max_m=30.))
            config = Path(tmp)/'config.json'; config.write_text(json.dumps(cfg))
            pre = fake_campaign(Path(tmp)/'pre', ['a', 'b', 'c'], rocks, 4)
            post = fake_campaign(Path(tmp)/'post', ['d', 'e', 'f'], [(r+.4, c-.3, b) for r, c, b in rocks], 5)
            argv = ['post_landing_check.py', '--pre', str(pre), '--post', str(post), '--output', str(Path(tmp)/'out'),
                    '--config', str(config)]
            with patch.object(sys, 'argv', argv):
                plc.main()
            result = json.loads((Path(tmp)/'out/post_landing_check.json').read_text())
            self.assertTrue(result['verdicts']['E1_casters_repeat'])
            self.assertTrue(result['verdicts']['E2_heights_agree'])
            self.assertTrue(result['verdicts']['E4_warnings_repeat'])
            self.assertGreater(result['sfs_detail']['observed'], .5)
            self.assertTrue((Path(tmp)/'out/REPORT.md').exists())
            self.assertTrue((Path(tmp)/'out/post_landing_check.png').exists())
            shared = fake_campaign(Path(tmp)/'shared', ['a', 'x', 'y'], rocks, 6)
            with self.assertRaises(ValueError):
                plc.compare(plc.load_campaign(pre), plc.load_campaign(shared), cfg)


if __name__ == '__main__':
    unittest.main()
