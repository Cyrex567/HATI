"""Regression tests for campaign inference, support matching and failure packaging."""
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from affine import Affine
from pyproj import CRS

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import run_saturation_campaign as campaign
from saturation_experiments import Experiment, fit_profile, load_inputs, maps, t1, t2, t3, t4, t5, t6, t7, t8
from src.hati_core.regional_shadow import RegionalConfig, assess_regions
from src.hati_core.shadow_likelihood import ShadowConfig
from src.hati_core.landing_terrain import LandingConfig
from src.hati_core.campaign_controls import render_control


def tiny_bundle(path, *, seed=12, height=.3):
    shape = (32, 32); az = np.array([0., 70., 145., 210.]); el = np.array([4., 4.2, 3.8, 4.5])
    sc = ShadowConfig(radius_px=6, root_support_px=3, supersample=2)
    rc = RegionalConfig(heights_m=(.3, .6), widths_m=(.6,), tile_px=16)
    stack = render_control(shape, az, el, pixel_m=.9, seed=seed, noise=.015, height=height, supersample=4)
    crs = CRS.from_proj4('+proj=stere +lat_0=-90 +lat_ts=-85 +lon_0=0 +R=1737400 +units=m')
    transform = Affine(.9, 0, 0, 0, -.9, 0); ids = ['test0', 'test1', 'test2', 'test3']
    run = dict(frames=[dict(pid=p) for p in ids], azimuths_map=az.tolist(), elevations=el.tolist(),
               transform=tuple(transform), crs=crs.to_wkt(), image_posting_m=.9,
               counterfactual=dict(row_px=16, col_px=16), demo=True,
               landing=asdict(LandingConfig(baselines_m=(4.,), footprint_diameter_m=4., navigation_margin_m=1.,
                                           horizon_distance_m=8., dem_vertical_sigma_m=0.)),
               shadow_configuration=dict(shadow=asdict(sc), regional=asdict(rc), noise_sigma=.015))
    buf = io.BytesIO()
    np.savez_compressed(buf, stack=stack, visibility=np.ones_like(stack), frame_ids=ids, azimuths=az, elevations=el,
                        slope_row=np.zeros(shape), slope_col=np.zeros(shape), transform=tuple(transform), crs=crs.to_wkt())
    payload = buf.getvalue(); sha = hashlib.sha256(payload).hexdigest()
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('aligned_stack.npz', payload)
        z.writestr('source_run.json', json.dumps(run))
        z.writestr('model_diagnostics.json', json.dumps(dict(provenance=dict(stack_sha256=sha))))
    return sha


class CampaignTests(unittest.TestCase):
    def test_ablation_freezes_parent_support_and_all_frames_preserve_results(self):
        rng = np.random.default_rng(3)
        stack = 1+rng.normal(0, .02, (4, 28, 28))
        stack[3, 10, 10] = np.nan
        cfg = ShadowConfig(radius_px=6, root_support_px=3, supersample=2)
        rc = RegionalConfig(heights_m=(.3,), widths_m=(.6,))
        args = (stack, [0, 60, 140, 220], [4]*4, .02, cfg, rc)
        base = assess_regions(*args)
        all_frames = assess_regions(*args, frame_indices=[0, 1, 2, 3])
        cut = assess_regions(*args, frame_indices=[0, 1, 2])
        np.testing.assert_allclose(base['score'], all_frames['score'], equal_nan=True)
        np.testing.assert_allclose(base['common_fraction'], cut['common_fraction'], equal_nan=True)
        self.assertLessEqual(np.nanmax(cut['frame_count']), 3)
        with self.assertRaises(ValueError):
            assess_regions(*args, frame_indices=[0, 0, 1])

    def test_profile_callback_agrees_with_saved_cell_maximum(self):
        a = render_control((28, 28), [0, 75, 150, 225], [4]*4, pixel_m=.9, seed=5, noise=.01, supersample=4)
        rows = []
        out = assess_regions(a, [0, 75, 150, 225], [4]*4, .01,
                             ShadowConfig(radius_px=6, root_support_px=3, supersample=2),
                             RegionalConfig(heights_m=(.3,), widths_m=(.6,)), audit_callback=rows.append)
        self.assertTrue(rows)
        for cell in rows:
            r, c = cell['row_px'], cell['col_px']
            self.assertAlmostEqual(np.max(cell['scores']), out['score'][r, c])

    def test_failed_missing_partial_cannot_become_scientific_pass(self):
        for status in ('FAILED', 'BLOCKED', 'PARTIAL', 'PENDING', 'INTERRUPTED'):
            result = campaign.verdict([dict(id='T8', status=status)])
            self.assertEqual(result['scientific_verdict'], 'INCONCLUSIVE')
            self.assertEqual(result['operational_verdict'], 'WITHHOLD_OPERATIONAL_USE')

    def test_zip_has_relative_paths_hashes_and_failure_details(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'run'; (output/'stages').mkdir(parents=True); (output/'logs').mkdir()
            (output/'logs/T1.log').write_text('specific failure details')
            records = [dict(id='T1', title='Physical test', status='FAILED', reason='calculation failed')]
            archive = campaign.package(output, records, dict(test=True))
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                self.assertIn('run/START_HERE.html', z.namelist())
                self.assertTrue(all(not name.startswith('/') and '..' not in Path(name).parts for name in z.namelist()))
                self.assertEqual(z.read('run/logs/T1.log'), b'specific failure details')
                for line in z.read('run/MANIFEST.sha256').decode().splitlines():
                    sha, name = line.split('  ', 1)
                    self.assertEqual(hashlib.sha256(z.read('run/'+name)).hexdigest(), sha)

    def test_resume_rejects_modified_stage_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'file'; p.write_text('complete')
            record = dict(status='COMPLETE', artifact_sha256={'file': campaign.digest(p)})
            self.assertTrue(campaign.stage_cached(Path(tmp), record))
            p.write_text('corrupt')
            self.assertFalse(campaign.stage_cached(Path(tmp), record))

    def test_cli_runs_in_order_continues_after_failure_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'tests').mkdir(); (root/'scripts').mkdir(); (root/'src').mkdir()
            for name in ('a', 'b'):
                (root/'tests'/f'test_{name}.py').write_text('test fixture')
            bundle = root/'bundle.zip'; bundle.write_text('input fixture')
            dem = root/'dem.tif'; dem.write_text('DEM fixture')
            config = root/'config.json'; config.write_text('{}')
            output = root/'run'; export = root/'export'; calls = []
            fail = [True]
            def logged(command, log):
                log.write_text('fixture log')
                if '--stage' not in command:
                    calls.append(Path(command[1]).stem); return 0
                stage = command[command.index('--stage')+1]; calls.append(stage)
                if stage == 'T2' and fail[0]:
                    return 7
                folder = Path(command[command.index('--output')+1])
                campaign.write_json(folder/'result.json', dict(status='COMPLETE', reason='fixture complete'))
                return 0
            argv = ['runner', '--bundle', str(bundle), '--dem', str(dem), '--config', str(config),
                    '--output', str(output), '--export-dir', str(export)]
            with patch.object(campaign, 'ROOT', root), patch.object(campaign, 'capture', return_value=dict(returncode=0, stdout='', stderr='')), \
                    patch.object(campaign, 'run_logged', side_effect=logged), patch.object(sys, 'argv', argv):
                self.assertEqual(campaign.main(), 1)
                self.assertEqual(calls, ['test_a', 'test_b']+[s for s, _ in campaign.STAGES])
                self.assertTrue((export/'run_results.zip').exists())
                fail[0] = False; calls.clear()
                with patch.object(sys, 'argv', [*argv, '--resume']):
                    self.assertEqual(campaign.main(), 0)
                self.assertEqual(calls, ['T2'])
                config.write_text('{"changed":true}')
                with patch.object(sys, 'argv', [*argv, '--resume']), self.assertRaises(SystemExit):
                    campaign.main()

    def test_failed_stage_does_not_reuse_stale_scientific_finding(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); (out/'stages/T5').mkdir(parents=True)
            campaign.write_json(out/'stages/T5/result.json', dict(widest_selected_fraction=.9))
            self.assertEqual(campaign.evidence_summary(out, [dict(id='T5', status='FAILED')]), [])

    def test_real_workers_t1_to_t8_on_small_independent_fixture(self):
        from argparse import Namespace
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); bundle = folder/'input.zip'; tiny_bundle(bundle)
            cfg = json.loads((ROOT/'configs/saturation_campaign.json').read_text())
            cfg.update(synthetic_seeds=1, synthetic_locations=[[.5, .5]], geometry_shifts=[1],
                       drop_frame_groups={}, profile_step_px=32, larger_support_px=5,
                       profile_heights_m=[.3, .6], registration_tiles_px=[24],
                       registration_planted_shifts_px=[[0, 0]])
            config = folder/'config.json'; config.write_text(json.dumps(cfg))
            for stage, fn in [('T1', t1), ('T2', t2), ('T3', t3), ('T4', t4), ('T5', t5), ('T6', t6), ('T7', t7), ('T8', t8)]:
                args = Namespace(stage=stage, bundle=bundle, config=config, output=folder/'stages'/stage,
                                 campaign=folder, dem=None, thermal=None, held_out=None)
                with patch('subprocess.run', side_effect=AssertionError('ISIS/external executable forbidden')):
                    result = fn(Experiment(args))
                self.assertIn(result['status'], ('COMPLETE', 'PARTIAL', 'BLOCKED'))
                if stage in ('T4', 'T8'):
                    self.assertEqual(result['status'], 'BLOCKED')
            self.assertTrue((folder/'stages/T5/expanded_width/profiles.npz').exists())
            self.assertTrue((folder/'stages/T6/height_controls.json').exists())

    def test_held_out_rejects_development_stack(self):
        from argparse import Namespace
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); bundle = folder/'input.zip'; tiny_bundle(bundle)
            manifest = folder/'heldout.json'
            manifest.write_text(json.dumps(dict(split='held_out', annotation_source='independent annotation',
                                independent_of_development=True, scenes=[dict(bundle='input.zip', labels='labels.npz')])))
            args = Namespace(stage='T8', bundle=bundle, config=ROOT/'configs/saturation_campaign.json',
                             output=folder/'T8', campaign=folder, dem=None, thermal=None, held_out=manifest)
            with self.assertRaisesRegex(ValueError, 'development stack'):
                t8(Experiment(args))

    def test_external_adapters_require_support_and_count_cells_once(self):
        from argparse import Namespace
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); bundle = folder/'input.zip'; tiny_bundle(bundle)
            held = folder/'held.zip'; held_sha = tiny_bundle(held, seed=35)
            manifest = folder/'held.json'
            labels = folder/'labels.npz'
            np.savez_compressed(labels, hazard=np.zeros((32, 32), bool), complete=np.ones((32, 32), bool), stack_sha256=held_sha)
            manifest.write_text(json.dumps(dict(split='held_out', annotation_source='fixture source',
                                independent_of_development=True, scenes=[dict(bundle='held.zip', labels='labels.npz')])))
            args = Namespace(stage='T8', bundle=bundle, config=ROOT/'configs/saturation_campaign.json',
                             output=folder/'T8', campaign=folder, dem=None, thermal=None, held_out=manifest)
            result = t8(Experiment(args))
            row = result['scenes'][0]
            self.assertEqual(row['true_negative']+row['false_positive']+row['unknown_negative'], 25)
            self.assertIsNone(row['recall_counting_unknown_as_missed'])
            self.assertFalse(row['meets_declared_targets'])
            base = folder/'stages/T1/baseline'; base.mkdir(parents=True)
            np.savez_compressed(base/'regional.npz', index=np.ones((32, 32))*.2)
            thermal = folder/'thermal.csv'
            thermal.write_text('source,footprint_id,row_start,row_stop,col_start,col_stop,rock_abundance,uncertainty\n'
                               'fixture,too-large,0,200,0,200,0.1,0.02\n')
            args.stage = 'T4'; args.thermal = thermal; args.output = folder/'T4'
            result = t4(Experiment(args))
            self.assertEqual(result['status'], 'PARTIAL')
            self.assertEqual(result['supported_footprints'], 0)

    def test_zero_support_never_produces_height_or_uncertainty(self):
        cfg = ShadowConfig(radius_px=6, root_support_px=3)
        result = fit_profile(np.full((4, 13, 13), np.nan), np.ones((4, 13, 13)),
                             np.array([0, 90, 180, 270]), np.array([4]*4), (0, 0), cfg, [.3, .6], [.6], .03, 1.)
        self.assertEqual(result['status'], 'insufficient_frames')
        self.assertNotIn('best', result)
        self.assertNotIn('descriptive_delta_set_m', result)

    def test_native_dem_replay_keeps_three_separate_maps_and_synthetic_label(self):
        from argparse import Namespace
        import rasterio
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); bundle = folder/'input.zip'; tiny_bundle(bundle)
            data, run, transform, crs, proof = load_inputs(bundle)
            dem = folder/'dem.tif'
            with rasterio.open(dem, 'w', driver='GTiff', width=96, height=96, count=1,
                               dtype='float32', crs=crs.to_wkt(), transform=transform*Affine.translation(-32, -32)) as f:
                f.write(np.zeros((96, 96), dtype='float32'), 1)
            args = Namespace(stage='maps', bundle=bundle, config=ROOT/'configs/saturation_campaign.json',
                             output=folder/'maps', campaign=folder, dem=dem, thermal=None, held_out=None)
            with patch('landing_maps.make_previews'):
                result = maps(Experiment(args))
            self.assertEqual(result['status'], 'COMPLETE')
            for name in ('terrain', 'shadow', 'fused'):
                self.assertTrue((folder/f'maps/maps/{name}_hazard.tif').exists())
            self.assertTrue(json.loads((folder/'maps/maps/run.json').read_text())['demo'])


if __name__ == '__main__':
    unittest.main()
