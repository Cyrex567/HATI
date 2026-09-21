"""Read-only observer, numerical invariance, path boundaries and heartbeat tests."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'dashboard')]
from live_feedback import LiveFeedback, Heartbeat, atomic_json
from hati_watch import WatchStore, make_handler, safe_file
from src.hati_core.regional_shadow import assess_regions, RegionalConfig
from src.hati_core.shadow_likelihood import ShadowConfig
from src.hati_core.campaign_controls import render_control


class LiveWatchTests(unittest.TestCase):
    def test_observer_preserves_every_scientific_array_and_reports_real_fit(self):
        az=np.array([0,75,150,225.]); el=np.full(4,4.)
        stack=render_control((32,32),az,el,pixel_m=.9,seed=12,noise=.015,supersample=4)
        original=stack.copy()
        sc=ShadowConfig(radius_px=6,root_support_px=3,supersample=2)
        rc=RegionalConfig(heights_m=(.3,.6),widths_m=(.6,),tile_px=4)
        base=assess_regions(stack,az,el,.015,sc,rc)
        with tempfile.TemporaryDirectory() as tmp:
            live=LiveFeedback(tmp,'T1',interval=0)
            observed=assess_regions(stack,az,el,.015,sc,rc,observer=lambda i:live.regional(i,'fixture'))
            for k,v in base.items():
                if isinstance(v,np.ndarray):np.testing.assert_array_equal(v,observed[k],err_msg=k)
            self.assertEqual(base['candidates'],observed['candidates'])
            np.testing.assert_array_equal(stack,original)
            snapshot=json.loads((Path(tmp)/'live/T1.json').read_text())
            fit=snapshot['fit']
            self.assertEqual(snapshot['cells_visited'],snapshot['cells_total'])
            self.assertAlmostEqual(fit['score']**2,fit['null_energy']-fit['fitted_energy'],places=8)
            self.assertAlmostEqual(sum(fit['frame_delta_chi2']),fit['score']**2,places=8)
            self.assertAlmostEqual(fit['index'],fit['score']/(fit['score']+fit['score_scale']))
            self.assertEqual(len(fit['observed']),len(fit['frames']))

    def test_display_write_failure_is_nonfatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            live=LiveFeedback(tmp,'T1',interval=0)
            with patch('live_feedback.atomic_json',side_effect=OSError('display disk unavailable')):
                live.update(force=True,message='test')
                live.field('test',np.ones((8,8)))
            self.assertTrue(live.warned)

    def test_observer_is_invariant_with_masked_slopes_ablation_and_failed_writes(self):
        az=np.array([0,75,150,225.]); el=np.full(4,4.)
        stack=render_control((28,28),az,el,pixel_m=.9,seed=31,noise=.015,slope_rc=(.01,-.02),supersample=4)
        stack[3,12:14,12:14]=np.nan
        visible=np.ones_like(stack);visible[0,8:10,8:10]=0
        sc=ShadowConfig(radius_px=6,root_support_px=3,supersample=2)
        rc=RegionalConfig(heights_m=(.3,.6),widths_m=(.6,),tile_px=8)
        original=stack.copy()
        for indices,failed in ((None,False),([0,1,2],False),(None,True)):
            with self.subTest(indices=indices,display_failure=failed), tempfile.TemporaryDirectory() as tmp:
                kwargs=dict(visible=visible,slope_row=np.full((28,28),.01),slope_col=np.full((28,28),-.02),frame_indices=indices)
                baseline=assess_regions(stack,az,el,.015,sc,rc,**kwargs)
                live=LiveFeedback(tmp,'T1',interval=0)
                with patch('live_feedback.atomic_json',side_effect=OSError('display unavailable') if failed else atomic_json):
                    observed=assess_regions(stack,az,el,.015,sc,rc,observer=lambda i:live.regional(i,'masked'),**kwargs)
                for key,value in baseline.items():
                    if isinstance(value,np.ndarray):np.testing.assert_array_equal(value,observed[key],err_msg=key)
                self.assertEqual(baseline['candidates'],observed['candidates'])
                np.testing.assert_array_equal(stack,original)
                self.assertEqual(live.warned,failed)
                if not failed:
                    fit=live.state['fit']
                    self.assertAlmostEqual(sum(fit['frame_delta_chi2']),fit['score']**2,places=8)
                    if indices:self.assertEqual(fit['frames'],indices)
                    live.update(force=True,kind='controls',message='Next calculation')
                    saved=json.loads((Path(tmp)/'live/T1.json').read_text())
                    self.assertEqual(saved['fit'],fit)

    def test_terrain_readout_preserves_native_measurements_and_unknowns(self):
        from src.hati_core.landing_terrain import LandingConfig, terrain_assessment
        yy,xx=np.indices((32,32));dem=.03*yy-.05*xx
        dem[14:17,14:17]+=1.2;original=dem.copy()
        cfg=LandingConfig(baselines_m=(4.,),footprint_diameter_m=4.)
        terrain=terrain_assessment(dem,1.,cfg)
        before={k:v.copy() for k,v in terrain['primary'].items() if isinstance(v,np.ndarray)}
        with tempfile.TemporaryDirectory() as tmp:
            live=LiveFeedback(tmp,'maps');live.terrain(terrain,cfg,1.,lambda a:a,(16,16))
            saved=WatchStore(tmp).state()['terrain']
            ratios=[]
            for key,entry in saved['sample'].items():
                self.assertEqual(entry['value'],terrain['primary'][key][16,16]);ratios.append(entry['ratio'])
            self.assertAlmostEqual(saved['index'],max(ratios)/(1+max(ratios)))
            live.terrain(terrain,cfg,1.,lambda a:a,(0,0))
            self.assertIsNone(live.state['terrain']['index'])
            self.assertTrue(all(m['value'] is None for m in live.state['terrain']['sample'].values()))
        for key,value in before.items():np.testing.assert_array_equal(value,terrain['primary'][key])
        np.testing.assert_array_equal(dem,original)

    def test_http_is_read_only_and_confined_to_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); run=root/'run'; run.mkdir()
            (root/'secret.txt').write_text('not served')
            (run/'logs').mkdir(); (run/'logs/T1.log').write_text('signed score output')
            atomic_json(run/'campaign.json',dict(stages=[dict(id='T1',title='Physical response',status='RUNNING')]))
            before={p.relative_to(run):p.read_bytes() for p in run.rglob('*') if p.is_file()}
            server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(WatchStore(run)))
            worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
            base=f'http://127.0.0.1:{server.server_address[1]}'
            try:
                with urlopen(base+'/api/state') as r:
                    state=json.load(r);self.assertEqual(state['selected_stage'],'T1');self.assertIn('signed score',state['log'])
                with urlopen(base+'/') as r:self.assertIn(b'OBSERVATION ONLY',r.read())
                for route in ('/artifact/../secret.txt','/artifact/%2e%2e/secret.txt','/api/start','/../secret.txt'):
                    with self.assertRaises(HTTPError) as cm:urlopen(base+route)
                    self.assertEqual(cm.exception.code,404)
                for method in ('POST','PUT','DELETE','PATCH'):
                    with self.assertRaises(HTTPError) as cm:urlopen(Request(base+'/api/state',data=b'{}',method=method))
                    self.assertEqual(cm.exception.code,405)
            finally:
                server.shutdown();server.server_close();worker.join()
            after={p.relative_to(run):p.read_bytes() for p in run.rglob('*') if p.is_file()}
            self.assertEqual(before,after)

    def test_stale_heartbeat_and_partial_json_remain_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);atomic_json(run/'campaign.json',dict(stages=[dict(id='T1',status='RUNNING')]))
            atomic_json(run/'live/runtime.json',dict(state='running',updated='2000-01-01T00:00:00+00:00'))
            (run/'live/T1.json').write_text('{partial')
            state=WatchStore(run).state()
            self.assertEqual(state['health'],'stale');self.assertIsNone(state['snapshot'])
            heartbeat=Heartbeat(run);heartbeat.start();heartbeat.close()
            self.assertEqual(WatchStore(run).state()['health'],'finished')

    def test_source_preview_of_existing_bundle_does_not_modify_run(self):
        from test_saturation_campaign import tiny_bundle
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);(run/'inputs').mkdir();tiny_bundle(run/'inputs/hati_diagnostic_bundle.zip')
            before=sorted(p.relative_to(run).as_posix() for p in run.rglob('*'))
            store=WatchStore(run);meta=store.inputs()
            self.assertEqual(len(meta['frames']),4)
            self.assertTrue(store.images['frame_00.png'].startswith(b'\x89PNG'))
            self.assertEqual(before,sorted(p.relative_to(run).as_posix() for p in run.rglob('*')))


if __name__=='__main__':unittest.main()
