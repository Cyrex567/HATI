"""Read-only observer, numerical invariance, path boundaries and heartbeat tests."""
from datetime import datetime, timedelta, timezone
import json
import os
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

    def test_runner_record_survives_a_viewer_read_during_its_update(self):
        from run_saturation_campaign import write_json
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'campaign.json';write_json(target,dict(stage=1))
            reader=open(target,'rb')                   # HATI Watch mid-read
            threading.Timer(.15,reader.close).start()  # the read finishes shortly after
            write_json(target,dict(stage=2))           # the runner's update waits instead of aborting
            self.assertEqual(json.loads(target.read_text())['stage'],2)
            self.assertEqual(list(Path(tmp).glob('*.part')),[])

    def test_snapshot_write_retries_while_the_viewer_holds_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            real=os.replace;calls=[0]
            def busy(src,dst):
                calls[0]+=1
                if calls[0]<3:raise PermissionError(5,'Access is denied')
                return real(src,dst)
            with patch('live_feedback.os.replace',side_effect=busy):
                atomic_json(Path(tmp)/'live/T14.json',dict(message='written'))
            self.assertEqual(json.loads((Path(tmp)/'live/T14.json').read_text())['message'],'written')
            self.assertEqual(calls[0],3);self.assertEqual(list((Path(tmp)/'live').glob('*.part')),[])

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

    def test_stages_started_without_the_runner_are_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);now=datetime.now(timezone.utc)
            atomic_json(run/'live/T13.json',dict(stage='T13',updated=(now-timedelta(minutes=30)).isoformat()))
            atomic_json(run/'stages/T13/result.json',dict(status='PARTIAL',reason='classes are research labels'))
            atomic_json(run/'live/T14.json',dict(stage='T14',kind='stage',message='T14: sizing',updated=now.isoformat()))
            # A result left by an earlier run of T14 must not mark the rerun finished.
            old=run/'stages/T14/result.json';atomic_json(old,dict(status='PARTIAL'))
            stamp=(now-timedelta(hours=2)).timestamp();os.utime(old,(stamp,stamp))
            state=WatchStore(run).state()
            self.assertEqual([(r['id'],r['status']) for r in state['stages']],[('T13','PARTIAL'),('T14','RUNNING')])
            self.assertEqual((state['active_stage'],state['health'],state['heartbeat_source']),('T14','live','inferred'))
            self.assertEqual(state['snapshot']['message'],'T14: sizing')
            # A long silent step is quiet, not stale; a stage silent for hours with no result is stale.
            for minutes,health,status in ((5,'quiet','RUNNING'),(120,'stale','STALE')):
                atomic_json(run/'live/T14.json',dict(stage='T14',updated=(now-timedelta(minutes=minutes)).isoformat()))
                stamp=(now-timedelta(minutes=minutes)).timestamp();os.utime(run/'live/T14.json',(stamp,stamp))
                os.utime(old,(stamp-7200,stamp-7200))
                state=WatchStore(run).state()
                self.assertEqual((state['health'],state['stages'][1]['status']),(health,status))

    def test_sizing_feed_streams_every_cell_without_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            live=LiveFeedback(tmp,'T14',interval=0)
            live.sizing_start('relief check',2,image_shape=(64,64),touchdown=(10.,10.),pixel_m=.9,clearance_m=.3)
            live.sizing_relief(5,5,'rock_like');live.sizing_relief(6,6,None)
            self.assertEqual(live.sizing_state['relief']['rock_like'],1);self.assertEqual(live.sizing_state['relief']['unchecked'],1)
            live.sizing_start('detections',2,image_shape=(64,64),touchdown=(10.,10.),pixel_m=.9,clearance_m=.3)
            live.sizing_row(dict(row_px=12,col_px=10,state='context_supported_unvalidated',height_lower_bound_m=.45,
                                 height_m=.6,censored=False,score=11.),relief='rock_like')
            live.sizing_row(dict(row_px=40,col_px=40,state='unresolved_scale_limit',height_lower_bound_m=None,height_m=float('nan')))
            s=json.loads((Path(tmp)/'live/T14.json').read_text())['sizing']
            self.assertEqual((s['phase'],s['done'],s['total']),('detections',2,2))
            self.assertEqual(s['counts'],dict(sized=2,with_warning_evidence=1,context_supported=1,exceeding_clearance=1))
            col={k:i for i,k in enumerate(s['columns'])}
            self.assertAlmostEqual(s['casters'][0][col['distance_to_touchdown_m']],2*.9)
            self.assertEqual(s['casters'][0][col['relief_check']],'rock_like')
            self.assertIsNone(s['casters'][1][col['height_m']])
            json.dumps(WatchStore(tmp).state(),allow_nan=False)   # the viewer re-serialises strictly
            self.assertFalse(live.warned)

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
