import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robotics import RoboticsManager, normalize_config, variation_nodes, summarize


def config():
    return dict(environment=dict(task='Lift', robots=['Panda'], controller='OSC_POSE', horizon=10),
                policy=dict(format='torchscript', source='local', name='/tmp/policy.pt'),
                action_space=dict(friction_scale=[.5, 1, 1.5]), episodes_per_variation=2)


class RoboticsTests(unittest.TestCase):
    def test_dqn_settings_episode_embeddings_and_live_statistics_are_preserved(self):
        def runner(job,directory,emit,cancel):
            if job['mode']=='analyze':
                emit(dict(type='record',record=dict(action_id='baseline',seed=42,success=True)))
                emit(dict(type='embedding',embedding=dict(version=1,source={'kind':'policy_hidden'},
                    projection={'rank':2},points=[dict(index=0,seed=42,name='Episode 1',position=[1,2,3],
                                                       feature_samples=10,image='baseline-seed-42.jpg')])))
            else:
                self.assertEqual(job['config']['search']['episodes'],20)
                emit(dict(type='progress',search={'episode':20,'updates':17,'model_calls':2,'cache_hits':18},
                    current={'input_index':0,'action_id':1},policy_statistics={'updates':17},
                    predicted_landscape={'policy_updates':17}))
        with tempfile.TemporaryDirectory() as tmp:
            manager=RoboticsManager(tmp,runner=runner)
            analysis=self.wait(manager,manager.start(config())['id'])
            self.assertTrue(analysis['embedding']['points'][0]['image'].startswith('/robotics/'))
            self.assertEqual(analysis['baseline']['records'][0]['input_index'],0)
            self.assertEqual(analysis['baseline']['records'][0]['embedding']['position'],[1,2,3])
            with self.assertRaises(ValueError): manager.start(dict(mode='discover',analysis_id=analysis['id'],search={'episodes':True}))
            result=self.wait(manager,manager.start(dict(mode='discover',analysis_id=analysis['id'],search={'episodes':20}))['id'])
            self.assertEqual(result['embedding'],analysis['embedding'])
            self.assertEqual(result['search']['updates'],17)
            self.assertEqual(result['predicted_landscape']['policy_updates'],17)
            self.assertIn('DQN',result['strategy'])
            self.assertIn('DQN',manager.artifact(result['id'],'report.html').read_text())
            self.assertIsNone(result['current'])

    def test_old_analysis_needs_reanalyze_before_dqn_but_legacy_sweep_still_works(self):
        def runner(job,directory,emit,cancel):
            if job['mode']=='analyze': emit(dict(type='record',record=dict(action_id='baseline',seed=42,success=True)))
        with tempfile.TemporaryDirectory() as tmp:
            manager=RoboticsManager(tmp,runner=runner)
            analysis=self.wait(manager,manager.start(config())['id'])
            with self.assertRaisesRegex(ValueError,'Analyze again'):
                manager.start(dict(mode='discover',analysis_id=analysis['id'],search={}))
            result=self.wait(manager,manager.start(dict(mode='discover',analysis_id=analysis['id']))['id'])
            self.assertEqual(result['status'],'completed')

    def test_environment_load_needs_no_policy_and_creates_no_failure_scores(self):
        def runner(job,directory,emit,cancel):
            self.assertNotIn('policy',job['config'])
            (directory/'live.jpg').write_bytes(b'preview')
            emit({'type':'progress','environment_info':{'task':'Lift','action_dimension':7}})
            emit({'type':'artifact','name':'preview','filename':'live.jpg'})
        with tempfile.TemporaryDirectory() as tmp:
            manager=RoboticsManager(tmp,runner=runner)
            job=self.wait(manager,manager.start({'mode':'environment','environment':{'task':'Lift'},'seed':7})['id'])
            self.assertEqual(job['status'],'completed')
            self.assertEqual(job['environment_info']['action_dimension'],7)
            self.assertEqual(job['nodes'],[])
            self.assertEqual(job['baseline']['episodes'],0)
            self.assertIsNone(job['baseline']['failure_rate'])
            self.assertNotIn('Baseline success',manager.artifact(job['id'],'report.html').read_text())
            with self.assertRaises(ValueError): manager.start({'mode':'discover','analysis_id':job['id']})
            for bad in [{'seed':True},{'environment':{'task':'../Lift'}},{'environment':{'controller':'wrong'}}]:
                with self.assertRaises(ValueError): manager.start(dict(mode='environment',**bad))

    def test_empty_environment_worker_does_not_claim_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager=RoboticsManager(tmp,runner=lambda *args:None)
            result=self.wait(manager,manager.start({'mode':'environment'})['id'])
            self.assertEqual(result['status'],'error')
            self.assertIn('preview',result['error'])

    def test_starter_catalog_only_exposes_complete_local_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager=RoboticsManager(tmp,runner=lambda *args:None)
            self.assertEqual(manager.starter_policies()['policies'],[])
            root=Path(tmp)/'starter-policies'; root.mkdir()
            (root/'lift.pt').write_bytes(b'weights')
            good={'id':'lift','label':'Lift','description':'Trained on demonstrations',
                  'checkpoint_sha256':hashlib.sha256(b'weights').hexdigest(),
                  'policy':{'format':'torchscript','source':'local','name':'lift.pt'},
                  'environment':{'task':'Lift','robots':['Panda'],'controller':'OSC_POSE'},
                  'mapping':{'state_keys':['robot0_eef_pos','cube_pos']},
                  'evaluation':{'successes':18,'episodes':20,'success_rate':.9},
                  'training':{'method':'behavior cloning'}}
            missing=dict(good,id='missing',policy=dict(good['policy'],name='missing.pt'))
            outside=dict(good,id='outside',policy=dict(good['policy'],name='../policy.pt'))
            bad_metrics=dict(good,id='metrics',evaluation={'successes':21,'episodes':20,'success_rate':1.05})
            (root/'catalog.json').write_text(json.dumps({'policies':[good,missing,outside,bad_metrics]}))
            catalog=manager.starter_policies()
            self.assertEqual([p['id'] for p in catalog['policies']],['lift'])
            self.assertEqual(catalog['policies'][0]['policy']['name'],str((root/'lift.pt').resolve()))
            self.assertTrue(catalog['warnings'])
            self.assertEqual(manager.catalog()['starter_policies'],catalog['policies'])

    def test_starter_scores_require_the_evaluated_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager=RoboticsManager(tmp,runner=lambda *args:None)
            root=Path(tmp)/'starter-policies'; root.mkdir()
            checkpoint=root/'best.pt'; checkpoint.write_bytes(b'evaluated weights')
            entry={'id':'lift','label':'Lift',
                   'policy':{'format':'torchscript','source':'local','name':'best.pt'},
                   'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   'evaluation':{'successes':18,'episodes':20}}
            manifest=root/'catalog.json'; manifest.write_text(json.dumps({'policies':[entry]}))
            self.assertEqual(len(manager.starter_policies()['policies']),1)
            checkpoint.write_bytes(b'retrained weights without evaluation')
            self.assertEqual(manager.starter_policies()['policies'],[])
            self.assertIn('checkpoint',manager.starter_policies()['warnings'][0].lower())
            checkpoint.write_bytes(b'evaluated weights')
            entry.pop('checkpoint_sha256')
            manifest.write_text(json.dumps({'policies':[entry]}))
            self.assertEqual(manager.starter_policies()['policies'],[])

    def test_config_rejects_unbounded_or_nonfinite_inputs(self):
        for change in [dict(episodes_per_variation=True), dict(seed=-1), dict(device='gpu'),
                       dict(action_space={'gravity_scale':[float('nan')]}),
                       dict(action_space={'unknown':[1]}), dict(mapping=[]),
                       dict(environment={'task':'../Lift'}), dict(variation_budget=0)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                normalize_config(dict(config(), **change))
        self.assertEqual(normalize_config(config())['environment']['control_freq'], 20)

    def test_graph_nodes_cover_combinations_not_fabricated_scores(self):
        nodes = variation_nodes({'mass_scale':[.5, 1], 'object_x_offset':[-.03, 0, .03]})
        self.assertEqual(len(nodes), 6)
        self.assertTrue(all(node['status']=='untested' and node['failure_rate'] is None for node in nodes))
        self.assertEqual(len({tuple(n['position']) for n in nodes}), 6)

    def test_errors_are_excluded_and_regressions_are_paired_by_seed(self):
        baseline=[dict(seed=4, success=True), dict(seed=5, success=False)]
        rows=[dict(seed=4, success=False), dict(seed=5, success=True), dict(seed=6, error='render error', success=None)]
        result=summarize(rows, baseline)
        self.assertEqual(result['failure_rate'], .5)
        self.assertEqual(result['episodes'], 2)
        self.assertEqual(result['errors'], 1)
        self.assertEqual((result['regressions'],result['recoveries']), (1,1))
        self.assertIsNone(summarize([rows[2]])['failure_rate'])

    def test_upload_offsets_retry_completion_and_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager=RoboticsManager(tmp, runner=lambda *args: None)
            for path in ['../policy.pt','/policy.pt','x/../../p.pt','x\\p.pt','x//p.pt']:
                with self.subTest(path=path), self.assertRaises(ValueError):
                    manager.create_upload({'format':'torchscript','files':[{'path':path,'size':6}]})
            upload=manager.create_upload({'format':'torchscript','files':[{'path':'policy.pt','size':6}]})
            uid=upload['id']
            with self.assertRaises(ValueError): manager.finish_upload(uid)
            manager.upload_chunk(uid,0,0,b'abc')
            manager.upload_chunk(uid,0,0,b'abc')
            with self.assertRaises(ValueError): manager.upload_chunk(uid,0,0,b'xyz')
            with self.assertRaises(ValueError): manager.upload_chunk(uid,0,4,b'de')
            manager.upload_chunk(uid,0,3,b'def')
            self.assertEqual(manager.finish_upload(uid)['status'],'ready')
            chosen=manager.resolve_policy({'format':'torchscript','source':'upload','upload_id':uid})
            self.assertEqual(Path(chosen['name']).read_bytes(),b'abcdef')
            with self.assertRaises(ValueError): manager.upload_chunk(uid,0,6,b'!')
            with self.assertRaises(FileNotFoundError): manager.artifact(uid,'policy.pt')

    def test_discovery_requires_completed_analysis_and_keeps_snapshot(self):
        def runner(job, directory, emit, cancel):
            if job['mode']=='analyze':
                emit({'type':'record','record':dict(action_id='baseline', seed=42,success=True)})
            else:
                emit({'type':'record','record':dict(action_id=0, seed=42,success=False)})
                emit({'type':'record','record':dict(action_id=1, seed=42,success=True,reused_baseline=True)})
        with tempfile.TemporaryDirectory() as tmp:
            manager=RoboticsManager(tmp, runner=runner)
            source=config()
            analysis=manager.start(dict(source, mode='analyze'))
            source['action_space']['friction_scale'].append(2)
            self.wait(manager,analysis['id'])
            discovery=manager.start(dict(mode='discover',analysis_id=analysis['id']))
            result=self.wait(manager,discovery['id'])
            self.assertEqual(len(result['nodes']),3)
            self.assertEqual(result['nodes'][0]['failure_rate'],1)
            self.assertIsNone(result['nodes'][2]['failure_rate'])
            self.assertIn('Baseline reused',manager.artifact(result['id'],'report.html').read_text())
            self.assertEqual(result['baseline']['success_rate'],1)
            reopened=RoboticsManager(tmp,runner=runner)
            self.assertEqual(reopened.get(result['id'])['status'],'completed')

    def test_native_upload_with_sidecars_resolves_main_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager=RoboticsManager(tmp,runner=lambda *args:None)
            upload=manager.create_upload({'format':'sb3','files':[
                {'path':'policy.metadata.json','size':2},{'path':'policy.zip','size':4},
                {'path':'policy.vecnormalize.pkl','size':3}]})
            for index,payload in enumerate([b'{}',b'zip!',b'pkl']): manager.upload_chunk(upload['id'],index,0,payload)
            manager.finish_upload(upload['id'])
            selected=manager.resolve_policy({'format':'sb3','source':'upload','upload_id':upload['id']})
            self.assertEqual(Path(selected['name']).name,'policy.zip')
            with self.assertRaises(ValueError):
                manager.create_upload({'format':'sb3','files':[{'path':'policy.zip','size':1},{'path':'code.py','size':1}]})

    def test_cancel_preserves_partial_results_and_single_active_worker(self):
        entered=threading.Event()
        def runner(job,directory,emit,cancel):
            emit({'type':'record','record':dict(action_id='baseline',seed=42,success=False)})
            entered.set()
            cancel.wait(3)
        with tempfile.TemporaryDirectory() as tmp:
            manager=RoboticsManager(tmp,runner=runner)
            job=manager.start(config()); self.assertTrue(entered.wait(1))
            with self.assertRaises(RuntimeError): manager.start(config())
            manager.cancel(job['id'])
            result=self.wait(manager,job['id'])
            self.assertEqual(result['status'],'cancelled')
            self.assertEqual(result['baseline']['episodes'],1)
            manager.close()

    def wait(self, manager, jid):
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            job=manager.get(jid)
            if job['status'] not in ('queued','running'): return job
            time.sleep(.01)
        self.fail('worker did not finish')


if __name__=='__main__': unittest.main()
