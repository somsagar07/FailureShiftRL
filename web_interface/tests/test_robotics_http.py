import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from test_http import MemoryConnection, MemoryServer
from server import create_server


class RoboticsHTTPTests(unittest.TestCase):
    def test_prediction_rows_are_versioned_finite_and_private(self):
        import numpy as np
        manager=self.server.robotics
        job=manager.start(dict(policy=dict(format='torchscript',source='local',name='/tmp/test.pt')))
        manager.threads[job['id']].join(timeout=2)
        jid=job['id']; route=f'/api/robotics/runs/{jid}/predictions'
        with manager.lock:
            saved=manager.jobs[jid]
            saved['embedding']={'points':[{'index':0,'seed':42},{'index':1,'seed':43}]}
            saved['predicted_landscape']=dict(policy_updates=0,available=False)
        code,body=self.request(route+'?input_index=1')
        self.assertEqual(code,200); self.assertEqual(json.loads(body)['actions'],[])
        self.assertFalse(json.loads(body)['available'])
        ids=[node['id'] for node in saved['nodes']]
        values=np.stack([np.arange(len(ids)),np.arange(len(ids))+.5]).astype('float32')
        np.save(manager.directory/jid/'dqn-predictions-3.npy',values)
        with manager.lock:
            saved['predicted_landscape']=dict(policy_updates=3,available=True,action_ids=ids,seeds=[42,43],values_file='dqn-predictions-3.npy')
        code,body=self.request(route+'?input_index=1'); response=json.loads(body)
        self.assertEqual(code,200); self.assertEqual(response['policy_updates'],3)
        self.assertEqual(response['actions'][0],dict(action_id=ids[0],q_value=.5))
        with manager.lock: saved['predicted_landscape']['seeds']=[43,42]
        self.assertEqual(self.request(route+'?input_index=1')[0],400)
        with manager.lock: saved['predicted_landscape']['seeds']=[42,43]
        self.assertEqual(self.request(f'/robotics/{jid}/artifacts/dqn-predictions-3.npy')[0],404)
        for query in ['', '?input_index=-1','?input_index=2','?input_index=x','?input_index=0&input_index=1']:
            self.assertEqual(self.request(route+query)[0],400)
        (manager.directory/jid/'dqn-predictions-3.npy').unlink()
        code,body=self.request(route+'?input_index=0')
        self.assertEqual(code,200); self.assertFalse(json.loads(body)['available'])

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        def runner(job,directory,emit,cancel):
            if job['mode']=='environment':
                (directory/'live.jpg').write_bytes(b'preview')
                emit({'type':'progress','environment_info':{'task':'Lift','action_dimension':7}})
                emit({'type':'artifact','name':'preview','filename':'live.jpg'})
                return
            emit({'type':'record','record':dict(action_id='baseline',seed=42,success=True)})
        with patch('server.ThreadingHTTPServer',MemoryServer):
            self.server=create_server(8000,Path(self.tmp.name)/'previews',robotics_runner=runner)

    def tearDown(self):
        self.server.robotics.close(); self.tmp.cleanup()

    def request(self,path,method='GET',data=None,raw=None,origin=None):
        body=raw if raw is not None else (json.dumps(data).encode() if data is not None else b'')
        headers=f'{method} {path} HTTP/1.1\r\nHost: localhost:8000\r\n'
        if origin: headers+=f'Origin: {origin}\r\n'
        kind='application/octet-stream' if raw is not None else 'application/json'
        headers+=f'Content-Type: {kind}\r\nContent-Length: {len(body)}\r\n\r\n'
        conn=MemoryConnection(headers.encode()+body)
        with patch('server.BaseHTTPRequestHandler.log_message'):
            self.server.handler(conn,('127.0.0.1',1),self.server)
        head,body=conn.output.split(b'\r\n\r\n',1)
        return int(head.split()[1]),body

    def test_catalog_policy_upload_and_run(self):
        code,body=self.request('/api/robotics/catalog'); self.assertEqual(code,200)
        self.assertEqual(len(json.loads(body)['policy_formats']),4)
        code,body=self.request('/api/robotics/policies','POST',dict(format='torchscript',files=[dict(path='test.pt',size=4)]))
        self.assertEqual(code,201); upload=json.loads(body)
        route=f'/api/robotics/policies/{upload["id"]}'
        self.assertEqual(self.request(route+'/files/0?offset=0','POST',raw=b'test')[0],200)
        self.assertEqual(self.request(route+'/finish','POST',{})[0],200)
        code,body=self.request('/api/robotics/runs','POST',dict(policy=dict(format='torchscript',source='upload',upload_id=upload['id'])))
        self.assertEqual(code,202); job=json.loads(body)
        deadline=time.monotonic()+2
        while job['status'] in ('running','queued') and time.monotonic()<deadline:
            time.sleep(.01); job=json.loads(self.request('/api/robotics/runs/'+job['id'])[1])
        self.assertEqual(job['status'],'completed')
        self.assertEqual(self.request(job['artifacts']['report'])[0],200)
        for name in ['request.json','worker.log','job.json','../test.pt']:
            self.assertEqual(self.request(f'/robotics/{job["id"]}/artifacts/{name}')[0],404)

    def test_invalid_upload_and_cross_origin_are_rejected(self):
        for data in [[],{},dict(format='lerobot',files=[dict(path='../config.json',size=2)])]:
            self.assertEqual(self.request('/api/robotics/policies','POST',data)[0],400)
        self.assertEqual(self.request('/api/robotics/runs','POST',{},origin='https://evil.example')[0],403)
        self.assertEqual(self.request('/api/robotics/runs','POST',dict(mode='discover',analysis_id='bad'))[0],404)

    def test_environment_preview_without_checkpoint_and_starter_endpoint(self):
        code,body=self.request('/api/robotics/runs','POST',{'mode':'environment','environment':{'task':'Lift'}})
        self.assertEqual(code,202); job=json.loads(body)
        deadline=time.monotonic()+2
        while job['status'] in ('running','queued') and time.monotonic()<deadline:
            time.sleep(.01); job=json.loads(self.request('/api/robotics/runs/'+job['id'])[1])
        self.assertEqual(job['status'],'completed')
        self.assertEqual(job['baseline']['episodes'],0)
        self.assertEqual(self.request(job['artifacts']['preview'])[0],200)
        code,body=self.request('/api/robotics/starter-policies')
        self.assertEqual(code,200); self.assertEqual(json.loads(body)['policies'],[])


if __name__=='__main__': unittest.main()
