"""Exercise the actual HTTP handler without requiring a listening socket."""
import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import create_server


class MemoryServer:
    def __init__(self, address, handler):
        self.server_port = address[1]
        self.handler = handler


class MemoryConnection:
    def __init__(self, payload):
        self.payload = payload
        self.output = b''
    def makefile(self, *args):
        return io.BytesIO(self.payload)
    def sendall(self, data):
        self.output += data


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        with patch('server.ThreadingHTTPServer', MemoryServer):
            self.server = create_server(8000, self.tmp.name, model_preparer=lambda model: (model, {}))
    def tearDown(self):
        self.tmp.cleanup()
    def request(self, path='/', method='GET', host='127.0.0.1:8000', origin=None, body=None):
        headers = f'{method} {path} HTTP/1.1\r\nHost: {host}\r\n'
        if origin:
            headers += 'Origin: ' + origin + '\r\n'
        payload = b'' if body is None else json.dumps(body).encode()
        if method == 'POST':
            headers += f'Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n'
        conn = MemoryConnection(headers.encode() + b'\r\n' + payload)
        with patch('server.BaseHTTPRequestHandler.log_message'):
            self.server.handler(conn, ('127.0.0.1', 1), self.server)
        raw_headers, raw_body = conn.output.split(b'\r\n\r\n', 1)
        return int(raw_headers.split()[1]), raw_body
    def test_page_assets_and_catalog(self):
        for path in ['/', '/app.js', '/graph.js', '/embedding-view.js', '/embedding-controller.js', '/style.css', '/api/catalog']:
            with self.subTest(path=path):
                code, body = self.request(path)
                self.assertEqual(code, 200)
                self.assertTrue(body)
    def test_reject_path_traversal(self):
        for path in ['/../server.py', '/%2e%2e/server.py', '/runs/../../server.py']:
            self.assertEqual(self.request(path)[0], 404)
    def test_reject_remote_hosts_and_origins(self):
        self.assertEqual(self.request(host='attacker.example')[0], 403)
        self.assertEqual(self.request('/api/previews', 'POST', origin='https://attacker.example', body={})[0], 403)
    def test_invalid_job_returns_actionable_client_error(self):
        code, body = self.request('/api/previews', 'POST', body={'action_space': {'rotation': [], 'darkness': [], 'saturation': []}})
        self.assertEqual(code, 400)
        self.assertIn('action space', json.loads(body)['error'])

    def test_invalid_embedding_requests_are_client_errors(self):
        for body in [None, [], {}, {'preview_id':'bad'},
                     {'preview_id': '0'*32, 'device': 'gpu'},
                     {'preview_id': '0'*32, 'device': None},
                     {'preview_id': '0'*32, 'device': ['cpu']}]:
            with self.subTest(body=body):
                code, payload = self.request('/api/embeddings', 'POST', body=body)
                self.assertEqual(code, 400)
                self.assertIn('error', json.loads(payload))

    def test_dataset_embedding_and_expansion_over_http(self):
        import numpy as np
        from PIL import Image

        class Encoder:
            feature_method = 'test RGB means'
            def __init__(self, model):
                self.model = model
            def encode(self, images):
                return np.array([list(np.asarray(image).mean(axis=(0,1)))+[1.] for image in images])

        with patch('server.HuggingFaceEncoder', Encoder), patch('server.ThreadingHTTPServer', MemoryServer):
            self.server = create_server(8000, self.tmp.name, model_preparer=lambda model: (model, {}))
        root = Path(self.tmp.name)/'embedding-input'
        root.mkdir()
        for index, color in enumerate(['red','green','blue','yellow']):
            Image.new('RGB', (8,8), color).save(root/f'{index}.png')

        def wait(path, job):
            deadline = time.monotonic()+5
            while job['status'] in ('loading','queued','running') and time.monotonic()<deadline:
                time.sleep(.01)
                code, body = self.request(path)
                self.assertEqual(code, 200)
                job = json.loads(body)
            self.assertNotEqual(job['status'], 'error', job.get('error'))
            return job

        code, body = self.request('/api/datasets','POST',body={'source':'folder','path':str(root)})
        self.assertEqual(code,202)
        dataset=json.loads(body);dataset=wait('/api/datasets/'+dataset['id'],dataset)
        config={'dataset_id':dataset['id'],'model':{'source':'huggingface','name':'test/model'},
                'action_space':{'rotation':[0],'darkness':[.1,.5],'saturation':[]}}
        code,body=self.request('/api/previews','POST',body=config)
        self.assertEqual(code,202)
        preview=json.loads(body);preview=wait('/api/previews/'+preview['id'],preview)
        code,body=self.request('/api/embeddings','POST',body={'preview_id':preview['id'],'device':'cpu'})
        self.assertEqual(code,202)
        job=json.loads(body);path='/api/embeddings/'+job['id'];job=wait(path,job)
        self.assertEqual(job['status'],'completed')
        self.assertEqual(job['device'], 'cpu')
        self.assertEqual(job['requested_device'], 'cpu')
        self.assertTrue(all(value >= 0 for value in job['timings_seconds'].values()))
        self.assertEqual(len(job['points']),4)
        self.assertEqual([p['index'] for p in job['points']],[0,1,2,3])
        self.assertEqual(len(job['points'][0]['position']),3)
        self.assertEqual(self.request(job['points'][3]['image'])[0],200)
        code,body=self.request(path+'/expand','POST',body={'input_index':2})
        self.assertEqual(code,202)
        expansion=json.loads(body)
        expansion=wait(path+'/expansions/'+expansion['id'],expansion)
        self.assertEqual(expansion['status'],'completed')
        self.assertEqual(expansion['input_index'],2)
        self.assertEqual(len(expansion['nodes']),2)
        self.assertEqual(expansion['original']['image'],job['points'][2]['image'])
        code,image=self.request(expansion['nodes'][1]['image'])
        self.assertEqual(code,200)
        self.assertEqual(Image.open(io.BytesIO(image)).getpixel((100,100)),(0,0,127))
        for index in [-1,4,True,'2']:
            self.assertEqual(self.request(path+'/expand','POST',body={'input_index':index})[0],400)

    def test_folder_to_graph_to_correct_image_over_http(self):
        from PIL import Image
        root = Path(self.tmp.name)/'input'
        root.mkdir()
        Image.new('RGB', (4, 4), 'red').save(root/'a.png')
        Image.new('RGB', (4, 4), 'blue').save(root/'b.png')
        code, body = self.request('/api/datasets', 'POST', body={'source':'folder', 'path':str(root)})
        self.assertEqual(code, 202)
        dataset = json.loads(body)
        deadline = time.monotonic() + 5
        while dataset['status'] == 'loading' and time.monotonic() < deadline:
            time.sleep(.01)
            dataset = json.loads(self.request('/api/datasets/'+dataset['id'])[1])
        self.assertEqual(dataset['status'], 'ready')
        # Dataset samples are viewable before creating any analysis or preview.
        for index, color in enumerate(((255, 0, 0), (0, 0, 255))):
            code, pixels = self.request(f'/datasets/{dataset["id"]}/images/{index}.png')
            self.assertEqual(code, 200)
            self.assertEqual(Image.open(io.BytesIO(pixels)).getpixel((100, 100)), color)
        self.assertEqual(self.request(f'/datasets/{dataset["id"]}/images/2.png')[0], 400)
        self.assertEqual(self.request('/datasets/'+'0'*32+'/images/0.png')[0], 404)
        config = {'dataset_id':dataset['id'], 'action_space':{'rotation':[0], 'darkness':[], 'saturation':[]}}
        code, body = self.request('/api/previews', 'POST', body=config)
        self.assertEqual(code, 202)
        job = json.loads(body)
        while job['status'] in ('queued', 'running') and time.monotonic() < deadline:
            time.sleep(.01)
            job = json.loads(self.request('/api/previews/'+job['id'])[1])
        code, body = self.request('/api/previews/'+job['id']+'?input=1')
        self.assertEqual(code, 200)
        selected = json.loads(body)
        self.assertEqual(selected['total'], 2)
        self.assertEqual(selected['input_name'], 'b.png')
        code, pixels = self.request(selected['nodes'][0]['image'])
        self.assertEqual(code, 200)
        self.assertEqual(Image.open(io.BytesIO(pixels)).getpixel((100,100)), (0,0,255))
        self.assertEqual(self.request('/api/previews/'+job['id']+'?input=2')[0], 400)
        self.assertEqual(self.request('/api/datasets/'+dataset['id']+'/items?offset=-1')[0], 400)

    def test_text_workspaces_analyze_and_expand_through_the_same_http_routes(self):
        import numpy as np

        seen = []
        class Encoder:
            feature_method = 'test text features'
            def __init__(self, model):
                seen.append(model)
            def encode(self, texts):
                return np.array([[1., len(text), sum(c.isupper() for c in text)] for text in texts])

        with patch('server.create_encoder', Encoder), patch('server.ThreadingHTTPServer', MemoryServer):
            self.server = create_server(8000, self.tmp.name, model_preparer=lambda model: (model, {}))

        def finish(path, job):
            deadline = time.monotonic()+5
            while job['status'] in ('queued', 'running') and time.monotonic() < deadline:
                time.sleep(.005)
                code, body = self.request(path)
                self.assertEqual(code, 200)
                job = json.loads(body)
            self.assertEqual(job['status'], 'completed', job.get('error'))
            return job

        for task, count in (('summarization', 14), ('generation', 27)):
            with self.subTest(task=task):
                config = {'task': task, 'model': {'source': 'huggingface', 'name': 'test/text-model'}}
                code, body = self.request('/api/previews', 'POST', body=config)
                self.assertEqual(code, 202)
                preview = json.loads(body)
                preview = finish('/api/previews/'+preview['id'], preview)
                code, body = self.request('/api/embeddings', 'POST', body={'preview_id': preview['id']})
                self.assertEqual(code, 202)
                job = json.loads(body)
                path = '/api/embeddings/'+job['id']
                job = finish(path, job)
                self.assertEqual(job['task'], task)
                self.assertEqual(job['projection']['fit_count'], count+1)
                self.assertEqual(job['points'][0]['text'], preview['original']['text'])
                code, body = self.request(path+'/expand', 'POST', body={'input_index': 0})
                self.assertEqual(code, 202)
                expansion = json.loads(body)
                expansion = finish(path+'/expansions/'+expansion['id'], expansion)
                self.assertEqual(len(expansion['nodes']), count)
                self.assertEqual(expansion['nodes'][0]['text'], preview['nodes'][0]['text'])
                self.assertIsInstance(expansion['nodes'][0]['similarity'], float)
                self.assertEqual(self.request(path+'/expand', 'POST', body={'input_index': 1})[0], 400)
        self.assertEqual([model['task'] for model in seen], ['summarization', 'generation'])


if __name__ == '__main__':
    unittest.main()
