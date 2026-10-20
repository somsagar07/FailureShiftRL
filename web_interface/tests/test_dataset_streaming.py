"""Row limits must stop source consumption before a complete split is downloaded."""
import io
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

from datasets import ClassLabel, Dataset, Features, Image as HFImage, IterableDataset, Value, load_dataset
from datasets.data_files import DataFilesDict, DataFilesList
from PIL import Image

from data_sources import DatasetManager


class DatasetStreamingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.manager = DatasetManager(self.temporary.name)
        buffer = io.BytesIO()
        Image.new('RGB', (8, 8), 'red').save(buffer, format='PNG')
        self.png = buffer.getvalue()
        self.features = Features({'image': HFImage(), 'label': ClassLabel(names=['cat', 'dog'])})

    def load(self, loader, counts, **fields):
        options = dict(config='default', splits=[dict(name=k, count=v) for k, v in counts.items()])
        selection = {'split': fields.pop('split')} if 'split' in fields else {'splits': list(counts)}
        with patch('data_sources.hf_dataset_options', return_value=options), \
                patch('datasets.load_dataset', side_effect=loader):
            job = self.manager.start(dict(source='huggingface', name='org/images',
                                          **selection, **fields))
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                info = self.manager.get(job['id'])
                if info['status'] != 'loading':
                    return info
                time.sleep(.01)
        self.fail('Dataset loading did not finish.')

    def image_stream(self, size, *, read_limit=None, seen=None):
        def rows():
            for index in range(size):
                if read_limit is not None and index >= read_limit:
                    raise AssertionError('Read past the requested row limit')
                if seen is not None:
                    seen.append(index)
                yield {'image': {'bytes': self.png, 'path': None}, 'label': index % 2}
        return IterableDataset.from_generator(rows, features=self.features)

    def test_first_200_of_million_rows_do_not_use_eager_loading_or_consume_row_201(self):
        seen = []
        def loader(*args, **kwargs):
            if not kwargs.get('streaming'):
                raise AssertionError('Eager loading downloads the entire source before subsetting')
            return self.image_stream(1_000_000, read_limit=200, seen=seen)
        info = self.load(loader, {'train': 1_000_000}, subset='200')
        self.assertEqual(info['status'], 'ready', info.get('error'))
        self.assertEqual(info['count'], 200)
        self.assertEqual(info['source_count'], 1_000_000)
        self.assertEqual(len(seen), 200)
        self.assertEqual(self.manager.labels(info['id'])['values']['199'], 'dog')
        self.assertEqual(self.manager.image(info['id'], 199).getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(self.manager.page(info['id'], 199)['items'][0]['source_index'], 199)
        # Repeated browsing/training reads must use retained rows, not reopen the stream.
        self.manager.image(info['id'], 0)
        self.assertEqual(len(seen), 200)
        self.assertEqual(list(Path(self.temporary.name).iterdir()), [])

    def test_range_skips_unneeded_splits_and_retains_original_row_numbers(self):
        seen = []
        def loader(*args, **kwargs):
            if kwargs['split'] != 'validation':
                raise AssertionError('Opened a split outside the requested range')
            if not kwargs.get('streaming'):
                raise AssertionError('Range used eager downloading')
            return self.image_stream(100, read_limit=30, seen=seen)
        info = self.load(loader, {'train': 1000, 'validation': 100, 'test': 300}, subset='[1020:1030]')
        self.assertEqual(info['status'], 'ready', info.get('error'))
        self.assertEqual(info['count'], 10)
        self.assertEqual(info['source_count'], 1400)
        self.assertEqual(len(seen), 30)
        item = self.manager.page(info['id'])['items'][0]
        self.assertEqual((item['source_split'], item['source_index']), ('validation', 20))

    def test_range_across_splits_keeps_labels_and_split_order(self):
        def loader(*args, **kwargs):
            if not kwargs.get('streaming'):
                raise AssertionError('Range used eager downloading')
            return self.image_stream(3 if kwargs['split'] == 'test' else 100,
                                     read_limit=3 if kwargs['split'] == 'test' else 2)
        info = self.load(loader, {'test': 3, 'train': 100}, subset='[2:5]')
        self.assertEqual(info['status'], 'ready', info.get('error'))
        items = self.manager.page(info['id'])['items']
        self.assertEqual([(x['source_split'], x['source_index'], x['label']) for x in items],
                         [('test', 2, 'cat'), ('train', 0, 'cat'), ('train', 1, 'dog')])

    def test_out_of_bounds_metadata_rejects_before_opening_data(self):
        def loader(*args, **kwargs):
            raise AssertionError('Invalid range must fail before starting a download')
        info = self.load(loader, {'train': 30}, subset='31')
        self.assertEqual(info['status'], 'error')
        self.assertIn('30 rows', info['error'])

    def test_legacy_split_expression_and_rows_keep_their_combined_selection(self):
        seen = {'train': [], 'validation': []}
        def loader(*args, **kwargs):
            split = kwargs['split']
            if split not in seen or not kwargs.get('streaming'):
                raise AssertionError('Resolve split expressions before bounded streaming.')
            return self.image_stream(20 if split == 'train' else 7, seen=seen[split],
                                     read_limit=10 if split == 'train' else 5)
        info = self.load(loader, {'train': 20, 'validation': 7},
                         split='train[:50%]+validation[-4:]', subset='[8:12]')
        self.assertEqual(info['status'], 'ready', info.get('error'))
        self.assertEqual(info['source_count'], 14)
        self.assertEqual(info['count'], 4)
        self.assertEqual(list(self.manager.labels(info['id'])['values'].values()),
                         ['cat', 'dog', 'dog', 'cat'])
        self.assertEqual(seen, {'train': list(range(10)), 'validation': list(range(5))})

    def test_reversed_legacy_range_does_not_shift_rows_into_the_next_split(self):
        def loader(*args, **kwargs):
            raise AssertionError('Invalid split ranges must fail before reading any data.')
        info = self.load(loader, {'train': 20, 'validation': 7},
                         split='train[10:5]+validation', subset='1')
        self.assertEqual(info['status'], 'error')
        self.assertIn('end no smaller than', info['error'])

    def test_unknown_total_stops_at_limit_without_exhausting_stream(self):
        def loader(*args, **kwargs):
            if not kwargs.get('streaming'):
                raise AssertionError('Unknown size used eager downloading')
            return self.image_stream(1000, read_limit=20)
        info = self.load(loader, {'train': None}, subset='20')
        self.assertEqual(info['status'], 'ready', info.get('error'))
        self.assertIsNone(info['source_count'])
        self.assertEqual(info['count'], 20)
        self.assertEqual(self.manager.page(info['id'], 19)['items'][0]['source_index'], 19)

    def test_unknown_short_stream_rejects_incomplete_subset(self):
        def loader(*args, **kwargs):
            return self.image_stream(7)
        info = self.load(loader, {'train': None}, subset='20')
        self.assertEqual(info['status'], 'error')
        self.assertIn('7 rows', info['error'])
        self.assertNotIn(info['id'], self.manager.records)

    def test_stream_errors_do_not_fall_back_to_a_full_download(self):
        def loader(*args, **kwargs):
            if not kwargs.get('streaming'):
                raise AssertionError('Silently attempted an eager download')
            raise RuntimeError('Dataset stream disconnected')
        info = self.load(loader, {'train': 1000}, subset='20')
        self.assertEqual(info['status'], 'error')
        self.assertIn('Dataset stream disconnected', info['error'])

    def test_summary_and_prompt_subsets_also_stop_before_extra_rows(self):
        for task in ('summarization', 'generation'):
            with self.subTest(task=task):
                def rows():
                    yield {'article': 'Source one', 'summary': 'Reference one', 'prompt': 'A cat'}
                    yield {'article': 'Source two', 'summary': 'Reference two', 'prompt': 'A dog'}
                    raise AssertionError('Read outside selected text rows')
                def loader(*args, **kwargs):
                    if not kwargs.get('streaming'):
                        raise AssertionError('Text subset used eager downloading')
                    return IterableDataset.from_generator(rows, features=Features({
                        'article': Value('string'), 'summary': Value('string'), 'prompt': Value('string')}))
                info = self.load(loader, {'train': 1000}, subset='2', task=task)
                self.assertEqual(info['status'], 'ready', info.get('error'))
                self.assertEqual(info['count'], 2)
                value = self.manager.text(info['id'], 1)
                self.assertEqual(value['text'], 'Source two' if task == 'summarization' else 'A dog')
                if task == 'summarization':
                    self.assertEqual(value['summary'], 'Reference two')

    def serve_files(self, files):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_HEAD(self):
                self.respond(False)

            def do_GET(self):
                self.respond(True)

            def respond(self, body):
                path = urlsplit(self.path).path
                payload = files.get(path)
                if payload is None:
                    self.send_error(404)
                    return
                start, end = 0, len(payload) - 1
                requested = self.headers.get('Range')
                if requested:
                    first, last = requested.removeprefix('bytes=').split('-')
                    if first:
                        start, end = int(first), min(int(last), end) if last else end
                    else:
                        start = max(0, len(payload) - int(last))
                self.send_response(206 if requested else 200)
                self.send_header('Accept-Ranges', 'bytes')
                self.send_header('Content-Length', str(end - start + 1))
                if requested:
                    self.send_header('Content-Range', f'bytes {start}-{end}/{len(payload)}')
                self.end_headers()
                if body:
                    requests.append((path, start, end))
                    try:
                        self.wfile.write(payload[start:end + 1])
                    except (BrokenPipeError, ConnectionResetError):
                        pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f'http://127.0.0.1:{server.server_port}', requests

    def test_remote_image_paths_fetch_only_selected_images_and_retain_their_bytes(self):
        base, requests = self.serve_files({'/selected.png': self.png, '/unselected.png': self.png})
        def rows():
            yield {'image': {'bytes': None, 'path': base + '/selected.png'}, 'label': 1}
            yield {'image': {'bytes': None, 'path': base + '/unselected.png'}, 'label': 0}
        def loader(*args, **kwargs):
            return IterableDataset.from_generator(rows, features=self.features)
        info = self.load(loader, {'train': 2}, subset='1')
        self.assertEqual(info['status'], 'ready', info.get('error'))
        self.assertEqual({path for path, _, _ in requests}, {'/selected.png'}, requests)
        before_preview = len(requests)
        self.assertEqual(self.manager.image(info['id'], 0).getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(self.manager.labels(info['id'])['values']['0'], 'dog')
        self.assertEqual(len(requests), before_preview, 'Preview must reuse retained image bytes.')
        self.assertEqual(list(Path(self.temporary.name).iterdir()), [])

    def test_real_parquet_stream_reads_first_shard_without_downloading_later_shards(self):
        import pyarrow.parquet as pq

        root = Path(self.temporary.name)
        features = Features({**self.features, 'unrelated': Value('binary')})
        files = {}
        for shard in range(3):
            path = root / f'shard-{shard}.parquet'
            records = Dataset.from_dict({
                'image': [{'bytes': self.png, 'path': None}] * 250,
                'label': [index % 2 for index in range(250)],
                'unrelated': [os.urandom(4096) for _ in range(250)]},
                features=features)
            records.to_parquet(path, batch_size=25)
            files['/' + path.name] = path.read_bytes()
        base, requests = self.serve_files(files)

        def loader(path, **kwargs):
            kwargs.pop('name', None)
            # Hub files are already resolved through repository metadata. Do
            # the same here: raw HTTP path discovery otherwise issues GET
            # status probes for every URL, without consuming their bodies.
            resolved = DataFilesDict(train=DataFilesList(
                [base + path for path in files], origin_metadata=[()] * len(files)))
            return load_dataset('parquet', data_files=resolved,
                                features=features, cache_dir=str(root / 'hf-cache'), **kwargs)

        readers = []
        open_reader = pq.ParquetFile
        def tracked_reader(*args, **kwargs):
            reader = open_reader(*args, **kwargs)
            readers.append(reader)
            return reader
        with patch.object(pq, 'ParquetFile', side_effect=tracked_reader):
            info = self.load(loader, {'train': 750}, subset='200')
        self.assertEqual(info['status'], 'ready', info.get('error'))
        self.assertEqual(info['count'], 200)
        self.assertTrue(readers)
        self.assertTrue(all(reader.closed for reader in readers), 'Readers must close after early termination.')
        self.assertTrue(requests, 'The test must exercise actual HTTP image-data reads.')
        self.assertEqual({path for path, _, _ in requests}, {'/shard-0.parquet'}, requests)
        self.assertFalse(list((root / 'hf-cache').rglob('*.arrow')))
        reads = len(requests)
        self.assertEqual(self.manager.image(info['id'], 199).getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(self.manager.labels(info['id'])['values']['199'], 'dog')
        self.assertEqual(len(requests), reads, 'Preview must reuse selected bytes.')


if __name__ == '__main__':
    unittest.main()
