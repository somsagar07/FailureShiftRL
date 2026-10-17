"""Image datasets from a local folder, browser folder upload, or Hugging Face."""
import copy
import io
import os
import re
import threading
import uuid
from pathlib import Path, PurePosixPath

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tif', '.tiff'}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def load_hf_images(name, split, config, column):
    try:
        from datasets import load_dataset, Image
    except ImportError as exc:
        raise ValueError('Hugging Face dataset dependencies are missing. Stop the server and run bash run.sh from web_interface; the launcher installs and checks its own environment.') from exc
    dataset = load_dataset(name, name=config or None, split=split)
    image_columns = [key for key, feature in dataset.features.items() if isinstance(feature, Image)]
    if not column and len(image_columns) == 1:
        column = image_columns[0]
    if column not in image_columns:
        raise ValueError('Choose an image column. Available image columns: ' + (', '.join(image_columns) or 'none'))
    # Selecting only this column avoids decoding unrelated audio/video fields.
    return dataset.select_columns([column]).cast_column(column, Image(decode=False)), column


class DatasetManager:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {}
        self.records = {}

    def get(self, dataset_id):
        with self.lock:
            return copy.deepcopy(self.jobs[dataset_id])

    def start(self, data):
        if not isinstance(data, dict) or data.get('source') not in ('folder', 'huggingface', 'upload'):
            raise ValueError('Choose a local folder or Hugging Face dataset.')
        source = data['source']
        dataset_id = uuid.uuid4().hex
        if source == 'folder':
            value = data.get('path', '')
            if not isinstance(value, str) or not value.strip() or len(value) > 4096:
                raise ValueError('Enter a folder path.')
            root = Path(value).expanduser().resolve()
            if not root.is_dir():
                raise ValueError('That folder does not exist on this computer.')
            args = (root,)
            name = str(root)
        elif source == 'huggingface':
            name = data.get('name', '')
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)?', name) or '..' in name:
                raise ValueError('Enter a Hugging Face dataset ID, such as owner/dataset.')
            split, config, column = data.get('split', 'train'), data.get('config', ''), data.get('column', '')
            if any(not isinstance(value, str) or len(value) > 200 for value in (split, config, column)) or not split.strip():
                raise ValueError('Enter a valid split, configuration and image column.')
            args = (name, split.strip(), config.strip(), column.strip())
        else:
            name = data.get('name', 'Selected folder')
            files = data.get('files')
            if not isinstance(name, str) or len(name) > 200:
                raise ValueError('Invalid folder name.')
            if not isinstance(files, list) or not 1 <= len(files) <= 10000:
                raise ValueError('Choose 1–10,000 images for browser upload. For larger folders, enter the folder path.')
            entries, seen = [], set()
            for item in files:
                if not isinstance(item, dict):
                    raise ValueError('Invalid file entry.')
                relative, size = item.get('name'), item.get('size')
                if not isinstance(relative, str) or not relative or len(relative) > 4096 or '\\' in relative or '\x00' in relative:
                    raise ValueError('Invalid image filename.')
                relative_path = PurePosixPath(relative)
                if relative_path.is_absolute() or '..' in relative_path.parts or relative in seen or relative_path.suffix.lower() not in IMAGE_SUFFIXES:
                    raise ValueError('Choose unique image paths inside the selected folder.')
                if type(size) is not int or not 0 < size <= MAX_UPLOAD_BYTES:
                    raise ValueError('Each uploaded image must be under 20 MB.')
                seen.add(relative)
                entries.append(dict(name=relative, size=size, path=self.directory/dataset_id/f'{len(entries)}{relative_path.suffix.lower()}'))
            (self.directory/dataset_id).mkdir()
            args = (entries,)
        with self.lock:
            job = dict(id=dataset_id, source=source, name=name, status='uploading' if source == 'upload' else 'loading', count=0)
            self.jobs[dataset_id] = job
            if source == 'upload':
                self.records[dataset_id] = args[0]
                job['expected_count'] = len(args[0])
            else:
                threading.Thread(target=self.load, args=(dataset_id, source, args), daemon=True).start()
            return copy.deepcopy(job)

    def load(self, dataset_id, source, args):
        try:
            if source == 'folder':
                root = args[0]
                records = []
                for parent, directories, names in os.walk(root, followlinks=False):
                    directories[:] = [name for name in directories if not (Path(parent)/name).is_symlink()]
                    for name in names:
                        path = Path(parent)/name
                        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file() and not path.is_symlink():
                            records.append(dict(name=path.relative_to(root).as_posix(), path=path))
                records.sort(key=lambda row: row['name'])
                extra = {}
            else:
                records, column = load_hf_images(*args)
                extra = dict(column=column, split=args[1], config=args[2])
            if not len(records):
                raise ValueError('No supported images were found in this folder or dataset split.')
            with self.lock:
                self.records[dataset_id] = records
                self.jobs[dataset_id].update(status='ready', count=len(records), **extra)
        except Exception as exc:
            with self.lock:
                self.jobs[dataset_id].update(status='error', error=str(exc))

    def upload(self, dataset_id, index, payload):
        with self.lock:
            job = self.jobs[dataset_id]
            if job['status'] != 'uploading' or type(index) is not int or not 0 <= index < len(self.records[dataset_id]):
                raise ValueError('Invalid upload file or dataset.')
            record = self.records[dataset_id][index]
            if len(payload) != record['size']:
                raise ValueError('The image upload was incomplete.')
            temporary = record['path'].with_suffix('.tmp')
            temporary.write_bytes(payload)
            temporary.replace(record['path'])

    def finish_upload(self, dataset_id):
        with self.lock:
            if self.jobs[dataset_id]['status'] != 'uploading':
                raise ValueError('This dataset is not accepting uploads.')
            records = self.records[dataset_id]
            if not all(row['path'].is_file() for row in records):
                raise ValueError('Upload all selected files before loading this folder.')
            self.jobs[dataset_id].update(status='ready', count=len(records))
            return self.get(dataset_id)

    def ready(self, dataset_id):
        info = self.get(dataset_id)
        if info['status'] != 'ready':
            raise ValueError('Load the folder or dataset before generating combinations.')
        return info

    def page(self, dataset_id, offset=0, limit=100):
        info = self.ready(dataset_id)
        if type(offset) is not int or type(limit) is not int or not 0 <= offset < info['count'] or not 1 <= limit <= 100:
            raise ValueError('Invalid dataset page.')
        records = self.records[dataset_id]
        items = [dict(index=i, name=records[i]['name'] if info['source'] != 'huggingface' else f'Image {i+1}')
                 for i in range(offset, min(offset+limit, info['count']))]
        return dict(items=items, count=info['count'], offset=offset)

    def image(self, dataset_id, index):
        from PIL import Image, ImageOps
        info = self.ready(dataset_id)
        if type(index) is not int or not 0 <= index < info['count']:
            raise ValueError('Image index is outside the selected dataset.')
        record = self.records[dataset_id][index]
        if info['source'] == 'huggingface':
            value = record[info['column']]
            if isinstance(value, dict):
                source = io.BytesIO(value['bytes']) if value.get('bytes') is not None else value.get('path')
            else:
                source = value
            if not isinstance(source, (str, Path, io.BytesIO)):
                raise ValueError('This row has no readable image.')
            if isinstance(source, str) and '://' in source:
                raise ValueError('This dataset stores remote image URLs. Choose a dataset with embedded or cached images.')
        else:
            source = record['path']
        try:
            with Image.open(source) as image:
                if image.width * image.height > 20_000_000:
                    raise ValueError('Use images under 20 megapixels.')
                return ImageOps.exif_transpose(image).convert('RGB').resize((224, 224))
        except (OSError, TypeError) as exc:
            raise ValueError(f'Could not read image {index+1}: {exc}') from exc
