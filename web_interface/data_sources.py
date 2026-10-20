"""Image, summarization, and generation datasets with paginated previews."""
import copy
import hashlib
import io
import os
import re
import threading
import uuid
from itertools import islice
from pathlib import Path, PurePosixPath

from classification_labels import extract_dataset_labels, label_columns
from generation_prompts import normalize_prompt_checks, validate_prompt

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tif', '.tiff'}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_SOURCE_CHARS = 200000
MAX_SUMMARY_CHARS = 50000
_SOURCE_COLUMN_NAMES = {'article', 'dialogue', 'document', 'text'}
_SUMMARY_COLUMN_NAMES = {'summary', 'highlights', 'abstract'}
_PROMPT_COLUMN_NAMES = {'prompt', 'caption', 'description', 'text'}
_CHECKS_COLUMN_NAMES = {'checks', 'vqa_list', 'prompt_checks'}
GENEVAL2_REVISION = 'a6e82d2289e8d418f27f0adee77908b07060eea3'
GENEVAL2_URL = f'https://raw.githubusercontent.com/facebookresearch/GenEval2/{GENEVAL2_REVISION}/geneval2_data.jsonl'
HF_DEPENDENCY_ERROR = ('Hugging Face dataset dependencies are missing. Stop the server and run bash run.sh '
                       'from web_interface; the launcher installs and checks its own environment.')


def _validate_hf_name(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)?', name) or '..' in name:
        raise ValueError('Enter a Hugging Face dataset ID, such as owner/dataset.')


def _text_column_options(features):
    from datasets import Value
    columns = [key for key, feature in features.items()
               if isinstance(feature, Value) and feature.dtype in ('string', 'large_string')]
    sources = [key for key in columns if key.lower() in _SOURCE_COLUMN_NAMES]
    summaries = [key for key in columns if key.lower() in _SUMMARY_COLUMN_NAMES]
    prompts = [key for key in columns if key.lower() in _PROMPT_COLUMN_NAMES]
    checks = [key for key, feature in features.items() if _prompt_checks_feature(feature)]
    named_checks = [key for key in checks if key.lower() in _CHECKS_COLUMN_NAMES]
    return dict(text_columns=columns,
                suggested_text_column=sources[0] if len(sources) == 1 else None,
                suggested_summary_column=summaries[0] if len(summaries) == 1 else None,
                prompt_columns=list(columns),
                suggested_prompt_column=prompts[0] if len(prompts) == 1 else None,
                checks_columns=checks,
                suggested_checks_column=named_checks[0] if len(named_checks) == 1 else None)


def _prompt_checks_feature(feature):
    """Identify pair/object-list columns without materializing any dataset rows."""
    from datasets import Value

    def item(value):
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return value[0]
        return getattr(value, 'feature', None)

    def string(value):
        return isinstance(value, Value) and value.dtype in ('string', 'large_string')

    child = item(feature)
    if isinstance(child, dict):
        return {'question', 'answer'} <= set(child) and not set(child) - {'question', 'answer', 'skill'} and all(
            string(value) or (isinstance(value, Value) and value.dtype == 'null') for value in child.values())
    return string(item(child)) or (isinstance(child, Value) and child.dtype == 'null') or (
        isinstance(feature, Value) and feature.dtype == 'null')


def benchmark_dataset_options(name, config=''):
    """Return the official benchmark's published source and honest single split."""
    if name != 'geneval2':
        raise ValueError('Choose the available generation benchmark: geneval2.')
    if config not in ('', 'default'):
        raise ValueError('Choose the available benchmark configuration: default.')
    return dict(source='benchmark', name='geneval2', title='GenEval2',
                configs=['default'], config='default', splits=[dict(name='test', count=800)],
                image_columns=[], label_columns=[], label_column_options=[],
                text_columns=['prompt'], prompt_columns=['prompt'],
                suggested_text_column=None, suggested_summary_column=None,
                suggested_prompt_column='prompt', checks_columns=['vqa_list'],
                suggested_checks_column='vqa_list', source_url=GENEVAL2_URL,
                source_revision=GENEVAL2_REVISION)


def hf_dataset_options(name, config=''):
    """Read Hub configuration, split, and feature metadata without loading rows."""
    _validate_hf_name(name)
    if not isinstance(config, str) or len(config) > 200:
        raise ValueError('Choose a valid dataset configuration.')
    config = config.strip()
    try:
        from datasets import get_dataset_config_names, get_dataset_config_info, Image, ClassLabel, Value
    except ImportError as exc:
        raise ValueError(HF_DEPENDENCY_ERROR) from exc
    try:
        configs = get_dataset_config_names(name)
    except Exception as exc:
        raise ValueError(f'Could not read metadata for {name}. Check the dataset ID, access permissions, '
                         f'and connection. {exc}') from exc
    if not configs or any(not isinstance(value, str) or not value for value in configs):
        raise ValueError(f'No dataset configurations could be discovered for {name}. Check its Hugging Face dataset page.')
    if config and config not in configs:
        raise ValueError('Choose an available dataset configuration: ' + ', '.join(configs))
    chosen = config or ('default' if 'default' in configs else configs[0])
    try:
        info = get_dataset_config_info(name, config_name=chosen)
    except Exception as exc:
        raise ValueError(f'Could not read split and column metadata for {name} ({chosen}). '
                         f'Check dataset access and its published metadata. {exc}') from exc
    if not info.splits:
        raise ValueError(f'No splits could be discovered for {name} ({chosen}). Check its published split metadata.')
    if not info.features:
        raise ValueError(f'No column metadata is published for {name} ({chosen}). '
                         'Choose a dataset configuration with published feature metadata.')
    splits = []
    for split, details in info.splits.items():
        count = details.get('num_examples') if isinstance(details, dict) else getattr(details, 'num_examples', None)
        splits.append(dict(name=str(split), count=count if type(count) is int and count >= 0 else None))
    return dict(name=name, configs=list(configs), config=chosen, splits=splits,
                image_columns=[key for key, feature in info.features.items() if isinstance(feature, Image)],
                label_columns=label_columns(info.features),
                label_column_options=[key for key, feature in info.features.items()
                    if isinstance(feature, ClassLabel) or (isinstance(feature, Value)
                        and (feature.dtype in ('string', 'large_string') or re.fullmatch(r'u?int(8|16|32|64)', feature.dtype)))],
                **_text_column_options(info.features))


def _subset_bounds(value):
    if not isinstance(value, str) or len(value) > 200:
        raise ValueError('Subset must be blank, a positive count such as 20, or a range such as [20:30].')
    value = value.strip()
    if not value or value.lower() == 'all':
        return None
    if re.fullmatch(r'[0-9]+', value) and int(value) > 0:
        return 0, int(value)
    match = re.fullmatch(r'\[([0-9]+):([0-9]+)\]', value)
    if match and int(match[1]) < int(match[2]):
        return int(match[1]), int(match[2])
    raise ValueError('Subset must be a positive count such as 20 or a range such as [20:30] '
                     'with a nonnegative start and larger exclusive end. Leave it blank for all rows.')


def load_hf_images(name, split, config, column, label_column='', *, streaming=False):
    try:
        from datasets import load_dataset, Image, ClassLabel, Value
    except ImportError as exc:
        raise ValueError(HF_DEPENDENCY_ERROR) from exc
    dataset = load_dataset(name, name=config or None, split=split, **({'streaming': True} if streaming else {}))
    if streaming:
        from dataset_streaming import bounded_parquet_stream
        dataset = bounded_parquet_stream(dataset)
    image_columns = [key for key, feature in dataset.features.items() if isinstance(feature, Image)]
    if not column and len(image_columns) == 1:
        column = image_columns[0]
    if column not in image_columns:
        raise ValueError('Choose an image column. Available image columns: ' + (', '.join(image_columns) or 'none'))
    # Keep scalar class labels without decoding unrelated audio/video fields.
    columns = [column, *label_columns(dataset.features)]
    if label_column in dataset.features:
        if not isinstance(dataset.features[label_column], (ClassLabel, Value)):
            raise ValueError('Choose a scalar label column containing class names or class labels.')
        columns.append(label_column)
    columns = list(dict.fromkeys(columns))
    return dataset.select_columns(columns).cast_column(column, Image(decode=False)), column


def load_hf_texts(name, split, config, text_column, summary_column='', *, streaming=False):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ValueError(HF_DEPENDENCY_ERROR) from exc
    dataset = load_dataset(name, name=config or None, split=split, **({'streaming': True} if streaming else {}))
    if streaming:
        from dataset_streaming import bounded_parquet_stream
        dataset = bounded_parquet_stream(dataset)
    options = _text_column_options(dataset.features)
    text_column = text_column or options['suggested_text_column']
    summary_column = summary_column or options['suggested_summary_column']
    available = ', '.join(options['text_columns']) or 'none'
    if text_column not in options['text_columns']:
        raise ValueError('Choose a source text column. Available string columns: ' + available)
    if summary_column not in options['text_columns']:
        raise ValueError('Choose a reference summary column. Available string columns: ' + available)
    if text_column == summary_column:
        raise ValueError('Choose different source text and reference summary columns.')
    # Retain Arrow storage and avoid decoding unrelated image, audio, or video fields.
    return dataset.select_columns([text_column, summary_column]), text_column, summary_column


def _select_prompt_columns(dataset, prompt_column, checks_column):
    options = _text_column_options(dataset.features)
    prompt_column = prompt_column or options['suggested_prompt_column']
    if prompt_column not in options['prompt_columns']:
        available = ', '.join(options['prompt_columns']) or 'none'
        raise ValueError('Choose a prompt column. Available string columns: ' + available)
    if checks_column and checks_column not in options['checks_columns']:
        available = ', '.join(options['checks_columns']) or 'none'
        raise ValueError('Choose a checks column containing question/answer pairs or objects. '
                         'Available checks columns: ' + available)
    columns = [prompt_column]
    if checks_column:
        columns.append(checks_column)
        if checks_column == 'vqa_list':
            columns.extend(key for key in ('skills', 'atom_count') if key in dataset.features)
    return dataset.select_columns(columns), prompt_column, checks_column


def load_hf_prompts(name, split, config, prompt_column, checks_column='', *, streaming=False):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ValueError(HF_DEPENDENCY_ERROR) from exc
    dataset = load_dataset(name, name=config or None, split=split, **({'streaming': True} if streaming else {}))
    if streaming:
        from dataset_streaming import bounded_parquet_stream
        dataset = bounded_parquet_stream(dataset)
    return _select_prompt_columns(dataset, prompt_column, checks_column)


def load_benchmark_prompts(name, split, config, prompt_column, checks_column):
    options = benchmark_dataset_options(name, config)
    if split != 'test':
        raise ValueError('GenEval2 publishes one benchmark split: test.')
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ValueError(HF_DEPENDENCY_ERROR) from exc
    dataset = load_dataset('json', data_files={'test': options['source_url']}, split='test')
    return _select_prompt_columns(dataset, prompt_column, checks_column)


def _load_hf_part(name, split, config, column, label_column, task, source, *, streaming=False):
    args = (name, split, config, column)
    options = {'streaming': True} if streaming else {}
    if task == 'summarization':
        return load_hf_texts(*args, summary_column=label_column, **options)
    if task == 'generation':
        loader = load_benchmark_prompts if source == 'benchmark' else load_hf_prompts
        return loader(*args, checks_column=label_column, **options)
    records, column = (load_hf_images(*args, label_column=label_column, **options)
                       if label_column else load_hf_images(*args, **options))
    return records, column, label_column


def _combine_parts(parts, task):
    if len(parts) == 1:
        return parts[0]
    from datasets import concatenate_datasets
    try:
        return concatenate_datasets(parts)
    except (TypeError, ValueError) as exc:
        column_kind = {'summarization': 'source text or reference summary',
                       'generation': 'prompt or checks'}.get(task, 'image or label')
        raise ValueError(f'The selected splits have incompatible {column_kind} columns. '
                         'Choose splits with the same feature definitions.') from exc


def _subset_error(subset, count):
    return ValueError(f'Subset {subset} exceeds the {count} rows in the selected splits. '
                      'Use a count or exclusive end no greater than that total.')


def _materialize_stream_range(records, start, end, image_column=None):
    """Retain just the requested rows in Arrow, without a second disk cache.

    Bounded writer batches avoid retaining a Python list of all selected images.
    Undecoded bytes remain available for random access during Analyze/RL/Adapt.
    """
    import pyarrow as pa
    from datasets import Dataset, DatasetInfo
    from datasets.arrow_writer import ArrowWriter
    from datasets.utils.file_utils import xopen

    buffer = pa.BufferOutputStream()
    writer = ArrowWriter(stream=buffer, features=records.features, writer_batch_size=64)
    iterator = iter(records)
    scanned = 0
    try:
        for index, row in enumerate(islice(iterator, end)):
            scanned = index + 1
            if index < start:
                continue
            if image_column:
                image = row.get(image_column)
                if (isinstance(image, dict) and image.get('bytes') is None
                        and isinstance(image.get('path'), str) and '://' in image['path']):
                    # Image-folder streams may expose remote/archive paths.
                    # Fetch only selected images so later random access is local.
                    with xopen(image['path'], 'rb') as handle:
                        row = {**row, image_column: {'bytes': handle.read(), 'path': None}}
            writer.write(row)
        writer.finalize()
        return Dataset.from_buffer(buffer.getvalue(), info=DatasetInfo(features=records.features)), scanned
    finally:
        close = getattr(iterator, 'close', None)
        if callable(close):
            close()
        writer.close()
        buffer.close()


def _load_hf_subset(name, splits, config, column, subset, label_column, task, options):
    start, end = _subset_bounds(subset)
    counts = {split: None for split in splits}
    if options is not None:
        counts.update({row['name']: row['count'] for row in options['splits'] if row['name'] in counts})
    if all(count is not None for count in counts.values()) and end > sum(counts.values()):
        raise _subset_error(subset, sum(counts.values()))
    parts, offset, selected = [], 0, 0
    for split in splits:
        if offset >= end:
            break
        count = counts[split]
        if count is not None and offset + count <= start:
            offset += count
            continue
        records, column, label_column = _load_hf_part(
            name, split, config, column, label_column, task, 'huggingface', streaming=True)
        info = getattr(records, 'info', None)
        config = config or getattr(info, 'config_name', None) or ''
        if count is None:
            # Streaming datasets normally retain published split sizes. A size
            # can be unknown; never exhaust a stream merely to count its rows.
            details = (getattr(info, 'splits', None) or {}).get(split)
            count = details.get('num_examples') if isinstance(details, dict) else getattr(details, 'num_examples', None)
            if count is None:
                count = getattr(records, 'num_rows', None)
            count = count if type(count) is int and count >= 0 else None
            counts[split] = count
        if count is not None and offset + count <= start:
            offset += count
            continue
        local_end = min(end - offset, count) if count is not None else end - offset
        part, scanned = _materialize_stream_range(
            records, max(0, start - offset), local_end, column if task == 'classification' else None)
        if scanned < local_end and count is not None:
            raise ValueError(f'Dataset split {split} ended at {scanned} rows before its published size; '
                             'refresh dataset options and retry.')
        if count is None and scanned < local_end:
            counts[split] = scanned  # End of this split was actually observed.
        if len(part):
            parts.append(part)
            selected += len(part)
        offset += count if count is not None else scanned
    source_count = sum(counts.values()) if all(value is not None for value in counts.values()) else None
    if selected != end - start:
        raise _subset_error(subset, source_count if source_count is not None else offset)
    return (_combine_parts(parts, task), column, label_column, config, source_count,
            [dict(name=split, count=counts[split]) for split in splits])


def _load_hf_expression_subset(name, selection, config, column, subset, label_column, task):
    """Translate legacy API split expressions before opening any data stream."""
    from datasets import ReadInstruction

    options = hf_dataset_options(name, config)
    config = options['config']
    sizes = {part['name']: part['count'] for part in options['splits']}
    expression = '+'.join(sizes) if selection == 'all' else selection
    try:
        instructions = ReadInstruction.from_spec(expression).to_absolute(sizes)
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError('Cannot resolve the split expression from published split sizes. '
                         'Select split names and use Rows for the count or range instead.') from exc
    if any(item.to < item.from_ for item in instructions):
        raise ValueError('Split expression ranges must have an end no smaller than their start.')
    total = sum(item.to - item.from_ for item in instructions)
    start, end = _subset_bounds(subset)
    if end > total:
        raise _subset_error(subset, total)
    parts, offset = [], 0
    for item in instructions:
        count = item.to - item.from_
        local_start, local_end = max(0, start - offset), min(count, end - offset)
        if local_start < local_end:
            absolute = f'[{item.from_ + local_start}:{item.from_ + local_end}]'
            part, column, label_column, _, _, _ = _load_hf_subset(
                name, [item.splitname], config, column, absolute, label_column, task, options)
            parts.append(part)
        offset += count
        if offset >= end:
            break
    return (_combine_parts(parts, task), column, label_column, config, total,
            [dict(name=selection, count=total)])


def _load_hf_selection(name, selection, config, column, subset, label_column,
                       task='classification', source='huggingface'):
    splits = [selection] if isinstance(selection, str) else selection
    bounds = _subset_bounds(subset)
    options = None
    if isinstance(selection, list):
        options = benchmark_dataset_options(name, config) if source == 'benchmark' else hf_dataset_options(name, config)
        available = [split['name'] for split in options['splits']]
        if any(split not in available for split in splits):
            raise ValueError('Choose available dataset splits: ' + ', '.join(available))
        config = options['config']
    if bounds is not None and source == 'huggingface':
        if isinstance(selection, str) and (selection == 'all' or any(char in selection for char in '[+]')):
            records, column, label_column, config, source_count, split_counts = _load_hf_expression_subset(
                name, selection, config, column, subset, label_column, task)
        else:
            records, column, label_column, config, source_count, split_counts = _load_hf_subset(
                name, splits, config, column, subset, label_column, task, options)
    else:
        parts = []
        for split in splits:
            records, column, label_column = _load_hf_part(name, split, config, column, label_column, task, source)
            parts.append(records)
        records = _combine_parts(parts, task)
        config = config or getattr(getattr(records, 'info', None), 'config_name', None) or ''
        source_count = len(records)
        split_counts = [dict(name=split, count=len(part)) for split, part in zip(splits, parts)]
        if bounds is not None:
            start, end = bounds
            if end > source_count:
                raise _subset_error(subset, source_count)
            records = records.select(range(start, end)) if hasattr(records, 'select') else records[start:end]
    extra = dict(split='+'.join(splits), splits=splits, subset=subset,
                 subset_start=bounds[0] if bounds is not None else 0, source_count=source_count, config=config)
    if task == 'summarization':
        extra.update(text_column=column, summary_column=label_column)
    elif task == 'generation':
        extra.update(prompt_column=column, checks_column=label_column)
    else:
        extra['column'] = column
    if isinstance(selection, list) or task == 'generation':
        extra['split_counts'] = split_counts
    if source == 'benchmark':
        extra.update(source_url=GENEVAL2_URL, source_revision=GENEVAL2_REVISION)
    return records, extra


def _text_row_identity(info, index):
    item = dict(index=index, name=f'Text {index+1}')
    source_index = info.get('subset_start', 0) + index
    for split in info.get('split_counts', []):
        if split['count'] is None or source_index < split['count']:
            item.update(name=f'{split["name"]}[{source_index}]',
                        source_split=split['name'], source_index=source_index)
            break
        source_index -= split['count']
    return item


def _summary_statistics(records, info):
    """Validate every selected row in bounded batches without copying the dataset."""
    references = invalid_texts = 0
    first_error = None
    index = 0
    for batch in records.iter(batch_size=256):
        for text, summary in zip(batch[info['text_column']], batch[info['summary_column']]):
            text_error = None
            if not isinstance(text, str) or not text.strip():
                text_error = 'is empty or missing; choose a range with nonempty source text'
            elif len(text) > MAX_SOURCE_CHARS:
                text_error = f'exceeds the {MAX_SOURCE_CHARS} character limit'
            if text_error:
                invalid_texts += 1
                if first_error is None:
                    name = _text_row_identity(info, index)['name']
                    first_error = f'Source text at row {index+1} ({name}) {text_error}.'
            if isinstance(summary, str) and summary.strip():
                if len(summary) <= MAX_SUMMARY_CHARS:
                    references += 1
                elif first_error is None:
                    name = _text_row_identity(info, index)['name']
                    first_error = f'Reference summary at row {index+1} ({name}) exceeds the {MAX_SUMMARY_CHARS} character limit.'
            index += 1
    return dict(reference_count=references, missing_reference_count=len(records)-references,
                invalid_text_count=invalid_texts), first_error


def _row_prompt_checks(record, info):
    column = info.get('checks_column')
    checks = normalize_prompt_checks(record[column]) if column else []
    if column == 'vqa_list' and 'skills' in record:
        skills = record['skills']
        if not checks and skills is None:
            return checks
        if not isinstance(skills, list) or len(skills) != len(checks):
            raise ValueError('The skills list must have one entry for each prompt check.')
        if any(not isinstance(skill, str) or not skill.strip() for skill in skills):
            raise ValueError('Each prompt-check skill must be nonempty text.')
        checks = normalize_prompt_checks([dict(check, skill=skill) for check, skill in zip(checks, skills)])
    return checks


def _prompt_statistics(records, info):
    """Validate bounded batches while retaining all selected prompts in Arrow."""
    checked = invalid_prompts = invalid_checks = 0
    first_error = None
    index = 0
    for batch in records.iter(batch_size=256):
        for batch_index, prompt in enumerate(batch[info['prompt_column']]):
            name = _text_row_identity(info, index)['name']
            try:
                validate_prompt(prompt)
            except ValueError as exc:
                invalid_prompts += 1
                if first_error is None:
                    first_error = f'Prompt at row {index+1} ({name}) {exc}.'
            record = {key: values[batch_index] for key, values in batch.items()
                      if key in (info.get('checks_column'), 'skills')}
            try:
                checked += bool(_row_prompt_checks(record, info))
            except ValueError as exc:
                invalid_checks += 1
                if first_error is None:
                    first_error = f'Prompt checks at row {index+1} ({name}): {exc}'
            index += 1
    return dict(checked_count=checked, missing_checks_count=len(records)-checked-invalid_checks,
                invalid_prompt_count=invalid_prompts, invalid_checks_count=invalid_checks), first_error


class DatasetManager:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {}
        self.records = {}
        self.label_sets = {}

    def get(self, dataset_id):
        with self.lock:
            return copy.deepcopy(self.jobs[dataset_id])

    def labels(self, dataset_id):
        self.ready(dataset_id)
        with self.lock:
            return copy.deepcopy(self.label_sets.get(dataset_id, dict(column=None, values={},
                message='No dataset labels. Upload JSON or enter labels.')))

    def start(self, data):
        if not isinstance(data, dict) or data.get('source') not in ('folder', 'huggingface', 'upload', 'benchmark'):
            raise ValueError('Choose a local folder, Hugging Face dataset, or generation benchmark.')
        source = data['source']
        task = data.get('task', 'classification')
        if task not in ('classification', 'summarization', 'generation'):
            raise ValueError('Choose classification, summarization, or generation for this dataset.')
        if task == 'summarization' and source != 'huggingface':
            raise ValueError('Choose a Hugging Face dataset for summarization.')
        if task == 'generation' and source not in ('huggingface', 'benchmark'):
            raise ValueError('Choose a Hugging Face prompt dataset or generation benchmark.')
        if source == 'benchmark' and task != 'generation':
            raise ValueError('Choose generation for this benchmark.')
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
        elif source in ('huggingface', 'benchmark'):
            name = data.get('name', '')
            if source == 'huggingface':
                _validate_hf_name(name)
            config = data.get('config', '')
            column_key = {'summarization': 'text_column', 'generation': 'prompt_column'}.get(task, 'column')
            label_key = {'summarization': 'summary_column', 'generation': 'checks_column'}.get(task, 'label_column')
            column = data.get(column_key, 'prompt' if source == 'benchmark' else '')
            label_column = data.get(label_key, 'vqa_list' if source == 'benchmark' else '')
            if any(not isinstance(value, str) or len(value) > 200 for value in (config, column, label_column)):
                fields = {'summarization': 'source text column, and reference summary column',
                          'generation': 'prompt column, and checks column'}.get(task, 'image column, and label column')
                raise ValueError(f'Enter a valid configuration, {fields}.')
            if 'splits' in data:
                if 'split' in data:
                    raise ValueError('Use either selected splits or a legacy split expression, not both.')
                selection = data['splits']
                if not isinstance(selection, list) or not selection or any(
                        not isinstance(value, str) or not value.strip() or len(value) > 200 for value in selection):
                    raise ValueError('Select one or more available dataset splits.')
                selection = [value.strip() for value in selection]
                if len(set(selection)) != len(selection):
                    raise ValueError('Choose each dataset split only once.')
            else:
                selection = data.get('split', 'test' if source == 'benchmark' else 'train')
                if not isinstance(selection, str) or not selection.strip() or len(selection) > 200:
                    raise ValueError('Enter a valid dataset split.')
                selection = selection.strip()
            if source == 'benchmark':
                benchmark_dataset_options(name, config.strip())
                if selection not in ('test', ['test']):
                    raise ValueError('GenEval2 publishes one benchmark split: test.')
                if column.strip() not in ('', 'prompt') or label_column.strip() not in ('', 'vqa_list'):
                    raise ValueError('GenEval2 uses the prompt and vqa_list columns.')
                column, label_column = 'prompt', 'vqa_list'
            subset = data.get('subset', '')
            _subset_bounds(subset)
            args = (name, selection, config.strip(), column.strip(), subset.strip(), label_column.strip())
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
            job = dict(id=dataset_id, source=source, task=task, name=name, status='uploading' if source == 'upload' else 'loading', count=0,
                       label_count=0, label_column=None, label_message='No dataset labels. Upload JSON or enter labels.')
            if source in ('huggingface', 'benchmark'):
                job.update(splits=[args[1]] if isinstance(args[1], str) else args[1], subset=args[4], config=args[2])
            if task == 'summarization':
                job.update(text_column=args[3], summary_column=args[5], reference_count=0,
                           missing_reference_count=0, invalid_text_count=0)
            elif task == 'generation':
                job.update(prompt_column=args[3], checks_column=args[5], checked_count=0,
                           missing_checks_count=0, invalid_prompt_count=0, invalid_checks_count=0)
                if source == 'benchmark':
                    job.update(source_url=GENEVAL2_URL, source_revision=GENEVAL2_REVISION)
            self.jobs[dataset_id] = job
            if source == 'upload':
                self.records[dataset_id] = args[0]
                job['expected_count'] = len(args[0])
            else:
                threading.Thread(target=self.load, args=(dataset_id, source, args), daemon=True).start()
            return copy.deepcopy(job)

    def load(self, dataset_id, source, args):
        try:
            task = self.get(dataset_id).get('task', 'classification')
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
                records, extra = _load_hf_selection(*args, task=task, source=source)
            if task in ('summarization', 'generation'):
                statistics, error = (_summary_statistics if task == 'summarization' else _prompt_statistics)(records, extra)
                with self.lock:
                    self.jobs[dataset_id].update(count=len(records), **extra, **statistics)
                    if not len(records):
                        raise ValueError('No text rows were found in the selected dataset splits.')
                    if error:
                        raise ValueError(error)
                    self.records[dataset_id] = records
                    self.jobs[dataset_id]['status'] = 'ready'
                return
            if not len(records):
                raise ValueError('No supported images were found in this folder or dataset split.')
            labels = extract_dataset_labels(records, column=args[5] or None) if source == 'huggingface' else extract_dataset_labels(records)
            with self.lock:
                self.records[dataset_id] = records
                self.label_sets[dataset_id] = labels
                self.jobs[dataset_id].update(status='ready', count=len(records), label_count=len(labels['values']),
                    label_column=labels['column'], label_message=labels['message'],
                    label_columns=labels.get('available_columns', []), label_diagnostic=labels.get('diagnostic', ''), **extra)
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
        if info.get('task') == 'generation':
            items = []
            for index in range(offset, min(offset+limit, info['count'])):
                row = records[index]
                items.append(dict(**_text_row_identity(info, index),
                                  text_preview=row[info['prompt_column']][:200],
                                  check_count=len(_row_prompt_checks(row, info))))
            return dict(items=items, count=info['count'], offset=offset)
        if info.get('task') == 'summarization':
            items = []
            for index in range(offset, min(offset+limit, info['count'])):
                row = records[index]
                summary = row[info['summary_column']]
                items.append(dict(**_text_row_identity(info, index),
                                  text_preview=row[info['text_column']][:200],
                                  has_reference=isinstance(summary, str) and bool(summary.strip())))
            return dict(items=items, count=info['count'], offset=offset)
        items = [dict(index=i, name=records[i]['name'] if info['source'] != 'huggingface' else f'Image {i+1}')
                 for i in range(offset, min(offset+limit, info['count']))]
        labels = self.label_sets.get(dataset_id, {}).get('values', {})
        for item in items:
            if info['source'] == 'huggingface' and info.get('split_counts'):
                source_index = info['subset_start'] + item['index']
                for split in info['split_counts']:
                    if split['count'] is None or source_index < split['count']:
                        item.update(name=f'{split["name"]}[{source_index}]',
                                    source_split=split['name'], source_index=source_index)
                        break
                    source_index -= split['count']
            if str(item['index']) in labels:
                item['label'] = labels[str(item['index'])]
        return dict(items=items, count=info['count'], offset=offset)

    def text(self, dataset_id, index):
        info = self.ready(dataset_id)
        if info.get('task') not in ('summarization', 'generation'):
            raise ValueError('Choose a summarization or generation text dataset.')
        if type(index) is not int or not 0 <= index < info['count']:
            raise ValueError('Text index is outside the selected dataset.')
        record = self.records[dataset_id][index]
        if info.get('task') == 'generation':
            return dict(**_text_row_identity(info, index), text=record[info['prompt_column']],
                        checks=_row_prompt_checks(record, info))
        summary = record[info['summary_column']]
        return dict(**_text_row_identity(info, index), text=record[info['text_column']],
                    summary=summary if isinstance(summary, str) and summary.strip() else None)

    def _image_source(self, dataset_id, index):
        info = self.ready(dataset_id)
        if info.get('task', 'classification') != 'classification':
            raise ValueError('Choose an image dataset for classification.')
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
        return source

    def image_identity(self, dataset_id, index):
        """Identify the bytes behind a row without decoding or copying the image."""
        source = self._image_source(dataset_id, index)
        if isinstance(source, io.BytesIO):
            return ('bytes', hashlib.sha256(source.getbuffer()).hexdigest())
        path = Path(source).expanduser().resolve()
        try:
            info = path.stat()
        except OSError as exc:
            raise ValueError(f'The source for image {index+1} changed or is unavailable. '
                             'Reload the dataset and Analyze again.') from exc
        return ('path', str(path), info.st_dev, info.st_ino, info.st_size,
                info.st_mtime_ns, info.st_ctime_ns)

    def image(self, dataset_id, index):
        from PIL import Image, ImageOps
        source = self._image_source(dataset_id, index)
        try:
            with Image.open(source) as image:
                if image.width * image.height > 20_000_000:
                    raise ValueError('Use images under 20 megapixels.')
                return ImageOps.exif_transpose(image).convert('RGB').resize((224, 224))
        except (OSError, TypeError) as exc:
            raise ValueError(f'Could not read image {index+1}: {exc}') from exc
