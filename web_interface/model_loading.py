"""Bounded checkpoint loading and selective preparation for Analyze.

Preparation includes Hub metadata, cache resolution and downloads. It does not
claim to measure network transfer separately from those operations.
"""
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path, PurePosixPath


_COMMIT = re.compile(r'[0-9a-fA-F]{40}')
_METADATA = {
    'config.json', 'generation_config.json', 'preprocessor_config.json',
    'processor_config.json', 'feature_extractor_config.json',
    'tokenizer_config.json', 'tokenizer.json', 'special_tokens_map.json',
    'added_tokens.json', 'vocab.json', 'vocab.txt', 'merges.txt',
    'chat_template.jinja',
}


def configure_parallel_loading():
    """Enable Transformers shard workers without overriding operator choices."""
    os.environ.setdefault('HF_ENABLE_PARALLEL_LOADING', 'true')
    os.environ.setdefault('HF_PARALLEL_LOADING_WORKERS', '4')


def _safe_relative_path(value):
    return (isinstance(value, str) and bool(value) and len(value) <= 1024
            and not any(character in '\\:*?[]' or ord(character) < 32 for character in value)
            and all(part not in ('', '.', '..') for part in value.split('/')))


def _validate_selection(model):
    if not isinstance(model, dict) or model.get('source') not in ('local', 'huggingface'):
        raise ValueError('Select a local or Hugging Face model.')
    task = model.get('task', 'classification')
    if task not in ('classification', 'summarization', 'generation'):
        raise ValueError('Choose an available workspace domain.')
    name = model.get('name')
    if not isinstance(name, str) or not name.strip() or len(name) > 1024:
        raise ValueError('Select a model before preparing Analyze.')
    if model['source'] == 'huggingface' and (
            not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)?', name)
            or '..' in name):
        raise ValueError('Use a Hugging Face model ID, for example google-t5/t5-small.')
    revision = model.get('revision')
    if revision is not None and (not _safe_relative_path(revision) or not revision.strip() or len(revision) > 256):
        raise ValueError('The model revision must be a branch, tag, or commit name.')
    if model['source'] == 'local':
        root = Path(name).expanduser()
        required = 'model_index.json' if task == 'generation' else 'config.json'
        if not root.is_dir() or not (root/required).is_file():
            raise ValueError(f'The local model directory must contain {required}.')
    return task


def _checkpoint_files(files, prefix, read_json):
    # Prefer exactly one format; indexes alone are not usable checkpoints.
    for filename, suffix in [('model.safetensors', '.safetensors'),
                             ('model.safetensors.index.json', '.safetensors'),
                             ('pytorch_model.bin', '.bin'),
                             ('pytorch_model.bin.index.json', '.bin')]:
        name = prefix + filename
        if name not in files:
            continue
        if not filename.endswith('.index.json'):
            return {name}
        index = read_json(name)
        mapping = index.get('weight_map') if isinstance(index, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError('The checkpoint index needs a nonempty weight_map.')
        selected = {name}
        for shard in mapping.values():
            if (not isinstance(shard, str) or not re.fullmatch(r'[A-Za-z0-9_./-]+', shard)
                    or shard.startswith('/') or any(part in ('.', '..') for part in shard.split('/'))
                    or not shard.endswith(suffix) or prefix + shard not in files):
                raise ValueError('The checkpoint index references a missing or unsupported shard.')
            selected.add(prefix + shard)
        return selected
    raise ValueError('Analyze requires a standard PyTorch pretrained checkpoint (safetensors or pytorch_model.bin).')


def _analysis_files(files, task, read_json):
    prefix = 'text_encoder/' if task == 'generation' else ''
    required = {'model_index.json', 'text_encoder/config.json'} if task == 'generation' else {'config.json'}
    if not required.issubset(files):
        raise ValueError('The checkpoint is missing its required model configuration.')
    selected = required | _checkpoint_files(files, prefix, read_json)
    for filename in files:
        path = PurePosixPath(filename)
        parent = str(path.parent)
        allowed_parent = parent in ('text_encoder', 'tokenizer') if task == 'generation' else parent == '.'
        if allowed_parent and (path.name in _METADATA or path.suffix in ('.model', '.spm', '.tiktoken')):
            selected.add(filename)
    # Transformers can redirect its standard files to versioned JSON assets.
    # Follow those declared names, retaining their component prefix so prompt
    # preparation cannot reach UNet/VAE files or executable remote code.
    for directory in ('text_encoder/', 'tokenizer/') if task == 'generation' else ('',):
        for filename, field in [('config.json', 'configuration_files'),
                                ('tokenizer_config.json', 'fast_tokenizer_files')]:
            name = directory + filename
            if name not in selected:
                continue
            metadata = read_json(name)
            if not isinstance(metadata, dict):
                raise ValueError('Model/tokenizer metadata must be a JSON object.')
            references = metadata.get(field, [])
            if not isinstance(references, list):
                raise ValueError('Referenced model/tokenizer metadata must be a list of JSON filenames.')
            for reference in references:
                if (not _safe_relative_path(reference) or not reference.endswith('.json')
                        or directory + reference not in files):
                    raise ValueError('Model/tokenizer metadata references a missing or unsupported relative JSON file.')
                selected.add(directory + reference)
    return sorted(selected)


def _cache_revision(snapshot, requested, revision):
    """Remember the resolved ref without requesting a mutable revision again."""
    reference = requested or 'main'
    if reference == revision:
        return
    if snapshot.parent.name != 'snapshots' or snapshot.name != revision:
        raise ValueError('The prepared snapshot does not match its immutable model revision.')
    storage = snapshot.parent.parent.resolve()
    refs = storage/'refs'
    target = refs/reference
    # Reject symlink redirects as well as traversal in the requested name.
    if not _safe_relative_path(reference) or not target.resolve().is_relative_to(refs):
        raise ValueError('The model revision must stay within its Hub cache refs directory.')
    temporary = None
    try:
        if target.is_file() and target.read_text(encoding='utf-8') == revision:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent,
                                         prefix='.ref-', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(revision)
        os.replace(temporary, target)
    except OSError as exc:
        # Match Hub cache behavior for read-only caches: valid weights remain
        # usable, even if this process cannot remember a new branch reference.
        logging.getLogger(__name__).warning('Could not cache model revision %s: %s', reference, exc)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_analysis_model(model):
    """Return local Analyze assets and an immutable revision with preparation time.

    Generation only prepares its manifest, CLIP text encoder and tokenizer. The
    original Hub identity belongs to the caller, which restores the resolved
    revision on the encoder created from the returned local snapshot.
    """
    started = time.perf_counter()
    task = _validate_selection(model)
    configure_parallel_loading()
    if model['source'] == 'local':
        return dict(model), {'revision': model.get('revision'),
                             'preparation_seconds': time.perf_counter() - started, 'files_reused': True}

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    repo_id, requested = model['name'], model.get('revision')
    cached = None
    try:
        info = HfApi().model_info(repo_id, revision=requested)
    except Exception as lookup_error:
        # Hub's local-only resolution knows branch/tag refs as well as commit
        # paths. Validate the required files below: a snapshot can be partial.
        try:
            cached = Path(snapshot_download(repo_id, revision=requested, local_files_only=True, max_workers=4))
        except Exception:
            raise lookup_error
        revision = cached.name
        if cached.parent.name != 'snapshots' or not _COMMIT.fullmatch(revision):
            raise ValueError('Could not resolve an immutable cached model revision.') from lookup_error
        files = {path.relative_to(cached).as_posix() for path in cached.rglob('*') if path.is_file()}
    else:
        revision = info.sha
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError('Could not resolve an immutable model revision.')
        files = {entry.rfilename for entry in info.siblings or []}
        if not files:
            files = set(HfApi().list_repo_files(repo_id, revision=revision))

    def read_json(filename):
        path = cached/filename if cached is not None else Path(hf_hub_download(repo_id, filename, revision=revision))
        return json.loads(path.read_text(encoding='utf-8'))

    selected = _analysis_files(files, task, read_json)
    snapshot = cached or Path(snapshot_download(repo_id, revision=revision, allow_patterns=selected, max_workers=4))
    if any(not (snapshot/filename).is_file() for filename in selected):
        raise ValueError('The prepared checkpoint is incomplete; required Analyze files are missing.')
    if cached is None:
        _cache_revision(snapshot, requested, revision)
    metadata = {'revision': revision, 'preparation_seconds': time.perf_counter() - started}
    if cached is not None:
        metadata['files_reused'] = True
    return {**model, 'source': 'local', 'name': str(snapshot), 'revision': revision}, metadata
