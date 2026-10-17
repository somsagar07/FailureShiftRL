"""Input representations from selected summarization and diffusion models.

These adapters never generate summaries or images. Diffusion loading is limited
to the pipeline manifest, tokenizer, and CLIP text encoder.
"""
import json
import math
import os
import re
from pathlib import Path

from model_loading import configure_parallel_loading


def _selection(model, task):
    if not isinstance(model, dict) or model.get('source') not in ('local', 'huggingface'):
        raise ValueError('Select a local or Hugging Face model for text embeddings.')
    if model.get('task', task) != task:
        raise ValueError(f'This text encoder requires the {task} task.')
    name = model.get('name')
    if not isinstance(name, str) or not name.strip() or len(name) > 1024:
        raise ValueError('Select a model before computing text embeddings.')
    name = name.strip()
    if model['source'] == 'local':
        path = Path(name).expanduser().resolve()
        if not path.is_dir():
            raise ValueError('Text embeddings require a local save_pretrained model or pipeline directory, not a checkpoint file.')
        required = 'model_index.json' if task == 'generation' else 'config.json'
        if not (path/required).is_file():
            raise ValueError(f'The local model directory must contain {required}.')
        name = str(path)
    elif not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)?', name) or '..' in name:
        raise ValueError('Use a Hugging Face model ID, for example google-t5/t5-small.')
    revision = model.get('revision')
    if revision is not None and (not isinstance(revision, str) or not revision.strip() or len(revision) > 256):
        raise ValueError('The model revision must be a branch, tag, or commit name.')
    return dict(source=model['source'], name=name, revision=revision)


def _dependencies():
    configure_parallel_loading()
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ValueError('Text embedding dependencies are missing. Stop the server and run bash run.sh from web_interface.') from exc
    version = re.match(r'^(\d+)\.(\d+)', torch.__version__)
    if not version or tuple(map(int, version.groups())) < (2, 6):
        raise ValueError('Text embeddings require PyTorch 2.6 or newer for safe weights-only loading.')
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    return torch, transformers


def _encoder_device(torch, device):
    if not isinstance(device, str) or device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Choose an analysis device: auto, cpu, or cuda.')
    if device == 'cpu':
        return 'cpu'
    available = torch.cuda.is_available()
    if device == 'cuda' and not available:
        raise ValueError('CUDA is unavailable. Choose CPU, or install CUDA-enabled PyTorch with a compatible NVIDIA driver.')
    if available:
        # Float32 tensors alone do not prevent cuDNN's default TF32 kernels.
        # Keep the original full-precision feature definition on the GPU too.
        if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
            torch.backends.cuda.matmul.fp32_precision = 'ieee'
            torch.backends.cudnn.conv.fp32_precision = 'ieee'
        else:  # PyTorch 2.6–2.8 use the original precision controls.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    return 'cuda' if available else 'cpu'


def _options(selection, revision=None):
    options = {'trust_remote_code': False, 'local_files_only': selection['source'] == 'local'}
    revision = revision or selection.get('revision')
    if revision:
        options['revision'] = revision
    return options


def _resolved_revision(selection, commit):
    if selection['source'] == 'local':
        return None
    commit = commit or selection.get('revision')
    if not isinstance(commit, str) or not re.fullmatch(r'[0-9a-fA-F]{40}', commit):
        raise ValueError('Could not resolve an immutable model revision; tokenizer and weights must use the same checkpoint.')
    return commit


def _pretrained(loaded, info, torch):
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError('The checkpoint is missing compatible pretrained weights; newly initialized features are not supported.')
    return loaded.to(device='cpu', dtype=torch.float32).eval()


def _token_limit(tokenizer, *configs, default=2048):
    # Generation max_length is an output limit and must never constrain inputs.
    candidates = [getattr(tokenizer, 'model_max_length', None)]
    for config in configs:
        for name in ('max_position_embeddings', 'max_encoder_position_embeddings', 'max_source_positions', 'n_positions'):
            candidates.append(getattr(config, name, None))
    limits = [int(value) for value in candidates if isinstance(value, (int, float)) and
              math.isfinite(value) and 0 < value < 10**9 and int(value) == value]
    # Tokenizers without a declared limit use a huge integer sentinel. Bound
    # inference even for relative-position models that permit arbitrary lengths.
    return min(2048, min(limits) if limits else default)


class _TextEncoder:
    def set_device(self, device):
        device = _encoder_device(self.torch, device)
        self.model.to(device=device, dtype=self.torch.float32)
        self.device = device
        return self

    def _prepare(self, texts, prefix=''):
        if not isinstance(texts, list) or not texts or not all(isinstance(text, str) for text in texts):
            raise ValueError('Provide a nonempty list of text inputs.')
        inputs = self.tokenizer([prefix+text for text in texts], padding=True, truncation=False,
                                return_tensors='pt', return_attention_mask=True,
                                return_token_type_ids=False, verbose=False)
        if 'input_ids' not in inputs or 'attention_mask' not in inputs:
            raise ValueError('The selected tokenizer must return input IDs and an attention mask.')
        ids, mask = inputs['input_ids'], inputs['attention_mask']
        if ids.ndim != 2 or mask.shape != ids.shape or ids.shape[0] != len(texts):
            raise ValueError('The selected tokenizer returned invalid text inputs.')
        for index, length in enumerate(mask.sum(dim=1).tolist()):
            if not length:
                raise ValueError(f'Text input {index+1} has no tokens.')
            if length > self.max_tokens:
                raise ValueError(f'Text input {index+1} has {length} tokens, exceeding this model\'s supported input limit of '
                                 f'{self.max_tokens} tokens (including the task prefix and special tokens). Shorten the input; truncation is disabled.')
        return {'input_ids': ids.to(self.device), 'attention_mask': mask.to(self.device)}

    def _pool(self, output, mask):
        torch = self.torch
        hidden = getattr(output, 'last_hidden_state', None)
        if not torch.is_tensor(hidden) or hidden.ndim != 3 or hidden.shape[:2] != mask.shape or not hidden.shape[2]:
            raise ValueError('The selected encoder must return final token last_hidden_state features.')
        weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
        features = (hidden*weights).sum(dim=1)/weights.sum(dim=1)
        if not torch.isfinite(features).all():
            raise ValueError('The selected encoder returned nonfinite text features.')
        return features.detach().to(device='cpu', dtype=torch.float64).numpy()

    def _configure_tokenizer(self):
        if self.tokenizer.pad_token_id is None:
            raise ValueError('The selected tokenizer needs a padding token for text embeddings.')
        # CLIP's absolute positions and causal mask require right padding for a
        # prompt's non-padding states to be independent of other batch members.
        self.tokenizer.padding_side = 'right'


class SummarizationEncoder(_TextEncoder):
    """Mask-mean final encoder states from an encoder-decoder summarizer."""
    def __init__(self, model):
        selection = _selection(model, 'summarization')
        self.torch, transformers = _dependencies()
        self.device = _encoder_device(self.torch, model.get('device', 'cpu'))
        try:
            config = transformers.AutoConfig.from_pretrained(selection['name'], **_options(selection))
            if not config.is_encoder_decoder:
                raise ValueError('Summarization embeddings require an encoder-decoder sequence-to-sequence model.')
            self.revision = _resolved_revision(selection, getattr(config, '_commit_hash', None))
            options = _options(selection, self.revision)
            loaded, info = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                selection['name'], config=config, weights_only=True, output_loading_info=True, **options)
            self.model = _pretrained(loaded, info, self.torch)
            self.set_device(self.device)
            self.encoder = self.model.get_encoder()
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(selection['name'], **options)
            self._configure_tokenizer()
            tasks = getattr(config, 'task_specific_params', None) or {}
            summary = tasks.get('summarization') or {}
            self.prefix = summary.get('prefix', getattr(config, 'prefix', None))
            is_t5 = config.model_type in ('t5', 'mt5', 'longt5', 'umt5')
            if self.prefix is None:
                self.prefix = 'summarize: ' if is_t5 else ''
            if not isinstance(self.prefix, str):
                raise ValueError('The summarization task prefix must be text.')
            self.max_tokens = _token_limit(self.tokenizer, config, getattr(self.encoder, 'config', None),
                                           default=512 if is_t5 else 2048)
        except Exception as exc:
            raise ValueError(f'Could not load summarization model/tokenizer and pretrained weights {selection["name"]}: {exc}') from exc
        self.feature_method = 'encoder last_hidden_state (non-padding token mean)'
        if self.prefix:
            self.feature_method += f'; summarization prefix {self.prefix!r}'

    def encode(self, texts):
        try:
            inputs = self._prepare(texts, self.prefix)
            with self.torch.inference_mode():
                output = self.encoder(**inputs, return_dict=True)
                return self._pool(output, inputs['attention_mask'])
        except Exception as exc:
            raise ValueError(f'Summarization input feature extraction failed: {exc}') from exc


class DiffusionPromptEncoder(_TextEncoder):
    """Stable Diffusion 1/2 CLIP conditioning features; no diffusion model load."""
    def __init__(self, model):
        selection = _selection(model, 'generation')
        self.torch, transformers = _dependencies()
        self.device = _encoder_device(self.torch, model.get('device', 'cpu'))
        try:
            if selection['source'] == 'local':
                index_path = Path(selection['name'])/'model_index.json'
                self.revision = None
            else:
                from huggingface_hub import hf_hub_download
                from transformers.utils.hub import extract_commit_hash
                index_path = Path(hf_hub_download(repo_id=selection['name'], filename='model_index.json',
                                                 revision=selection.get('revision')))
                self.revision = _resolved_revision(selection, extract_commit_hash(str(index_path), None))
            index = json.loads(index_path.read_text(encoding='utf-8'))
            if not isinstance(index, dict) or index.get('_class_name') != 'StableDiffusionPipeline':
                raise ValueError('Only Stable Diffusion 1/2 StableDiffusionPipeline prompt embeddings are supported; XL, SD3, FLUX, and custom pipelines are unsupported.')
            if index.get('text_encoder') != ['transformers', 'CLIPTextModel']:
                raise ValueError('The Stable Diffusion pipeline must declare a Transformers CLIPTextModel text_encoder.')
            tokenizers = {'CLIPTokenizer': transformers.CLIPTokenizer, 'CLIPTokenizerFast': transformers.CLIPTokenizerFast}
            tokenizer = index.get('tokenizer')
            if not isinstance(tokenizer, list) or len(tokenizer) != 2 or tokenizer[0] != 'transformers' or tokenizer[1] not in tokenizers:
                raise ValueError('The Stable Diffusion pipeline must declare a Transformers CLIP tokenizer.')
            if any(value not in (None, [None, None]) and key.startswith(('text_encoder_', 'tokenizer_'))
                   for key, value in index.items()):
                raise ValueError('Stable Diffusion pipelines with additional text encoders are unsupported.')
            options = _options(selection, self.revision)
            config = transformers.AutoConfig.from_pretrained(selection['name'], subfolder='text_encoder', **options)
            if not isinstance(config, transformers.CLIPTextConfig):
                raise ValueError('The declared text_encoder must contain a compatible CLIP text configuration.')
            loaded, info = transformers.CLIPTextModel.from_pretrained(
                selection['name'], subfolder='text_encoder', config=config,
                weights_only=True, output_loading_info=True, **options)
            self.model = _pretrained(loaded, info, self.torch)
            self.set_device(self.device)
            self.tokenizer = tokenizers[tokenizer[1]].from_pretrained(selection['name'], subfolder='tokenizer', **options)
            if tokenizer[1] == 'CLIPTokenizer' and not callable(getattr(self.tokenizer, 'fix_text', None)):
                raise ValueError('CLIP tokenization requires ftfy for consistent text normalization. Run bash setup.sh from web_interface.')
            self._configure_tokenizer()
            self.max_tokens = _token_limit(self.tokenizer, config, default=77)
        except Exception as exc:
            raise ValueError(f'Could not load diffusion prompt tokenizer and pretrained weights {selection["name"]}: {exc}') from exc
        self.feature_method = 'CLIP last_hidden_state (non-padding token mean; prompt conditioning)'

    def encode(self, texts):
        try:
            inputs = self._prepare(texts)
            # Match StableDiffusionPipeline.encode_prompt's attention-mask rule.
            mask = inputs['attention_mask'] if getattr(self.model.config, 'use_attention_mask', False) else None
            with self.torch.inference_mode():
                output = self.model(input_ids=inputs['input_ids'], attention_mask=mask, return_dict=True)
                return self._pool(output, inputs['attention_mask'])
        except Exception as exc:
            raise ValueError(f'Diffusion prompt feature extraction failed: {exc}') from exc
