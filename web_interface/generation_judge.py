"""Local visual-question checks for generated images; scores remain review signals."""
import copy
from pathlib import Path
import re
import unicodedata

from text_encoders import _dependencies, _options, _resolved_revision, _selection, _token_limit


DEFAULT_JUDGE = 'CountingSheep/vev-4b'
COMBINED_JUDGES = (DEFAULT_JUDGE, 'Qwen/Qwen3-VL-2B-Instruct')
JUDGE_MODELS = (*COMBINED_JUDGES, 'Salesforce/blip-vqa-base')
_NUMBERS = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split(), map(str, range(21))))


def create_judge(selection, device):
    local_vev=(selection.get('source')=='local' and isinstance(selection.get('name'),str)
               and (Path(selection['name']).expanduser()/'vev.json').is_file())
    if selection.get('name') == DEFAULT_JUDGE or local_vev:
        from vev_judge import VevJudge
        return VevJudge(selection, device)
    return QuestionJudge(selection, device)


def normalize_answer(value):
    """Compare complete short answers; never reward an answer substring."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError('The visual judge returned an empty or invalid answer.')
    value = unicodedata.normalize('NFKC', value).casefold().strip()
    value = re.sub(r'^[\s\"\'.,!?;:]+|[\s\"\'.,!?;:]+$', '', value)
    value = ' '.join(value.split())
    if not value:
        raise ValueError('The visual judge returned an empty answer.')
    return _NUMBERS.get(value, value)


def check_answers(checks, answers):
    if not checks or not isinstance(answers, list) or len(checks) != len(answers):
        raise ValueError('The visual judge must return one answer for every requirement.')
    rows = []
    for check, answer in zip(checks, answers):
        passed = normalize_answer(answer) == normalize_answer(check['answer'])
        rows.append(dict(question=check['question'], expected=check['answer'], answer=answer,
                         passed=passed, **({'skill': check['skill']} if check.get('skill') else {})))
    count = sum(row['passed'] for row in rows)
    return dict(checks=rows, checks_passed=count, checks_total=len(rows), quality=count / len(rows))


class QuestionJudge:
    """An independent pretrained judge. Expected answers never enter its prompt."""

    def __init__(self, selection, device, *, batch_size=None, batch_token_limit=4096):
        self.torch, transformers = _dependencies()
        self.device = device
        self.batch_size = (4 if device == 'cuda' else 1) if batch_size is None else batch_size
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError('The visual judge batch size must be a positive integer.')
        self.batch_token_limit = batch_token_limit
        if type(batch_token_limit) is not int or batch_token_limit < 1:
            raise ValueError('The visual judge batch token limit must be a positive integer.')
        # The judge is a Transformers checkpoint, independent of the diffusion
        # pipeline's model_index.json contract.
        selection = _selection(selection, 'summarization')
        config = transformers.AutoConfig.from_pretrained(selection['name'], **_options(selection))
        revision = _resolved_revision(selection, getattr(config, '_commit_hash', None))
        options = _options(selection, revision)
        self.kind = config.model_type
        if self.kind == 'blip':
            cls, processor = transformers.BlipForQuestionAnswering, transformers.BlipProcessor
        elif self.kind == 'qwen3_vl':
            cls, processor = transformers.Qwen3VLForConditionalGeneration, transformers.AutoProcessor
        else:
            raise ValueError('Visual checks support Qwen3-VL and BLIP question-answering checkpoints.')
        self.dtype = self.torch.float16 if device == 'cuda' else self.torch.float32
        loaded, info = cls.from_pretrained(selection['name'], config=config, weights_only=True,
            output_loading_info=True, dtype=self.dtype, **options)
        if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
            raise ValueError('The visual judge checkpoint is missing compatible pretrained weights.')
        self.model = loaded.to(device=device, dtype=self.dtype).eval()
        self.processor = processor.from_pretrained(selection['name'], use_fast=self.kind == 'qwen3_vl', **options)
        if self.kind == 'qwen3_vl':
            # Transformers 4.57 selects logits from the final prompt column.
            # Right padding therefore changes decoder-only batched answers.
            self.processor.tokenizer.padding_side = 'left'
        self.max_tokens = _token_limit(self.processor.tokenizer, getattr(config, 'text_config', config))
        self.identity = dict(name=selection['name'], revision=revision, architecture=self.kind)

    def answer(self, image, question):
        return self.answer_many(image, [question])[0]

    def _prepare(self, image, questions):
        if self.kind == 'blip':
            inputs = self.processor(images=[image] * len(questions), text=questions,
                                    return_tensors='pt', padding=True, truncation=False)
        else:
            texts = []
            for question in questions:
                messages = [{'role': 'user', 'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': 'Answer the question about the image. Use only a short answer '
                     '(yes, no, a number, or a few words), without explanation.\n' + question}]}]
                texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            # Each question has its own prompt and image. No expected answers,
            # paths, URLs, or other questions are included as context.
            inputs = self.processor(text=texts, images=[image] * len(questions),
                                    return_tensors='pt', padding=True, truncation=False)
        if inputs['input_ids'].shape[-1] + 16 > self.max_tokens:
            raise ValueError('A visual-check question exceeds the judge token limit; shorten the question.')
        return inputs

    def _batch_count(self, prepared):
        count, width = prepared['input_ids'].shape
        if self.kind == 'blip':
            # BLIP 4.57's decoder cross-attends to every question embedding,
            # including padding. Equal lengths preserve single-question behavior.
            lengths = prepared['attention_mask'].sum(dim=1).tolist()
            count = next((index for index, length in enumerate(lengths) if length != lengths[0]), count)
        count = min(count, max(1, self.batch_token_limit // (width + 16)))
        if self.device == 'cuda' and count > 1:
            try:
                free, _ = self.torch.cuda.mem_get_info(self.device)
            except RuntimeError:
                return 1
            config = getattr(self.model.config, 'text_config', self.model.config)
            heads = getattr(config, 'num_attention_heads', 1)
            head_dim = getattr(config, 'head_dim', None) or config.hidden_size // heads
            cache_bytes = (2 * config.num_hidden_layers * getattr(config, 'num_key_value_heads', heads)
                           * head_dim * (width + 16) * self.torch.empty((), dtype=self.dtype).element_size())
            input_bytes = sum(value.numel() * value.element_size() for value in prepared.values())
            # Leave a quarter of free VRAM (at least 256 MiB) for other work.
            # The estimate covers KV storage plus input/workspace headroom;
            # actual model allocation failures still reduce the lifetime limit.
            reserve = max(256 * 1024**2, free // 4)
            per_question = max(256 * 1024**2, 2 * cache_bytes + 4 * input_bytes // prepared['input_ids'].shape[0])
            count = min(count, max(1, (free - reserve) // per_question))
        return count

    def _generate(self, prepared):
        inputs, output, tokens = {}, None, None
        try:
            for key, value in prepared.items():
                inputs[key] = value.to(device=self.device, dtype=self.dtype if value.is_floating_point() else value.dtype)
            generation = copy.deepcopy(self.model.generation_config)
            generation.do_sample = False
            generation.num_beams = 1
            generation.num_beam_groups = 1
            generation.num_return_sequences = 1
            generation.min_length = 0
            generation.min_new_tokens = 0
            generation.max_new_tokens = 16
            generation.early_stopping = False
            generation.length_penalty = 1.
            generation.return_dict_in_generate = False
            generation.output_scores = False
            generation.output_attentions = False
            generation.output_hidden_states = False
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, generation_config=generation)
            count = inputs['input_ids'].shape[0]
            if not self.torch.is_tensor(output) or output.ndim != 2 or output.shape[0] != count:
                raise ValueError('The visual judge must return one answer for every question.')
            tokens = output[:, inputs['input_ids'].shape[-1]:] if self.kind == 'qwen3_vl' else output
            decoded = self.processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)
            if not isinstance(decoded, list) or len(decoded) != count:
                raise ValueError('The visual judge must return one answer for every question.')
            for answer in decoded:
                normalize_answer(answer)  # Invalid output must not become a failure observation.
            return [answer.strip() for answer in decoded]
        finally:
            inputs.clear()
            output = tokens = None

    @staticmethod
    def _check_cancelled(cancelled):
        if cancelled is not None:
            if not callable(cancelled):
                raise ValueError('Generation cancellation must be a callable returning a boolean.')
            state = cancelled()
            if not isinstance(state, bool):
                raise ValueError('Generation cancellation must return a boolean.')
            if state:
                raise InterruptedError('Cancelled by user.')

    def answer_many(self, image, questions, *, cancelled=None):
        """Answer independent questions in order, with bounded inference batches.

        ``batch_size`` is a tunable upper bound, lowered for this model's
        lifetime after CUDA OOM. The padded-token budget is soft for a single
        valid question; the per-question context limit is always enforced.
        No image tensors or answers are retained between calls.
        """
        if not isinstance(questions, (list, tuple)) or not questions or not all(
                isinstance(question, str) and question.strip() for question in questions):
            raise ValueError('Provide a nonempty list of visual-check questions.')
        answers = []
        while len(answers) < len(questions):
            self._check_cancelled(cancelled)
            pending = questions[len(answers):len(answers) + self.batch_size]
            prepared, oom = None, None
            count = len(pending)
            try:
                prepared = self._prepare(image, pending)
                count = self._batch_count(prepared)
                if count < len(pending):
                    prepared.clear()
                    prepared = self._prepare(image, pending[:count])
                self._check_cancelled(cancelled)
                batch = self._generate(prepared)
            except self.torch.cuda.OutOfMemoryError as error:
                # A traceback can retain every failed CUDA intermediate. Drop
                # it before leaving the handler and releasing the allocator.
                oom = error.with_traceback(None)
                oom.__context__ = oom.__cause__ = None
            finally:
                if prepared is not None:
                    prepared.clear()
            if oom is not None:
                self.batch_size = max(1, count // 2)
                self.torch.cuda.empty_cache()
                self._check_cancelled(cancelled)
                if count == 1:
                    raise oom
                continue  # Retry the same pending questions; never skip or offload.
            self._check_cancelled(cancelled)
            answers.extend(batch)
        return answers

    def evaluate(self, image, checks):
        from generation_prompts import normalize_prompt_checks
        checks = normalize_prompt_checks(checks)
        return check_answers(checks, self.answer_many(image, [check['question'] for check in checks]))
