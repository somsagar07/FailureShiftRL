"""Bounded NLI support with evidence and a whitespace sensitivity diagnostic.

The raw NLI score is retained, including when a probe scores higher. Reliability
only describes coverage and this consistency check; it is not factuality
calibration. In particular, stable NLI can miss speaker and relation errors.
"""
import math
import re

from model_loading import configure_parallel_loading
from text_encoders import (_dependencies, _options, _pretrained, _resolved_revision,
                           _selection, _token_limit)


def _collapse(text):
    return ' '.join(text.split())


class GroundedScorer:
    SCORER_REVISION = 'nli-evidence-v1'
    POLICY_REVISION = 'raw-support-format-probe-v1'
    FORMAT_DELTA_THRESHOLD = .20
    MAX_CLAIMS = 32
    MAX_SOURCE_WINDOWS = 32
    BATCH_SIZE = 16
    LIMITATIONS = ('NLI support is an uncalibrated review signal, not verified factuality. '
                  'A stable score can miss wrong speakers, entities, and relations. '
                  'The probe only compares raw source text with collapsed whitespace; '
                  'claims are sentence-like spans, not verified atomic facts. '
                  'Evidence is a decoded token window and may not be an exact source quotation.')

    def __init__(self, model, device):
        configure_parallel_loading()
        self.torch, transformers = _dependencies()
        selection = _selection(model, 'summarization')
        cfg = transformers.AutoConfig.from_pretrained(selection['name'], **_options(selection))
        revision = _resolved_revision(selection, getattr(cfg, '_commit_hash', None))
        options = _options(selection, revision)
        ids = [int(key) for key, label in cfg.id2label.items()
               if label.lower() in ('entailment', 'entail')]
        if len(ids) != 1 or not 0 <= ids[0] < cfg.num_labels:
            raise ValueError('NLI scorer must declare an unambiguous entailment label in id2label.')
        loaded, info = transformers.AutoModelForSequenceClassification.from_pretrained(
            selection['name'], config=cfg, weights_only=True, output_loading_info=True, **options)
        self.model = _pretrained(loaded, info, self.torch).to(device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(selection['name'], **options)
        self.entailment_id = ids[0]
        self.num_labels = cfg.num_labels
        self.limit = _token_limit(self.tokenizer, cfg, default=512)
        if self.limit < 16:
            raise ValueError('Grounded NLI needs an input limit of at least 16 tokens.')
        self.device = device
        self.identity = {'name': selection['name'], 'revision': revision, **self._policy()}

    def _policy(self):
        return {'scorer_revision': self.SCORER_REVISION,
                'policy_revision': self.POLICY_REVISION,
                'format_delta_threshold': self.FORMAT_DELTA_THRESHOLD,
                'max_claims': self.MAX_CLAIMS,
                'max_source_windows': self.MAX_SOURCE_WINDOWS,
                'batch_size': self.BATCH_SIZE,
                'batch_policy': 'halve on CUDA OutOfMemoryError; preserve pairs; fail at one pair',
                'input_token_limit': self.limit,
                'window_tokens': max(4, (self.limit - 8) // 2),
                'aggregation': 'mean of per-claim maximum raw-window entailment; null if incomplete',
                'reliability': 'complete raw/probe coverage, unchanged lexical windows, '
                               'and every claim/aggregate format delta <= threshold'}

    def score(self, source, summary):
        """Legacy raw scalar API. Production decisions should use ``assess``.

        Unstable but complete measurements retain their raw score. Incomplete
        measurements raise instead of silently averaging only the easiest claims.
        """
        result = self.assess(source, summary)
        if result['source_support'] is None:
            raise ValueError('Grounded NLI support is incomplete; use assess() for coverage and warnings.')
        return result['source_support']

    def _windows(self, source, window):
        if not source.strip():
            return [], 0, 0
        ids = self.tokenizer.encode(source, add_special_tokens=False)
        stride = max(1, window // 2)
        total = math.ceil(len(ids) / stride)
        windows = []
        for start in range(0, min(len(ids), stride * self.MAX_SOURCE_WINDOWS), stride):
            end = min(start + window, len(ids))
            windows.append({'text': self.tokenizer.decode(ids[start:end], skip_special_tokens=True),
                            'token_span': [start, end]})
        return windows, len(ids), total

    def _infer(self, pairs):
        """Score unique pairs in bounded batches; oversized pairs stay unscored."""
        scores = [None] * len(pairs)
        start, batch_size = 0, self.BATCH_SIZE
        while start < len(pairs):
            batch = pairs[start:start + batch_size]
            try:
                probabilities = self._infer_batch(batch)
            except self.torch.cuda.OutOfMemoryError as exc:
                if len(batch) == 1:
                    raise self.torch.cuda.OutOfMemoryError(
                        'Grounded NLI cannot fit a single pair in CUDA memory; free GPU memory or use CPU.') from exc
                batch_size = max(1, len(batch) // 2)
                # The failed frame and its tensors are released when this
                # exception block exits. Retry the same start index, never skip.
                continue
            scores[start:start + len(batch)] = probabilities
            start += len(batch)
        return scores

    def _infer_batch(self, batch):
        # Keep device tensors local so a failed batch can be released before a
        # smaller retry. Other RuntimeErrors must propagate without retrying.
        scores = [None] * len(batch)
        inputs = self.tokenizer([pair[0] for pair in batch], [pair[1] for pair in batch],
                                return_tensors='pt', padding=True, truncation=False)
        ids, mask = inputs.get('input_ids'), inputs.get('attention_mask')
        if (ids is None or mask is None or ids.ndim != 2 or mask.shape != ids.shape
                or ids.shape[0] != len(batch)):
            raise ValueError('NLI tokenizer returned invalid batch inputs.')
        lengths = mask.sum(dim=1).tolist()
        valid = [index for index, length in enumerate(lengths) if 0 < length <= self.limit]
        if not valid:
            return scores
        # Trim padding only. Oversized real sequences are omitted with an
        # explicit pair warning in assess(), never truncated into valid ones.
        width = max(lengths[index] for index in valid)
        positions = slice(-width, None) if getattr(self.tokenizer, 'padding_side', 'right') == 'left' else slice(None, width)
        selected = {key: value[valid, positions].to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            logits = self.model(**selected).logits
            if (logits.ndim != 2 or logits.shape != (len(valid), getattr(self, 'num_labels', 3))
                    or not self.torch.isfinite(logits).all()):
                raise ValueError('NLI returned invalid or nonfinite logits.')
            probabilities = logits.float().softmax(-1)[:, self.entailment_id].cpu().tolist()
        for index, probability in zip(valid, probabilities):
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError('NLI returned invalid or nonfinite support.')
            scores[index] = float(probability)
        return scores

    def assess(self, source, summary):
        """Return raw support, paired evidence, coverage, and reliability warnings.

        Whitespace is collapsed before separately tokenizing the probe source.
        Changed lexical windows invalidate comparability even if scores coincide.
        Both the aggregate and each claim must have absolute delta <= 0.20.
        No source, claim, or pair truncation is hidden: incomplete aggregate scores
        are null, and available per-claim scores remain marked partial/unscored.
        """
        if not isinstance(source, str) or not isinstance(summary, str):
            raise ValueError('Grounded NLI needs source and summary strings.')
        claims = [{'text': text.strip(), 'support': None, 'collapsed_support': None,
                   'evidence': '', 'collapsed_evidence': '', 'evidence_token_span': None,
                   'collapsed_evidence_token_span': None, 'format_delta': None,
                   'reliable': False, 'status': 'unscored',
                   'raw_windows_scored': 0, 'collapsed_windows_scored': 0}
                  for text in re.split(r'(?<=[.!?])\s+|\n+', summary) if text.strip()]
        warnings = []

        def warn(code, message, **details):
            warnings.append({'code': code, 'message': message, **details})

        coverage = {'claims_total': len(claims), 'claims_scored': 0, 'claims_unscored': len(claims),
                    'raw_source_tokens': 0, 'collapsed_source_tokens': 0,
                    'raw_windows_total': 0, 'collapsed_windows_total': 0,
                    'raw_windows_scored': 0, 'collapsed_windows_scored': 0,
                    'source_truncated': False, 'format_windowing_changed': False,
                    'pairs_scored': 0, 'pairs_unscored': 0, 'complete': False}
        result = {'source_support': None, 'collapsed_source_support': None, 'reliable': False,
                  'format_delta': None, 'max_claim_format_delta': None,
                  'claims': claims, 'warnings': warnings, 'coverage': coverage,
                  'identity': {**self.identity, **self._policy()}, 'limitations': self.LIMITATIONS}
        if not claims:
            result.update(source_support=0., collapsed_source_support=0., format_delta=0.,
                          max_claim_format_delta=0.)
            warn('empty_summary', 'The summary has no claims to assess; support is zero by convention.')
            return result

        window = max(4, (self.limit - 8) // 2)
        raw, raw_tokens, raw_total = self._windows(source, window)
        collapsed, collapsed_tokens, collapsed_total = self._windows(_collapse(source), window)
        coverage.update(raw_source_tokens=raw_tokens, collapsed_source_tokens=collapsed_tokens,
                        raw_windows_total=raw_total, collapsed_windows_total=collapsed_total)
        if not raw or not collapsed:
            warn('empty_source', 'The source has no usable tokens; every claim remains unscored.')
            return result
        truncated = raw_total > len(raw) or collapsed_total > len(collapsed)
        coverage['source_truncated'] = truncated
        if truncated:
            warn('source_window_budget_exceeded',
                 'Only the first bounded set of source windows was assessed; aggregate support is unavailable.',
                 max_windows=self.MAX_SOURCE_WINDOWS)
        changed = (raw_total != collapsed_total or
                   [_collapse(item['text']) for item in raw] != [_collapse(item['text']) for item in collapsed])
        coverage['format_windowing_changed'] = changed
        if changed:
            warn('format_windowing_changed',
                 'Whitespace collapse changed lexical window boundaries or coverage; '
                 'the support difference cannot be attributed only to formatting.')
        if len(claims) > self.MAX_CLAIMS:
            warn('claim_budget_exceeded',
                 'Claims beyond the assessment budget remain visible and unscored.',
                 max_claims=self.MAX_CLAIMS)

        pairs, pair_indices, claim_pairs = [], {}, []
        for index, claim in enumerate(claims):
            variant_pairs = {'raw': [], 'collapsed': []}
            claim_pairs.append(variant_pairs)
            if index >= self.MAX_CLAIMS:
                continue
            if len(self.tokenizer.encode(claim['text'], add_special_tokens=False)) > window:
                warn('claim_too_long', 'This claim exceeds the token budget and was not truncated or scored.',
                     claim_index=index, max_tokens=window)
                continue
            for variant, windows in (('raw', raw), ('collapsed', collapsed)):
                for evidence in windows:
                    pair = (evidence['text'], claim['text'])
                    if pair not in pair_indices:
                        pair_indices[pair] = len(pairs)
                        pairs.append(pair)
                    variant_pairs[variant].append(pair_indices[pair])
        scores = self._infer(pairs)
        coverage['pairs_scored'] = sum(score is not None for score in scores)
        coverage['pairs_unscored'] = len(scores) - coverage['pairs_scored']
        if coverage['pairs_unscored']:
            warn('pair_input_limit',
                 'Some decoded premise/claim pairs exceed the model input limit; '
                 'they were left unscored, without truncation.',
                 pairs_unscored=coverage['pairs_unscored'])

        scored_windows = {'raw': set(), 'collapsed': set()}
        for index, (claim, variant_pairs) in enumerate(zip(claims, claim_pairs)):
            complete = not truncated
            for variant, windows, prefix in (('raw', raw, ''), ('collapsed', collapsed, 'collapsed_')):
                indices = variant_pairs[variant]
                available = [(window_index, scores[pair_index]) for window_index, pair_index in enumerate(indices)
                             if scores[pair_index] is not None]
                claim[f'{variant}_windows_scored'] = len(available)
                complete = complete and len(available) == len(windows)
                if available:
                    evidence_index, support = max(available, key=lambda item: item[1])
                    evidence = windows[evidence_index]
                    claim[f'{prefix}support'] = support
                    claim[f'{prefix}evidence'] = evidence['text']
                    claim[f'{prefix}evidence_token_span'] = evidence['token_span']
                    scored_windows[variant].update(item[0] for item in available)
            claim['status'] = 'scored' if complete else ('partial' if claim['support'] is not None else 'unscored')
            if claim['support'] is not None and claim['collapsed_support'] is not None:
                delta = abs(claim['support'] - claim['collapsed_support'])
                claim['format_delta'] = delta
                if delta > self.FORMAT_DELTA_THRESHOLD:
                    warn('format_sensitive',
                         'NLI support for this claim changes materially under collapsed source whitespace.',
                         claim_index=index, delta=delta, threshold=self.FORMAT_DELTA_THRESHOLD)
                claim['reliable'] = complete and not changed and delta <= self.FORMAT_DELTA_THRESHOLD

        coverage['raw_windows_scored'] = len(scored_windows['raw'])
        coverage['collapsed_windows_scored'] = len(scored_windows['collapsed'])
        coverage['claims_scored'] = sum(claim['status'] == 'scored' for claim in claims)
        coverage['claims_unscored'] = len(claims) - coverage['claims_scored']
        coverage['complete'] = coverage['claims_scored'] == len(claims)
        deltas = [claim['format_delta'] for claim in claims if claim['format_delta'] is not None]
        result['max_claim_format_delta'] = max(deltas, default=None)
        if coverage['complete']:
            result['source_support'] = sum(claim['support'] for claim in claims) / len(claims)
            result['collapsed_source_support'] = sum(claim['collapsed_support'] for claim in claims) / len(claims)
            result['format_delta'] = abs(result['source_support'] - result['collapsed_source_support'])
            result['reliable'] = (all(claim['reliable'] for claim in claims)
                                  and result['format_delta'] <= self.FORMAT_DELTA_THRESHOLD)
        return result
