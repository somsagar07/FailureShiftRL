"""Optional, explicitly configured matched comparisons of prompt-fulfillment errors.

The observed gap describes only the supplied suite under its declared reference.
It is neither a failure probability nor evidence of a confirmed causal bias.
"""

import copy
import json
import math

from core import apply_prompt
from generation_prompts import normalize_prompt_checks, validate_prompt


MAX_CASES = 32
MAX_GROUPS = 8
MAX_NAME_CHARS = 80


def _fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f'{label} must contain exactly {", ".join(sorted(expected))}.')


def _name(value, label):
    if (not isinstance(value, str) or not value.strip() or len(value) > MAX_NAME_CHARS
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError(f'{label} must be nonempty text of at most {MAX_NAME_CHARS} characters.')
    return value.strip()


def normalize_bias_suite(value):
    """Validate a declared comparison reference and copy its complete independent checks."""
    if value is None:
        return None
    _fields(value, {'name', 'reference', 'cases'}, 'Bias suite')
    name = _name(value['name'], 'Bias suite name')
    if value['reference'] != 'equal_error_rates':
        raise ValueError('Bias suite reference must be equal_error_rates.')
    if not isinstance(value['cases'], list) or not 1 <= len(value['cases']) <= MAX_CASES:
        raise ValueError(f'Bias suite cases must be a list of 1 to {MAX_CASES} cases.')

    cases, case_names, expected_groups = [], set(), None
    for case_index, case in enumerate(value['cases'], 1):
        label = f'Bias suite case {case_index}'
        _fields(case, {'name', 'groups'}, label)
        case_name = _name(case['name'], f'{label} name')
        if case_name in case_names:
            raise ValueError(f'{label} has a duplicate case name: {case_name}.')
        case_names.add(case_name)
        if not isinstance(case['groups'], list) or not 2 <= len(case['groups']) <= MAX_GROUPS:
            raise ValueError(f'{label} groups must be a list of 2 to {MAX_GROUPS} groups.')
        groups, group_names = [], set()
        for group_index, group in enumerate(case['groups'], 1):
            group_label = f'{label} group {group_index}'
            _fields(group, {'name', 'prompt', 'checks'}, group_label)
            group_name = _name(group['name'], f'{group_label} name')
            if group_name in group_names:
                raise ValueError(f'{label} has a duplicate group name: {group_name}.')
            group_names.add(group_name)
            try:
                prompt = validate_prompt(group['prompt'])
                checks = normalize_prompt_checks(group['checks'])
                if not checks:
                    raise ValueError('nonempty independent prompt checks are required')
            except ValueError as exc:
                raise ValueError(f'{group_label}: {exc}') from exc
            groups.append({'name': group_name, 'prompt': prompt, 'checks': checks})
        if expected_groups is None:
            expected_groups = group_names
        elif group_names != expected_groups:
            raise ValueError(f'{label} must contain the same group names as every other case.')
        cases.append({'name': case_name, 'groups': groups})
    return {'name': name, 'reference': 'equal_error_rates', 'cases': cases}


def _integer(value, minimum, maximum, label):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f'{label} must be an integer from {minimum} to {maximum}.')
    return value


def _action_snapshot(action):
    if action is not None and not isinstance(action, dict):
        raise ValueError('Bias comparison action must be a complete action object or None.')
    try:
        key = json.dumps(action, sort_keys=True, allow_nan=False, separators=(',', ':'))
        snapshot = json.loads(key)
        if snapshot != action:
            raise ValueError('Action must use JSON values and text object keys.')
    except (TypeError, ValueError) as exc:
        raise ValueError('Bias comparison action must contain finite JSON values and text keys.') from exc
    return key, snapshot


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise InterruptedError('Matched bias comparison cancelled.')


def _content_failure(output):
    value = output.get('content_failure') if isinstance(output, dict) else None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not 0 <= value <= 1 or not math.isfinite(value)):
        raise ValueError('Bias comparison content_failure must be a finite number from 0 to 1.')
    return float(value)


def _check_results(output, checks):
    """Copy an optional complete QA trace without rewriting the judge's answers."""
    if 'checks' not in output:
        return None
    rows = output['checks']
    if not isinstance(rows, list) or len(rows) != len(checks):
        raise ValueError('Bias comparison checks must contain one trace row per configured check.')
    required = {'question', 'expected', 'answer', 'passed'}
    for index, (row, check) in enumerate(zip(rows, checks), 1):
        label = f'Bias comparison check result {index}'
        if (not isinstance(row, dict) or not required <= set(row)
                or set(row) - required - {'skill'}):
            raise ValueError(f'{label} must contain question, expected, answer, passed and optional skill.')
        text_fields = ('question', 'expected', 'answer') + (('skill',) if 'skill' in row else ())
        if (not all(isinstance(row[field], str) for field in text_fields)
                or not isinstance(row['passed'], bool)):
            raise ValueError(f'{label} requires text answers and a boolean passed value.')
        # Reuse the independent-check text bounds; retain the original answer,
        # including its whitespace, as the trace of what the judge returned.
        normalize_prompt_checks([{
            'question': row['question'], 'answer': row['answer'],
            **({'skill': row['skill']} if 'skill' in row else {}),
        }])
        if row['question'] != check['question'] or row['expected'] != check['answer']:
            raise ValueError(f'{label} must match its configured question and expected answer.')
    return copy.deepcopy(rows)


def _check_evidence(output, checks, check_results):
    """Retain optional structured evidence only when its complete trace agrees."""
    if 'check_evidence' not in output:
        return None
    from generation_failure import _decision_requirement_scores, _requirement_questions, _unit

    rows = output['check_evidence']
    if check_results is None or not isinstance(rows, list) or len(rows) != len(checks):
        raise ValueError('Bias comparison check evidence needs one row per recorded check result.')
    questions = _requirement_questions(checks)
    decisions = []
    for row, question in zip(rows, questions):
        if not isinstance(row, dict) or not isinstance(row.get('passed'), bool):
            raise ValueError('Bias comparison check evidence needs objects with boolean passed values.')
        probabilities = row.get('probabilities')
        if not isinstance(probabilities, list) or len(probabilities) != 2:
            raise ValueError('Bias comparison check evidence needs two decision probabilities.')
        probabilities = [_unit(value, 'Bias comparison check probability') for value in probabilities]
        value = _unit(row.get('value'), 'Bias comparison check value')
        _unit(row.get('threshold'), 'Bias comparison check threshold')
        if (not math.isclose(math.fsum(probabilities), 1., rel_tol=0., abs_tol=1e-5)
                or not math.isclose(value, probabilities[1], rel_tol=0., abs_tol=1e-5)):
            raise ValueError('Bias comparison check probabilities must sum to one and agree with the value.')
        decisions.append(dict(id=question['id'], type='noul', probabilities=probabilities, value=value))
    # Reconstruct the canonical fields using the same target-verification and
    # binary-readout logic that produced the evidence. This also checks IDs,
    # order, question isolation, methods, labels, threshold and trace outcomes.
    expected = _decision_requirement_scores(checks, questions, iter(decisions))
    if rows != expected['check_evidence'] or check_results != expected['checks']:
        raise ValueError('Bias comparison check evidence must match its questions and recorded outcomes.')
    return copy.deepcopy(expected['check_evidence'])


class BiasProbe:
    """Evaluate paired samples and cache completed results for this fixed suite/run.

    ``predict`` owns model execution and artifact handling. Its content error
    score, optional QA trace and structured check evidence are retained; images
    are never retained or written.
    """

    def __init__(self, suite, repeats=3, seed=42):
        self._suite = normalize_bias_suite(suite)
        self._repeats = _integer(repeats, 2, 16, 'Bias comparison repeats')
        self._seed = _integer(seed, 0, 2**32 - 1, 'Bias comparison seed')
        self._cache = {}

    def _result(self):
        return {
            'score': None,
            'status': 'not_evaluated',
            'name': 'quality_disparity',
            'method': 'max_group_mean_content_failure_gap',
            'scope': 'matched_suite_action',
            'reference': self._suite['reference'] if self._suite else None,
            'suite_name': self._suite['name'] if self._suite else None,
            'seed': self._seed,
            'repeats': self._repeats,
            'group_rates': {},
            'group_sample_counts': {},
            'case_count': 0,
            'group_count': 0,
            'sample_count': 0,
            'cases': [],
            'interpretation': ('Observed prompt-fulfillment quality disparity across the explicitly '
                               'configured groups; not a failure probability or confirmed causal bias.'),
        }

    def evaluate(self, action, predict, cancelled=None, progress=None):
        """Return complete score evidence, raising on errors or cancellation.

        Every group in a case/repeat receives the same deterministic unsigned seed.
        Equal case and repeat counts make each observation equally weighted in its
        group's mean. The final score is the largest group mean minus the smallest.
        """
        _check_cancelled(cancelled)
        result = self._result()
        if self._suite is None:
            result['reason'] = 'No matched comparison suite was configured.'
            return result

        key, action = _action_snapshot(action)
        if key in self._cache:
            return copy.deepcopy(self._cache[key])

        names = [group['name'] for group in self._suite['cases'][0]['groups']]
        scores = {name: [] for name in names}
        total = len(self._suite['cases']) * len(names) * self._repeats
        completed = 0
        for case_index, case in enumerate(self._suite['cases']):
            case_scores = {name: [] for name in names}
            case_result = {'name': case['name'], 'group_rates': {}, 'samples': []}
            for repeat in range(self._repeats):
                seed = (self._seed + case_index * 1000003 + repeat) % 2**32
                sample = {'repeat': repeat, 'seed': seed, 'groups': []}
                for group_index, group in enumerate(case['groups']):
                    if progress is not None:
                        progress(phase='matched bias comparison', completed=completed, total=total,
                                 case=case['name'], group=group['name'], repeat=repeat, seed=seed)
                    _check_cancelled(cancelled)
                    prompt = group['prompt']
                    transformed = apply_prompt(prompt, action) if action is not None else prompt
                    output = predict(
                        transformed, checks=copy.deepcopy(group['checks']), reference_prompt=prompt,
                        seed=seed,
                        artifact_name=f'bias-c{case_index + 1}-r{repeat + 1}-g{group_index + 1}',
                    )
                    _check_cancelled(cancelled)
                    failure = _content_failure(output)
                    check_results = _check_results(output, group['checks'])
                    check_evidence = _check_evidence(output, group['checks'], check_results)
                    scores[group['name']].append(failure)
                    case_scores[group['name']].append(failure)
                    observation = {
                        'name': group['name'], 'prompt': prompt, 'transformed_prompt': transformed,
                        'checks': copy.deepcopy(group['checks']), 'content_failure': failure,
                    }
                    if check_results is not None:
                        observation['check_results'] = check_results
                    if check_evidence is not None:
                        observation['check_evidence'] = check_evidence
                    sample['groups'].append(observation)
                    completed += 1
                case_result['samples'].append(sample)
            case_result['group_rates'] = {
                name: math.fsum(values) / self._repeats for name, values in case_scores.items()
            }
            result['cases'].append(case_result)

        rates = {name: math.fsum(values) / len(values) for name, values in scores.items()}
        result.update({
            'score': max(rates.values()) - min(rates.values()),
            'status': 'evaluated',
            'group_rates': rates,
            'group_sample_counts': {name: len(values) for name, values in scores.items()},
            'case_count': len(self._suite['cases']),
            'group_count': len(names),
            'sample_count': completed,
        })
        if progress is not None:
            progress(phase='matched bias comparison', completed=completed, total=total)
        _check_cancelled(cancelled)
        self._cache[key] = copy.deepcopy(result)
        return result
