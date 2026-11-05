"""Explicit held-out generation panels and conservative paired comparisons.

This module never activates a checkpoint. Scores remain the selected evaluator's
uncalibrated review signals; training loss is not part of the comparison.
"""
import json
import math
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from core import apply_prompt
from generation_prompts import normalize_prompt_checks, validate_prompt


MAX_VALIDATION_PROMPTS = 64
MAX_PANEL_IMAGES = 4096
MAX_PREVIEW_IMAGES = 64
MAX_PREVIEW_BYTES = 32 * 1024**2


def _normalized(prompt):
    return ' '.join(unicodedata.normalize('NFKC', prompt).casefold().split())


def validate_validation_data(records, training_prompts, *, actions=()):
    """Copy 1–64 explicit scenes and reject normalized training/validation leaks.

    The caller supplies both original training scenes and effective training
    captions. Original and action-applied validation captions must be disjoint
    from that corpus. No training image is opened or converted by this validation.
    """
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_VALIDATION_PROMPTS:
        raise ValueError('Provide 1 to 64 explicit held-out validation prompts.')
    training = {_normalized(prompt) for prompt in training_prompts}
    result, seen = [], set()
    for record in records:
        if (not isinstance(record, dict) or 'prompt' not in record or
                set(record) - {'prompt', 'checks'}):
            raise ValueError('Each validation record needs prompt and optional checks only.')
        prompt = validate_prompt(record['prompt'])
        key = _normalized(prompt)
        if key in training or any(_normalized(apply_prompt(prompt, action)) in training for action in actions):
            raise ValueError('Validation prompts overlap training scenes or captions.')
        if key in seen:
            raise ValueError('Remove duplicate validation prompts.')
        seen.add(key)
        result.append({'prompt': prompt, 'checks': normalize_prompt_checks(record.get('checks'))})
    return result


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _check(cancelled):
    if cancelled is not None and cancelled():
        raise InterruptedError('Held-out generation validation cancelled.')


def _integer(value, minimum, maximum, name):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer from {minimum} to {maximum}.')
    return value


def evaluate_generation(model, config, records, actions, directory, progress, cancelled,
                        evaluator_factory=None, expected_scorer=None):
    """Render and score a fresh, bounded panel, always releasing its evaluator.

    Each scene's original and selected actions use the same noise seed. Repeats
    advance by the number of scenes, avoiding collisions between scene/repeat
    coordinates (apart from the documented uint32 wrap). ``validation_phase``
    controls artifact names only and must differ for before and after panels.
    The factory owns fresh resources; callers must not pass a retaining cache.
    """
    records = validate_validation_data(records, [])
    actions = _json(actions)
    if not isinstance(actions, list):
        raise ValueError('Validation actions must be a list of selected action objects.')
    identities = set()
    for action in actions:
        if not isinstance(action, dict) or type(action.get('id')) is not int or action['id'] < 0:
            raise ValueError('Validation actions require nonnegative integer IDs.')
        if action['id'] in identities:
            raise ValueError('Validation action IDs must be distinct.')
        identities.add(action['id'])
        apply_prompt(records[0]['prompt'], action)
    seed = _integer(config.get('seed', 42), 0, 2**32-1, 'Validation seed')
    repeats = _integer(config.get('validation_repeats', 1), 1, 4, 'Validation repeats')
    total = len(records) * (1 + len(actions)) * repeats
    if total > MAX_PANEL_IMAGES:
        raise ValueError('Validation is limited to 4096 images per panel; reduce actions or repeats.')
    phase = config.get('validation_phase', 'panel')
    if not isinstance(phase, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,32}', phase):
        raise ValueError('Use a simple validation phase name without path components.')
    settings = {
        'seed': seed, 'validation_repeats': repeats,
        'generation_steps': config.get('generation_steps', 12),
        'generation_size': config.get('generation_size', 256),
        'generation_metric': config.get('generation_metric', 'combined'),
        'generation_precision_requested': config.get('generation_precision', 'auto'),
        'device': config.get('device', 'cpu'),
        'failure_threshold': config.get('failure_threshold', .5),
        'generation_bias_suite': config.get('generation_bias_suite'),
        'generation_bias_repeats': config.get('generation_bias_repeats', 3),
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _check(cancelled)
    if evaluator_factory is None:
        from discovery_models import create_evaluator
        evaluator_factory = create_evaluator
    evaluator = None
    try:
        evaluator = evaluator_factory(model, {**config, 'task': 'generation'}, directory, progress=progress)
        callbacks = getattr(evaluator, 'set_runtime_callbacks', None)
        if callable(callbacks):
            callbacks(cancelled=cancelled, progress=progress)
        _check(cancelled)
        scoring = _json(evaluator.describe())
        scorer = scoring.get('scorer')
        if not isinstance(scorer, dict) or not scorer:
            raise ValueError('The validation evaluator must report its scorer identity.')
        if expected_scorer is not None and scorer != expected_scorer:
            raise ValueError('Validation scorer identity changed between before and after panels.')
        settings['generation_precision'] = scoring.get('generation_precision')
        settings['scoring_policy'] = {key: scoring[key] for key in (
            'criterion', 'failure_threshold', 'limitations', 'reward_formula', 'threshold_policy',
            'rubric', 'answer_policy', 'calibration', 'min_clip_similarity', 'bias_policy') if key in scoring}
        pairs = []
        retained_images = retained_bytes = 0
        for index, record in enumerate(records):
            for repeat in range(repeats):
                noise_seed = (seed + index + repeat * len(records)) % 2**32
                for action in [None, *actions]:
                    _check(cancelled)
                    identity = None if action is None else action['id']
                    prompt = record['prompt'] if action is None else apply_prompt(record['prompt'], action)
                    suffix = 'original' if action is None else f'action_{identity}'
                    name = f'validation-{phase}-{index}-{repeat}-{suffix}'
                    output = _json(evaluator.predict(prompt, seed=noise_seed, artifact_name=name,
                        checks=record['checks'], reference_prompt=record['prompt'], action=action))
                    _check(cancelled)
                    score = _json(evaluator.score(output, output))
                    # Retention depends on fixed traversal order and byte/count
                    # budgets, never on the measured score. Keep every score.
                    artifact = directory / (name + '.png')
                    if output.get('image') != artifact.name or artifact.is_symlink() or not artifact.is_file():
                        raise ValueError('Validation must return its own generated PNG artifact.')
                    size = artifact.stat().st_size
                    retained = retained_images < MAX_PREVIEW_IMAGES and retained_bytes + size <= MAX_PREVIEW_BYTES
                    output['image_retained'] = retained
                    if retained:
                        retained_images += 1
                        retained_bytes += size
                    else:
                        artifact.unlink()
                        output.pop('image')
                    pairs.append(dict(input_index=index, action_id=identity, repeat=repeat, seed=noise_seed,
                                      prompt=prompt, reference_prompt=record['prompt'], checks=record['checks'],
                                      output=output, score=score))
                    if progress is not None:
                        progress(phase=f'validating {phase} outputs', validation_completed=len(pairs),
                                 validation_total=total)
                    _check(cancelled)
        return _json(dict(status='completed', model=model, records=records, actions=actions,
                          settings=settings, scoring=scoring, scorer=scorer, pairs=pairs,
                          artifacts=dict(evaluated_images=total, retained_images=retained_images,
                              retained_bytes=retained_bytes, max_images=MAX_PREVIEW_IMAGES,
                              max_bytes=MAX_PREVIEW_BYTES, policy='First previews within fixed count and byte budgets; all measurements retained.')))
    finally:
        if evaluator is not None:
            evaluator.close()


def _unit(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f'{name} must be a finite score from zero to one.')
    return float(value)


def _coordinates(panel):
    """Require every original/action/repeat coordinate, not a cherry-picked panel."""
    if not isinstance(panel, dict) or panel.get('status') != 'completed':
        raise ValueError('Compare two complete validation panels.')
    records = validate_validation_data(panel.get('records'), [])
    actions = panel.get('actions')
    settings = panel.get('settings')
    if not isinstance(actions, list) or not isinstance(settings, dict):
        raise ValueError('Validation panels need action and generation settings snapshots.')
    action_lookup = {None: None}
    for action in actions:
        if (not isinstance(action, dict) or type(action.get('id')) is not int or
                action['id'] < 0 or action['id'] in action_lookup):
            raise ValueError('Validation action IDs must be distinct nonnegative integers.')
        action_lookup[action['id']] = action
    repeats = _integer(settings.get('validation_repeats'), 1, 4, 'Validation repeats')
    seed = _integer(settings.get('seed'), 0, 2**32-1, 'Validation seed')
    expected = {(index, identity, repeat) for index in range(len(records))
                for identity in action_lookup for repeat in range(repeats)}
    pairs = panel.get('pairs')
    if not isinstance(pairs, list) or len(pairs) != len(expected) or len(pairs) > MAX_PANEL_IMAGES:
        raise ValueError('Validation pairs are missing, duplicated, or exceed the panel limit.')
    found = set()
    for row in pairs:
        if not isinstance(row, dict):
            raise ValueError('Validation pairs must be objects.')
        index, identity, repeat = row.get('input_index'), row.get('action_id'), row.get('repeat')
        if (type(index) is not int or type(repeat) is not int or
                (identity is not None and type(identity) is not int)):
            raise ValueError('Validation pair coordinates must use integer indices and action IDs.')
        key = index, identity, repeat
        if key not in expected or key in found:
            raise ValueError('Validation pairs are missing or duplicated.')
        found.add(key)
        record, action = records[index], action_lookup[identity]
        prompt = record['prompt'] if action is None else apply_prompt(record['prompt'], action)
        if (row.get('seed') != (seed + index + repeat * len(records)) % 2**32 or
                type(row.get('seed')) is not int or row.get('reference_prompt') != record['prompt'] or
                row.get('prompt') != prompt or row.get('checks') != record['checks']):
            raise ValueError('Validation pair prompts, checks, actions or seeds do not match their panel.')
    return pairs


def _interval(values):
    """Percentile paired bootstrap, with one independent unit per scene."""
    if not values:
        return None
    if len(values) == 1 or max(values) == min(values):
        return [values[0], values[0]]
    rng = random.Random(42)
    count = len(values)
    draws = sorted(math.fsum(rng.choices(values, k=count)) / count for _ in range(4000))
    def percentile(fraction):
        position = (len(draws)-1) * fraction
        lower = int(position)
        return draws[lower] + (draws[min(lower+1, len(draws)-1)]-draws[lower]) * (position-lower)
    return [percentile(.025), percentile(.975)]


def _means(rows):
    """Return equal-scene means, retaining all matched seeds/actions per scene."""
    grouped = defaultdict(list)
    for index, before, after in rows:
        grouped[index].append((before, after))
    before_means, after_means, differences = [], [], []
    for pairs in grouped.values():
        before = math.fsum(pair[0] for pair in pairs) / len(pairs)
        after = math.fsum(pair[1] for pair in pairs) / len(pairs)
        before_means.append(before)
        after_means.append(after)
        differences.append(after-before)
    count = len(grouped)
    return dict(count=len(rows), prompt_count=count,
        before_mean=math.fsum(before_means)/count if count else None,
        after_mean=math.fsum(after_means)/count if count else None,
        mean_change=math.fsum(differences)/count if count else None,
        ci95=_interval(differences))


def _summarize(rows):
    values = _means([(row['input_index'], row['before_severity'], row['after_severity']) for row in rows])
    values['before_mean_severity'] = values.pop('before_mean')
    values['after_mean_severity'] = values.pop('after_mean')
    flags = _means([(row['input_index'], row['before_failure'], row['after_failure']) for row in rows])
    values.update(before_failure_rate=flags['before_mean'], after_failure_rate=flags['after_mean'],
                  failure_rate_change=flags['mean_change'])
    names = sorted({name for row in rows for name in row['components']})
    values['components'] = {name: _means([
        (row['input_index'], row['components'][name]['before'], row['components'][name]['after'])
        for row in rows if row['components'].get(name, {}).get('before') is not None]) for name in names}
    return values


def compare_generation(before, after):
    """Compare a fixed held-out panel, never promoting or deleting a checkpoint.

    Improvement requires at least eight distinct scenes, selected mean severity
    decreasing by at least .01 with bootstrap upper bound below zero, no selected
    flag-rate increase, and clean/all mean increases no greater than .01.
    Regression requires eight scenes and any scope increasing by more than .01
    with bootstrap lower bound above zero. All other outcomes are inconclusive.
    These are explicit review rules, not calibrated statistical guarantees.
    """
    before, after = _json(before), _json(after)
    left, right = _coordinates(before), _coordinates(after)
    for key in ('records', 'actions', 'settings', 'scorer'):
        if before.get(key) != after.get(key):
            raise ValueError(f'Before and after validation {key} must match exactly.')
    if not isinstance(before.get('scorer'), dict) or not before['scorer']:
        raise ValueError('Before and after validation need an identified scorer.')
    provenance = ('input_index', 'action_id', 'repeat', 'seed', 'prompt', 'reference_prompt', 'checks')
    paired = []
    for original, candidate in zip(left, right):
        if any(original.get(key) != candidate.get(key) for key in provenance):
            raise ValueError('Before and after validation pair provenance must match exactly.')
        scores = []
        components = []
        for row in (original, candidate):
            score, output = row.get('score'), row.get('output')
            if not isinstance(score, dict) or not isinstance(output, dict) or type(score.get('failure')) is not bool:
                raise ValueError('Validation pairs need explicit boolean failure flags and output scores.')
            scores.append(_unit(score.get('severity'), 'Validation severity'))
            raw = output.get('failure_scores', {})
            if not isinstance(raw, dict):
                raise ValueError('Validation failure components must be a score object.')
            components.append({key: None if value is None else _unit(value, f'Validation {key}')
                               for key, value in raw.items()})
        if (set(components[0]) != set(components[1]) or any(
                (components[0][key] is None) != (components[1][key] is None) for key in components[0])):
            raise ValueError('Before and after component evaluation coverage must match.')
        paired.append({**{key: original[key] for key in provenance},
            'before_severity': scores[0], 'after_severity': scores[1], 'mean_change': scores[1]-scores[0],
            'before_failure': original['score']['failure'], 'after_failure': candidate['score']['failure'],
            'components': {key: {'before': components[0][key], 'after': components[1][key],
                'mean_change': None if components[0][key] is None else components[1][key]-components[0][key]}
                for key in components[0]}})
    scopes = dict(clean=_summarize([row for row in paired if row['action_id'] is None]),
                  selected=_summarize([row for row in paired if row['action_id'] is not None]),
                  all=_summarize(paired))
    count = len(before['records'])
    verdict, reason = 'inconclusive', 'The fixed panel does not meet the declared improvement or regression rules.'
    if count < 8:
        reason = 'At least 8 distinct held-out prompts are required for an improvement or regression verdict.'
    elif any(summary['mean_change'] is not None and summary['mean_change'] > .01 + 1e-12 and
             summary['ci95'][0] > 0 for summary in scopes.values()):
        verdict = 'regressed'
        reason = 'A scope worsened by more than .01 severity with its paired interval entirely above zero.'
    else:
        selected = scopes['selected']
        if (count >= 8 and selected['count'] and selected['mean_change'] <= -.01 + 1e-12 and
                selected['ci95'][1] < 0 and selected['failure_rate_change'] <= 0 and
                scopes['clean']['mean_change'] <= .01 + 1e-12 and scopes['all']['mean_change'] <= .01 + 1e-12):
            verdict = 'improved'
            reason = 'Selected severity improved with a negative paired interval and all declared guardrails passed.'
    per_prompt = []
    for index, record in enumerate(before['records']):
        rows = [row for row in paired if row['input_index'] == index]
        per_prompt.append(dict(input_index=index, prompt=record['prompt'],
            clean=_summarize([row for row in rows if row['action_id'] is None]),
            selected=_summarize([row for row in rows if row['action_id'] is not None]), all=_summarize(rows)))
    return _json(dict(status='completed', verdict=verdict, reason=reason, prompt_count=count,
        scorer=before['scorer'], settings=before['settings'], **scopes, per_pair=paired, per_prompt=per_prompt,
        bootstrap=dict(method='paired percentile bootstrap over per-prompt mean changes', draws=4000,
                       seed=42, confidence=.95, independent_unit='distinct prompt'),
        limitations=[
            'Scores are uncalibrated automatic review signals, not guarantees of output quality.',
            'Intervals resample distinct prompts; action and seed repeats are not independent prompts.',
            'Matched bias-suite scores may be shared across prompts for an action. Prompt intervals hold that suite fixed and do not estimate its uncertainty.',
            'This validation panel supports checkpoint review. Final testing requires new untouched prompts.',
            'No checkpoint is activated or deleted by this report.']))
