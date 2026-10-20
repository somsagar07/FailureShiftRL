"""Aggregate measured actions and index explicit-target training plans lazily."""
import copy
import json
import operator
import random
import re
from collections.abc import Sequence
from pathlib import Path

from core import apply_text
from data_sources import MAX_UPLOAD_BYTES, MAX_SOURCE_CHARS


def summarize(discovery, context):
    """Count each measured input/action pair once, irrespective of RL visits."""
    preview = context['preview']
    actions = {action['id']: action for action in preview['actions']}
    pairs = {}
    for record in discovery['records']:
        pair = record['input_index'], record['action_id']
        if pair[1] not in actions or not 0 <= pair[0] < preview['input_count']:
            raise ValueError('Discovery measurements do not match this analysis. Analyze again.')
        pairs.setdefault(pair, record)
    if not pairs:
        raise ValueError('Adaptation requires at least one measured input/action pair.')
    preferences = {row['action_id']: row for row in (discovery.get('policy_statistics') or {}).get('actions', [])}
    rows = []
    for action_id, action in actions.items():
        measured = [record for (index, identity), record in pairs.items() if identity == action_id]
        count = len(measured)
        failures = sum(bool(record['failure']) for record in measured)
        representative = min(measured, key=lambda row: (-bool(row['failure']), -row['severity'], row['input_index'])) if count else None
        preference = preferences.get(action_id, {})
        rows.append(dict(action_id=action_id, action=copy.deepcopy(action), evaluated_inputs=count,
                         failure_count=failures, failure_rate=failures / count if count else None,
                         mean_severity=sum(record['severity'] for record in measured) / count if count else None,
                         mean_reward=sum(record['reward'] for record in measured) / count if count else None,
                         preference_mean=preference.get('mean'), preference_std=preference.get('std'),
                         example_input_index=representative['input_index'] if representative else None))
    rows.sort(key=lambda row: (row['failure_rate'] is None, -(row['failure_rate'] or 0),
                              -row['evaluated_inputs'], -(row['mean_severity'] or 0),
                              -(row['mean_reward'] or 0), row['action_id']))
    defaults = dict(include_originals=discovery['task'] != 'generation',
                    reference_summary=discovery['config'].get('reference_summary', ''),
                    source_text=preview['config'].get('text', ''))
    if discovery['task'] == 'summarization' and context.get('summary_dataset'):
        defaults.update(uses_dataset=True, dataset=copy.deepcopy(context['summary_dataset']))
    elif discovery['task'] == 'generation' and context.get('generation_dataset'):
        defaults.update(uses_dataset=True, dataset=copy.deepcopy(context['generation_dataset']))
    return dict(discovery_id=discovery['id'], embedding_id=discovery['embedding_id'], task=discovery['task'],
                input_count=preview['input_count'], max_examples=None, actions=rows,
                suggested_action_ids=[row['action_id'] for row in rows if row['failure_count'] > 0][:3],
                defaults=defaults)


def _text(value, name, maximum):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f'{name} must be nonempty text of at most {maximum:,} characters.')
    return value


def _records(data, name='training_data'):
    if not isinstance(data, list) or not data:
        raise ValueError(f'Provide a nonempty {name} list with explicit desired targets.')
    try:
        size = len(json.dumps(data, allow_nan=False, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
    except (TypeError, ValueError) as exc:
        raise ValueError('Training data must contain valid JSON records.') from exc
    if size > MAX_UPLOAD_BYTES:
        raise ValueError('Training data exceeds 20 MB.')
    return data


def _image_path(value):
    from PIL import Image
    _text(value, 'Desired training image path', 50000)
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise ValueError('Supply an existing absolute path to a desired training image.')
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError('Desired training images must be at most 20 MB each.')
    try:
        with Image.open(path) as image:
            if image.width * image.height > 20_000_000:
                raise ValueError('Use desired images under 20 megapixels.')
            image.verify()
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError('Desired training image path must identify a valid image.') from exc
    return str(path.resolve())


def prepare(discovery, context, config):
    """Validate selection and supervision without materializing training inputs."""
    summary = summarize(discovery, context)
    action_ids = config['action_ids']
    catalog = {row['action_id']: row for row in summary['actions']}
    if (not isinstance(action_ids, list) or not action_ids or
            any(type(identity) is not int or identity not in catalog for identity in action_ids) or
            len(set(action_ids)) != len(action_ids)):
        raise ValueError('Select distinct action IDs from this analysis action catalog.')
    include_originals = config.get('include_originals', summary['defaults']['include_originals'])
    if type(include_originals) is not bool:
        raise ValueError('include_originals must be true or false.')
    task = discovery['task']
    training_data, base_targets, dataset_id = None, None, None
    if task == 'classification':
        if 'training_data' in config:
            raise ValueError('Classification uses the analyzed image dataset and its reviewed labels.')
        input_count = context['preview']['input_count']
        labels = discovery['config'].get('label_map', {})
        base_targets = []
        for index in range(input_count):
            target = labels.get(str(index))
            if not ((type(target) is int and target >= 0) or
                    (isinstance(target, str) and target.strip() and len(target) <= 512)):
                raise ValueError('Every analyzed image needs its original reviewed target label before adaptation.')
            base_targets.append(target)
    elif task == 'summarization':
        changed_facts = {str(action_id) for action_id in action_ids
                         if catalog[action_id]['action'].get('name') == 'negate_sentence'}
        if summary['defaults'].get('uses_dataset') and 'training_data' not in config:
            dataset = context['summary_dataset']
            input_count, dataset_id = dataset['count'], dataset['id']
            if dataset['reference_count'] != input_count:
                raise ValueError('Every training article needs a nonempty reference summary.')
            if changed_facts:
                raise ValueError('negate_sentence changes facts; supply training JSON with a corrected '
                                 'action_summaries target for every article, or remove that action.')
        else:
            training_data = _records(config.get('training_data', [
                {'text': summary['defaults']['source_text'], 'summary': summary['defaults']['reference_summary']}]))
            input_count = len(training_data)
        for index, record in enumerate(training_data or []):
            if (not isinstance(record, dict) or not {'text', 'summary'} <= set(record) or
                    set(record) - {'text', 'summary', 'action_summaries'}):
                raise ValueError('Each training record needs text and summary, with optional action_summaries.')
            _text(record['text'], 'Training text', MAX_SOURCE_CHARS if summary['defaults'].get('uses_dataset') else 20000)
            _text(record['summary'], 'Reference summary', 50000)
            overrides = record.get('action_summaries', {})
            if (not isinstance(overrides, dict) or any(
                    not isinstance(key, str) or not re.fullmatch(r'0|[1-9][0-9]*', key) or int(key) not in catalog
                    for key in overrides)):
                raise ValueError('action_summaries must map catalog action IDs to corrected summaries.')
            for value in overrides.values():
                _text(value, 'Corrected action summary', 50000)
            if not changed_facts <= overrides.keys():
                raise ValueError('negate_sentence changes facts; supply an action_summaries target for every training record.')
    elif task == 'generation':
        if include_originals:
            raise ValueError('Generation cannot include originals without an explicit desired original image.')
        records = _records(config.get('training_data'))
        input_count = len(records)
        training_data = []
        covered, pairs, inputs = set(), set(), set()
        dataset = context.get('generation_dataset')
        if dataset:
            dataset_id = dataset['id']
        for index, record in enumerate(records):
            if (not isinstance(record, dict) or not {'action_id', 'image_path'} <= set(record) or
                    set(record) - {'input_index', 'action_id', 'image_path', 'prompt'}):
                raise ValueError('Each desired-image record needs input_index, action_id and image_path.')
            action_id = record['action_id']
            if type(action_id) is not int or action_id not in action_ids:
                raise ValueError('Desired-image records must use only selected action IDs.')
            if dataset and 'input_index' not in record:
                raise ValueError('Dataset targets require input_index to identify the prompt for each desired image.')
            input_index = record.get('input_index', 0)
            if type(input_index) is not int or not 0 <= input_index < context['preview']['input_count']:
                raise ValueError('Desired-image input_index is outside this analysis.')
            if dataset and 'prompt' in record:
                raise ValueError('Dataset targets use the selected prompt and action combination; a prompt override would change that pair.')
            pair = input_index, action_id
            if dataset and pair in pairs:
                raise ValueError('Each prompt/action pair must appear only once in the training data.')
            target = _image_path(record['image_path'])
            prompt = _text(record['prompt'], 'Reference prompt', 20000) if 'prompt' in record else None
            training_data.append(dict(input_index=input_index, action_id=action_id, image_path=target, prompt=prompt))
            covered.add(action_id)
            pairs.add(pair)
            inputs.add(input_index)
        if covered != set(action_ids):
            raise ValueError('Every selected action needs at least one explicitly supplied desired training image.')
        if dataset:
            input_count = len(inputs)
    else:
        raise ValueError('Unsupported adaptation task.')
    example_count = len(training_data) if task == 'generation' else input_count * (len(action_ids) + int(include_originals))
    if not example_count:
        raise ValueError('Adaptation needs at least one training example.')
    selection = dict(action_ids=list(action_ids), actions=[copy.deepcopy(catalog[identity]) for identity in action_ids],
                     input_count=input_count, example_count=example_count, include_originals=include_originals)
    plan = dict(kind=task, action_ids=list(action_ids), input_count=input_count,
                example_count=example_count, include_originals=include_originals)
    if dataset_id:
        plan['dataset_id'] = dataset_id
        selection['dataset'] = copy.deepcopy(context['generation_dataset' if task == 'generation' else 'summary_dataset'])
    result = dict(training_plan=plan, adaptation_selection=selection)
    if base_targets is not None:
        result['base_targets'] = base_targets
    if training_data is not None:
        result['training_data'] = copy.deepcopy(training_data)
    return result


class TrainingExamples(Sequence):
    """A sized training corpus whose inputs exist only while a caller uses them.

    The report stores base records and the action axis, never their Cartesian
    product. ``target`` lets a trainer validate supervision before loading any
    image; integer indexing materializes exactly one input/action combination.
    """

    def __init__(self, job, context, input_loader, check=None):
        self.job, self.context, self.input_loader = job, context, input_loader
        self.check = check
        self.plan = job.get('training_plan')
        self.truncate_inputs = bool(context['preview']['config']['task'] == 'summarization' and
                                    context['preview']['config'].get('dataset_id'))
        self._last_article_index, self._last_article = None, None
        self.actions = {row['action_id']: row['action']
                        for row in job.get('adaptation_selection', {}).get('actions', [])}
        self.action_ids = ([None] if self.plan and self.plan['include_originals'] else [])
        if self.plan:
            self.action_ids += self.plan['action_ids']

    def __len__(self):
        return self.plan['example_count'] if self.plan else len(self.job['examples'])

    def _check(self):
        if self.check is not None:
            self.check()

    def _index(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError('Training example index is outside this corpus.')
        return index

    def _coordinates(self, index):
        input_index, action_index = divmod(index, len(self.action_ids))
        return input_index, self.action_ids[action_index]

    def _article(self, index):
        if self.plan.get('dataset_id'):
            if index != self._last_article_index:
                self._last_article = self.context['summary_row'](index)
                self._last_article_index = index
            return self._last_article
        return self.job['training_data'][index]

    def target(self, index):
        self._check()
        index = self._index(index)
        if self.plan is None:
            return self.job['examples'][index]['target']
        if self.plan['kind'] == 'generation':
            return self.job['training_data'][index]['image_path']
        input_index, action_id = self._coordinates(index)
        if self.plan['kind'] == 'classification':
            return self.job['base_targets'][input_index]
        record = self._article(input_index)
        return record.get('action_summaries', {}).get(str(action_id), record['summary'])

    def __getitem__(self, index):
        self._check()
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        index = self._index(index)
        target = self.target(index)
        if self.plan is None:
            record = self.job['examples'][index]
            source = self.input_loader(self.context, record['input_index'], record['action_id'])
        elif self.plan['kind'] == 'generation':
            record = self.job['training_data'][index]
            source = record.get('prompt')
            if source is None:
                source = self.input_loader(self.context, record.get('input_index', 0), record['action_id'])
        else:
            input_index, action_id = self._coordinates(index)
            if self.plan['kind'] == 'classification' or self.plan.get('dataset_id'):
                source = self.input_loader(self.context, input_index, action_id)
            else:
                source = self.job['training_data'][input_index]['text']
                if action_id is not None:
                    seed = (self.job['settings']['seed'] + input_index * 1000003 + action_id) % 2**32
                    source = apply_text(source, self.actions[action_id], random.Random(seed))
        self._check()
        return dict(input=source, target=target)
