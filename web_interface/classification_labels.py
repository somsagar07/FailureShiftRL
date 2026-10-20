"""Retain dataset class names and match them to a model's declared classes."""
import re
import json
import math
from functools import lru_cache
from pathlib import Path


_LABEL_COLUMNS = {
    'label', 'labels', 'target', 'targets', 'class', 'classes', 'category', 'categories',
    'class_name', 'class_names', 'class_id', 'class_label', 'label_name', 'label_id',
    'category_name', 'category_id', 'fine', 'coarse', 'fine_label', 'fine_labels',
    'coarse_label', 'coarse_labels',
}
_METADATA_COLUMNS = {'camera', 'camera_id', 'source', 'image_source', 'split', 'site',
                     'location', 'origin', 'device', 'annotator', 'annotator_id'}


def label_columns(features):
    """Include named categorical features and scalar fields conventionally used as labels."""
    from datasets import ClassLabel, Value
    return [name for name, feature in features.items()
            if isinstance(feature, ClassLabel)
            or (name.lower() in _LABEL_COLUMNS and isinstance(feature, Value))]


def extract_dataset_labels(records, column=None):
    features = getattr(records, 'features', {})
    columns = label_columns(features) if features else []
    result = dict(column=None, values={}, original_values={}, available_columns=columns,
                  loaded=0, missing=len(records), diagnostic='no_label_column',
                  message='No dataset label column found. Select a label column, upload JSON or enter labels.')
    if not features:
        if column is not None:
            raise ValueError('Selected label column does not exist in this dataset.')
        return result
    from datasets import ClassLabel, Value
    if column is not None:
        if not isinstance(column, str) or column not in features:
            raise ValueError('Selected label column does not exist in this dataset.')
        if not isinstance(features[column], (ClassLabel, Value)):
            raise ValueError('Selected label column must contain a single class per image.')
        if column not in columns:
            result['available_columns'] = [*columns, column]
    elif len(columns) == 1:
        column = columns[0]
    else:
        canonical = [name for name in columns if name.lower() in ('label', 'labels')]
        if len(canonical) == 1 and all(name == canonical[0] or name.lower() in _METADATA_COLUMNS for name in columns):
            column = canonical[0]
        else:
            if columns:
                result['diagnostic'] = 'ambiguous_label_columns'
                result['message'] = f'Multiple label columns found ({", ".join(columns)}). Select the label column to use.'
            return result
    result['column'] = column
    feature = features[column]
    names = getattr(feature, 'names', None)
    if names is not None:
        result['class_names'] = list(names)
    unnamed_numeric = 0
    for index, value in enumerate(records[column]):
        if isinstance(value, str) and value.strip() and len(value) <= 512:
            result['original_values'][str(index)] = value.strip()
        elif type(value) in (int, float) and math.isfinite(value):
            result['original_values'][str(index)] = value
            if names is None:
                unnamed_numeric += 1
        if names is not None:
            value = names[value] if type(value) is int and 0 <= value < len(names) else None
        if isinstance(value, str) and value.strip() and len(value) <= 512:
            result['values'][str(index)] = value.strip()
    result['loaded'] = len(result['values'])
    result['missing'] = len(records) - result['loaded']
    if unnamed_numeric:
        result['diagnostic'] = 'unnamed_numeric_labels'
        result['message'] = 'Dataset labels contain numeric IDs without class names. Provide class names or an explicit mapping to this model.'
    elif result['missing']:
        result['diagnostic'] = 'missing_labels'
        result['message'] = f'{result["missing"]} inputs have no usable dataset class name. Upload JSON or enter the missing labels.'
    else:
        result['diagnostic'] = ''
        result['message'] = ''
    return result


def _name(value):
    return re.sub(r'[\s_-]+', ' ', value.strip()).lower()


@lru_cache(maxsize=1)
def _pet_class_names():
    return json.loads(Path(__file__).with_name('imagenet_pet_labels.json').read_text())['groups']


def model_label_groups(model_labels):
    """Match complete, declared ImageNet pet vocabularies without assuming class IDs."""
    names = {}
    for index, name in model_labels.items():
        if str(index).isdigit() and isinstance(name, str):
            names.setdefault(name.strip(), set()).add(int(index))
    groups = {}
    for group, members in _pet_class_names().items():
        ids = []
        for member in members:
            # HF uses full synonym lists; some checkpoints use the first name only.
            # Preserve case: Cardigan (corgi) and cardigan (clothing) are distinct.
            matches = names.get(member, set()) | names.get(member.split(',')[0], set())
            if len(matches) != 1:
                break
            ids.extend(matches)
        if len(set(ids)) == len(members):
            groups[group] = sorted(ids)
    return groups


def resolve_target(target, model_labels, groups=None):
    if type(target) is int and target >= 0 and str(target) in model_labels:
        return target
    if isinstance(target, str) and target.strip():
        matches = [int(key) for key, name in model_labels.items() if _name(name) == _name(target)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError('Ambiguous class name; enter a model class ID.')
        if _name(target) in (groups or {}):
            return _name(target)
    raise ValueError('Label does not match this model. Enter a model class ID or class name.')


def _imagenet_class_offset(dataset_labels, model_labels):
    """Verify the entire class order before resolving ImageNet's duplicate cranes.

    The official dataset names class 517 'crane2'; the model presets call both
    134 and 517 'crane'. MobileNet also has a leading background output. A class
    count or numeric dataset ID alone is insufficient to establish this mapping.
    """
    names = dataset_labels.get('class_names')
    if (not isinstance(names, list) or len(names) != 1000 or
            names[0] != 'tench, Tinca tinca' or names[134] != 'crane' or names[517] != 'crane2'):
        return None
    offset = len(model_labels) - len(names)
    if offset not in (0, 1) or (offset == 1 and _name(model_labels.get('0', '')) != 'background'):
        return None
    for index, name in enumerate(names):
        expected = 'crane' if index == 517 else name
        actual = model_labels.get(str(index+offset))
        if not isinstance(expected, str) or not isinstance(actual, str) or _name(expected) != _name(actual):
            return None
    return offset


def match_dataset_labels(dataset_labels, model_labels, total):
    model_labels = {str(key): value for key, value in model_labels.items()
                    if str(key).isdigit() and isinstance(value, str)}
    values = {str(index): value for index, value in dataset_labels.get('values', {}).items()
              if isinstance(value, str) and value.strip()}
    groups = model_label_groups(model_labels)
    offset = _imagenet_class_offset(dataset_labels, model_labels)
    original_values = dataset_labels.get('original_values', {})
    mapped = {}
    resolved = {}
    for index, value in values.items():
        source_id = original_values.get(index)
        if (offset is not None and type(source_id) is int and 0 <= source_id < 1000
                and value == dataset_labels['class_names'][source_id]):
            mapped[index] = source_id + offset
            continue
        if value not in resolved:
            try:
                resolved[value] = resolve_target(value, model_labels, groups)
            except ValueError:
                resolved[value] = None
        if resolved[value] is not None:
            mapped[index] = resolved[value]
    unmatched_names = sorted({values[index] for index in values if index not in mapped})
    return dict(column=dataset_labels.get('column'), values=values, label_map=mapped,
                model_labels=model_labels, groups=groups, loaded=len(values), ready=len(mapped), total=total,
                missing=max(0, total-len(values)), unmatched=len(values)-len(mapped),
                unmatched_classes=unmatched_names[:3], unmatched_class_count=len(unmatched_names),
                original_values=dataset_labels.get('original_values', {}),
                available_columns=dataset_labels.get('available_columns', []),
                diagnostic=dataset_labels.get('diagnostic', ''),
                message=dataset_labels.get('message', ''))
