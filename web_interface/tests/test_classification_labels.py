"""Dataset labels must keep their meaning when model class IDs differ."""
import unittest
import json
import builtins
from pathlib import Path
from unittest.mock import patch

from classification_labels import extract_dataset_labels, label_columns, match_dataset_labels, resolve_target


class DatasetLabelTests(unittest.TestCase):
    def test_imagenet_dataset_classes_map_with_background_offset_and_duplicate_cranes(self):
        from datasets import ClassLabel
        names = [f'class {index}' for index in range(1000)]
        names[0], names[134], names[517] = 'tench, Tinca tinca', 'crane', 'crane2'
        records = self.dataset({'label': [0, 134, 517, 999, -1]}, {'label': ClassLabel(names=names)})
        extracted = extract_dataset_labels(records)
        for offset in (0, 1):
            labels = {str(index + offset): ('crane' if index == 517 else name) for index, name in enumerate(names)}
            if offset:
                labels['0'] = 'background'
            with self.subTest(offset=offset):
                result = match_dataset_labels(extracted, labels, 5)
                self.assertEqual(result['label_map'], {'0': offset, '1': 134+offset, '2': 517+offset, '3': 999+offset})
                self.assertEqual(result['ready'], 4)
                self.assertEqual(result['missing'], 1, 'Unlabeled test images must remain unlabeled')
                with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                    resolve_target('crane', labels)
                reordered = dict(labels)
                reordered[str(10+offset)], reordered[str(11+offset)] = reordered[str(11+offset)], reordered[str(10+offset)]
                unmatched = match_dataset_labels(extracted, reordered, 5)
                self.assertNotIn('1', unmatched['label_map'], 'An inconsistent full vocabulary must not imply ID alignment')
                self.assertNotIn('2', unmatched['label_map'])
        without_names = match_dataset_labels({'values': extracted['values'], 'original_values': extracted['original_values']}, labels, 5)
        self.assertNotIn('1', without_names['label_map'], 'Numeric dataset IDs alone never resolve ambiguous names')

    def test_local_folder_records_do_not_require_the_hugging_face_datasets_package(self):
        original_import = builtins.__import__

        def without_datasets(name, *args, **kwargs):
            if name == 'datasets':
                raise ImportError('Hugging Face datasets is not installed')
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=without_datasets):
            result = extract_dataset_labels([Path('cat.png'), Path('dog.png')])
        self.assertEqual(result['values'], {})
        self.assertEqual(result['missing'], 2)
        self.assertEqual(result['diagnostic'], 'no_label_column')

    def dataset(self, columns, features):
        from datasets import Dataset, Features
        return Dataset.from_dict(columns, features=Features(features))

    def test_beans_class_names_load_automatically_and_match_reordered_model_ids(self):
        from datasets import ClassLabel
        records = self.dataset({'labels': [0, 1, 2, 0]},
                               {'labels': ClassLabel(names=['angular_leaf_spot', 'bean_rust', 'healthy'])})
        extracted = extract_dataset_labels(records)
        result = match_dataset_labels(extracted, {'0': 'healthy', '1': 'bean rust', '2': 'angular leaf spot'}, 4)
        self.assertEqual(result['values'], {'0': 'angular_leaf_spot', '1': 'bean_rust', '2': 'healthy', '3': 'angular_leaf_spot'})
        self.assertEqual(result['label_map'], {'0': 2, '1': 1, '2': 0, '3': 2})
        self.assertEqual((result['loaded'], result['ready'], result['missing'], result['unmatched']), (4, 4, 0, 0))

    def test_canonical_label_is_chosen_over_unrelated_categorical_metadata(self):
        from datasets import ClassLabel, Value
        for label in (ClassLabel(names=['cat']), Value('string')):
            with self.subTest(feature=label):
                records = self.dataset({'labels': [0 if isinstance(label, ClassLabel) else 'cat'], 'camera': [0]},
                                       {'labels': label, 'camera': ClassLabel(names=['camera-a'])})
                self.assertEqual(extract_dataset_labels(records)['values'], {'0': 'cat'})
                self.assertEqual(set(label_columns(records.features)), {'labels', 'camera'})

    def test_ambiguous_taxonomies_require_a_column_and_explicit_choice_decodes_it(self):
        from datasets import ClassLabel
        for fine in ('fine_label', 'label'):
            with self.subTest(fine=fine):
                records = self.dataset({fine: [0], 'coarse_label': [0]},
                                       {fine: ClassLabel(names=['cat']), 'coarse_label': ClassLabel(names=['animal'])})
                result = extract_dataset_labels(records)
                self.assertEqual(result['values'], {})
                self.assertIsNone(result['column'])
                self.assertEqual(result['diagnostic'], 'ambiguous_label_columns')
                self.assertEqual(set(result['available_columns']), {fine, 'coarse_label'})
                self.assertIn('Multiple', result['message'])
                self.assertEqual(extract_dataset_labels(records, column='coarse_label')['values'], {'0': 'animal'})

    def test_string_label_synonyms_are_detected_and_explicit_scalar_fields_are_supported(self):
        from datasets import ClassLabel, Value
        records = self.dataset({'class_name': ['cat'], 'camera': [0], 'reviewed': ['dog']},
                               {'class_name': Value('string'), 'camera': ClassLabel(names=['camera-a']), 'reviewed': Value('string')})
        self.assertIn('class_name', label_columns(records.features))
        self.assertEqual(extract_dataset_labels(records, column='class_name')['values'], {'0': 'cat'})
        self.assertEqual(extract_dataset_labels(records, column='reviewed')['values'], {'0': 'dog'})
        single = self.dataset({'category': [' healthy ', None, '']}, {'category': Value('string')})
        result = extract_dataset_labels(single)
        self.assertEqual(result['values'], {'0': 'healthy'})
        self.assertEqual(result['missing'], 2)
        self.assertEqual(result['diagnostic'], 'missing_labels')

    def test_invalid_explicit_label_columns_fail_instead_of_falling_back(self):
        from datasets import ClassLabel, List, Value
        records = self.dataset({'label': [0], 'objects': [['cat']]},
                               {'label': ClassLabel(names=['cat']), 'objects': List(Value('string'))})
        for column in ('absent', 'objects', 3):
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, 'label column'):
                extract_dataset_labels(records, column=column)

    def test_numeric_labels_without_class_names_remain_unresolved_and_preserve_raw_values(self):
        from datasets import Value
        records = self.dataset({'label_id': [0, 1, None]}, {'label_id': Value('int64')})
        extracted = extract_dataset_labels(records)
        self.assertEqual(extracted['column'], 'label_id')
        self.assertEqual(extracted['values'], {})
        self.assertEqual(extracted['original_values'], {'0': 0, '1': 1})
        self.assertEqual(extracted['diagnostic'], 'unnamed_numeric_labels')
        self.assertIn('class names', extracted['message'])
        result = match_dataset_labels(extracted, {'0': 'cat', '1': 'dog'}, 3)
        self.assertEqual(result['label_map'], {})
        self.assertEqual((result['missing'], result['unmatched']), (3, 0))

    def test_null_and_invalid_class_ids_report_missing_labels(self):
        from datasets import ClassLabel
        records = self.dataset({'label': [0, -1, None]}, {'label': ClassLabel(names=['cat'])})
        result = extract_dataset_labels(records)
        self.assertEqual(result['values'], {'0': 'cat'})
        self.assertEqual(result['diagnostic'], 'missing_labels')
        self.assertEqual(result['missing'], 2)

    def test_absent_labels_report_missing_without_predicting_them(self):
        from datasets import Value
        records = self.dataset({'description': ['photo of a cat']}, {'description': Value('string')})
        result = extract_dataset_labels(records)
        self.assertEqual(result['values'], {})
        self.assertEqual(result['diagnostic'], 'no_label_column')
        self.assertEqual(result['missing'], 1)


def imagenet_labels(offset=0, short=False):
    """Offline model vocabulary with canonical pet classes and unrelated distractors."""
    pets = json.loads((Path(__file__).parents[1] / 'imagenet_pet_labels.json').read_text())['groups']
    labels = {str(i + offset): f'other-{i}' for i in range(1000)}
    for group, start in [('dog', 151), ('cat', 281)]:
        for index, name in enumerate(pets[group], start):
            labels[str(index + offset)] = name.split(',')[0] if short else name
    for index, name in {269: 'timber wolf, grey wolf', 273: 'dingo', 275: 'African hunting dog',
                        286: 'cougar', 292: 'tiger', 383: 'Madagascar cat',
                        474: 'cardigan', 934: 'hotdog, hot dog, red hot'}.items():
        labels[str(index + offset)] = name
    if offset:
        labels['0'] = 'background'
    return labels


class LabelMappingTests(unittest.TestCase):
    def test_full_dataset_resolves_each_repeated_class_once_including_unmatched_names(self):
        values = {str(index): ['cat', 'dog', 'unknown'][index % 3] for index in range(50000)}
        with patch('classification_labels.resolve_target', wraps=resolve_target) as resolve:
            result = match_dataset_labels({'values': values}, imagenet_labels(), 50000)
        self.assertEqual(resolve.call_count, 3)
        self.assertEqual((result['loaded'], result['ready'], result['unmatched']), (50000, 33334, 16666))
        self.assertEqual(result['label_map']['0'], 'cat')
        self.assertEqual(result['label_map']['1'], 'dog')
        self.assertNotIn('2', result['label_map'])

    def test_loaded_cifar_labels_distinguish_model_mismatches_from_missing_labels(self):
        names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        values = {str(index): names[index % 10] for index in range(20)}
        result = match_dataset_labels({'column': 'label', 'values': values}, imagenet_labels(), 20)
        self.assertEqual((result['loaded'], result['ready'], result['missing'], result['unmatched']), (20, 4, 0, 16))
        self.assertEqual(result['unmatched_classes'], ['airplane', 'automobile', 'bird'])
        self.assertEqual(result['unmatched_class_count'], 8)
        self.assertEqual(result['values'], values)
        self.assertEqual(result['label_map'], {'3': 'cat', '5': 'dog', '13': 'cat', '15': 'dog'})

    def test_beans_labels_never_become_imagenet_targets_by_numeric_position(self):
        result = match_dataset_labels({'values': {'0': 'angular_leaf_spot', '1': 'bean_rust', '2': 'healthy'}},
                                      imagenet_labels(), 3)
        self.assertEqual(result['label_map'], {})
        self.assertEqual((result['loaded'], result['ready'], result['missing'], result['unmatched']), (3, 0, 0, 3))
        self.assertEqual(result['unmatched_class_count'], 3)
        numeric = match_dataset_labels({'values': {'0': 0, '1': 1}}, {'0': 'cat', '1': 'dog'}, 2)
        self.assertEqual(numeric['label_map'], {}, 'dataset numeric IDs are not model IDs')

    def test_cat_dog_labels_are_ready_for_both_imagenet_class_counts_and_name_formats(self):
        values = {str(index): 'cat' if index % 2 else 'dog' for index in range(50)}
        for offset in (0, 1):
            for short in (False, True):
                with self.subTest(offset=offset, short=short):
                    result = match_dataset_labels({'column': 'labels', 'values': values},
                                                  imagenet_labels(offset, short), 50)
                    self.assertEqual(result['ready'], 50)
                    self.assertEqual(result['label_map'], values)
                    self.assertEqual(result['groups']['cat'], list(range(281 + offset, 286 + offset)))
                    self.assertEqual(result['groups']['dog'], list(range(151 + offset, 269 + offset)))
                    self.assertNotIn(474 + offset, result['groups']['dog'], 'cardigan clothing is not a corgi')
                    self.assertEqual(resolve_target(' CAT ', result['model_labels'], result['groups']), 'cat')

    def test_pet_groups_follow_class_names_when_model_ids_are_reordered(self):
        labels = {str(999 - int(index)): name for index, name in imagenet_labels().items()}
        result = match_dataset_labels({'values': {'0': 'cat', '1': 'dog'}}, labels, 2)
        self.assertEqual(result['ready'], 2)
        self.assertEqual(set(result['groups']['cat']), {999 - index for index in range(281, 286)})
        self.assertIn(999 - 264, result['groups']['dog'])
        self.assertNotIn(999 - 474, result['groups']['dog'])

    def test_incomplete_or_ambiguous_pet_vocabularies_still_need_review(self):
        for labels in (dict(imagenet_labels(), **{'284': 'LABEL_284'}),
                       dict(imagenet_labels(), **{'400': 'Persian cat'}),
                       {str(i): f'LABEL_{i}' for i in range(1000)}):
            with self.subTest(labels=labels.get('284')):
                result = match_dataset_labels({'values': {'0': 'cat'}}, labels, 1)
                self.assertEqual(result['ready'], 0)
                self.assertNotIn('cat', result['groups'])

    def test_names_map_to_model_ids_and_missing_or_incompatible_labels_need_review(self):
        result = match_dataset_labels(
            {'column': 'labels', 'values': {'0': 'cat', '1': 'dog', '2': 'bird'}},
            {'0': 'dog', '4': 'cat'}, 4)
        self.assertEqual(result['label_map'], {'0': 4, '1': 0})
        self.assertEqual(result['loaded'], 3)
        self.assertEqual(result['ready'], 2)
        self.assertEqual(result['total'], 4)
        self.assertEqual(result['values']['2'], 'bird')
        self.assertEqual((result['missing'], result['unmatched']), (1, 1))

    def test_cosmetic_spelling_matches_but_ambiguous_and_generic_classes_do_not(self):
        result = match_dataset_labels(
            {'values': {'0': 'Angular_leaf_spot', '1': 'CAT', '2': 'dog'}},
            {'0': 'angular leaf spot', '1': 'Cat', '2': 'cat', '3': 'LABEL_3'}, 3)
        self.assertEqual(result['label_map'], {'0': 0})
        self.assertEqual(result['ready'], 1)
        with self.assertRaises(ValueError):
            resolve_target('cat', result['model_labels'])
        self.assertEqual(resolve_target(2, result['model_labels']), 2)
        for value in (True, -1, 4, 'dog'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_target(value, result['model_labels'])

    def test_no_labels_never_manufactures_targets_from_model_predictions(self):
        result = match_dataset_labels({'values': {}}, {'0': 'cat'}, 50)
        self.assertEqual(result['label_map'], {})
        self.assertEqual(result['ready'], 0)
        self.assertEqual(result['total'], 50)

    def test_case_normalization_matches_manual_editor_without_transliteration(self):
        self.assertEqual(resolve_target('straße', {'0': 'Straße'}), 0)
        with self.assertRaises(ValueError):
            resolve_target('STRASSE', {'0': 'Straße'})


if __name__ == '__main__':
    unittest.main()
