import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import image_actions, select_actions, classification_outcome, rouge_l


class CoreTests(unittest.TestCase):
    def test_original_action_order_and_values(self):
        actions = image_actions()
        self.assertEqual(len(actions), 125)
        self.assertEqual(actions[0], {'id': 0, 'rotation': -10, 'darkness': .1, 'saturation': .5})
        self.assertEqual(actions[124], {'id': 124, 'rotation': 10, 'darkness': .5, 'saturation': .7})

    def test_random_search_is_bounded_and_reproducible(self):
        actions = image_actions()
        first = select_actions(actions, 12, 42, 'random')
        self.assertEqual(first, select_actions(actions, 12, 42, 'random'))
        self.assertEqual(len({x['id'] for x in first}), 12)
        self.assertEqual(len(select_actions(actions, 200, 42, 'grid')), 125)
        self.assertEqual(actions[0]['id'], 0)

    def test_unlabelled_changes_are_only_candidates(self):
        self.assertEqual(classification_outcome(2, 3, None), 'candidate')
        self.assertEqual(classification_outcome(2, 2, None), 'stable')

    def test_baseline_errors_are_not_new_failures(self):
        self.assertEqual(classification_outcome(2, 3, 4), 'baseline_error')
        self.assertEqual(classification_outcome(2, 3, 2), 'label_violation')
        self.assertEqual(classification_outcome(2, 2, 2), 'stable')

    def test_summary_metric_uses_order_and_handles_empty_text(self):
        self.assertEqual(rouge_l('a b c', 'a b c'), 1)
        self.assertAlmostEqual(rouge_l('a b c', 'c b a'), 1/3)
        self.assertEqual(rouge_l('', 'words'), 0)

    def test_selected_image_dimensions_use_identity_when_removed(self):
        from core import actions_for
        actions = actions_for('classification', {'rotation': [-5, 20], 'darkness': [.2], 'saturation': []})
        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[1], dict(id=1, rotation=20, darkness=.2, saturation=1))

    def test_disabled_dimensions_do_not_change_the_image(self):
        from core import actions_for, apply_image
        from PIL import Image
        original = Image.new('RGB', (8, 8), (100, 140, 200))
        action = actions_for('classification', {'rotation': [0], 'darkness': [], 'saturation': []})[0]
        self.assertEqual(apply_image(original, action).tobytes(), original.tobytes())

    def test_selected_text_actions_preserve_original_ids(self):
        from core import actions_for
        self.assertEqual(actions_for('summarization', {'actions': ['to_lowercase']}),
                         [dict(id=2, name='to_lowercase')])

    def test_generation_uses_added_and_removed_values(self):
        from core import actions_for
        actions = actions_for('generation', {'description': ['careful'], 'profession': ['doctor', 'artist'], 'setting': ['a hospital']})
        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0]['profession'], 'doctor')
        self.assertEqual(actions[1]['setting'], 'a hospital')

    def test_invalid_action_spaces_are_rejected(self):
        from core import actions_for
        cases = [
            ('classification', {'rotation': [], 'darkness': [], 'saturation': []}),
            ('classification', {'rotation': [float('nan')], 'darkness': [], 'saturation': []}),
            ('classification', {'rotation': [True], 'darkness': [], 'saturation': []}),
            ('classification', {'rotation': [181], 'darkness': [], 'saturation': []}),
            ('classification', {'rotation': [0], 'darkness': [-1], 'saturation': []}),
            ('classification', {'rotation': [0], 'darkness': [], 'saturation': [4]}),
            ('classification', {'rotation': list(range(25)), 'darkness': [i/25 for i in range(25)], 'saturation': []}),
            ('classification', {'rotation': [0], 'darkness': [], 'saturation': [], 'blur': [1]}),
            ('summarization', {'actions': []}),
            ('summarization', {'actions': ['unknown_function']}),
            ('generation', {'description': [], 'profession': ['doctor'], 'setting': ['lab']}),
        ]
        for task, space in cases:
            with self.subTest(task=task, space=space), self.assertRaises(ValueError):
                actions_for(task, space)

    def test_duplicate_values_are_not_evaluated_twice(self):
        from core import actions_for
        actions = actions_for('classification', {'rotation': [0, 0.0], 'darkness': [.1, .1], 'saturation': []})
        self.assertEqual(len(actions), 1)


if __name__ == '__main__':
    unittest.main()
