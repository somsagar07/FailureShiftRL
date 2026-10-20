import importlib
import importlib.util
import unittest

import core


class GenerationStyleTests(unittest.TestCase):
    def test_default_groups_cover_the_cartesian_product(self):
        space = core.default_action_space('generation')
        self.assertEqual(list(space), ['style', 'lighting', 'detail'])
        self.assertEqual([len(values) for values in space.values()], [5, 5, 5])
        actions = core.actions_for('generation')
        self.assertEqual(len(actions), 125)
        self.assertEqual(len({action['id'] for action in actions}), 125)
        self.assertEqual(len({(action['style'], action['lighting'], action['detail']) for action in actions}),125)
        for key, values in space.items():
            self.assertEqual({action[key] for action in actions}, set(values))

    def test_saved_style_only_actions_remain_usable(self):
        self.assertEqual(core.actions_for('generation', {'styles': ['3D render', 'watercolor', 'watercolor']}),
                         [{'id': 4, 'style': '3D render'}, {'id': 1, 'style': 'watercolor'}])

    def test_removed_groups_are_omitted_and_subset_ids_remain_stable(self):
        space = core.default_action_space('generation')
        full = core.actions_for('generation', space)
        chosen = {key:[values[-1]] for key,values in space.items()}
        self.assertEqual(core.actions_for('generation', chosen), [full[-1]])
        space['lighting'] = []
        two_groups = core.actions_for('generation', space)
        self.assertEqual(len(two_groups),25)
        self.assertTrue(all(action['lighting'] is None for action in two_groups))
        space['detail'] = []
        only_style = core.actions_for('generation', space)
        self.assertEqual(len(only_style),5)
        for action in only_style:
            text = core.apply_prompt('two green cats', action)
            self.assertNotIn('lighting',text)
            self.assertNotIn('detail',text)
            self.assertTrue(text.endswith('two green cats'))
        space['style'] = []
        space['lighting'] = ['soft']
        action = core.actions_for('generation', space)[0]
        self.assertEqual(core.apply_prompt('two green cats',action), 'Soft lighting. Scene: two green cats')

    def test_combination_preserves_the_full_scene_and_describes_every_enabled_group(self):
        space = {'style':['watercolor'], 'lighting':['soft'], 'detail':['intricate']}
        actions = core.actions_for('generation',space)
        prompt = '  Two green cats to the left of one red bowl.\nA sign says {home}.  '
        self.assertEqual(core.apply_prompt(prompt,actions[0]),
                         'Watercolor style. Soft lighting. Detail level: intricate. Scene: '+prompt)

    def test_group_values_are_validated_and_at_least_one_group_is_required(self):
        default = core.default_action_space('generation')
        for space in ({'style':[], 'lighting':[], 'detail':[]},
                      {**default,'lighting':['add three cats']},
                      {**default,'detail':[float('nan')]},
                      {'style':['watercolor'],'lighting':['soft']}):
            with self.subTest(space=space), self.assertRaises(ValueError):
                core.actions_for('generation',space)

    def test_style_wrapper_preserves_scene_counts_relations_and_literal_braces(self):
        self.assertTrue(callable(getattr(core, 'apply_prompt', None)), 'A shared prompt transformer is required')
        prompt = '  Two blue cups to the left of one red bowl.\nA sign reads {home}.  '
        self.assertEqual(core.apply_prompt(prompt, {'id': 1, 'style': 'watercolor'}),
                         'Watercolor style. Scene: ' + prompt)
        self.assertEqual(core.apply_prompt(prompt, {'id': 4, 'style': '3D render'}),
                         '3D render style. Scene: ' + prompt)

    def test_arbitrary_content_cannot_be_submitted_as_a_style(self):
        for space in ({'styles': []}, {'styles': ['add three cats']}, {'styles': ['Watercolor']},
                      {'styles': ['photograph'], 'profession': ['doctor']}, {'styles': [float('nan')]}):
            with self.subTest(space=space), self.assertRaises(ValueError):
                core.actions_for('generation', space)

    def test_explicit_legacy_template_spaces_remain_usable(self):
        self.assertTrue(callable(getattr(core, 'apply_prompt', None)), 'Legacy actions use the shared transformer')
        action = core.actions_for('generation', {'description': ['careful'], 'profession': ['doctor'],
                                                  'setting': ['a hospital']})[0]
        self.assertEqual(core.apply_prompt('A {description} {profession} in {setting}.', action),
                         'A careful doctor in a hospital.')
        self.assertEqual(core.apply_prompt('{"scene": "A {description} {profession} in {setting}."}', action),
                         '{"scene": "A careful doctor in a hospital."}')


class PromptCheckTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('generation_prompts'), 'Shared prompt-check normalization is required')
        self.normalize = importlib.import_module('generation_prompts').normalize_prompt_checks

    def test_pairs_and_dict_checks_preserve_order_and_optional_skill(self):
        self.assertEqual(self.normalize([['How many cups?', 'two'],
                                         {'question': 'Is the bowl red?', 'answer': 'Yes', 'skill': 'attribute'}]),
                         [{'question': 'How many cups?', 'answer': 'two'},
                          {'question': 'Is the bowl red?', 'answer': 'Yes', 'skill': 'attribute'}])
        self.assertEqual(self.normalize(None), [])
        self.assertEqual(self.normalize([]), [])

    def test_normalized_checks_are_independent_of_caller_mutation(self):
        source = [{'question': ' Is there a cup? ', 'answer': ' Yes ', 'skill': ' object '}]
        normalized = self.normalize(source)
        self.assertEqual(normalized, [{'question': 'Is there a cup?', 'answer': 'Yes', 'skill': 'object'}])
        source[0]['answer'] = 'No'
        self.assertEqual(normalized[0]['answer'], 'Yes')

    def test_malformed_checks_are_rejected_instead_of_dropped_or_coerced(self):
        invalid = [float('nan'), '', {}, [None], [['question']], [['q', 'a', 'extra']],
                   [{'question': 'q'}], [{'answer': 'a'}], [['', 'Yes']], [['q', None]],
                   [['q', float('nan')]], [['q', True]], [{'question': 'q', 'answer': 'a', 'skill': ''}],
                   [{'question': 'q', 'answer': 'a', 'skill': float('nan')}],
                   [{'question': 'q', 'answer': 'a', 'unsupported': 'would be dropped'}],
                   [['q', 'a'], ['bad', '']], [['q\x00', 'a']], [['q' * 4001, 'a']],
                   [['q', 'a' * 1001]], [['q', 'a']] * 129]
        for value in invalid:
            with self.subTest(value=repr(value)[:120]), self.assertRaises(ValueError):
                self.normalize(value)


if __name__ == '__main__':
    unittest.main()
