"""Fine-tuning defaults agree across API and direct training entrypoints."""
import unittest

from discovery import DiscoveryManager
from fine_tuning import _settings


class TrainingSettingsTests(unittest.TestCase):
    def test_omitted_learning_rate_uses_the_training_task(self):
        manager = DiscoveryManager.__new__(DiscoveryManager)
        for task, expected in [('classification', .001), ('summarization', .00002), ('generation', .0001)]:
            with self.subTest(task=task):
                discovery = {'task': task, 'config': {'seed': 42, 'generation_size': 256}}
                self.assertEqual(manager._finetune_settings({}, discovery)['learning_rate'], expected)
                self.assertEqual(_settings({}, task=task)['learning_rate'], expected)

    def test_explicit_learning_rate_survives_task_defaults(self):
        manager = DiscoveryManager.__new__(DiscoveryManager)
        config = {'learning_rate': .000007}
        for task in ('classification', 'summarization', 'generation'):
            with self.subTest(task=task):
                discovery = {'task': task, 'config': {'seed': 42, 'generation_size': 256}}
                self.assertEqual(manager._finetune_settings(config, discovery)['learning_rate'], .000007)
                self.assertEqual(_settings(config, task=task)['learning_rate'], .000007)
        self.assertEqual(config, {'learning_rate': .000007})

    def test_legacy_settings_call_retains_its_default(self):
        self.assertEqual(_settings({})['learning_rate'], .0001)
