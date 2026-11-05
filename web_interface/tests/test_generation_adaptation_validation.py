"""Exercise held-out validation through the real adaptation worker and reports."""
import json
import copy
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from discovery import DiscoveryManager
from generation_failure import combined_score
import test_generation_dataset_discovery as dataset_tests


class GenerationAdaptationValidationTests(unittest.TestCase):
    setUp = dataset_tests.GenerationDatasetDiscoveryTests.setUp
    discover = dataset_tests.GenerationDatasetDiscoveryTests.discover

    def configure(self):
        discovery = self.discover()
        target = self.root / 'target.png'
        Image.new('RGB', (48, 32), 'green').save(target)
        self.events, self.evaluators = [], []
        self.config = dict(discovery_id=discovery['id'], action_ids=[0],
            training_data=[dict(input_index=0, action_id=0, image_path=str(target))],
            cleanup_after_finetune=False, batch_size=1,
            validation_data=[dict(prompt=f'A held-out scene with {i} red apples') for i in range(8)])
        owner = self

        class Evaluator:
            def __init__(self, model, config, directory, progress=None):
                self.after = model['source'] == 'local'
                self.model, self.config, self.directory = model, config, directory
                self.closed = False
                owner.events.append('load-after' if self.after else 'load-before')
                owner.evaluators.append(self)

            def describe(self):
                return dict(revision='a' * 40, generation_precision='float32',
                    scorer=dict(name='changed' if self.after and getattr(owner, 'changed_scorer', False)
                                else 'fixture/judge', revision='b' * 40))

            def predict(self, value, **options):
                if getattr(owner, 'cancel_phase', '') == ('after' if self.after else 'before'):
                    owner.manager._events[owner.manager.active].set()
                filename = options['artifact_name'] + '.png'
                Image.new('RGB', (8, 8)).save(self.directory / filename)
                severity = .2 if self.after or options['action'] is None else .8
                return dict(image=filename, prompt=value, seed=options['seed'],
                    failure_scores=dict(content=severity, instructions=.1, visual=.1, bias=None))

            def score(self, output, baseline):
                return combined_score(output, baseline)

            def close(self):
                self.closed = True
                owner.events.append('close-after' if self.after else 'close-before')

        def train(model, task, examples, directory, settings, progress, cancelled):
            self.events.append('train')
            self.training_model, self.training_settings = model, settings
            self.assertTrue(all(e.closed for e in self.evaluators))
            self.assertEqual(len(examples), 1)
            checkpoint = directory / 'checkpoint'
            checkpoint.mkdir()
            if getattr(self, 'cancel_after_save', False):
                self.manager._events[self.manager.active].set()
            return dict(checkpoint=str(checkpoint), steps=1, training_loss=[.2])

        self.manager.evaluator_factory, self.manager.finetune_fn = Evaluator, train
        return discovery

    def finish(self, **updates):
        job = self.manager.start_finetune({**self.config, **updates})
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            report = self.manager.get_finetune(job['id'])
            if report['status'] in ('completed', 'error', 'cancelled'):
                return report
            time.sleep(.01)
        self.fail('Adaptation did not finish')

    def test_fresh_matched_panels_surround_training_and_publish_images(self):
        discovery = self.configure()
        report = self.finish(generation_framing='preserve', generation_size='auto')
        self.assertEqual(report['status'], 'completed', report.get('error'))
        self.assertEqual(self.events, ['load-before', 'close-before', 'train', 'load-after', 'close-after'])
        self.assertEqual(self.training_model['revision'], 'a' * 40)
        self.assertEqual(self.training_settings['generation_size'], 'auto')
        validation = report['validation']
        self.assertEqual(validation['status'], 'completed')
        self.assertEqual(validation['comparison']['verdict'], 'improved')
        self.assertEqual(validation['before']['settings'], validation['after']['settings'])
        self.assertEqual(validation['before']['settings']['generation_size'], discovery['config']['generation_size'])
        before, after = validation['before']['pairs'], validation['after']['pairs']
        self.assertEqual([(r['prompt'], r['seed'], r['checks']) for r in before],
                         [(r['prompt'], r['seed'], r['checks']) for r in after])
        image = before[0]['output']['image']
        self.assertTrue(image.startswith(f"/discoveries/{report['id']}/artifacts/validation-before-"))
        self.assertTrue(self.manager.artifact(report['id'], image.rsplit('/', 1)[1]).is_file())
        persisted = json.loads(self.manager.artifact(report['id'], 'report.json').read_text())
        self.assertEqual(persisted['validation'], validation)
        restarted = DiscoveryManager(self.previews, self.analyses, self.manager.directory)
        self.assertEqual(restarted.get_finetune(report['id'])['validation'], validation)

    def test_defaults_do_not_copy_preview_resolution_into_training(self):
        self.configure()
        self.config.pop('validation_data')
        report = self.finish()
        self.assertEqual(report['status'], 'completed', report.get('error'))
        self.assertEqual(self.training_settings['generation_size'], 'auto')
        self.assertEqual(self.training_settings['generation_framing'], 'preserve')
        self.assertEqual(report['validation']['status'], 'not_requested')
        self.assertEqual(self.events, ['train'])

    def test_overlapping_scene_or_caption_rejected_before_launch(self):
        self.configure()
        for prompt in ('  TWO green CATS ', 'Photograph STYLE. Scene: two green cats'):
            with self.subTest(prompt=prompt), self.assertRaisesRegex(ValueError, 'overlap'):
                self.manager.start_finetune({**self.config, 'validation_data': [{'prompt': prompt}]})
        self.assertEqual(self.events, [])
        self.assertFalse(self.manager.finetunes)

    def test_action_applied_holdout_cannot_match_a_training_prompt_override(self):
        self.configure()
        # A manually entered source permits explicit captions for desired photos.
        self.analyses.context['preview']['config'].pop('dataset_id')
        scene = 'a fox beside a green bicycle'
        self.config['training_data'][0]['prompt'] = 'Photograph style. Scene: ' + scene
        self.config['validation_data'] = [{'prompt': scene}]
        with patch.object(self.manager, '_launch', side_effect=AssertionError('Leaking holdout reached launch')):
            with self.assertRaisesRegex(ValueError, 'overlap'):
                self.manager.start_finetune(self.config)
        self.assertEqual(self.events, [])

    def test_invalid_options_rejected_before_launch(self):
        self.configure()
        for values in (dict(generation_size=True), dict(generation_size=511),
                       dict(generation_size='preview'), dict(generation_framing='stretch'),
                       dict(validation_repeats=0), dict(validation_repeats=True),
                       dict(validation_data=[])):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.manager.start_finetune({**self.config, **values})
        self.assertFalse(self.manager.finetunes)

    def test_generation_options_cannot_leak_into_other_workspaces(self):
        discovery = self.configure()
        for task in ('classification', 'summarization'):
            self.manager.jobs[discovery['id']]['task'] = task
            for field, value in (('generation_size', 'auto'), ('generation_framing', 'preserve'),
                                 ('validation_data', self.config['validation_data']), ('validation_repeats', 1)):
                with self.subTest(task=task, field=field), self.assertRaisesRegex(ValueError, 'only to image generation'):
                    self.manager._finetune_settings({field: value}, self.manager.jobs[discovery['id']])
        self.assertFalse(self.manager.finetunes)

    def test_reviewed_pair_training_uses_the_same_validation_pipeline(self):
        self.configure()
        record = self.config.pop('training_data')[0]
        self.config.pop('action_ids')
        self.config['examples'] = [dict(input_index=record['input_index'], action_id=record['action_id'],
                                       target=record['image_path'])]
        report = self.finish(generation_size=512, generation_framing='center_crop')
        self.assertEqual(report['status'], 'completed', report.get('error'))
        self.assertEqual(report['validation']['comparison']['verdict'], 'improved')
        self.assertEqual(self.training_settings['generation_size'], 512)
        self.assertEqual(self.training_settings['generation_framing'], 'center_crop')

    def test_scorer_change_preserves_candidate_but_reports_validation_error(self):
        self.configure()
        self.changed_scorer = True
        report = self.finish()
        self.assertEqual(report['status'], 'error')
        self.assertIn('scorer', report['error'])
        self.assertEqual(report['validation']['status'], 'error')
        self.assertTrue(Path(report['checkpoint']).is_dir())
        self.assertTrue(all(e.closed for e in self.evaluators))

    def test_cancellation_before_training_releases_evaluator(self):
        self.configure()
        self.cancel_phase = 'before'
        report = self.finish()
        self.assertEqual(report['status'], 'cancelled')
        self.assertEqual(report['validation']['status'], 'cancelled')
        self.assertIsNone(report['checkpoint'])
        self.assertNotIn('train', self.events)
        self.assertTrue(all(e.closed for e in self.evaluators))

    def test_cancellation_after_save_preserves_checkpoint_and_stops_validation(self):
        self.configure()
        self.cancel_after_save = True
        report = self.finish()
        self.assertEqual(report['status'], 'cancelled')
        self.assertEqual(report['validation']['status'], 'cancelled')
        self.assertTrue(Path(report['checkpoint']).is_dir())
        self.assertNotIn('load-after', self.events)

    def test_cancellation_during_candidate_validation_closes_evaluator(self):
        self.configure()
        self.cancel_phase = 'after'
        report = self.finish()
        self.assertEqual(report['status'], 'cancelled')
        self.assertEqual(report['validation']['status'], 'cancelled')
        self.assertTrue(Path(report['checkpoint']).is_dir())
        self.assertTrue(all(e.closed for e in self.evaluators))

    def test_finetune_artifacts_do_not_expose_arbitrary_files_or_symlinks(self):
        self.configure()
        report = self.finish()
        self.assertEqual(report['status'], 'completed', report.get('error'))
        for filename in ('../../target.png', 'target.png', 'policy.pt'):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                self.manager.artifact(report['id'], filename)
        link = self.manager.directory / report['id'] / 'validation-after-999-0-original.png'
        link.symlink_to(self.root / 'target.png')
        with self.assertRaises(FileNotFoundError):
            self.manager.artifact(report['id'], link.name)

    def test_restart_marks_interrupted_validation_as_error_and_keeps_saved_candidate(self):
        self.configure()
        report = self.finish()
        self.assertEqual(report['status'], 'completed', report.get('error'))
        path = self.manager.artifact(report['id'], 'report.json')
        for status in ('pending', 'running', 'completed', 'not_requested'):
            interrupted = copy.deepcopy(report)
            interrupted.update(status='running', phase='validating after outputs')
            interrupted['validation']['status'] = status
            path.write_text(json.dumps(interrupted))
            restarted = DiscoveryManager(self.previews, self.analyses, self.manager.directory)
            recovered = restarted.get_finetune(report['id'])
            self.assertEqual(recovered['status'], 'error')
            self.assertEqual(recovered['checkpoint'], report['checkpoint'])
            expected = 'error' if status in ('pending', 'running') else status
            with self.subTest(status=status):
                self.assertEqual(recovered['validation']['status'], expected)
                if expected == 'error':
                    self.assertIn('stopped', recovered['validation']['error'])


if __name__ == '__main__':
    unittest.main()
