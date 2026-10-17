"""Reviewed training must update real weights and save independent checkpoints."""
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoModelForImageClassification, AutoModelForSeq2SeqLM
import test_discovery_models as fixtures

import fine_tuning as ft


class FineTuningTests(unittest.TestCase):
    setUpClass=classmethod(fixtures.EvaluatorTests.setUpClass.__func__)
    setUp=fixtures.EvaluatorTests.setUp
    model=fixtures.EvaluatorTests.model

    def train(self,task,examples,**settings):
        return ft.fine_tune(self.model(task),task,examples,Path(self.scratch.name)/'training',
            {'epochs':2,'batch_size':1,'learning_rate':.01,'seed':8,**settings},lambda **kw:None,lambda:False)

    def test_classifier_only_head_updates_and_reload_is_independent(self):
        original=load_file(self.vision/'model.safetensors')
        examples=[{'input':Image.new('RGB',(16,16),'red'),'target':'dog'}]
        result=self.train('classification',examples)
        checkpoint=Path(result['checkpoint'])
        saved=load_file(checkpoint/'model.safetensors')
        changed={key for key in saved if not torch.equal(original[key],saved[key])}
        self.assertTrue(changed)
        self.assertTrue(all(key.startswith('classifier.') for key in changed),changed)
        self.assertEqual(result['steps'],2)
        model=AutoModelForImageClassification.from_pretrained(checkpoint,local_files_only=True)
        self.assertEqual(model.config.id2label[1],'dog')
        self.assertTrue(all(torch.equal(original[key],load_file(self.vision/'model.safetensors')[key]) for key in original))

    def test_summary_supervised_weights_update_and_reload(self):
        original=load_file(self.summary/'model.safetensors')
        result=self.train('summarization',[{'input':'a cat','target':'cat in the garden'}])
        saved=load_file(Path(result['checkpoint'])/'model.safetensors')
        self.assertTrue(any(not torch.equal(original[key],saved[key]) for key in saved))
        model=AutoModelForSeq2SeqLM.from_pretrained(result['checkpoint'],local_files_only=True)
        self.assertEqual(model.config.model_type,'t5')
        self.assertTrue(torch.isfinite(torch.tensor(result['training_loss'])))

    def test_all_targets_validated_before_any_updates_or_save(self):
        for task,examples in [('classification',[{'input':Image.new('RGB',(16,16)),'target':'dog'},
            {'input':Image.new('RGB',(16,16)),'target':'dataset-dog'}]),
            ('summarization',[{'input':'a cat','target':'cat'}, {'input':'a cat','target':''}])]:
            with self.subTest(task=task),self.assertRaises(ValueError):
                self.train(task,examples)
        self.assertFalse((Path(self.scratch.name)/'training'/'checkpoint').exists())

    def test_cancel_never_advertises_or_saves_final_checkpoint(self):
        with self.assertRaises(InterruptedError):
            ft.fine_tune(self.model(),'classification',[{'input':Image.new('RGB',(16,16)),'target':1}],
                self.scratch.name,{},lambda **kw:None,lambda:True)
        self.assertFalse((Path(self.scratch.name)/'checkpoint').exists())

    def test_cancellation_after_update_removes_staged_checkpoint(self):
        cancelled=[False]
        def progress(**fields):
            if fields.get('completed',0)>0:
                cancelled[0]=True
        with self.assertRaises(InterruptedError):
            ft.fine_tune(self.model(),'classification',[{'input':Image.new('RGB',(16,16)),'target':1}],
                self.scratch.name,{'epochs':3},progress,lambda:cancelled[0])
        self.assertFalse((Path(self.scratch.name)/'checkpoint').exists())
        self.assertFalse(list(Path(self.scratch.name).glob('.checkpoint-*')))

    def test_real_summary_batches_mask_padding_and_are_repeatable(self):
        examples=[{'input':'a cat','target':'cat'}, {'input':'a cat in the garden','target':'a cat in the garden'}]
        settings={'epochs':1,'batch_size':2,'learning_rate':.001,'seed':6}
        first=ft.fine_tune(self.model('summarization'),'summarization',examples,Path(self.scratch.name)/'first',
            settings,lambda **kw:None,lambda:False)
        second=ft.fine_tune(self.model('summarization'),'summarization',examples,Path(self.scratch.name)/'second',
            settings,lambda **kw:None,lambda:False)
        self.assertEqual(first['training_loss'],second['training_loss'])
        a=load_file(Path(first['checkpoint'])/'model.safetensors')
        b=load_file(Path(second['checkpoint'])/'model.safetensors')
        self.assertTrue(all(torch.equal(a[key],b[key]) for key in a))

    def test_invalid_settings_and_source_overwrite_rejected(self):
        examples=[{'input':Image.new('RGB',(16,16)),'target':1}]
        for settings in [{'epochs':0},{'epochs':True},{'learning_rate':float('nan')},{'batch_size':0}]:
            with self.subTest(settings=settings),self.assertRaises(ValueError):
                self.train('classification',examples,**settings)
        with self.assertRaisesRegex(ValueError,'original|source|separate'):
            ft.fine_tune(self.model(),'classification',examples,self.vision,{},lambda **kw:None,lambda:False)
