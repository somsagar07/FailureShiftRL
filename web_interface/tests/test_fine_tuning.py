"""Reviewed training must update real weights and save independent checkpoints."""
import copy
import itertools
import random
import tracemalloc
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import (AutoModelForImageClassification, AutoModelForSeq2SeqLM,
                          BertConfig, EncoderDecoderConfig, EncoderDecoderModel,
                          MBartConfig, MBartForConditionalGeneration, MBartTokenizerFast,
                          T5ForConditionalGeneration, ViTForImageClassification, ViTImageProcessor)
from test_text_encoders import save_tokenizer
import test_discovery_models as fixtures

import fine_tuning as ft


class FineTuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.EvaluatorTests.setUpClass.__func__(cls)
        cls.grouped_vision=cls.root/'grouped-vision'
        config=copy.deepcopy(cls.reference.config)
        config.id2label={int(index):name for index,name in fixtures.imagenet_labels(offset=1).items()}
        cls.grouped_reference=ViTForImageClassification(config).eval()
        cls.grouped_reference.save_pretrained(cls.grouped_vision)
        cls.processor=ViTImageProcessor(size={'height':16,'width':16})
        cls.processor.save_pretrained(cls.grouped_vision)

    setUp=fixtures.EvaluatorTests.setUp
    model=fixtures.EvaluatorTests.model

    def train(self,task,examples,**settings):
        return ft.fine_tune(self.model(task),task,examples,Path(self.scratch.name)/'training',
            {'epochs':2,'batch_size':1,'learning_rate':.01,'seed':8,**settings},lambda **kw:None,lambda:False)

    def test_automatic_device_falls_back_to_cpu_when_cuda_is_unavailable(self):
        with patch.object(torch.cuda,'is_available',return_value=False):
            result=self.train('classification',[{'input':Image.new('RGB',(16,16)),'target':1}],epochs=1)
        self.assertEqual(result.get('device'),'cpu')
        self.assertEqual(result.get('device_name'),'CPU')
        self.assertEqual(result['settings'].get('device'),'auto')
        self.assertEqual(result['steps'],1)

    def test_explicit_cpu_reports_cpu_even_when_cuda_is_available(self):
        result=self.train('classification',[{'input':Image.new('RGB',(16,16)),'target':1}],device='cpu',epochs=1)
        self.assertEqual(result.get('device'),'cpu')
        self.assertEqual(result['settings']['device'],'cpu')

    def test_invalid_or_unavailable_device_fails_before_checkpoint_creation(self):
        for index,device in enumerate(('gpu','cuda:99',None,True,'cuda')):
            destination=Path(self.scratch.name)/f'invalid-{index}'
            with self.subTest(device=device),patch.object(torch.cuda,'is_available',return_value=False):
                with self.assertRaisesRegex(ValueError,'device|CUDA'):
                    ft.fine_tune(self.model(),'classification',
                        [{'input':Image.new('RGB',(16,16)),'target':1}],destination,{'device':device},
                        lambda **kw:None,lambda:False)
            self.assertFalse(destination.exists())

    def test_invalid_optimizer_controls_fail_before_model_loading_or_checkpoint_creation(self):
        cases={'weight_decay':(-.01,1.01,True,None,'0.1',float('nan'),float('inf'),10**400),
               'max_grad_norm':(-.01,100.01,True,None,'1',float('nan'),float('inf'),10**400)}
        for key,values in cases.items():
            for index,value in enumerate(values):
                destination=Path(self.scratch.name)/f'{key}-{index}'
                with self.subTest(setting=key,value=value):
                    with self.assertRaisesRegex(ValueError,key):
                        ft.fine_tune(self.model(path=Path(self.scratch.name)/'missing-model'),'classification',
                            [{'input':Image.new('RGB',(16,16)),'target':1}],destination,{key:value},
                            lambda **kw:None,lambda:False)
                    self.assertFalse(destination.exists())

    def test_default_optimizer_controls_preserve_classifier_and_summary_checkpoints(self):
        cases=[('classification',[{'input':Image.new('RGB',(16,16),'red'),'target':1}]),
               ('summarization',[{'input':'a cat in the garden','target':'cat'}])]
        for task,examples in cases:
            results=[]
            for index,controls in enumerate(({}, {'weight_decay':.01,'max_grad_norm':1.})):
                results.append(ft.fine_tune(self.model(task),task,examples,
                    Path(self.scratch.name)/f'{task}-{index}',
                    {'epochs':2,'learning_rate':.01,'seed':8,'device':'cpu',**controls},
                    lambda **kw:None,lambda:False))
            with self.subTest(task=task):
                self.assertEqual(results[0]['training_loss'],results[1]['training_loss'])
                first=load_file(Path(results[0]['checkpoint'])/'model.safetensors')
                second=load_file(Path(results[1]['checkpoint'])/'model.safetensors')
                self.assertTrue(all(torch.equal(value,second[key]) for key,value in first.items()))
                for result in results:
                    report=json.loads((Path(result['checkpoint'])/'training_report.json').read_text())
                    self.assertEqual(report['settings'].get('weight_decay'),.01)
                    self.assertEqual(report['settings'].get('max_grad_norm'),1.)

    def test_weight_decay_changes_real_classifier_and_summary_updates(self):
        cases=[('classification',self.vision,[{'input':Image.new('RGB',(16,16),'red'),'target':1}]),
               ('summarization',self.summary,[{'input':'a cat in the garden','target':'cat'}])]
        for task,source,examples in cases:
            checkpoints=[]
            for index,decay in enumerate((0.,1.)):
                result=ft.fine_tune(self.model(task),task,examples,
                    Path(self.scratch.name)/f'{task}-{index}',
                    {'epochs':1,'learning_rate':.01,'seed':8,'device':'cpu','weight_decay':decay},
                    lambda **kw:None,lambda:False)
                checkpoints.append(load_file(Path(result['checkpoint'])/'model.safetensors'))
            with self.subTest(task=task):
                before=load_file(source/'model.safetensors')
                keys=[key for key in checkpoints[0] if task=='summarization' or key.startswith('classifier.')]
                self.assertTrue(any(not torch.equal(checkpoints[0][key],checkpoints[1][key]) for key in keys))
                # AdamW's first update has identical data gradients in both runs.
                # Decay 1 at learning rate .01 subtracts exactly 1% of the base weight.
                for key in keys:
                    torch.testing.assert_close(checkpoints[0][key]-checkpoints[1][key],before[key]*.01,
                                               rtol=1e-4,atol=2e-7)

    def test_gradient_limit_bounds_real_updates_and_zero_disables_clipping(self):
        cases=[('classification',[{'input':Image.new('RGB',(16,16),'red'),'target':1}]),
               ('summarization',[{'input':'a cat in the garden','target':'cat'}])]
        optimizer_type=torch.optim.AdamW
        for task,examples in cases:
            checkpoints=[]
            for index,limit in enumerate((0.,100.,.05)):
                norms=[]
                class ObservedAdamW(optimizer_type):
                    def step(self,*args,**kwargs):
                        gradients=[parameter.grad for group in self.param_groups for parameter in group['params']
                                   if parameter.grad is not None]
                        norms.append(float(torch.stack([gradient.norm() for gradient in gradients]).norm()))
                        return super().step(*args,**kwargs)
                with patch.object(torch.optim,'AdamW',ObservedAdamW):
                    result=ft.fine_tune(self.model(task),task,examples,
                        Path(self.scratch.name)/f'{task}-{index}',
                        {'epochs':2,'learning_rate':.01,'seed':8,'device':'cpu','max_grad_norm':limit},
                        lambda **kw:None,lambda:False)
                checkpoints.append(load_file(Path(result['checkpoint'])/'model.safetensors'))
                with self.subTest(task=task,limit=limit):
                    self.assertEqual(len(norms),2)
                    if limit==.05:
                        self.assertTrue(all(0<norm<=.050001 for norm in norms),norms)
                    else:
                        self.assertGreater(max(norms),1.,'zero must retain the original gradients above the legacy limit')
            with self.subTest(task=task):
                self.assertTrue(all(torch.equal(value,checkpoints[1][key]) for key,value in checkpoints[0].items()),
                                'zero clipping must match a bound above every actual gradient norm')
                self.assertTrue(any(not torch.equal(value,checkpoints[2][key]) for key,value in checkpoints[0].items()),
                                'the configured gradient bound must affect the saved trained parameters')

    @unittest.skipUnless(torch.cuda.is_available(),'Real CUDA training requires CUDA')
    def test_automatic_classifier_and_summarizer_train_on_cuda_and_report_device(self):
        cases=[('classification',ViTForImageClassification,
                [{'input':Image.new('RGB',(16,16)),'target':1}]),
               ('summarization',T5ForConditionalGeneration,[{'input':'a cat','target':'cat'}])]
        for task,model_class,examples in cases:
            with self.subTest(task=task):
                observed=[]
                original=model_class.forward
                def forward(model,*args,**kwargs):
                    self.assertEqual(next(model.parameters()).device.type,'cuda')
                    self.assertTrue(all(value.device.type=='cuda' for value in kwargs.values()
                                        if isinstance(value,torch.Tensor)))
                    observed.append(True)
                    return original(model,*args,**kwargs)
                progress=[]
                with patch.object(model_class,'forward',forward):
                    result=ft.fine_tune(self.model(task),task,examples,Path(self.scratch.name)/task,
                        {'epochs':1},lambda **fields:progress.append(fields),lambda:False)
                self.assertTrue(observed,'the selected model must execute a training forward pass')
                self.assertEqual(result.get('device'),'cuda')
                self.assertEqual(result.get('device_name'),torch.cuda.get_device_name())
                self.assertEqual(result['settings'].get('device'),'auto')
                self.assertTrue(any(fields.get('device')=='cuda' for fields in progress))
                report=json.loads((Path(result['checkpoint'])/'training_report.json').read_text())
                self.assertEqual(report['device'],'cuda')
                self.assertEqual(report['device_name'],torch.cuda.get_device_name())

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA RNG isolation requires CUDA')
    def test_cuda_training_repeats_seeded_updates_and_restores_rng_on_completion_and_cancel(self):
        before_cpu=torch.random.get_rng_state()
        before_cuda=torch.cuda.get_rng_state_all()
        examples=[{'input':'a cat','target':'cat'}, {'input':'a cat in the garden','target':'cat in the garden'}]
        results=[]
        try:
            for attempt in range(3):
                cancelled=[False]
                def progress(**fields):
                    if attempt==2 and fields['phase']=='training reviewed examples':
                        cancelled[0]=True
                destination=Path(self.scratch.name)/f'cuda-{attempt}'
                def train():
                    return ft.fine_tune(self.model('summarization'),'summarization',examples,destination,
                        {'device':'cuda','seed':16,'batch_size':2},progress,lambda:cancelled[0])
                if attempt==2:
                    with self.assertRaises(InterruptedError):
                        train()
                    self.assertFalse((destination/'checkpoint').exists())
                    self.assertFalse(list(destination.glob('.checkpoint-*')))
                else:
                    results.append(train())
                    self.assertEqual(results[-1].get('device'),'cuda')
                self.assertTrue(torch.equal(before_cpu,torch.random.get_rng_state()))
                self.assertTrue(all(torch.equal(before,after) for before,after in
                                    zip(before_cuda,torch.cuda.get_rng_state_all())))
            self.assertEqual(results[0]['training_loss'],results[1]['training_loss'])
            first=load_file(Path(results[0]['checkpoint'])/'model.safetensors')
            second=load_file(Path(results[1]['checkpoint'])/'model.safetensors')
            self.assertTrue(all(torch.equal(first[key],second[key]) for key in first))
        finally:
            torch.random.set_rng_state(before_cpu)
            torch.cuda.set_rng_state_all(before_cuda)

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

    def test_group_target_updates_only_head_and_increases_category_probability(self):
        image=Image.new('RGB',(16,16),'red')
        inputs=self.processor(images=image,return_tensors='pt')
        with torch.inference_mode():
            before=self.grouped_reference(**inputs).logits.softmax(-1)[0]
        original=load_file(self.grouped_vision/'model.safetensors')
        result=ft.fine_tune(self.model(path=self.grouped_vision),'classification',
            [{'input':image,'target':'cat'},{'input':image,'target':' CAT '}],
            Path(self.scratch.name)/'training',{'epochs':1,'batch_size':2,'learning_rate':.01},
            lambda **kw:None,lambda:False)
        checkpoint=Path(result['checkpoint'])
        saved=load_file(checkpoint/'model.safetensors')
        changed={key for key in saved if not torch.equal(original[key],saved[key])}
        self.assertTrue(changed)
        self.assertTrue(all(key.startswith('classifier.') for key in changed),changed)
        self.assertTrue(torch.all(saved['classifier.bias'][282:287]>original['classifier.bias'][282:287]),
                        'all allowed cat classes receive the category gradient; none is chosen as a breed target')
        reloaded=AutoModelForImageClassification.from_pretrained(checkpoint,local_files_only=True).eval()
        with torch.inference_mode():
            after=reloaded(**inputs).logits.softmax(-1)[0]
        self.assertGreater(float(after[282:287].sum()),float(before[282:287].sum()))
        self.assertAlmostEqual(result['training_loss'],-math.log(float(before[282:287].sum())),places=5)
        self.assertTrue(all(torch.equal(original[key],load_file(self.grouped_vision/'model.safetensors')[key])
                            for key in original))
        report=json.loads((checkpoint/'training_report.json').read_text())
        self.assertEqual(report['objective'],'negative_log_target_probability')
        self.assertEqual(report['target_groups'],{'cat':[282,283,284,285,286]})
        self.assertEqual(report['target_counts'],{'single_class':0,'groups':{'cat':2}})
        self.assertEqual(report['method'],'classification head')

    def test_mixed_singleton_cat_and_dog_batch_uses_each_targets_probability_mass(self):
        images=[Image.new('RGB',(16,16),color) for color in ('red','green','blue')]
        with torch.inference_mode():
            probabilities=self.grouped_reference(**self.processor(images=images,return_tensors='pt')).logits.double().softmax(-1)
        expected=(-probabilities[0,282:287].sum().log()-probabilities[1,152:270].sum().log()
                  -probabilities[2,17].log())/3
        result=ft.fine_tune(self.model(path=self.grouped_vision),'classification',
            [{'input':image,'target':target} for image,target in zip(images,['cat','dog',17])],
            Path(self.scratch.name)/'training',{'epochs':1,'batch_size':3,'learning_rate':.01},
            lambda **kw:None,lambda:False)
        self.assertEqual(result['steps'],1)
        self.assertAlmostEqual(result['training_loss'],float(expected),places=5)
        self.assertEqual(result['target_groups'],{'cat':list(range(282,287)),'dog':list(range(152,270))})
        self.assertEqual(result['target_counts'],{'single_class':1,'groups':{'cat':1,'dog':1}})

    def test_singleton_target_batch_is_equivalent_to_cross_entropy(self):
        images=[Image.new('RGB',(16,16),color) for color in ('red','blue')]
        with torch.inference_mode():
            logits=self.reference(**self.processor(images=images,return_tensors='pt')).logits
            expected=torch.nn.functional.cross_entropy(logits,torch.tensor([0,2]))
        result=self.train('classification',[{'input':images[0],'target':'cat'},{'input':images[1],'target':2}],
                          epochs=1,batch_size=2)
        self.assertAlmostEqual(result['training_loss'],float(expected),places=6)
        self.assertEqual(result['target_groups'],{})
        self.assertEqual(result['target_counts'],{'single_class':2,'groups':{}})

    def test_group_loss_stays_finite_when_target_probability_underflows(self):
        path=Path(self.scratch.name)/'extreme-classifier'
        model=copy.deepcopy(self.grouped_reference)
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.fill_(1000.)
            model.classifier.bias[282:287]=-1000.
        model.save_pretrained(path)
        self.processor.save_pretrained(path)
        result=ft.fine_tune(self.model(path=path),'classification',
            [{'input':Image.new('RGB',(16,16)),'target':'cat'}],Path(self.scratch.name)/'training',
            {'epochs':1},lambda **kw:None,lambda:False)
        self.assertTrue(math.isfinite(result['training_loss']))
        self.assertAlmostEqual(result['training_loss'],2000+math.log(996)-math.log(5),places=3)

    def test_unknown_targets_after_valid_group_fail_before_optimizer_or_checkpoint(self):
        for target in ('dataset-dog',-1,1001,True):
            with self.subTest(target=target),patch.object(torch.optim,'AdamW',
                    side_effect=AssertionError('optimizer created before target validation')):
                with self.assertRaisesRegex(ValueError,'explicit model class|exact model label'):
                    ft.fine_tune(self.model(path=self.grouped_vision),'classification',
                        [{'input':Image.new('RGB',(16,16)),'target':'cat'},
                         {'input':Image.new('RGB',(16,16)),'target':target}],
                        Path(self.scratch.name)/'training',{},lambda **kw:None,lambda:False)
        self.assertFalse((Path(self.scratch.name)/'training'/'checkpoint').exists())

    def test_group_class_ids_outside_head_fail_before_optimizer(self):
        path=Path(self.scratch.name)/'invalid-vocabulary'
        model=copy.deepcopy(self.grouped_reference)
        model.config.id2label[10001]=model.config.id2label.pop(282)
        model.save_pretrained(path)
        self.processor.save_pretrained(path)
        with patch.object(torch.optim,'AdamW',side_effect=AssertionError('optimizer created before ID validation')):
            with self.assertRaisesRegex(ValueError,'class IDs'):
                ft.fine_tune(self.model(path=path),'classification',
                    [{'input':Image.new('RGB',(16,16)),'target':'cat'}],Path(self.scratch.name)/'training',
                    {},lambda **kw:None,lambda:False)
        self.assertFalse((Path(self.scratch.name)/'training'/'checkpoint').exists())

    def test_large_ordered_source_shuffle_is_dispersed_repeatable_and_bounded(self):
        tracemalloc.start()
        try:
            first=list(itertools.islice(ft._shuffled_indices(10000000,random.Random(6)),512))
            _,peak=tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak,1024*1024)
        self.assertGreaterEqual(len({index//156250 for index in first}),32,
                                'early training must mix distant regions of an input/class-ordered corpus')
        self.assertEqual(len(set(first)),512)
        self.assertEqual(first,list(itertools.islice(ft._shuffled_indices(10000000,random.Random(6)),512)))
        self.assertNotEqual(first,list(itertools.islice(ft._shuffled_indices(10000000,random.Random(7)),512)))
        for count in (1,2,31,4096,10001):
            order=list(ft._shuffled_indices(count,random.Random(4)))
            self.assertEqual(sorted(order),list(range(count)))

    def test_lazy_source_trains_every_example_without_eager_loading(self):
        # Eager preparation, dropped final batches, and repeated samples all break
        # this source's bounded demand/coverage contract.
        count=4103
        loaded=[]
        completed=[0]
        class Source:
            def __len__(self):
                return count
            def target(self,index):
                return index%3
            def __getitem__(self,index):
                if not 0<=index<count:
                    raise IndexError(index)
                if len(loaded)-completed[0]>=32:
                    raise AssertionError('training read ahead beyond its bounded active batch')
                loaded.append(index)
                return {'input':Image.new('RGB',(16,16),(index%256,0,0)),'target':self.target(index)}
        events=[]
        def progress(**fields):
            events.append(fields)
            completed[0]=fields.get('processed_examples',0)
        result=ft.fine_tune(self.model(),'classification',Source(),Path(self.scratch.name)/'large',
            {'epochs':2,'batch_size':32,'learning_rate':.001,'device':'cpu','seed':9},progress,lambda:False)
        self.assertEqual(result['steps'],258)
        self.assertEqual(result['processed_examples'],8206)
        self.assertEqual(result['examples_per_epoch'],4103)
        self.assertEqual(result['total_examples'],8206)
        self.assertEqual(result['epoch'],2)
        self.assertEqual(result['epochs'],2)
        self.assertEqual(result['effective_batch_size'],32)
        self.assertLessEqual(result['micro_batch_size'],32)
        self.assertEqual(sorted(loaded[:count]),list(range(count)))
        self.assertEqual(sorted(loaded[count:]),list(range(count)))
        self.assertNotEqual(loaded[:count],loaded[count:])
        self.assertTrue(all(fields['total']==258 for fields in events))
        self.assertTrue(all(fields['completed']<=258 for fields in events))
        self.assertEqual(events[-1]['processed_examples'],8206)
        self.assertTrue(math.isfinite(result['training_loss']))

    def test_invalid_lazy_target_is_rejected_without_loading_images_or_creating_optimizer(self):
        class Source:
            def __len__(self):
                return 10000
            def target(self,index):
                return 'dataset-dog' if index==9999 else 1
            def __getitem__(self,index):
                raise AssertionError('target validation decoded an image')
        with patch.object(torch.optim,'AdamW',side_effect=AssertionError('unvalidated optimizer')):
            with self.assertRaisesRegex(ValueError,'explicit model class|exact model label'):
                ft.fine_tune(self.model(),'classification',Source(),Path(self.scratch.name)/'invalid',
                    {'device':'cpu'},lambda **fields:None,lambda:False)
        self.assertFalse((Path(self.scratch.name)/'invalid'/'checkpoint').exists())

    def test_default_auto_and_large_manual_batches_report_actual_example_progress(self):
        examples=[{'input':Image.new('RGB',(16,16)),'target':1}]*9
        for index,batch in enumerate(('auto',1024)):
            result=ft.fine_tune(self.model(),'classification',examples,Path(self.scratch.name)/str(index),
                {'batch_size':batch,'device':'cpu'},lambda **fields:None,lambda:False)
            self.assertEqual(result['settings']['batch_size'],batch)
            self.assertEqual(result['processed_examples'],9)
            self.assertEqual(result['total_examples'],9)
            self.assertEqual(result['steps'],math.ceil(9/result['effective_batch_size']))
            self.assertLessEqual(result['micro_batch_size'],result['effective_batch_size'])
        for batch in (None,True,0,1025,1.5,'16'):
            with self.subTest(batch=batch),self.assertRaisesRegex(ValueError,'batch_size'):
                ft._settings({'batch_size':batch})

    def test_uneven_summary_microbatches_match_full_batch_token_normalization(self):
        source=Path(self.scratch.name)/'no-dropout'
        model=T5ForConditionalGeneration.from_pretrained(self.summary,local_files_only=True)
        model.config.dropout_rate=0.
        model.save_pretrained(source)
        save_tokenizer(source)
        examples=[{'input':'a cat','target':target} for target in
                  ('cat','a cat in the garden','cat in the garden','a cat','cat')]
        results=[]
        for micro in (5,2,1):
            seen=[]
            original=T5ForConditionalGeneration.forward
            def forward(model,*args,**kwargs):
                seen.append(int(kwargs['input_ids'].shape[0]))
                return original(model,*args,**kwargs)
            with patch.object(ft,'_batch_sizes',return_value=(5,micro),create=True), \
                    patch.object(T5ForConditionalGeneration,'forward',forward):
                result=ft.fine_tune(self.model('summarization',source),'summarization',examples,
                    Path(self.scratch.name)/f'micro-{micro}',
                    {'batch_size':5,'seed':12,'device':'cpu','learning_rate':.001},
                    lambda **fields:None,lambda:False)
            self.assertEqual(max(seen),micro)
            self.assertEqual(sum(seen),5)
            self.assertEqual(result['steps'],1)
            results.append(result)
        reference=load_file(Path(results[0]['checkpoint'])/'model.safetensors')
        for result in results[1:]:
            self.assertAlmostEqual(results[0]['training_loss'],result['training_loss'],places=5)
            saved=load_file(Path(result['checkpoint'])/'model.safetensors')
            for key in reference:
                torch.testing.assert_close(saved[key],reference[key],rtol=2e-5,atol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA OOM recovery requires CUDA')
    def test_oom_after_partial_backward_replays_whole_update_and_restores_rng(self):
        examples=[{'input':Image.new('RGB',(16,16),color),'target':index%3}
                  for index,color in enumerate(('red','green','blue','white','black'))]
        results=[]
        for retry in (False,True):
            seen=[]
            states=[]
            failed=[False]
            original=ViTForImageClassification.forward
            def forward(model,*args,**kwargs):
                states.append((torch.random.get_rng_state(),torch.cuda.get_rng_state()))
                torch.rand(())
                torch.rand((),device='cuda')
                size=int(kwargs['pixel_values'].shape[0])
                seen.append(size)
                result=original(model,*args,**kwargs)
                if retry and not failed[0] and len(seen)==2:
                    def fail_backward(gradient):
                        failed[0]=True
                        raise torch.cuda.OutOfMemoryError('simulated CUDA allocation after prior gradients')
                    result.logits.register_hook(fail_backward)
                return result
            with patch.object(ft,'_batch_sizes',return_value=(5,4 if retry else 2),create=True), \
                    patch.object(ViTForImageClassification,'forward',forward):
                result=ft.fine_tune(self.model(),'classification',examples,
                    Path(self.scratch.name)/f'retry-{retry}',
                    {'batch_size':5,'device':'cuda','seed':17,'learning_rate':.001},
                    lambda **fields:None,lambda:False)
            results.append(result)
            self.assertEqual(seen,[4,1,2,2,1] if retry else [2,2,1])
            self.assertEqual(result['steps'],1)
            self.assertEqual(result['processed_examples'],5)
            self.assertEqual(result['oom_retries'],int(retry))
            self.assertEqual(result['micro_batch_size'],2)
            if retry:
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(states[0],states[2])),
                                'failed attempts must restore CPU and CUDA RNG before replay')
        self.assertAlmostEqual(results[0]['training_loss'],results[1]['training_loss'],places=6)
        reference=load_file(Path(results[0]['checkpoint'])/'model.safetensors')
        saved=load_file(Path(results[1]['checkpoint'])/'model.safetensors')
        for key in reference:
            torch.testing.assert_close(saved[key],reference[key],rtol=1e-6,atol=1e-7)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA OOM behavior requires CUDA')
    def test_single_example_oom_is_actionable_and_optimizer_oom_is_never_retried(self):
        examples=[{'input':Image.new('RGB',(16,16)),'target':1}]
        original_step=torch.optim.AdamW.step
        updates=[]
        def partial_step(optimizer,*args,**kwargs):
            original_step(optimizer,*args,**kwargs)
            updates.append(1)
            raise torch.cuda.OutOfMemoryError('simulated optimizer-state allocation failure')
        for phase in ('forward','optimizer'):
            destination=Path(self.scratch.name)/phase
            target,method=(ViTForImageClassification,'forward') if phase=='forward' else (torch.optim.AdamW,'step')
            failure=partial_step if phase=='optimizer' else lambda *args,**kwargs: (_ for _ in ()).throw(
                torch.cuda.OutOfMemoryError('simulated single-example failure'))
            with patch.object(target,method,failure):
                with self.assertRaisesRegex(ValueError,'[Cc][Pp][Uu]|smaller model|optimizer update'):
                    ft.fine_tune(self.model(),'classification',examples,destination,
                        {'batch_size':1,'device':'cuda'},lambda **fields:None,lambda:False)
            self.assertFalse((destination/'checkpoint').exists())
            self.assertFalse(list(destination.glob('.checkpoint-*')))
        self.assertEqual(updates,[1],'a potentially partial optimizer step must never replay')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA OOM behavior requires CUDA')
    def test_final_single_example_oom_is_not_retried_with_same_actual_batch(self):
        examples=[{'input':Image.new('RGB',(16,16)),'target':1}]*5
        sizes=[]
        original=ViTForImageClassification.forward
        def forward(model,*args,**kwargs):
            size=int(kwargs['pixel_values'].shape[0])
            sizes.append(size)
            if size==1:
                raise torch.cuda.OutOfMemoryError('simulated longer final example cannot fit')
            return original(model,*args,**kwargs)
        with patch.object(ft,'_batch_sizes',return_value=(4,4)), \
                patch.object(ViTForImageClassification,'forward',forward):
            with self.assertRaisesRegex(ValueError,'one training example'):
                ft.fine_tune(self.model(),'classification',examples,Path(self.scratch.name)/'final-oom',
                    {'batch_size':4,'device':'cuda'},lambda **fields:None,lambda:False)
        self.assertEqual(sizes,[4,1], 'retrying a batch already containing one example cannot reduce memory')
        self.assertFalse((Path(self.scratch.name)/'final-oom'/'checkpoint').exists())

    def test_cancellation_during_large_target_validation_never_materializes_images(self):
        count=[0]
        class Source:
            def __len__(self):
                return 100000000
            def target(self,index):
                count[0]+=1
                return 1
            def __getitem__(self,index):
                raise AssertionError('cancelled target validation decoded an image')
        with patch.object(torch.optim,'AdamW',side_effect=AssertionError('cancelled optimizer')):
            with self.assertRaises(InterruptedError):
                ft.fine_tune(self.model(),'classification',Source(),Path(self.scratch.name)/'cancel',
                    {'device':'cpu'},lambda **fields:None,lambda:count[0]>=3)
        self.assertEqual(count[0],3)
        self.assertFalse((Path(self.scratch.name)/'cancel'/'checkpoint').exists())

    def test_summary_supervised_weights_update_and_reload(self):
        original=load_file(self.summary/'model.safetensors')
        result=self.train('summarization',[{'input':'a cat','target':'cat in the garden'}])
        saved=load_file(Path(result['checkpoint'])/'model.safetensors')
        self.assertTrue(any(not torch.equal(original[key],saved[key]) for key in saved))
        model=AutoModelForSeq2SeqLM.from_pretrained(result['checkpoint'],local_files_only=True)
        self.assertEqual(model.config.model_type,'t5')
        self.assertTrue(torch.isfinite(torch.tensor(result['training_loss'])))

    def test_summary_uses_target_language_tokens_for_reviewed_labels(self):
        directory=Path(self.scratch.name)/'multilingual-summary'
        save_tokenizer(directory)
        tokenizer=MBartTokenizerFast(tokenizer_file=str(directory/'tokenizer.json'),
            bos_token='[BOS]',eos_token='[EOS]',pad_token='[PAD]',unk_token='[UNK]',
            src_lang='en_XX',tgt_lang='de_DE',model_max_length=12)
        tokenizer.save_pretrained(directory)
        config=MBartConfig(vocab_size=len(tokenizer),d_model=12,encoder_layers=1,decoder_layers=1,
            encoder_attention_heads=3,decoder_attention_heads=3,encoder_ffn_dim=24,decoder_ffn_dim=24,
            max_position_embeddings=12,pad_token_id=0,bos_token_id=2,eos_token_id=3,
            decoder_start_token_id=tokenizer.lang_code_to_id['de_DE'])
        MBartForConditionalGeneration(config).save_pretrained(directory)

        adapter,prepared,_=ft._summarizer(self.model('summarization',directory),
            [{'input':'cat','target':'cat'}])

        self.assertEqual(prepared[0][0]['input_ids'].tolist(),[[8,3,22]])
        self.assertEqual(prepared[0][1].tolist(),[[8,3,21]],
                         'reviewed targets must carry the configured target language, not the source language')

    def encoder_decoder(self,encoder_limit,decoder_limit):
        directory=Path(self.scratch.name)/'encoder-decoder'
        def component(limit):
            return BertConfig(vocab_size=16,hidden_size=12,intermediate_size=24,
                num_hidden_layers=1,num_attention_heads=3,max_position_embeddings=limit)
        config=EncoderDecoderConfig.from_encoder_decoder_configs(component(encoder_limit),component(decoder_limit))
        config.pad_token_id=0
        config.decoder_start_token_id=2
        config.eos_token_id=3
        EncoderDecoderModel(config=config).save_pretrained(directory)
        save_tokenizer(directory)
        return self.model('summarization',directory)

    def test_summary_target_can_exceed_encoder_capacity_when_decoder_supports_it(self):
        model=self.encoder_decoder(4,8)
        result=ft.fine_tune(model,'summarization',[{'input':'cat','target':'a cat in the garden'}],
            Path(self.scratch.name)/'training',{},lambda **kw:None,lambda:False)
        self.assertEqual(result['steps'],1)
        self.assertTrue(math.isfinite(result['training_loss']))
        reloaded=AutoModelForSeq2SeqLM.from_pretrained(result['checkpoint'],local_files_only=True)
        self.assertEqual(reloaded.config.decoder.max_position_embeddings,8)

    def test_summary_checks_decoder_capacity_before_starting_optimizer(self):
        model=self.encoder_decoder(8,4)
        with patch.object(torch.optim,'AdamW',side_effect=AssertionError('optimizer created before target validation')):
            with self.assertRaisesRegex(ValueError,'target summary.*token limit'):
                ft.fine_tune(model,'summarization',
                    [{'input':'cat','target':'cat'},{'input':'cat','target':'a cat in the garden'}],
                    Path(self.scratch.name)/'training',{},lambda **kw:None,lambda:False)
        self.assertFalse((Path(self.scratch.name)/'training'/'checkpoint').exists())

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

    def test_cancellation_at_validation_progress_stops_before_model_loading(self):
        cancelled=[False]
        def progress(**fields):
            cancelled[0]=True
        for task in ('classification','summarization','generation'):
            with self.subTest(task=task):
                cancelled[0]=False
                with self.assertRaises(InterruptedError):
                    ft.fine_tune(self.model(task,path=Path(self.scratch.name)/'missing-model'),task,
                        [{'input':'source','target':'reviewed target'}],Path(self.scratch.name)/task,{},
                        progress,lambda:cancelled[0])

    def test_cancellation_at_save_progress_stops_before_serializing(self):
        cancelled=[False]
        def progress(**fields):
            if fields['phase']=='saving separate checkpoint':
                cancelled[0]=True
        with patch.object(ViTForImageClassification,'save_pretrained',
                          side_effect=AssertionError('cancelled job reached checkpoint serialization')):
            with self.assertRaises(InterruptedError):
                ft.fine_tune(self.model(),'classification',[{'input':Image.new('RGB',(16,16)),'target':1}],
                    self.scratch.name,{},progress,lambda:cancelled[0])
        self.assertFalse((Path(self.scratch.name)/'checkpoint').exists())
        self.assertFalse(list(Path(self.scratch.name).glob('.checkpoint-*')))

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

    def test_cancellation_stops_between_training_input_preparations(self):
        examples={'classification':[{'input':Image.new('RGB',(16,16)),'target':1}]*3,
                  'summarization':[{'input':'a cat','target':'cat'}]*3}
        for task in examples:
            with self.subTest(task=task):
                prepared=[]
                cancelled=[False]
                cls,method=(ViTImageProcessor,'preprocess') if task=='classification' else (ft.SummarizationEncoder,'_prepare')
                original=getattr(cls,method)
                def prepare(instance,*args,**kwargs):
                    result=original(instance,*args,**kwargs)
                    prepared.append(result)
                    cancelled[0]=True
                    return result
                with patch.object(cls,method,prepare):
                    with self.assertRaises(InterruptedError):
                        ft.fine_tune(self.model(task),task,examples[task],
                            Path(self.scratch.name)/task,{},lambda **kw:None,lambda:cancelled[0])
                self.assertEqual(len(prepared),1,'cancellation should stop before preprocessing the remaining corpus')
                self.assertFalse((Path(self.scratch.name)/task/'checkpoint').exists())

    def test_real_summary_batches_mask_padding_and_are_repeatable(self):
        examples=[{'input':'a cat','target':'cat'}, {'input':'a cat in the garden','target':'a cat in the garden'}]
        settings={'epochs':1,'batch_size':2,'learning_rate':.001,'seed':6,'device':'cpu'}
        first=ft.fine_tune(self.model('summarization'),'summarization',examples,Path(self.scratch.name)/'first',
            settings,lambda **kw:None,lambda:False)
        second=ft.fine_tune(self.model('summarization'),'summarization',examples,Path(self.scratch.name)/'second',
            settings,lambda **kw:None,lambda:False)
        self.assertEqual(first['training_loss'],second['training_loss'])
        a=load_file(Path(first['checkpoint'])/'model.safetensors')
        b=load_file(Path(second['checkpoint'])/'model.safetensors')
        self.assertTrue(all(torch.equal(a[key],b[key]) for key in a))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA RNG isolation requires CUDA')
    def test_cpu_training_preserves_existing_cpu_and_cuda_random_state(self):
        before_cpu=torch.random.get_rng_state()
        before_cuda=torch.cuda.get_rng_state_all()
        try:
            result=self.train('classification',[{'input':Image.new('RGB',(16,16)),'target':1}],epochs=1,device='cpu')
            self.assertEqual(result['steps'],1)
            self.assertTrue(torch.equal(before_cpu,torch.random.get_rng_state()))
            self.assertTrue(all(torch.equal(before,after) for before,after in
                                zip(before_cuda,torch.cuda.get_rng_state_all())),
                            'CPU-only fine-tuning must not reseed CUDA used by other jobs')
        finally:
            torch.cuda.set_rng_state_all(before_cuda)

    def test_invalid_settings_and_source_overwrite_rejected(self):
        examples=[{'input':Image.new('RGB',(16,16)),'target':1}]
        for settings in [{'epochs':0},{'epochs':True},{'learning_rate':float('nan')},{'batch_size':0}]:
            with self.subTest(settings=settings),self.assertRaises(ValueError):
                self.train('classification',examples,**settings)
        with self.assertRaisesRegex(ValueError,'original|source|separate'):
            ft.fine_tune(self.model(),'classification',examples,self.vision,{},lambda **kw:None,lambda:False)
