"""Real offline model inference and domain reward checks."""
import json
import copy
import math
import shutil
import tempfile
import unittest
import warnings
from unittest.mock import patch
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor
from test_text_encoders import VOCAB, save_tokenizer
from transformers import T5Config, T5ForConditionalGeneration

import discovery_models as dm
from test_classification_labels import imagenet_labels


class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.vision = cls.root/'vision'
        cls.summary = cls.root/'summary'
        torch.manual_seed(9)
        cls.reference = ViTForImageClassification(ViTConfig(
            image_size=16, patch_size=8, hidden_size=12, intermediate_size=24,
            num_hidden_layers=1, num_attention_heads=3, num_labels=3,
            id2label={0:'cat',1:'dog',2:'bird'})).eval()
        cls.reference.save_pretrained(cls.vision)
        ViTImageProcessor(size={'height':16,'width':16}).save_pretrained(cls.vision)
        T5ForConditionalGeneration(T5Config(vocab_size=len(VOCAB),d_model=12,d_kv=4,d_ff=24,
            num_layers=1,num_decoder_layers=1,num_heads=3,pad_token_id=0,eos_token_id=3,
            decoder_start_token_id=0)).save_pretrained(cls.summary)
        save_tokenizer(cls.summary)

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)

    def model(self, task='classification', path=None):
        return {'source':'local','name':str(path or (self.vision if task=='classification' else self.summary)), 'task':task}

    def evaluator(self, task='classification', **config):
        evaluator = dm.create_evaluator(self.model(task), {'task':task,**config}, self.scratch.name)
        self.addCleanup(evaluator.close)
        return evaluator

    def test_real_classifier_logits_and_model_labels(self):
        evaluator = self.evaluator()
        image = Image.new('RGB',(20,20),'red')
        result = evaluator.predict(image)
        with torch.inference_mode():
            expected = self.reference(**evaluator.processor(images=image,return_tensors='pt')).logits.softmax(-1)[0]
        torch.testing.assert_close(torch.tensor(result['probabilities']), expected)
        self.assertEqual(result['label'], self.reference.config.id2label[result['class_id']])
        json.dumps(result,allow_nan=False)

    def test_label_resolution_rejects_dataset_names_and_ambiguous_labels(self):
        evaluator = self.evaluator()
        self.assertEqual(evaluator.resolve_label('cat'),0)
        self.assertEqual(evaluator.resolve_label(2),2)
        for value in [True,'0','CAT','dataset-cat',-1,3,None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluator.resolve_label(value)
        evaluator.model.config.id2label[1]='cat'
        with self.assertRaisesRegex(ValueError,'ambiguous'):
            evaluator.resolve_label('cat')

    def test_classification_rewards_baseline_errors_and_new_label_errors_separately(self):
        evaluator = self.evaluator()
        baseline={'class_id':0,'probabilities':[.8,.15,.05]}
        bad={'class_id':1,'probabilities':[.2,.7,.1]}
        good=evaluator.score(baseline,baseline,target='cat')
        bad_score=evaluator.score(bad,baseline,target='cat')
        self.assertEqual(good['reward'],0)
        self.assertEqual(good['severity'],0)
        self.assertTrue(bad_score['failure'])
        self.assertGreater(bad_score['reward'],.5)
        self.assertFalse(bad_score['metrics']['baseline_error'])
        existing=evaluator.score(bad,bad,target='cat')
        self.assertTrue(existing['metrics']['baseline_error'])
        self.assertAlmostEqual(existing['reward'],.675)
        self.assertEqual(existing['severity'],1)
        proxy=evaluator.score(bad,baseline)
        self.assertIn('candidate',proxy['criterion'])
        self.assertNotIn('labelled',proxy['criterion'])

    def test_grouped_pet_targets_score_actual_top_class_and_summed_target_confidence(self):
        path = Path(self.scratch.name) / 'imagenet-head'
        config = copy.deepcopy(self.reference.config)
        config.id2label = {int(index): name for index, name in imagenet_labels(offset=1).items()}
        ViTForImageClassification(config).save_pretrained(path)
        ViTImageProcessor(size={'height':16,'width':16}).save_pretrained(path)
        evaluator = dm.create_evaluator(self.model(path=path), {'task':'classification'}, self.scratch.name)
        self.addCleanup(evaluator.close)

        def prediction(values):
            vector = [values.get(index, 0.) for index in range(1001)]
            return {'class_id': max(range(1001), key=vector.__getitem__), 'probabilities': vector}

        baseline = prediction({282:.6, 284:.2, 208:.2})
        other_cat = prediction({282:.2, 284:.6, 208:.2})
        score = evaluator.score(other_cat, baseline, target='cat')
        self.assertFalse(score['failure'], 'changing domestic cat breed is still correct for a cat label')
        self.assertEqual(score['reward'], 0)
        self.assertEqual(score['metrics']['target_label'], 'cat')
        self.assertEqual(score['metrics']['target_class_count'], 5)
        self.assertAlmostEqual(score['metrics']['target_probability'], .8)
        self.assertNotIn('target_id', score['metrics'], 'a category is not one invented breed target')

        dog = prediction({282:.2, 284:.1, 208:.7})
        bad = evaluator.score(dog, baseline, target='cat')
        self.assertTrue(bad['failure'])
        self.assertTrue(bad['metrics']['new_error'])
        self.assertAlmostEqual(bad['metrics']['confidence_drop'], .625)
        self.assertAlmostEqual(bad['reward'], .83125)
        self.assertFalse(evaluator.score(other_cat, dog, target='cat')['failure'])
        self.assertTrue(evaluator.score(other_cat, dog, target='cat')['metrics']['baseline_error'])
        # Category probabilities are used for reward only, not to replace the actual top-1 prediction.
        spread_cats = prediction({282:.3, 284:.3, 208:.4})
        self.assertTrue(evaluator.score(spread_cats, baseline, target='cat')['failure'])
        self.assertFalse(evaluator.score(dog, dog, target='dog')['failure'])
        for wrong in (270, 274, 276, 287, 293, 384, 475, 935):
            for target in ('cat', 'dog'):
                self.assertTrue(evaluator.score(prediction({wrong:1.}), baseline, target=target)['failure'])
        self.assertEqual(evaluator.describe()['label_groups']['cat'], [282,283,284,285,286])
        with self.assertRaises(ValueError):
            dm.resolve_label(evaluator.model.config, 'cat')  # The legacy resolver accepts individual classes only.

    def test_summary_real_generation_and_reference_quality(self):
        evaluator=self.evaluator('summarization',reference_summary='a cat in the garden',
                                 source_text='a cat in the garden',summary_metric='reference',max_new_tokens=4)
        actual=evaluator.predict('a cat')
        self.assertIsInstance(actual['summary'],str)
        self.assertTrue(math.isfinite(actual['quality']))
        perfect=evaluator.measure_summary('a cat in the garden')
        empty=evaluator.measure_summary('')
        repeat=evaluator.measure_summary('cat cat cat cat cat cat')
        self.assertEqual(perfect['quality'],1)
        self.assertEqual(empty['quality'],0)
        self.assertLess(repeat['quality'],.3)
        loss=evaluator.score(empty,perfect)
        self.assertEqual(loss['reward'],1)
        self.assertTrue(loss['failure'])
        self.assertIn('reference',loss['criterion'])
        self.assertEqual(evaluator.score(empty,empty)['reward'],.5)

    def test_summary_generation_overrides_conflicting_checkpoint_defaults(self):
        evaluator=self.evaluator('summarization',reference_summary='a cat',max_new_tokens=4)
        config=evaluator.model.generation_config
        config.min_length=56
        config.early_stopping=True
        config.length_penalty=2.
        config.num_beams=4
        lengths=[]
        effective=[]
        prepare=evaluator.model._prepare_generation_config
        def observed_config(*args,**kwargs):
            result=prepare(*args,**kwargs)
            effective.append(copy.deepcopy(result[0]))
            return result
        original=evaluator.model.generate
        def observed(*args,**kwargs):
            value=original(*args,**kwargs)
            lengths.append(value.shape[1])
            return value
        with warnings.catch_warnings(record=True) as caught, patch.object(evaluator.model,'generate',side_effect=observed), patch.object(evaluator.model,'_prepare_generation_config',side_effect=observed_config):
            warnings.simplefilter('always')
            evaluator.predict('a cat')
        self.assertLessEqual(lengths[0],5)
        self.assertEqual(effective[0].min_length,0)
        self.assertEqual(effective[0].num_beams,1)
        self.assertFalse(effective[0].early_stopping)
        self.assertFalse(any('min_length' in str(item.message) or 'generation flags' in str(item.message) for item in caught),caught)
        self.assertEqual(config.min_length,56)
        self.assertEqual(config.num_beams,4)

    def test_summary_generation_respects_decoder_position_capacity(self):
        from transformers import (BartConfig, BartForConditionalGeneration, BertConfig,
                                  EncoderDecoderConfig, EncoderDecoderModel)

        for architecture in ('bart', 'encoder-decoder'):
            with self.subTest(architecture=architecture):
                path = Path(self.scratch.name)/architecture
                if architecture == 'bart':
                    config = BartConfig(vocab_size=len(VOCAB), d_model=12, encoder_layers=1, decoder_layers=1,
                        encoder_attention_heads=3, decoder_attention_heads=3, encoder_ffn_dim=24, decoder_ffn_dim=24,
                        max_position_embeddings=8, pad_token_id=0, bos_token_id=2, eos_token_id=3,
                        decoder_start_token_id=2, forced_eos_token_id=None)
                    model = BartForConditionalGeneration(config)
                else:
                    def component(limit):
                        return BertConfig(vocab_size=len(VOCAB), hidden_size=12, intermediate_size=24,
                            num_hidden_layers=1, num_attention_heads=3, max_position_embeddings=limit)
                    config = EncoderDecoderConfig.from_encoder_decoder_configs(component(4), component(8))
                    config.pad_token_id, config.decoder_start_token_id, config.eos_token_id = 0, 2, 3
                    model = EncoderDecoderModel(config=config)
                model.generation_config.eos_token_id = None
                model.save_pretrained(path)
                save_tokenizer(path)
                evaluator = dm.create_evaluator(self.model('summarization', path),
                    {'task':'summarization', 'summary_metric':'reference', 'reference_summary':'a cat'}, self.scratch.name)
                self.addCleanup(evaluator.close)
                lengths = []
                generate = evaluator.model.generate
                def observed(*args, **kwargs):
                    output = generate(*args, **kwargs)
                    lengths.append(output.shape[1])
                    return output
                with patch.object(evaluator.model, 'generate', side_effect=observed):
                    try:
                        evaluator.predict('a cat')
                    except (IndexError, RuntimeError) as exc:
                        self.fail(f'Generation exceeded decoder capacity: {exc}')
                self.assertEqual(lengths, [8], 'the decoder start token shares the output position budget')
                self.assertEqual(evaluator.describe()['max_new_tokens'], 7)

    def test_relative_position_summary_output_is_not_limited_by_encoder_tokens(self):
        evaluator = self.evaluator('summarization', reference_summary='a cat', max_new_tokens=16)
        evaluator.model.generation_config.eos_token_id = None
        lengths = []
        generate = evaluator.model.generate
        def observed(*args, **kwargs):
            output = generate(*args, **kwargs)
            lengths.append(output.shape[1])
            return output
        with patch.object(evaluator.model, 'generate', side_effect=observed):
            evaluator.predict('a cat')
        self.assertEqual(lengths, [17], 'relative-position T5 can generate beyond the encoder token limit')
        self.assertEqual(evaluator.describe().get('max_new_tokens'), 16)

    def test_missing_reference_never_silently_uses_another_metric(self):
        with self.assertRaisesRegex(ValueError,'reference'):
            self.evaluator('summarization',summary_metric='reference')

    def test_unsupported_multilabel_and_regression_checkpoints_rejected(self):
        for problem_type,count in [('multi_label_classification',3),('regression',1),(None,1)]:
            with self.subTest(problem_type=problem_type,count=count):
                directory=Path(self.scratch.name)/(str(problem_type)+str(count))
                config=ViTConfig(image_size=16,patch_size=8,hidden_size=12,intermediate_size=24,
                    num_hidden_layers=1,num_attention_heads=3,num_labels=count,problem_type=problem_type)
                ViTForImageClassification(config).save_pretrained(directory)
                ViTImageProcessor(size={'height':16,'width':16}).save_pretrained(directory)
                with self.assertRaisesRegex(ValueError,'single.label|multiclass|regression|multi.label'):
                    dm.create_evaluator(self.model(path=directory),{'task':'classification'},self.scratch.name)

    def test_checkpoint_rejects_missing_classifier_head_weights(self):
        directory=Path(shutil.copytree(self.vision,Path(self.scratch.name)/'broken'))
        path=directory/'model.safetensors'
        state=load_file(path)
        del state['classifier.weight']
        save_file(state,path,metadata={'format':'pt'})
        with self.assertRaisesRegex(ValueError,'weights|checkpoint'):
            dm.create_evaluator(self.model(path=directory),{'task':'classification'},self.scratch.name)

    def test_confident_wrong_prediction_has_higher_reward_than_uncertain_wrong(self):
        evaluator=self.evaluator()
        baseline={'class_id':0,'probabilities':[.8,.15,.05]}
        uncertain={'class_id':1,'probabilities':[.2,.45,.35]}
        confident={'class_id':1,'probabilities':[.2,.75,.05]}
        low=evaluator.score(uncertain,baseline,target=0)
        high=evaluator.score(confident,baseline,target=0)
        self.assertGreater(high['reward'],low['reward'])
        self.assertEqual(high['metrics']['wrong_confidence'],.75)

    def test_grounding_uses_evaluated_input_and_real_nli_model(self):
        from transformers import BertConfig,BertForSequenceClassification
        path=Path(self.scratch.name)/'nli'
        model=BertForSequenceClassification(BertConfig(vocab_size=len(VOCAB),hidden_size=12,
            intermediate_size=24,num_hidden_layers=1,num_attention_heads=3,
            max_position_embeddings=128,num_labels=3,id2label={0:'contradiction',1:'entailment',2:'neutral'}))
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.copy_(torch.tensor([-4.,4.,-4.]))
        model.save_pretrained(path)
        save_tokenizer(path,limit=128)
        evaluator=self.evaluator('summarization',reference_summary='a cat',source_text='a cat in the garden',
            summary_metric='grounded',max_new_tokens=3,_nli_model={'source':'local','name':str(path)})
        measured=evaluator.measure_summary('a cat')
        self.assertGreater(measured['components']['source_support'],.99)
        with patch.object(evaluator.nli,'score',wraps=evaluator.nli.score) as observed:
            evaluator.predict('a dog runs')
        self.assertEqual(observed.call_args.args[0],'a dog runs')
        self.assertIn('evaluated input',evaluator.describe()['grounding_policy'])

    def test_rewards_and_severities_remain_bounded_for_extreme_probabilities(self):
        evaluator=self.evaluator()
        for probabilities in [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1/3,1/3,1/3],[1e-30,.9,.1]]:
            output={'class_id':max(range(3),key=probabilities.__getitem__),'probabilities':probabilities}
            for baseline_probs in [[1.,0.,0.],[0.,1.,0.],[1e-30,.9,.1]]:
                baseline={'class_id':max(range(3),key=baseline_probs.__getitem__),'probabilities':baseline_probs}
                for target in [0,1,2,None]:
                    result=evaluator.score(output,baseline,target=target)
                    self.assertTrue(0<=result['reward']<=1)
                    self.assertTrue(0<=result['severity']<=1)
                    json.dumps(result,allow_nan=False)

    def test_invalid_numeric_scores_rejected(self):
        evaluator=self.evaluator()
        for bad in [float('nan'),float('inf'),-.1,1.1]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                evaluator.score({'class_id':0,'probabilities':[bad,.1,.1]},
                                {'class_id':0,'probabilities':[.8,.1,.1]})


if __name__=='__main__':
    unittest.main()
