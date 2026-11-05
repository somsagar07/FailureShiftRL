"""Discovery-only adapters: real tiny model computations, no network access."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (AutoTokenizer, GPT2Config, GPT2LMHeadModel, CLIPConfig,
    CLIPModel, CLIPProcessor, CLIPImageProcessor, CLIPTokenizerFast)

from test_text_encoders import save_tokenizer, VOCAB, clip_vocabulary
from text_encoders import SummarizationEncoder
from embeddings import create_encoder, _selection, _model_key
from discovery_models import create_evaluator


class GeneralModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.causal = cls.root/'causal'
        GPT2LMHeadModel(GPT2Config(vocab_size=len(VOCAB), n_embd=12,
            n_layer=1, n_head=3, n_positions=128, pad_token_id=0,
            eos_token_id=3, bos_token_id=2)).save_pretrained(cls.causal)
        save_tokenizer(cls.causal, limit=128)
        cls.clip = cls.root/'clip'
        config = CLIPConfig(text_config=dict(vocab_size=len(clip_vocabulary()[0]),
            hidden_size=12, intermediate_size=24, num_hidden_layers=1,
            num_attention_heads=3, max_position_embeddings=77, bos_token_id=2,
            eos_token_id=3, pad_token_id=0), vision_config=dict(hidden_size=12,
            intermediate_size=24, num_hidden_layers=1, num_attention_heads=3,
            image_size=16, patch_size=8), projection_dim=8)
        CLIPModel(config).save_pretrained(cls.clip)
        save_tokenizer(cls.clip, clip=True, limit=77)
        CLIPProcessor(tokenizer=CLIPTokenizerFast.from_pretrained(cls.clip),
            image_processor=CLIPImageProcessor(size={'shortest_edge':16},
                crop_size={'height':16,'width':16})).save_pretrained(cls.clip)

    def test_causal_embeddings_are_masked_final_source_states(self):
        encoder = SummarizationEncoder(dict(source='local', name=str(self.causal), task='summarization'))
        actual = encoder.encode(['a cat', 'a cat in the garden'])
        self.assertEqual(actual.shape, (2,12))
        np.testing.assert_allclose(actual[0], encoder.encode(['a cat'])[0], atol=2e-6)
        tokens = encoder._prepare(['a cat'])
        with torch.inference_mode():
            expected = encoder.model.base_model(**tokens, return_dict=True, use_cache=False).last_hidden_state[0]
        np.testing.assert_allclose(actual[0], expected.mean(0).numpy(), atol=2e-6)

    def test_causal_summary_decodes_only_completion_and_reserves_context(self):
        selection = dict(source='local', name=str(self.causal), task='summarization')
        evaluator = create_evaluator(selection, dict(task='summarization',device='cpu',
            reference_summary='a cat',max_new_tokens=8), self.root/'outputs')
        self.addCleanup(evaluator.close)
        # Fixed token generation isolates the prompt slicing contract; the
        # source embedding test above exercises the actual causal forward pass.
        def generate(**kwargs):
            ids=kwargs['input_ids']
            self.assertLessEqual(ids.shape[-1]+kwargs['generation_config'].max_new_tokens,128)
            return torch.cat([ids,torch.tensor([[8,3]])],dim=1)
        from unittest.mock import patch
        with patch.object(evaluator.model,'generate',generate):
            out=evaluator.predict('a cat in the garden')
        self.assertEqual(out['summary'],'cat')
        with self.assertRaisesRegex(ValueError,'limit|context|tokens'):
            evaluator.predict('a cat '*150)

    def test_causal_dataset_truncation_keeps_instruction_and_reports_visible_source(self):
        evaluator = create_evaluator(dict(source='local',name=str(self.causal),task='summarization'),
            dict(task='summarization',device='cpu',dataset_id='fixture',max_new_tokens=8),self.root/'dataset')
        self.addCleanup(evaluator.close)
        out=evaluator.predict('a cat '*150,reference_summary='a cat')
        self.assertTrue(out['input_coverage']['truncated'])
        self.assertLessEqual(out['input_coverage']['model_tokens']+8,128)

    def test_clip_features_and_labelled_discovery_use_supplied_vocabulary(self):
        selection=dict(source='local',name=str(self.clip),task='classification',candidate_labels=['cat','dog'])
        encoder=create_encoder(selection)
        image=Image.new('RGB',(20,20),'red')
        features=encoder.encode([image])
        self.assertEqual(features.shape,(1,8))
        self.assertEqual(encoder.class_labels,{0:'cat',1:'dog'})
        evaluator=create_evaluator(selection,dict(task='classification',device='cpu'),self.root/'classification',encoder=encoder)
        self.addCleanup(evaluator.close)
        output=evaluator.predict(image)
        self.assertEqual(len(output['probabilities']),2)
        self.assertAlmostEqual(sum(output['probabilities']),1.,places=5)
        score=evaluator.score(output,output,target=output['label'])
        self.assertFalse(score['failure'])
        self.assertEqual(evaluator.resolve_label('dog'),1)

    def test_candidate_vocabulary_is_validated_and_part_of_identity(self):
        base=dict(source='local',name=str(self.clip),task='classification')
        a=_selection(dict(base,candidate_labels=['cat','dog']))
        b=_selection(dict(base,candidate_labels=['cat','bird']))
        self.assertNotEqual(_model_key(a),_model_key(b))
        for labels in (['cat'],['cat','cat'],['cat',7]):
            with self.assertRaises(ValueError):_selection(dict(base,candidate_labels=labels))

    def test_preview_keeps_class_vocabulary(self):
        from server import validate_config
        model=dict(source='huggingface',name='openai/clip-vit-base-patch32',candidate_labels=['cat','dog'])
        result=validate_config(dict(task='classification',dataset_id='a'*32,model=model))
        self.assertEqual(result['model']['candidate_labels'],['cat','dog'])

    def test_renamed_local_clip_automatically_uses_dataset_classes(self):
        import shutil,time
        from server import PreviewManager,validate_config
        from embeddings import EmbeddingManager
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model=root/'vision-backbone';shutil.copytree(self.clip,model)
            for label in ('cat','dog'):
                folder=root/'images'/label;folder.mkdir(parents=True)
                Image.new('RGB',(20,20),'red').save(folder/'image.png')
            previews=PreviewManager(root/'previews');manager=EmbeddingManager(previews,root/'embeddings')
            def wait(getter,job):
                deadline=time.monotonic()+10
                while time.monotonic()<deadline:
                    value=getter(job['id'])
                    if value['status'] in ('ready','completed'):return value
                    if value['status']=='error':self.fail(value['error'])
                    time.sleep(.01)
                self.fail('Worker timed out')
            dataset=wait(previews.datasets.get,previews.datasets.start(dict(source='folder',path=str(root/'images'))))
            # Dataset metadata boundary, as returned by HF ClassLabel. Folder
            # names alone are not treated as reviewed ground-truth labels.
            previews.datasets.label_sets[dataset['id']]=dict(class_names=['cat','dog'],
                column='label',values={'0':'cat','1':'dog'},original_values={'0':0,'1':1})
            config=validate_config(dict(task='classification',dataset_id=dataset['id'],model=dict(source='local',name=str(model)),
                action_space=dict(rotation=[0],darkness=[],saturation=[])))
            preview=wait(previews.get,previews.start(config))
            result=wait(manager.get,manager.start(preview['id'],device='cpu'))
            self.assertEqual(result['labels']['loaded'],2)
            self.assertEqual(result['model']['candidate_labels'],['cat','dog'])
            manager.release_encoder()


if __name__=='__main__': unittest.main()
