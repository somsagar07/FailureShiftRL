"""Actual tiny Stable Diffusion, independent CLIP scoring and LoRA optimization."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file,save_file
from diffusers import AutoencoderKL,DDPMScheduler,StableDiffusionPipeline,UNet2DConditionModel
from transformers import CLIPConfig,CLIPModel,CLIPProcessor,CLIPTextConfig,CLIPTextModel,CLIPTokenizerFast,CLIPImageProcessor,CLIPVisionConfig
from test_text_encoders import save_tokenizer,clip_vocabulary
import discovery_models as dm
import fine_tuning as ft


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary=tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root=Path(cls.temporary.name)
        cls.pipeline=cls.root/'pipeline'
        cls.clip=cls.root/'clip'
        tokenizer_path=cls.root/'tokenizer'
        save_tokenizer(tokenizer_path,clip=True)
        tokenizer=CLIPTokenizerFast.from_pretrained(tokenizer_path,local_files_only=True)
        text_cfg=CLIPTextConfig(vocab_size=len(clip_vocabulary()[0]),hidden_size=12,intermediate_size=24,
            num_hidden_layers=1,num_attention_heads=3,max_position_embeddings=12,bos_token_id=2,eos_token_id=3,pad_token_id=0)
        torch.manual_seed(23)
        unet=UNet2DConditionModel(sample_size=16,in_channels=4,out_channels=4,
            down_block_types=('CrossAttnDownBlock2D','DownBlock2D'),up_block_types=('UpBlock2D','CrossAttnUpBlock2D'),
            block_out_channels=(16,32),layers_per_block=1,cross_attention_dim=12,attention_head_dim=4,norm_num_groups=4)
        vae=AutoencoderKL(in_channels=3,out_channels=3,
            down_block_types=('DownEncoderBlock2D','DownEncoderBlock2D'),up_block_types=('UpDecoderBlock2D','UpDecoderBlock2D'),
            block_out_channels=(16,32),layers_per_block=1,latent_channels=4,norm_num_groups=4,sample_size=32)
        pipeline=StableDiffusionPipeline(unet=unet,vae=vae,text_encoder=CLIPTextModel(text_cfg),tokenizer=tokenizer,
            scheduler=DDPMScheduler(num_train_timesteps=20,clip_sample=False,steps_offset=1),safety_checker=None,feature_extractor=None,
            requires_safety_checker=False)
        pipeline.save_pretrained(cls.pipeline,safe_serialization=True)
        vision_cfg=CLIPVisionConfig(hidden_size=12,intermediate_size=24,num_hidden_layers=1,num_attention_heads=3,
                                   image_size=16,patch_size=8)
        CLIPModel(CLIPConfig(text_config=text_cfg.to_dict(),vision_config=vision_cfg.to_dict(),projection_dim=8)).save_pretrained(cls.clip)
        CLIPProcessor(tokenizer=tokenizer,image_processor=CLIPImageProcessor(size={'shortest_edge':16},crop_size={'height':16,'width':16})).save_pretrained(cls.clip)

    def setUp(self):
        self.scratch=tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)

    def model(self,path=None):
        return {'source':'local','name':str(path or self.pipeline),'task':'generation'}

    def evaluator(self,path=None):
        evaluator=dm.create_evaluator(self.model(path),{'task':'generation','generation_steps':2,'generation_size':32,
            '_clip_model':{'source':'local','name':str(self.clip)}},self.scratch.name)
        self.addCleanup(evaluator.close)
        return evaluator

    def test_real_generation_same_noise_seed_and_independent_clip_scores(self):
        evaluator=self.evaluator()
        first=evaluator.predict('a cat',seed=3,artifact_name='first')
        again=evaluator.predict('a cat',seed=3,artifact_name='again')
        self.assertEqual(first['clip_similarity'],again['clip_similarity'])
        np.testing.assert_array_equal(np.asarray(Image.open(Path(self.scratch.name)/first['image'])),
                                      np.asarray(Image.open(Path(self.scratch.name)/again['image'])))
        self.assertTrue(-1<=first['clip_similarity']<=1)
        self.assertIn('demographic',evaluator.describe()['limitations'])
        score=evaluator.score(first,first)
        self.assertEqual(score['metrics']['quality_drop'],0)
        json.dumps(score,allow_nan=False)
        with self.assertRaises(ValueError):
            evaluator.predict('a cat',artifact_name='../escape')

    def test_lora_changes_denoiser_saved_pipeline_reloads_and_original_unchanged(self):
        original=load_file(self.pipeline/'unet'/'diffusion_pytorch_model.safetensors')
        desired=Path(self.scratch.name)/'desired.png'
        Image.new('RGB',(32,32),'red').save(desired)
        result=ft.fine_tune(self.model(),'generation',[{'input':'a cat','target':str(desired)}],
            Path(self.scratch.name)/'training',{'epochs':2,'learning_rate':.01,'generation_size':32,'seed':4},
            lambda **kw:None,lambda:False)
        checkpoint=Path(result['checkpoint'])
        saved=load_file(checkpoint/'unet'/'diffusion_pytorch_model.safetensors')
        changed={key for key in original if not torch.equal(original[key],saved[key])}
        self.assertTrue(changed)
        self.assertTrue(all(any(piece in key for piece in ['to_q','to_k','to_v','to_out']) for key in changed),changed)
        self.assertTrue((checkpoint/'lora'/'pytorch_lora_weights.safetensors').is_file())
        unchanged=load_file(self.pipeline/'unet'/'diffusion_pytorch_model.safetensors')
        self.assertTrue(all(torch.equal(original[key],unchanged[key]) for key in original))
        evaluator=self.evaluator(checkpoint)
        self.assertTrue((Path(self.scratch.name)/evaluator.predict('a cat',seed=2)['image']).is_file())

    def test_generation_alignment_threshold_flags_review_candidates(self):
        evaluator=self.evaluator()
        weak={'quality':.6,'clip_similarity':.2,'components':{'clip_similarity':.2}}
        strong={'quality':.7,'clip_similarity':.4,'components':{'clip_similarity':.4}}
        self.assertTrue(evaluator.score(weak,weak)['failure'])
        self.assertFalse(evaluator.score(strong,strong)['failure'])
        self.assertEqual(evaluator.describe()['min_clip_similarity'],.25)

    def test_pipeline_rejects_custom_component_classes(self):
        broken=Path(shutil.copytree(self.pipeline,Path(self.scratch.name)/'custom'))
        path=broken/'model_index.json'
        index=json.loads(path.read_text())
        index['scheduler']=['custom_module','SchedulerWithSideEffects']
        path.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError,'scheduler|component|standard'):
            self.evaluator(broken)

    def test_generation_missing_denoiser_weights_are_rejected(self):
        broken=Path(shutil.copytree(self.pipeline,Path(self.scratch.name)/'broken'))
        path=broken/'unet'/'diffusion_pytorch_model.safetensors'
        weights=load_file(path)
        del weights[next(iter(weights))]
        save_file(weights,path,metadata={'format':'pt'})
        with self.assertRaisesRegex(ValueError,'weights|checkpoint|missing'):
            self.evaluator(broken)

    def test_generation_and_training_reject_overlength_prompts(self):
        evaluator=self.evaluator()
        prompt=' '.join(['cat']*11)
        with self.assertRaisesRegex(ValueError,'token limit'):
            evaluator.predict(prompt)
        desired=Path(self.scratch.name)/'target.png'
        Image.new('RGB',(32,32)).save(desired)
        with self.assertRaisesRegex(ValueError,'token limit'):
            ft.fine_tune(self.model(),'generation',[{'input':prompt,'target':str(desired)}],
                Path(self.scratch.name)/'training',{'generation_size':32},lambda **kw:None,lambda:False)

    def test_missing_desired_image_is_not_replaced_by_generated_failure(self):
        with self.assertRaisesRegex(ValueError,'desired|absolute'):
            ft.fine_tune(self.model(),'generation',[{'input':'a cat','target':'failed.png'}],self.scratch.name,
                {'generation_size':32},lambda **kw:None,lambda:False)
        self.assertFalse((Path(self.scratch.name)/'checkpoint').exists())
