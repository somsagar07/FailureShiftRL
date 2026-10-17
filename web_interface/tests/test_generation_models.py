"""Actual tiny Stable Diffusion, independent CLIP scoring and LoRA optimization."""
import json
import gc
import os
import shutil
import tempfile
import threading
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

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

    def test_component_workers_overlap_real_loads_without_global_init_patches(self):
        barrier=threading.Barrier(2)
        threads=set()
        original_register=torch.nn.Module.register_parameter
        encoder=dm.DiffusionPromptEncoder(self.model())
        originals={cls:cls.from_pretrained for cls in (UNet2DConditionModel,AutoencoderKL)}
        def observed(cls):
            def load(*args,**kwargs):
                threads.add(threading.get_ident())
                try:
                    barrier.wait(timeout=5)
                except threading.BrokenBarrierError:
                    self.fail('Independent component loads did not overlap.')
                self.assertFalse(kwargs.get('low_cpu_mem_usage',True))
                self.assertIs(torch.nn.Module.register_parameter,original_register)
                return originals[cls](*args,**kwargs)
            return load
        with patch.dict(os.environ,{'FAILURESHIFT_COMPONENT_WORKERS':'2'}), \
                patch.object(UNet2DConditionModel,'from_pretrained',side_effect=observed(UNet2DConditionModel)), \
                patch.object(AutoencoderKL,'from_pretrained',side_effect=observed(AutoencoderKL)):
            pipeline,_=dm.load_diffusion(self.model(),encoder=encoder)
        self.assertEqual(len(threads),2)
        self.assertNotIn(threading.get_ident(),threads)
        self.assertIs(torch.nn.Module.register_parameter,original_register)
        for component in (pipeline.unet,pipeline.vae):
            self.assertTrue(all(parameter.dtype==torch.float32 and parameter.device.type=='cpu'
                                for parameter in component.parameters()))

    def test_serial_and_parallel_components_preserve_weights_and_seeded_image(self):
        loaded=[]
        for workers in ('1','2'):
            with patch.dict(os.environ,{'FAILURESHIFT_COMPONENT_WORKERS':workers}):
                pipeline,_=dm.load_diffusion(self.model())
            pipeline.set_progress_bar_config(disable=True)
            with torch.inference_mode():
                image=pipeline('a cat',num_inference_steps=2,height=32,width=32,
                               generator=torch.Generator(device='cpu').manual_seed(7)).images[0]
            loaded.append((pipeline,np.asarray(image)))
        np.testing.assert_array_equal(loaded[0][1],loaded[1][1])
        for name in ('unet','vae','text_encoder'):
            original=getattr(loaded[0][0],name).state_dict()
            parallel=getattr(loaded[1][0],name).state_dict()
            self.assertEqual(set(original),set(parallel))
            self.assertTrue(all(torch.equal(original[key],parallel[key]) for key in original))

    def test_bin_components_load_with_weights_only_and_match_safetensors(self):
        reference,_=dm.load_diffusion(self.model())
        directory=Path(shutil.copytree(self.pipeline,Path(self.scratch.name)/'bin'))
        for name in ('unet','vae'):
            component=directory/name
            (component/'diffusion_pytorch_model.safetensors').unlink()
            getattr(reference,name).save_pretrained(component,safe_serialization=False)
        calls=[]
        original_load=torch.load
        def observed(*args,**kwargs):
            calls.append(kwargs.get('weights_only'))
            return original_load(*args,**kwargs)
        for workers in ('1','2'):
            with self.subTest(workers=workers), patch.dict(os.environ,{'FAILURESHIFT_COMPONENT_WORKERS':workers}), \
                    patch.object(torch,'load',side_effect=observed):
                loaded,_=dm.load_diffusion(self.model(directory))
                for name in ('unet','vae'):
                    self.assertTrue(all(torch.equal(value,getattr(loaded,name).state_dict()[key])
                                        for key,value in getattr(reference,name).state_dict().items()))
                loaded.set_progress_bar_config(disable=True)
                with torch.inference_mode():
                    image=loaded('a cat',num_inference_steps=2,height=32,width=32,
                                 generator=torch.Generator(device='cpu').manual_seed(7)).images[0]
                self.assertEqual(image.size,(32,32))
        self.assertTrue(calls)
        self.assertTrue(all(flag is True for flag in calls))

    def test_sharded_bin_components_use_sequential_memory_efficient_loading(self):
        directory=Path(shutil.copytree(self.pipeline,Path(self.scratch.name)/'bin-sharded'))
        original=UNet2DConditionModel.from_pretrained(self.pipeline,subfolder='unet',local_files_only=True)
        (directory/'unet'/'diffusion_pytorch_model.safetensors').unlink()
        original.save_pretrained(directory/'unet',safe_serialization=False,max_shard_size='100KB')
        self.assertTrue((directory/'unet'/'diffusion_pytorch_model.bin.index.json').is_file())
        observed=[]
        loader=UNet2DConditionModel.from_pretrained
        def load_unet(*args,**kwargs):
            observed.append((threading.get_ident(),kwargs.get('low_cpu_mem_usage'),kwargs.get('use_safetensors')))
            return loader(*args,**kwargs)
        with patch.dict(os.environ,{'FAILURESHIFT_COMPONENT_WORKERS':'2'}), \
                patch.object(UNet2DConditionModel,'from_pretrained',side_effect=load_unet):
            pipeline,_=dm.load_diffusion(self.model(directory))
        self.assertEqual(observed,[(threading.get_ident(),True,False)])
        self.assertTrue(all(torch.equal(value,pipeline.unet.state_dict()[key]) for key,value in original.state_dict().items()))

    def test_safetensors_remain_preferred_when_bin_files_also_exist(self):
        directory=Path(shutil.copytree(self.pipeline,Path(self.scratch.name)/'both-formats'))
        for name in ('unet','vae'):
            (directory/name/'diffusion_pytorch_model.bin').write_bytes(b'not a usable checkpoint')
        loaded,_=dm.load_diffusion(self.model(directory))
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in loaded.unet.parameters()))

    def test_offline_component_format_lookup_uses_the_pinned_snapshot(self):
        revision='a'*40
        with patch.dict(os.environ,{'FAILURESHIFT_COMPONENT_WORKERS':'2'}), \
                patch('huggingface_hub.list_repo_files',side_effect=ConnectionError('offline')), \
                patch('huggingface_hub.snapshot_download',return_value=str(self.pipeline)) as cached:
            workers,formats=dm._component_loading_options({'source':'huggingface','name':'test/pipeline'},revision)
        cached.assert_called_once_with('test/pipeline',revision=revision,local_files_only=True)
        self.assertEqual(workers,2)
        self.assertEqual(formats,{'unet':True,'vae':True})

    def test_default_component_loading_keeps_memory_efficient_initialization(self):
        observed=[]
        loader=UNet2DConditionModel.from_pretrained
        def load_unet(*args,**kwargs):
            observed.append((threading.get_ident(),kwargs.get('low_cpu_mem_usage')))
            return loader(*args,**kwargs)
        with patch.dict(os.environ) as environment, \
                patch.object(UNet2DConditionModel,'from_pretrained',side_effect=load_unet):
            environment.pop('FAILURESHIFT_COMPONENT_WORKERS',None)
            dm.load_diffusion(self.model())
        self.assertEqual(observed,[(threading.get_ident(),True)])

    def test_component_loading_preserves_rng_for_later_seeded_training(self):
        encoder=dm.DiffusionPromptEncoder(self.model())
        for workers in ('1','2'):
            with self.subTest(workers=workers), patch.dict(os.environ,{'FAILURESHIFT_COMPONENT_WORKERS':workers}):
                before=torch.random.get_rng_state()
                dm.load_diffusion(self.model(),encoder=encoder)
                self.assertTrue(torch.equal(before,torch.random.get_rng_state()))

    def test_component_worker_configuration_rejects_unbounded_or_invalid_counts(self):
        encoder=dm.DiffusionPromptEncoder(self.model())
        for workers in ('0','3','-1','all','1.5'):
            with self.subTest(workers=workers), patch.dict(os.environ,{'FAILURESHIFT_COMPONENT_WORKERS':workers}):
                with self.assertRaisesRegex(ValueError,'COMPONENT_WORKERS|workers'):
                    dm.load_diffusion(self.model(),encoder=encoder)

    def test_failed_component_does_not_keep_other_component_alive(self):
        encoder=dm.DiffusionPromptEncoder(self.model())
        references=[]
        vae_loaded=threading.Event()
        original=AutoencoderKL.from_pretrained
        def load_vae(*args,**kwargs):
            model,info=original(*args,**kwargs)
            references.append(weakref.ref(model))
            vae_loaded.set()
            return model,info
        def broken_unet(*args,**kwargs):
            self.assertTrue(vae_loaded.wait(timeout=5),'The other component never loaded.')
            raise ValueError('invalid checkpoint')
        with patch.dict(os.environ,{'FAILURESHIFT_COMPONENT_WORKERS':'2'}), \
                patch.object(AutoencoderKL,'from_pretrained',side_effect=load_vae), \
                patch.object(UNet2DConditionModel,'from_pretrained',side_effect=broken_unet):
            with self.assertRaisesRegex(ValueError,'invalid checkpoint'):
                dm.load_diffusion(self.model(),encoder=encoder)
        gc.collect()
        self.assertEqual(len(references),1)
        self.assertTrue(all(reference() is None for reference in references))

    def test_sharded_components_preserve_accelerate_loading_and_serialize_initialization(self):
        directory=Path(shutil.copytree(self.pipeline,Path(self.scratch.name)/'sharded'))
        original=UNet2DConditionModel.from_pretrained(self.pipeline,subfolder='unet',local_files_only=True)
        original.save_pretrained(directory/'unet',safe_serialization=True,max_shard_size='100KB')
        self.assertTrue((directory/'unet'/'diffusion_pytorch_model.safetensors.index.json').is_file())
        observed=[]
        loader=UNet2DConditionModel.from_pretrained
        def load_unet(*args,**kwargs):
            observed.append((threading.get_ident(),kwargs.get('low_cpu_mem_usage')))
            return loader(*args,**kwargs)
        with patch.dict(os.environ,{'FAILURESHIFT_COMPONENT_WORKERS':'2'}), \
                patch.object(UNet2DConditionModel,'from_pretrained',side_effect=load_unet):
            pipeline,_=dm.load_diffusion(self.model(directory))
        self.assertEqual(observed,[(threading.get_ident(),True)])
        self.assertTrue(all(torch.equal(value,pipeline.unet.state_dict()[key]) for key,value in original.state_dict().items()))

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
