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
from PIL import Image,ImageOps
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
            'generation_metric':'clip',
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

    def test_real_generation_scores_visual_requirements_and_preserves_judge_trace(self):
        from test_generation_judge import save_blip_fixture
        judge_path=Path(self.scratch.name)/'judge'
        save_blip_fixture(judge_path)
        evaluator=dm.create_evaluator(self.model(), {'task':'generation','generation_steps':1,
            'generation_size':32,'generation_metric':'checks',
            '_judge_model':{'source':'local','name':str(judge_path)}},self.scratch.name)
        self.addCleanup(evaluator.close)
        checks=[{'question':'is there cat ?', 'answer':'yes'}]
        result=evaluator.predict('a cat',seed=3,checks=checks)
        self.assertEqual(result['checks_total'],1)
        self.assertTrue(result['checks'][0]['answer'])
        self.assertTrue((Path(self.scratch.name)/result['image']).is_file())
        self.assertNotIn('clip_similarity',result)
        self.assertIn('not style compliance',evaluator.describe()['limitations'])
        self.assertEqual(evaluator.score(result,result)['metrics']['quality_drop'],0.)
        with patch.object(evaluator.pipeline,'__call__',side_effect=AssertionError('Missing checks must fail early')):
            with self.assertRaisesRegex(ValueError,'questions|requirements'):
                evaluator.predict('a cat')
        worse={**result,'quality':.25,'components':{'checks_passed':1,'checks_total':4}}
        baseline={**result,'quality':.75}
        scored=evaluator.score(worse,baseline)
        self.assertEqual(scored['severity'],.75)
        self.assertEqual(scored['reward'],.625)
        self.assertTrue(scored['failure'])

    def test_generation_requirement_judge_reuses_weights_and_clip_uses_original_scene(self):
        from test_generation_judge import save_blip_fixture
        from embeddings import EmbeddingManager
        judge_path=Path(self.scratch.name)/'judge'
        save_blip_fixture(judge_path)
        manager=EmbeddingManager(None,Path(self.scratch.name)/'embeddings')
        self.addCleanup(manager.release_encoder)
        settings={'task':'generation','generation_steps':1,'generation_size':32,'generation_metric':'checks',
                  '_judge_model':{'source':'local','name':str(judge_path)}}
        first=manager.evaluation_model(self.model(),settings,self.scratch.name)
        judge=first.judge
        first.close()
        with patch('generation_judge.QuestionJudge',side_effect=AssertionError('Judge loaded twice')):
            second=manager.evaluation_model(self.model(),{**settings,'generation_steps':2},self.scratch.name)
        self.assertIs(second.judge,judge)
        self.assertEqual(second.steps,2)
        second.close()
        clip=manager.evaluation_model(self.model(),{**settings,'generation_metric':'clip',
            '_clip_model':{'source':'local','name':str(self.clip)}},self.scratch.name)
        self.addCleanup(clip.close)
        result=clip.predict('a red cat',reference_prompt='a cat')
        self.assertEqual(result['prompt'],'a red cat')
        self.assertEqual(result['alignment_prompt'],'a cat')
        self.assertEqual(clip.describe()['generation_metric'],'clip')

    def test_combined_reward_defaults_to_maximum_and_retains_missing_bias_evidence(self):
        from test_generation_judge import save_qwen_fixture
        judge_path=Path(self.scratch.name)/'judge'
        save_qwen_fixture(judge_path)
        evaluator=dm.create_evaluator(self.model(), {'task':'generation','generation_metric':'combined','generation_steps':1,
            'generation_size':32, '_judge_model':{'source':'local','name':str(judge_path)}},self.scratch.name)
        self.addCleanup(evaluator.close)
        with patch.object(evaluator.judge,'answer_many',side_effect=[['0','0'],['0','3','1']]):
            baseline=evaluator.predict('a cat',seed=3)
            output=evaluator.predict('a red cat',seed=3,reference_prompt='a cat',
                                     action={'id':36,'style':'photograph','lighting':None,'detail':None})
        score=evaluator.score(output,baseline)
        self.assertEqual(score['reward'],.75)
        self.assertEqual(score['severity'],.75)
        self.assertTrue(score['failure'])
        self.assertIsNone(output['failure_scores']['bias'])
        self.assertEqual(output['failure_evidence']['bias']['status'],'not_evaluated')
        self.assertIsNone(baseline['failure_scores']['instructions'])
        self.assertEqual(evaluator.describe()['generation_metric'],'combined')
        self.assertEqual(score['metrics']['baseline_severity'],0.)
        self.assertEqual(score['metrics']['severity_increase'],.75)
        self.assertTrue((Path(self.scratch.name)/output['image']).is_file())
        json.dumps(score,allow_nan=False)

    def test_combined_bias_comparison_is_counted_cached_and_saves_no_probe_images(self):
        from test_generation_judge import save_qwen_fixture
        judge_path=Path(self.scratch.name)/'judge'
        save_qwen_fixture(judge_path)
        suite={'name':'paired fixture','reference':'equal_error_rates','cases':[
            {'name':'cat scene','groups':[
                {'name':'A','prompt':'a cat','checks':[{'question':'is there cat ?', 'answer':'yes'}]},
                {'name':'B','prompt':'a red cat','checks':[{'question':'is there cat ?', 'answer':'no'}]}]}]}
        settings={'task':'generation','generation_metric':'combined','generation_steps':1,
            'generation_size':32,'generation_bias_suite':suite,'generation_bias_repeats':2,
            '_judge_model':{'source':'local','name':str(judge_path)}}
        evaluator=dm.create_evaluator(self.model(),settings,self.scratch.name)
        self.addCleanup(evaluator.close)
        calls=[]
        evaluator.set_runtime_callbacks(on_model_call=lambda:calls.append(True))
        # Main prompt has dataset checks; the four comparison samples answer yes.
        with patch.object(evaluator.judge,'answer_many',side_effect=[['yes','0']]+[['yes']]*4):
            output=evaluator.predict('a cat',artifact_name='original',checks=[{'question':'is there cat ?', 'answer':'yes'}])
        self.assertEqual(output['failure_scores']['bias'],1.)
        self.assertEqual(len(calls),4)
        self.assertEqual(sorted(path.name for path in Path(self.scratch.name).glob('*.png')),['original.png'])
        with patch.object(evaluator.judge,'answer_many',side_effect=[['yes','0']]):
            second=evaluator.predict('a cat',artifact_name='second',checks=[{'question':'is there cat ?', 'answer':'yes'}])
        self.assertEqual(len(calls),4,'a completed suite is reused only within this run/action')
        self.assertEqual(second['failure_evidence']['bias']['sample_count'],4)
        self.assertEqual(evaluator.score(second,output)['reward'],1.)

        fresh=dm.create_evaluator(self.model(),{**settings,'seed':17},self.scratch.name,shared=evaluator)
        self.addCleanup(fresh.close)
        fresh.set_runtime_callbacks(on_model_call=lambda:calls.append(True))
        self.assertIs(fresh.judge,evaluator.judge)
        with patch.object(fresh.judge,'answer_many',side_effect=[['yes','0']]+[['yes']]*4):
            rerun=fresh.predict('a cat',checks=[{'question':'is there cat ?', 'answer':'yes'}])
        self.assertEqual(len(calls),8,'shared weights do not reuse a previous run\'s comparison measurements')
        self.assertEqual(rerun['failure_evidence']['bias']['seed'],17)

    def test_actual_qwen_rubric_protocol_runs_on_generated_image(self):
        from test_generation_judge import save_qwen_fixture
        judge_path=Path(self.scratch.name)/'judge'
        save_qwen_fixture(judge_path,rating=2)
        evaluator=dm.create_evaluator(self.model(),{'task':'generation','generation_steps':1,
            'generation_size':32,'_judge_model':{'source':'local','name':str(judge_path)}},self.scratch.name)
        self.addCleanup(evaluator.close)
        result=evaluator.predict('a cat')
        self.assertEqual(result['failure_scores'],{'content':.5,'instructions':None,'visual':.5,'bias':None})
        self.assertEqual(result['failure_evidence']['visual']['answer'],'2')
        self.assertEqual(evaluator.score(result,result)['reward'],.5)
        self.assertEqual(evaluator.describe()['generation_metric'],'combined')


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

    @unittest.skipUnless(torch.cuda.is_available(),'Real diffusion CUDA training requires CUDA')
    def test_automatic_lora_training_uses_cuda_for_both_prediction_objectives(self):
        desired=Path(self.scratch.name)/'desired.png'
        Image.new('RGB',(32,32),'red').save(desired)
        original=load_file(self.pipeline/'unet'/'diffusion_pytorch_model.safetensors')
        for prediction_type in ('epsilon','v_prediction'):
            with self.subTest(prediction_type=prediction_type):
                pipeline,revision=dm.load_diffusion(self.model())
                pipeline.scheduler.register_to_config(prediction_type=prediction_type)
                observed=[]
                def require_cuda(module,args,kwargs):
                    self.assertEqual(next(module.parameters()).device.type,'cuda')
                    tensors=[value for value in (*args,*kwargs.values()) if isinstance(value,torch.Tensor)]
                    self.assertTrue(tensors)
                    self.assertTrue(all(value.device.type=='cuda' for value in tensors))
                    observed.append(type(module).__name__)
                handles=[component.register_forward_pre_hook(require_cuda,with_kwargs=True)
                         for component in (pipeline.vae.encoder,pipeline.text_encoder,pipeline.unet)]
                before_cpu=torch.random.get_rng_state()
                before_cuda=torch.cuda.get_rng_state_all()
                try:
                    with patch.object(ft,'load_diffusion',return_value=(pipeline,revision)):
                        result=ft.fine_tune(self.model(),'generation',[{'input':'a cat','target':str(desired)}],
                            Path(self.scratch.name)/prediction_type,
                            {'epochs':1,'learning_rate':.01,'generation_size':32,'seed':4},
                            lambda **fields:None,lambda:False)
                    self.assertEqual(result.get('device'),'cuda')
                    self.assertEqual(len(observed),3)
                    self.assertTrue(torch.equal(before_cpu,torch.random.get_rng_state()))
                    self.assertTrue(all(torch.equal(before,after) for before,after in
                                        zip(before_cuda,torch.cuda.get_rng_state_all())))
                    checkpoint=Path(result['checkpoint'])
                    saved=load_file(checkpoint/'unet'/'diffusion_pytorch_model.safetensors')
                    changed={key for key in original if not torch.equal(original[key],saved[key])}
                    self.assertTrue(changed)
                    self.assertTrue(all(any(piece in key for piece in ('to_q','to_k','to_v','to_out'))
                                        for key in changed),changed)
                    self.assertTrue((checkpoint/'lora'/'pytorch_lora_weights.safetensors').is_file())
                    reloaded,_=dm.load_diffusion(self.model(checkpoint))
                    self.assertTrue(all(torch.equal(value,reloaded.unet.state_dict()[key])
                                        for key,value in saved.items()))
                finally:
                    for handle in handles:
                        handle.remove()
                    torch.random.set_rng_state(before_cpu)
                    torch.cuda.set_rng_state_all(before_cuda)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA diffusion OOM recovery requires CUDA')
    def test_lora_oom_retry_restores_noise_generator_and_replays_one_update(self):
        desired=Path(self.scratch.name)/'desired.png'
        Image.new('RGB',(32,32),'red').save(desired)
        examples=[{'input':'a cat','target':str(desired)}]*3
        results=[]
        for retry in (False,True):
            seen=[]
            failed=[False]
            original=UNet2DConditionModel.forward
            def forward(model,sample,*args,**kwargs):
                seen.append(int(sample.shape[0]))
                result=original(model,sample,*args,**kwargs)
                if retry and not failed[0] and len(seen)==2:
                    def fail_backward(gradient):
                        failed[0]=True
                        raise torch.cuda.OutOfMemoryError('simulated diffusion partial backward allocation')
                    result.sample.register_hook(fail_backward)
                return result
            with patch.object(ft,'_batch_sizes',return_value=(3,2 if retry else 1)), \
                    patch.object(UNet2DConditionModel,'forward',forward):
                result=ft.fine_tune(self.model(),'generation',examples,
                    Path(self.scratch.name)/f'retry-{retry}',
                    {'batch_size':3,'device':'cuda','seed':29,'generation_size':32,'learning_rate':.001},
                    lambda **fields:None,lambda:False)
            results.append(result)
            self.assertEqual(seen,[2,1,1,1,1] if retry else [1,1,1])
            self.assertEqual(result['steps'],1)
            self.assertEqual(result['processed_examples'],3)
            self.assertEqual(result['micro_batch_size'],1)
            self.assertEqual(result['oom_retries'],int(retry))
        self.assertAlmostEqual(results[0]['training_loss'],results[1]['training_loss'],places=6)
        reference=load_file(Path(results[0]['checkpoint'])/'unet'/'diffusion_pytorch_model.safetensors')
        saved=load_file(Path(results[1]['checkpoint'])/'unet'/'diffusion_pytorch_model.safetensors')
        for key in reference:
            torch.testing.assert_close(saved[key],reference[key],rtol=1e-5,atol=1e-7)

    def test_lora_checkpoint_packs_noncontiguous_weights_only_for_serialization(self):
        pipeline,revision=dm.load_diffusion(self.model())
        pipeline.unet.to(memory_format=torch.channels_last)
        pipeline.vae.to(memory_format=torch.channels_last)
        original={name:{key:value.clone() for key,value in getattr(pipeline,name).state_dict().items()}
                  for name in ('unet','vae','text_encoder')}
        strides={name:{key:value.stride() for key,value in getattr(pipeline,name).state_dict().items()}
                 for name in ('unet','vae')}
        self.assertFalse(pipeline.unet.conv_in.weight.is_contiguous())
        self.assertFalse(pipeline.vae.encoder.conv_in.weight.is_contiguous())
        desired=Path(self.scratch.name)/'desired.png'
        Image.new('RGB',(32,32),'red').save(desired)
        with patch.object(ft,'load_diffusion',return_value=(pipeline,revision)):
            result=ft.fine_tune(self.model(),'generation',[{'input':'a cat','target':str(desired)}],
                Path(self.scratch.name)/'training',{'epochs':2,'learning_rate':.01,'generation_size':32,'seed':4,'device':'cpu'},
                lambda **kw:None,lambda:False)
        checkpoint=Path(result['checkpoint'])
        saved=load_file(checkpoint/'unet'/'diffusion_pytorch_model.safetensors')
        changed={key for key,value in original['unet'].items() if not torch.equal(value,saved[key])}
        self.assertTrue(changed)
        self.assertTrue(all(any(piece in key for piece in ('to_q','to_k','to_v','to_out')) for key in changed),changed)
        self.assertTrue(all(torch.equal(value,pipeline.unet.state_dict()[key]) for key,value in saved.items()))
        for name in ('vae','text_encoder'):
            self.assertTrue(all(torch.equal(value,getattr(pipeline,name).state_dict()[key])
                                for key,value in original[name].items()))
        for name in strides:
            self.assertEqual({key:value.stride() for key,value in getattr(pipeline,name).state_dict().items()},strides[name],
                             'packing checkpoint tensors must not change live tensors or leave serialization hooks active')
        source=load_file(self.pipeline/'unet'/'diffusion_pytorch_model.safetensors')
        self.assertTrue(all(torch.equal(value,source[key]) for key,value in original['unet'].items()))

        reloaded,_=dm.load_diffusion(self.model(checkpoint))
        adapter_only,_=dm.load_diffusion(self.model())
        adapter_only.load_lora_weights(checkpoint/'lora',weight_name='pytorch_lora_weights.safetensors')
        adapter_only.unet.fuse_lora(safe_fusing=True)
        adapter_only.unet.unload_lora()
        for key,value in saved.items():
            torch.testing.assert_close(adapter_only.unet.state_dict()[key],value)
        noise=torch.linspace(-1,1,4*16*16).reshape(1,4,16,16)
        condition=torch.linspace(-1,1,3*12).reshape(1,3,12)
        pipeline.unet.eval()
        reloaded.unet.eval()
        with torch.inference_mode():
            expected=pipeline.unet(noise,2,encoder_hidden_states=condition).sample
            actual=reloaded.unet(noise,2,encoder_hidden_states=condition).sample
        torch.testing.assert_close(actual,expected)

    def test_lora_training_is_repeatable_on_each_available_device(self):
        desired=Path(self.scratch.name)/'desired.png'
        Image.new('RGB',(32,32),'red').save(desired)
        devices=['cpu','cuda'] if torch.cuda.is_available() else ['cpu']
        for device in devices:
            with self.subTest(device=device):
                results=[]
                for attempt in range(2):
                    results.append(ft.fine_tune(self.model(),'generation',
                        [{'input':'a cat','target':str(desired)}],Path(self.scratch.name)/f'{device}-{attempt}',
                        {'epochs':1,'learning_rate':.01,'generation_size':32,'seed':7,'device':device},
                        lambda **fields:None,lambda:False))
                self.assertEqual(results[0]['training_loss'],results[1]['training_loss'])
                checkpoints=[load_file(Path(result['checkpoint'])/'unet'/'diffusion_pytorch_model.safetensors')
                             for result in results]
                for key,value in checkpoints[0].items():
                    if device=='cpu':
                        self.assertTrue(torch.equal(value,checkpoints[1][key]),key)
                    else:
                        # cuDNN backward reductions can differ by float32 roundoff.
                        torch.testing.assert_close(value,checkpoints[1][key],rtol=1e-5,atol=1e-7)

    def test_default_optimizer_controls_preserve_lora_checkpoint(self):
        desired=Path(self.scratch.name)/'desired.png'
        Image.new('RGB',(32,32),'red').save(desired)
        results=[]
        for index,controls in enumerate(({}, {'weight_decay':.01,'max_grad_norm':1.})):
            results.append(ft.fine_tune(self.model(),'generation',[{'input':'a cat','target':str(desired)}],
                Path(self.scratch.name)/str(index),
                {'epochs':2,'learning_rate':.01,'generation_size':32,'seed':7,'device':'cpu',**controls},
                lambda **fields:None,lambda:False))
        self.assertEqual(results[0]['training_loss'],results[1]['training_loss'])
        checkpoints=[load_file(Path(result['checkpoint'])/'unet'/'diffusion_pytorch_model.safetensors')
                     for result in results]
        self.assertTrue(all(torch.equal(value,checkpoints[1][key]) for key,value in checkpoints[0].items()))
        for result in results:
            report=json.loads((Path(result['checkpoint'])/'training_report.json').read_text())
            self.assertEqual(report['settings'].get('weight_decay'),.01)
            self.assertEqual(report['settings'].get('max_grad_norm'),1.)

    def test_weight_decay_changes_real_lora_updates(self):
        desired=Path(self.scratch.name)/'desired.png'
        Image.new('RGB',(32,32),'red').save(desired)
        checkpoints=[]
        for index,decay in enumerate((0.,1.)):
            result=ft.fine_tune(self.model(),'generation',[{'input':'a cat','target':str(desired)}],
                Path(self.scratch.name)/str(index),
                {'epochs':1,'learning_rate':.01,'generation_size':32,'seed':7,'device':'cpu','weight_decay':decay},
                lambda **fields:None,lambda:False)
            checkpoints.append(load_file(Path(result['checkpoint'])/'lora'/'pytorch_lora_weights.safetensors'))
        down=[key for key in checkpoints[0] if key.endswith('.lora.down.weight')]
        self.assertTrue(down)
        self.assertTrue(any(not torch.equal(checkpoints[0][key],checkpoints[1][key]) for key in down))
        # LoRA's zero-initialized up matrix gives the down matrix no data gradient
        # in the first update, so decay 1 at learning rate .01 scales it by .99.
        for key in down:
            torch.testing.assert_close(checkpoints[1][key],checkpoints[0][key]*.99,rtol=1e-6,atol=1e-8)

    def test_gradient_limit_bounds_real_lora_updates_and_zero_disables_clipping(self):
        desired=Path(self.scratch.name)/'desired.png'
        Image.new('RGB',(32,32),'red').save(desired)
        optimizer_type=torch.optim.AdamW
        checkpoints=[]
        recorded=[]
        for index,limit in enumerate((0.,100.,1e-5)):
            norms=[]
            class ObservedAdamW(optimizer_type):
                def step(self,*args,**kwargs):
                    gradients=[parameter.grad for group in self.param_groups for parameter in group['params']
                               if parameter.grad is not None]
                    norms.append(float(torch.stack([gradient.norm() for gradient in gradients]).norm()))
                    return super().step(*args,**kwargs)
            with patch.object(torch.optim,'AdamW',ObservedAdamW):
                result=ft.fine_tune(self.model(),'generation',[{'input':'a cat','target':str(desired)}],
                    Path(self.scratch.name)/str(index),
                    {'epochs':2,'learning_rate':.01,'generation_size':32,'seed':7,'device':'cpu','max_grad_norm':limit},
                    lambda **fields:None,lambda:False)
            checkpoints.append(load_file(Path(result['checkpoint'])/'lora'/'pytorch_lora_weights.safetensors'))
            recorded.append(norms)
            self.assertEqual(len(norms),2)
            if limit:
                self.assertTrue(all(0<norm<=limit*1.00001 for norm in norms),norms)
        self.assertGreater(min(recorded[0]),1e-5)
        self.assertEqual(recorded[0],recorded[1])
        self.assertTrue(all(torch.equal(value,checkpoints[1][key]) for key,value in checkpoints[0].items()))
        self.assertTrue(any(not torch.equal(value,checkpoints[2][key]) for key,value in checkpoints[0].items()),
                        'the gradient bound must affect the saved adapter weights')

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

    def test_training_cancellation_stops_preparing_desired_images(self):
        desired=Path(self.scratch.name)/'desired.png'
        Image.new('RGB',(32,32),'red').save(desired)
        prepared=[]
        cancelled=[False]
        fit=ImageOps.fit
        def prepare(*args,**kwargs):
            result=fit(*args,**kwargs)
            prepared.append(result)
            cancelled[0]=True
            return result
        with patch.object(ImageOps,'fit',prepare):
            with self.assertRaises(InterruptedError):
                ft.fine_tune(self.model(),'generation',[{'input':'a cat','target':str(desired)}]*3,
                    Path(self.scratch.name)/'training',{'generation_size':32},
                    lambda **kw:None,lambda:cancelled[0])
        self.assertEqual(len(prepared),1,'cancellation should stop before loading and resizing remaining desired images')
        self.assertFalse((Path(self.scratch.name)/'training'/'checkpoint').exists())
