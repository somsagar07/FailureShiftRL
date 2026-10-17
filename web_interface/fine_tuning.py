"""Train reviewed targets into separate, independently loadable checkpoints."""
import gc
import json
import math
import os
import random
import shutil
import tempfile
from pathlib import Path

from discovery_models import ClassificationEvaluator, load_diffusion
from text_encoders import SummarizationEncoder


def _settings(settings):
    result={'epochs':1,'learning_rate':1e-4,'seed':0,'batch_size':1,'generation_size':256,**settings}
    for key,low,high in [('epochs',1,100),('batch_size',1,32),('seed',0,2**32-1),('generation_size',32,1024)]:
        value=result[key]
        if isinstance(value,bool) or not isinstance(value,int) or not low<=value<=high:
            raise ValueError(f'{key} must be an integer from {low} to {high}.')
    rate=result['learning_rate']
    if isinstance(rate,bool) or not isinstance(rate,(int,float)) or not math.isfinite(rate) or not 0<rate<=.1:
        raise ValueError('learning_rate must be finite and greater than zero, at most 0.1.')
    if result['generation_size']%8:
        raise ValueError('generation_size must be a multiple of eight.')
    return result


def _check(cancelled):
    if cancelled():
        raise InterruptedError('Fine-tuning cancelled; no final checkpoint was saved.')


def _examples(examples):
    if not isinstance(examples,list) or not examples or len(examples)>256:
        raise ValueError('Provide between 1 and 256 reviewed examples.')
    for example in examples:
        if not isinstance(example,dict) or 'input' not in example or 'target' not in example or example['target'] is None:
            raise ValueError('Every example requires an explicit reviewed target.')


def _classifier(model,examples,directory):
    from PIL import Image
    adapter=ClassificationEvaluator(model,{'task':'classification'},directory)
    prepared=[]
    for example in examples:
        if not isinstance(example['input'],Image.Image):
            raise ValueError('Classifier training input must be an image.')
        target=adapter.resolve_label(example['target'])
        inputs=adapter.processor(images=example['input'],return_tensors='pt')
        prepared.append((dict(inputs),adapter.torch.tensor([target],dtype=adapter.torch.long)))
    base_ids={id(parameter) for parameter in adapter.model.base_model.parameters()}
    trainable=[]
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(id(parameter) not in base_ids)
        if parameter.requires_grad:
            trainable.append(parameter)
    if not trainable:
        raise ValueError('This classifier has no separately identifiable pretrained classification head.')
    adapter.model.train()
    adapter.model.base_model.eval()  # preserve encoder dropout and batch statistics
    return adapter,prepared,trainable


def _summarizer(model,examples):
    adapter=SummarizationEncoder(model)
    prepared=[]
    for example in examples:
        source,target=example['input'],example['target']
        if not isinstance(source,str) or not source.strip() or not isinstance(target,str) or not target.strip():
            raise ValueError('Summarization training needs text inputs and nonempty reviewed target summaries.')
        inputs=adapter._prepare([source],adapter.prefix)
        labels=adapter._prepare([target])['input_ids']
        # decoder positional capacity can differ from encoder; never truncate labels.
        limit=getattr(adapter.model.config,'max_position_embeddings',adapter.max_tokens)
        if labels.shape[-1]>min(adapter.max_tokens,limit):
            raise ValueError('Reviewed target summary exceeds the model output token limit.')
        prepared.append((inputs,labels))
    adapter.model.train()
    return adapter,prepared,list(adapter.model.parameters())


def _pad_batch(torch,examples,pad_token_id,labels_pad=-100):
    keys=examples[0][0].keys()
    batch={}
    for key in keys:
        tensors=[item[0][key] for item in examples]
        if tensors[0].ndim==2:
            limit=max(value.shape[1] for value in tensors)
            pad=pad_token_id if key=='input_ids' else 0
            batch[key]=torch.cat([torch.nn.functional.pad(value,(0,limit-value.shape[1]),value=pad) for value in tensors])
        else:
            batch[key]=torch.cat(tensors)
    labels=[item[1] for item in examples]
    if labels[0].ndim==2:
        limit=max(value.shape[1] for value in labels)
        batch['labels']=torch.cat([torch.nn.functional.pad(value,(0,limit-value.shape[1]),value=labels_pad) for value in labels])
    else:
        batch['labels']=torch.cat(labels)
    return batch


def _generation(model,examples,settings):
    import numpy as np
    import torch
    from PIL import Image,ImageOps
    from diffusers import DDPMScheduler
    from peft import LoraConfig
    pipeline,revision=load_diffusion(model)
    pipeline.to('cpu')
    size=settings['generation_size']
    prepared=[]
    for example in examples:
        caption,target=example['input'],example['target']
        if not isinstance(caption,str) or not caption.strip() or not isinstance(target,str):
            raise ValueError('Generation training requires captions and explicit desired-image absolute paths.')
        path=Path(target)
        if not path.is_absolute() or not path.is_file():
            raise ValueError('Reviewed desired image must be an existing absolute image path.')
        with Image.open(path) as image:
            image=ImageOps.fit(image.convert('RGB'),(size,size),method=Image.Resampling.LANCZOS)
            pixels=torch.from_numpy(np.asarray(image,dtype=np.float32).copy()).permute(2,0,1).unsqueeze(0)/127.5-1
        tokens=pipeline.tokenizer(caption,padding='max_length',truncation=False,return_tensors='pt',verbose=False)
        if tokens['input_ids'].shape[-1]>min(pipeline.tokenizer.model_max_length,pipeline.text_encoder.config.max_position_embeddings):
            raise ValueError('Training caption exceeds diffusion model token limit.')
        prepared.append((pixels,tokens))
    for component in (pipeline.vae,pipeline.text_encoder,pipeline.unet):
        component.requires_grad_(False)
        component.eval()
    pipeline.unet.add_adapter(LoraConfig(r=4,lora_alpha=4,init_lora_weights='gaussian',
                                         target_modules=['to_q','to_k','to_v','to_out.0']))
    pipeline.unet.train()
    trainable=[parameter for parameter in pipeline.unet.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError('No trainable diffusion LoRA weights were created.')
    scheduler=DDPMScheduler.from_config(pipeline.scheduler.config)
    if scheduler.config.prediction_type not in ('epsilon','v_prediction'):
        raise ValueError('Only epsilon or v_prediction diffusion training is supported.')
    return pipeline,prepared,trainable,scheduler,revision


def fine_tune(model,task,examples,directory,settings,progress,cancelled):
    """Optimize explicit user targets; original files are never overwritten.

    The checkpoint represents training on reviewed examples only. Training loss
    is not a held-out evaluation or evidence of model improvement.
    """
    import torch
    settings=_settings(settings)
    _examples(examples)
    _check(cancelled)
    destination=Path(directory).resolve()
    if model.get('source')=='local':
        source=Path(model['name']).expanduser().resolve()
        if destination==source or source in destination.parents:
            raise ValueError('Save into a separate directory outside the original source model.')
    final=destination/'checkpoint'
    if final.exists():
        raise ValueError('Checkpoint destination already exists; choose a separate training directory.')
    destination.mkdir(parents=True,exist_ok=True)
    if task not in ('classification','summarization','generation'):
        raise ValueError('Unsupported fine-tuning task.')
    adapter=pipeline=None
    staging=None
    try:
        progress(phase='validating reviewed examples',completed=0,total=len(examples))
        # Isolate this job's RNG state from unrelated embedding or RL operations.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(settings['seed'])
            if task=='classification':
                adapter,prepared,trainable=_classifier(model,examples,destination)
            elif task=='summarization':
                adapter,prepared,trainable=_summarizer(model,examples)
            else:
                pipeline,prepared,trainable,scheduler,revision=_generation(model,examples,settings)
            _check(cancelled)
            optimizer=torch.optim.AdamW(trainable,lr=settings['learning_rate'])
            rng=random.Random(settings['seed'])
            generator=torch.Generator(device='cpu').manual_seed(settings['seed'])
            total=settings['epochs']*math.ceil(len(prepared)/settings['batch_size'])
            steps=0
            losses=[]
            for epoch in range(settings['epochs']):
                order=list(range(len(prepared)))
                rng.shuffle(order)
                for offset in range(0,len(order),settings['batch_size']):
                    _check(cancelled)
                    chosen=[prepared[index] for index in order[offset:offset+settings['batch_size']]]
                    optimizer.zero_grad(set_to_none=True)
                    if pipeline is None:
                        pad=adapter.tokenizer.pad_token_id if task=='summarization' else 0
                        batch=_pad_batch(torch,chosen,pad)
                        loss=adapter.model(**batch).loss
                    else:
                        pixels=torch.cat([value[0] for value in chosen])
                        ids=torch.cat([value[1]['input_ids'] for value in chosen])
                        mask=torch.cat([value[1]['attention_mask'] for value in chosen])
                        with torch.no_grad():
                            latents=pipeline.vae.encode(pixels).latent_dist.sample(generator=generator)*pipeline.vae.config.scaling_factor
                            condition=pipeline.text_encoder(ids,attention_mask=mask if getattr(pipeline.text_encoder.config,'use_attention_mask',False) else None)[0]
                        noise=torch.randn(latents.shape,generator=generator)
                        timesteps=torch.randint(0,scheduler.config.num_train_timesteps,(latents.shape[0],),generator=generator)
                        noisy=scheduler.add_noise(latents,noise,timesteps)
                        prediction=pipeline.unet(noisy,timesteps,encoder_hidden_states=condition).sample
                        target=noise if scheduler.config.prediction_type=='epsilon' else scheduler.get_velocity(latents,noise,timesteps)
                        loss=torch.nn.functional.mse_loss(prediction.float(),target.float())
                    if not torch.isfinite(loss):
                        raise ValueError('Training produced a nonfinite loss; no checkpoint saved.')
                    loss.backward()
                    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in trainable):
                        raise ValueError('Training produced nonfinite gradients; no checkpoint saved.')
                    torch.nn.utils.clip_grad_norm_(trainable,1.)
                    optimizer.step()
                    steps+=1
                    losses.append(float(loss.detach()))
                    progress(phase='training reviewed examples',completed=steps,total=total,loss=losses[-1])
            _check(cancelled)
            progress(phase='saving separate checkpoint',completed=steps,total=total,loss=losses[-1])
            staging=Path(tempfile.mkdtemp(prefix='.checkpoint-',dir=destination))
            if pipeline is None:
                adapter.model.save_pretrained(staging,safe_serialization=True)
                if task=='classification':
                    adapter.processor.save_pretrained(staging)
                else:
                    adapter.tokenizer.save_pretrained(staging)
            else:
                from peft.utils import get_peft_model_state_dict
                from diffusers.utils import convert_state_dict_to_diffusers
                from diffusers import StableDiffusionPipeline
                state=convert_state_dict_to_diffusers(get_peft_model_state_dict(pipeline.unet))
                StableDiffusionPipeline.save_lora_weights(staging/'lora',unet_lora_layers=state,safe_serialization=True)
                pipeline.unet.fuse_lora(safe_fusing=True)
                pipeline.unet.unload_lora()
                pipeline.save_pretrained(staging,safe_serialization=True)
            metadata={'task':task,'steps':steps,'training_loss':sum(losses)/len(losses),
                      'reviewed_examples':len(examples),'settings':settings,
                      'evaluation':'Training loss on reviewed examples only; no held-out improvement measured.',
                      'method':{'classification':'classification head','summarization':'supervised seq2seq','generation':'UNet attention LoRA, fused loadable pipeline plus separate adapter'}[task]}
            (staging/'training_report.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
            _check(cancelled)
            os.replace(staging,final)
            staging=None
            return {'checkpoint':str(final),**metadata}
    finally:
        if staging is not None:
            shutil.rmtree(staging,ignore_errors=True)
        if adapter is not None and callable(getattr(adapter,'close',None)):
            adapter.close()
        del adapter,pipeline
        gc.collect()
