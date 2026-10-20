"""Stream explicit reviewed targets into independent, loadable checkpoints."""
import gc
import json
import math
import os
import random
import shutil
import tempfile
from contextlib import ExitStack
from functools import lru_cache
from itertools import islice
from pathlib import Path

from discovery_models import ClassificationEvaluator, load_diffusion
from text_encoders import SummarizationEncoder, _encoder_device


def _settings(settings):
    result={'epochs':1,'learning_rate':1e-4,'seed':0,'batch_size':'auto','generation_size':256,'device':'auto',
            'weight_decay':.01,'max_grad_norm':1.,**settings}
    if not isinstance(result['device'],str) or result['device'] not in ('auto','cpu','cuda'):
        raise ValueError('Choose a fine-tuning device: auto, cpu, or cuda.')
    batch=result['batch_size']
    if batch!='auto' and (type(batch) is not int or not 1<=batch<=1024):
        raise ValueError('batch_size must be auto or an integer from 1 to 1024.')
    for key,low,high in [('epochs',1,100),('seed',0,2**32-1),('generation_size',32,1024)]:
        value=result[key]
        if isinstance(value,bool) or not isinstance(value,int) or not low<=value<=high:
            raise ValueError(f'{key} must be an integer from {low} to {high}.')
    rate=result['learning_rate']
    if isinstance(rate,bool) or not isinstance(rate,(int,float)) or not math.isfinite(rate) or not 0<rate<=.1:
        raise ValueError('learning_rate must be finite and greater than zero, at most 0.1.')
    for key,high in [('weight_decay',1.),('max_grad_norm',100.)]:
        value=result[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not 0<=value<=high or not math.isfinite(value):
            raise ValueError(f'{key} must be finite and from 0 to {high:g}.')
    if result['generation_size']%8:
        raise ValueError('generation_size must be a multiple of eight.')
    return result


def _check(cancelled):
    if cancelled():
        raise InterruptedError('Fine-tuning cancelled; no final checkpoint was saved.')


def _examples(examples):
    # Accept the sequence protocol as well as lists without traversing or copying
    # the source. A lazy source may represent millions of transformed examples.
    if isinstance(examples,(str,bytes,dict)) or not callable(getattr(examples,'__getitem__',None)):
        raise ValueError('Provide a nonempty sized, indexable sequence of reviewed examples.')
    try:
        count=len(examples)
    except (TypeError,OverflowError) as exc:
        raise ValueError('Provide a nonempty sized, indexable sequence of reviewed examples.') from exc
    if count<1:
        raise ValueError('Provide at least one reviewed example.')


def _example(examples,index):
    example=examples[index]
    if not isinstance(example,dict) or 'input' not in example or 'target' not in example or example['target'] is None:
        raise ValueError('Every example requires an explicit reviewed target.')
    return example


def _targets(examples,cancelled,validated):
    target=getattr(examples,'target',None)
    for index in range(len(examples)):
        _check(cancelled)
        value=target(index) if callable(target) else _example(examples,index)['target']
        if value is None:
            raise ValueError('Every example requires an explicit reviewed target.')
        yield value
        _check(cancelled)
        if (index+1)%256==0 or index+1==len(examples):
            validated(index+1)
            _check(cancelled)


class _PreparedExamples:
    """Preprocess only requested examples; never cache the full tensor corpus."""
    def __init__(self,examples,prepare,cancelled):
        self.examples,self.prepare,self.cancelled=examples,prepare,cancelled

    def __len__(self):
        return len(self.examples)

    def __getitem__(self,index):
        _check(self.cancelled)
        result=self.prepare(_example(self.examples,index))
        _check(self.cancelled)
        return result


def _classifier(model,examples,directory,cancelled=lambda:False,*,device='cpu',validated=lambda count:None):
    from PIL import Image
    adapter=ClassificationEvaluator(model,{'task':'classification','device':'cpu'},directory)
    groups={}
    counts={'single_class':0,'groups':{}}
    num_labels=adapter.model.config.num_labels

    @lru_cache(maxsize=256,typed=True)
    def resolve_known(target):
        target=adapter.resolve_label(target)
        ids=list(adapter.label_groups[target]) if isinstance(target,str) else [target]
        if not ids or any(type(index) is not int or not 0<=index<num_labels for index in ids) or len(set(ids))!=len(ids):
            raise ValueError('Classifier target must contain distinct valid model class IDs.')
        return target,ids

    def resolve(target):
        if not isinstance(target,(int,str)):
            raise ValueError('Supply an explicit model class ID or exact model label.')
        return resolve_known(target)

    for target in _targets(examples,cancelled,validated):
        target,ids=resolve(target)
        if isinstance(target,str):
            groups[target]=ids
            counts['groups'][target]=counts['groups'].get(target,0)+1
        else:
            counts['single_class']+=1

    def prepare(example):
        if not isinstance(example['input'],Image.Image):
            raise ValueError('Classifier training input must be an image.')
        _,ids=resolve(example['target'])
        inputs=adapter.processor(images=example['input'],return_tensors='pt')
        allowed=adapter.torch.zeros((1,num_labels),dtype=adapter.torch.bool)
        allowed[0,ids]=True
        return dict(inputs),allowed

    base_ids={id(parameter) for parameter in adapter.model.base_model.parameters()}
    trainable=[]
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(id(parameter) not in base_ids)
        if parameter.requires_grad:
            trainable.append(parameter)
    if not trainable:
        raise ValueError('This classifier has no separately identifiable pretrained classification head.')
    adapter.model.to(device)
    adapter.device=device
    adapter.model.train()
    adapter.model.base_model.eval()  # preserve encoder dropout and batch statistics
    objective={'objective':'negative_log_target_probability',
               'objective_formula':'logsumexp(all_logits) - logsumexp(target_logits)',
               'target_groups':groups,'target_counts':counts}
    return adapter,_PreparedExamples(examples,prepare,cancelled),trainable,objective


def _summarizer(model,examples,cancelled=lambda:False,*,device='cpu',validated=lambda count:None):
    adapter=SummarizationEncoder({**model,'device':'cpu'})
    limits=[getattr(adapter.tokenizer,'model_max_length',None)]
    for config in (adapter.model.config,adapter.model.get_decoder().config):
        limits.extend(getattr(config,name,None) for name in
                      ('max_position_embeddings','max_decoder_position_embeddings','max_target_positions','n_positions'))
    limits=[int(value) for value in limits if isinstance(value,(int,float)) and
            math.isfinite(value) and 0<value<10**9 and int(value)==value]
    output_limit=min(2048,min(limits) if limits else adapter.max_tokens)

    def labels_for(target):
        if not isinstance(target,str) or not target.strip():
            raise ValueError('Summarization training needs text inputs and nonempty reviewed target summaries.')
        # Target mode preserves decoder-specific vocabularies and language tokens.
        labels=adapter.tokenizer(text_target=[target],padding=True,truncation=False,return_tensors='pt',
                                 return_attention_mask=True,return_token_type_ids=False,verbose=False)
        if labels['input_ids'].shape[-1]>output_limit:
            raise ValueError('Reviewed target summary exceeds the model output token limit.')
        result=labels['input_ids'].masked_fill(labels['attention_mask']==0,-100)
        if not (result!=-100).any():
            raise ValueError('Reviewed target summary must contain output tokens.')
        return result

    for target in _targets(examples,cancelled,validated):
        labels_for(target)

    def prepare(example):
        source=example['input']
        if not isinstance(source,str) or not source.strip():
            raise ValueError('Summarization training needs text inputs and nonempty reviewed target summaries.')
        options={'truncate':True} if getattr(examples,'truncate_inputs',False) else {}
        return adapter._prepare([source],adapter.prefix,**options),labels_for(example['target'])

    # The adapter's tokenizer/_prepare device remains CPU. The model itself
    # moves to the training device; only the active padded batch follows it.
    adapter.model.to(device=device,dtype=adapter.torch.float32)
    adapter.model.train()
    return adapter,_PreparedExamples(examples,prepare,cancelled),list(adapter.model.parameters())


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


def _generation(model,examples,settings,cancelled=lambda:False,*,device='cpu',validated=lambda count:None):
    import numpy as np
    import torch
    from PIL import Image,ImageOps
    from diffusers import DDPMScheduler
    from peft import LoraConfig
    pipeline,revision=load_diffusion(model)
    size=settings['generation_size']

    def desired_path(target):
        if not isinstance(target,str):
            raise ValueError('Generation training requires captions and explicit desired-image absolute paths.')
        path=Path(target)
        if not path.is_absolute() or not path.is_file():
            raise ValueError('Reviewed desired image must be an existing absolute image path.')
        return path

    for target in _targets(examples,cancelled,validated):
        # Verify every desired image before any update, without resizing it into
        # a retained training tensor or decoding augmented classifier inputs.
        try:
            with Image.open(desired_path(target)) as image:
                image.verify()
        except (OSError,SyntaxError) as exc:
            raise ValueError('Reviewed desired image must be a readable image.') from exc

    def prepare(example):
        caption=example['input']
        if not isinstance(caption,str) or not caption.strip():
            raise ValueError('Generation training requires captions and explicit desired-image absolute paths.')
        with Image.open(desired_path(example['target'])) as image:
            image=ImageOps.fit(image.convert('RGB'),(size,size),method=Image.Resampling.LANCZOS)
            pixels=torch.from_numpy(np.asarray(image,dtype=np.float32).copy()).permute(2,0,1).unsqueeze(0)/127.5-1
        tokens=pipeline.tokenizer(caption,padding='max_length',truncation=False,return_tensors='pt',verbose=False)
        if tokens['input_ids'].shape[-1]>min(pipeline.tokenizer.model_max_length,pipeline.text_encoder.config.max_position_embeddings):
            raise ValueError('Training caption exceeds diffusion model token limit.')
        return pixels,tokens

    for component in (pipeline.vae,pipeline.text_encoder,pipeline.unet):
        component.requires_grad_(False)
        component.eval()
    pipeline.unet.add_adapter(LoraConfig(r=4,lora_alpha=4,init_lora_weights='gaussian',
                                         target_modules=['to_q','to_k','to_v','to_out.0']))
    pipeline.to(device)
    pipeline.unet.train()
    trainable=[parameter for parameter in pipeline.unet.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError('No trainable diffusion LoRA weights were created.')
    scheduler=DDPMScheduler.from_config(pipeline.scheduler.config)
    if scheduler.config.prediction_type not in ('epsilon','v_prediction'):
        raise ValueError('Only epsilon or v_prediction diffusion training is supported.')
    return pipeline,_PreparedExamples(examples,prepare,cancelled),trainable,scheduler,revision


def _batch_sizes(settings,task,device,count):
    """Use free device memory as a hint; actual OOM feedback remains decisive."""
    auto={'classification':8,'summarization':4,'generation':1}
    if device=='cuda':
        import torch
        free,_=torch.cuda.mem_get_info()
        tier=0 if free<8*1024**3 else 1 if free<16*1024**3 else 2
        auto={'classification':(8,16,32)[tier],'summarization':(2,4,8)[tier],'generation':(1,1,2)[tier]}
    effective=min(count,auto[task]) if settings['batch_size']=='auto' else settings['batch_size']
    return effective,min(count,effective,auto[task])


def _shuffled_indices(count,rng,buffer_size=4096):
    # A seeded affine bijection first disperses class-/input-ordered corpora.
    # The bounded buffer then mixes nearby members of that traversal. This is
    # not a uniform random permutation; it visits every index exactly once
    # without allocating an index array proportional to the corpus size.
    start=rng.randrange(count)
    stride=1
    if count>1:
        stride=rng.randrange(1,count)
        while math.gcd(stride,count)!=1:
            stride=rng.randrange(1,count)
    source=((start+index*stride)%count for index in range(count))
    buffer=list(islice(source,buffer_size))
    for index in source:
        slot=rng.randrange(len(buffer))
        yield buffer[slot]
        buffer[slot]=index
    rng.shuffle(buffer)
    yield from buffer


def _index_batches(indices,size):
    while batch:=list(islice(indices,size)):
        yield batch


def _rng_state(torch,device,generator):
    return (torch.random.get_rng_state(),torch.cuda.get_rng_state() if device=='cuda' else None,
            generator.get_state())


def _restore_rng(torch,device,generator,state):
    torch.random.set_rng_state(state[0])
    if device=='cuda':
        torch.cuda.set_rng_state(state[1])
    generator.set_state(state[2])


def _is_cuda_oom(torch,error,device):
    return device=='cuda' and (isinstance(error,torch.OutOfMemoryError) or 'CUDA out of memory' in str(error))


def _backward_chunk(torch,prepared,indices,loss_for,effective_count,cancelled):
    # This frame owns all active input/activation tensors. On failure its
    # traceback is discarded before empty_cache and a smaller replay begins.
    chosen=[prepared[index] for index in indices]
    _check(cancelled)
    loss,weight=loss_for(chosen)
    if not torch.isfinite(loss):
        raise ValueError('Training produced a nonfinite loss; no checkpoint saved.')
    (loss*(weight/effective_count)).backward()
    return float(loss.detach())*weight,weight


def _accumulate(torch,prepared,indices,micro_size,loss_for,trainable,cancelled):
    numerator=0.
    denominator=0
    for offset in range(0,len(indices),micro_size):
        _check(cancelled)
        value,weight=_backward_chunk(torch,prepared,indices[offset:offset+micro_size],loss_for,len(indices),cancelled)
        numerator+=value
        denominator+=weight
    # Seq2seq loss is a mean over non-padding target tokens, not examples or
    # microbatches. Scaling the accumulated token sum once preserves that mean.
    scale=len(indices)/denominator
    for parameter in trainable:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)
    return numerator/denominator


def _save_diffusion_pipeline(pipeline,directory):
    """Pack checkpoint tensors without changing live parameter layouts or values."""
    import torch

    def pack_state(module,state,prefix,metadata):
        for key,value in state.items():
            if isinstance(value,torch.Tensor) and not value.is_contiguous():
                state[key]=value.contiguous()

    # Some pretrained convolutions use channels-last storage. Safetensors needs
    # contiguous checkpoint copies; leave the trained module and optimizer alone.
    with ExitStack() as cleanup:
        for component in pipeline.components.values():
            if isinstance(component,torch.nn.Module):
                cleanup.callback(component.register_state_dict_post_hook(pack_state).remove)
        pipeline.save_pretrained(directory,safe_serialization=True)


def fine_tune(model,task,examples,directory,settings,progress,cancelled):
    """Train all explicit targets with bounded preprocessing and safe OOM replay.

    Batch size is effective examples per optimizer update. The microbatch size
    can shrink without changing that update's examples or target weighting.
    Training loss is not a held-out evaluation or evidence of improvement.
    """
    import torch
    settings=_settings(settings)
    _examples(examples)
    _check(cancelled)
    if task not in ('classification','summarization','generation'):
        raise ValueError('Unsupported fine-tuning task.')
    device=_encoder_device(torch,settings['device'])
    execution={'device':device,'device_name':torch.cuda.get_device_name() if device=='cuda' else 'CPU'}
    effective,micro_size=_batch_sizes(settings,task,device,len(examples))
    total=settings['epochs']*((len(examples)+effective-1)//effective)
    status={'effective_batch_size':effective,'micro_batch_size':micro_size,'oom_retries':0,
            'epoch':0,'epochs':settings['epochs'],'processed_examples':0,
            'examples_per_epoch':len(examples),'total_examples':len(examples)*settings['epochs']}
    destination=Path(directory).resolve()
    if model.get('source')=='local':
        source=Path(model['name']).expanduser().resolve()
        if destination==source or source in destination.parents:
            raise ValueError('Save into a separate directory outside the original source model.')
    final=destination/'checkpoint'
    if final.exists():
        raise ValueError('Checkpoint destination already exists; choose a separate training directory.')
    destination.mkdir(parents=True,exist_ok=True)
    adapter=pipeline=None
    staging=None
    steps=0

    def emit(phase,**fields):
        progress(phase=phase,completed=steps,total=total,**execution,**status,**fields)

    def validated(count):
        emit('validating reviewed examples',validated_examples=count)

    try:
        emit('validating reviewed examples',validated_examples=0)
        _check(cancelled)
        cuda_devices=[torch.cuda.current_device()] if device=='cuda' else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(settings['seed'])
            if cuda_devices:
                torch.cuda.manual_seed(settings['seed'])
            objective={}
            if task=='classification':
                adapter,prepared,trainable,objective=_classifier(model,examples,destination,cancelled,device=device,validated=validated)
            elif task=='summarization':
                adapter,prepared,trainable=_summarizer(model,examples,cancelled,device=device,validated=validated)
                objective['input_policy']=dict(max_input_tokens=adapter.max_tokens,
                    truncate=bool(getattr(examples,'truncate_inputs',False)),
                    policy=('Dataset articles use the model input limit; target summaries must fit the output limit.'
                            if getattr(examples,'truncate_inputs',False) else
                            'Source text and target summaries must fit the model token limits.'))
            else:
                pipeline,prepared,trainable,scheduler,revision=_generation(model,examples,settings,cancelled,device=device,validated=validated)
            _check(cancelled)
            optimizer=torch.optim.AdamW(trainable,lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
            rng=random.Random(settings['seed'])
            generator=torch.Generator(device=device).manual_seed(settings['seed'])

            def loss_for(chosen):
                if pipeline is None:
                    pad=adapter.tokenizer.pad_token_id if task=='summarization' else 0
                    batch={key:value.to(device) for key,value in _pad_batch(torch,chosen,pad).items()}
                    if task=='classification':
                        allowed=batch.pop('labels')
                        logits=adapter.model(**batch).logits
                        loss=(logits.logsumexp(-1)-logits.masked_fill(~allowed,-torch.inf).logsumexp(-1)).mean()
                        weight=len(chosen)
                    else:
                        weight=int((batch['labels']!=-100).sum())
                        loss=adapter.model(**batch).loss
                    return loss,weight
                pixels=torch.cat([value[0] for value in chosen]).to(device=device,dtype=pipeline.vae.dtype)
                ids=torch.cat([value[1]['input_ids'] for value in chosen]).to(device)
                mask=torch.cat([value[1]['attention_mask'] for value in chosen]).to(device)
                with torch.no_grad():
                    latents=pipeline.vae.encode(pixels).latent_dist.sample(generator=generator)*pipeline.vae.config.scaling_factor
                    condition=pipeline.text_encoder(ids,attention_mask=mask if getattr(pipeline.text_encoder.config,'use_attention_mask',False) else None)[0]
                noise=torch.randn(latents.shape,generator=generator,device=latents.device,dtype=latents.dtype)
                timesteps=torch.randint(0,scheduler.config.num_train_timesteps,(latents.shape[0],),
                                        generator=generator,device=latents.device)
                noisy=scheduler.add_noise(latents,noise,timesteps)
                prediction=pipeline.unet(noisy,timesteps,encoder_hidden_states=condition).sample
                target=noise if scheduler.config.prediction_type=='epsilon' else scheduler.get_velocity(latents,noise,timesteps)
                return torch.nn.functional.mse_loss(prediction.float(),target.float()),len(chosen)

            mean_loss=0.
            for epoch in range(settings['epochs']):
                status['epoch']=epoch+1
                order=_shuffled_indices(len(prepared),rng)
                for indices in _index_batches(order,effective):
                    _check(cancelled)
                    state=_rng_state(torch,device,generator)
                    while True:
                        optimizer.zero_grad(set_to_none=True)
                        try:
                            loss=_accumulate(torch,prepared,indices,status['micro_batch_size'],loss_for,trainable,cancelled)
                        except RuntimeError as exc:
                            if not _is_cuda_oom(torch,exc,device):
                                raise
                            # Release failed graph frames before freeing the CUDA
                            # cache. No optimizer update occurred in this region.
                            exc.__traceback__=None
                            optimizer.zero_grad(set_to_none=True)
                            _restore_rng(torch,device,generator,state)
                            gc.collect()
                            torch.cuda.empty_cache()
                            attempted_size=min(status['micro_batch_size'],len(indices))
                            if attempted_size==1:
                                raise ValueError('CUDA cannot fit one training example. Free GPU memory, choose a smaller model or generation size, or select CPU; no checkpoint was saved.') from None
                            status['micro_batch_size']=max(1,attempted_size//2)
                            status['oom_retries']+=1
                            emit('retrying smaller training microbatch')
                            _check(cancelled)
                            continue
                        break
                    _check(cancelled)
                    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in trainable):
                        raise ValueError('Training produced nonfinite gradients; no checkpoint saved.')
                    if settings['max_grad_norm']>0:
                        torch.nn.utils.clip_grad_norm_(trainable,settings['max_grad_norm'])
                    try:
                        optimizer.step()
                    except RuntimeError as exc:
                        if _is_cuda_oom(torch,exc,device):
                            raise ValueError('CUDA ran out of memory during the optimizer update, which may be partial and was not retried. Free GPU memory, use a smaller model, or select CPU; no checkpoint was saved.') from exc
                        raise
                    steps+=1
                    mean_loss+=(loss-mean_loss)/steps
                    status['processed_examples']+=len(indices)
                    emit('training reviewed examples',loss=loss)
            _check(cancelled)
            emit('saving separate checkpoint',loss=loss)
            _check(cancelled)
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
                _save_diffusion_pipeline(pipeline,staging)
            metadata={'task':task,'steps':steps,'training_loss':mean_loss,
                      'reviewed_examples':len(examples),'settings':settings,
                      'evaluation':'Training loss on reviewed examples only; no held-out improvement measured.',
                      'method':{'classification':'classification head','summarization':'supervised seq2seq','generation':'UNet attention LoRA, fused loadable pipeline plus separate adapter'}[task],
                      **execution,**objective,**status}
            (staging/'training_report.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
            _check(cancelled)
            os.replace(staging,final)
            staging=None
            return {'checkpoint':str(final),**metadata}
    except torch.OutOfMemoryError as exc:
        if device!='cuda':
            raise
        raise ValueError('CUDA could not fit model setup or checkpoint saving. Free GPU memory, use a smaller model, or select CPU; no checkpoint was saved.') from exc
    finally:
        if staging is not None:
            shutil.rmtree(staging,ignore_errors=True)
        if adapter is not None and callable(getattr(adapter,'close',None)):
            adapter.close()
        del adapter,pipeline
        gc.collect()
