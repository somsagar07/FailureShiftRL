"""Real model inference and domain-specific, baseline-aware review signals."""
import gc
import copy
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path

from classification_labels import model_label_groups
from model_loading import configure_parallel_loading
from text_encoders import (_dependencies, _encoder_device, _options, _pretrained, _resolved_revision,
                           _selection, _token_limit, SummarizationEncoder, DiffusionPromptEncoder)


def unit(value, name='score'):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
        raise ValueError(f'{name} must be finite and between zero and one.')
    return float(value)


def resolve_label(config,target):
    if isinstance(target,int) and not isinstance(target,bool) and 0<=target<config.num_labels:
        return target
    if isinstance(target,str):
        matches=[int(key) for key,label in config.id2label.items() if label==target]
        if len(matches)==1 and 0<=matches[0]<config.num_labels:
            return matches[0]
        if len(matches)>1:
            raise ValueError('Model label is ambiguous; supply an explicit model class ID.')
    raise ValueError('Supply an explicit model class ID or exact model label; dataset IDs are not mapped automatically.')


def _device(torch,config):
    return _encoder_device(torch,config.get('device','cpu'))


def _evaluation_encoder(cls,model,device,encoder,shared):
    selected=shared.encoder if shared is not None else encoder
    if selected is not None and selected.device==device:
        return selected
    if shared is not None:
        raise ValueError('Shared evaluation resources must use the requested device.')
    # Keep an Analyze encoder on its own device if evaluation requests another.
    return cls({**model,'device':device})


class Evaluator:
    def __init__(self,config,directory):
        configure_parallel_loading()
        self.config=dict(config)
        self.directory=Path(directory).resolve()
        self.directory.mkdir(parents=True,exist_ok=True)
        self.threshold=unit(config.get('failure_threshold',.15),'failure_threshold')

    def close(self):
        if not getattr(self,'_shared_resources',False):
            close=getattr(getattr(self,'judge',None),'close',None)
            if callable(close):
                close()
        for name in ('model','pipeline','encoder','scorer','nli','processor','tokenizer','judge','_generation_runtime'):
            if hasattr(self,name):
                delattr(self,name)
        if getattr(self,'_shared_resources',False):
            return
        gc.collect()
        if hasattr(self,'torch') and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def describe(self):
        return {'task':self.config['task'],'criterion':self.criterion,'revision':self.revision,
                'failure_threshold':self.threshold,'limitations':self.limitations}


class ClassificationEvaluator(Evaluator):
    criterion='labelled classification error or original-prediction consistency candidate'
    limitations='Unlabelled inputs measure prediction consistency, not verified errors.'

    def __init__(self,model,config,directory,*,encoder=None,shared=None):
        super().__init__(config,directory)
        selection=_selection(model,'classification')
        self.torch,transformers=_dependencies()
        self.device=_device(self.torch,config)
        if shared is not None:
            if shared.device!=self.device:
                raise ValueError('Shared evaluation resources must use the requested device.')
            self.model,self.processor,self.revision=shared.model,shared.processor,shared.revision
            return
        try:
            cfg=transformers.AutoConfig.from_pretrained(selection['name'],**_options(selection))
            if cfg.num_labels<2 or getattr(cfg,'problem_type',None) not in (None,'single_label_classification'):
                raise ValueError('Only single-label multiclass image classifiers are supported; regression and multi-label checkpoints need different metrics.')
            self.revision=_resolved_revision(selection,getattr(cfg,'_commit_hash',None))
            options=_options(selection,self.revision)
            loaded,info=transformers.AutoModelForImageClassification.from_pretrained(
                selection['name'],config=cfg,weights_only=True,output_loading_info=True,**options)
            self.model=_pretrained(loaded,info,self.torch).to(self.device)
            self.processor=transformers.AutoImageProcessor.from_pretrained(selection['name'],use_fast=False,**options)
        except Exception as exc:
            raise ValueError(f'Could not load classifier pretrained weights/processor: {exc}') from exc

    def resolve_label(self,target):
        try:
            return resolve_label(self.model.config,target)
        except ValueError:
            if isinstance(target,str) and target.strip().lower() in self.label_groups:
                return target.strip().lower()
            raise

    @cached_property
    def label_groups(self):
        return model_label_groups(self.model.config.id2label)

    def predict(self,value,*,seed=0,artifact_name='sample'):
        inputs=self.processor(images=value,return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            logits=self.model(**inputs).logits
            if logits.shape!=(1,self.model.config.num_labels) or not self.torch.isfinite(logits).all():
                raise ValueError('Classifier returned invalid or nonfinite logits.')
            probabilities=logits.softmax(-1)[0].cpu().tolist()
        index=max(range(len(probabilities)),key=probabilities.__getitem__)
        return {'class_id':index,'label':self.model.config.id2label[index],
                'confidence':probabilities[index],'probabilities':probabilities}

    def describe(self):
        return {**super().describe(),
                'reward_formula':'0.5*error + 0.25*error*max_wrong_probability + 0.25*max(0,(baseline_target_probability-target_probability)/baseline_target_probability)',
                'severity_formula':'1 for a target error, otherwise relative target-confidence drop',
                'label_policy':'Dataset/manual targets map to model classes or declared ImageNet pet groups. Group errors use the actual top-1 class; target confidence sums group probabilities. Legacy unlabelled scores measure original-prediction consistency.',
                'label_groups':self.label_groups}

    def _probabilities(self,output):
        vector=output['probabilities']
        if not isinstance(vector,list) or len(vector)!=self.model.config.num_labels:
            raise ValueError('Wrong classifier probability vector.')
        vector=[unit(value,'probability') for value in vector]
        if abs(sum(vector)-1)>1e-4:
            raise ValueError('Probabilities must sum to one.')
        index=output['class_id']
        if isinstance(index,bool) or not isinstance(index,int) or index!=max(range(len(vector)),key=vector.__getitem__):
            raise ValueError('Class ID does not match probabilities.')
        return vector

    def score(self,output,baseline,*,target=None):
        current,original=self._probabilities(output),self._probabilities(baseline)
        labelled=target is not None
        target_id=self.resolve_label(target) if labelled else baseline['class_id']
        grouped=isinstance(target_id,str)
        target_ids=set(self.label_groups[target_id]) if grouped else {target_id}
        current_target=sum(current[index] for index in target_ids)
        original_target=sum(original[index] for index in target_ids)
        error=output['class_id'] not in target_ids
        baseline_error=baseline['class_id'] not in target_ids
        drop=max(0.,(original_target-current_target)/max(original_target,1e-8))
        # Actual labelled mistakes retain search signal even when present before perturbation.
        wrong_confidence=max((probability for index,probability in enumerate(current) if index not in target_ids),default=0.)
        reward=.5*float(error)+.25*float(error)*wrong_confidence+.25*drop
        severity=1. if error else drop
        target_info={'target_label':target_id,'target_class_count':len(target_ids)} if grouped else {
            'target_id':target_id,'target_label':self.model.config.id2label[target_id]}
        return {'reward':reward,'severity':severity,'failure':bool(error),
                'criterion':'labelled classification error' if labelled else 'original-prediction consistency candidate',
                'metrics':{**target_info,
                'target_probability':current_target,'baseline_target_probability':original_target,
                'confidence_drop':drop,'wrong_confidence':wrong_confidence,'baseline_error':bool(baseline_error),'error':bool(error),
                'new_error':bool(error and not baseline_error)}}


def rouge_l(reference,summary):
    reference,summary=[re.findall(r'\w+',text.casefold()) for text in (reference,summary)]
    if not reference or not summary:
        return 0.
    previous=[0]*(len(summary)+1)
    for token in reference:
        current=[0]
        for index,other in enumerate(summary):
            current.append(previous[index]+1 if token==other else max(previous[index+1],current[-1]))
        previous=current
    return 2*previous[-1]/(len(reference)+len(summary))


def nonrepetition(text):
    tokens=re.findall(r'\w+',text.casefold())
    if not tokens:
        return 0.
    if len(tokens)<3:
        return 1.
    grams=list(zip(tokens,tokens[1:]))
    return len(set(grams))/len(grams)


class GroundedScorer:
    def __init__(self,model,device):
        configure_parallel_loading()
        self.torch,transformers=_dependencies()
        selection=_selection(model,'summarization')
        cfg=transformers.AutoConfig.from_pretrained(selection['name'],**_options(selection))
        revision=_resolved_revision(selection,getattr(cfg,'_commit_hash',None))
        options=_options(selection,revision)
        loaded,info=transformers.AutoModelForSequenceClassification.from_pretrained(
            selection['name'],config=cfg,weights_only=True,output_loading_info=True,**options)
        self.model=_pretrained(loaded,info,self.torch).to(device)
        self.tokenizer=transformers.AutoTokenizer.from_pretrained(selection['name'],**options)
        ids=[int(key) for key,label in cfg.id2label.items() if label.lower() in ('entailment','entail')]
        if len(ids)!=1:
            raise ValueError('NLI scorer must declare an unambiguous entailment label in id2label.')
        self.entailment_id=ids[0]
        self.limit=_token_limit(self.tokenizer,cfg,default=512)
        self.device=device
        self.identity={'name':selection['name'],'revision':revision}

    def score(self,source,summary):
        if not summary.strip():
            return 0.
        source_ids=self.tokenizer.encode(source,add_special_tokens=False)
        window=max(4,(self.limit-8)//2)
        windows=[self.tokenizer.decode(source_ids[i:i+window],skip_special_tokens=True)
                 for i in range(0,len(source_ids),max(1,window//2))]
        if not windows:
            raise ValueError('Grounded scoring needs nonempty source text.')
        claims=[text.strip() for text in re.split(r'(?<=[.!?])\s+|\n+',summary) if text.strip()]
        scores=[]
        for claim in claims:
            if len(self.tokenizer.encode(claim,add_special_tokens=False))>window:
                raise ValueError('Generated summary sentence exceeds grounded scorer input limit.')
            support=[]
            for premise in windows:
                inputs=self.tokenizer(premise,claim,return_tensors='pt',truncation=False).to(self.device)
                if inputs['input_ids'].shape[-1]>self.limit:
                    raise ValueError('Grounded NLI pair exceeds scorer input limit.')
                with self.torch.inference_mode():
                    logits=self.model(**inputs).logits
                    if not self.torch.isfinite(logits).all():
                        raise ValueError('NLI returned nonfinite logits.')
                    support.append(float(logits.softmax(-1)[0,self.entailment_id]))
            scores.append(max(support))
        return sum(scores)/len(scores) if scores else 0.


class SummarizationEvaluator(Evaluator):
    limitations='ROUGE measures reference overlap; NLI estimates source support. Both are automated review signals.'

    def __init__(self,model,config,directory,*,encoder=None,shared=None):
        super().__init__(config,directory)
        self.metric=config.get('summary_metric','reference')
        if self.metric not in ('reference','grounded'):
            raise ValueError('Summary metric must be reference or grounded.')
        self.reference=config.get('reference_summary','')
        self.source=config.get('source_text','')
        self.dataset_references=bool(config.get('dataset_id'))
        self.truncate_inputs=self.dataset_references
        if not self.dataset_references and (not isinstance(self.reference,str) or not self.reference.strip()):
            raise ValueError('Summary evaluation requires an explicit reference summary.')
        if not self.dataset_references and self.metric=='grounded' and (not isinstance(self.source,str) or not self.source.strip()):
            raise ValueError('Grounded summary scoring needs original source text.')
        self.torch,_=_dependencies()
        self.device=_device(self.torch,config)
        self.encoder=_evaluation_encoder(SummarizationEncoder,model,self.device,encoder,shared)
        self.model,self.tokenizer,self.torch=self.encoder.model,self.encoder.tokenizer,self.encoder.torch
        self.revision=self.encoder.revision
        self.max_new_tokens=config.get('max_new_tokens',96)
        if isinstance(self.max_new_tokens,bool) or not isinstance(self.max_new_tokens,int) or not 1<=self.max_new_tokens<=512:
            raise ValueError('max_new_tokens must be an integer from 1 to 512.')
        limits=[]
        for decoder_config in (self.model.config,self.model.get_decoder().config):
            limits.extend(getattr(decoder_config,name,None) for name in
                          ('max_position_embeddings','max_decoder_position_embeddings','max_target_positions','n_positions'))
        limits=[int(value) for value in limits if not isinstance(value,bool) and isinstance(value,(int,float))
                and 0<value<10**9 and int(value)==value]
        if limits:
            # The decoder start token shares absolute positional capacity with
            # generated tokens. Encoder/tokenizer input limits do not apply here.
            self.max_new_tokens=min(self.max_new_tokens,min(limits)-1)
            if self.max_new_tokens<1:
                raise ValueError('The decoder needs capacity for a start token and generated output.')
        self.criterion='reference ROUGE-L review signal' if self.metric=='reference' else 'reference ROUGE-L and grounded NLI review signal'
        if self.metric=='grounded':
            self.nli=shared.nli if shared is not None else GroundedScorer(config.get('_nli_model',{'source':'huggingface','name':'cross-encoder/nli-MiniLM2-L6-H768'}),self.device)

    def measure_summary(self,text,source=None,*,reference=None):
        reference=self.reference if reference is None else reference
        if not isinstance(reference,str) or not reference.strip():
            raise ValueError('Summary evaluation requires this article’s explicit reference summary.')
        overlap,repetition=rouge_l(reference,text),nonrepetition(text)
        metrics={'rouge_l':overlap,'nonrepetition':repetition}
        quality=overlap*repetition
        if self.metric=='grounded':
            metrics['source_support']=self.nli.score(self.source if source is None else source,text)
            quality=math.sqrt(quality*metrics['source_support'])
        return {'summary':text,'quality':quality,'components':metrics}

    def predict(self,value,*,seed=0,artifact_name='sample',reference_summary=None):
        if self.dataset_references and (not isinstance(reference_summary,str) or not reference_summary.strip()):
            raise ValueError('Dataset inference requires this article’s explicit reference summary.')
        options={'truncate':True} if self.truncate_inputs else {}
        inputs={key:tensor.to(self.device) for key,tensor in self.encoder._prepare([value],self.encoder.prefix,**options).items()}
        generation=copy.deepcopy(self.model.generation_config)
        generation.min_length=0
        generation.min_new_tokens=0
        generation.early_stopping=False
        generation.length_penalty=1.
        generation.num_beams=1
        generation.num_beam_groups=1
        generation.num_return_sequences=1
        generation.do_sample=False
        generation.max_new_tokens=self.max_new_tokens
        with self.torch.inference_mode():
            generated=self.model.generate(**inputs,generation_config=generation,use_model_defaults=False)
        return self.measure_summary(self.tokenizer.decode(generated[0],skip_special_tokens=True),source=value,
                                    reference=reference_summary)

    def score(self,output,baseline,*,target=None):
        quality,original=unit(output['quality']),unit(baseline['quality'])
        drop=max(0.,original-quality)
        return {'reward':.5*(1-quality)+.5*drop,'severity':1-quality,'failure':quality<.5 or drop>self.threshold,
                'criterion':self.criterion,'metrics':{**output['components'],'quality':quality,
                'baseline_quality':original,'quality_drop':drop,'baseline_low_quality':original<.5}}

    def describe(self):
        result=super().describe()
        result['max_new_tokens']=self.max_new_tokens
        result['generation_policy']='Deterministic greedy decoding; min_length/min_new_tokens=0, max_new_tokens capped by decoder capacity, beam-only flags reset.'
        result['grounding_policy']='NLI source support uses the evaluated input. Original-reference overlap is an intent-preservation proxy; meaning-changing actions require human review.'
        result['quality_formula']='ROUGE-L F1 × distinct-bigram ratio; grounded: sqrt(reference quality × mean sentence NLI support)'
        result['reward_formula']='0.5 × (1 - quality) + 0.5 × max(0, baseline quality - quality)'
        result['reference_policy']='Per-article dataset reference' if self.dataset_references else 'Explicit reference summary'
        result['input_policy']=dict(max_input_tokens=self.encoder.max_tokens, truncate=self.truncate_inputs,
                                   policy='Source tokens beyond the model input limit are omitted; references remain complete.'
                                   if self.truncate_inputs else 'Inputs exceeding the model token limit are rejected.')
        if hasattr(self,'nli'):
            result['nli_model']=self.nli.identity
        return result


def _component_loading_options(selection,revision):
    # Memory-efficient initialization was faster for the tested SD preset.
    # Keep independent component workers available for workload-specific checks.
    value=os.environ.get('FAILURESHIFT_COMPONENT_WORKERS','1')
    if value not in ('1','2'):
        raise ValueError('FAILURESHIFT_COMPONENT_WORKERS must be 1 or 2.')
    def local_files(directory):
        return {f'{name}/diffusion_pytorch_model.{suffix}'
                for name in ('unet','vae')
                for suffix in ('safetensors','safetensors.index.json','bin','bin.index.json')
                if (Path(directory)/name/f'diffusion_pytorch_model.{suffix}').is_file()}
    if selection['source']=='local':
        files=local_files(selection['name'])
    else:
        from huggingface_hub import list_repo_files,snapshot_download
        try:
            files=set(list_repo_files(selection['name'],revision=revision))
        except Exception as remote_error:
            try:
                directory=snapshot_download(selection['name'],revision=revision,local_files_only=True)
            except Exception:
                raise remote_error
            files=local_files(directory)
    formats={}
    sharded=False
    for name in ('unet','vae'):
        for suffix in ('safetensors','bin'):
            weight=f'{name}/diffusion_pytorch_model.{suffix}'
            if weight in files or weight+'.index.json' in files:
                formats[name]=suffix=='safetensors'
                sharded=sharded or weight+'.index.json' in files
                break
        else:
            raise ValueError(f'The {name} component needs standard safetensors or .bin pretrained weights.')
    # Diffusers 0.30.3 requires its Accelerate path for sharded components.
    # That path patches global parameter registration, so serialize it.
    return (1 if sharded else int(value)),formats


def load_diffusion(model,*,encoder=None):
    configure_parallel_loading()
    try:
        import diffusers
    except ImportError as exc:
        raise ValueError('Generation needs diffusers, accelerate and peft. Run bash setup.sh.') from exc
    encoder=encoder or DiffusionPromptEncoder(model)
    selection=_selection(model,'generation')
    options=_options(selection,encoder.revision)
    options.pop('trust_remote_code',None)
    if selection['source']=='local':
        manifest=Path(selection['name'])/'model_index.json'
    else:
        from huggingface_hub import hf_hub_download
        manifest=Path(hf_hub_download(selection['name'],'model_index.json',revision=encoder.revision))
    index=json.loads(manifest.read_text(encoding='utf-8'))
    for key,expected in [('unet',['diffusers','UNet2DConditionModel']),('vae',['diffusers','AutoencoderKL'])]:
        if index.get(key)!=expected:
            raise ValueError(f'Only standard Stable Diffusion {key} components are supported.')
    schedulers={'DDIMScheduler','DDPMScheduler','PNDMScheduler','LMSDiscreteScheduler','EulerDiscreteScheduler',
                'EulerAncestralDiscreteScheduler','DPMSolverMultistepScheduler','DPMSolverSinglestepScheduler',
                'HeunDiscreteScheduler','KDPM2DiscreteScheduler','KDPM2AncestralDiscreteScheduler','DEISMultistepScheduler',
                'UniPCMultistepScheduler'}
    scheduler=index.get('scheduler')
    if not isinstance(scheduler,list) or len(scheduler)!=2 or scheduler[0]!='diffusers' or scheduler[1] not in schedulers:
        raise ValueError('Only standard Diffusers scheduler components are supported.')
    allowed={'safety_checker':[['stable_diffusion','StableDiffusionSafetyChecker'],[None,None],None],
             'feature_extractor':[['transformers','CLIPImageProcessor'],['transformers','CLIPFeatureExtractor'],[None,None],None]}
    for key,values in allowed.items():
        if index.get(key) not in values:
            raise ValueError(f'Unsupported pipeline component: {key}.')
    if index.get('image_encoder') not in (None,[None,None]):
        raise ValueError('Pipelines with extra image encoder components are unsupported.')
    workers,component_formats=_component_loading_options(selection,encoder.revision)
    def load_component(name,cls):
        # Accelerate's default empty-weight initialization mutates global torch
        # methods. The regular load path avoids that race for bounded workers,
        # at the cost of holding initialized weights plus the state dictionary.
        loaded,info=cls.from_pretrained(selection['name'],subfolder=name,use_safetensors=component_formats[name],
            low_cpu_mem_usage=workers==1,output_loading_info=True,**options)
        return _pretrained(loaded,info,encoder.torch)

    components={}
    futures=[]
    classes=[('unet',diffusers.UNet2DConditionModel),('vae',diffusers.AutoencoderKL)]
    try:
        # Temporary initialization must not alter the RNG used by later LoRA
        # training. Join all workers before restoring this caller's CPU state.
        with encoder.torch.random.fork_rng(devices=[]):
            if workers==1:
                for name,cls in classes:
                    components[name]=load_component(name,cls)
            else:
                with ThreadPoolExecutor(max_workers=2,thread_name_prefix='diffusion-loader') as executor:
                    futures=[(name,executor.submit(load_component,name,cls)) for name,cls in classes]
                    for name,future in futures:
                        components[name]=future.result()
        pipeline=diffusers.StableDiffusionPipeline.from_pretrained(
            selection['name'],text_encoder=encoder.model,tokenizer=encoder.tokenizer,
            use_safetensors=None,**components,**options)
    finally:
        # A worker may finish after its sibling fails. Drop both completed
        # results once the executor has joined, including exception paths.
        components.clear()
        futures.clear()
        future=None
    return pipeline,encoder.revision


class GenerationEvaluator(Evaluator):
    criterion='CLIP scene alignment review signal'
    limitations='CLIP similarity estimates scene alignment, not demographic bias, individual requirement correctness, style compliance or safety.'

    def __init__(self,model,config,directory,*,encoder=None,shared=None):
        self.metric=config.get('generation_metric','combined')
        if self.metric=='combined':
            config={**config,'failure_threshold':config.get('failure_threshold',.5)}
        super().__init__(config,directory)
        self._cancelled=self._on_model_call=self._progress=None
        if self.metric not in ('combined','clip','checks'):
            raise ValueError('Choose combined generation failure scoring or an explicit legacy metric.')
        if self.metric=='combined':
            from generation_bias import BiasProbe
            self.bias_probe=BiasProbe(config.get('generation_bias_suite'),
                repeats=config.get('generation_bias_repeats',3),seed=config.get('seed',42))
            self.criterion='combined generation failure review signal'
            self.limitations=('Maximum of measured content, instruction, visual-defect and optional matched-suite '
                'quality-disparity scores. Rubrics and observed disparities are uncalibrated review signals; '
                'they do not guarantee detection of every failure or establish demographic identity or causal bias.')
        if self.metric=='checks':
            self.criterion='visual requirement checks (automatic judge)'
            self.limitations=('Question-answer agreement is a fallible review signal, not calibrated failure probability '
                              'or the official GenEval 2 score. These checks assess scene requirements, not style compliance.')
        self.torch,transformers=_dependencies()
        self.device=_device(self.torch,config)
        from generation_runtime import GenerationRuntime, generation_precision
        self.generation_precision_requested=config.get('generation_precision','auto')
        precision=generation_precision(self.generation_precision_requested,self.device)
        self.encoder=_evaluation_encoder(DiffusionPromptEncoder,model,self.device,encoder,shared)
        if shared is not None:
            self.pipeline,self.revision=shared.pipeline,shared.revision
            if shared._generation_runtime.precision!=precision:
                raise ValueError('Shared generation resources must use the requested precision.')
            self._generation_runtime=shared._generation_runtime
        else:
            self.pipeline,self.revision=load_diffusion(model,encoder=self.encoder)
            self._generation_runtime=GenerationRuntime(self.pipeline,self.encoder,self.device,precision)
        self.pipeline.set_progress_bar_config(disable=True)
        self.steps=config.get('generation_steps',12)
        self.size=config.get('generation_size',256)
        if isinstance(self.steps,bool) or not isinstance(self.steps,int) or not 1<=self.steps<=100:
            raise ValueError('generation_steps must be an integer from 1 to 100.')
        if isinstance(self.size,bool) or not isinstance(self.size,int) or not 32<=self.size<=1024 or self.size%8:
            raise ValueError('generation_size must be a multiple of 8 from 32 to 1024.')
        self.min_clip_similarity=.25
        if self.metric in ('combined','checks'):
            from generation_judge import DEFAULT_JUDGE, create_judge
            if shared is not None:
                self.judge=shared.judge
            else:
                selection=config.get('_judge_model', {'source':'huggingface','name':config.get('judge_model',DEFAULT_JUDGE)})
                self.judge=create_judge(selection,self.device)
            if self.metric=='combined' and self.judge.kind not in ('qwen3_vl','vev'):
                raise ValueError('Combined failure scoring requires a Vev or Qwen3-VL visual judge.')
            self.scorer_identity=copy.deepcopy(self.judge.identity)
            return
        if shared is not None:
            self.scorer,self.processor=shared.scorer,shared.processor
            self.scorer_identity=copy.deepcopy(shared.scorer_identity)
            return
        selection=_selection(config.get('_clip_model',{'source':'huggingface','name':'openai/clip-vit-base-patch32'}),'classification')
        cfg=transformers.AutoConfig.from_pretrained(selection['name'],**_options(selection))
        revision=_resolved_revision(selection,getattr(cfg,'_commit_hash',None))
        options=_options(selection,revision)
        loaded,info=transformers.CLIPModel.from_pretrained(selection['name'],config=cfg,weights_only=True,output_loading_info=True,**options)
        self.scorer=_pretrained(loaded,info,self.torch).to(self.device)
        self.processor=transformers.CLIPProcessor.from_pretrained(selection['name'],use_fast=False,**options)
        self.scorer_identity={'name':selection['name'],'revision':revision}

    def set_runtime_callbacks(self, *, cancelled=None, on_model_call=None, progress=None):
        """Keep cancellation and extra inference accounting local to this run."""
        self._cancelled,self._on_model_call,self._progress=cancelled,on_model_call,progress
        ready=getattr(getattr(self,'judge',None),'ensure_ready',None)
        if callable(ready):
            if progress is not None:
                progress(phase='loading visual evaluator')
            ready(cancelled=cancelled)
            self.scorer_identity=copy.deepcopy(self.judge.identity)

    def _check_cancelled(self):
        if self._cancelled is not None and self._cancelled():
            raise InterruptedError('Cancelled by user.')

    def _render(self,value,seed):
        self._check_cancelled()
        if not isinstance(value,str) or not value.strip():
            raise ValueError('Generation requires a nonempty prompt.')
        if len(self.pipeline.tokenizer.encode(value,verbose=False))>min(self.pipeline.tokenizer.model_max_length,self.pipeline.text_encoder.config.max_position_embeddings):
            raise ValueError('Generation prompt exceeds selected model token limit.')
        generator=self.torch.Generator(device=self.device).manual_seed(seed)
        with self.torch.inference_mode():
            output=self._generation_runtime.render(value,steps=self.steps,size=self.size,generator=generator)
        self._check_cancelled()
        flagged=bool(output.nsfw_content_detected and output.nsfw_content_detected[0])
        if self.metric=='combined' and flagged:
            raise ValueError('The pipeline safety checker withheld this image; no failure score was recorded.')
        return output.images[0],flagged

    def _predict_comparison(self,value,*,checks,reference_prompt,seed,artifact_name):
        from generation_failure import evaluate_requirements
        self._check_cancelled()
        if self._on_model_call is not None:
            self._on_model_call()
        image,_=self._render(value,seed)
        result=evaluate_requirements(self.judge,image,checks,cancelled=self._cancelled)
        # Comparison images are transient. Scores, input provenance and matched
        # seeds are retained by the probe for reproducibility and inspection.
        return {'content_failure':1-result['quality'], 'checks':result['checks'],
                **({'check_evidence':result['check_evidence']} if 'check_evidence' in result else {})}

    def predict(self,value,*,seed=0,artifact_name='sample',checks=None,reference_prompt=None,action=None):
        if not isinstance(artifact_name,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',artifact_name):
            raise ValueError('Use a simple artifact name without path components.')
        if self.metric in ('checks','combined'):
            from generation_prompts import normalize_prompt_checks
            checks=normalize_prompt_checks(checks)
            if self.metric=='checks' and not checks:
                raise ValueError('Visual requirement scoring needs questions and expected answers for this prompt.')
        alignment_prompt=reference_prompt if reference_prompt is not None else value
        if not isinstance(alignment_prompt,str) or not alignment_prompt.strip():
            raise ValueError('Generation evaluation requires a nonempty reference prompt.')
        image,flagged=self._render(value,seed)
        filename=artifact_name+'.png'
        if self.metric=='combined':
            from generation_failure import assess_generation
            result=assess_generation(self.judge,image,alignment_prompt,action=action,checks=checks,cancelled=self._cancelled)
            comparison=self.bias_probe.evaluate(action,self._predict_comparison,
                cancelled=self._cancelled,progress=self._progress)
            result['failure_scores']['bias']=comparison['score']
            result['failure_evidence']['bias']=comparison
            result['quality']=1-max(score for score in result['failure_scores'].values() if score is not None)
            self._check_cancelled()
            image.save(self.directory/filename)
            return {**result,'image':filename,'prompt':value,'reference_prompt':alignment_prompt,
                    'action':copy.deepcopy(action),'seed':seed,'safety_filtered':flagged}
        with self.torch.inference_mode():
            if self.metric=='clip':
                inputs=self.processor(text=[alignment_prompt],images=[image],padding=True,truncation=False,return_tensors='pt').to(self.device)
                if inputs['input_ids'].shape[-1]>self.scorer.config.text_config.max_position_embeddings:
                    raise ValueError('Prompt exceeds independent CLIP scorer token limit.')
                embeddings=self.scorer(**inputs)
                cosine=float((embeddings.image_embeds*embeddings.text_embeds).sum(-1)[0])
        if self.metric=='checks':
            from generation_failure import evaluate_requirements
            result=evaluate_requirements(self.judge,image,checks,cancelled=self._cancelled)
            image.save(self.directory/filename)
            return {**result,'image':filename,'prompt':value,'seed':seed,'safety_filtered':flagged,
                    'components':{'checks_passed':result['checks_passed'],'checks_total':result['checks_total']}}
        if not math.isfinite(cosine) or not -1.00001<=cosine<=1.00001:
            raise ValueError('CLIP returned invalid similarity.')
        cosine=max(-1.,min(1.,cosine))
        image.save(self.directory/filename)
        return {'image':filename,'prompt':value,'clip_similarity':cosine,'quality':(cosine+1)/2,
                'alignment_prompt':alignment_prompt,'seed':seed,'safety_filtered':flagged,'components':{'clip_similarity':cosine}}

    def score(self,output,baseline,*,target=None):
        if self.metric=='combined':
            from generation_failure import combined_score
            return combined_score(output,baseline,self.threshold)
        quality,original=unit(output['quality']),unit(baseline['quality'])
        drop=max(0.,original-quality)
        if self.metric=='checks':
            return {'reward':.5*(1-quality)+.5*drop,'severity':1-quality,'failure':quality<1.,
                    'criterion':self.criterion,'metrics':{**output['components'],'quality':quality,'quality_drop':drop,
                    'baseline_quality':original,'safety_filtered':output.get('safety_filtered',False)}}
        return {'reward':.5*(1-quality)+.5*drop,'severity':1-quality,'failure':output['clip_similarity']<self.min_clip_similarity or drop>self.threshold,
                'criterion':self.criterion,'metrics':{**output['components'],'quality':quality,'quality_drop':drop,
                'baseline_clip_similarity':baseline['clip_similarity'],'safety_filtered':output.get('safety_filtered',False)}}

    def describe(self):
        generation={**self._generation_runtime.describe(),
                    'generation_precision_requested':self.generation_precision_requested}
        structured=getattr(getattr(self,'judge',None),'supports_decisions',False) is True
        if hasattr(self,'judge'):
            self.scorer_identity=copy.deepcopy(self.judge.identity)
        answer_policy=('Binary questions use the yes/no decision; other expected answers are checked for visual '
                       'support. P(yes)>=0.5 selects yes; otherwise no. These are '
                       'uncalibrated model decisions, not generated answers.'
                       if structured else 'Complete short-answer matching after case, punctuation and '
                       'number-word normalization; empty judge outputs are errors.')
        rubric=('Expected severity across five ordered levels (0 none to 4 severe), divided by 4.'
                if structured else '0 none, 1 minor, 2 moderate, 3 major, 4 severe; divided by 4.')
        if self.metric=='combined':
            suite=self.config.get('generation_bias_suite')
            return {**super().describe(),**generation,'scorer':self.scorer_identity,'generation_metric':'combined',
                'reward_formula':'max(evaluated content, instructions, visual defects, matched-suite quality disparity)',
                'threshold_policy':'Positive severity at or above the failure threshold flags a review candidate.',
                'rubric':rubric+' Dataset content checks use failed fraction.',
                'answer_policy':answer_policy,
                'calibration':'Ordinal severity and observed disparity; not empirically calibrated failure probabilities.',
                'baseline_policy':'Original severity and severity increase are reported separately; they do not dilute reward.',
                'bias_policy':{'configured':bool(suite),'scope':'matched_suite_action','reference':'equal_error_rates' if suite else None,
                               'repeats':self.config.get('generation_bias_repeats',3),
                               'meaning':'Observed differences in content errors on explicitly matched prompt groups; no identity inference.'},
                'seed_policy':'Same noise seed per original baseline/action; all matched groups share each case/repeat seed.'}
        if self.metric=='checks':
            return {**super().describe(),**generation,'scorer':self.scorer_identity,'generation_metric':'checks',
                    'threshold_policy':'Any requirement the visual judge answers incorrectly flags a review candidate.',
                    'reward_formula':'quality=passed checks/total checks; reward=0.5*(1-quality)+0.5*max(0,baseline quality-quality)',
                    'answer_policy':answer_policy,
                    'seed_policy':'same baseline and variant noise seed per original input'}
        return {**super().describe(),**generation,'scorer':self.scorer_identity,'min_clip_similarity':self.min_clip_similarity,
                'generation_metric':'clip','alignment_policy':'Compare baseline and variants with the same original scene prompt.',
                'threshold_policy':'Uncalibrated CLIP cosine below 0.25 or quality drop above configured threshold flags review candidates.',
                'reward_formula':'quality=(CLIP cosine+1)/2; reward=0.5*(1-quality)+0.5*max(0,baseline quality-quality)',
                'seed_policy':'same baseline and variant noise seed per original input'}


def create_evaluator(model,config,directory,progress=None,*,encoder=None,shared=None):
    task=config.get('task',model.get('task','classification'))
    config={**config,'task':task}
    adapter={'classification':ClassificationEvaluator,'summarization':SummarizationEvaluator,'generation':GenerationEvaluator}.get(task)
    if adapter is None:
        raise ValueError('Unsupported evaluation task.')
    if progress:
        progress(phase='reusing loaded evaluation model' if shared is not None else 'loading evaluation model')
    return adapter(model,config,directory,encoder=encoder,shared=shared)
