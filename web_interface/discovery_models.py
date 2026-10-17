"""Real model inference and domain-specific, baseline-aware review signals."""
import gc
import copy
import json
import math
import re
from pathlib import Path

from text_encoders import (_dependencies, _options, _pretrained, _resolved_revision,
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
    value=config.get('device','cpu')
    if value not in ('cpu','auto'):
        raise ValueError('Device must be cpu or auto.')
    return 'cuda' if value=='auto' and torch.cuda.is_available() else 'cpu'


class Evaluator:
    def __init__(self,config,directory):
        self.config=dict(config)
        self.directory=Path(directory).resolve()
        self.directory.mkdir(parents=True,exist_ok=True)
        self.threshold=unit(config.get('failure_threshold',.15),'failure_threshold')

    def close(self):
        for name in ('model','pipeline','encoder','scorer','nli','processor','tokenizer'):
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
        return resolve_label(self.model.config,target)

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
                'label_policy':'Explicit model IDs or exact unique model labels; absent labels use original predictions as consistency candidates.'}

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
        error=output['class_id']!=target_id
        baseline_error=baseline['class_id']!=target_id
        drop=max(0.,(original[target_id]-current[target_id])/max(original[target_id],1e-8))
        # Actual labelled mistakes retain search signal even when present before perturbation.
        wrong_confidence=max(probability for index,probability in enumerate(current) if index!=target_id) if len(current)>1 else 0.
        reward=.5*float(error)+.25*float(error)*wrong_confidence+.25*drop
        severity=1. if error else drop
        return {'reward':reward,'severity':severity,'failure':bool(error),
                'criterion':'labelled classification error' if labelled else 'original-prediction consistency candidate',
                'metrics':{'target_id':target_id,'target_label':self.model.config.id2label[target_id],
                'target_probability':current[target_id],'baseline_target_probability':original[target_id],
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
        if not isinstance(self.reference,str) or not self.reference.strip():
            raise ValueError('Summary evaluation requires an explicit reference summary.')
        if self.metric=='grounded' and (not isinstance(self.source,str) or not self.source.strip()):
            raise ValueError('Grounded summary scoring needs original source text.')
        self.encoder=shared.encoder if shared is not None else encoder or SummarizationEncoder(model)
        self.model,self.tokenizer,self.torch=self.encoder.model,self.encoder.tokenizer,self.encoder.torch
        self.revision=self.encoder.revision
        self.device=_device(self.torch,config)
        self.model.to(self.device)
        self.max_new_tokens=config.get('max_new_tokens',96)
        if isinstance(self.max_new_tokens,bool) or not isinstance(self.max_new_tokens,int) or not 1<=self.max_new_tokens<=512:
            raise ValueError('max_new_tokens must be an integer from 1 to 512.')
        self.criterion='reference ROUGE-L review signal' if self.metric=='reference' else 'reference ROUGE-L and grounded NLI review signal'
        if self.metric=='grounded':
            self.nli=shared.nli if shared is not None else GroundedScorer(config.get('_nli_model',{'source':'huggingface','name':'cross-encoder/nli-MiniLM2-L6-H768'}),self.device)

    def measure_summary(self,text,source=None):
        overlap,repetition=rouge_l(self.reference,text),nonrepetition(text)
        metrics={'rouge_l':overlap,'nonrepetition':repetition}
        quality=overlap*repetition
        if self.metric=='grounded':
            metrics['source_support']=self.nli.score(self.source if source is None else source,text)
            quality=math.sqrt(quality*metrics['source_support'])
        return {'summary':text,'quality':quality,'components':metrics}

    def predict(self,value,*,seed=0,artifact_name='sample'):
        inputs={key:value.to(self.device) for key,value in self.encoder._prepare([value],self.encoder.prefix).items()}
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
        return self.measure_summary(self.tokenizer.decode(generated[0],skip_special_tokens=True),source=value)

    def score(self,output,baseline,*,target=None):
        quality,original=unit(output['quality']),unit(baseline['quality'])
        drop=max(0.,original-quality)
        return {'reward':.5*(1-quality)+.5*drop,'severity':1-quality,'failure':quality<.5 or drop>self.threshold,
                'criterion':self.criterion,'metrics':{**output['components'],'quality':quality,
                'baseline_quality':original,'quality_drop':drop,'baseline_low_quality':original<.5}}

    def describe(self):
        result=super().describe()
        result['generation_policy']='Deterministic greedy decoding; min_length/min_new_tokens=0, max_new_tokens configured, beam-only flags reset.'
        result['grounding_policy']='NLI source support uses the evaluated input. Original-reference overlap is an intent-preservation proxy; meaning-changing actions require human review.'
        result['quality_formula']='ROUGE-L F1 × distinct-bigram ratio; grounded: sqrt(reference quality × mean sentence NLI support)'
        result['reward_formula']='0.5 × (1 - quality) + 0.5 × max(0, baseline quality - quality)'
        if hasattr(self,'nli'):
            result['nli_model']=self.nli.identity
        return result


def load_diffusion(model,*,encoder=None):
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
    components={}
    for name,cls in [('unet',diffusers.UNet2DConditionModel),('vae',diffusers.AutoencoderKL)]:
        loaded,info=cls.from_pretrained(selection['name'],subfolder=name,use_safetensors=True,
                                        output_loading_info=True,**options)
        components[name]=_pretrained(loaded,info,encoder.torch)
    pipeline=diffusers.StableDiffusionPipeline.from_pretrained(
        selection['name'],text_encoder=encoder.model,tokenizer=encoder.tokenizer,
        use_safetensors=True,**components,**options)
    return pipeline,encoder.revision


class GenerationEvaluator(Evaluator):
    criterion='CLIP prompt alignment review signal'
    limitations='CLIP similarity estimates prompt alignment, not demographic bias, image fidelity or safety.'

    def __init__(self,model,config,directory,*,encoder=None,shared=None):
        super().__init__(config,directory)
        self.torch,transformers=_dependencies()
        self.device=_device(self.torch,config)
        self.encoder=shared.encoder if shared is not None else encoder or DiffusionPromptEncoder(model)
        if shared is not None:
            self.pipeline,self.revision=shared.pipeline,shared.revision
        else:
            self.pipeline,self.revision=load_diffusion(model,encoder=self.encoder)
        self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)
        self.steps=config.get('generation_steps',12)
        self.size=config.get('generation_size',256)
        if isinstance(self.steps,bool) or not isinstance(self.steps,int) or not 1<=self.steps<=100:
            raise ValueError('generation_steps must be an integer from 1 to 100.')
        if isinstance(self.size,bool) or not isinstance(self.size,int) or not 32<=self.size<=1024 or self.size%8:
            raise ValueError('generation_size must be a multiple of 8 from 32 to 1024.')
        self.min_clip_similarity=.25
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

    def predict(self,value,*,seed=0,artifact_name='sample'):
        if not isinstance(value,str) or not value.strip():
            raise ValueError('Generation requires a nonempty prompt.')
        if len(self.pipeline.tokenizer.encode(value,verbose=False))>min(self.pipeline.tokenizer.model_max_length,self.pipeline.text_encoder.config.max_position_embeddings):
            raise ValueError('Generation prompt exceeds selected model token limit.')
        if not isinstance(artifact_name,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',artifact_name):
            raise ValueError('Use a simple artifact name without path components.')
        generator=self.torch.Generator(device=self.device).manual_seed(seed)
        with self.torch.inference_mode():
            output=self.pipeline(value,num_inference_steps=self.steps,height=self.size,width=self.size,generator=generator)
            image=output.images[0]
            inputs=self.processor(text=[value],images=[image],padding=True,truncation=False,return_tensors='pt').to(self.device)
            if inputs['input_ids'].shape[-1]>self.scorer.config.text_config.max_position_embeddings:
                raise ValueError('Prompt exceeds independent CLIP scorer token limit.')
            embeddings=self.scorer(**inputs)
            cosine=float((embeddings.image_embeds*embeddings.text_embeds).sum(-1)[0])
        if not math.isfinite(cosine) or not -1.00001<=cosine<=1.00001:
            raise ValueError('CLIP returned invalid similarity.')
        cosine=max(-1.,min(1.,cosine))
        filename=artifact_name+'.png'
        image.save(self.directory/filename)
        flagged=bool(output.nsfw_content_detected and output.nsfw_content_detected[0])
        return {'image':filename,'prompt':value,'clip_similarity':cosine,'quality':(cosine+1)/2,
                'seed':seed,'safety_filtered':flagged,'components':{'clip_similarity':cosine}}

    def score(self,output,baseline,*,target=None):
        quality,original=unit(output['quality']),unit(baseline['quality'])
        drop=max(0.,original-quality)
        return {'reward':.5*(1-quality)+.5*drop,'severity':1-quality,'failure':output['clip_similarity']<self.min_clip_similarity or drop>self.threshold,
                'criterion':self.criterion,'metrics':{**output['components'],'quality':quality,'quality_drop':drop,
                'baseline_clip_similarity':baseline['clip_similarity'],'safety_filtered':output.get('safety_filtered',False)}}

    def describe(self):
        return {**super().describe(),'scorer':self.scorer_identity,'min_clip_similarity':self.min_clip_similarity,
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
