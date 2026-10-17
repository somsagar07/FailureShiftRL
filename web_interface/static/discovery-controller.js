'use strict';

// Polling never changes the embedding layout or starts an action expansion.
class DiscoveryController {
  constructor({api, explorer, onCheckpoint, onChange=()=>{}}) {
    Object.assign(this,{api,explorer,onCheckpoint,onChange});
    this.el=id=>document.getElementById(id);
    this.learningRates={classification:'0.001',summarization:'0.00002',generation:'0.0001'};
    this.version=0;this.job=null;this.training=null;this.examples=[];this.selected=null;this.context=null;
    const bind=(id,event,fn)=>this.el(id).addEventListener(event,fn);
    bind('find-failure','click',()=>{this.el('discovery-panel').hidden=false;this.configure();});
    bind('discovery-start','click',()=>this.start());
    bind('discovery-cancel','click',()=>this.cancel('discoveries'));
    bind('finetune-cancel','click',()=>this.cancel('finetunes'));
    bind('discovery-retry','click',()=>this.poll(this.trainingActive?'finetunes':'discoveries',this.version));
    bind('discovery-pair','change',()=>{this.selected=this.el('discovery-pair').value;this.selectionTouched=true;this.inspectSelected(true);});
    bind('finetune-add','click',()=>this.addExample());
    bind('finetune-start','click',()=>this.startTraining());
    bind('finetune-use','click',()=>{if(this.attached&&this.training?.status==='completed'&&this.training.checkpoint)this.onCheckpoint(this.training.checkpoint);});
    this.configure();this.controls();
  }
  get active() {return this.pending||[this.job,this.training].some(job=>job&&['queued','running'].includes(job.status));}
  get trainingActive() {return this.training&&['queued','running'].includes(this.training.status);}
  get analysisReady() {return Boolean(this.context&&this.context===this.explorer.job?.id);}
  get attached() {return Boolean(this.analysisReady&&this.job?.embedding_id===this.context);}
  ready(job) {
    this.context=job.id;this.configure();
    if(!this.attached){this.hideResults();this.examples=[];this.selected=null;this.el('finetune-examples').replaceChildren();}
    this.controls();
  }
  hideResults() {
    for(const id of ['discovery-inspector','discovery-records','finetune-panel','severity-legend','finetune-checkpoint'])this.el(id).hidden=true;
    this.explorer.view.setDiscovery?.(null);
  }
  configure() {
    const task=this.explorer.task;
    if(this.rateTask!==task) {
      if(this.rateTask)this.learningRates[this.rateTask]=this.el('finetune-lr').value;
      this.el('finetune-lr').value=this.learningRates[task];this.rateTask=task;
    }
    this.el('discovery-threshold-field').hidden=task==='classification';
    this.el('discovery-labels').hidden=task!=='classification';
    this.el('discovery-summary').hidden=task!=='summarization';
    this.el('discovery-generation').hidden=task!=='generation';
    this.el('generation-options').hidden=task!=='generation';
    this.el('finetune-target-label').textContent=task==='classification'?'Desired class ID or exact model label':task==='summarization'?'Corrected target summary':'Absolute server path to desired image (caption uses the selected prompt)';
  }
  invalidate() {
    this.context=null;this.selected=null;this.inspectedIndex=null;this.selectionTouched=false;this.examples=[];
    this.el('discovery-inspector').hidden=true;this.el('discovery-records').hidden=true;
    this.el('finetune-panel').hidden=true;this.el('severity-legend').hidden=true;
    this.el('finetune-target').value='';this.el('finetune-examples').replaceChildren();
    this.el('finetune-checkpoint').hidden=true;
    this.explorer.view.setDiscovery?.(null);
    if(!this.active) {
      this.version++;clearTimeout(this.timer);this.job=null;this.training=null;
      this.el('discovery-panel').hidden=true;
      this.el('discovery-status').textContent='';this.el('discovery-metrics').textContent='';
      this.el('discovery-criterion').textContent='';this.el('discovery-artifacts').replaceChildren();
    } else this.el('discovery-status').textContent='Inputs changed. The previous job can still be cancelled below.';
    this.controls();
  }
  controls() {
    this.el('find-failure').disabled=!this.explorer.job||this.active;
    this.el('discovery-settings').disabled=this.active;
    this.el('discovery-start').disabled=!this.analysisReady||this.active;
    this.el('discovery-cancel').hidden=!this.job||!['queued','running'].includes(this.job.status);
    this.el('finetune-cancel').hidden=!this.trainingActive;
    const terminal=this.job&&!['queued','running'].includes(this.job.status);
    this.el('finetune-panel').hidden=!this.attached||!terminal||!this.job.records?.length;
    this.el('finetune-add').disabled=!this.attached||this.active||!this.job?.records?.some(record=>this.key(record)===this.selected);
    this.el('finetune-start').disabled=this.active||!this.examples.length||!this.attached;
    this.el('finetune-target').disabled=this.active;
    this.onChange();
  }
  message(message) {this.el('discovery-error').textContent=message;this.el('discovery-error').hidden=!message;}
  number(id,min,max) {
    const n=Number(this.el(id).value);
    if(!this.el(id).value.trim()||!Number.isFinite(n)||n<min||n>max)throw new Error(`Check ${id.replaceAll('-',' ')} (${min}–${max}).`);
    return n;
  }
  async start() {
    if(this.active||!this.analysisReady)return;
    const version=++this.version,embedding=this.context;
    try {
      const task=this.explorer.task;
      const config={embedding_id:embedding,episodes:this.number('discovery-episodes',1,10000),seed:this.number('discovery-seed',0,2147483647),device:this.el('discovery-device').value};
      if(!Number.isInteger(config.episodes)||!Number.isInteger(config.seed))throw new Error('Episodes and seed must be whole numbers.');
      if(task!=='classification')config.failure_threshold=this.number('discovery-threshold',0,1);
      if(task==='classification') {
        config.label_map=this.el('discovery-label-map').value.trim()?JSON.parse(this.el('discovery-label-map').value):{};
        if(!config.label_map||typeof config.label_map!=='object'||Array.isArray(config.label_map))throw new Error('Labels must be a JSON object keyed by input index.');
      }
      if(task==='summarization'){config.reference_summary=this.el('discovery-reference').value.trim();config.summary_metric=this.el('discovery-metric').value;if(!config.reference_summary)throw new Error('Enter a human reference summary before discovery; both summary metrics require it.');}
      if(task==='generation'){config.generation_steps=this.number('discovery-steps',1,100);config.generation_size=Number(this.el('discovery-size').value);}
      this.pending=true;this.message('');this.controls();
      this.el('discovery-status').textContent='Starting discovery…';
      const job=await this.api('/api/discoveries',config);
      if(version!==this.version)return;
      this.pending=false;this.job=job;this.examples=[];this.selected=null;this.inspectedIndex=null;this.selectionTouched=false;this.training=null;
      this.el('finetune-examples').replaceChildren();this.el('finetune-checkpoint').hidden=true;
      this.el('finetune-status').textContent='';this.receive('discoveries',job,version);
    } catch(error) {if(version!==this.version)return;this.pending=false;this.message(error.message);this.controls();}
  }
  receive(kind,job,version) {
    if(version!==this.version)return;
    this.pending=false;
    if(kind==='finetunes'){this.training=job;this.renderTraining();}else{this.job=job;this.render();}
    this.controls();this.el('discovery-retry').hidden=true;
    clearTimeout(this.timer);
    if(['queued','running'].includes(job.status))this.timer=setTimeout(()=>this.poll(kind,version),650);
  }
  async poll(kind,version) {
    const job=kind==='finetunes'?this.training:this.job;if(!job)return;
    try {const result=await this.api(`/api/${kind}/${job.id}`);if(version!==this.version)return;this.message('');this.receive(kind,result,version);}
    catch(error){if(version!==this.version)return;this.message(`Status unavailable: ${error.message}. Retry status or cancel the job.`);this.el('discovery-retry').hidden=false;this.controls();}
  }
  async cancel(kind) {
    const job=kind==='finetunes'?this.training:this.job;if(!job)return;
    const version=this.version;
    try {this.message('');const result=await this.api(`/api/${kind}/${job.id}/cancel`,{});if(version===this.version)this.receive(kind,result,version);}
    catch(error){if(version===this.version){this.message(error.message);this.el('discovery-retry').hidden=false;}}
  }
  format(n) {return typeof n==='number'&&Number.isFinite(n)?n.toFixed(3):'—';}
  key(record) {return `${record.input_index}:${record.action_id}`;}
  render() {
    const job=this.job;
    this.el('discovery-status').textContent=`${job.status} · ${job.phase||'Discovery'} · episode ${job.episode||0}/${job.episodes||0}${!this.attached?' · previous input/model':''}`;
    const totalPairs=this.attached&&(this.explorer.job?.points?.length||0)*(this.explorer.preview?.action_count||0);
    const coverage=totalPairs?` · coverage ${job.unique_evaluations||0}/${totalPairs} (${(100*(job.unique_evaluations||0)/totalPairs).toFixed(1)}%)`:'';
    this.el('discovery-metrics').textContent=`Loss ${this.format(job.loss)} · epsilon ${this.format(job.epsilon)} · reward ${this.format(job.reward)} · mean ${this.format(job.mean_reward)} · ${job.unique_evaluations||0} measured pairs · ${job.model_calls||0} selected-model calls${coverage}${job.current?` · current input ${job.current.input_index}, action ${job.current.action_id}`:''}`;
    this.el('discovery-criterion').textContent=job.criterion||'';
    this.el('discovery-progress').hidden=false;this.el('discovery-progress').max=job.episodes||1;this.el('discovery-progress').value=job.episode||0;
    if(job.error)this.message(job.error);
    const links=[];
    for(const [name,url] of Object.entries(job.artifacts||{})){if(typeof url!=='string'||!url.startsWith('/discoveries/'))continue;const a=document.createElement('a');a.textContent=`Download ${name}`;a.href=url;links.push(a);}
    this.el('discovery-artifacts').replaceChildren(...links);
    if(!this.attached){this.hideResults();return;}
    this.el('severity-legend').hidden=false;this.explorer.view.setDiscovery(job);
    const records=job.records||[],selector=this.el('discovery-pair');
    selector.replaceChildren(...records.map(record=>{const option=document.createElement('option');option.value=this.key(record);option.textContent=`Input ${record.input_index} · action ${record.action_id} · severity ${this.format(record.severity)} (${record.visits} visits)`;return option;}));
    if(!this.selected&&!this.selectionTouched)this.selected=records.length?this.key(records[0]):null;
    selector.value=this.selected||'';this.el('discovery-records').hidden=!records.length;
    this.inspectSelected(false);
  }
  inspectPair(index,action) {
    if(!this.attached||!this.job)return;
    this.selectionTouched=true;this.inspectedIndex=index;this.selected=action===null?null:`${index}:${action}`;
    this.el('discovery-pair').value=this.selected||'';
    this.inspectSelected(false,index);this.controls();
  }
  paragraph(text) {const el=document.createElement('p');el.textContent=text;return el;}
  percent(value) {return typeof value==='number'&&Number.isFinite(value)?`${(100*value).toFixed(1)}%`:'—';}
  output(id,output) {
    const children=[];
    if(!output){this.el(id).replaceChildren(this.paragraph('Not evaluated.'));return;}
    if(output.image&&typeof output.image==='string'&&output.image.startsWith('/discoveries/')) {
      const img=document.createElement('img');img.src=output.image;img.alt='Generated model output';children.push(img);
    }
    if(output.summary!==undefined) {
      for(const paragraph of String(output.summary).split(/\n\s*\n/))children.push(this.paragraph(paragraph||'The model returned an empty summary.'));
    } else if(output.label!==undefined||output.class_id!==undefined) {
      children.push(this.paragraph(`Prediction: ${output.label??'Unnamed class'} (class ${output.class_id??'—'})`));
      children.push(this.paragraph(`Confidence: ${this.percent(output.confidence)}`));
    }
    if(output.clip_similarity!==undefined)children.push(this.paragraph(`CLIP prompt alignment: ${this.format(output.clip_similarity)}`));
    if(output.prompt)children.push(this.paragraph(`Prompt: ${output.prompt}`));
    if(output.seed!==undefined)children.push(this.paragraph(`Noise seed: ${output.seed}`));
    if(output.quality!==undefined)children.push(this.paragraph(`Quality review score: ${this.format(output.quality)}`));
    if(!children.length)children.push(this.paragraph('No model output available.'));
    this.el(id).replaceChildren(...children);
  }
  metricRows(record) {
    const labels={target_id:'Target class ID',target_label:'Target label',target_probability:'Target confidence',baseline_target_probability:'Original target confidence',target_confidence:'Target confidence',baseline_target_confidence:'Original target confidence',error:'Prediction differs from target',new_error:'New error after action',safety_filtered:'Image filtered by safety checker',confidence_drop:'Target confidence drop',baseline_error:'Original already incorrect',prediction_changed:'Prediction changed',labelled:'Human label supplied',rouge_l:'Reference overlap (ROUGE-L F1)',nonrepetition:'Distinct bigram ratio',source_support:'Source support (NLI)',quality:'Quality review score',baseline_quality:'Original quality score',quality_drop:'Quality drop from original',baseline_low_quality:'Original has low quality',clip_similarity:'CLIP prompt alignment',baseline_clip_similarity:'Original CLIP prompt alignment'};
    const rows=[['Criterion',record.criterion||this.job?.criterion||'—'],['Severity',this.percent(record.severity)],['Search reward',this.format(record.reward)],['Observed failure rate',this.percent(record.failure_rate)]];
    for(const [key,value] of Object.entries(record.metrics||{})) {
      const label=labels[key]||key.replaceAll('_',' ').replace(/^./,letter=>letter.toUpperCase());
      const display=typeof value==='boolean'?(value?'Yes':'No'):typeof value==='number'?this.format(value):String(value??'—');
      rows.push([label,display]);
    }
    const list=document.createElement('dl');list.className='metric-rows';
    for(const [label,value] of rows){const name=document.createElement('dt'),detail=document.createElement('dd');name.textContent=label;detail.textContent=value;list.append(name,detail);}
    this.el('discovery-components').textContent='';this.el('discovery-components').replaceChildren(list);
  }
  inspectSelected(focus=false,baselineIndex=null) {
    if(!this.attached){this.hideResults();return;}
    const record=this.job?.records?.find(record=>this.key(record)===this.selected);
    const index=record?.input_index??baselineIndex??(this.selected?Number(this.selected.split(':')[0]):this.inspectedIndex);
    const baseline=this.job?.baseline?.find(item=>item.input_index===index);
    this.el('discovery-inspector').hidden=index===null||index===undefined;
    if(index===null||index===undefined)return;
    this.output('discovery-baseline',baseline?.output);this.output('discovery-output',record?.output);
    if(record)this.metricRows(record);else {this.el('discovery-components').replaceChildren();this.el('discovery-components').textContent='This action has not been evaluated.';}
    if(focus&&record)this.explorer.focusMeasured(record.input_index,record.action_id);
    this.controls();
  }
  addExample() {
    if(this.active||!this.attached)return;
    const record=this.job?.records?.find(record=>this.key(record)===this.selected),text=this.el('finetune-target').value.trim();
    if(!record||!text){this.message('Select a measured action and enter its explicit desired target.');return;}
    const target=this.job.task==='classification'&&/^\d+$/.test(text)?Number(text):text;
    const example={input_index:record.input_index,action_id:record.action_id,target};
    const old=this.examples.findIndex(item=>this.key(item)===this.key(example));
    if(old>=0)this.examples[old]=example;else this.examples.push(example);
    this.message('');this.renderExamples();this.controls();
  }
  renderExamples() {
    this.el('finetune-examples').replaceChildren(...this.examples.map((example,index)=>{const li=document.createElement('li'),label=document.createElement('span'),remove=document.createElement('button');label.textContent=`Input ${example.input_index}, action ${example.action_id}: ${example.target}`;remove.type='button';remove.textContent='Remove';remove.disabled=this.active;remove.addEventListener('click',()=>{if(this.active)return;this.examples.splice(index,1);this.renderExamples();this.controls();});li.append(label,remove);return li;}));
  }
  async startTraining() {
    if(this.active||!this.examples.length||!this.attached)return;
    let version=this.version;
    try {
      const config={discovery_id:this.job.id,examples:this.examples.map(example=>({...example})),epochs:this.number('finetune-epochs',1,100),learning_rate:this.number('finetune-lr',1e-8,.1),batch_size:1,seed:Number(this.el('discovery-seed').value)};
      if(!Number.isInteger(config.epochs))throw new Error('Epochs must be a whole number.');
      // Training owns a new polling generation. An in-flight discovery response
      // cannot replace its timer or resurrect the terminal discovery.
      version=++this.version;clearTimeout(this.timer);
      this.pending=true;this.controls();this.message('');this.el('finetune-status').textContent='Starting fine-tuning…';
      const job=await this.api('/api/finetunes',config);if(version!==this.version)return;
      this.pending=false;this.receive('finetunes',job,version);this.renderExamples();
    } catch(error){if(version!==this.version)return;this.pending=false;this.message(error.message);this.controls();}
  }
  renderTraining() {
    const job=this.training;
    this.el('finetune-status').textContent=`${job.status} · ${job.phase||'Training'} · ${job.completed||0}/${job.total||0} steps · loss ${this.format(job.loss)}`;
    if(job.error)this.message(job.error);
    this.el('finetune-checkpoint').hidden=job.status!=='completed'||!job.checkpoint||!this.attached;
    this.el('finetune-path').value=job.checkpoint||'';
  }
}
