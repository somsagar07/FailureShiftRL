'use strict';

// Polling never changes the embedding layout or starts an action expansion.
class DiscoveryController {
  constructor({api, explorer, onCheckpoint, onChange=()=>{}}) {
    Object.assign(this,{api,explorer,onCheckpoint,onChange});
    this.el=id=>document.getElementById(id);
    this.learningRates={classification:'0.001',summarization:'0.00002',generation:'0.0001'};
    this.failureThresholds={classification:'0.15',summarization:'0.15',generation:'0.5'};
    this.trainingOptions={};this.trainingDefaults={'finetune-epochs':'1','finetune-device':'auto','finetune-batch-size':'auto','finetune-seed':'42','finetune-weight-decay':'0.01','finetune-max-grad-norm':'1','finetune-cleanup':true};
    this.version=0;this.job=null;this.training=null;this.selected=null;this.context=null;this.inspectionOpen=false;
    this.pending=false;this.pendingKind=null;this.polling=null;this.cancelling=null;
    this.landscapeMode='predicted';
    this.explorer.onPredictions=()=>this.inspectSelected(false);
    this.labelRead=0;this.labelLoading=false;this.labelError='';this.labelSource='';this.labelScope=null;this.labelMap={};
    this.labelInfo={};this.labelPage=0;this.labelViewing=false;this.editingLabels=false;this.labelMissing=0;
    const bind=(id,event,fn)=>this.el(id).addEventListener(event,fn);
    bind('find-failure','click',()=>this.showDiscovery());
    bind('discovery-start','click',()=>this.start());
    bind('discovery-cancel','click',()=>this.cancel('discoveries'));
    bind('finetune-cancel','click',()=>this.cancel('finetunes'));
    bind('discovery-retry','click',()=>this.poll(this.trainingActive?'finetunes':'discoveries',this.version));
    bind('discovery-pair','change',()=>{this.selected=this.el('discovery-pair').value||null;this.inspectionOpen=Boolean(this.selected);this.inspectSelected(true);this.controls();});
    bind('landscape-mode','change',()=>{this.landscapeMode=this.el('landscape-mode').value;this.renderLegend();});
    this.el('discovery-judge-model').value='CountingSheep/vev-4b';
    this.el('discovery-generation-precision').value='auto';
    this.el('discovery-bias-repeats').value='3';
    bind('discovery-label-upload','click',()=>this.el('discovery-label-file').click());
    bind('discovery-label-file','change',()=>this.uploadLabels());
    bind('discovery-label-edit','click',()=>this.editLabels(!this.editingLabels));
    bind('discovery-label-view','click',()=>{const open=this.editingLabels||!this.labelViewing;this.editLabels(false);this.labelViewing=open;this.renderLabels();});
    bind('discovery-label-previous','click',()=>{this.labelPage=Math.max(0,this.labelPage-5);this.renderLabels();});
    bind('discovery-label-next','click',()=>{this.labelPage+=5;this.renderLabels();});
    bind('discovery-label-clear','click',()=>{if(!this.active)this.clearLabels();});
    bind('discovery-label-map','input',()=>{
      this.labelRead++;this.labelLoading=false;this.labelSource='Edited labels';
      this.updateLabels();this.controls();
    });
    this.adaptation=new AdaptationController({api,explorer,discovery:this});
    bind('finetune-start','click',()=>this.adaptation.start());
    bind('finetune-use','click',()=>{if(this.attached&&this.training?.status==='completed'&&this.training.checkpoint)this.onCheckpoint(this.training.checkpoint);});
    this.editLabels(false);this.updateLabels();this.configure();this.controls();
  }
  get active() {return this.pending||[this.job,this.training].some(job=>job&&['queued','running'].includes(job.status));}
  get trainingActive() {return this.pendingKind==='finetunes'||Boolean(this.training&&['queued','running'].includes(this.training.status));}
  get analysisReady() {return Boolean(this.context&&this.context===this.explorer.job?.id);}
  get attached() {return Boolean(this.analysisReady&&this.job?.embedding_id===this.context);}
  get summaryDataset() {return this.explorer.task==='summarization'&&Boolean(this.explorer.job?.references?.dataset_id||this.explorer.preview?.dataset||this.explorer.preview?.config.dataset_id);}
  get summaryReferences() {
    const info=this.explorer.job?.references,dataset=this.explorer.preview?.dataset;
    const total=info?.total??dataset?.count??this.explorer.job?.points?.length??0;
    const count=info?.count??dataset?.reference_count??0;
    return {total,count,missing:info?.missing??dataset?.missing_reference_count??total-count};
  }
  get generationRequirements() {
    const info=this.explorer.job?.requirements,dataset=this.explorer.preview?.dataset;
    const total=info?.total??dataset?.count??this.explorer.job?.points?.length??1;
    const count=info?.count??dataset?.checked_count??0;
    return {total,count,missing:info?.missing??dataset?.missing_checks_count??total-count};
  }
  showDiscovery() {
    this.adaptation.close();this.el('discovery-panel').hidden=false;
    this.configure();this.controls();
  }
  ready(job) {
    if(this.context!==job.id)this.landscapeMode='predicted';
    if(this.context!==job.id&&['cpu','cuda'].includes(job.device))this.el('discovery-device').value=job.device;
    this.context=job.id;this.configure();
    if(this.explorer.task==='summarization'&&!this.summaryDataset) {
      const text=job.points?.[0]?.text??this.explorer.preview?.config.text;
      if(typeof text==='string') {
        if(this.referenceText!==undefined&&this.referenceText!==text)this.el('discovery-reference').value='';
        this.referenceText=text;
      }
    }
    if(this.explorer.task==='classification') {
      const config=this.explorer.preview?.config;
      const scope=JSON.stringify([...(config?[config.dataset_id,config.model?.source,config.model?.name]:[job.id]),job.labels?.model_labels,job.labels?.groups]);
      this.labelInfo=job.labels||{};
      if(this.labelScope!==scope){this.labelViewing=false;this.labelPage=0;this.clearLabels();}
      this.labelScope=scope;this.updateLabels(this.labelError);
    }
    if(!this.attached){this.hideResults();this.selected=null;this.inspectionOpen=false;this.adaptation.reset();}
    this.controls();
  }
  hideResults() {
    for(const id of ['discovery-inspector','discovery-records','severity-legend','finetune-checkpoint','finetune-report','finetune-cleanup-status'])this.el(id).hidden=true;
    if(this.explorer.setDiscovery)this.explorer.setDiscovery(null);else this.explorer.view.setDiscovery?.(null);
  }
  configure() {
    const task=this.explorer.task;
    if(this.rateTask!==task) {
      if(this.rateTask) {
        this.learningRates[this.rateTask]=this.el('finetune-lr').value;
        this.failureThresholds[this.rateTask]=this.el('discovery-threshold').value;
        this.trainingOptions[this.rateTask]=Object.fromEntries(Object.keys(this.trainingDefaults).map(id=>[id,this.el(id)[id==='finetune-cleanup'?'checked':'value']]));
      }
      for(const [id,value] of Object.entries(this.trainingOptions[task]||this.trainingDefaults))this.el(id)[id==='finetune-cleanup'?'checked':'value']=value;
      this.el('finetune-lr').value=this.learningRates[task];this.el('discovery-threshold').value=this.failureThresholds[task];this.rateTask=task;
    }
    this.el('discovery-generation-metric').value='combined';
    this.el('discovery-threshold-field').hidden=task==='classification';
    this.el('discovery-threshold-label').textContent=task==='generation'?'Failure severity threshold':'Quality-drop threshold';
    this.el('discovery-labels').hidden=task!=='classification';
    this.el('discovery-summary').hidden=task!=='summarization';
    this.el('discovery-reference-field').hidden=this.summaryDataset;
    this.el('discovery-reference-status').hidden=!this.summaryDataset;
    if(this.summaryDataset) {
      const {total,count,missing}=this.summaryReferences;
      this.el('discovery-reference-status').textContent=`${count} / ${total} reference summaries loaded${missing?` · ${missing} missing. Choose a reference column or another dataset before discovery.`:''}`;
      this.el('discovery-reference-status').className=missing?'hint needs-review':'hint';
    } else this.el('discovery-reference-status').textContent='';
    this.el('discovery-generation').hidden=task!=='generation';
    this.el('discovery-generation-fields').hidden=task!=='generation';
    this.el('generation-options').hidden=task!=='generation';
    this.el('discovery-judge-field').hidden=task!=='generation';
    if(task==='generation') {
      const {total,count,missing}=this.generationRequirements;
      this.el('discovery-requirements-status').textContent=`${count} / ${total} prompts have requirements${missing?` · ${missing} use visual content review against the prompt.`:'. Dataset checks review content.'}`;
      this.el('discovery-requirements-status').className='hint';
      this.el('discovery-generation').textContent='One reward: the largest measured content, instruction, visual defect, or bias score (0–1). Style, lighting, and detail are reviewed when enabled. Scores flag candidates for inspection.';
    }
  }
  invalidate() {
    this.labelRead++;this.labelLoading=false;
    this.context=null;this.selected=null;this.inspectionOpen=false;
    this.el('discovery-inspector').hidden=true;this.el('discovery-records').hidden=true;
    this.el('severity-legend').hidden=true;
    this.el('finetune-checkpoint').hidden=true;this.el('finetune-report').hidden=true;
    this.el('finetune-cleanup-status').hidden=true;this.el('finetune-cleanup-status').textContent='';
    this.adaptation.reset();
    if(this.explorer.setDiscovery)this.explorer.setDiscovery(null);else this.explorer.view.setDiscovery?.(null);
    if(!this.active) {
      this.version++;clearTimeout(this.timer);this.job=null;this.training=null;
      this.el('discovery-panel').hidden=true;
      this.el('discovery-status').textContent='';this.el('discovery-metrics').textContent='';
      this.el('discovery-criterion').textContent='';this.el('discovery-artifacts').replaceChildren();
    } else this.el('discovery-status').textContent='Inputs changed. The previous job can still be cancelled below.';
    this.controls();
  }
  controls() {
    this.el('find-failure').disabled=!this.explorer.job&&!this.active;
    this.el('discovery-settings').disabled=this.active;
    this.el('discovery-start').disabled=!this.analysisReady||this.active||(this.summaryDataset&&this.summaryReferences.missing>0)||(this.explorer.task==='classification'&&(this.labelLoading||Boolean(this.labelError)||this.labelMissing>0));
    this.el('discovery-cancel').hidden=!this.job||!['queued','running'].includes(this.job.status);
    this.el('finetune-cancel').hidden=!this.training||!['queued','running'].includes(this.training.status);
    for(const id of ['discovery-cancel','finetune-cancel'])this.el(id).disabled=this.cancelling===this.version;
    this.el('discovery-retry').disabled=this.cancelling===this.version||this.polling?.version===this.version;
    this.adaptation.controls();
    this.onChange();
  }
  message(message) {this.el('discovery-error').textContent=message;this.el('discovery-error').hidden=!message;}
  number(id,min,max) {
    const n=Number(this.el(id).value);
    if(!this.el(id).value.trim()||!Number.isFinite(n)||n<min||n>max)throw new Error(`Check ${id.replaceAll('-',' ')} (${min}–${max}).`);
    return n;
  }
  biasSuite() {
    const text=this.el('discovery-bias-suite').value.trim();
    if(!text)return null;
    let suite;
    try{suite=JSON.parse(text);}catch{throw new Error('Bias comparison JSON is invalid. Use the example object format.');}
    if(suite!==null&&(typeof suite!=='object'||Array.isArray(suite)))throw new Error('Bias comparison must be a JSON object in the example format.');
    return suite;
  }
  editLabels(open) {
    this.editingLabels=open;this.labelViewing=open;
    this.el('discovery-label-editor').hidden=!open;
    this.el('discovery-label-edit').setAttribute('aria-expanded',String(open));
    this.renderLabels();
  }
  formatLabels(labels) {
    const formatted=JSON.stringify(labels,null,2);
    return formatted.length<=2*1024*1024?formatted:JSON.stringify(labels);
  }
  writeLabel(index,value) {
    if(this.active)return;
    this.labelRead++;this.labelLoading=false;this.labelSource='Edited labels';
    const labels={...(this.labelMap||{})},text=value.trim();
    if(!text)delete labels[index];
    else if(Object.values(this.labelInfo.model_labels||{}).some(name=>name.trim()===text))labels[index]=text;
    else labels[index]=/^#?(0|[1-9][0-9]*)$/.test(text)?Number(text.replace(/^#/,'')):text;
    this.el('discovery-label-map').value=this.formatLabels(labels);
    this.updateLabels('',true);this.controls();
  }
  resolveLabel(value) {
    const classes=this.labelInfo.model_labels||{};
    if(Number.isSafeInteger(value)&&value>=0&&Object.hasOwn(classes,String(value)))return value;
    const normalize=name=>name.trim().replace(/[\s_-]+/g,' ').toLowerCase();
    if(typeof value==='string'&&value.trim()) {
      const matches=Object.entries(classes).filter(([,name])=>normalize(name)===normalize(value));
      if(matches.length===1)return Number(matches[0][0]);
      if(matches.length>1)throw new Error('Ambiguous class name; enter a model class ID.');
      if(Object.hasOwn(this.labelInfo.groups||{},normalize(value)))return normalize(value);
    }
    throw new Error('Label does not match this model.');
  }
  parseLabels(text=this.el('discovery-label-map').value) {
    text=text.replace(/^\uFEFF/,'').trim();
    if(!text)return {};
    if(text.length>2*1024*1024)throw new Error('Use JSON under 2 MB.');
    let labels;
    try{labels=JSON.parse(text);}catch{throw new Error('Invalid JSON. Use an object such as {"0": "cat", "1": 281}.');}
    if(!labels||typeof labels!=='object'||Array.isArray(labels))throw new Error('Labels must be a JSON object keyed by input index.');
    const count=this.explorer.job?.points?.length;
    for(const [index,label] of Object.entries(labels)) {
      if(!/^(0|[1-9][0-9]*)$/.test(index)||!Number.isSafeInteger(Number(index))||(count!==undefined&&Number(index)>=count))throw new Error(`Input index ${index} is outside this analysis. Indices start at 0.`);
      if(!((typeof label==='number'&&Number.isSafeInteger(label)&&label>=0)||(typeof label==='string'&&label.trim()&&label.length<=512)))throw new Error(`Input ${index}: use a nonnegative class ID or an exact model label name.`);
    }
    return labels;
  }
  updateLabels(uploadError='',preserveInputs=false) {
    this.labelError=uploadError;
    try{this.labelMap=this.parseLabels();}
    catch(error){this.labelMap=null;this.labelError=error.message;}
    this.el('discovery-label-error').textContent=this.labelError;
    this.el('discovery-label-error').hidden=!this.labelError;
    this.labelResolved={};this.labelIssues={};
    const resolutions=new Map();
    for(const [index,value] of Object.entries(this.labelMap||{})) {
      if(!resolutions.has(value)) {
        try{resolutions.set(value,{target:this.resolveLabel(value)});}
        catch(error){resolutions.set(value,{error:error.message});}
      }
      const resolved=resolutions.get(value);
      if(resolved.error)this.labelIssues[index]=resolved.error;
      else this.labelResolved[index]=resolved.target;
    }
    this.labelMissing=Math.max(0,(this.explorer.job?.points?.length||0)-Object.keys(this.labelResolved).length);
    this.renderLabels(preserveInputs);
  }
  clearLabels() {
    this.labelRead++;this.labelLoading=false;this.labelSource='';
    const labels=this.labelInfo.label_map||{};
    this.el('discovery-label-map').value=Object.keys(labels).length?this.formatLabels(labels):'';this.el('discovery-label-file').value='';
    this.editLabels(false);this.updateLabels();this.controls();
  }
  async uploadLabels() {
    if(this.active)return;
    const file=this.el('discovery-label-file').files?.[0];if(!file)return;
    this.el('discovery-label-file').value='';
    const read=++this.labelRead;
    this.labelLoading=true;this.updateLabels();this.controls();
    try {
      if(file.size>2*1024*1024)throw new Error('Use a JSON file under 2 MB.');
      const text=await file.text();
      if(read!==this.labelRead)return;
      const labels={...(this.labelMap||this.labelInfo.label_map||{}),...this.parseLabels(text)};
      this.el('discovery-label-map').value=this.formatLabels(labels);
      this.labelSource=file.name;this.editLabels(false);this.updateLabels();
    } catch(error) {
      if(read===this.labelRead)this.updateLabels(`File not applied. ${error.message}`);
    } finally {
      if(read===this.labelRead){this.labelLoading=false;this.renderLabels();this.controls();}
    }
  }
  renderLabels(preserveInputs=false) {
    const points=this.explorer.job?.points||[],total=points.length,classes=this.labelInfo.model_labels||{};
    const ready=Object.keys(this.labelResolved||{}).length;
    const mismatches=points.filter(point=>{
      const index=String(point.index),original=this.labelInfo.values?.[index],provided=this.labelMap?.[index];
      return this.labelResolved?.[index]===undefined&&original!==undefined&&this.labelInfo.label_map?.[index]===undefined&&(provided===undefined||provided===original);
    });
    const mismatchIndices=new Set(mismatches.map(point=>String(point.index)));
    const missing=Math.max(0,total-ready-mismatches.length);
    this.el('discovery-label-summary').textContent=this.labelLoading?'Reading JSON…':!this.labelMap?'Check JSON formatting':
      `${ready} / ${total} labels ready${mismatches.length?` · ${mismatches.length} model mismatch${mismatches.length===1?'':'es'}`:''}${missing?` · ${missing} need review`:''}`;
    this.el('discovery-label-summary').className=`label-summary${missing||mismatches.length?' needs-review':''}`;
    const loaded=this.labelInfo.loaded??Object.keys(this.labelInfo.values||{}).length;
    const names=[...new Set(mismatches.map(point=>this.labelInfo.values[String(point.index)]))].sort();
    const mismatchMessage=mismatches.length?`Choose a model with matching classes. Unmatched: ${names.slice(0,3).join(', ')}${names.length>3?` (+${names.length-3} more)`:''}.`:'';
    const datasetMessage=total?`${loaded} / ${total} dataset labels loaded${this.labelInfo.column?` · Column: ${this.labelInfo.column}`:''}`:'No dataset labels available.';
    this.el('discovery-label-source').textContent=[datasetMessage,missing?this.labelInfo.message:'',mismatchMessage,this.labelSource].filter(Boolean).join(' · ');
    this.el('discovery-label-clear').disabled=this.active||(!this.labelSource&&!this.labelError&&!this.labelLoading);
    this.el('discovery-label-preview').hidden=!this.labelViewing||!total;
    this.el('discovery-label-view').setAttribute('aria-expanded',String(this.labelViewing&&!this.editingLabels));
    this.labelPage=Math.min(this.labelPage,Math.max(0,Math.floor((total-1)/5)*5));
    const samples=points.slice(this.labelPage,this.labelPage+5);
    const body=this.el('discovery-label-preview-rows');
    const reuse=preserveInputs&&this.editingLabels&&body.children.length===samples.length;
    const rows=samples.map((point,offset)=>{
      const index=String(point.index),row=document.createElement('tr');
      const mapped=this.labelResolved?.[index],provided=this.labelMap?.[index],original=this.labelInfo.values?.[index],raw=this.labelInfo.original_values?.[index];
      const known=mapped!==undefined;
      const grouped=known&&typeof mapped==='string',groupSize=grouped?this.labelInfo.groups[mapped].length:0;
      const fromDataset=(original!==undefined||raw!==undefined)&&(provided===undefined||provided===original||(known&&mapped===this.labelInfo.label_map?.[index]));
      const namesMissing=!known&&provided===undefined&&typeof raw==='number'&&this.labelInfo.diagnostic==='unnamed_numeric_labels';
      const values=[`${index}. ${point.name}`,fromDataset?'Dataset':provided!==undefined?'Manual':'—',
        known?(grouped?mapped:`${classes[mapped]} (class ${mapped})`):provided??original??raw??'—',
        known?`Ready${grouped?` · ${groupSize} model classes`:''}`:mismatchIndices.has(index)?'Model class mismatch':namesMissing?'Class names missing':`Needs review${this.labelIssues?.[index]?` · ${this.labelIssues[index]}`:' · Missing label'}`];
      if(reuse) {
        const current=body.children[offset];current.children[1].textContent=values[1];current.children[3].textContent=values[3];
        current.children[3].className=known?'':'needs-review';return current;
      }
      for(const [column,value] of values.entries()) {
        const cell=document.createElement('td');
        if(column===2&&this.editingLabels) {
          const input=document.createElement('input');input.type='text';input.value=typeof provided==='number'?`#${provided}`:provided??(mismatchIndices.has(index)?original:'');input.maxLength=512;
          input.placeholder='Class name or #ID';input.setAttribute('list','discovery-label-classes');input.setAttribute('aria-label',`Label for input ${index}: ${point.name}`);
          input.disabled=this.active;
          for(const event of ['input','change'])input.addEventListener(event,()=>this.writeLabel(index,input.value));cell.append(input);
        } else cell.textContent=value;
        if(column===0&&point.image) {
          const image=document.createElement('img');image.src=point.image;image.alt=point.name;image.loading='lazy';image.className='label-thumbnail';cell.append(image);
        }
        if(column===3&&!known)cell.className='needs-review';
        row.append(cell);
      }
      return row;
    });
    if(!reuse)body.replaceChildren(...rows);
    this.el('discovery-label-preview-count').textContent=total?`${this.labelPage+1}–${this.labelPage+samples.length} of ${total} inputs`:'';
    this.el('discovery-label-previous').disabled=this.labelPage===0;
    this.el('discovery-label-next').disabled=this.labelPage+5>=total;
    if(this.labelClasses!==classes||this.labelGroups!==this.labelInfo.groups) {
      this.labelClasses=classes;this.labelGroups=this.labelInfo.groups;
      const groups=Object.entries(this.labelGroups||{}).map(([name,ids])=>{
        const option=document.createElement('option');option.value=name;option.textContent=`${ids.length} model classes`;return option;
      });
      this.el('discovery-label-classes').replaceChildren(...groups,...Object.entries(classes).map(([id,name])=>{
        const option=document.createElement('option');option.value=name;option.textContent=`Class ${id}`;return option;
      }));
    }
  }
  async start() {
    if(this.active||!this.analysisReady)return;
    if(this.summaryDataset&&this.summaryReferences.missing>0){this.configure();this.controls();return;}
    if(this.explorer.task==='classification') {
      this.updateLabels(this.labelError);this.controls();
      if(this.labelLoading||this.labelError||this.labelMissing>0)return;
    }
    const version=++this.version,embedding=this.context;
    try {
      const task=this.explorer.task;
      const config={embedding_id:embedding,episodes:this.number('discovery-episodes',1,10000),seed:this.number('discovery-seed',0,2147483647),device:this.el('discovery-device').value};
      if(!Number.isInteger(config.episodes)||!Number.isInteger(config.seed))throw new Error('Episodes and seed must be whole numbers.');
      if(task!=='classification')config.failure_threshold=this.number('discovery-threshold',0,1);
      if(task==='classification') {
        config.label_map=this.parseLabels();this.updateLabels();
      }
      if(task==='summarization') {
        config.summary_metric=this.el('discovery-metric').value;
        if(!this.summaryDataset) {
          config.reference_summary=this.el('discovery-reference').value.trim();
          if(!config.reference_summary)throw new Error('Enter a human reference summary before discovery; both summary metrics require it.');
        }
      }
      if(task==='generation') {
        config.generation_metric='combined';
        config.generation_precision=this.el('discovery-generation-precision').value;
        config.judge_model=this.el('discovery-judge-model').value;
        if(!['CountingSheep/vev-4b','Qwen/Qwen3-VL-2B-Instruct'].includes(config.judge_model))throw new Error('Choose Vev 4B or Qwen3-VL 2B for combined review.');
        config.generation_bias_suite=this.biasSuite();
        config.generation_bias_repeats=this.number('discovery-bias-repeats',2,16);
        if(!Number.isInteger(config.generation_bias_repeats))throw new Error('Bias comparison repeats must be a whole number from 2 to 16.');
        config.generation_steps=this.number('discovery-steps',1,100);config.generation_size=Number(this.el('discovery-size').value);
      }
      this.pending=true;this.pendingKind='discoveries';this.message('');this.controls();
      this.el('discovery-status').textContent='Starting discovery…';
      const job=await this.api('/api/discoveries',config);
      if(version!==this.version)return;
      this.pending=false;this.job=job;this.training=null;this.adaptation.reset();
      const point=this.explorer.point,action=this.explorer.action;
      this.inspectionOpen=false;
      this.selected=point&&action!==null&&action!==undefined?`${point.index}:${action}`:null;
      this.el('finetune-checkpoint').hidden=true;this.el('finetune-report').hidden=true;
      this.el('finetune-progress').hidden=true;this.el('finetune-progress').value=0;this.el('finetune-path').value='';
      this.el('finetune-status').textContent='';this.el('finetune-batch-info').textContent='';this.receive('discoveries',job,version);
      this.el('finetune-cleanup-status').hidden=true;this.el('finetune-cleanup-status').textContent='';
    } catch(error) {
      if(version!==this.version)return;
      if(this.pendingKind==='discoveries')this.el('discovery-status').textContent='Could not start discovery.';
      this.pending=false;this.pendingKind=null;this.message(error.message);this.controls();
    }
  }
  receive(kind,job,version) {
    if(version!==this.version)return;
    this.pending=false;this.pendingKind=null;
    if(kind==='finetunes'){this.training=job;this.renderTraining();}else{this.job=job;this.render();}
    this.controls();this.el('discovery-retry').hidden=true;
    clearTimeout(this.timer);
    if(['queued','running'].includes(job.status))this.timer=setTimeout(()=>this.poll(kind,version),650);
  }
  async poll(kind,version) {
    if(version!==this.version||this.cancelling===version||this.polling?.version===version)return;
    const job=kind==='finetunes'?this.training:this.job;if(!job)return;
    const request={kind,version};this.polling=request;clearTimeout(this.timer);this.controls();
    const message=value=>kind==='finetunes'?this.adaptation.message(value):this.message(value);
    try {const result=await this.api(`/api/${kind}/${job.id}`);if(version!==this.version)return;message('');this.receive(kind,result,version);}
    catch(error){if(version!==this.version)return;message(`Status unavailable: ${error.message}. Retry status or cancel the job.`);this.el('discovery-retry').hidden=false;this.controls();}
    finally{if(this.polling===request){this.polling=null;this.controls();}}
  }
  async cancel(kind) {
    const job=kind==='finetunes'?this.training:this.job;
    if(!job||!['queued','running'].includes(job.status)||this.cancelling===this.version)return;
    // Retire reads taken before cancellation so they cannot restore a running job.
    const version=++this.version;this.cancelling=version;clearTimeout(this.timer);this.controls();
    const message=value=>kind==='finetunes'?this.adaptation.message(value):this.message(value);
    try {message('');const result=await this.api(`/api/${kind}/${job.id}/cancel`,{});if(version===this.version)this.receive(kind,result,version);}
    catch(error){if(version===this.version){message(error.message);this.el('discovery-retry').hidden=false;}}
    finally{if(this.cancelling===version){this.cancelling=null;this.controls();}}
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
    for(const [name,url] of Object.entries(job.artifacts||{})){if(typeof url!=='string'||!url.startsWith('/discoveries/'))continue;const a=document.createElement('a');a.textContent=`Open ${name} ↗`;a.href=url;a.target='_blank';a.rel='noopener noreferrer';links.push(a);}
    this.el('discovery-artifacts').replaceChildren(...links);
    if(!this.attached){this.hideResults();return;}
    if(this.explorer.task==='classification')this.renderLabels();
    this.el('severity-legend').hidden=false;
    if(this.explorer.setDiscovery)this.explorer.setDiscovery(job);else this.explorer.view.setDiscovery(job);
    this.renderLegend();
    this.inspectionOptions();
    this.inspectSelected(false);
  }
  renderLegend() {
    const predicted=this.landscapeMode==='predicted',available=this.job?.predicted_landscape?.policy_updates>0;
    this.el('landscape-mode').value=this.landscapeMode;
    this.explorer.view.setLandscapeMode?.(this.landscapeMode);
    this.el('landscape-scale-label').textContent=predicted?'Predicted reward':'Measured severity';
    this.el('landscape-scale').title=predicted?
      'DQN expected reward, not failure probability. Originals average all action rewards. Colors use 0 to 1; out-of-range estimates use the nearest end color.':
      'Measured severity from model evaluations. Originals average their tested action pairs only.';
    this.el('landscape-unavailable').textContent=predicted?(available?'Unavailable':'Awaiting learning'):'Unevaluated';
    this.el('landscape-evidence').hidden=!predicted||!available;
  }
  inspectionOptions() {
    const severity=record=>Number.isFinite(record.severity)?record.severity:-Infinity;
    const records=[...(this.job?.records||[])].sort((a,b)=>severity(b)-severity(a)||a.input_index-b.input_index||a.action_id-b.action_id);
    const selector=this.el('discovery-pair');
    selector.title='Highest measured severity first; unevaluated actions last.';
    const option=(value,label)=>{const el=document.createElement('option');el.value=value;el.textContent=label;return el;};
    const choices=records.map(record=>option(this.key(record),`Input ${record.input_index} · action ${record.action_id} · severity ${this.percent(record.severity)} (${record.visits} visits)`));
    if(this.selected&&!records.some(record=>this.key(record)===this.selected)) {
      const [index,action]=this.selected.split(':');
      choices.push(option(this.selected,`Input ${index} · action ${action} · unevaluated`));
    }
    selector.replaceChildren(option('','Select an input/action…'),...choices);
    selector.value=this.inspectionOpen?this.selected||'':'';
    this.el('discovery-records').hidden=!choices.length;
  }
  inspectPair(index,action) {
    if(!this.attached||!this.job)return;
    this.selected=index===null||action===null?null:`${index}:${action}`;this.inspectionOpen=false;
    this.adaptation.setGraphAction(index,action);
    this.inspectionOptions();this.inspectSelected(false);this.controls();
  }
  paragraph(text) {const el=document.createElement('p');el.textContent=text;return el;}
  percent(value) {return typeof value==='number'&&Number.isFinite(value)?`${(100*value).toFixed(1)}%`:'—';}
  failureReview(output) {
    const scores=document.createElement('dl'),details=document.createElement('details'),summary=document.createElement('summary');
    scores.className='metric-rows failure-scores';details.className='reference-disclosure failure-evidence';summary.textContent='Failure evidence';details.append(summary);
    for(const [key,label] of [['content','Content failure'],['instructions','Instructions'],['visual','Visual defects'],['bias','Bias disparity']]) {
      const score=output.failure_scores[key],evidence=output.failure_evidence?.[key]||{};
      const measured=typeof score==='number'&&Number.isFinite(score);
      const status=evidence.status||(measured?'evaluated':'not_evaluated');
      const statusLabel=String(status).replaceAll('_',' ').replace(/^./,letter=>letter.toUpperCase());
      const name=document.createElement('dt'),value=document.createElement('dd');name.textContent=label;
      value.textContent=measured?`${this.format(score)} · ${statusLabel}`:statusLabel;scores.append(name,value);
      const section=document.createElement('section'),heading=document.createElement('h5');heading.textContent=label;section.append(heading);
      for(const [field,title] of [['method','Method'],['question','Question'],['answer','Answer'],['reason','Reason']]) {
        if(evidence[field]!==undefined&&evidence[field]!==null)section.append(this.paragraph(`${title}: ${evidence[field]}`));
      }
      if(evidence.readout==='vev_expected_ordinal') {
        section.append(this.paragraph(`Expected severity: ${this.format(evidence.value)} / 4`));
        if(Array.isArray(evidence.probabilities))section.append(this.paragraph(`Rating weights (0–4): ${evidence.probabilities.map(value=>this.percent(value)).join(' · ')}`));
      }
      if(key==='bias'&&evidence.status==='evaluated'&&evidence.scope==='matched_suite_action')section.append(this.paragraph('Measured across the matched suite for this action; the observed quality disparity contributes to every input tested with this action.'));
      if(evidence.reference)section.append(this.paragraph(`Reference: ${evidence.reference}`));
      if(evidence.suite_name)section.append(this.paragraph(`Suite: ${evidence.suite_name}`));
      for(const [group,rate] of Object.entries(evidence.group_rates||{})) {
        const count=evidence.group_sample_counts?.[group];
        section.append(this.paragraph(`${group}: ${this.format(rate)} mean content failure${Number.isInteger(count)?` · ${count} samples`:''}`));
      }
      if(Array.isArray(evidence.cases)&&evidence.cases.length) {
        const cases=document.createElement('details'),title=document.createElement('summary'),records=document.createElement('pre');
        title.textContent='Matched prompts, seeds, and case results';records.textContent=JSON.stringify(evidence.cases,null,2);cases.append(title,records);section.append(cases);
      }
      if(!evidence.method&&!evidence.reason)section.append(this.paragraph(statusLabel));
      details.append(section);
    }
    return [scores,details];
  }
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
    if(output.clip_similarity!==undefined)children.push(this.paragraph(`CLIP scene alignment: ${this.format(output.clip_similarity)}`));
    if(output.prompt)children.push(this.paragraph(`Prompt: ${output.prompt}`));
    if(output.seed!==undefined)children.push(this.paragraph(`Noise seed: ${output.seed}`));
    if(output.failure_scores)children.push(...this.failureReview(output));
    if(Array.isArray(output.checks)&&output.checks.length) {
      const detail=document.createElement('details'),summary=document.createElement('summary'),list=document.createElement('ol');
      detail.className='reference-disclosure';
      summary.textContent=`Content requirements: ${output.checks_passed??output.checks.filter(check=>check.passed).length} / ${output.checks_total??output.checks.length} passed`;
      for(const [index,check] of output.checks.entries()) {
        const evidence=output.check_evidence?.[index],verification=evidence?.method==='vev_target_verification';
        const answer=verification?`Target: ${check.expected} · Supported: ${check.answer} · Model support: ${this.percent(evidence.probabilities?.[1])}`:`Expected: ${check.expected} · Answer: ${check.answer}`;
        const row=document.createElement('li');row.textContent=`${check.passed?'Passed':'Failed'} · ${check.question} · ${answer}${check.skill?` · ${check.skill}`:''}`;list.append(row);
      }
      detail.append(summary,list);children.push(detail);
    } else if(output.quality!==undefined&&!output.failure_scores)children.push(this.paragraph(`Quality review score: ${this.format(output.quality)}`));
    if(!children.length)children.push(this.paragraph('No model output available.'));
    this.el(id).replaceChildren(...children);
  }
  policyRows(action) {
    const statistics=this.job?.policy_statistics;
    if(!statistics||action===null||action===undefined)return [];
    const entry=statistics.actions?.find(item=>item.action_id===action),updated=statistics.policy_updates>0;
    const number=value=>Number.isFinite(value)?value.toPrecision(3):'Unavailable';
    const reason=!updated?'no policy updates yet':statistics.input_count<2?'requires at least 2 original inputs':'not available';
    const variation=updated&&statistics.input_count>1&&Number.isFinite(entry?.std)?number(entry.std):`Unavailable (${reason})`;
    return [['Mean DQN action preference',updated?number(entry?.mean):'Unavailable (no policy updates yet)'],
      ['Preference variation (sample SD)',variation],['Original inputs in policy statistics',String(statistics.input_count)],
      ['Tested inputs for this action',`${entry?.evaluated_inputs??'—'} / ${statistics.input_count}`]];
  }
  metricRows(record,action=record?.action_id) {
    const labels={target_id:'Target class ID',target_label:'Target label',target_probability:'Target confidence',baseline_target_probability:'Original target confidence',target_confidence:'Target confidence',baseline_target_confidence:'Original target confidence',error:'Prediction differs from target',new_error:'New error after action',safety_filtered:'Image filtered by safety checker',confidence_drop:'Target confidence drop',baseline_error:'Original already incorrect',prediction_changed:'Prediction changed',labelled:'Human label supplied',rouge_l:'Reference overlap (ROUGE-L F1)',nonrepetition:'Distinct bigram ratio',source_support:'Source support (NLI)',quality:'Quality review score',baseline_quality:'Original quality score',quality_drop:'Quality drop from original',baseline_low_quality:'Original has low quality',clip_similarity:'CLIP scene alignment',baseline_clip_similarity:'Original CLIP scene alignment',checks_passed:'Requirements passed',checks_total:'Requirements evaluated'};
    Object.assign(labels,{content_failure:'Content failure score',instructions_failure:'Instruction failure score',visual_failure:'Visual defect score',bias_failure:'Bias disparity score',baseline_severity:'Baseline severity',severity_increase:'Severity increase from baseline',failure_threshold:'Failure severity threshold'});
    if(record?.output?.failure_scores)Object.assign(labels,{quality:'Combined quality (1 − severity)',baseline_quality:'Baseline quality (1 − severity)'});
    else if(record?.output?.checks?.length)Object.assign(labels,{quality:'Content requirements passed (fraction)',baseline_quality:'Original requirements passed (fraction)',quality_drop:'Decrease in passed requirements (fraction)'});
    const rows=record?[['Criterion',record.criterion||this.job?.criterion||'—'],['Severity',this.percent(record.severity)],['Search reward',this.format(record.reward)],['Observed failure rate',this.percent(record.failure_rate)]]:[];
    const prediction=this.explorer.predictionFor?.(Number(this.selected?.split(':')[0]),action);
    if(prediction)rows.push(['Predicted reward',this.format(prediction.q_value)],
      ['DQN action preference (this input)',this.percent(prediction.preference)],['Evidence',record?'Model evaluated':'Estimated only']);
    rows.push(...this.policyRows(action));
    for(const [key,value] of Object.entries(record?.metrics||{})) {
      const label=labels[key]||key.replaceAll('_',' ').replace(/^./,letter=>letter.toUpperCase());
      const display=typeof value==='boolean'?(value?'Yes':'No'):typeof value==='number'?this.format(value):String(value??'—');
      rows.push([label,display]);
    }
    const list=document.createElement('dl');list.className='metric-rows';
    for(const [label,value] of rows){const name=document.createElement('dt'),detail=document.createElement('dd');name.textContent=label;detail.textContent=value;list.append(name,detail);}
    this.el('discovery-components').replaceChildren();this.el('discovery-components').textContent=record?'':'This action has not been evaluated.';
    if(rows.length)this.el('discovery-components').append(list);
  }
  inspectSelected(focus=false) {
    if(!this.attached){this.hideResults();return;}
    this.el('discovery-inspector').hidden=!this.inspectionOpen||!this.selected;
    if(!this.inspectionOpen||!this.selected)return;
    const record=this.job?.records?.find(record=>this.key(record)===this.selected);
    const index=record?.input_index??Number(this.selected.split(':')[0]);
    const baseline=this.job?.baseline?.find(item=>item.input_index===index);
    this.output('discovery-baseline',baseline?.output);this.output('discovery-output',record?.output);
    this.metricRows(record,record?.action_id??(this.selected?Number(this.selected.split(':')[1]):null));
    if(focus&&record){this.explorer.focusMeasured(record.input_index,record.action_id);this.adaptation.setGraphAction(record.input_index,record.action_id);}
    this.controls();
  }
  async startTraining(selection) {
    if(this.active||!this.attached||!this.adaptation.current||!selection?.action_ids?.length)return;
    let version=this.version;
    try {
      const config={...selection,discovery_id:this.job.id,...this.adaptation.trainingSettings()};
      if(!Number.isInteger(config.epochs)||!Number.isInteger(config.seed))throw new Error('Epochs and seed must be whole numbers.');
      // Training owns a new polling generation. An in-flight discovery response
      // cannot replace its timer or resurrect the terminal discovery.
      version=++this.version;clearTimeout(this.timer);
      this.pending=true;this.pendingKind='finetunes';this.controls();this.message('');this.adaptation.message('');this.el('finetune-status').textContent='Starting fine-tuning…';
      this.el('finetune-checkpoint').hidden=true;this.el('finetune-report').hidden=true;
      this.el('finetune-progress').hidden=true;this.el('finetune-progress').value=0;
      this.el('finetune-batch-info').textContent='';
      this.el('finetune-cleanup-status').hidden=true;this.el('finetune-cleanup-status').textContent='';
      const job=await this.api('/api/finetunes',config);if(version!==this.version)return;
      this.pending=false;this.receive('finetunes',job,version);
    } catch(error){
      if(version!==this.version)return;
      if(this.pendingKind==='finetunes')this.el('finetune-status').textContent='Could not start fine-tuning.';
      this.pending=false;this.pendingKind=null;this.adaptation.message(error.message);this.controls();
    }
  }
  renderTraining() {
    const job=this.training;
    const device=job.device==='cuda'?`CUDA GPU${job.device_name?` · ${job.device_name}`:''}`:job.device==='cpu'?'CPU':'';
    const count=value=>(value||0).toLocaleString('en-US'),parts=[job.status];
    if(job.phase&&job.phase!==job.status)parts.push(job.phase);
    if(job.epoch&&job.epochs)parts.push(`Epoch ${job.epoch}/${job.epochs}`);
    if(job.total_examples)parts.push(`${count(job.processed_examples)}/${count(job.total_examples)} examples`);
    if(job.total)parts.push(`${count(job.completed)}/${count(job.total)} updates`);
    if(Number.isFinite(job.loss))parts.push(`loss ${this.format(job.loss)}`);
    if(device)parts.push(device);
    this.el('finetune-status').textContent=parts.join(' · ');
    this.el('finetune-batch-info').textContent=job.effective_batch_size?`Batch ${job.effective_batch_size} · processed in chunks of ${job.micro_batch_size||job.effective_batch_size}${job.oom_retries?` · ${job.oom_retries} memory adjustment${job.oom_retries===1?'':'s'}`:''}`:'';
    const cleanup=this.el('finetune-cleanup-status'),cleanupMessage=job.status==='completed'&&typeof job.cleanup?.message==='string'?job.cleanup.message:'';
    cleanup.textContent=cleanupMessage;cleanup.hidden=!cleanupMessage;
    if(job.error)this.adaptation.message(job.error);
    this.el('finetune-progress').hidden=!job.total;this.el('finetune-progress').max=job.total||1;this.el('finetune-progress').value=job.completed||0;
    this.el('finetune-checkpoint').hidden=job.status!=='completed'||!job.checkpoint||!this.attached;
    this.el('finetune-path').value=job.checkpoint||'';
    const url=job.artifacts?.report,report=this.el('finetune-report');
    report.hidden=!this.attached||typeof url!=='string'||!url.startsWith('/discoveries/');
    if(!report.hidden){report.href=url;report.target='_blank';report.rel='noopener noreferrer';}
  }
}
