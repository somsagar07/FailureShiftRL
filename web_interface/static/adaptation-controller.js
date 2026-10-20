'use strict';

// Select global action combinations; examples are built by the server across training inputs.
class AdaptationController {
  constructor({api,explorer,discovery}) {
    Object.assign(this,{api,explorer,discovery});this.el=id=>document.getElementById(id);
    this.version=0;this.fileRead=0;this.selected=new Set();this.targets=new Map();this.targetFields=new Map();
    this.open=false;this.summary=null;this.graphAction=null;this.graphSelection=0;this.loading=false;this.reading=false;
    this.explorer.view.onRender=()=>this.positionGraphAction();
    const bind=(id,event,fn)=>this.el(id).addEventListener(event,fn);
    bind('adapt','click',()=>this.show());bind('adapt-close','click',()=>this.discovery.showDiscovery?.());
    bind('adapt-toggle-action','click',()=>this.toggle(this.graphAction));
    bind('adapt-graph-toggle','click',()=>this.toggle(this.graphAction));
    bind('adapt-action-choice','change',()=>{const value=this.el('adapt-action-choice').value;this.setGraphAction(null,value===''?null:Number(value));});
    bind('adapt-data-source','change',()=>{this.fileRead++;this.reading=false;this.message('');this.renderTargets();this.controls();});
    bind('adapt-include-originals','change',()=>this.controls());
    bind('adapt-summary-reference','input',()=>this.controls());
    bind('adapt-data-json','input',()=>{this.fileRead++;this.reading=false;this.message('');this.controls();});
    bind('adapt-data-file','change',()=>this.upload());
    for(const id of ['finetune-epochs','finetune-lr','finetune-seed','finetune-weight-decay','finetune-max-grad-norm'])bind(id,'input',()=>this.controls());
    for(const id of ['finetune-device','finetune-batch-size','finetune-cleanup'])bind(id,'change',()=>this.controls());
    this.controls();
  }
  get available() {
    const d=this.discovery;
    return Boolean(d.attached&&d.job&&['completed','cancelled','error'].includes(d.job.status)&&d.job.records?.length);
  }
  get current() {return this.available&&this.summary?.discovery_id===this.discovery.job.id;}
  get task() {return this.summary?.task||this.explorer.task;}
  get summaryDataset() {return this.task==='summarization'&&Boolean(this.summary?.defaults.uses_dataset||this.summary?.defaults.dataset);}
  get generationDataset() {return this.task==='generation'&&Boolean(this.summary?.defaults.uses_dataset||this.summary?.defaults.dataset||this.summary?.input_count>1);}
  reset() {
    this.version++;this.fileRead++;this.loading=false;this.reading=false;this.open=Boolean(this.discovery.trainingActive);
    this.summary=null;this.selected.clear();this.targets.clear();this.targetFields.clear();this.graphAction=null;
    this.el('adapt-data-json').value='';this.el('adapt-data-file').value='';this.el('adapt-selection').replaceChildren();
    this.el('adapt-targets').replaceChildren();this.message('');this.syncGraph();this.controls();
  }
  message(value) {this.el('adapt-error').textContent=value;this.el('adapt-error').hidden=!value;}
  async show() {
    if((!this.available&&!this.discovery.trainingActive)||(this.discovery.active&&!this.discovery.trainingActive))return;
    this.open=true;this.el('discovery-panel').hidden=true;this.controls();
    this.el('embedding-wrap').scrollIntoView?.({block:'start'});
    if(this.current||!this.available){this.syncGraph();return;}
    const version=++this.version,id=this.discovery.job.id,graphSelection=this.graphSelection;
    this.loading=true;this.message('');this.controls();
    try {
      const summary=await this.api(`/api/discoveries/${id}/adaptation`);
      if(version!==this.version||!this.available||this.discovery.job.id!==id)return;
      this.summary=summary;this.selected=new Set(summary.suggested_action_ids);this.targets.clear();
      this.el('adapt-include-originals').checked=summary.defaults.include_originals;
      this.el('adapt-data-source').value=this.generationDataset?'json':'current';
      this.el('adapt-summary-reference').value=summary.defaults.reference_summary||'';
      this.el('adapt-data-json').value='';this.el('adapt-data-file').value='';
      this.renderActions();this.renderTargets();this.syncGraph();
      const first=this.selected.values().next().value;
      if(graphSelection===this.graphSelection&&this.graphAction===null) {
        if(this.explorer.action!==null&&this.explorer.action!==undefined)this.setGraphAction(this.explorer.point?.index,this.explorer.action);
        else if(first!==undefined)this.focusAction(first);
      }
    } catch(error){if(version===this.version)this.message(error.message);}
    finally{if(version===this.version){this.loading=false;this.controls();}}
  }
  close() {
    this.version++;this.loading=false;this.fileRead++;this.reading=false;
    this.open=false;this.syncGraph();this.controls();
  }
  syncGraph() {
    this.explorer.view.setAdaptation?.(this.open&&this.current?[...this.selected]:[]);
    this.el('adapt-legend').hidden=!(this.open&&this.current&&this.selected.size);
  }
  setGraphAction(index,action) {
    this.graphSelection++;this.graphAction=action;
    this.el('adapt-action-choice').value=action===null?'':String(action);this.controls();
  }
  positionGraphAction() {
    const menu=this.el('adapt-graph-action');if(menu.hidden)return;
    const view=this.explorer.view,node=view.projected?.find(point=>point.kind==='action'&&point.selected&&point.data.id===this.graphAction);
    if(!node||node.x<0||node.y<0||node.x>view.width||node.y>view.height) {
      Object.assign(menu.style,{left:'',top:'',bottom:''});return;
    }
    const {width,height}=menu.getBoundingClientRect(),margin=8,gap=node.radius+14;
    let left=node.x+gap,top=node.y-height/2;
    if(left+width>view.width-margin) {
      left=node.x-gap-width;
      if(left<margin) {
        left=node.x-width/2;
        top=node.y+gap+height<=view.height-margin?node.y+gap:node.y-gap-height;
      }
    }
    left=Math.max(margin,Math.min(left,view.width-width-margin));
    top=Math.max(margin,Math.min(top,view.height-height-margin));
    Object.assign(menu.style,{left:`${left}px`,top:`${top}px`,bottom:'auto'});
  }
  toggle(id) {
    if(this.discovery.active||!this.current||!this.summary.actions.some(action=>action.action_id===id))return;
    if(this.selected.has(id))this.selected.delete(id);else this.selected.add(id);
    this.message('');this.renderActions();this.renderTargets();this.syncGraph();this.controls();
  }
  describe(row) {return `Action ${row.action_id} · ${this.explorer.describeAction?.(row.action)||row.action.name?.replaceAll('_',' ')||'combination'}`;}
  focusAction(id) {
    const row=this.summary?.actions.find(row=>row.action_id===id);if(!row)return;
    const index=row.example_input_index??this.explorer.point?.index??0;
    if(this.explorer.focusAction)this.explorer.focusAction(index,id);else this.explorer.focusMeasured(index,id);
    this.setGraphAction(index,id);
  }
  renderActions() {
    const option=(value,label)=>{const node=document.createElement('option');node.value=value;node.textContent=label;return node;};
    this.el('adapt-action-choice').replaceChildren(option('','Select a graph node or action…'),...this.summary.actions.map(row=>option(String(row.action_id),this.describe(row))));
    this.el('adapt-action-choice').value=this.graphAction===null?'':String(this.graphAction);
    this.actionButtons=[];
    const rows=this.summary.actions.filter(row=>this.selected.has(row.action_id)).map(row=>{
      const tr=document.createElement('tr');
      const values=[this.describe(row),Number.isFinite(row.failure_rate)?`${Math.round(100*row.failure_rate)}% · ${row.failure_count}/${row.evaluated_inputs}`:'Unevaluated',
        `${row.evaluated_inputs} / ${this.summary.input_count}`,Number.isFinite(row.mean_severity)?`${Math.round(100*row.mean_severity)}%`:'—'];
      for(const [index,value] of values.entries()) {
        const cell=document.createElement('td');cell.textContent=value;
        if(index)cell.setAttribute('data-label',['','Failure rate','Tested inputs','Severity'][index]);
        tr.append(cell);
      }
      const actions=document.createElement('td');actions.className='adapt-row-buttons';
      for(const [text,fn] of [['View',()=>this.focusAction(row.action_id)],['Remove',()=>this.toggle(row.action_id)]]){
        const button=document.createElement('button');button.type='button';button.textContent=text;button.addEventListener('click',fn);
        button.setAttribute('aria-label',`${text} ${this.describe(row)}`);actions.append(button);this.actionButtons.push({button,mutates:text==='Remove'});
      }
      tr.append(actions);return tr;
    });
    this.el('adapt-selection').replaceChildren(...rows);
  }
  renderTargets() {
    if(!this.summary)return;
    const task=this.task,json=this.el('adapt-data-source').value==='json';
    this.el('adapt-data-source-field').hidden=task==='classification'||this.generationDataset;
    this.el('adapt-current-option').textContent=task==='generation'?'Reference images by action':this.summaryDataset?'Analyzed dataset':'Analyzed article';
    this.el('adapt-json-fields').hidden=task==='classification'||!json;
    this.el('adapt-summary-field').hidden=task!=='summarization'||json||this.summaryDataset;
    this.el('adapt-originals-field').hidden=task==='generation';
    this.el('adapt-data-help').textContent=task==='classification'?`${this.summary.input_count} analyzed images · labels from discovery`:
      this.summaryDataset&&!json?`${this.summary.input_count} analyzed articles · ${this.summary.defaults.dataset?.reference_count??0} / ${this.summary.input_count} reference summaries. Selected actions use each article’s reference.`:
      task==='summarization'?'Apply selected actions to each article. Check references against the edited text.':this.generationDataset?
        `${this.summary.input_count} analyzed prompts. Provide approved reference images for specific prompt/action pairs. Only supplied pairs train; each selected action needs at least one target.`:'Provide desired reference images for each selected action.';
    this.el('adapt-json-help').textContent=this.generationDataset?'Each record: input_index (starts at 0), action_id, image_path (absolute server path). Use the analyzed prompt and action combination; one target per pair.':task==='generation'?'Each record: action_id, image_path (absolute server path), and optional prompt caption.':
      'Each record: text and summary. For negation, add action_summaries with the action ID and corrected summary.';
    const exampleAction=this.selected.values().next().value??this.summary.actions[0]?.action_id;
    const imageExample={...(this.generationDataset?{input_index:0}:{}),action_id:exampleAction,image_path:'/data/desired.png'};
    this.el('adapt-data-json').placeholder=task==='generation'?JSON.stringify([imageExample],null,2):'[{"text": "Article…", "summary": "Reference…"}]';
    this.targetFields=new Map();
    const rows=[];
    if(!json&&task!=='classification'&&!this.summaryDataset&&!this.generationDataset)for(const row of this.summary.actions.filter(row=>this.selected.has(row.action_id))) {
      if(task==='summarization'&&row.action.name!=='negate_sentence')continue;
      const wrapper=document.createElement('div'),label=document.createElement('label'),input=document.createElement('textarea');
      const title=task==='generation'?`Desired images · ${this.describe(row)}`:`Corrected reference · ${this.describe(row)}`;
      const id=`adapt-target-${row.action_id}`;label.htmlFor=id;label.textContent=title;input.id=id;input.rows=2;
      input.maxLength=50000;input.value=this.targets.get(row.action_id)||'';
      input.placeholder=task==='generation'?'Absolute server paths, one image per line':'Reference summary matching the negated article';
      input.setAttribute('aria-label',title);
      input.addEventListener('input',()=>{this.targets.set(row.action_id,input.value);this.controls();});
      this.targetFields.set(row.action_id,input);wrapper.append(label,input);rows.push(wrapper);
    }
    this.el('adapt-targets').replaceChildren(...rows);
  }
  parseDataset(text) {
    if(new TextEncoder().encode(text).length>20*1024*1024)throw Error('Use training JSON under 20 MB.');
    let records;try{records=JSON.parse(text.replace(/^\uFEFF/,''));}catch{throw Error('Training data must be valid JSON.');}
    if(!Array.isArray(records)||!records.length||records.some(row=>!row||typeof row!=='object'||Array.isArray(row)))throw Error('Use a nonempty JSON array of training records.');
    return records;
  }
  async upload() {
    if(this.discovery.active)return;
    const file=this.el('adapt-data-file').files?.[0];if(!file)return;
    this.el('adapt-data-file').value='';const read=++this.fileRead;this.reading=true;this.controls();
    try {
      if(file.size>20*1024*1024)throw Error('Use training JSON under 20 MB.');
      const text=await file.text();if(read!==this.fileRead)return;
      const records=this.parseDataset(text);this.el('adapt-data-json').value=JSON.stringify(records);
      this.el('adapt-data-source').value='json';this.message('');this.renderTargets();
    } catch(error){if(read===this.fileRead)this.message(`File not applied. ${error.message}`);}
    finally{if(read===this.fileRead){this.reading=false;this.controls();}}
  }
  trainingConfig() {
    if(!this.current)throw Error('Finish discovery for this analysis first.');
    if(!this.selected.size)throw Error('Select at least one action.');
    const action_ids=[...this.selected],task=this.task,json=this.el('adapt-data-source').value==='json';
    const include_originals=task!=='generation'&&this.el('adapt-include-originals').checked;
    const config={action_ids,include_originals};
    let count;
    if(task==='classification')count=this.summary.input_count*(action_ids.length+Number(include_originals));
    else if(this.summaryDataset&&!json) {
      if(this.summary.actions.some(action=>this.selected.has(action.action_id)&&action.action.name==='negate_sentence'))
        throw Error('Remove Negate sentence or supply training JSON with corrected summaries for each article.');
      if((this.summary.defaults.dataset?.reference_count??0)<this.summary.input_count)
        throw Error('Each article needs a reference summary. Choose another dataset or supply training JSON.');
      count=this.summary.input_count*(action_ids.length+Number(include_originals));
    } else if(task==='summarization') {
      const action_summaries=Object.fromEntries([...this.targets].filter(([id,value])=>this.selected.has(id)&&value.trim()).map(([id,value])=>[id,value.trim()]));
      const base={text:this.summary.defaults.source_text,summary:this.el('adapt-summary-reference').value.trim()};
      if(Object.keys(action_summaries).length)base.action_summaries=action_summaries;
      const records=json?this.parseDataset(this.el('adapt-data-json').value):[base];
      for(const row of records) {
        if(typeof row.text!=='string'||!row.text.trim()||typeof row.summary!=='string'||!row.summary.trim())throw Error('Each training pair needs article text and a reference summary.');
        for(const action of this.summary.actions.filter(action=>this.selected.has(action.action_id)&&action.action.name==='negate_sentence'))
          if(typeof row.action_summaries?.[action.action_id]!=='string'||!row.action_summaries[action.action_id].trim())throw Error('Add a corrected reference for the negation action.');
      }
      config.training_data=records;count=records.length*(action_ids.length+Number(include_originals));
    } else {
      if(this.generationDataset&&!json)throw Error('Provide training JSON with an input_index, action_id, and image_path for each approved pair.');
      const corpus=json?this.parseDataset(this.el('adapt-data-json').value):action_ids.flatMap(action_id=>(this.targets.get(action_id)||'').split(/\r?\n/).map(value=>value.trim()).filter(Boolean).map(image_path=>({action_id,image_path})));
      const catalog=new Set(this.summary.actions.map(row=>row.action_id));
      if(corpus.some(row=>!Number.isInteger(row.action_id)||!catalog.has(row.action_id)))throw Error('Reference images need valid action IDs from this analysis.');
      const records=corpus.filter(row=>this.selected.has(row.action_id));
      const pairs=new Set();
      for(const row of records) {
        if(typeof row.image_path!=='string'||!row.image_path.startsWith('/'))throw Error('Use absolute server paths for desired reference images.');
        if(this.generationDataset) {
          if(!Number.isInteger(row.input_index)||row.input_index<0||row.input_index>=this.summary.input_count)
            throw Error(`Each target needs an input_index from 0 to ${this.summary.input_count-1}.`);
          if(Object.hasOwn(row,'prompt'))throw Error('Prompt overrides are unavailable for dataset targets; use the analyzed prompt and action combination.');
          const pair=`${row.input_index}:${row.action_id}`;
          if(pairs.has(pair))throw Error('Remove duplicate prompt/action target pairs.');
          pairs.add(pair);
        }
      }
      if(action_ids.some(id=>!records.some(row=>row.action_id===id)))throw Error('Add desired reference images for every selected action.');
      config.training_data=records;count=records.length;
    }
    // A running older server may still advertise its former materialization limit.
    if(Number.isFinite(this.summary.max_examples)&&this.summary.max_examples>0&&count>this.summary.max_examples)
      throw Error(`This server still has a ${this.summary.max_examples.toLocaleString('en-US')}-example limit. Restart the server to use streaming training.`);
    this.exampleCount=count;return config;
  }
  trainingSettings() {
    const number=id=>{const value=this.el(id).value.trim();return value===''?NaN:Number(value);};
    const epochs=number('finetune-epochs'),learning_rate=number('finetune-lr');
    const value=this.el('finetune-batch-size').value||'auto',batch_size=value==='auto'?'auto':Number(value);
    const device=this.el('finetune-device').value||'auto',seed=number('finetune-seed');
    const weight_decay=number('finetune-weight-decay'),max_grad_norm=number('finetune-max-grad-norm');
    if(!Number.isInteger(epochs)||epochs<1||epochs>100)throw Error('Choose 1–100 whole epochs.');
    if(!Number.isFinite(learning_rate)||learning_rate<1e-8||learning_rate>.1)throw Error('Choose a learning rate from 0.00000001 to 0.1.');
    if(batch_size!=='auto'&&(!Number.isInteger(batch_size)||batch_size<1||batch_size>1024))throw Error('Choose Automatic or a whole batch size from 1 to 1024.');
    if(!['auto','cuda','cpu'].includes(device))throw Error('Choose Automatic, CUDA GPU, or CPU for training.');
    if(!Number.isInteger(seed)||seed<0||seed>4294967295)throw Error('Choose a whole training seed from 0 to 4294967295.');
    if(!Number.isFinite(weight_decay)||weight_decay<0||weight_decay>1)throw Error('Choose weight decay from 0 to 1.');
    if(!Number.isFinite(max_grad_norm)||max_grad_norm<0||max_grad_norm>100)throw Error('Choose a gradient clipping norm from 0 to 100 (0 disables clipping).');
    return {epochs,learning_rate,device,batch_size,seed,weight_decay,max_grad_norm,cleanup_after_finetune:this.el('finetune-cleanup').checked};
  }
  controls() {
    const active=Boolean(this.discovery.active),current=this.current;
    this.el('adapt').disabled=(!this.available&&!this.discovery.trainingActive)||(active&&!this.discovery.trainingActive);
    this.el('adapt').setAttribute('aria-expanded',String(this.open));
    this.el('adapt').setAttribute('aria-pressed',String(this.open));
    this.el('find-failure').setAttribute('aria-pressed',String(!this.open&&!this.el('discovery-panel').hidden));
    this.el('workspace-layout').setAttribute('data-stage',this.open?'adapt':'analyze');
    this.el('adapt-panel').hidden=!this.open;
    if(this.open)this.el('discovery-panel').hidden=true;
    this.el('adapt-settings').disabled=active||this.loading||!current;
    const action=current&&this.summary.actions.find(row=>row.action_id===this.graphAction);
    const label=this.selected.has(this.graphAction)?'Remove from FT actions':'Add to FT actions';
    for(const id of ['adapt-toggle-action','adapt-graph-toggle']) {
      this.el(id).disabled=active||this.loading||!action;this.el(id).textContent=label;
    }
    this.el('adapt-graph-action').hidden=!this.open||!action||this.loading;
    this.el('adapt-graph-description').textContent=action?this.describe(action):'';
    this.el('adapt-graph-description').title=action?this.describe(action):'';
    this.el('adapt-graph-toggle').title=action?this.describe(action):'';
    this.positionGraphAction();
    this.el('adapt-summary').textContent=this.loading?'Summarizing discovery…':current?`${this.selected.size} FT action${this.selected.size===1?'':'s'} · ${this.generationDataset?'uses supplied prompt/action pairs':'applies across training inputs'}`:'';
    for(const {button,mutates} of this.actionButtons||[])button.disabled=(!current)||(mutates&&active);
    for(const field of this.targetFields.values())field.disabled=active;
    let valid=false,description='',blocker='';
    if(current&&!this.loading)try {
      this.trainingConfig();const {epochs}=this.trainingSettings();valid=true;
      description=`${this.exampleCount.toLocaleString('en-US')} example${this.exampleCount===1?'':'s'} per epoch · ${(this.exampleCount*epochs).toLocaleString('en-US')} total`;
    } catch(error){blocker=error.message;}
    if(this.loading)blocker='Loading training selection…';
    else if(this.reading)blocker='Reading training JSON…';
    else if(active)blocker=this.discovery.trainingActive?'Training is in progress.':'Wait for the current job to finish.';
    else if(!current)blocker='Finish discovery for this analysis first.';
    this.el('adapt-training-count').textContent=description;
    this.el('adapt-training-blocker').textContent=blocker;this.el('adapt-training-blocker').hidden=!blocker;
    this.el('finetune-start').disabled=!valid||active||this.loading||this.reading;
  }
  async start() {
    if(this.discovery.active||this.loading||this.reading)return;
    try{const config=this.trainingConfig();this.trainingSettings();this.message('');await this.discovery.startTraining(config);}
    catch(error){this.message(error.message);this.controls();}
  }
}
