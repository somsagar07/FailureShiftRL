'use strict';

// Robotics keeps the shared camera, hit testing, focus and live outlines.
// Its observed original color describes the baseline episode itself.
class RoboticsEmbeddingView extends EmbeddingView {
  nodeColor(node) {
    if(node.data.error)return '#9a72bc';
    if(this.landscapeMode!=='predicted') {
      if(node.kind==='sample')return this.severityColor(typeof node.data.success==='boolean'?Number(!node.data.success):null);
    }
    return super.nodeColor(node);
  }
  movePointer(event) {
    if(this.drag){super.movePointer(event);return;}
    const point=this.pick(event);this.canvas.style.cursor=point?'pointer':'grab';
    if(!point){this.canvas.title='';return;}
    const data=point.data,original=point.kind==='sample';
    const title=original?data.name:Object.entries(data.action||{}).map(([key,value])=>`${key}: ${value}`).join(', ');
    const outcome=data.error?'rollout error':typeof data.success==='boolean'?(data.success?'observed task success':'observed task failure'):'untested';
    const location=original?'episode feature mean':data.pending?'pending layout · feature coordinates unavailable':`measured episode features${Number.isFinite(data.similarity)?` · cosine similarity: ${data.similarity.toFixed(3)}`:''}`;
    this.canvas.title=`${title} · seed ${data.seed} · ${outcome} · ${location}${this.landscapeMode==='predicted'?this.predictionHint(original?data.index:this.focusIndex,original?null:data.id):''}${!original&&this.discovery?.policy_statistics?this.policyHint(data.id):''}`;
  }
}

// This workspace deliberately owns its catalog, upload state, and poll timer.
// Legacy classification/text jobs can finish while the user inspects a robot run.
class RoboticsController {
  constructor({api,onChange=()=>{}}) {
    this.api=api;this.onChange=onChange;this.active=false;this.catalog=null;
    this.loading=false;this.submitting=false;this.uploading=false;this.upload=null;
    this.job=null;this.analysis=null;this.revision=0;this.analysisRevision=-1;
    this.evaluationJob=null;this.environmentJob=null;this.environmentSignature=null;
    this.starterPolicies=[];this.selectedStarter=null;this.starterSignature=null;this.starterChanged=false;this.refreshingStarters=false;
    this.runGeneration=0;this.uploadGeneration=0;this.pollTimer=null;
    this.groups={};this.editingGroup=null;this.selectedId=null;this.selectedInput=null;
    this.graphDataset=null;this.graphPopulated=false;this.predictionRequestKey=null;
    this.landscapeJob=null;this.landscapeChoice=null;
    this.$=id=>document.getElementById(`robotics-${id}`);
    this.$('form').addEventListener('submit',event=>{event.preventDefault();this.analyze();});
    this.$('discover').addEventListener('click',()=>this.discover());
    this.$('search-panel').addEventListener('submit',event=>{event.preventDefault();this.startDiscovery();});
    this.$('close-search').addEventListener('click',()=>this.closeSearch());
    this.$('load-environment').addEventListener('click',()=>this.loadEnvironment());
    this.$('show-evaluation').addEventListener('click',()=>{if(!this.busy&&this.evaluationJob){this.error();this.applyJob(this.evaluationJob);}});
    this.$('starter-policy').addEventListener('change',()=>this.selectStarter());
    this.$('refresh-starters').addEventListener('click',()=>this.refreshStarters());
    this.$('cancel').addEventListener('click',()=>this.cancel());
    this.$('runtime-retry').addEventListener('click',()=>this.loadCatalog());
    this.$('task').addEventListener('change',()=>this.taskChanged());
    this.$('robot').addEventListener('change',()=>this.robotChanged());
    this.$('policy-format').addEventListener('change',()=>this.formatChanged());
    this.$('policy-source').addEventListener('change',()=>this.sourceChanged());
    this.$('choose-folder').addEventListener('click',()=>this.$('policy-folder').click());
    this.$('policy-file').addEventListener('change',()=>this.uploadFiles([...this.$('policy-file').files],false));
    this.$('policy-folder').addEventListener('change',()=>this.uploadFiles([...this.$('policy-folder').files],true));
    this.$('config').addEventListener('change',()=>this.invalidate());
    this.$('config').addEventListener('input',()=>this.invalidate());
    this.$('action-add').addEventListener('change',()=>{
      const key=this.$('action-add').value,group=this.metadata(key);
      if(!group||Object.hasOwn(this.groups,key))return;
      this.groups[key]=[...(this.catalog.default_action_space?.[key]||group.values||[])];this.editingGroup=key;
      this.invalidate();this.renderActions();this.$(`edit-${key}`)?.focus();
    });
    this.$('reset-actions').addEventListener('click',()=>{this.restoreActions();this.invalidate();});
    this.$('episode').addEventListener('change',()=>this.selectEpisode(this.$('episode').value));
    this.$('inspect').addEventListener('change',()=>this.selectNode(this.$('inspect').value));
    this.$('landscape-mode').addEventListener('change',()=>{this.landscapeChoice=this.$('landscape-mode').value;this.predictionRequestKey=null;this.graph.setLandscapeMode(this.landscapeChoice);this.renderLandscapeNote();this.renderInspection();this.loadPredictions();});
    this.$('graph-fit').addEventListener('click',()=>this.fitGraph());
    this.$('graph-zoom-in').addEventListener('click',()=>this.graph.zoom(1.25));
    this.$('graph-zoom-out').addEventListener('click',()=>this.graph.zoom(.8));
    this.graph=new RoboticsEmbeddingView(this.$('graph'),{onSelectSample:point=>this.selectEpisode(point.index),onSelectAction:node=>this.selectNode(node.id),onClear:()=>this.selectEpisode('')});
    this.graph.onRender=()=>{this.$('graph-zoom').textContent=`${Math.round(100*this.graph.fitDistance/this.graph.distance)}%`;};
  }

  get busy() {return this.submitting||this.uploading||this.cancelling||this.isRunning(this.job);}
  isRunning(job) {return Boolean(job&&['queued','running','loading','preparing','cancelling'].includes(job.status));}
  node(tag,text,className) {
    const result=document.createElement(tag);
    if(text!==undefined)result.textContent=text;
    if(className)result.className=className;
    return result;
  }
  error(message='') {this.$('error').textContent=message;this.$('error').hidden=!message;}
  metadata(id) {return this.catalog?.action_groups?.find(group=>group.id===id);}
  groupName(id) {return this.metadata(id)?.label||id.replaceAll('_',' ');}
  percent(value) {return Number.isFinite(value)?`${Math.round(value*100)}%`:'—';}
  number(value) {return Number.isFinite(Number(value))?Number(value).toLocaleString('en-US'):'—';}
  baselineRecords() {const baseline=this.job?.baseline;return Array.isArray(baseline)?baseline:baseline?.records||[];}
  measured(records) {return records.filter(record=>!record.error&&typeof record.success==='boolean');}
  summary(records) {
    const valid=this.measured(records),successes=valid.filter(record=>record.success).length;
    return {episodes:valid.length,errors:records.filter(record=>record.error).length,success_rate:valid.length?successes/valid.length:null,failure_rate:valid.length?1-successes/valid.length:null};
  }
  baselineSummary() {
    const records=this.baselineRecords();
    return records.length?this.summary(records):this.job?.baseline||{episodes:0,errors:0,success_rate:null};
  }
  async enter() {
    this.active=true;this.graph.render();
    if(!this.catalog&&!this.loading)await this.loadCatalog();
    this.updateControls();
  }
  leave() {this.active=false;}
  closeSearch() {this.$('search-panel').hidden=true;this.$('discover').setAttribute('aria-expanded','false');}

  setOptions(id,options,value) {
    this.$(id).replaceChildren(...options.map(option=>{
      const data=typeof option==='string'?{id:option,label:option}:option;
      const element=this.node('option',data.label||data.id);element.value=data.id;return element;
    }));
    if(value!==undefined)this.$(id).value=value;
  }
  async loadCatalog() {
    if(this.loading||this.busy)return;
    this.loading=true;this.error();this.$('runtime-status').textContent='Checking the simulation runtime…';this.updateControls();
    try {
      const catalog=await this.api('/api/robotics/catalog');
      const first=!this.catalog;this.catalog=catalog;
      const runtime=catalog.runtime||{};
      this.$('runtime-status').textContent=runtime.message||(runtime.available?'Simulation runtime ready.':'Simulation runtime is unavailable.');
      this.$('runtime-status').className=runtime.available?'hint':'hint robotics-runtime-unavailable';
      this.$('runtime-retry').hidden=Boolean(runtime.available);
      this.$('runtime-command').hidden=Boolean(runtime.available)||!runtime.command;
      this.$('runtime-command').textContent=runtime.command?`Set up the server runtime: ${runtime.command}`:'';
      if(first) {
        const defaults=catalog.defaults||{},environment=defaults.environment||{};
        this.setOptions('task',[...(catalog.tasks||[]),{id:'__custom__',label:'Other installed task…'}],environment.task||catalog.tasks?.[0]?.id);
        this.setOptions('robot',catalog.robots||[],environment.robots?.[0]);
        this.setOptions('controller',catalog.controllers||[],environment.controller);
        this.setOptions('policy-format',catalog.policy_formats||[]);
        this.$('robots').value=(environment.robots||[this.$('robot').value]).join(', ');
        for(const [id,value] of Object.entries({horizon:environment.horizon,frequency:environment.control_freq,episodes:defaults.episodes_per_variation,budget:defaults.variation_budget,seed:defaults.seed,device:defaults.device})) {
          if(value!==undefined)this.$(id).value=value;
        }
        this.$('record-video').checked=defaults.record_video!==false;
        this.$('budget').max=catalog.limits?.max_actions||5000;
        this.taskChanged();this.formatChanged();this.restoreActions();
      }
      this.setStarters(catalog.starter_policies||[]);
    } catch(error) {
      this.$('runtime-status').textContent='Could not read the robotics catalog.';
      this.$('runtime-retry').hidden=false;this.error(error.message);
    } finally {this.loading=false;this.updateControls();}
  }
  setStarters(policies) {
    this.starterPolicies=Array.isArray(policies)?policies:[];
    const previous=this.selectedStarter,selected=previous?.id||'';
    const refreshed=selected?this.starterPolicies.find(item=>item.id===selected)||null:null;
    const changed=Boolean(previous)&&(!refreshed||this.starterContract(previous)!==this.starterContract(refreshed));
    this.selectedStarter=refreshed;
    this.setOptions('starter-policy',[{id:'',label:'Custom policy'},...this.starterPolicies],selected);
    if(selected&&!refreshed) {
      this.starterSignature=null;this.starterChanged=false;this.$('starter-policy').value='';
      this.$('starter-status').textContent='The selected starter is no longer available. Its current settings have been kept.';
    } else {
      if(changed)this.starterChanged=true;
      this.$('starter-status').textContent=refreshed?'Starter details refreshed. Current settings have been kept.':this.starterPolicies.length?'Choose a trained checkpoint to fill its task, simulator settings, and mapping.':'No trained starter policies are available. Refresh after training finishes.';
    }
    if(changed)this.invalidate();else this.renderStarterDetails();
  }
  starterContract(starter) {
    const canonical=value=>Array.isArray(value)?value.map(canonical):value&&typeof value==='object'?Object.fromEntries(Object.keys(value).sort().map(key=>[key,canonical(value[key])])):value;
    return JSON.stringify(canonical({checkpoint_sha256:starter.checkpoint_sha256,environment:starter.environment,policy:starter.policy,mapping:starter.mapping||{},instruction:starter.instruction||''}));
  }
  async refreshStarters() {
    if(this.busy||this.refreshingStarters||!this.catalog)return;
    this.refreshingStarters=true;this.$('starter-status').textContent='Checking for trained starter policies…';this.updateControls();
    try {const result=await this.api('/api/robotics/starter-policies');this.setStarters(result.policies||[]);}
    catch(error) {this.$('starter-status').textContent=`Could not refresh starter policies. ${error.message}`;}
    finally {this.refreshingStarters=false;this.updateControls();}
  }
  chooseOption(id,value) {
    const field=this.$(id);
    if(![...field.options].some(option=>option.value===value)) {
      const option=this.node('option',value);option.value=value;field.append(option);
    }
    field.value=value;
  }
  selectStarter() {
    const selected=this.starterPolicies.find(item=>item.id===this.$('starter-policy').value);
    this.selectedStarter=selected||null;this.starterSignature=null;this.starterChanged=false;
    if(!selected){this.updateControls();return;}
    const environment=selected.environment,policy=selected.policy;
    const knownTask=this.catalog.tasks?.some(task=>task.id===environment.task);
    this.$('task').value=knownTask?environment.task:'__custom__';
    this.taskChanged();this.$('custom-task').value=knownTask?'':environment.task;
    this.chooseOption('robot',environment.robots[0]);this.robotChanged();
    this.$('robots').value=environment.robots.join(', ');
    this.$('robots-fields').hidden=!(!knownTask||environment.task.startsWith('TwoArm')||environment.robots.length>1);
    this.chooseOption('controller',environment.controller);
    this.$('frequency').value=environment.control_freq;this.$('horizon').value=environment.horizon;
    this.$('policy-format').value=policy.format;this.formatChanged();
    this.$('policy-source').value=policy.source;this.sourceChanged();
    this.$('policy-name').value=policy.name;this.$('policy-revision').value=policy.revision||'';
    this.$('mapping').value=JSON.stringify(selected.mapping||{},null,2);
    this.$('instruction').value=selected.instruction||'';
    this.starterSignature=this.starterSnapshot();this.invalidate();
  }
  starterSnapshot() {
    return JSON.stringify(['task','custom-task','robots','controller','frequency','horizon','policy-format','policy-source','policy-name','policy-revision','mapping','instruction'].map(id=>this.$(id).value));
  }
  renderStarterDetails() {
    const selected=this.selectedStarter,details=this.$('starter-details');details.hidden=!selected;
    if(!selected){details.textContent='';return;}
    const evaluation=selected.evaluation||{},episodes=evaluation.episodes,successes=evaluation.successes;
    const measured=Number.isInteger(episodes)&&episodes>0&&Number.isInteger(successes)&&successes>=0&&successes<=episodes;
    const parts=[selected.description,`${selected.environment.task} · ${measured?`${successes}/${episodes} measured successes (${this.percent(successes/episodes)})`:'No measured evaluation results reported.'}`,selected.training?.method?`Training: ${selected.training.method}`:''];
    if(this.starterChanged)parts.unshift('Starter checkpoint or configuration updated. These results describe the refreshed starter. Current inputs have been kept and may differ; select the starter again to apply its updated configuration.');
    else if(this.starterSignature&&this.starterSignature!==this.starterSnapshot())parts.unshift('Settings edited. These recorded results describe the original starter configuration.');
    details.textContent=parts.filter(Boolean).join('\n');
  }
  taskChanged() {
    const task=this.$('task').value,custom=task==='__custom__';
    this.$('custom-task-fields').hidden=!custom;this.$('custom-task').required=custom;
    const preset=this.catalog?.tasks?.find(item=>item.id===task);
    this.$('instruction').value=preset?.instruction||'';
    this.$('robots-fields').hidden=!custom&&!task.startsWith('TwoArm');
    if(!custom)this.robotChanged();
  }
  robotChanged() {
    const robot=this.$('robot').value,task=this.$('task').value;
    this.$('robots').value=task.startsWith('TwoArm')&&robot!=='Baxter'?`${robot}, ${robot}`:robot;
  }
  formatChanged() {
    const format=this.$('policy-format').value;
    const sources=this.catalog?.policy_formats?.find(item=>item.id===format)?.sources||['local','upload'];
    const previous=this.$('policy-source').value;
    const names={local:'Server file or folder',upload:'Upload from this computer',huggingface:'Hugging Face (LeRobot)'};
    this.setOptions('policy-source',sources.map(id=>({id,label:names[id]||id})),sources.includes(previous)?previous:sources[0]);
    this.upload=null;this.$('policy-file').value='';this.$('policy-folder').value='';
    this.$('upload-status').textContent='Choose a policy to upload.';
    this.$('upload-progress').hidden=true;
    this.$('sb3-fields').hidden=format!=='sb3';
    this.$('policy-help').textContent={
      robomimic:'Use a Robomimic checkpoint trained for this task, robot, and controller. Saved environment metadata is checked.',
      torchscript:'Use an exported TorchScript policy with matching state inputs and controller actions. Configure its mapping below.',
      sb3:'Use a saved Stable-Baselines3 model with the matching algorithm, observation space, and controller actions.',
      lerobot:'Supports compatible LeRobot policies, including VLA policies. Include saved processor files and map camera, state, and action features below.'
    }[format]||'';
    this.sourceChanged();
  }
  sourceChanged() {
    const source=this.$('policy-source').value,format=this.$('policy-format').value,folder=format==='lerobot';
    this.$('policy-name-fields').hidden=source==='upload';
    this.$('policy-name').required=source!=='upload';
    this.$('policy-name-label').textContent=source==='huggingface'?'Hugging Face policy ID':folder?'Server policy folder':'Server checkpoint path';
    this.$('policy-name').placeholder=source==='huggingface'?'owner/saved-policy':folder?'/path/to/lerobot-policy':'/path/to/policy-checkpoint';
    this.$('policy-revision-fields').hidden=source!=='huggingface';
    this.$('policy-upload-fields').hidden=source!=='upload';
    this.$('checkpoint-file-fields').hidden=folder;
    this.$('policy-folder-fields').hidden=!folder;
    this.$('policy-file').accept=({robomimic:'.pth,.pt,.ckpt',torchscript:'.pt,.pth,.jit,.torchscript,.ckpt',sb3:'.zip'}[format]||'')+',.json,.pkl';
    this.updateControls();
  }
  restoreActions() {
    this.groups=Object.fromEntries(Object.entries(this.catalog?.default_action_space||{}).map(([key,values])=>[key,[...values]]));
    this.editingGroup=null;this.renderActions();
  }
  actionCount() {
    const values=Object.values(this.groups);
    return values.length?values.reduce((count,group)=>count*group.length,1):0;
  }
  actionError() {
    const count=this.actionCount(),limit=this.catalog?.limits?.max_actions||5000;
    if(!count)return 'Add a variation group and select at least one value.';
    if(count>limit)return `${this.number(count)} combinations exceed the ${this.number(limit)} limit. Select fewer values or groups.`;
    return '';
  }
  renderActions() {
    const chips=this.$('action-chips'),editor=this.$('action-editor');chips.replaceChildren();editor.replaceChildren();
    for(const [key,values] of Object.entries(this.groups)) {
      const chip=this.node('div',undefined,'action-chip'),edit=this.node('button',undefined,'action-chip-label');
      edit.type='button';edit.id=`robotics-edit-${key}`;edit.textContent=`${this.groupName(key)} (${values.length}) ⌄`;
      edit.setAttribute('aria-expanded',String(this.editingGroup===key));edit.setAttribute('aria-controls','robotics-action-editor');
      edit.addEventListener('click',()=>{this.editingGroup=this.editingGroup===key?null:key;this.renderActions();this.$(`edit-${key}`).focus();});
      const remove=this.node('button','×','action-chip-remove');remove.type='button';remove.id=`robotics-remove-${key}`;
      remove.setAttribute('aria-label',`Remove ${this.groupName(key)}`);
      remove.addEventListener('click',()=>{delete this.groups[key];if(this.editingGroup===key)this.editingGroup=null;this.invalidate();this.renderActions();this.$('action-add').focus();});
      chip.append(edit,remove);chips.append(chip);
    }
    const available=(this.catalog?.action_groups||[]).filter(group=>!Object.hasOwn(this.groups,group.id));
    this.setOptions('action-add',[{id:'',label:available.length?'＋ Add variation group…':'All groups added'},...available]);
    this.$('action-add').disabled=!available.length;
    if(this.editingGroup&&this.groups[this.editingGroup]) {
      const key=this.editingGroup,metadata=this.metadata(key),values=metadata?.values||[],selected=this.groups[key];
      const title=this.node('p',`${this.groupName(key)} values`,'robotics-value-title');editor.append(title);
      const options=this.node('div',undefined,'robotics-value-options');options.setAttribute('role','group');options.setAttribute('aria-label',`${this.groupName(key)} values`);
      values.forEach(value=>{
        const label=this.node('label',undefined,'robotics-value-option'),checkbox=this.node('input');
        checkbox.type='checkbox';checkbox.checked=selected.includes(value);
        checkbox.addEventListener('change',()=>{
          this.groups[key]=values.filter(item=>item===value?checkbox.checked:this.groups[key].includes(item));
          this.$(`edit-${key}`).textContent=`${this.groupName(key)} (${this.groups[key].length}) ⌄`;this.invalidate();
        });
        label.append(checkbox,document.createTextNode(`${value}${value===metadata.neutral?' (neutral)':''}`));options.append(label);
      });
      editor.append(options);
      if(metadata?.description)editor.append(this.node('p',metadata.description,'hint action-guidance'));
    }
    editor.hidden=!this.editingGroup;this.updateControls();
  }
  invalidate() {this.revision++;this.updateControls();}
  updateControls() {
    const available=this.catalog?.runtime?.available!==false&&Boolean(this.catalog),busy=this.busy;
    const count=this.actionCount(),actionError=this.actionError();
    this.$('config').disabled=this.loading||busy||!this.catalog;
    this.$('action-count').textContent=`${this.number(count)} combination${count===1?'':'s'} · neutral settings may reuse baseline measurements`;
    this.$('action-error').textContent=actionError;this.$('action-error').hidden=!this.catalog||!actionError;
    const policyReady=this.$('policy-source').value==='upload'?Boolean(this.upload):Boolean(this.$('policy-name').value.trim());
    this.$('analyze').disabled=!available||this.loading||busy||Boolean(actionError)||!policyReady;
    this.$('analyze').textContent=this.submitting&&this.submittingMode==='analyze'?'Starting…':this.isRunning(this.job)&&this.job.mode==='analyze'?'Analyzing…':'Analyze';
    this.$('load-environment').disabled=!available||this.loading||busy;
    this.$('load-environment').textContent=this.submitting&&this.submittingMode==='environment'?'Starting…':this.isRunning(this.job)&&this.job.mode==='environment'?'Loading environment…':'Load environment';
    this.$('refresh-starters').disabled=this.loading||busy||this.refreshingStarters||!this.catalog;
    this.$('starter-policy').disabled=this.refreshingStarters||!this.starterPolicies.length;
    const ready=this.analysis?.status==='completed'&&this.analysisRevision===this.revision;
    const baseline=this.analysis?.baseline;
    const baselineCount=Array.isArray(baseline)?this.measured(baseline).length:baseline?.episodes||this.measured(baseline?.records||[]).length;
    this.$('discover').disabled=busy||!available||!ready||!baselineCount||this.job?.mode==='environment';
    this.$('discover').textContent=this.isRunning(this.job)&&this.job.mode==='discover'?'Finding failures…':'Find failure';
    this.$('start-search').disabled=this.$('discover').disabled;
    this.$('search-fields').disabled=busy;
    if(this.job?.mode==='environment')this.closeSearch();
    this.$('cancel').hidden=!(this.uploading||this.cancelling||this.isRunning(this.job));
    this.$('cancel').disabled=Boolean(this.cancelling);
    this.$('cancel').textContent=this.uploading?'Cancel upload':this.cancelling?'Cancelling…':'Cancel';
    this.$('stale').hidden=!this.analysis||this.analysisRevision===this.revision;
    this.$('runtime-retry').disabled=this.loading||busy;
    this.$('show-evaluation').hidden=this.job?.mode!=='environment'||!this.evaluationJob;
    this.$('show-evaluation').disabled=busy;
    this.renderStarterDetails();this.renderEnvironment();
    this.onChange();
  }

  async uploadFiles(files,directory) {
    if(!files.length||this.busy)return;
    this.invalidate();this.error();this.upload=null;
    const format=this.$('policy-format').value;
    const checkpoints=files.filter(file=>!/\.(json|pkl)$/i.test(file.name));
    if(!directory&&checkpoints.length!==1){this.error('Choose exactly one native checkpoint, plus its optional metadata JSON and normalization statistics files.');return;}
    const commonRoot=directory?files[0].webkitRelativePath.split('/')[0]:null;
    const manifest=files.map(file=>{
      let path=directory?file.webkitRelativePath:file.name;
      if(directory&&path.startsWith(commonRoot+'/'))path=path.slice(commonRoot.length+1);
      return {path,size:file.size};
    });
    if(manifest.some(file=>!file.path||file.path.startsWith('/')||file.path.includes('\\')||file.path.split('/').some(part=>!part||part==='.'||part==='..'))) {
      this.error('The selected folder contains an invalid relative path.');return;
    }
    const total=files.reduce((sum,file)=>sum+file.size,0),generation=++this.uploadGeneration;
    this.uploading=true;this.uploadAbort=new AbortController();this.$('upload-progress').hidden=false;this.$('upload-progress').value=0;
    this.$('upload-status').textContent=`Preparing ${files.length} file${files.length===1?'':'s'}…`;this.updateControls();
    try {
      const info=await this.api('/api/robotics/policies',{format,files:manifest,...(!directory?{primary:checkpoints[0].name}:{})});
      const chunkBytes=Math.min(8*1024*1024,Math.max(1,Number(info.chunk_bytes)||8*1024*1024));
      let uploaded=0;
      for(let index=0;index<files.length;index++) {
        for(let offset=0;offset<files[index].size;offset+=chunkBytes) {
          if(generation!==this.uploadGeneration)throw new DOMException('Upload cancelled','AbortError');
          const chunk=files[index].slice(offset,Math.min(offset+chunkBytes,files[index].size));
          const response=await fetch(`/api/robotics/policies/${encodeURIComponent(info.id)}/files/${index}?offset=${offset}`,{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:chunk,signal:this.uploadAbort.signal});
          const result=await response.json();
          if(!response.ok)throw new Error(result.error||`Upload failed (${response.status})`);
          uploaded+=chunk.size;this.$('upload-progress').value=total?100*uploaded/total:100;
          this.$('upload-status').textContent=`Uploading ${manifest[index].path} · ${(uploaded/1048576).toFixed(1)} / ${(total/1048576).toFixed(1)} MiB`;
        }
      }
      if(generation!==this.uploadGeneration)return;
      const finished=await this.api(`/api/robotics/policies/${encodeURIComponent(info.id)}/finish`,{});
      if(generation!==this.uploadGeneration)return;
      if(finished.status!=='ready')throw new Error(finished.error||'The uploaded policy could not be finalized.');
      this.upload={id:finished.id||info.id,format};this.$('upload-progress').value=100;
      this.$('upload-status').textContent=`Ready · ${files.length} file${files.length===1?'':'s'} · ${(total/1048576).toFixed(1)} MiB`;
    } catch(error) {
      if(generation!==this.uploadGeneration||error.name==='AbortError')return;
      this.$('upload-status').textContent='Upload failed. Choose the policy again to retry.';this.error(error.message);
    } finally {
      if(generation===this.uploadGeneration){this.uploading=false;this.uploadAbort=null;this.updateControls();}
      this.$(directory?'policy-folder':'policy-file').value='';
    }
  }
  numeric(id,label) {
    const field=this.$(id),value=Number(field.value);
    if(!field.value.trim()||!Number.isInteger(value)||value<Number(field.min)||(field.max&&value>Number(field.max)))throw new Error(`${label} must be an integer between ${field.min} and ${field.max||'the supported limit'}.`);
    return value;
  }
  environmentConfig() {
    const task=this.$('task').value==='__custom__'?this.$('custom-task').value.trim():this.$('task').value;
    const robots=this.$('robots').value.split(',').map(value=>value.trim()).filter(Boolean);
    if(!task)throw new Error('Choose or enter an installed Robosuite task.');
    if(!robots.length)throw new Error('Choose at least one robot.');
    const controller=this.$('controller').value;if(!controller)throw new Error('Choose a robot controller.');
    return {mode:'environment',environment:{task,robots,controller,control_freq:this.numeric('frequency','Control frequency'),horizon:this.numeric('horizon','Episode step limit')},seed:this.numeric('seed','Seed')};
  }
  environmentSnapshot() {
    try {return JSON.stringify(this.environmentConfig());}catch{return null;}
  }
  config() {
    let mapping;
    try {mapping=JSON.parse(this.$('mapping').value||'{}');}
    catch {throw new Error('Observation and action mapping must be valid JSON. Open “Observation and action mapping” to correct it.');}
    if(!mapping||typeof mapping!=='object'||Array.isArray(mapping))throw new Error('Adapter mapping must be a JSON object.');
    const source=this.$('policy-source').value,format=this.$('policy-format').value;
    const policy={format,source};
    if(source==='upload') {
      if(!this.upload||this.upload.format!==format)throw new Error('Upload a policy before analyzing.');
      policy.upload_id=this.upload.id;
    } else {
      policy.name=this.$('policy-name').value.trim();if(!policy.name)throw new Error('Enter a policy path or Hugging Face policy ID.');
      if(source==='huggingface'&&this.$('policy-revision').value.trim())policy.revision=this.$('policy-revision').value.trim();
    }
    if(format==='sb3')policy.algorithm=this.$('sb3-algorithm').value;
    const environment=this.environmentConfig();
    const actionError=this.actionError();if(actionError)throw new Error(actionError);
    return {...environment,mode:'analyze',policy,mapping,
      instruction:this.$('instruction').value.trim(),action_space:Object.fromEntries(Object.entries(this.groups).map(([key,values])=>[key,[...values]])),episodes_per_variation:this.numeric('episodes','Baseline episodes'),variation_budget:this.numeric('budget','Variation budget'),device:this.$('device').value,record_video:this.$('record-video').checked};
  }
  async loadEnvironment() {
    if(this.busy||!this.catalog||this.catalog.runtime?.available===false)return;
    this.error();
    try {await this.start(this.environmentConfig());}catch(error){this.error(error.message);}
  }
  async analyze() {
    if(this.busy||!this.catalog||this.catalog.runtime?.available===false)return;
    this.error();
    try {await this.start(this.config());}catch(error){this.error(error.message);}
  }
  discover() {
    if(this.$('discover').disabled||this.busy)return;
    this.error();this.$('search-error').hidden=true;this.$('search-panel').hidden=false;
    this.$('discover').setAttribute('aria-expanded','true');this.$('search-episodes').focus({preventScroll:true});
    this.$('search-panel').scrollIntoView({block:'nearest'});
  }
  searchSettings() {
    const decimal=(id,label)=>{
      const field=this.$(id),value=Number(field.value);
      if(!field.value.trim()||!Number.isFinite(value)||value<Number(field.min)||value>Number(field.max))throw new Error(`${label} must be between ${field.min} and ${field.max}.`);
      return value;
    };
    const settings={episodes:this.numeric('search-episodes','RL steps'),rollout_budget:this.numeric('search-rollouts','New rollout budget'),
      epsilon_start:decimal('search-epsilon-start','Starting exploration'),epsilon_end:decimal('search-epsilon-end','Final exploration'),
      learning_rate:decimal('search-learning-rate','Learning rate'),batch_size:this.numeric('search-batch-size','Batch size'),
      replay_capacity:this.numeric('search-replay-capacity','Replay capacity'),seed:this.numeric('search-seed','Search seed'),criterion:this.$('search-criterion').value};
    if(settings.epsilon_end>settings.epsilon_start)throw new Error('Final exploration must not exceed starting exploration.');
    if(settings.replay_capacity<settings.batch_size)throw new Error('Replay capacity must be at least the batch size.');
    if(!['task_failure','new_failure'].includes(settings.criterion))throw new Error('Choose a reward criterion.');
    return settings;
  }
  async startDiscovery() {
    if(this.$('start-search').disabled||this.busy)return;
    this.$('search-error').hidden=true;this.error();
    try {await this.start({mode:'discover',analysis_id:this.analysis.id,search:this.searchSettings()});}
    catch(error){this.$('search-error').textContent=error.message;this.$('search-error').hidden=false;}
  }
  async start(config) {
    const generation=++this.runGeneration,revision=this.revision;clearTimeout(this.pollTimer);
    this.renderPolicyNotes(null);
    this.submitting=true;this.submittingMode=config.mode;this.cancelling=false;this.updateControls();
    try {
      const job=await this.api('/api/robotics/runs',config);
      if(generation!==this.runGeneration)return;
      if(config.mode==='analyze'){this.analysisRevision=revision;this.selectedId=null;this.selectedInput=null;this.graphDataset=null;this.graphPopulated=false;this.graph.reset();}
      if(config.mode==='discover')this.closeSearch();
      if(config.mode==='environment')this.environmentSignature=JSON.stringify(config);
      this.applyJob(job);
      if(this.isRunning(job))this.schedulePoll(generation);
    } catch(error) {if(generation===this.runGeneration){this.error(error.message);if(config.mode==='environment')this.applyJob({mode:'environment',status:'error',message:'Environment load failed.',error:error.message});}}
    finally {if(generation===this.runGeneration){this.submitting=false;this.submittingMode=null;this.updateControls();}}
  }
  schedulePoll(generation,delay=800) {clearTimeout(this.pollTimer);this.pollTimer=setTimeout(()=>this.poll(generation),delay);}
  async poll(generation) {
    if(generation!==this.runGeneration||!this.isRunning(this.job))return;
    const id=this.job.id;
    try {
      const job=await this.api(`/api/robotics/runs/${encodeURIComponent(id)}`);
      if(generation!==this.runGeneration||id!==this.job?.id||!this.isRunning(this.job))return;
      this.applyJob(job);if(this.isRunning(this.job))this.schedulePoll(generation);
    } catch(error) {
      if(generation!==this.runGeneration)return;
      this.$('status').textContent=`Connection interrupted; retrying the running job. ${error.message}`;
      this.schedulePoll(generation,2000);
    }
  }
  async cancel() {
    if(this.uploading) {
      this.uploadGeneration++;this.uploadAbort?.abort();this.uploadAbort=null;this.uploading=false;this.upload=null;
      this.$('upload-status').textContent='Upload cancelled. Choose the policy again to retry.';this.updateControls();return;
    }
    if(!this.isRunning(this.job)||this.cancelling)return;
    const id=this.job.id;let generation=this.runGeneration;
    const current=()=>id===this.job?.id&&generation===this.runGeneration;
    this.cancelling=true;this.updateControls();
    try {
      const result=await this.api(`/api/robotics/runs/${encodeURIComponent(id)}/cancel`,{});
      if(!current())return;
      generation=++this.runGeneration;clearTimeout(this.pollTimer);
      this.applyJob({...this.job,...result});if(this.isRunning(this.job))this.schedulePoll(generation);
    } catch(error) {if(current())this.error(`Could not cancel the worker: ${error.message}`);}
    finally {if(current()){this.cancelling=false;this.updateControls();}}
  }

  applyJob(job) {
    this.job=job;if(job.mode==='analyze')this.analysis=job;
    const environment=job.mode==='environment';
    if(environment)this.environmentJob=job;else this.evaluationJob=job;
    this.reusedIds=new Set((job.records||[]).filter(record=>record.reused_baseline).map(record=>String(record.action_id)));
    this.$('status').textContent=job.message||`${environment?'Environment':job.mode==='analyze'?'Baseline analysis':job.search?'DQN discovery':'Variation sweep'} ${job.status}.`;
    this.$('results-title').textContent=environment?'Environment preview':'Policy episode landscape';
    this.$('results-subtitle').textContent=environment?'Reset scene · no policy evaluation':'One original per episode · expand to inspect environment variations';
    const parts=[];
    if(job.phase)parts.push(job.phase.replaceAll('_',' '));
    if(!environment&&Number.isFinite(job.completed)&&Number.isFinite(job.total))parts.push(`${job.completed} / ${job.total} episodes`);
    if(!environment&&Number.isFinite(job.episode))parts.push(`episode ${job.episode}`);
    if(!environment&&Number.isFinite(job.step))parts.push(`step ${job.step}${job.horizon?` / ${job.horizon}`:''}`);
    this.$('step-status').textContent=parts.join(' · ');
    this.$('progress').hidden=!this.isRunning(job);this.$('progress').value=job.search?.episodes?100*(job.search.episode||0)/job.search.episodes:job.total?100*job.completed/job.total:0;
    if(job.error)this.error(typeof job.error==='string'?job.error:JSON.stringify(job.error));
    this.$('search-space').hidden=environment;
    this.renderPolicyNotes(environment?null:job);this.renderSearchProgress();
    if(environment) {
      for(const id of ['metrics','baseline-errors','inspector','live'])this.$(id).hidden=true;
    } else {this.renderMetrics();this.renderGraph();this.renderPicker();this.renderInspection();this.renderLive();}
    this.renderArtifacts();this.updateControls();
  }
  renderEnvironment() {
    const job=this.environmentJob,info=job?.status==='completed'?job.environment_info:null;
    const stale=Boolean(info)&&this.environmentSignature!==this.environmentSnapshot();
    const pending=this.submitting&&this.submittingMode==='environment';
    this.$('environment-status').textContent=pending?'Starting environment…':this.isRunning(job)?'Loading the configured scene…':info?stale?'Settings changed. Load again to update the scene.':'Environment loaded. Choose a policy and Analyze to measure task success.':job?.status==='error'?'Environment load failed. Check the error and retry.':job?.status==='cancelled'?'Environment load cancelled.':'Preview this scene before choosing a policy.';
    const visible=Boolean(info)&&this.job?.mode==='environment'&&!pending;
    this.$('environment-summary').hidden=!visible;this.$('environment-stale').hidden=!stale;
    const preview=visible?this.artifactUrl(job?.artifacts?.preview):null;
    this.$('environment-preview').hidden=!preview;
    if(!info)return;
    this.$('environment-details').textContent=`${info.task} · ${(info.robots||[]).join(', ')} · ${info.controller} · ${info.action_dimension} action dimensions · ${info.control_freq} Hz · ${info.horizon} steps · seed ${info.seed}`;
    if(preview&&this.$('environment-image').getAttribute('src')!==preview)this.$('environment-image').src=preview;
    this.$('environment-caption').textContent=`${info.preview_camera||'Scene'} camera · initial environment state`;
    const signature=JSON.stringify([info.observation_shapes,info.cameras]);
    if(signature!==this.environmentMetadataSignature) {
      const details=this.$('environment-observations');details.replaceChildren(this.node('dt','Cameras'),this.node('dd',(info.cameras||[]).join(', ')||'None reported'));
      for(const [key,shape] of Object.entries(info.observation_shapes||{}))details.append(this.node('dt',key),this.node('dd',Array.isArray(shape)?`[${shape.join(', ')}]`:String(shape)));
      this.environmentMetadataSignature=signature;
    }
    if(visible&&this.active&&this.scrolledEnvironmentId!==job.id&&window.matchMedia('(max-width: 900px)').matches) {
      this.scrolledEnvironmentId=job.id;this.$('environment-summary').scrollIntoView({block:'start'});
    }
  }
  renderPolicyNotes(job) {
    const list=value=>Array.isArray(value)?value:typeof value==='string'?[value]:[];
    const seen=new Set(),warnings=[];
    for(const warning of [...list(job?.warnings),...list(job?.policy_info?.warnings)]) {
      const raw=typeof warning==='string'?warning:warning?.message;
      if(typeof raw!=='string')continue;
      const message=raw.trim().replace(/\s+/g,' ');
      if(!message||seen.has(message))continue;
      seen.add(message);warnings.push(message);
    }
    const details=this.$('policy-notes'),id=job?.id||null;
    if(this.policyNotesRunId!==id||!warnings.length)details.open=false;
    this.policyNotesRunId=id;details.hidden=!warnings.length;
    this.$('policy-notes-title').textContent=`Policy notes (${warnings.length})`;
    const signature=JSON.stringify(warnings);
    if(signature!==this.policyNotesSignature) {
      this.$('policy-notes-list').replaceChildren(...warnings.map(message=>this.node('li',message)));
      this.policyNotesSignature=signature;
    }
  }
  renderMetrics() {
    const baseline=this.baselineSummary(),records=this.job?.records||[],summary=this.summary(records.filter(record=>!record.reused_baseline));
    this.$('metrics').hidden=!this.job;
    this.$('baseline-rate').textContent=this.percent(baseline.success_rate);
    this.$('baseline-detail').textContent=`${baseline.episodes||0} measured episode${baseline.episodes===1?'':'s'}${baseline.errors?` · ${baseline.errors} error${baseline.errors===1?'':'s'}`:''}`;
    this.$('failure-rate').textContent=this.percent(summary.failure_rate);
    const tested=(this.job?.nodes||[]).filter(node=>node.episodes>0&&!this.reusedIds.has(String(node.id))).length,reused=records.filter(record=>record.reused_baseline).length;
    this.$('failure-detail').textContent=(summary.episodes?`${summary.episodes} measured episodes · ${tested} measured variations`:'No variation episodes measured yet')+(reused?` · ${reused} baseline reuses excluded`:'');
    const regressions=(this.job?.nodes||[]).reduce((sum,node)=>sum+(Number(node.regressions)||0),0),recoveries=(this.job?.nodes||[]).reduce((sum,node)=>sum+(Number(node.recoveries)||0),0);
    this.$('paired').textContent=`${regressions} new failures · ${recoveries} recoveries`;
    const errors=(baseline.errors||0)+summary.errors;
    this.$('error-count').textContent=`Matched seeds · ${errors} rollout error${errors===1?'':'s'} excluded from rates`;
    const failed=this.baselineRecords().filter(record=>record.error);
    this.$('baseline-errors').hidden=!failed.length;
    this.$('baseline-errors').open=failed.length>0&&!baseline.episodes;
    this.$('baseline-error-title').textContent=`Baseline rollout errors (${failed.length})`;
    this.$('baseline-error-list').replaceChildren(...failed.map(record=>this.node('li',`Seed ${record.seed??'—'}: ${record.error}`)));
  }
  renderSearchProgress() {
    const search=this.job?.mode==='discover'?this.job.search:null;
    this.$('search-progress').hidden=!search;
    if(!search)return;
    this.$('search-steps').textContent=`${this.number(search.episode||0)} / ${this.number(search.episodes)}`;
    this.$('search-model-calls').textContent=`${this.number(search.model_calls||0)} / ${this.number(search.rollout_budget)}`;
    this.$('search-cache-hits').textContent=this.number(search.cache_hits||0);
    this.$('search-epsilon').textContent=Number.isFinite(search.epsilon)?Number(search.epsilon.toFixed(3)).toString():'—';
    this.$('search-loss').textContent=Number.isFinite(search.loss)?Number(search.loss.toPrecision(4)).toString():'—';
    this.$('search-updates').textContent=this.number(search.updates||0);
    const current=this.job.current,point=this.episodePoints().find(point=>point.index===current?.input_index);
    const action=(this.job.nodes||[]).find(node=>String(node.id)===String(current?.action_id));
    this.$('search-current').textContent=current?`${this.isRunning(this.job)?'Current':'Last'} pair: ${point?.name||`episode ${Number(current.input_index)+1}`} · ${action?.label||`variation ${Number(current.action_id)+1}`}`:'';
  }
  episodeEmbedding() {return this.job?.embedding||(this.job?.mode==='discover'?this.analysis?.embedding:null);}
  episodePoints() {
    return (this.episodeEmbedding()?.points||[]).filter(point=>Number.isInteger(point.index)&&Array.isArray(point.position)&&point.position.length===3&&point.position.every(Number.isFinite));
  }
  selectedPoint() {return this.episodePoints().find(point=>point.index===this.selectedInput);}
  pairRecord(point,actionId) {
    const records=this.job?.records;
    if(!point||!Array.isArray(records))return null;
    if(this.recordIndexRecords!==records) {
      this.recordIndexRecords=records;this.recordIndex=new Map();
      for(const record of records)this.recordIndex.set(Number.isInteger(record.input_index)?`input:${record.input_index}:${record.action_id}`:`seed:${record.seed}:${record.action_id}`,record);
    }
    return this.recordIndex.get(`input:${point.index}:${actionId}`)||this.recordIndex.get(`seed:${point.seed}:${actionId}`)||null;
  }
  baselineRecord(point) {
    if(!point)return null;
    const record=this.baselineRecords().find(record=>String(record.seed)===String(point.seed));
    return {...point,...record};
  }
  pairState(record) {return record?.error?'error':typeof record?.success==='boolean'?(record.success?'success':'failure'):'untested';}
  nodeLabel(node,point=this.selectedPoint()) {
    const record=this.pairRecord(point,node.id),state=this.pairState(record);
    return `${node.label||`Variation ${Number(node.id)+1}`} · ${state==='error'?'rollout error':state==='untested'?'untested':`observed ${state}`}${record?.reused_baseline?' · baseline reused':''}`;
  }
  expansion(point) {
    const nodes=this.job?.nodes||[],points=this.episodePoints();
    const spread=Math.max(...[0,1,2].map(axis=>Math.max(...points.map(item=>item.position[axis]))-Math.min(...points.map(item=>item.position[axis]))))/2||1;
    return nodes.map((node,index)=>{
      const record=this.pairRecord(point,node.id),measured=record?.embedding?.position;
      const position=Array.isArray(measured)&&measured.length===3&&measured.every(Number.isFinite)?measured:
        record?.reused_baseline&&!record.error?point.position:null;
      // This deterministic placeholder arrangement is never labeled as PCA.
      const angle=index*Math.PI*(3-Math.sqrt(5)),radius=spread*(.24+.38*Math.sqrt((index+1)/Math.max(1,nodes.length)));
      return {id:node.id,action:node.values||{},seed:point.seed,success:record?.success,error:record?.error,
        position:position||[point.position[0]+radius*Math.cos(angle),point.position[1]+radius*Math.sin(angle),point.position[2]],
        pending:!position,similarity:Number.isFinite(record?.embedding?.similarity)?record.embedding.similarity:null};
    });
  }
  graphDiscovery(points) {
    const bySeed=new Map(points.map(point=>[String(point.seed),point.index]));
    const records=(this.job?.records||[]).filter(record=>!record.error&&typeof record.success==='boolean').map(record=>({...record,
      input_index:Number.isInteger(record.input_index)?record.input_index:bySeed.get(String(record.seed)),severity:Number(!record.success)}));
    return {...this.job,records:records.filter(record=>Number.isInteger(record.input_index))};
  }
  renderGraph() {
    const points=this.episodePoints(),embedding=this.episodeEmbedding(),key=this.job?.mode==='discover'?(this.job.analysis_id||this.analysis?.id):this.job?.id;
    if(this.landscapeJob!==this.job?.id){this.landscapeJob=this.job?.id;this.landscapeChoice=null;this.predictionRequestKey=null;}
    // Start with observed baselines; learned estimates cover all actions once
    // trained. An explicit user choice survives subsequent progress polls.
    this.$('landscape-mode').value=this.landscapeChoice||(this.job?.predicted_landscape?.policy_updates>0?'predicted':'measured');
    this.$('graph').hidden=!points.length;this.$('graph-empty').hidden=Boolean(points.length);
    for(const id of ['graph-fit','graph-zoom-in','graph-zoom-out','landscape-mode'])this.$(id).disabled=!points.length;
    if(this.graphDataset!==key||(!this.graphPopulated&&points.length)) {
      this.graph.setData(points);this.graphDataset=key;this.graphPopulated=Boolean(points.length);
    } else this.graph.points=points;
    if(this.selectedInput!==null&&!points.some(point=>point.index===this.selectedInput)){this.selectedInput=null;this.selectedId=null;this.graph.clearFocus();}
    this.graph.setLandscapeMode(this.$('landscape-mode').value);
    this.graph.setDiscovery(this.graphDiscovery(points));
    const point=this.selectedPoint();
    if(point) {
      if(this.graph.focusIndex!==point.index)this.graph.focus(point.index);
      this.graph.setExpansion(point.index,this.expansion(point));
      const action=(this.job.nodes||[]).find(node=>String(node.id)===this.selectedId);
      if(this.selectedId!==null&&!action)this.selectedId=null;
      this.graph.selectAction(action?.id??null);
    }
    const measured=(this.job?.records||[]).filter(record=>!record.error&&typeof record.success==='boolean').length;
    this.$('graph-count').textContent=points.length?`${this.number(points.length)} original${points.length===1?'':'s'} · ${this.number(this.job?.nodes?.length||0)} variations / episode${measured?` · ${this.number(measured)} measured pairs`:''}`:'No episode features available';
    const source=embedding?.source||{},kind=String(source.kind||''),layer=source.source_layer?` · ${source.source_layer}`:'';
    const featureName=/observation/.test(kind)?'Observation features':/policy|hidden/.test(kind)?'Policy features':kind?`${kind.replaceAll('_',' ')} features`:'Episode features';
    const method=embedding?.projection?.method||'PCA',projectionName=/^PCA/i.test(method)?'PCA':method;
    this.$('coordinate-note').textContent=points.length?`${featureName}${layer} · ${projectionName} · episode means`:'Analyze to capture episode features. Older runs need a new analysis.';
    this.$('coordinate-note').title=`${method}. Each original pools fresh decision features from one baseline episode. Measured variations use the same frozen baseline projection; untested variations use an explicit pending layout.`;
    this.$('size-note').hidden=!this.job?.policy_statistics;
    this.renderLandscapeNote();this.loadPredictions();
  }
  fitGraph() {this.graph.fit();}
  renderLandscapeNote(message) {
    const predicted=this.$('landscape-mode').value==='predicted';
    this.$('observed-legend').hidden=predicted;this.$('estimate-legend').hidden=!predicted;
    this.$('landscape-note').textContent=message||(predicted?'DQN expected-reward estimates · not calibrated failure probabilities.':'Green: task success · red: task failure · gray: untested.');
  }
  async loadPredictions() {
    if(this.$('landscape-mode').value!=='predicted')return;
    const job=this.job,point=this.selectedPoint(),updates=job?.predicted_landscape?.policy_updates;
    if(!updates){this.renderLandscapeNote('DQN estimates unavailable until the first policy update.');return;}
    if(!point){this.renderLandscapeNote('Originals: mean Q · select an episode for action estimates.');return;}
    const key=`${job.id}:${point.index}:${updates}`;
    if(this.predictionRequestKey===key)return;
    this.predictionRequestKey=key;
    try {
      const result=await this.api(`/api/robotics/runs/${encodeURIComponent(job.id)}/predictions?input_index=${point.index}`);
      if(this.predictionRequestKey!==key||this.job?.id!==job.id||this.selectedInput!==point.index||this.job?.predicted_landscape?.policy_updates!==updates)return;
      this.graph.setPredictions({...result,discovery_id:job.id});
      this.renderInspection();
      if(result.available===false){this.predictionRequestKey=null;this.renderLandscapeNote('DQN estimates are not available yet.');}
    } catch(error) {if(this.predictionRequestKey===key){this.predictionRequestKey=null;this.renderLandscapeNote(`DQN estimates unavailable. ${error.message}`);}}
  }
  renderPicker() {
    const points=this.episodePoints(),point=this.selectedPoint();this.$('inspector').hidden=!points.length;
    const originals=points.map(item=>({id:String(item.index),label:`${item.name||`Episode ${item.index+1}`} · ${this.pairState(item)}`}));
    const episodeSignature=JSON.stringify(originals);
    if(episodeSignature!==this.episodePickerSignature){this.setOptions('episode',[{id:'',label:'Select an episode…'},...originals],this.selectedInput===null?'':String(this.selectedInput));this.episodePickerSignature=episodeSignature;}
    this.$('episode').value=this.selectedInput===null?'':String(this.selectedInput);
    const priority={failure:0,success:1,error:2,untested:3};
    const nodes=point?[...(this.job?.nodes||[])].sort((a,b)=>priority[this.pairState(this.pairRecord(point,a.id))]-priority[this.pairState(this.pairRecord(point,b.id))]||Number(a.id)-Number(b.id)):[];
    const options=nodes.map(node=>({id:String(node.id),label:this.nodeLabel(node,point)})),signature=JSON.stringify(options);
    // Leave an open native picker alone during step-only progress updates.
    if(signature!==this.pickerSignature){this.setOptions('inspect',[{id:'',label:'Baseline episode'},...options],this.selectedId??'');this.pickerSignature=signature;}
    this.$('inspect').value=this.selectedId??'';this.$('inspect').disabled=!point;
  }
  selectEpisode(index) {
    this.selectedInput=index===''?null:Number(index);this.selectedId=null;this.predictionRequestKey=null;
    if(this.selectedInput===null)this.graph.clearFocus();else this.graph.focus(this.selectedInput);
    this.renderGraph();this.renderPicker();this.renderInspection();
  }
  selectNode(id) {this.selectedId=id===''?null:String(id);this.renderGraph();this.renderPicker();this.renderInspection();}
  artifactUrl(value) {
    const url=typeof value==='string'?value:value?.url;
    return typeof url==='string'&&/^\/robotics\/[^/]+\/artifacts\/[^/]+(?:\?.*)?$/.test(url)?url:null;
  }
  renderMedia(id,record,empty) {
    const container=this.$(id),video=this.artifactUrl(record?.video),image=this.artifactUrl(record?.image),url=video||image;
    const key=url||empty;if(container.dataset.media===key)return;container.dataset.media=key;container.replaceChildren();
    if(!url){container.append(this.node('p',empty,'hint'));return;}
    const isVideo=video&&!/\.(gif|png|jpe?g|webp)(?:\?|$)/i.test(video);
    const media=this.node(isVideo?'video':'img');media.src=url;
    if(isVideo){media.controls=true;media.preload='metadata';media.playsInline=true;}
    else media.alt=video?'Representative episode replay':'Recorded episode frame';
    media.addEventListener('error',()=>{if(container.contains(media)){media.remove();container.prepend(this.node('p','This replay could not be loaded. Open the saved artifact to inspect it.','hint'));}});
    const link=this.node('a',video?'Open replay':'Open frame');link.href=url;link.target='_blank';link.rel='noopener noreferrer';link.className='report-link';container.append(media,link);
  }
  renderInspection() {
    const point=this.selectedPoint(),selected=(this.job?.nodes||[]).find(node=>String(node.id)===this.selectedId);
    this.$('selection').hidden=!point;if(!point)return;
    const baseline=this.baselineRecord(point),record=selected?this.pairRecord(point,selected.id):baseline,state=this.pairState(record);
    this.$('selected-title').textContent=selected?`${selected.label||`Variation ${Number(selected.id)+1}`} · seed ${point.seed}`:`${point.name||`Episode ${point.index+1} · seed ${point.seed}`} · baseline`;
    this.$('selected-values').textContent=selected?Object.entries(selected.values||{}).map(([key,value])=>`${this.groupName(key)}: ${value}`).join(' · '):'Original episode · variations expanded in the graph';
    this.$('selected-outcome').textContent=state==='untested'?'Untested: no observed task outcome.':state==='error'?'Rollout error: no task success or failure was measured.':`Observed task ${state}${Number.isFinite(record?.steps)?` · ${record.steps} steps`:''}`;
    const notes=[];
    if(this.$('landscape-mode').value==='predicted') {
      const value=selected?this.graph.prediction(point.index,selected.id)?.q_value:this.job?.predicted_landscape?.input_mean_reward?.[point.index];
      if(Number.isFinite(value))notes.push(`${selected?'Estimated reward (Q)':'Mean estimated reward (Q)'}: ${Number(value.toPrecision(3))}`);
    }
    if(record?.error)notes.push(typeof record.error==='string'?record.error:JSON.stringify(record.error));
    if(record?.embedding_warning)notes.push(String(record.embedding_warning));
    if(record?.reused_baseline)notes.push('Neutral condition: this episode’s baseline measurement was reused');
    if(selected) {
      if(typeof record?.success==='boolean'&&!record.error&&typeof baseline?.success==='boolean') {
        if(baseline.success&&!record.success)notes.push('New failure from a successful baseline');
        if(!baseline.success&&record.success)notes.push('Recovery from a failed baseline');
      }
      if(record?.embedding?.position)notes.push(`Measured episode features${Number.isFinite(record.embedding.similarity)?` · cosine similarity ${record.embedding.similarity.toFixed(3)}`:''}`);
      else if(!record?.reused_baseline)notes.push('Pending layout · no measured feature coordinates');
    }
    this.$('selected-note').textContent=notes.join(' · ');
    this.$('baseline-outcome').textContent=`Seed ${point.seed} · ${this.pairState(baseline)==='error'?'rollout error':`observed task ${this.pairState(baseline)}`}`;
    this.$('variation-replay').hidden=!selected;
    this.renderMedia('baseline-media',baseline,`No baseline replay was recorded for seed ${point.seed}.`);
    this.renderMedia('variation-media',selected?record:null,state==='untested'?`Evaluate this variation for seed ${point.seed} to record a replay.`:`No replay was recorded for this variation at seed ${point.seed}.`);
    const container=this.$('episode-table');container.replaceChildren();
    if(!selected||!record)return;
    const table=this.node('table'),head=this.node('thead'),header=this.node('tr');
    for(const text of ['Seed','Baseline task outcome','Variation task outcome','Steps'])header.append(this.node('th',text));head.append(header);table.append(head);
    const body=this.node('tbody'),row=this.node('tr'),outcome=item=>item?.error?`Error: ${item.error}`:item?.success===true?'Success':item?.success===false?'Failure':'Not measured';
    row.append(this.node('td',point.seed),this.node('td',outcome(baseline)),this.node('td',outcome(record)),this.node('td',record.steps??'—'));body.append(row);table.append(body);container.append(table);
  }
  renderArtifacts() {
    const container=this.$('artifacts');container.replaceChildren();
    for(const [key,value] of Object.entries(this.job?.artifacts||{})) {
      if(/preview|image|video|replay/i.test(key))continue;
      const url=this.artifactUrl(value);if(!url)continue;
      const link=this.node('a',key==='report'?'Open run report':`Open ${key.replaceAll('_',' ')}`,'report-link');link.href=url;link.target='_blank';link.rel='noopener noreferrer';container.append(link);
    }
  }
  renderLive() {
    const base=this.artifactUrl(this.job?.artifacts?.preview);this.$('live').hidden=!base;
    if(!base)return;
    const frame=encodeURIComponent([this.job.phase,this.job.completed,this.job.episode,this.job.step].join('-'));
    const url=base.includes('?')?base:`${base}?frame=${frame}`;
    if(this.$('live-image').getAttribute('src')!==url)this.$('live-image').src=url;
    this.$('live-caption').textContent=`${this.isRunning(this.job)?'Live episode preview':'Last recorded preview'} · ${this.$('step-status').textContent}`;
  }
}
