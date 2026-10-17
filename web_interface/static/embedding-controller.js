'use strict';

// Job/selection state is separate from the canvas camera and from the input form.
class EmbeddingExplorer {
  constructor({api,onInspect,onStatus,onError,onBusy,onSelectInput,describeAction}) {
    Object.assign(this,{api,onInspect,onStatus,onError,onBusy,onSelectInput});
    this.describeAction=describeAction||(action=>action?.name?.replaceAll('_',' ')||Object.entries(action||{}).map(([key,value])=>`${key}: ${value}`).join(' · '));
    this.control=id=>document.getElementById(id);
    this.run=0;this.selection=0;this.timer=null;this.expansionTimer=null;
    this.dispatch=Promise.resolve();
    this.view=new EmbeddingView(this.control('embedding-canvas'),{
      onSelectSample:point=>this.focus(point.index),
      onSelectAction:node=>this.inspect(node),
      onClear:()=>this.clearFocus()
    });
    this.control('embedding-fit').addEventListener('click',()=>this.view.fit());
    this.control('embedding-zoom-in').addEventListener('click',()=>this.view.zoom(1.25));
    this.control('embedding-zoom-out').addEventListener('click',()=>this.view.zoom(1/1.25));
    this.control('embedding-back').addEventListener('click',()=>this.clearFocus());
    this.control('embedding-action').addEventListener('change',()=>{
      const id=this.control('embedding-action').value;
      this.inspect(id==='original'?null:this.expansion?.nodes.find(node=>String(node.id)===id));
    });
    this.reset();
  }

  reset() {
    this.run++;this.selection++;clearTimeout(this.timer);clearTimeout(this.expansionTimer);
    this.job=null;this.preview=null;this.point=null;this.action=null;this.expansion=null;this.expansionCache=new Map();
    this.discovery=null;this.measuredCache=new Map();
    this.view.reset();this.control('embedding-wrap').hidden=true;
    this.control('embedding-actions').hidden=true;
    this.control('embedding-back').disabled=true;
    this.control('embedding-meta').textContent='';
    this.control('embedding-focus').textContent='';
  }

  get task() {return this.job?.task||this.preview?.config.task||'classification';}

  async load(preview) {
    this.reset();this.preview=preview;const run=this.run;
    this.onError('');this.onBusy(true);this.onStatus('Loading the model and computing embeddings…');
    try {
      const job=await this.api('/api/embeddings',{preview_id:preview.id});
      if(run!==this.run)return;
      await this.receive(job,run);
    } catch(error) {this.fail(error,run);}
  }

  async receive(job,run) {
    if(run!==this.run)return;
    if(job.status==='queued'||job.status==='running') {
      this.onStatus(`${job.phase||'Computing embeddings'}${job.total?` · ${job.completed} / ${job.total}`:''}`);
      this.timer=setTimeout(async()=>{
        if(run!==this.run)return;
        try{await this.receive(await this.api(`/api/embeddings/${job.id}`),run);}catch(error){this.fail(error,run);}
      },500);return;
    }
    if(job.status!=='completed')throw new Error(job.error||'Could not compute this model’s embeddings.');
    this.job=job;this.control('embedding-wrap').hidden=false;
    this.view.setData(job.points);
    const variance=Math.round(job.projection.variance*100);
    const images=this.task==='classification';
    const representation=images?'embeddings':this.task==='generation'?'CLIP prompt conditioning':'summarization encoder embeddings';
    const count=job.points.length,fitCount=job.projection.fit_count;
    const population=images?'PCA fit on original images only':fitCount?`PCA fit on the original + ${fitCount-1} action variant${fitCount===2?'':'s'} (${fitCount} total)`:'PCA fit on the original and all action variants';
    this.control('embedding-meta').textContent=`${count} ${images?'image':'input'}${count===1?'':'s'} · ${job.feature_dim}D ${representation} → 3D PCA · ${variance}% variance retained · ${population}`;
    this.control('embedding-meta').title=`${job.model?.name||this.preview.config.model.name} · ${job.feature_method} · ${job.projection.method}. Positions approximate full-space similarity; cosine uses the full embeddings. No failure score is computed.`;
    this.onStatus('Select an input to explore its action combinations.');this.onBusy(false);this.onReady?.(job);
    if(!images&&job.points.length)await this.focus(job.points[0].index);
  }

  fail(error,run) {
    if(run!==this.run)return;
    this.onBusy(false);this.onError(error.message);this.onStatus('Could not load the embedding view. Check the model and try again.');
  }

  async focus(index) {
    const point=this.job?.points.find(point=>point.index===index);
    if(!point)return;
    const run=this.run,selection=++this.selection;
    clearTimeout(this.expansionTimer);this.point=point;this.action=null;this.expansion=null;
    this.view.focus(index);this.control('embedding-back').disabled=false;
    this.control('embedding-actions').hidden=true;
    this.control('embedding-focus').textContent=`${point.name} · computing action embeddings…`;
    this.onError('');if(this.task==='classification')this.onSelectInput(index);this.onInspect(point,null,this.preview.config);
    this.onInspectPair?.(index,null);
    if(this.expansionCache.has(index)) {
      await this.receiveExpansion(this.expansionCache.get(index),run,selection);return;
    }
    const measured=this.measuredExpansion(index);
    if(measured)this.showExpansion(measured);
    if(this.isDiscoveryActive?.()) {
      if(!measured)this.control('embedding-focus').textContent=`${point.name} · live discovery · waiting for measured actions…`;
      return;
    }
    try {
      // Server supersession follows arrival order. Wait for the previous POST's
      // acknowledgement, but never wait for its inference job to finish.
      const request=this.dispatch.then(()=>{
        if(run!==this.run||selection!==this.selection)return null;
        return this.api(`/api/embeddings/${this.job.id}/expand`,{input_index:index});
      });
      this.dispatch=request.then(()=>undefined,()=>undefined);
      const job=await request;
      if(run!==this.run||selection!==this.selection)return;
      await this.receiveExpansion(job,run,selection);
    } catch(error) {this.failExpansion(error,run,selection);}
  }

  async receiveExpansion(job,run,selection) {
    if(run!==this.run||selection!==this.selection)return;
    if(job.status==='queued'||job.status==='running') {
      this.control('embedding-focus').textContent=`${this.point.name} · ${job.phase||'Computing actions'}${job.total?` · ${job.completed} / ${job.total}`:''}`;
      this.expansionTimer=setTimeout(async()=>{
        if(run!==this.run||selection!==this.selection)return;
        try{await this.receiveExpansion(await this.api(`/api/embeddings/${this.job.id}/expansions/${job.id}`),run,selection);}
        catch(error){this.failExpansion(error,run,selection);}
      },350);return;
    }
    if(job.status!=='completed')throw new Error(job.error||'Could not compute the action embeddings.');
    if(job.input_index!==this.point.index)return;
    this.expansionCache.set(job.input_index,job);this.showExpansion(job);
  }

  measuredExpansion(index) {
    const nodes=this.measuredCache.get(index);
    return nodes?.size?{input_index:index,nodes:[...nodes.values()].sort((a,b)=>a.id-b.id),partial:true}:null;
  }

  showExpansion(expansion) {
    if(expansion.input_index!==this.point?.index)return;
    this.expansion=expansion;this.view.setExpansion(expansion.input_index,expansion.nodes);
    this.view.selectAction(this.action);
    const choice=(value,label)=>{const el=document.createElement('option');el.value=value;el.textContent=label;return el;};
    this.control('embedding-action').replaceChildren(choice('original','Original input'),...expansion.nodes.map(node=>choice(String(node.id),`Combination ${node.id+1} · ${this.describeAction(node.action)}${node.unchanged?' · unchanged':''}`)));
    this.control('embedding-action').value=this.action===null?'original':String(this.action);this.control('embedding-actions').hidden=false;
    this.control('embedding-focus').textContent=`${this.point.name} · ${expansion.nodes.length} ${expansion.partial?'measured actions':'action combinations'}${expansion.partial&&this.isDiscoveryActive?.()?' · live discovery':''}`;
  }

  setDiscovery(job) {
    if(!job||job.id!==this.discovery?.id)this.measuredCache.clear();
    this.discovery=job;this.view.setDiscovery(job);
    for(const record of job?.records||[]) {
      const node=record.node;
      // Plot only model coordinates supplied for this input/action pair.
      if(!node||node.id!==record.action_id||!Array.isArray(node.position)||node.position.length!==3||!node.position.every(Number.isFinite)||!this.job?.points.some(point=>point.index===record.input_index))continue;
      if(!this.measuredCache.has(record.input_index))this.measuredCache.set(record.input_index,new Map());
      this.measuredCache.get(record.input_index).set(node.id,node);
    }
    if(!this.point)return;
    const expansion=this.expansionCache.get(this.point.index)||this.measuredExpansion(this.point.index);
    if(expansion)this.showExpansion(expansion);
    else if(this.expansion?.partial) {
      this.expansion=null;this.view.setExpansion(this.point.index,[]);
      this.control('embedding-actions').hidden=true;
      this.control('embedding-focus').textContent=this.point.name;
    }
  }

  failExpansion(error,run,selection) {
    if(run!==this.run||selection!==this.selection)return;
    this.onError(error.message);
    this.control('embedding-focus').textContent=`${this.point.name} · could not compute actions. Select the input to retry.`;
  }

  inspect(node) {
    if(!this.point)return;
    this.action=node?.id??null;this.view.selectAction(this.action);
    this.control('embedding-action').value=node?String(node.id):'original';
    this.onInspect(this.point,node||null,this.preview.config);this.onInspectPair?.(this.point.index,node?.id??null);
  }

  focusMeasured(index,action) {
    const point=this.job?.points.find(point=>point.index===index);if(!point)return;
    this.selection++;clearTimeout(this.expansionTimer);this.point=point;this.action=action;
    this.expansion=this.expansionCache.get(index)||this.measuredExpansion(index);
    this.view.focus(index);
    if(this.expansion)this.showExpansion(this.expansion);
    this.view.selectAction(action);
    this.control('embedding-back').disabled=false;
    if(!this.expansion)this.control('embedding-focus').textContent=`${point.name} · measured action ${action}`;
    this.control('embedding-actions').hidden=!this.expansion;
    if(this.task==='classification')this.onSelectInput(index);
    this.onInspect(point,this.expansion?.nodes.find(node=>node.id===action)||null,this.preview.config);
  }

  clearFocus() {
    this.selection++;clearTimeout(this.expansionTimer);this.point=null;this.action=null;this.expansion=null;
    this.view.clearFocus();this.control('embedding-actions').hidden=true;
    this.control('embedding-back').disabled=true;this.control('embedding-focus').textContent='';
    this.onInspect(null,null);this.onInspectPair?.(null,null);this.onError('');
    if(this.job)this.onStatus('Select an input to explore its action combinations.');
  }
}
