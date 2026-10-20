'use strict';

// Job/selection state is separate from the canvas camera and from the input form.
class EmbeddingExplorer {
  constructor({api,onInspect,onStatus,onError,onBusy,onSelectInput,describeAction}) {
    Object.assign(this,{api,onInspect,onStatus,onError,onBusy,onSelectInput});
    this.describeAction=describeAction||(action=>action?.name?.replaceAll('_',' ')||Object.entries(action||{}).filter(([key,value])=>key!=='id'&&value!==null&&value!==undefined).map(([key,value])=>`${key}: ${value}`).join(' · '));
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
    this.job=null;this.preview=null;this.point=null;this.action=null;this.pendingAction=null;this.expansion=null;this.expansionCache=new Map();this.originalCache=new Map();
    this.discovery=null;this.measuredCache=new Map();this.partialExpansions=new Map();
    this.predictionRevision='';this.predictionCache=new Map();this.predictionRequests=new Map();
    this.predictionError('');
    this.control('embedding-input-dot').hidden=false;this.control('embedding-action-dot').hidden=false;
    this.view.reset();this.control('embedding-wrap').hidden=true;
    this.control('embedding-actions').hidden=true;
    this.control('embedding-back').disabled=true;
    this.control('embedding-meta').textContent='';
    this.control('embedding-focus').textContent='';
    this.control('analysis-performance').hidden=true;
    this.control('analysis-performance-summary').textContent='';
    this.control('analysis-timings').replaceChildren();
    this.control('analysis-report').hidden=true;this.control('analysis-report').href='';
  }

  get task() {return this.job?.task||this.preview?.config.task||'classification';}
  get summaryDataset() {return this.task==='summarization'&&Boolean(this.preview?.dataset||this.preview?.config.dataset_id||this.job?.references?.dataset_id);}
  get generationDataset() {return this.task==='generation'&&Boolean(this.preview?.dataset||this.preview?.config.dataset_id||this.job?.requirements?.dataset_id);}
  get textDataset() {return this.summaryDataset||this.generationDataset;}

  async load(preview,device='auto') {
    this.reset();this.preview=preview;const run=this.run;
    this.onError('');this.onBusy(true);this.onStatus('Loading the model and computing embeddings…');
    try {
      const job=await this.api('/api/embeddings',{preview_id:preview.id,device});
      if(run!==this.run)return;
      await this.receive(job,run);
    } catch(error) {this.fail(error,run);}
  }

  async receive(job,run) {
    if(run!==this.run)return;
    if(job.status==='queued'||job.status==='running') {
      this.onStatus(`${job.phase||'Computing embeddings'}${job.total?` · ${job.completed} / ${job.total}`:''}${job.device?` · ${this.deviceLabel(job)}`:''}`);
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
    const population=images?'PCA fit on original images only':this.textDataset?`PCA fit on original ${this.generationDataset?'prompts':'articles'} only`:fitCount?`PCA fit on the original + ${fitCount-1} action variant${fitCount===2?'':'s'} (${fitCount} total)`:'PCA fit on the original and all action variants';
    this.control('embedding-meta').textContent=`${count} ${images?'image':this.summaryDataset?'article':this.generationDataset?'prompt':'input'}${count===1?'':'s'} · ${job.feature_dim}D ${representation} → 3D PCA · ${variance}% variance retained · ${population}`;
    this.control('embedding-meta').title=`${job.model?.name||this.preview.config.model.name} · ${job.feature_method} · ${job.projection.method}. Positions approximate full-space similarity; cosine uses the full embeddings. No failure score is computed.`;
    this.showPerformance(job);
    const report=this.control('analysis-report');report.href=`/api/embeddings/${job.id}`;
    report.target='_blank';report.rel='noopener noreferrer';report.hidden=false;
    this.onStatus('Select an input to explore its action combinations.');this.onBusy(false);this.onReady?.(job);
    if(!images&&!this.textDataset&&job.points.length)await this.focus(job.points[0].index);
  }

  deviceLabel(job) {
    if(!job.device)return 'Device not reported';
    const device=job.device.toUpperCase();
    return job.device_name&&job.device_name.toUpperCase()!==device?`${device} · ${job.device_name}`:device;
  }

  showPerformance(job) {
    const timings=job.timings_seconds;
    const duration=value=>value===0?'0 s':value<.01?'<0.01 s':`${value.toFixed(2)} s`;
    this.control('analysis-timings').replaceChildren();
    this.control('analysis-performance').hidden=!timings;
    if(!timings)return;
    const valid=value=>Number.isFinite(value)&&value>=0;
    const total=valid(timings.total)?` · ${duration(timings.total)}`:'';
    this.control('analysis-performance-summary').textContent=`Analyze${total} · ${this.deviceLabel(job)}`;
    const rows=[['Files / download','preparation'],['Weight loading','model_loading'],['Embeddings','embedding'],['PCA','pca']];
    for(const [label,key] of rows) {
      if(!valid(timings[key]))continue;
      const name=document.createElement('dt'),value=document.createElement('dd');
      name.textContent=label;value.textContent=duration(timings[key]);
      if(timings[key]===0&&(job.features_reused||(key==='model_loading'&&job.model_reused)))value.textContent+=' · reused';
      this.control('analysis-timings').append(name,value);
    }
    if(valid(job.queue_wait_seconds)&&job.queue_wait_seconds>=.01) {
      const name=document.createElement('dt'),value=document.createElement('dd');
      name.textContent='Queue wait';value.textContent=duration(job.queue_wait_seconds);
      this.control('analysis-timings').append(name,value);
    }
  }

  fail(error,run) {
    if(run!==this.run)return;
    this.onBusy(false);this.onError(error.message);this.onStatus('Could not load the embedding view. Check the model and try again.');
  }

  async focusAction(index,action) {
    if(this.availableExpansion(index)?.nodes.some(node=>node.id===action)) {
      // Inspect an existing node without cancelling its input's live expansion.
      this.onError('');this.focusMeasured(index,action);this.onInspectPair?.(index,action);return;
    }
    return this.focus(index,action);
  }

  async focus(index,action=null) {
    const point=this.job?.points.find(point=>point.index===index);
    if(!point)return;
    const run=this.run,selection=++this.selection;
    clearTimeout(this.expansionTimer);this.point=point;this.action=action;this.pendingAction=action;this.expansion=null;
    this.view.focus(index);this.control('embedding-back').disabled=false;
    this.loadPredictions(index);
    this.control('embedding-actions').hidden=true;
    this.control('embedding-focus').textContent=`${point.name} · computing action embeddings…`;
    this.onError('');this.syncInput(index);
    // An uncached action is still loading; do not present it as the original.
    this.onInspect(action===null?this.originalInput():null,null,this.preview.config);
    this.onInspectPair?.(index,action);
    const cached=this.expansionCache.get(index);
    if(cached&&(action===null||cached.nodes.some(node=>node.id===action))) {
      await this.receiveExpansion(cached,run,selection);return;
    }
    const available=this.availableExpansion(index);
    if(available)this.showExpansion(available);
    if(action!==null&&this.pendingAction===null)return;
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
    if(job.input_index!==this.point?.index)return;
    if(job.status==='queued'||job.status==='running') {
      if(job.nodes?.length) {
        // A superseded expansion may restart with a smaller first batch.
        // Keep its already computed nodes selectable until the full job arrives.
        const nodes=new Map((this.partialExpansions.get(job.input_index)?.nodes||[]).map(node=>[node.id,node]));
        for(const node of job.nodes)nodes.set(node.id,node);
        this.partialExpansions.set(job.input_index,{...job,nodes:[...nodes.values()]});
        this.showExpansion(this.availableExpansion(job.input_index));
      }
      this.control('embedding-focus').textContent=`${this.point.name} · ${job.phase||'Computing actions'}${job.total?` · ${job.completed} / ${job.total}`:''}`;
      this.expansionTimer=setTimeout(async()=>{
        if(run!==this.run||selection!==this.selection)return;
        try{await this.receiveExpansion(await this.api(`/api/embeddings/${this.job.id}/expansions/${job.id}`),run,selection);}
        catch(error){this.failExpansion(error,run,selection);}
      },350);return;
    }
    if(job.status!=='completed')throw new Error(job.error||'Could not compute the action embeddings.');
    this.partialExpansions.delete(job.input_index);
    this.expansionCache.set(job.input_index,job);this.showExpansion(job);
    if(this.pendingAction!==null)throw new Error(`Action ${this.pendingAction} is not available for this input.`);
  }

  availableExpansion(index) {
    if(this.expansionCache.has(index))return this.expansionCache.get(index);
    const progress=this.partialExpansions.get(index),nodes=new Map(this.measuredCache.get(index));
    for(const node of progress?.nodes||[])nodes.set(node.id,node);
    return nodes.size?{...progress,input_index:index,nodes:[...nodes.values()].sort((a,b)=>a.id-b.id),partial:true,measuredOnly:!progress}:null;
  }

  showExpansion(expansion) {
    if(expansion.input_index!==this.point?.index)return;
    this.expansion=expansion;this.view.setExpansion(expansion.input_index,expansion.nodes);
    if(expansion.original)this.originalCache.set(expansion.input_index,expansion.original);
    this.view.selectAction(this.action);
    const choice=(value,label)=>{const el=document.createElement('option');el.value=value;el.textContent=label;return el;};
    this.control('embedding-action').replaceChildren(choice('original','Original input'),...expansion.nodes.map(node=>choice(String(node.id),`${this.task==='generation'?`Action ${node.id}`:`Combination ${node.id+1}`} · ${this.describeAction(node.action)}${node.unchanged?' · unchanged':''}`)));
    this.control('embedding-action').value=this.action===null?'original':String(this.action);this.control('embedding-actions').hidden=false;
    const count=expansion.partial&&!expansion.measuredOnly?`${expansion.nodes.length} / ${expansion.total}`:expansion.nodes.length;
    this.control('embedding-focus').textContent=`${this.point.name} · ${count} ${expansion.measuredOnly?'measured actions':'action combinations'}${expansion.partial&&this.isDiscoveryActive?.()?' · live discovery':''}`;
    // The requested node may arrive in any batch. A later inspection cancels it.
    if(this.pendingAction!==null) {
      const requested=expansion.nodes.find(node=>node.id===this.pendingAction);
      if(requested)this.inspect(requested);
    } else if(this.textDataset&&this.action===null)this.onInspect(this.originalInput(),null,this.preview.config);
  }

  setDiscovery(job) {
    this.control('embedding-input-dot').hidden=Boolean(job);this.control('embedding-action-dot').hidden=Boolean(job);
    if(!job||job.id!==this.discovery?.id)this.measuredCache.clear();
    const revision=job?.predicted_landscape?`${job.id}:${job.predicted_landscape.policy_updates}`:'';
    if(revision!==this.predictionRevision) {
      this.predictionRevision=revision;this.predictionCache.clear();this.predictionRequests=new Map();
      this.predictionError('');
    }
    this.discovery=job;this.view.setDiscovery(job);
    for(const record of job?.records||[]) {
      const node=record.node;
      // Plot only model coordinates supplied for this input/action pair.
      if(!node||node.id!==record.action_id||!Array.isArray(node.position)||node.position.length!==3||!node.position.every(Number.isFinite)||!this.job?.points.some(point=>point.index===record.input_index))continue;
      if(!this.measuredCache.has(record.input_index))this.measuredCache.set(record.input_index,new Map());
      this.measuredCache.get(record.input_index).set(node.id,node);
    }
    if(!this.point)return;
    this.loadPredictions(this.point.index);
    const expansion=this.availableExpansion(this.point.index);
    if(expansion)this.showExpansion(expansion);
    else if(this.expansion?.partial) {
      this.expansion=null;this.view.setExpansion(this.point.index,[]);
      this.control('embedding-actions').hidden=true;
      this.control('embedding-focus').textContent=this.point.name;
    }
  }

  predictionFor(index,action) {
    return this.predictionCache.get(index)?.actions.find(item=>item.action_id===action);
  }

  predictionError(message) {
    this.control('landscape-error').textContent=message;this.control('landscape-error').hidden=!message;
  }

  async loadPredictions(index) {
    if(!this.predictionRevision)return;
    const cached=this.predictionCache.get(index);
    if(cached) {
      if(this.point?.index===index){this.predictionError('');this.view.setPredictions(cached);this.onPredictions?.();}
      return;
    }
    if(this.predictionRequests.has(index))return;
    const revision=this.predictionRevision,run=this.run,job=this.discovery,requests=this.predictionRequests;
    const request=this.api(`/api/discoveries/${job.id}/predictions?input=${index}`);
    requests.set(index,request);
    try {
      const result=await request;
      if(run!==this.run||revision!==this.predictionRevision||result.discovery_id!==job.id||
         result.input_index!==index||result.policy_updates!==job.predicted_landscape.policy_updates)return;
      this.predictionCache.set(index,result);
      if(this.predictionCache.size>32)this.predictionCache.delete(this.predictionCache.keys().next().value);
      if(this.point?.index===index){this.predictionError('');this.view.setPredictions(result);this.onPredictions?.();}
    } catch(error) {
      if(run===this.run&&revision===this.predictionRevision&&this.point?.index===index)
        this.predictionError(`Could not load predicted action values: ${error.message}. Select the input to retry.`);
    } finally {requests.delete(index);}
  }

  failExpansion(error,run,selection) {
    if(run!==this.run||selection!==this.selection)return;
    this.pendingAction=null;
    this.onError(error.message);
    this.control('embedding-focus').textContent=`${this.point.name} · could not compute actions. Select the input to retry.`;
  }

  inspect(node) {
    if(!this.point)return;
    this.pendingAction=null;
    this.action=node?.id??null;this.view.selectAction(this.action);
    this.control('embedding-action').value=node?String(node.id):'original';
    this.onInspect(this.originalInput(),node||null,this.preview.config);this.onInspectPair?.(this.point.index,node?.id??null);
  }

  originalInput() {
    return this.expansion?.original||this.originalCache.get(this.point?.index)||(this.textDataset?null:this.point);
  }

  syncInput(index) {
    if(this.task!=='classification'&&!this.textDataset)return;
    const run=this.run,selection=this.selection;
    Promise.resolve(this.onSelectInput?.(index)).then(original=>{
      if(!original||run!==this.run||selection!==this.selection||this.point?.index!==index)return;
      this.originalCache.set(index,original);
      const node=this.expansion?.nodes.find(node=>node.id===this.action);
      if(this.action===null||node)this.onInspect(this.originalInput(),node||null,this.preview.config);
    }).catch(error=>{if(run===this.run&&selection===this.selection)this.onError(error.message);});
  }

  focusMeasured(index,action) {
    const point=this.job?.points.find(point=>point.index===index);if(!point)return;
    this.pendingAction=null;
    if(this.point?.index!==index){this.selection++;clearTimeout(this.expansionTimer);}
    this.point=point;this.action=action;
    this.expansion=this.availableExpansion(index);
    this.view.focus(index);
    this.loadPredictions(index);
    if(this.expansion)this.showExpansion(this.expansion);
    this.view.selectAction(action);
    this.control('embedding-back').disabled=false;
    if(!this.expansion)this.control('embedding-focus').textContent=`${point.name} · measured action ${action}`;
    this.control('embedding-actions').hidden=!this.expansion;
    this.syncInput(index);
    const node=this.expansion?.nodes.find(node=>node.id===action);
    this.onInspect(node||action===null?this.originalInput():null,node||null,this.preview.config);
  }

  clearFocus() {
    this.selection++;clearTimeout(this.expansionTimer);this.point=null;this.action=null;this.pendingAction=null;this.expansion=null;
    this.view.clearFocus();this.control('embedding-actions').hidden=true;
    this.control('embedding-back').disabled=true;this.control('embedding-focus').textContent='';
    this.onInspect(null,null);this.onInspectPair?.(null,null);this.onError('');
    if(this.job)this.onStatus('Select an input to explore its action combinations.');
  }
}
