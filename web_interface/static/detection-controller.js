'use strict';

// This workspace owns its dataset, jobs, selection and polling lifecycle.
class DetectionController {
  constructor({api}) {
    this.api=api;this.active=false;this.dataset=null;this.analysis=null;this.discovery=null;
    this.selected=null;this.selectedRecord=null;this.busy=false;this.generation=0;this.timer=null;
    this.selectionGeneration=0;this.expansion=null;this.inspection=null;this.graphAnalysis=null;
    this.root=document.getElementById('detection-workspace');
    if(!this.root)throw new Error('Detection workspace is missing.');
    this.root.innerHTML=`
      <div class="detection-layout"><div class="detection-settings">
        <h2>Model</h2><label for="detection-model-source">Model source</label>
        <select id="detection-model-source"><option value="preset">DETR ResNet-50</option><option value="huggingface">Hugging Face model</option><option value="local">Local model path</option></select>
        <div id="detection-model-custom" hidden><label for="detection-model-name">Model ID or path</label><input id="detection-model-name" type="text" placeholder="organization/model or local path"></div>
        <h2>Images</h2><label for="detection-source">Dataset source</label><select id="detection-source"><option value="folder">Local folder</option><option value="huggingface">Hugging Face dataset</option></select>
        <div id="detection-folder-fields"><label for="detection-folder-path">Folder path</label><input id="detection-folder-path" type="text" placeholder="/path/to/images"></div>
        <div id="detection-hf-fields" hidden><label for="detection-dataset-preset">Hugging Face dataset</label><select id="detection-dataset-preset"><option value="detection-datasets/coco">COCO · everyday objects</option><option value="cvdl/oxford-pets">Oxford Pets · cats and dogs</option><option value="merve/cppe-5">CPPE-5 · protective equipment</option><option value="other">Other…</option></select><div id="detection-dataset-name-field" hidden><label for="detection-dataset-name">Dataset ID</label><input id="detection-dataset-name" type="text" placeholder="organization/dataset"></div><details class="dataset-options"><summary>Dataset options</summary><label for="detection-config">Configuration (optional)</label><input id="detection-config" type="text"><label for="detection-split">Split</label><input id="detection-split" type="text" value="val"><label for="detection-subset">Rows (optional)</label><input id="detection-subset" type="text" value="20" placeholder="e.g. [0:100]"><p class="hint">20 loads the first 20 images. Leave blank for all rows.</p><label for="detection-column">Image column (optional)</label><input id="detection-column" type="text" value="image"></details><p class="hint">Images are used for prediction consistency; dataset bounding boxes are not evaluated yet.</p></div>
        <button id="detection-load" type="button">Load images</button><p id="detection-dataset-status" class="hint" role="status">Load a dataset to begin.</p>
        <h2>Action space</h2>
        <label class="detection-check"><input id="detection-darkness" type="checkbox" checked> Darkness <span>0.1, 0.2, 0.3, 0.4, 0.5</span></label>
        <label class="detection-check"><input id="detection-saturation" type="checkbox" checked> Saturation <span>0.5, 0.55, 0.6, 0.65, 0.7</span></label>
        <button id="detection-analyze" class="primary" type="button" disabled>Analyze</button>
        <p id="detection-error" class="error" role="alert"></p>
      </div><div class="detection-results"><div class="detection-heading"><h2>Search space</h2><span id="detection-progress" role="status">No analysis yet.</span></div>
        <div class="detection-graph-controls"><button id="detection-back" type="button">Show all</button><button id="detection-fit" type="button">Fit</button><button id="detection-zoom-in" type="button">+</button><button id="detection-zoom-out" type="button">−</button></div>
        <canvas id="detection-canvas" aria-label="3D detector feature graph"></canvas><p id="detection-graph-status" class="hint">Analyze images to see the search space.</p>
        <label for="detection-action">Inspect</label><select id="detection-action" disabled><option value="original">Original image</option></select>
        <div class="detection-legend" aria-label="Graph legend"><span>Green → red: consistency change</span><span>Gray: unmeasured</span><span>Drag: rotate · Shift-drag: pan · Scroll: zoom</span></div>
        <div class="detection-workflow-actions"><button id="detection-find-failure" type="button" disabled aria-controls="detection-discovery-panel" aria-expanded="false">Find failure</button><button id="detection-fine-tune" type="button" disabled title="Fine-tuning is not available for object detection yet">Adapt</button></div>
        <section id="detection-discovery-panel" class="detection-discovery-panel" hidden><h3>Find failure</h3><div class="detection-numbers"><label>Episodes<input id="detection-episodes" type="number" min="1" max="10000" value="20"></label><label>Seed<input id="detection-seed" type="number" min="0" max="4294967295" value="42"></label></div><button id="detection-discover" type="button" disabled>Start discovery</button>
        <h3>Measured action outcomes</h3><div id="detection-records" class="detection-records"><p class="hint">No transformations measured yet.</p></div></section>
      </div><aside class="detection-inspector"><h2>Inspect</h2><div id="detection-selection" hidden><h3 id="detection-selected-name"></h3><div class="detection-images"><figure><img id="detection-original" alt="Original image with detection boxes"><figcaption id="detection-original-caption">Original detections</figcaption></figure><figure id="detection-transformed-wrap" hidden><img id="detection-transformed" alt="Transformed image with detection boxes"><figcaption>Transformed detections</figcaption></figure></div><div id="detection-details" aria-live="polite"></div></div><p id="detection-inspector-empty" class="hint">Select a point to view its images and predicted labels.</p></aside></div>`;
    this.$=id=>this.root.querySelector(`#detection-${id}`);
    this.view=new EmbeddingView(this.$('canvas'),{onSelectSample:point=>this.select(point.index),onSelectAction:node=>this.selectAction(node.id),onClear:()=>this.clearSelection()});
    this.$('back').addEventListener('click',()=>this.clearSelection());this.$('fit').addEventListener('click',()=>this.view.fit());
    this.$('zoom-in').addEventListener('click',()=>this.view.zoom(1.25));this.$('zoom-out').addEventListener('click',()=>this.view.zoom(1/1.25));
    this.$('action').addEventListener('change',()=>this.selectAction(this.$('action').value==='original'?null:Number(this.$('action').value)));
    this.$('load').addEventListener('click',()=>this.loadDataset());
    this.$('analyze').addEventListener('click',()=>this.analyze());
    this.$('discover').addEventListener('click',()=>this.discover());
    this.$('find-failure').addEventListener('click',()=>{this.$('discovery-panel').hidden=!this.$('discovery-panel').hidden;this.$('find-failure').setAttribute('aria-expanded',String(!this.$('discovery-panel').hidden));});
    this.$('source').addEventListener('change',()=>{this.$('folder-fields').hidden=this.$('source').value!=='folder';this.$('hf-fields').hidden=this.$('source').value!=='huggingface';this.invalidateDataset();});
    this.$('dataset-preset').addEventListener('change',()=>this.selectDatasetPreset());
    this.$('model-source').addEventListener('change',()=>{this.$('model-custom').hidden=this.$('model-source').value==='preset';this.invalidateAnalysis();});
    this.$('model-name').addEventListener('input',()=>this.invalidateAnalysis());
    for(const id of ['folder-path','dataset-name','config','split','subset','column'])this.$(id).addEventListener('input',()=>this.invalidateDataset());
    for(const id of ['darkness','saturation'])this.$(id).addEventListener('change',()=>{this.discovery=null;this.clearSelection();this.controls();});
  }
  enter(){this.active=true;if(this.pending)this.poll(this.pending.kind,this.pending.id,this.generation);this.controls();}
  leave(){this.active=false;clearTimeout(this.timer);this.timer=null;}
  selectDatasetPreset(){
    const preset=this.$('dataset-preset').value;
    const options={'detection-datasets/coco':{split:'val',column:'image'},'cvdl/oxford-pets':{split:'test',column:'img'},'merve/cppe-5':{split:'test',column:'image'}};
    const selected=options[preset];this.$('dataset-name-field').hidden=Boolean(selected);
    this.$('dataset-name').value=selected?preset:'';this.$('config').value='';
    this.$('split').value=selected?.split||'train';this.$('column').value=selected?.column||'';
    this.$('subset').value='20';this.invalidateDataset();
  }
  invalidateDataset(){this.dataset=null;this.$('dataset-status').textContent='Load the edited dataset to continue.';this.invalidateAnalysis();}
  invalidateAnalysis(){if(this.busy)return;this.analysis=null;this.discovery=null;this.selected=null;this.selectedRecord=null;this.pending=null;this.generation++;this.selectionGeneration++;this.graphAnalysis=null;this.$('discovery-panel').hidden=true;this.$('find-failure').setAttribute('aria-expanded','false');this.view.reset();this.render();}
  controls(){for(const id of ['source','folder-path','dataset-preset','dataset-name','config','split','subset','column','model-source','model-name','episodes','seed','darkness','saturation'])this.$(id).disabled=this.busy;this.$('load').disabled=this.busy;this.$('analyze').disabled=this.busy||!this.dataset;this.$('find-failure').disabled=!this.analysis;this.$('discover').disabled=this.busy||!this.analysis||(!this.$('darkness').checked&&!this.$('saturation').checked);}
  error(message=''){this.$('error').textContent=message;}
  async loadDataset(){
    if(this.busy)return;this.error();this.dataset=null;this.invalidateAnalysis();const token=++this.generation;this.busy=true;this.controls();
    const source=this.$('source').value;
    const preset=this.$('dataset-preset').value;
    const data=source==='folder'?{source,path:this.$('folder-path').value.trim()}:{source,name:preset==='other'?this.$('dataset-name').value.trim():preset,config:this.$('config').value.trim(),split:this.$('split').value.trim(),subset:this.$('subset').value.trim(),column:this.$('column').value.trim()};
    try{let job=await this.api('/api/datasets',data);while(job.status==='loading'){this.$('dataset-status').textContent='Loading images…';await new Promise(r=>setTimeout(r,700));if(token!==this.generation)return;job=await this.api(`/api/datasets/${encodeURIComponent(job.id)}`);}
      if(token!==this.generation)return;if(job.status!=='ready')throw Error(job.error||'The dataset could not be loaded.');this.dataset=job;this.$('dataset-status').textContent=`${job.count} images loaded.`;
    }catch(e){if(token===this.generation){this.error(e.message);this.$('dataset-status').textContent='Images could not be loaded.';}}
    finally{if(token===this.generation){this.busy=false;this.controls();}}
  }
  async analyze(){
    if(this.busy||!this.dataset)return;const source=this.$('model-source').value;
    const model=source==='preset'?{source:'huggingface',name:'facebook/detr-resnet-50'}:{source,name:this.$('model-name').value.trim()};
    if(!model.name){this.error('Enter a model ID or local path.');return;}
    this.error();this.analysis=null;this.discovery=null;this.selected=null;this.selectedRecord=null;this.selectionGeneration++;this.graphAnalysis=null;this.view.reset();this.render();
    await this.start('analysis','/api/detection/analyze',{dataset_id:this.dataset.id,model,device:'auto'});
  }
  async discover(){
    if(this.busy||!this.analysis)return;const actions=['darkness','saturation'].filter(id=>this.$(id).checked),episodes=Number(this.$('episodes').value),seed=Number(this.$('seed').value);
    if(!actions.length||!this.$('episodes').value||!Number.isSafeInteger(episodes)||episodes<1||episodes>10000||!this.$('seed').value||!Number.isSafeInteger(seed)||seed<0||seed>=2**32){this.error('Choose an action, 1–10000 episodes and a seed from 0 to 4294967295.');return;}
    this.error();this.discovery=null;this.selectedRecord=null;this.render();
    await this.start('discovery','/api/detection/discover',{analysis_id:this.analysis.id,episodes,seed,actions});
  }
  async start(kind,path,payload){
    const token=++this.generation;this.busy=true;this.controls();this.$('progress').textContent='Starting…';
    try{const job=await this.api(path,payload);if(token!==this.generation)return;this.applyJob(kind,job);if(!this.finished(job)){this.pending={kind,id:job.id};if(this.active)this.poll(kind,job.id,token);}}
    catch(e){if(token===this.generation){this.error(e.message);this.$('progress').textContent='Request failed.';this.busy=false;this.controls();}}
  }
  finished(job){return ['completed','ready','failed','error','cancelled'].includes(job.status);}
  async poll(kind,id,token){
    clearTimeout(this.timer);this.timer=setTimeout(async()=>{if(!this.active||token!==this.generation)return;
      try{const job=await this.api(`/api/detection/jobs/${encodeURIComponent(id)}`);if(token!==this.generation||!this.active)return;this.applyJob(kind,job);if(!this.finished(job))this.poll(kind,id,token);}
      catch(e){if(token===this.generation){this.error(e.message);this.$('progress').textContent='Could not refresh job status.';this.pending=null;this.busy=false;if(kind==='analysis')this.analysis=null;this.render();this.controls();}}
    },700);
  }
  applyJob(kind,job){
    const progress=Number.isFinite(job.completed)&&Number.isFinite(job.total)?` · ${job.completed} / ${job.total}`:'';
    this.$('progress').textContent=`${job.phase||job.status||'Working'}${progress}`;
    if(['failed','error','cancelled'].includes(job.status)){this.error(job.error||`Job ${job.status}.`);this.pending=null;this.busy=false;if(kind==='analysis')this.analysis=null;this.render();this.controls();return;}
    if(kind==='analysis'&&job.points){this.analysis=job;}
    if(kind==='discovery'){this.discovery=job;}
    if(this.finished(job)){if(kind==='analysis')this.analysis=job;this.pending=null;this.busy=false;}
    this.render();this.controls();
  }
  selectedActions(){return ['darkness','saturation'].filter(id=>this.$(id).checked);}
  clearSelection(){this.selectionGeneration++;this.selected=null;this.selectedRecord=null;this.expansion=null;this.inspection=null;this.view.clearFocus();this.render();}
  async waitForJob(job,token,onProgress){
    while(!this.finished(job)){
      if(token!==this.selectionGeneration)return null;
      if(onProgress)onProgress(job);
      await new Promise(resolve=>setTimeout(resolve,400));
      if(token!==this.selectionGeneration)return null;
      job=await this.api(`/api/detection/jobs/${encodeURIComponent(job.id)}`);
    }
    if(token!==this.selectionGeneration)return null;
    if(job.status!=='completed')throw Error(job.error||'Could not inspect this image.');
    if(onProgress)onProgress(job);
    return job;
  }
  async select(index,record=null){
    if(!this.analysis||this.analysis.status!=='completed')return;
    const token=++this.selectionGeneration;this.selected=index;this.selectedRecord=null;this.inspection=null;this.expansion=null;
    this.view.focus(index);this.render();this.$('graph-status').textContent='Computing perturbation embeddings…';
    try{
      let expansion=await this.api('/api/detection/expand',{analysis_id:this.analysis.id,input_index:index,actions:this.selectedActions()});
      if(token!==this.selectionGeneration)return;
      this.expansion=expansion;
      const inspectPromise=this.inspectAction(null,token);
      expansion=await this.waitForJob(expansion,token,job=>{this.expansion=job;this.view.setExpansion(index,job.nodes||[]);this.renderActions();this.$('graph-status').textContent=`Perturbations · ${job.completed} / ${job.total}`;});
      if(!expansion)return;
      await inspectPromise;
      if(token===this.selectionGeneration){this.$('graph-status').textContent=`${expansion.nodes.length} perturbations · select a node`;if(record)this.selectAction(record.action_id);}
    }catch(e){if(token===this.selectionGeneration)this.error(e.message);}
  }
  async selectAction(actionId){
    if(!this.expansion||this.selected===null)return;
    this.selectedRecord=actionId===null?null:(this.discovery?.records||[]).find(r=>r.input_index===this.selected&&r.action_id===actionId)||null;
    this.view.selectAction(actionId);this.$('action').value=actionId===null?'original':String(actionId);
    const token=++this.selectionGeneration;
    // Keep the in-progress expansion poll alive for the selected image.
    if(this.expansion.status!=='completed')this.continueExpansion(this.expansion,token);
    await this.inspectAction(actionId,token);
  }
  async continueExpansion(job,token){
    try{await this.waitForJob(job,token,value=>{this.expansion=value;this.view.setExpansion(value.input_index,value.nodes||[]);this.renderActions();this.$('graph-status').textContent=`Perturbations · ${value.completed} / ${value.total}`;});}
    catch(e){if(token===this.selectionGeneration)this.error(e.message);}
  }
  async inspectAction(actionId,token){
    this.inspection=null;this.renderInspection();this.$('details').textContent='Running detector…';
    try{
      let job=await this.api('/api/detection/inspect',{expansion_id:this.expansion.id,action_id:actionId});
      job=await this.waitForJob(job,token);
      if(!job)return;this.inspection=job;this.renderInspection();
    }catch(e){if(token===this.selectionGeneration){this.error(e.message);this.$('details').textContent='Could not run the detector.';}}
  }
  renderActions(){
    const selected=this.view.selectedAction;const options=[new Option('Original image','original')];
    for(const node of this.expansion?.nodes||[])options.push(new Option(`Darkness ${node.action.darkness} · Saturation ${node.action.saturation}`,String(node.id)));
    this.$('action').replaceChildren(...options);this.$('action').disabled=!this.expansion?.nodes?.length;
    this.$('action').value=selected===null?'original':String(selected);
  }
  renderInspection(){
    const point=(this.analysis?.points||[]).find(p=>p.index===this.selected),job=this.inspection;
    this.$('selection').hidden=!point;this.$('inspector-empty').hidden=Boolean(point);
    if(!point)return;
    this.$('selected-name').textContent=point.name||`Image ${point.index+1}`;
    const base=job?job.id:this.analysis.id;
    this.$('original').src=`/api/detection/jobs/${encodeURIComponent(base)}/images/${point.index}/original`;
    this.$('original-caption').textContent=job?'Original detections':'Original image';
    this.$('transformed-wrap').hidden=!job||job.action_id===null;
    if(job&&job.action_id!==null)this.$('transformed').src=`/api/detection/jobs/${encodeURIComponent(job.id)}/images/${point.index}/action-${job.action_id}`;
    const lines=(title,prediction)=>[title,...(prediction?.detections?.length?prediction.detections.map(d=>`${d.label} · ${(100*d.score).toFixed(1)}%`):['No detections above threshold'])].join('\n');
    this.$('details').textContent=job?[lines('Original labels',job.baseline),...(job.output?[lines('Perturbed labels',job.output),`Consistency change: ${(100*job.metrics.severity).toFixed(1)}%`]:[]),'Dataset bounding-box labels are not loaded; compare predictions for review.'].join('\n\n'):'';
  }
  render(){
    const points=this.analysis?.points||[];
    if(this.analysis?.status==='completed'&&this.graphAnalysis!==this.analysis.id){this.graphAnalysis=this.analysis.id;this.view.setData(points.map(point=>({...point,position:[point.x,point.y,point.z||0]})));}
    this.view.setLandscapeMode('measured');this.view.setDiscovery(this.discovery);this.renderInspection();
    const records=this.discovery?.records||[],list=this.$('records');list.replaceChildren();
    if(!records.length){const p=document.createElement('p');p.className='hint';p.textContent='No transformations measured yet.';list.append(p);}
    else records.forEach(item=>{const button=document.createElement('button'),action=this.discovery.actions?.find(a=>a.id===item.action_id),image=points.find(p=>p.index===item.input_index);button.type='button';button.className='detection-record';button.textContent=`${image?.name||`Image ${item.input_index+1}`} · ${action?`darkness ${action.darkness}, saturation ${action.saturation}`:item.action_id} · severity ${item.severity ?? '—'} · reward ${item.reward ?? '—'}`;button.addEventListener('click',()=>this.select(item.input_index,item));list.append(button);});
    this.controls();
  }
}
