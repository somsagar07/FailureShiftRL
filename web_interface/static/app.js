'use strict';
const $ = id => document.getElementById(id);
let sidebarCollapsed=false;
try{sidebarCollapsed=window.localStorage.getItem('failureshift.workspace-sidebar')==='collapsed';}catch{}
function renderSidebar() {
  $('workspace-shell').setAttribute('data-sidebar-collapsed',String(sidebarCollapsed));
  const button=$('workspace-sidebar-toggle'),label=`${sidebarCollapsed?'Expand':'Collapse'} workspace sidebar`;
  button.setAttribute('aria-expanded',String(!sidebarCollapsed));button.setAttribute('aria-label',label);button.title=label;
}
$('workspace-sidebar-toggle').addEventListener('click',()=>{
  sidebarCollapsed=!sidebarCollapsed;renderSidebar();
  try{window.localStorage.setItem('failureshift.workspace-sidebar',sidebarCollapsed?'collapsed':'expanded');}catch{}
});
renderSidebar();
const titles = {classification:'Image classification', summarization:'Summarization', generation:'Image generation'};
const labels = {rotation:'Rotation',darkness:'Darkness',saturation:'Saturation',style:'Style',lighting:'Lighting',detail:'Detail'};
const modelPresets = {
  classification:[{id:'google/mobilenet_v2_1.0_224',label:'MobileNetV2 · ImageNet'},{id:'microsoft/resnet-18',label:'ResNet18 · ImageNet'},
    {id:'nateraw/vit-base-beans',label:'ViT · Beans'},{id:'aaraki/vit-base-patch16-224-in21k-finetuned-cifar10',label:'ViT · CIFAR-10'}],
  summarization:[{id:'Falconsai/text_summarization',label:'T5 Small summarizer'},{id:'sshleifer/distilbart-cnn-12-6',label:'DistilBART CNN'}],
  generation:[{id:'segmind/tiny-sd',label:'Tiny Stable Diffusion'},{id:'segmind/small-sd',label:'Small Stable Diffusion'}]
};
const datasetPresets = [
  {id:'microsoft/cats_vs_dogs',label:'Cats and dogs',config:'default'},
  {id:'AI-Lab-Makerere/beans',label:'Beans',config:'default',model:'nateraw/vit-base-beans'},
  {id:'uoft-cs/cifar10',label:'CIFAR-10',config:'plain_text',model:'aaraki/vit-base-patch16-224-in21k-finetuned-cifar10'},
  {id:'ILSVRC/imagenet-1k',label:'ImageNet-1k',config:'default',model:'google/mobilenet_v2_1.0_224',
    models:['google/mobilenet_v2_1.0_224','microsoft/resnet-18']}
];
const summaryDatasetPresets=[
  {id:'knkarthick/samsum',label:'SAMSum',config:'default'},
  {id:'abisee/cnn_dailymail',label:'CNN/DailyMail',config:'3.0.0'}
];
const generationDatasetPresets=[
  {id:'geneval2',label:'GenEval 2',config:'default',source:'benchmark'},
  {id:'nateraw/parti-prompts',label:'PartiPrompts',config:'default',source:'huggingface'}
];
// Summary inputs retain their own selections when another workspace is opened.
const summaryInput={dataset:null,loading:false,options:null,optionsLoading:false,optionsRequest:0,optionsTimer:null,
  splits:[],index:0,offset:0,items:[],pageRequest:0,pageLoading:false,sample:null,sampleRequest:0,sampleLoading:false};
const generationInput={dataset:null,loading:false,options:null,optionsLoading:false,optionsRequest:0,optionsTimer:null,
  splits:[],index:0,offset:0,items:[],pageRequest:0,pageLoading:false,sample:null,sampleRequest:0,sampleLoading:false};
let catalog, task='classification', preview=null, busy=false;
let previewRequest=0,previewTimer=null;
let dataset=null, datasetLoading=false, imageIndex=0, imageOffset=0, imageItems=[];
let pageRequest=0, pageLoading=false;
let datasetOptions=null, datasetOptionsLoading=false, datasetOptionsRequest=0, datasetOptionsTimer=null;
let selectedSplits=[];
const selections={}, textInputs={};
let editingAction=null;
const embeddingExplorer=new EmbeddingExplorer({api,onInspect:inspectEmbedding,onSelectInput:syncDatasetImage,describeAction:actionDescription,
  onStatus:message=>{$('preview-status').textContent=message;},onError:error,
  onBusy:value=>{busy=value;updateControls();}});

let discovery;
discovery=new DiscoveryController({api,explorer:embeddingExplorer,onChange:()=>updateControls(),onCheckpoint:path=>{
  $('model-source').value='local';$('model-name').value=path;modelChoiceChanged();
  $('model-status').textContent='Saved checkpoint selected. Click Analyze to evaluate it.';
}});
embeddingExplorer.onReady=job=>{discovery.ready(job);showDatasetSample();};
embeddingExplorer.onInspectPair=(index,action)=>discovery.inspectPair(index,action);
embeddingExplorer.isDiscoveryActive=()=>Boolean(discovery.active);

function element(tag,text,className) {
  const el=document.createElement(tag);
  if(text!==undefined)el.textContent=text;
  if(className)el.className=className;
  return el;
}
function show(id,value) { $(id).hidden=!value; }
function error(message) { $('form-error').textContent=message;show('form-error',Boolean(message)); }
async function api(path,data) {
  const response=await fetch(path,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const result=await response.json();
  if(!response.ok)throw new Error(result.error || `Request failed (${response.status})`);
  return result;
}
function actionSpace() {
  return Object.fromEntries(Object.entries(selections[task]).map(([key,group])=>[key,group.enabled?[...group.values]:[]]));
}
function actionCount() {
  const groups=Object.values(selections[task] || {});
  if(!groups.some(group=>group.enabled && group.values.length))return 0;
  if(groups.some(group=>group.enabled && !group.values.length))return 0;
  return task==='summarization'?groups[0].values.length:groups.reduce((count,group)=>count*(group.enabled?group.values.length:1),1);
}
function usesSummaryDataset() {return task==='summarization'&&$('summary-source').value!=='text';}
function usesGenerationDataset() {return task==='generation'&&$('generation-source').value!=='text';}
function activeTextDataset() {return usesSummaryDataset()?summaryInput:usesGenerationDataset()?generationInput:null;}
function updateControls() {
  const count=actionCount();
  $('action-count').textContent=`${count} combination${count===1?'':'s'}${task==='classification'?' per image':task==='generation'?' per prompt':''}`;
  $('action-error').textContent=count?'':'Add at least one action to analyze.';
  show('action-error',!count);
  const images=task==='classification',textDataset=activeTextDataset(),activeDataset=images?dataset:textDataset?.dataset;
  const inputReady=images||textDataset?Boolean(activeDataset):Boolean($('source-text').value.trim());
  const inputLoading=images?(datasetLoading||pageLoading):Boolean(textDataset&&(textDataset.loading||textDataset.pageLoading));
  $('generate').disabled=busy||Boolean(discovery?.active)||inputLoading||!count||!inputReady||!selectedModel().name;
  $('generate').textContent=busy?'Analyzing…':'Analyze';
  $('config-fields').disabled=busy||(images&&datasetLoading)||Boolean(textDataset?.loading);
  $('load-dataset').disabled=datasetLoading||busy||($('dataset-source').value==='huggingface'&&(!datasetOptions||datasetOptionsLoading||!selectedSplits.length));
  $('load-summary-dataset').disabled=busy||summaryInput.loading||summaryInput.optionsLoading||!summaryInput.options||!summaryInput.splits.length;
  $('load-generation-dataset').disabled=busy||generationInput.loading||generationInput.optionsLoading||!generationInput.options||!generationInput.splits.length;
  const imagesUnavailable=(!images&&!textDataset)||busy||inputLoading||!activeDataset;
  $('image-select').disabled=imagesUnavailable;
  const offset=images?imageOffset:textDataset?.offset||0;
  $('previous-images').disabled=imagesUnavailable||offset===0;
  $('next-images').disabled=imagesUnavailable||offset+100>=activeDataset?.count;
  $('dataset-summary').textContent=images&&dataset?`${dataset.count} images × ${count} combinations per image = ${dataset.count*count} combinations.`:'';
  $('summary-dataset-count').textContent=usesSummaryDataset()&&activeDataset?`${activeDataset.count} articles × ${count} edits per article = ${activeDataset.count*count} combinations.`:'';
  $('generation-dataset-count').textContent=usesGenerationDataset()&&activeDataset?`${activeDataset.count} prompts × ${count} combinations per prompt = ${activeDataset.count*count} combinations.`:'';
  document.querySelectorAll('.domain').forEach(button=>button.disabled=busy||datasetLoading||summaryInput.loading||generationInput.loading);
}
function clearPreview() {
  previewRequest++;clearTimeout(previewTimer);busy=false;
  discovery.invalidate();embeddingExplorer.reset();
  preview=null;show('graph-wrap',false);show('details',false);
  $('preview-status').textContent='Choose your input, then Analyze to build the search space.';
  showDatasetSample();
  updateControls();
}
function renderActions() {
  if(!selections[task])selections[task]=Object.fromEntries(Object.entries(catalog.tasks[task]).map(([key,values])=>[key,{enabled:true,values:[...values]}]));
  const editor=$('action-editor'),chips=$('action-chips'),add=$('action-add');editor.replaceChildren();chips.replaceChildren();
  const text=task==='summarization',groupKey='actions';
  const keys=text?catalog.tasks[task][groupKey]:Object.keys(catalog.tasks[task]);
  const active=key=>text?selections[task][groupKey].values.includes(key):selections[task][key].enabled;
  const name=key=>text?key.replaceAll('_',' ').replace(/^./,letter=>letter.toUpperCase()):labels[key];
  const available=keys.filter(key=>!active(key));
  const placeholder=element('option',available.length?'＋ Add action…':'All actions added');placeholder.value='';
  add.replaceChildren(placeholder,...available.map(key=>{const option=element('option',name(key));option.value=key;return option;}));
  add.value='';add.disabled=!available.length;
  if(!keys.includes(editingAction)||!active(editingAction))editingAction=null;
  for(const key of keys) {
    if(!active(key))continue;
    const chip=element('div',undefined,'action-chip');
    const label=element(text?'span':'button',undefined,'action-chip-label');
    const caption=element('span',text?name(key):`${name(key)} (${selections[task][key].values.length})`);caption.id=`action-caption-${key}`;
    if(!text) {
      label.type='button';label.id=`edit-action-${key}`;label.title=`Edit ${name(key).toLowerCase()} values`;
      label.setAttribute('aria-expanded',String(editingAction===key));label.setAttribute('aria-controls','action-editor');
      const symbol=element('span',({rotation:'↻',darkness:'◐',saturation:'◒',style:'◈',lighting:'☼',detail:'≋'})[key],'action-symbol');symbol.setAttribute('aria-hidden','true');
      label.append(symbol,caption);
      const arrow=element('span','⌄','action-chevron');arrow.setAttribute('aria-hidden','true');label.append(arrow);
      label.addEventListener('click',()=>{editingAction=editingAction===key?null:key;renderActions();$(`edit-action-${key}`).focus();});
    } else label.append(caption);
    const remove=element('button','×','action-chip-remove');remove.type='button';remove.id=`remove-action-${key}`;
    remove.setAttribute('aria-label',`Remove ${name(key)}`);remove.title=`Remove ${name(key).toLowerCase()}`;
    remove.addEventListener('click',()=>{
      if(text){const group=selections[task][groupKey];group.values=group.values.filter(value=>value!==key);group.enabled=Boolean(group.values.length);}
      else selections[task][key].enabled=false;
      if(editingAction===key)editingAction=null;
      clearPreview();renderActions();add.focus();
    });
    chip.append(label,remove);chips.append(chip);
  }
  if(editingAction) {
    const key=editingAction,defaults=catalog.tasks[task][key],group=selections[task][key];
    const label=element('label',`${name(key)} values`);label.setAttribute('for',`action-${key}`);
    const select=element('select');select.id=`action-${key}`;
    const all=element('option',`All values (${defaults.length})`);all.value='all';select.append(all);
    defaults.forEach((value,index)=>{const option=element('option',`${value}${key==='rotation'?'°':''}`);option.value=`value-${index}`;select.append(option);});
    select.value=group.values.length===defaults.length?'all':`value-${defaults.indexOf(group.values[0])}`;
    select.addEventListener('change',()=>{
      group.values=select.value==='all'?[...defaults]:[defaults[Number(select.value.slice(6))]];
      $(`action-caption-${key}`).textContent=`${name(key)} (${group.values.length})`;
      clearPreview();updateControls();
    });
    editor.append(label,select);
  }
  editor.hidden=!editingAction;
  updateControls();
}
$('action-add').addEventListener('change',()=>{
  const key=$('action-add').value;if(!key)return;
  if(task==='summarization') {
    const group=selections[task].actions,defaults=catalog.tasks[task].actions;
    if(!defaults.includes(key)||group.values.includes(key))return;
    group.values=defaults.filter(value=>value===key||group.values.includes(value));group.enabled=true;
  } else {
    const group=selections[task][key];if(!group||group.enabled)return;
    group.enabled=true;editingAction=key;
  }
  clearPreview();renderActions();(task==='summarization'?$(`remove-action-${key}`):$(`edit-action-${key}`)).focus();
});
function setTask(next) {
  pageRequest++;pageLoading=false;
  summaryInput.pageRequest++;summaryInput.pageLoading=false;summaryInput.sampleRequest++;summaryInput.sampleLoading=false;
  generationInput.pageRequest++;generationInput.pageLoading=false;generationInput.sampleRequest++;generationInput.sampleLoading=false;
  if(catalog && task!=='classification')textInputs[task]=$('source-text').value;
  if(task!==next)$('model-name').value='';
  task=next;editingAction=null;
  $('domain-title').textContent=titles[task];
  $('workspace-description').textContent=task==='classification'?'Choose a model, load your images, and explore actions.':task==='summarization'?'Choose a model, load your articles, and explore summarization failures.':'Choose a model, load your prompts, and explore style, lighting, and detail combinations.';
  $('viewer-title').textContent='Search space';
  document.querySelectorAll('.domain').forEach(button=>{const selected=button.dataset.task===task;button.classList.toggle('active',selected);button.setAttribute('aria-current',selected?'page':'false');});
  show('image-fields',task==='classification');show('summary-fields',task==='summarization');show('generation-fields',task==='generation');
  show('text-fields',task!=='classification'&&!activeTextDataset());
  $('source-label').textContent=task==='generation'?'Prompt':'Text';
  $('source-text').value=textInputs[task] ?? (task==='generation'?catalog.prompt:catalog.text);
  $('source-text').required=task!=='classification'&&!activeTextDataset();
  $('source-help').textContent=task==='generation'?'Describe any scene. Analyze compares prompt combinations; Find failure generates images and reviews content, instructions, and visual defects.':'Analyze compares text edits; Find failure generates and evaluates summaries.';
  $('action-help').textContent=task==='classification'?'':task==='generation'?'Combine Style, Lighting, and Detail values. Removed groups add no instruction to the original prompt.':'Text edits run separately.';
  show('action-help',task!=='classification');
  populatePresets('model-preset',modelPresets[task]);modelChoiceChanged();
  clearPreview();renderInputBrowser();
  if(usesSummaryDataset()&&summaryInput.dataset)loadSummarySample();
  if(usesGenerationDataset()&&generationInput.dataset)loadGenerationSample();
  error('');renderActions();
}
function populatePresets(id,choices) {
  $(id).replaceChildren(...[...choices,{id:'other',label:'Other…'}].map(choice=>{const option=element('option',choice.label);option.value=choice.id;return option;}));
  $(id).value=choices[0].id;
}
function selectedModel() {
  const source=$('model-source').value;
  return {source,name:source==='huggingface'&&$('model-preset').value!=='other'?$('model-preset').value:$('model-name').value.trim()};
}
function modelChoiceChanged() {
  $('model-status').textContent='';clearPreview();
  const huggingface=$('model-source').value==='huggingface';
  const custom=huggingface&&$('model-preset').value==='other';
  show('hf-model-fields',huggingface);show('model-name-fields',!huggingface||custom);
  $('model-label').textContent=huggingface?'Hugging Face model ID':'Local model directory';
  $('model-name').placeholder=huggingface?`e.g. ${modelPresets[task][0].id}`:task==='generation'?'/path/to/stable-diffusion-pipeline':'/path/to/saved-huggingface-model';
  $('model-name').required=custom||!huggingface;
  const help={
    classification:huggingface?'Runs locally · downloads on first use.':'Choose a saved Hugging Face model directory with pretrained weights and image processor configuration.',
    summarization:huggingface?'Uses the selected encoder-decoder model’s summarization encoder. Required model and tokenizer files download on first use.':'Choose a saved encoder-decoder model directory with pretrained weights and tokenizer. Uses its summarization encoder.',
    generation:huggingface?'Stable Diffusion 1/2 pipelines use their CLIP text encoder. Only the required conditioning files download; cached files are reused. SDXL and FLUX are unsupported.':'Choose a Diffusers Stable Diffusion 1/2 pipeline directory with its tokenizer and CLIP text encoder. SDXL and FLUX are unsupported.'
  };
  $('model-help').textContent=help[task];
  updateDatasetModelChoice();
  updateControls();
}
function modelSourceChanged() {
  $('model-name').value='';modelChoiceChanged();
}
function actionDescription(action) {
  if(!action)return 'Original input.';
  if('rotation' in action)return `Rotation: ${action.rotation}° · Darkness: ${action.darkness} · Saturation: ${action.saturation}`;
  if(action.name)return action.name.replaceAll('_',' ');
  return Object.entries(action).filter(([key,value])=>key!=='id'&&value!==null&&value!==undefined).map(([key,value])=>`${labels[key]||key}: ${value}`).join(' · ');
}
function displayInput(container,input) {
  const el=input.image?element('img'):element('pre',input.text);
  if(input.image){el.src=input.image;el.alt=input.name||'Input preview';el.addEventListener('error',()=>{if($(container).children[0]===el)error('Could not read this dataset image. Check the file/image column or choose another image.');});}
  $(container).replaceChildren(el);
}
function displayRequirements(container,checks=[]) {
  const disclosure=$(container);disclosure.hidden=!checks.length;
  const list=element('ol');
  list.append(...checks.map(check=>element('li',`${check.question} · Expected: ${check.answer}${check.skill?` · ${check.skill}`:''}`)));
  disclosure.replaceChildren(element('summary',`Prompt requirements (${checks.length})`),list);
}
function showDatasetSample() {
  const summary=usesSummaryDataset(),textDataset=activeTextDataset(),currentDataset=textDataset?textDataset.dataset:dataset;
  const index=textDataset?textDataset.index:imageIndex,items=textDataset?textDataset.items:imageItems;
  const item=items.find(item=>item.index===index);
  const visible=(task==='classification'||Boolean(textDataset))&&Boolean(currentDataset)&&Boolean(item)&&!embeddingExplorer.job;
  show('dataset-preview',visible);
  show('dataset-preview-reference',false);$('dataset-preview-reference-text').textContent='';
  displayRequirements('dataset-preview-requirements');
  if(!visible){$('dataset-preview-image').replaceChildren();$('dataset-preview-name').textContent='';return;}
  const name=`${item.index+1}. ${item.name}`;
  if(textDataset) {
    const sample=textDataset.sample?.index===index?textDataset.sample:null;
    displayInput('dataset-preview-image',{text:sample?.text||`Loading ${summary?'article':'prompt'}…`});
    if(summary) {
      $('dataset-preview-reference-text').textContent=sample?.summary||'No reference summary for this article.';
      show('dataset-preview-reference',Boolean(sample));
    } else displayRequirements('dataset-preview-requirements',sample?.checks||[]);
  } else displayInput('dataset-preview-image',{image:`/datasets/${dataset.id}/images/${imageIndex}.png`,name});
  $('dataset-preview-name').textContent=name;
  $('preview-status').textContent='';
}
function inspectEmbedding(original,node,config) {
  if(!original){show('details',false);return;}
  displayInput('original-view',original);displayInput('selected-view',node||original);
  const reference=config.task==='summarization'?(original.reference_summary??original.summary):null;
  show('original-reference',Boolean(reference));$('original-reference-text').textContent=reference||'';
  displayRequirements('original-requirements',config.task==='generation'?original.checks||[]:[]);
  $('selected-title').textContent=node?(config.task==='generation'?`Action ${node.id}`:`Combination ${node.id+1}`):'Original input';
  const description=node&&config.task==='classification'?Object.entries({rotation:'Rotation',darkness:'Darkness',saturation:'Saturation'}).map(([key,label])=>{
    const value=config.action_space?.[key]?.length===0?'Off':`${node.action[key]}${key==='rotation'?'°':''}`;
    return `${label}: ${value}`;
  }).join('\n'):actionDescription(node?.action);
  $('selected-action').textContent=description+(node?.unchanged?' — This edit did not change this input.':'');
  $('embedding-similarity').textContent=node?`Cosine similarity to original: ${node.similarity.toFixed(3)}`:'';
  show('embedding-similarity',Boolean(node));show('details',true);
}
async function pollPreview(id,request) {
  if(request!==previewRequest)return;
  try {
    const job=await api(`/api/previews/${id}?input=${task==='classification'?imageIndex:activeTextDataset()?.index||0}`);
    if(request!==previewRequest)return;
    preview=job;
    if(job.status==='queued'||job.status==='running') {
      $('preview-status').textContent=`Generating ${job.completed} of ${job.total} combinations…`;
      previewTimer=setTimeout(()=>pollPreview(id,request),250);return;
    }
    busy=false;updateControls();
    if(job.status==='error'){error(job.error);$('preview-status').textContent='Could not generate combinations.';return;}
    await embeddingExplorer.load(job,$('analysis-device').value||'auto');
  } catch(e){if(request!==previewRequest)return;busy=false;updateControls();error(e.message);$('preview-status').textContent='Connection interrupted. Click Analyze to try again.';}
}
$('preview-form').addEventListener('submit',async event=>{
  event.preventDefault();error('');if(busy||discovery.active||!actionCount()||(task==='classification'&&(datasetLoading||pageLoading||!dataset)))return;
  const textDataset=activeTextDataset();
  if(textDataset&&(textDataset.loading||textDataset.pageLoading||!textDataset.dataset))return;
  const model=selectedModel();
  if(!model.name){error('Choose a model to compute embeddings.');return;}
  if(task!=='classification'&&!textDataset&&!$('source-text').value.trim()){error('Enter source text or a prompt to analyze.');return;}
  const config={task,model,dataset_id:task==='classification'?dataset.id:textDataset?.dataset.id||'',text:textDataset?'':$('source-text').value,action_space:actionSpace()};
  clearPreview();const request=previewRequest;
  busy=true;updateControls();
  $('preview-status').textContent='Generating input combinations…';
  try {
    const job=await api('/api/previews',config);if(request!==previewRequest)return;
    preview=job;await pollPreview(job.id,request);
  } catch(e){if(request!==previewRequest)return;busy=false;updateControls();error(e.message);$('preview-status').textContent='Could not generate combinations.';}
});
$('model-source').addEventListener('change',modelSourceChanged);
$('model-preset').addEventListener('change',modelChoiceChanged);
$('model-name').addEventListener('input',modelChoiceChanged);
$('source-text').addEventListener('input',clearPreview);
$('analysis-device').addEventListener('change',clearPreview);
$('reset-actions').addEventListener('click',()=>{delete selections[task];editingAction=null;clearPreview();renderActions();});
document.querySelectorAll('.domain').forEach(button=>button.addEventListener('click',()=>setTask(button.dataset.task)));
function clearDataset() {
  pageRequest++;pageLoading=false;
  dataset=null;imageIndex=0;imageOffset=0;imageItems=[];clearPreview();show('input-browser',false);
  $('image-select').replaceChildren();$('image-page').textContent='';
  $('dataset-label-status').textContent='';show('dataset-label-status',false);
  $('dataset-status').textContent='Select and load your input data.';updateControls();
}
function datasetPresetChanged() {
  const preset=datasetPresets.find(choice=>choice.id===$('dataset-preset').value);
  show('dataset-name-fields',!preset);
  $('dataset-name').value=preset?.id||'';
  setDatasetOptions('dataset-config',preset?[preset.config]:[],true);$('dataset-config').value=preset?.config||'';
  $('dataset-subset').value='';
  resetDatasetOptions();clearDataset();updateDatasetModelChoice();
  if($('dataset-source').value==='huggingface')return loadDatasetOptions();
}
function setDatasetOptions(id,values,automatic=false) {
  $(id).replaceChildren(...(automatic?['',...values]:values).map(value=>{
    const option=element('option',value||'Automatic');option.value=value;return option;
  }));
  $(id).value=automatic?'':values[0]||'';
}
function resetDatasetOptions() {
  clearTimeout(datasetOptionsTimer);datasetOptionsRequest++;datasetOptions=null;datasetOptionsLoading=false;selectedSplits=[];
  $('dataset-split-options').replaceChildren();$('dataset-splits').open=false;
  $('dataset-split-summary').textContent='Select a dataset';$('dataset-options-status').textContent='';show('dataset-refresh',false);
  for(const id of ['dataset-column','dataset-label-column'])setDatasetOptions(id,[],true);
}
function updateDatasetModelChoice() {
  const preset=datasetPresets.find(choice=>choice.id===$('dataset-name').value.trim());
  const model=preset?.model&&modelPresets.classification.find(choice=>choice.id===preset.model);
  const available=task==='classification'&&$('dataset-source').value==='huggingface';
  const compatible=Boolean(model)&&(preset.models||[model.id]).includes(selectedModel().name);
  const visible=available&&model&&!compatible;
  show('dataset-access-note',available&&preset?.id==='ILSVRC/imagenet-1k');
  show('dataset-model-choice',Boolean(visible));
  if(visible){$('dataset-model-help').textContent=`Model for ${preset.label} labels: ${model.label}.`;$('dataset-use-model').textContent=`Use ${model.label}`;}
}
function renderDatasetSplits() {
  $('dataset-split-summary').textContent=selectedSplits.length?selectedSplits.join(' + '):'Choose splits';
  $('dataset-split-options').replaceChildren(...datasetOptions.splits.map(split=>{
    const row=element('label'),input=element('input');input.type='checkbox';input.value=split.name;input.checked=selectedSplits.includes(split.name);
    input.addEventListener('change',()=>{
      selectedSplits=datasetOptions.splits.filter(option=>option.name===split.name?input.checked:selectedSplits.includes(option.name)).map(option=>option.name);
      $('dataset-split-summary').textContent=selectedSplits.length?selectedSplits.join(' + '):'Choose splits';
      clearDataset();
    });
    row.append(input,element('span',split.name));
    if(Number.isSafeInteger(split.count))row.append(element('span',`${split.count.toLocaleString()} rows`,'split-count'));
    return row;
  }));
}
async function loadDatasetOptions() {
  clearTimeout(datasetOptionsTimer);
  const name=$('dataset-name').value.trim(),config=$('dataset-config').value;
  resetDatasetOptions();
  if(!name||$('dataset-source').value!=='huggingface'){updateControls();return;}
  const request=datasetOptionsRequest;
  datasetOptionsLoading=true;$('dataset-split-summary').textContent='Loading splits…';updateControls();
  try {
    const info=await api(`/api/datasets/options?name=${encodeURIComponent(name)}&config=${encodeURIComponent(config)}`);
    if(request!==datasetOptionsRequest)return;
    datasetOptions=info;
    setDatasetOptions('dataset-config',info.configs);$('dataset-config').value=info.config;
    setDatasetOptions('dataset-column',info.image_columns,true);setDatasetOptions('dataset-label-column',info.label_column_options||info.label_columns,true);
    selectedSplits=[info.splits.find(split=>split.name==='train')?.name||info.splits[0].name];
    renderDatasetSplits();
    $('dataset-options-status').textContent=info.label_columns.length?'Labels read from the dataset.':info.label_column_options?.length?'Choose a label column in Dataset options.':'No label column found.';
  } catch(e) {
    if(request!==datasetOptionsRequest)return;
    $('dataset-split-summary').textContent='Splits unavailable';$('dataset-options-status').textContent=e.message;show('dataset-refresh',true);
  } finally {
    if(request===datasetOptionsRequest){datasetOptionsLoading=false;updateControls();}
  }
}
async function loadImagePage(offset,selectedIndex=offset) {
  if(task!=='classification'||!dataset)return false;
  const datasetId=dataset.id,request=++pageRequest;
  pageLoading=true;updateControls();
  try {
    const page=await api(`/api/datasets/${datasetId}/items?offset=${offset}&limit=100`);
    if(request!==pageRequest||dataset?.id!==datasetId||task!=='classification')return false;
    imageItems=page.items;
    imageOffset=page.offset;imageIndex=page.items.some(item=>item.index===selectedIndex)?selectedIndex:page.items[0].index;
    $('image-select').replaceChildren(...page.items.map(item=>{const option=element('option',`${item.index+1}. ${item.name}`);option.value=String(item.index);return option;}));
    $('image-select').value=String(imageIndex);
    $('image-page').textContent=`${offset+1}–${offset+page.items.length} of ${page.count}`;
    show('input-browser',true);
    showDatasetSample();
    return true;
  } catch(e) {
    if(request===pageRequest&&dataset?.id===datasetId)throw e;
    return false;
  } finally {
    if(request===pageRequest){pageLoading=false;updateControls();}
  }
}
async function syncDatasetImage(index) {
  if(usesGenerationDataset()&&generationInput.dataset) {
    generationInput.index=index;
    if(generationInput.items.some(item=>item.index===index)) {
      generationInput.pageRequest++;generationInput.pageLoading=false;renderInputBrowser();await loadGenerationSample();
    } else try{await loadGenerationPage(Math.floor(index/100)*100,index);}catch(e){error(e.message);}
    return generationInput.sample?.index===index?generationInput.sample:null;
  }
  if(usesSummaryDataset()&&summaryInput.dataset) {
    summaryInput.index=index;
    if(summaryInput.items.some(item=>item.index===index)) {
      summaryInput.pageRequest++;summaryInput.pageLoading=false;renderInputBrowser();await loadSummarySample();
    } else try{await loadSummaryPage(Math.floor(index/100)*100,index);}catch(e){error(e.message);}
    return summaryInput.sample?.index===index?{...summaryInput.sample,reference_summary:summaryInput.sample.summary}:null;
  }
  if(task!=='classification'||!dataset)return;
  imageIndex=index;
  if([...$('image-select').children].some(option=>option.value===String(index))) {
    pageRequest++;pageLoading=false;$('image-select').value=String(index);updateControls();return;
  }
  try{await loadImagePage(Math.floor(index/100)*100,index);}catch(e){error(e.message);}
}
async function datasetReady(info) {
  while(info.status==='loading') {
    $('dataset-status').textContent=info.source==='huggingface'?'Loading the Hugging Face dataset…':'Reading the image folder…';
    await new Promise(resolve=>setTimeout(resolve,700));info=await api(`/api/datasets/${info.id}`);
  }
  if(info.status!=='ready')throw new Error(info.error||'The dataset could not be loaded.');
  dataset=info;await loadImagePage(0);
  const splitText=info.splits?.length?` · ${info.splits.join(' + ')}`:'';
  $('dataset-status').textContent=`${info.count.toLocaleString()} images loaded${splitText}${info.subset?` · rows ${info.subset}`:''}.`;
  const loaded=info.label_count||0,missing=info.count-loaded;
  $('dataset-label-status').textContent=`${loaded} / ${info.count} dataset labels loaded${missing?` · ${info.label_message||`${missing} missing labels`}`:''}`;
  $('dataset-label-status').className=missing?'hint needs-review':'hint';show('dataset-label-status',true);
}
$('load-dataset').addEventListener('click',async()=>{
  if(busy||datasetLoading||($('dataset-source').value==='huggingface'&&(!datasetOptions||datasetOptionsLoading||!selectedSplits.length)))return;
  error('');clearDataset();datasetLoading=true;updateControls();
  const source=$('dataset-source').value;
  const data=source==='folder'?{source,path:$('folder-path').value}:{source,name:$('dataset-name').value.trim(),splits:[...selectedSplits],subset:$('dataset-subset').value.trim(),config:$('dataset-config').value,column:$('dataset-column').value,label_column:$('dataset-label-column').value};
  try{await datasetReady(await api('/api/datasets',data));}
  catch(e){dataset=null;error(e.message);$('dataset-status').textContent='Input data could not be loaded.';}
  finally{datasetLoading=false;updateControls();}
});
$('dataset-source').addEventListener('change',()=>{show('folder-fields',$('dataset-source').value==='folder');show('hf-dataset-fields',$('dataset-source').value==='huggingface');clearDataset();updateDatasetModelChoice();return loadDatasetOptions();});
$('dataset-preset').addEventListener('change',datasetPresetChanged);
for(const id of ['folder-path','dataset-subset'])$(id).addEventListener('input',clearDataset);
for(const id of ['dataset-column','dataset-label-column'])$(id).addEventListener('change',clearDataset);
$('dataset-name').addEventListener('input',()=>{
  resetDatasetOptions();setDatasetOptions('dataset-config',[],true);clearDataset();updateDatasetModelChoice();
  datasetOptionsTimer=setTimeout(loadDatasetOptions,500);
});
$('dataset-name').addEventListener('change',()=>{clearDataset();return loadDatasetOptions();});
$('dataset-config').addEventListener('change',()=>{clearDataset();return loadDatasetOptions();});
$('dataset-refresh').addEventListener('click',loadDatasetOptions);
$('dataset-use-model').addEventListener('click',()=>{
  const preset=datasetPresets.find(choice=>choice.id===$('dataset-name').value.trim());
  if(!preset?.model)return;
  $('model-source').value='huggingface';$('model-preset').value=preset.model;modelChoiceChanged();
});
$('browse-folder').addEventListener('click',()=>$('folder-files').click());
$('folder-files').addEventListener('change',async()=>{
  const files=Array.from($('folder-files').files).filter(file=>/\.(png|jpe?g|webp|bmp|gif|tiff?)$/i.test(file.name));
  if(!files.length){error('Choose a folder containing images.');return;}
  if(files.length>10000||files.some(file=>file.size>20*1024*1024)){error('Browser uploads support up to 10,000 images under 20 MB each. You can enter a folder path instead.');return;}
  error('');clearDataset();datasetLoading=true;updateControls();
  try {
    const name=(files[0].webkitRelativePath||files[0].name).split('/')[0];
    const info=await api('/api/datasets',{source:'upload',name,files:files.map(file=>({name:file.webkitRelativePath||file.name,size:file.size}))});
    for(let index=0;index<files.length;index++) {
      $('dataset-status').textContent=`Loading image ${index+1} of ${files.length}…`;
      const response=await fetch(`/api/datasets/${info.id}/files/${index}`,{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:files[index]});
      if(!response.ok)throw new Error((await response.json()).error||'Image upload failed.');
    }
    await datasetReady(await api(`/api/datasets/${info.id}/finish`,{}));
  } catch(e){dataset=null;error(e.message);$('dataset-status').textContent='Folder could not be loaded.';}
  finally{datasetLoading=false;updateControls();$('folder-files').value='';}
});
async function showSelectedImage() {
  const textDataset=activeTextDataset();
  if(textDataset) {
    if(embeddingExplorer.job)await embeddingExplorer.focus(textDataset.index);
    else {showDatasetSample();await (usesGenerationDataset()?loadGenerationSample():loadSummarySample());}
  } else if(task==='classification') {
    if(embeddingExplorer.job)await embeddingExplorer.focus(imageIndex);
    else showDatasetSample();
  }
}
$('image-select').addEventListener('change',async()=>{
  const textDataset=activeTextDataset();
  if(textDataset)textDataset.index=Number($('image-select').value);else if(task==='classification')imageIndex=Number($('image-select').value);
  await showSelectedImage();
});
for(const [id,direction] of [['previous-images',-1],['next-images',1]])$(id).addEventListener('click',async()=>{
  try {
    const loaded=usesGenerationDataset()?await loadGenerationPage(Math.max(0,generationInput.offset+direction*100)):usesSummaryDataset()?await loadSummaryPage(Math.max(0,summaryInput.offset+direction*100)):
      await loadImagePage(Math.max(0,imageOffset+direction*100));
    if(loaded)await showSelectedImage();
  } catch(e){error(e.message);}
});
function renderInputBrowser() {
  const summary=usesSummaryDataset(),textDataset=activeTextDataset(),info=textDataset?textDataset.dataset:task==='classification'?dataset:null;
  show('input-browser',Boolean(info));
  const noun=summary?'article':usesGenerationDataset()?'prompt':'image';
  $('input-browser-label').textContent=`Dataset ${noun}`;
  $('previous-images').textContent=`Previous ${noun}s`;$('next-images').textContent=`Next ${noun}s`;
  const items=textDataset?textDataset.items:imageItems,index=textDataset?textDataset.index:imageIndex,offset=textDataset?textDataset.offset:imageOffset;
  $('image-select').replaceChildren(...(info?items:[]).map(item=>{const option=element('option',`${item.index+1}. ${item.name}`);option.value=String(item.index);return option;}));
  $('image-select').value=info?String(index):'';
  $('image-page').textContent=info&&items.length?`${offset+1}–${offset+items.length} of ${info.count}`:'';
  updateControls();
}
function clearSummaryDataset() {
  summaryInput.pageRequest++;summaryInput.sampleRequest++;summaryInput.loadRequest=(summaryInput.loadRequest||0)+1;
  Object.assign(summaryInput,{dataset:null,index:0,offset:0,items:[],sample:null,loading:false,pageLoading:false,sampleLoading:false});
  $('summary-dataset-status').textContent='Select and load your articles.';
  $('summary-dataset-reference-status').textContent='';show('summary-dataset-reference-status',false);
  if(usesSummaryDataset()){clearPreview();renderInputBrowser();}else updateControls();
}
function resetSummaryOptions() {
  clearTimeout(summaryInput.optionsTimer);summaryInput.optionsRequest++;
  Object.assign(summaryInput,{options:null,optionsLoading:false,splits:[]});
  $('summary-dataset-split-options').replaceChildren();$('summary-dataset-splits').open=false;
  $('summary-dataset-split-summary').textContent='Select a dataset';$('summary-dataset-options-status').textContent='';show('summary-dataset-refresh',false);
  for(const id of ['summary-dataset-text-column','summary-dataset-summary-column'])setDatasetOptions(id,[],true);
}
function summaryDatasetPresetChanged() {
  const preset=summaryDatasetPresets.find(choice=>choice.id===$('summary-dataset-preset').value);
  show('summary-dataset-name-fields',!preset);$('summary-dataset-name').value=preset?.id||'';
  setDatasetOptions('summary-dataset-config',preset?[preset.config]:[],true);$('summary-dataset-config').value=preset?.config||'';
  $('summary-dataset-subset').value='';resetSummaryOptions();clearSummaryDataset();return loadSummaryOptions();
}
function renderSummarySplits() {
  $('summary-dataset-split-summary').textContent=summaryInput.splits.length?summaryInput.splits.join(' + '):'Choose splits';
  $('summary-dataset-split-options').replaceChildren(...summaryInput.options.splits.map(split=>{
    const row=element('label'),input=element('input');input.type='checkbox';input.value=split.name;input.checked=summaryInput.splits.includes(split.name);
    input.addEventListener('change',()=>{
      summaryInput.splits=summaryInput.options.splits.filter(option=>option.name===split.name?input.checked:summaryInput.splits.includes(option.name)).map(option=>option.name);
      $('summary-dataset-split-summary').textContent=summaryInput.splits.length?summaryInput.splits.join(' + '):'Choose splits';clearSummaryDataset();
    });
    row.append(input,element('span',split.name));
    if(Number.isSafeInteger(split.count))row.append(element('span',`${split.count.toLocaleString()} rows`,'split-count'));
    return row;
  }));
}
async function loadSummaryOptions() {
  const name=$('summary-dataset-name').value.trim(),config=$('summary-dataset-config').value;
  resetSummaryOptions();if(!name){updateControls();return;}
  const request=summaryInput.optionsRequest;summaryInput.optionsLoading=true;
  $('summary-dataset-split-summary').textContent='Loading splits…';updateControls();
  try {
    const info=await api(`/api/datasets/options?name=${encodeURIComponent(name)}&config=${encodeURIComponent(config)}`);
    if(request!==summaryInput.optionsRequest)return;
    summaryInput.options=info;
    setDatasetOptions('summary-dataset-config',info.configs);$('summary-dataset-config').value=info.config;
    for(const [id,key] of [['summary-dataset-text-column','suggested_text_column'],['summary-dataset-summary-column','suggested_summary_column']]) {
      setDatasetOptions(id,info.text_columns||[],true);$(id).value=info[key]||'';
    }
    summaryInput.splits=[info.splits.find(split=>split.name==='train')?.name||info.splits[0]?.name].filter(Boolean);renderSummarySplits();
    $('summary-dataset-options-status').textContent=info.suggested_summary_column?'Article and reference columns selected automatically.':'Choose a reference summary column in Dataset options.';
  } catch(e) {
    if(request!==summaryInput.optionsRequest)return;
    $('summary-dataset-split-summary').textContent='Splits unavailable';$('summary-dataset-options-status').textContent=e.message;show('summary-dataset-refresh',true);
  } finally {if(request===summaryInput.optionsRequest){summaryInput.optionsLoading=false;updateControls();}}
}
async function loadSummarySample() {
  if(!usesSummaryDataset()||!summaryInput.dataset)return;
  const index=summaryInput.index,datasetId=summaryInput.dataset.id;
  if(summaryInput.sample?.index===index){showDatasetSample();return;}
  const request=++summaryInput.sampleRequest;summaryInput.sampleLoading=true;summaryInput.sample=null;showDatasetSample();
  try {
    const sample=await api(`/api/datasets/${datasetId}/text/${index}`);
    if(request!==summaryInput.sampleRequest||summaryInput.dataset?.id!==datasetId||summaryInput.index!==index||!usesSummaryDataset())return;
    summaryInput.sample=sample;showDatasetSample();
  } catch(e) {
    if(request===summaryInput.sampleRequest&&usesSummaryDataset()) {
      error(e.message);if(!embeddingExplorer.job)displayInput('dataset-preview-image',{text:'Could not load this article. Select it again to retry.'});
    }
  } finally {if(request===summaryInput.sampleRequest)summaryInput.sampleLoading=false;}
}
async function loadSummaryPage(offset,selectedIndex=offset) {
  if(!usesSummaryDataset()||!summaryInput.dataset||offset>=summaryInput.dataset.count)return false;
  const datasetId=summaryInput.dataset.id,request=++summaryInput.pageRequest;
  summaryInput.pageLoading=true;updateControls();
  try {
    const page=await api(`/api/datasets/${datasetId}/items?offset=${offset}&limit=100`);
    if(request!==summaryInput.pageRequest||summaryInput.dataset?.id!==datasetId||!usesSummaryDataset())return false;
    summaryInput.items=page.items;summaryInput.offset=page.offset;
    summaryInput.index=page.items.some(item=>item.index===selectedIndex)?selectedIndex:page.items[0]?.index;
    renderInputBrowser();await loadSummarySample();
    return request===summaryInput.pageRequest&&summaryInput.dataset?.id===datasetId&&usesSummaryDataset();
  } catch(e) {if(request===summaryInput.pageRequest)throw e;return false;}
  finally {if(request===summaryInput.pageRequest){summaryInput.pageLoading=false;updateControls();}}
}
$('summary-source').addEventListener('change',()=>{
  summaryInput.pageRequest++;summaryInput.sampleRequest++;summaryInput.pageLoading=false;summaryInput.sampleLoading=false;
  show('summary-dataset-fields',usesSummaryDataset());show('text-fields',!usesSummaryDataset());$('source-text').required=!usesSummaryDataset();
  clearPreview();renderInputBrowser();if(usesSummaryDataset())loadSummarySample();
});
$('summary-dataset-preset').addEventListener('change',summaryDatasetPresetChanged);
$('summary-dataset-subset').addEventListener('input',clearSummaryDataset);
for(const id of ['summary-dataset-text-column','summary-dataset-summary-column'])$(id).addEventListener('change',clearSummaryDataset);
$('summary-dataset-name').addEventListener('input',()=>{
  resetSummaryOptions();setDatasetOptions('summary-dataset-config',[],true);clearSummaryDataset();summaryInput.optionsTimer=setTimeout(loadSummaryOptions,500);
});
for(const id of ['summary-dataset-name','summary-dataset-config'])$(id).addEventListener('change',()=>{clearSummaryDataset();return loadSummaryOptions();});
$('summary-dataset-refresh').addEventListener('click',loadSummaryOptions);
$('load-summary-dataset').addEventListener('click',async()=>{
  if(!usesSummaryDataset()||busy||summaryInput.loading||summaryInput.optionsLoading||!summaryInput.options||!summaryInput.splits.length)return;
  error('');clearSummaryDataset();summaryInput.loading=true;updateControls();const request=summaryInput.loadRequest;
  const data={source:'huggingface',task:'summarization',name:$('summary-dataset-name').value.trim(),config:$('summary-dataset-config').value,
    splits:[...summaryInput.splits],subset:$('summary-dataset-subset').value.trim(),text_column:$('summary-dataset-text-column').value,summary_column:$('summary-dataset-summary-column').value};
  try {
    let info=await api('/api/datasets',data);
    while(info.status==='loading'&&request===summaryInput.loadRequest&&usesSummaryDataset()) {
      $('summary-dataset-status').textContent='Loading the Hugging Face dataset…';
      await new Promise(resolve=>setTimeout(resolve,700));info=await api(`/api/datasets/${info.id}`);
    }
    if(request!==summaryInput.loadRequest||!usesSummaryDataset())return;
    if(info.status!=='ready')throw new Error(info.error||'The dataset could not be loaded.');
    summaryInput.dataset=info;await loadSummaryPage(0);
    if(request!==summaryInput.loadRequest||!usesSummaryDataset())return;
    $('summary-dataset-status').textContent=`${info.count.toLocaleString()} articles loaded${info.splits?.length?` · ${info.splits.join(' + ')}`:''}${info.subset?` · rows ${info.subset}`:''}.`;
    const count=info.reference_count||0,missing=info.missing_reference_count??info.count-count;
    $('summary-dataset-reference-status').textContent=`${count} / ${info.count} reference summaries loaded${missing?` · ${missing} missing. Choose a reference column or another dataset before discovery.`:''}`;
    $('summary-dataset-reference-status').className=missing?'hint needs-review':'hint';show('summary-dataset-reference-status',true);
  } catch(e) {
    if(request===summaryInput.loadRequest){summaryInput.dataset=null;renderInputBrowser();showDatasetSample();error(e.message);$('summary-dataset-status').textContent='Input data could not be loaded.';}
  } finally {if(request===summaryInput.loadRequest){summaryInput.loading=false;updateControls();}}
});
function clearGenerationDataset() {
  generationInput.pageRequest++;generationInput.sampleRequest++;generationInput.loadRequest=(generationInput.loadRequest||0)+1;
  Object.assign(generationInput,{dataset:null,index:0,offset:0,items:[],sample:null,loading:false,pageLoading:false,sampleLoading:false});
  $('generation-dataset-status').textContent='Select and load your prompts.';
  $('generation-dataset-checks-status').textContent='';show('generation-dataset-checks-status',false);
  if(usesGenerationDataset()){clearPreview();renderInputBrowser();}else updateControls();
}
function resetGenerationOptions() {
  clearTimeout(generationInput.optionsTimer);generationInput.optionsRequest++;
  Object.assign(generationInput,{options:null,optionsLoading:false,splits:[]});
  $('generation-dataset-split-options').replaceChildren();$('generation-dataset-splits').open=false;
  $('generation-dataset-split-summary').textContent='Select a dataset';$('generation-dataset-options-status').textContent='';show('generation-dataset-refresh',false);
  for(const id of ['generation-dataset-prompt-column','generation-dataset-checks-column'])setDatasetOptions(id,[],true);
  $('generation-dataset-checks-column').children[0].textContent='None';
}
function generationDatasetPresetChanged() {
  const preset=generationDatasetPresets.find(choice=>choice.id===$('generation-dataset-preset').value);
  show('generation-dataset-name-fields',!preset);$('generation-dataset-name').value=preset?.id||'';
  setDatasetOptions('generation-dataset-config',preset?[preset.config]:[],true);$('generation-dataset-config').value=preset?.config||'';
  $('generation-dataset-subset').value='';resetGenerationOptions();clearGenerationDataset();return loadGenerationOptions();
}
function renderGenerationSplits() {
  $('generation-dataset-split-summary').textContent=generationInput.splits.length?generationInput.splits.join(' + '):'Choose splits';
  $('generation-dataset-split-options').replaceChildren(...generationInput.options.splits.map(split=>{
    const row=element('label'),input=element('input');input.type='checkbox';input.value=split.name;input.checked=generationInput.splits.includes(split.name);
    input.addEventListener('change',()=>{
      generationInput.splits=generationInput.options.splits.filter(option=>option.name===split.name?input.checked:generationInput.splits.includes(option.name)).map(option=>option.name);
      $('generation-dataset-split-summary').textContent=generationInput.splits.length?generationInput.splits.join(' + '):'Choose splits';clearGenerationDataset();
    });
    row.append(input,element('span',split.name));
    if(Number.isSafeInteger(split.count))row.append(element('span',`${split.count.toLocaleString()} rows`,'split-count'));
    return row;
  }));
}
async function loadGenerationOptions() {
  const name=$('generation-dataset-name').value.trim(),config=$('generation-dataset-config').value;
  resetGenerationOptions();if(!name){updateControls();return;}
  const request=generationInput.optionsRequest;generationInput.optionsLoading=true;
  $('generation-dataset-split-summary').textContent='Loading splits…';updateControls();
  try {
    const info=await api(`/api/datasets/options?name=${encodeURIComponent(name)}&config=${encodeURIComponent(config)}`);
    if(request!==generationInput.optionsRequest)return;
    generationInput.options=info;
    setDatasetOptions('generation-dataset-config',info.configs||[],true);$('generation-dataset-config').value=info.config||'';
    setDatasetOptions('generation-dataset-prompt-column',info.text_columns||[],true);$('generation-dataset-prompt-column').value=info.suggested_prompt_column||'';
    setDatasetOptions('generation-dataset-checks-column',info.checks_columns||[],true);$('generation-dataset-checks-column').value=info.suggested_checks_column||'';
    $('generation-dataset-checks-column').children[0].textContent='None';
    generationInput.splits=[info.splits.find(split=>split.name===(name==='geneval2'?'test':'train'))?.name||info.splits[0]?.name].filter(Boolean);renderGenerationSplits();
    $('generation-dataset-options-status').textContent=name==='geneval2'?'Official GenEval 2 prompts · 800 rows. Discovery reviews content requirements; it does not report an official benchmark score.':info.suggested_checks_column?'Prompt and requirements columns selected automatically.':'No requirements selected. Discovery uses visual content review against each prompt.';
  } catch(e) {
    if(request!==generationInput.optionsRequest)return;
    $('generation-dataset-split-summary').textContent='Splits unavailable';$('generation-dataset-options-status').textContent=e.message;show('generation-dataset-refresh',true);
  } finally {if(request===generationInput.optionsRequest){generationInput.optionsLoading=false;updateControls();}}
}
async function loadGenerationSample() {
  if(!usesGenerationDataset()||!generationInput.dataset)return;
  const index=generationInput.index,datasetId=generationInput.dataset.id;
  if(generationInput.sample?.index===index){showDatasetSample();return;}
  const request=++generationInput.sampleRequest;generationInput.sampleLoading=true;generationInput.sample=null;showDatasetSample();
  try {
    const sample=await api(`/api/datasets/${datasetId}/text/${index}`);
    if(request!==generationInput.sampleRequest||generationInput.dataset?.id!==datasetId||generationInput.index!==index||!usesGenerationDataset())return;
    generationInput.sample=sample;showDatasetSample();
  } catch(e) {
    if(request===generationInput.sampleRequest&&usesGenerationDataset()) {
      error(e.message);if(!embeddingExplorer.job)displayInput('dataset-preview-image',{text:'Could not load this prompt. Select it again to retry.'});
    }
  } finally {if(request===generationInput.sampleRequest)generationInput.sampleLoading=false;}
}
async function loadGenerationPage(offset,selectedIndex=offset) {
  if(!usesGenerationDataset()||!generationInput.dataset||offset>=generationInput.dataset.count)return false;
  const datasetId=generationInput.dataset.id,request=++generationInput.pageRequest;
  generationInput.pageLoading=true;updateControls();
  try {
    const page=await api(`/api/datasets/${datasetId}/items?offset=${offset}&limit=100`);
    if(request!==generationInput.pageRequest||generationInput.dataset?.id!==datasetId||!usesGenerationDataset())return false;
    generationInput.items=page.items;generationInput.offset=page.offset;
    generationInput.index=page.items.some(item=>item.index===selectedIndex)?selectedIndex:page.items[0]?.index;
    renderInputBrowser();await loadGenerationSample();
    return request===generationInput.pageRequest&&generationInput.dataset?.id===datasetId&&usesGenerationDataset();
  } catch(e) {if(request===generationInput.pageRequest)throw e;return false;}
  finally {if(request===generationInput.pageRequest){generationInput.pageLoading=false;updateControls();}}
}
$('generation-source').addEventListener('change',()=>{
  generationInput.pageRequest++;generationInput.sampleRequest++;generationInput.pageLoading=false;generationInput.sampleLoading=false;
  show('generation-dataset-fields',usesGenerationDataset());show('text-fields',!usesGenerationDataset());$('source-text').required=!usesGenerationDataset();
  clearPreview();renderInputBrowser();if(usesGenerationDataset())loadGenerationSample();
});
$('generation-dataset-preset').addEventListener('change',generationDatasetPresetChanged);
$('generation-dataset-subset').addEventListener('input',clearGenerationDataset);
for(const id of ['generation-dataset-prompt-column','generation-dataset-checks-column'])$(id).addEventListener('change',clearGenerationDataset);
$('generation-dataset-name').addEventListener('input',()=>{
  resetGenerationOptions();setDatasetOptions('generation-dataset-config',[],true);clearGenerationDataset();generationInput.optionsTimer=setTimeout(loadGenerationOptions,500);
});
for(const id of ['generation-dataset-name','generation-dataset-config'])$(id).addEventListener('change',()=>{clearGenerationDataset();return loadGenerationOptions();});
$('generation-dataset-refresh').addEventListener('click',loadGenerationOptions);
$('load-generation-dataset').addEventListener('click',async()=>{
  if(!usesGenerationDataset()||busy||generationInput.loading||generationInput.optionsLoading||!generationInput.options||!generationInput.splits.length)return;
  error('');clearGenerationDataset();generationInput.loading=true;updateControls();const request=generationInput.loadRequest;
  const name=$('generation-dataset-name').value.trim();
  const data={source:name==='geneval2'?'benchmark':'huggingface',task:'generation',name,config:$('generation-dataset-config').value,
    splits:[...generationInput.splits],subset:$('generation-dataset-subset').value.trim(),prompt_column:$('generation-dataset-prompt-column').value,checks_column:$('generation-dataset-checks-column').value};
  try {
    let info=await api('/api/datasets',data);
    while(info.status==='loading'&&request===generationInput.loadRequest&&usesGenerationDataset()) {
      $('generation-dataset-status').textContent='Loading the prompt dataset…';
      await new Promise(resolve=>setTimeout(resolve,700));info=await api(`/api/datasets/${info.id}`);
    }
    if(request!==generationInput.loadRequest||!usesGenerationDataset())return;
    if(info.status!=='ready')throw new Error(info.error||'The dataset could not be loaded.');
    generationInput.dataset=info;await loadGenerationPage(0);
    if(request!==generationInput.loadRequest||!usesGenerationDataset())return;
    $('generation-dataset-status').textContent=`${info.count.toLocaleString()} prompts loaded${info.splits?.length?` · ${info.splits.join(' + ')}`:''}${info.subset?` · rows ${info.subset}`:''}.`;
    const count=info.checked_count||0,missing=info.missing_checks_count??info.count-count;
    $('generation-dataset-checks-status').textContent=`${count} / ${info.count} prompts have requirements${missing?` · ${missing} use visual content review against the prompt.`:'. Content checks are ready.'}`;
    $('generation-dataset-checks-status').className='hint';show('generation-dataset-checks-status',true);
  } catch(e) {
    if(request===generationInput.loadRequest){generationInput.dataset=null;renderInputBrowser();showDatasetSample();error(e.message);$('generation-dataset-status').textContent='Input data could not be loaded.';}
  } finally {if(request===generationInput.loadRequest){generationInput.loading=false;updateControls();}}
});
(async()=>{try{
  catalog=await api('/api/catalog');$('summary-source').value='dataset';$('generation-source').value='dataset';
  populatePresets('generation-dataset-preset',generationDatasetPresets);generationDatasetPresetChanged();
  populatePresets('summary-dataset-preset',summaryDatasetPresets);summaryDatasetPresetChanged();
  populatePresets('dataset-preset',datasetPresets);datasetPresetChanged();setTask('classification');
}catch(e){error(`Could not load the local server: ${e.message}`);}})();
