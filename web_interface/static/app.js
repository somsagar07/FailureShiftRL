'use strict';
const $ = id => document.getElementById(id);
const titles = {classification:'Image classification', summarization:'Summarization', generation:'Image generation'};
const labels = {rotation:'Rotation',darkness:'Darkness',saturation:'Saturation',description:'Description',profession:'Profession',setting:'Setting',actions:'Text action'};
const modelPresets = {
  classification:[{id:'google/mobilenet_v2_1.0_224',label:'MobileNetV2'},{id:'microsoft/resnet-18',label:'ResNet18'}],
  summarization:[{id:'Falconsai/text_summarization',label:'T5 Small summarizer'},{id:'sshleifer/distilbart-cnn-12-6',label:'DistilBART CNN'}],
  generation:[{id:'segmind/tiny-sd',label:'Tiny Stable Diffusion'},{id:'segmind/small-sd',label:'Small Stable Diffusion'}]
};
const datasetPresets = [
  {id:'hf-internal-testing/cats_vs_dogs_sample',label:'Cats and dogs — 50 demo images',split:'train',config:'',column:'image'},
  {id:'AI-Lab-Makerere/beans',label:'Beans — 20 test images',split:'test[:20]',config:'default',column:'image'},
  {id:'uoft-cs/cifar10',label:'CIFAR-10 — 20 test images',split:'test[:20]',config:'plain_text',column:'img'}
];
let catalog, task='classification', preview=null, busy=false;
let previewRequest=0,previewTimer=null;
let dataset=null, datasetLoading=false, imageIndex=0, imageOffset=0;
let pageRequest=0, pageLoading=false;
const selections={}, textInputs={};
const embeddingExplorer=new EmbeddingExplorer({api,onInspect:inspectEmbedding,onSelectInput:syncDatasetImage,describeAction:actionDescription,
  onStatus:message=>{$('preview-status').textContent=message;},onError:error,
  onBusy:value=>{busy=value;updateControls();}});

let discovery;
discovery=new DiscoveryController({api,explorer:embeddingExplorer,onChange:()=>updateControls(),onCheckpoint:path=>{
  $('model-source').value='local';$('model-name').value=path;modelChoiceChanged();
  $('model-status').textContent='Saved checkpoint selected. Click Analyze to evaluate it.';
}});
embeddingExplorer.onReady=job=>discovery.ready(job);
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
function updateControls() {
  const count=actionCount();
  $('action-count').textContent=`${count} combination${count===1?'':'s'}${task==='classification'?' per image':''}`;
  $('action-error').textContent=count?'':'Keep at least one action. Each enabled group needs at least one value.';
  show('action-error',!count);
  const images=task==='classification',inputReady=images?Boolean(dataset):Boolean($('source-text').value.trim());
  const inputLoading=images&&(datasetLoading||pageLoading);
  $('generate').disabled=busy||Boolean(discovery?.active)||inputLoading||!count||!inputReady||!selectedModel().name;
  $('generate').textContent=busy?'Analyzing…':'Analyze';
  $('config-fields').disabled=busy||(images&&datasetLoading);
  const imagesUnavailable=!images||busy||datasetLoading||pageLoading||!dataset;
  $('image-select').disabled=imagesUnavailable;
  $('previous-images').disabled=imagesUnavailable||imageOffset===0;
  $('next-images').disabled=imagesUnavailable||imageOffset+100>=dataset.count;
  $('dataset-summary').textContent=images&&dataset?`${dataset.count} images × ${count} combinations per image = ${dataset.count*count} combinations.`:'';
  document.querySelectorAll('.domain').forEach(button=>button.disabled=busy||datasetLoading);
}
function clearPreview() {
  previewRequest++;clearTimeout(previewTimer);busy=false;
  discovery.invalidate();embeddingExplorer.reset();
  preview=null;show('graph-wrap',false);show('details',false);
  $('preview-status').textContent='Click Analyze to explore model embeddings and action combinations.';
  updateControls();
}
function renderActions() {
  if(!selections[task])selections[task]=Object.fromEntries(Object.entries(catalog.tasks[task]).map(([key,values])=>[key,{enabled:true,values:[...values]}]));
  const editor=$('action-editor');editor.replaceChildren();
  const keys=task==='classification'?['rotation','darkness','saturation']:Object.keys(catalog.tasks[task]);
  for(const key of keys) {
    const defaults=catalog.tasks[task][key],group=selections[task][key],block=element('div',undefined,'action-group');
    const select=element('select');select.id=`action-${key}`;
    const label=element('label');
    if(task==='classification') {
      label.className='action-label';label.setAttribute('for',`enable-${key}`);
      const toggle=element('input');toggle.id=`enable-${key}`;toggle.type='checkbox';toggle.checked=group.enabled;
      label.append(toggle,element('span',labels[key]));
      select.setAttribute('aria-label',`${labels[key]} values`);select.disabled=!group.enabled;
      block.classList.toggle('inactive',!group.enabled);
      toggle.addEventListener('change',()=>{
        group.enabled=toggle.checked;select.disabled=!group.enabled;block.classList.toggle('inactive',!group.enabled);
        clearPreview();updateControls();
      });
    } else {label.textContent=labels[key];label.setAttribute('for',select.id);}
    const all=element('option',`All values (${defaults.length})`);all.value='all';select.append(all);
    defaults.forEach((value,index)=>{const option=element('option',key==='actions'?value.replaceAll('_',' '):`${value}${key==='rotation'?'°':''}`);option.value=`value-${index}`;select.append(option);});
    select.value=group.values.length===defaults.length?'all':`value-${defaults.indexOf(group.values[0])}`;
    select.addEventListener('change',()=>{
      group.values=select.value==='all'?[...defaults]:[defaults[Number(select.value.slice(6))]];
      clearPreview();updateControls();
    });
    block.append(label,select);editor.append(block);
  }
  updateControls();
}
function setTask(next) {
  pageRequest++;pageLoading=false;
  if(catalog && task!=='classification')textInputs[task]=$('source-text').value;
  if(task!==next)$('model-name').value='';
  task=next;
  $('domain-title').textContent=titles[task];
  $('workspace-description').textContent=task==='classification'?'Explore model embeddings. Select an image to see how actions change it.':task==='summarization'?'Explore how text edits change the selected summarization encoder’s embeddings.':'Explore how prompt changes affect the selected model’s CLIP text conditioning.';
  $('viewer-title').textContent='Embedding space';
  document.querySelectorAll('.domain').forEach(button=>button.classList.toggle('active',button.dataset.task===task));
  show('image-fields',task==='classification');show('text-fields',task!=='classification');
  $('source-label').textContent=task==='generation'?'Prompt template':'Text';
  $('source-text').value=textInputs[task] ?? (task==='generation'?catalog.prompt:catalog.text);
  $('source-text').required=task!=='classification';
  $('source-help').textContent=task==='generation'?'Keep {description}, {profession} and {setting}. The original uses the first default for each slot. Points compare prompt conditioning. No images are generated.':'Points compare original and edited source text using the summarization encoder. No summaries are generated.';
  $('action-help').textContent=task==='classification'?'':task==='generation'?'Choose all fixed options or one option for each prompt slot.':'Choose all 14 implemented text edits or one edit. Inflection and the unfinished verb-tense action are not included.';
  show('action-help',task!=='classification');
  populatePresets('model-preset',modelPresets[task]);modelChoiceChanged();
  clearPreview();show('input-browser',task==='classification'&&Boolean(dataset));
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
    classification:huggingface?'The selected vision model runs locally on the chosen device. Model files download on first use.':'Choose a saved Hugging Face model directory with pretrained weights and image processor configuration.',
    summarization:huggingface?'Uses the selected encoder-decoder model’s summarization encoder. Required model and tokenizer files download on first use.':'Choose a saved encoder-decoder model directory with pretrained weights and tokenizer. Uses its summarization encoder.',
    generation:huggingface?'Stable Diffusion 1/2 pipelines use their CLIP text encoder. Only the required conditioning files download; cached files are reused. SDXL and FLUX are unsupported.':'Choose a Diffusers Stable Diffusion 1/2 pipeline directory with its tokenizer and CLIP text encoder. SDXL and FLUX are unsupported.'
  };
  $('model-help').textContent=help[task];
  updateControls();
}
function modelSourceChanged() {
  $('model-name').value='';modelChoiceChanged();
}
function actionDescription(action) {
  if(!action)return 'Original input.';
  if('rotation' in action)return `Rotation: ${action.rotation}° · Darkness: ${action.darkness} · Saturation: ${action.saturation}`;
  if(action.name)return action.name.replaceAll('_',' ');
  return `Description: ${action.description} · Profession: ${action.profession} · Setting: ${action.setting}`;
}
function displayInput(container,input) {
  const el=input.image?element('img'):element('pre',input.text);
  if(input.image){el.src=input.image;el.alt='Input preview';el.addEventListener('error',()=>error('Could not read this dataset image. Check the file/image column or choose another image.'));}
  $(container).replaceChildren(el);
}
function inspectEmbedding(original,node,config) {
  if(!original){show('details',false);return;}
  displayInput('original-view',original);displayInput('selected-view',node||original);
  $('selected-title').textContent=node?`Combination ${node.id+1}`:'Original input';
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
    const job=await api(`/api/previews/${id}?input=${task==='classification'?imageIndex:0}`);
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
  const model=selectedModel();
  if(!model.name){error('Choose a model to compute embeddings.');return;}
  if(task!=='classification'&&!$('source-text').value.trim()){error('Enter source text or a prompt template to analyze.');return;}
  const config={task,model,dataset_id:task==='classification'?dataset.id:'',text:$('source-text').value,action_space:actionSpace()};
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
$('reset-actions').addEventListener('click',()=>{delete selections[task];clearPreview();renderActions();});
document.querySelectorAll('.domain').forEach(button=>button.addEventListener('click',()=>setTask(button.dataset.task)));
function clearDataset() {
  pageRequest++;pageLoading=false;
  dataset=null;imageIndex=0;imageOffset=0;clearPreview();show('input-browser',false);
  $('image-select').replaceChildren();$('image-page').textContent='';
  $('dataset-status').textContent='Select and load your input data.';updateControls();
}
function datasetPresetChanged() {
  const preset=datasetPresets.find(choice=>choice.id===$('dataset-preset').value);
  show('dataset-name-fields',!preset);
  $('dataset-name').value=preset?.id||'';$('dataset-split').value=preset?.split||'train';
  $('dataset-config').value=preset?.config||'';$('dataset-column').value=preset?.column||'';
  clearDataset();
}
async function loadImagePage(offset,selectedIndex=offset) {
  if(task!=='classification'||!dataset)return false;
  const datasetId=dataset.id,request=++pageRequest;
  pageLoading=true;updateControls();
  try {
    const page=await api(`/api/datasets/${datasetId}/items?offset=${offset}&limit=100`);
    if(request!==pageRequest||dataset?.id!==datasetId||task!=='classification')return false;
    imageOffset=page.offset;imageIndex=page.items.some(item=>item.index===selectedIndex)?selectedIndex:page.items[0].index;
    $('image-select').replaceChildren(...page.items.map(item=>{const option=element('option',`${item.index+1}. ${item.name}`);option.value=String(item.index);return option;}));
    $('image-select').value=String(imageIndex);
    $('image-page').textContent=`${offset+1}–${offset+page.items.length} of ${page.count}`;
    show('input-browser',true);
    return true;
  } catch(e) {
    if(request===pageRequest&&dataset?.id===datasetId)throw e;
    return false;
  } finally {
    if(request===pageRequest){pageLoading=false;updateControls();}
  }
}
async function syncDatasetImage(index) {
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
  $('dataset-status').textContent=`${info.count} images loaded from ${info.name}.`;
}
$('load-dataset').addEventListener('click',async()=>{
  error('');clearDataset();datasetLoading=true;updateControls();
  const source=$('dataset-source').value;
  const data=source==='folder'?{source,path:$('folder-path').value}:{source,name:$('dataset-name').value.trim(),split:$('dataset-split').value.trim(),config:$('dataset-config').value.trim(),column:$('dataset-column').value.trim()};
  try{await datasetReady(await api('/api/datasets',data));}
  catch(e){dataset=null;error(e.message);$('dataset-status').textContent='Input data could not be loaded.';}
  finally{datasetLoading=false;updateControls();}
});
$('dataset-source').addEventListener('change',()=>{show('folder-fields',$('dataset-source').value==='folder');show('hf-dataset-fields',$('dataset-source').value==='huggingface');clearDataset();});
$('dataset-preset').addEventListener('change',datasetPresetChanged);
for(const id of ['folder-path','dataset-name','dataset-split','dataset-config','dataset-column'])$(id).addEventListener('input',clearDataset);
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
  if(task==='classification'&&preview)await embeddingExplorer.focus(imageIndex);
}
$('image-select').addEventListener('change',async()=>{imageIndex=Number($('image-select').value);await showSelectedImage();});
$('previous-images').addEventListener('click',async()=>{try{if(await loadImagePage(Math.max(0,imageOffset-100)))await showSelectedImage();}catch(e){error(e.message);}});
$('next-images').addEventListener('click',async()=>{try{if(await loadImagePage(imageOffset+100))await showSelectedImage();}catch(e){error(e.message);}});
(async()=>{try{catalog=await api('/api/catalog');populatePresets('dataset-preset',datasetPresets);datasetPresetChanged();setTask('classification');}catch(e){error(`Could not load the local server: ${e.message}`);}})();
