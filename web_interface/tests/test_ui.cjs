// Run with Node.js: node tests/test_ui.cjs
// Execute the actual browser controller against a small DOM and fetch boundary.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const elements=new Map();
class Element {
  constructor(){this.value='';this.children=[];this.events={};this.dataset={};this.attrs={};this.style={};this.classList={toggle(){}};}
  set id(value){this._id=value;elements.set(value,this);}
  get id(){return this._id;}
  replaceChildren(...children){this.children=children;}
  append(...children){this.children.push(...children);}
  addEventListener(name,handler){this.events[name]=handler;}
  setAttribute(key,value){this.attrs[key]=value;if(key==='data-node')this.dataset.node=value;}
  getBoundingClientRect(){return {left:0,top:0,width:800,height:600};}
  getContext(){return new Proxy({measureText:()=>({width:40})},{get:(target,key)=>target[key]||(()=>{})});}
  closest(){return this.dataset.node===undefined?null:this;}
  setPointerCapture(){}
  hasPointerCapture(){return true;}
  releasePointerCapture(){}
  focus(){}
  querySelectorAll(){const all=[];const visit=node=>{if(node.dataset?.node!==undefined)all.push(node);for(const child of node.children||[])visit(child);};visit(this);return all;}
}
const get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
const domains=['classification','summarization','generation'].map(task=>{const el=new Element();el.dataset.task=task;return el;});
const document={getElementById:get,createElement:()=>new Element(),createElementNS:()=>new Element(),querySelectorAll:selector=>selector==='.domain'?domains:[]};
get('model-source').value='local';get('dataset-source').value='folder';
for(const [id,value] of Object.entries({'finetune-epochs':'1','finetune-lr':'0.001','finetune-device':'auto','finetune-batch-size':'auto','finetune-seed':'42','finetune-weight-decay':'0.01','finetune-max-grad-norm':'1'}))get(id).value=value;
get('finetune-cleanup').checked=true;
const preferences=new Map([['failureshift.workspace-sidebar','collapsed']]);let storageUnavailable=false;
const spaces={classification:{rotation:[-10,-5,0,5,10],darkness:[.1,.2,.3,.4,.5],saturation:[.5,.55,.6,.65,.7]},summarization:{actions:['to_lowercase','remove_punctuation']},generation:{style:['photograph','watercolor','pencil drawing','digital illustration','3D render'],lighting:['even','soft','directional','high-key','low-key'],detail:['minimal','simple','moderate','detailed','intricate']}};
let submitted,modelRequested,embeddingRequested,job,datasetRequested,resolveOldPage,resolvePreview;
let deferNextPreview=false,resolveOldOptions;
const requests=[];
const datasetId='c'.repeat(32),hfDatasetId='d'.repeat(32);
const point=index=>({index,name:index===0?'cat.png':index===1?'dog.png':`image-${index}.png`,image:index===0?'/original.png':index===1?'/dog-original.png':`/original-${index}.png`,position:[index,0,0]});
const context={document,console,setTimeout:()=>1,clearTimeout(){},window:{addEventListener(){},devicePixelRatio:1,localStorage:{getItem:key=>preferences.get(key),setItem:(key,value)=>{if(storageUnavailable)throw Error('Storage unavailable');preferences.set(key,value);}}},fetch:async(url,options)=>{
  requests.push({url,body:options.body});
  let data;
  if(url==='/api/catalog')data={tasks:spaces,text:'HELLO World',prompt:'A unique scientist in a corporate office.'};
  else if(url.startsWith('/api/datasets/options?')) {
    const query=new URL(url,'http://localhost').searchParams,name=query.get('name');
    data={name,configs:['default'],config:'default',splits:[{name:'train',count:100},{name:'test',count:30}],image_columns:['image'],label_columns:['label']};
    if(name==='uoft-cs/cifar10')Object.assign(data,{configs:['plain_text'],config:'plain_text',image_columns:['img']});
    if(name==='ILSVRC/imagenet-1k')Object.assign(data,{splits:[{name:'train',count:1281167},{name:'validation',count:50000},{name:'test',count:100000}]});
    if(name==='org/photos')Object.assign(data,{splits:[{name:'validation',count:40},{name:'test',count:30}],image_columns:['picture'],label_columns:[],label_column_options:['species']});
    if(name==='org/slow')return new Promise(resolve=>{resolveOldOptions=()=>resolve({ok:true,json:async()=>data});});
  }
  else if(url==='/api/datasets'){
    datasetRequested=JSON.parse(options.body);
    data=datasetRequested.source==='folder'?{id:datasetId,source:'folder',name:'/data/images',status:'ready',count:201}:{id:hfDatasetId,source:'huggingface',name:datasetRequested.name,status:'ready',count:1};
  }
  else if(url.startsWith(`/api/datasets/${datasetId}/items`)){
    const offset=Number(new URL(url,'http://localhost').searchParams.get('offset'));
    data={items:Array.from({length:Math.min(100,201-offset)},(_,i)=>({index:offset+i,name:['cat.png','dog.png'][offset+i]||`image-${offset+i}.png`})),count:201,offset};
    if(offset===100)return new Promise(resolve=>{resolveOldPage=()=>resolve({ok:true,json:async()=>data});});
  }
  else if(url.startsWith(`/api/datasets/${hfDatasetId}/items`))data={items:[{index:0,name:'Image 1'}],count:1,offset:0};
  else if(url==='/api/previews'){
    submitted=JSON.parse(options.body);
    if(submitted.task==='classification')job={id:'a'.repeat(32),config:submitted,status:'completed',input_count:201,action_count:25,total:5025,completed:5025,input_index:0,input_name:'cat.png',original:{image:'/original.png'},nodes:[{id:0,action:{rotation:0,darkness:0,saturation:1},image:'/changed.png'}]};
    else {
      const generation=submitted.task==='generation';
      const original={text:generation?'A unique scientist in a corporate office.':submitted.text};
      const nodes=generation?[{id:36,action:{id:36,style:'photograph',lighting:null,detail:null},text:'A unique scientist in a corporate office. Style: photograph.'}]:[
        {id:0,action:{name:'to_lowercase'},text:submitted.text.toLowerCase(),unchanged:false},
        {id:1,action:{name:'remove_punctuation'},text:submitted.text,unchanged:true}
      ];
      job={id:'a'.repeat(32),config:submitted,status:'completed',input_count:1,input_index:0,input_name:generation?'Baseline prompt':'Source text',original,nodes,action_count:nodes.length,total:nodes.length,completed:nodes.length};
    }
    data=job;
    if(deferNextPreview){deferNextPreview=false;return new Promise(resolve=>{resolvePreview=()=>resolve({ok:true,json:async()=>data});});}
  } else if(url.startsWith(`/api/previews/${'a'.repeat(32)}`))data=url.includes('input=1')?{...job,input_index:1,input_name:'dog.png',original:{image:'/dog-original.png'},nodes:[{...job.nodes[0],image:'/dog-changed.png'}]}:job;
  else if(url==='/api/models') {modelRequested=JSON.parse(options.body);data={...modelRequested,id:'b'.repeat(32),status:'ready'};}
  else if(url==='/api/embeddings') {
    embeddingRequested=JSON.parse(options.body);
    const images=submitted.task==='classification';
    data={id:'e'.repeat(32),task:submitted.task,status:'completed',points:images?Array.from({length:201},(_,index)=>point(index)):[{index:0,name:job.input_name,...job.original,position:[0,0,0]}],projection:{method:images?'PCA fitted on original images':'PCA fitted on original and action variants',variance:.85,rank:1,components:1,...(!images?{fit_count:job.nodes.length+1}:{})},feature_dim:images?4:768,feature_method:images?'test features':'final encoder token states, attention-mask mean pooled',model:submitted.model};
  } else if(url===`/api/embeddings/${'e'.repeat(32)}/expand`) {
    const index=JSON.parse(options.body).input_index;
    data=submitted.task==='classification'?{id:'f'.repeat(32),status:'completed',input_index:index,original:point(index),nodes:[{...job.nodes[0],image:index===0?'/changed.png':'/dog-changed.png',position:[index,.1,0],similarity:.9}]}:
      {id:'f'.repeat(32),status:'completed',input_index:0,original:{index:0,name:job.input_name,...job.original,position:[0,0,0]},nodes:job.nodes.map((node,index)=>({...node,position:[index+1,.1,0],similarity:node.unchanged?1:.875,neighbors:[]}))};
  }
  else throw new Error('Unexpected URL '+url);
  return {ok:true,json:async()=>data};
}};
vm.createContext(context);
for(const script of ['graph.js','embedding-view.js','embedding-controller.js','adaptation-controller.js','discovery-controller.js','app.js'])vm.runInContext(fs.readFileSync(path.join(__dirname,'../static',script),'utf8'),context);
async function flush(){for(let i=0;i<25;i++)await Promise.resolve();}
function addAction(key){get('action-add').value=key;get('action-add').events.change();}
function removeAction(key){get(`remove-action-${key}`).events.click();}
function chooseValues(key,value){
  if(get(`edit-action-${key}`).attrs['aria-expanded']!=='true')get(`edit-action-${key}`).events.click();
  get(`action-${key}`).value=value;get(`action-${key}`).events.change();
}
(async()=>{
  await flush();
  assert.equal(get('workspace-shell').attrs['data-sidebar-collapsed'],'true','startup restores the compact workspace preference');
  assert.equal(get('workspace-sidebar-toggle').attrs['aria-expanded'],'false');
  assert.match(get('workspace-sidebar-toggle').attrs['aria-label'],/expand.*workspace/i);
  const requestsBeforeSidebar=requests.length;
  get('workspace-sidebar-toggle').events.click();
  assert.equal(get('workspace-shell').attrs['data-sidebar-collapsed'],'false');
  assert.equal(get('workspace-sidebar-toggle').attrs['aria-expanded'],'true');
  assert.equal(preferences.get('failureshift.workspace-sidebar'),'expanded');
  storageUnavailable=true;get('workspace-sidebar-toggle').events.click();
  assert.equal(get('workspace-shell').attrs['data-sidebar-collapsed'],'true','the sidebar still toggles when storage is unavailable');
  storageUnavailable=false;get('workspace-sidebar-toggle').events.click();
  assert.equal(requests.length,requestsBeforeSidebar,'collapsing navigation does not reload model or analysis data');
  assert.equal(domains[0].attrs['aria-current'],'page');
  assert.equal(get('generate').disabled,true,'data must be selected before preview generation');
  assert.equal(get('action-chips').children.length,3,'classification starts with three active action chips');
  assert.equal(get('action-count').textContent,'125 combinations per image');
  assert.equal(get('action-add').disabled,true,'the add menu excludes already selected actions');
  removeAction('saturation');
  assert.equal(get('action-count').textContent,'25 combinations per image');
  assert.equal(get('action-chips').children.length,2);
  assert.deepEqual(get('action-add').children.slice(1).map(option=>option.value),['saturation']);
  for(const action of ['rotation','darkness'])removeAction(action);
  assert.equal(get('action-count').textContent,'0 combinations per image');
  assert.equal(get('generate').disabled,true,'at least one action must be selected');
  get('reset-actions').events.click();
  assert.equal(get('action-count').textContent,'125 combinations per image');
  removeAction('saturation');chooseValues('rotation','value-0');
  assert.equal(get('action-count').textContent,'5 combinations per image');
  removeAction('rotation');addAction('rotation');
  assert.equal(get('action-rotation').value,'value-0','reselecting an action preserves its selected value');
  addAction('rotation');assert.equal(get('action-chips').children.length,2,'an already active action cannot be duplicated');
  chooseValues('rotation','all');
  get('dataset-source').value='folder';get('folder-path').value='/data/images';
  await get('load-dataset').events.click();await flush();
  assert.equal(get('generate').disabled,true,'classification embeddings require a selected model');
  assert.ok(get('dataset-summary').textContent.includes('5025'));
  assert.equal(get('dataset-preview').hidden,false,'loading data shows its first sample before Analyze or model selection');
  assert.equal(get('dataset-preview-image').children[0].src,`/datasets/${datasetId}/images/0.png`);
  assert.equal(get('dataset-preview-name').textContent,'1. cat.png');
  assert.equal(get('preview-status').textContent,'','the selected sample replaces the Analyze instruction');
  get('image-select').value='1';await get('image-select').events.change();
  assert.equal(get('dataset-preview-image').children[0].src,`/datasets/${datasetId}/images/1.png`);
  assert.equal(get('dataset-preview-name').textContent,'2. dog.png');
  assert.equal(requests.some(request=>request.url==='/api/previews'||request.url==='/api/embeddings'),false,'browsing samples does not run analysis');
  get('image-select').value='0';await get('image-select').events.change();
  get('model-source').value='huggingface';get('model-source').events.change();
  assert.equal(get('hf-model-fields').hidden,false);
  assert.equal(get('model-name-fields').hidden,true);
  assert.equal(get('model-name').required,false,'a hidden custom ID must not block preset submission');
  get('model-preset').value='microsoft/resnet-18';get('model-preset').events.change();
  await get('preview-form').events.submit({preventDefault(){}});await flush();
  assert.deepEqual(submitted.action_space,{rotation:[-10,-5,0,5,10],darkness:[.1,.2,.3,.4,.5],saturation:[]});
  assert.equal(submitted.dataset_id,datasetId);
  assert.deepEqual(submitted.model,{source:'huggingface',name:'microsoft/resnet-18'});
  assert.equal(embeddingRequested.preview_id,job.id);
  assert.equal(modelRequested,undefined,'embedding loading handles model download without a second snapshot job');
  assert.equal(get('embedding-wrap').hidden,false);
  assert.equal(get('dataset-preview').hidden,true,'the completed graph replaces the raw sample preview');
  assert.equal(get('graph-wrap').hidden,true);
  assert.equal(get('details').hidden,true,'the overview starts with all originals, before selecting an image');
  get('image-select').value='0';await get('image-select').events.change();await flush();
  assert.equal(get('selected-title').textContent,'Original input');
  assert.equal(get('selected-view').children[0].src,'/original.png');
  get('embedding-action').value='0';get('embedding-action').events.change();
  assert.equal(get('selected-view').children[0].src,'/changed.png');
  assert.equal(get('selected-action').textContent,'Rotation: 0°\nDarkness: 0\nSaturation: Off','the selected sample shows all three dimensions, including disabled actions');
  assert.ok(get('embedding-similarity').textContent.includes('0.900'),'cosine similarity comes from the embedding response');
  assert.equal(get('generate').disabled,false);
  get('image-select').value='1';await get('image-select').events.change();await flush();
  assert.equal(get('selected-view').children[0].src,'/dog-original.png');
  get('embedding-action').value='0';get('embedding-action').events.change();
  assert.equal(get('selected-view').children[0].src,'/dog-changed.png');
  get('model-preset').value='other';get('model-preset').events.change();
  assert.equal(get('dataset-preview').hidden,false,'invalidating analysis restores the currently selected sample');
  assert.equal(get('dataset-preview-image').children[0].src,`/datasets/${datasetId}/images/1.png`);
  assert.equal(get('model-name-fields').hidden,false);
  assert.equal(get('model-name').required,true);
  get('model-name').value='org/model';get('model-name').events.input();
  await get('preview-form').events.submit({preventDefault(){}});await flush();
  assert.deepEqual(submitted.model,{source:'huggingface',name:'org/model'},'Other submits the custom model, not the previous preset');
  assert.equal(get('next-images').disabled,false);
  const pendingPage=get('next-images').events.click();await flush();
  const paginationDisabledWhileLoading=get('next-images').disabled;
  get('dataset-source').value='huggingface';get('dataset-source').events.change();await flush();
  assert.equal(get('dataset-preview').hidden,true,'changing datasets clears the previous image immediately');
  assert.equal(get('dataset-name-fields').hidden,true);
  await get('load-dataset').events.click();await flush();
  assert.deepEqual(datasetRequested,{source:'huggingface',name:'microsoft/cats_vs_dogs',splits:['train'],subset:'',config:'default',column:'',label_column:''});
  assert.ok(get('dataset-preset').children.every(option=>!/(demo|sample|20 test)/i.test(option.textContent)),'presets select full real datasets');
  resolveOldPage();await pendingPage;await flush();
  assert.equal(get('image-select').children.length,1,'a delayed folder response cannot replace the new dataset');
  assert.equal(get('image-select').children[0].textContent,'1. Image 1');
  assert.equal(get('image-select').value,'0');
  assert.equal(get('dataset-preview').hidden,false);
  assert.equal(get('dataset-preview-image').children[0].src,`/datasets/${hfDatasetId}/images/0.png`,'a delayed old dataset page cannot replace the selected sample');
  assert.equal(get('image-page').textContent,'1–1 of 1');
  assert.equal(get('next-images').disabled,true);
  assert.equal(get('graph-wrap').hidden,true,'changing the dataset clears the old graph');
  assert.equal(get('embedding-wrap').hidden,true,'changing the dataset clears the old embedding view');
  assert.equal(paginationDisabledWhileLoading,true,'disable pagination while a page is loading');
  get('dataset-preset').value='uoft-cs/cifar10';get('dataset-preset').events.change();await flush();
  assert.equal(get('generate').disabled,true,'changing a preset invalidates the previously loaded dataset');
  await get('load-dataset').events.click();await flush();
  assert.deepEqual(datasetRequested,{source:'huggingface',name:'uoft-cs/cifar10',splits:['train'],subset:'',config:'plain_text',column:'',label_column:''});
  const testSplit=get('dataset-split-options').children[1].children[0];
  testSplit.checked=true;testSplit.events.change();
  get('dataset-subset').value='[20:30]';get('dataset-subset').events.input();
  assert.equal(get('generate').disabled,true,'changing splits or rows invalidates loaded data');
  await get('load-dataset').events.click();await flush();
  assert.deepEqual(datasetRequested.splits,['train','test']);assert.equal(datasetRequested.subset,'[20:30]');
  assert.equal(get('dataset-model-choice').hidden,false,'a dataset-specific classifier is offered without replacing the chosen model');
  get('dataset-use-model').events.click();
  assert.equal(get('model-preset').value,'aaraki/vit-base-patch16-224-in21k-finetuned-cifar10');
  assert.equal(get('dataset-model-choice').hidden,true);
  assert.equal(get('dataset-preview').hidden,false,'using a matching model keeps the loaded images');
  for(const row of get('dataset-split-options').children){row.children[0].checked=false;row.children[0].events.change();}
  assert.equal(get('load-dataset').disabled,true,'at least one split must be selected');
  const callsBeforeEmptySplits=requests.length;await get('load-dataset').events.click();
  assert.equal(requests.length,callsBeforeEmptySplits,'an empty split selection cannot start a load');
  assert.ok(get('dataset-preset').children.some(option=>option.value==='ILSVRC/imagenet-1k'),'ImageNet models need the real matching dataset preset');
  get('dataset-preset').value='ILSVRC/imagenet-1k';get('dataset-preset').events.change();await flush();
  assert.equal(get('dataset-access-note').hidden,false);
  assert.equal(get('dataset-subset').value,'','ImageNet defaults to all rows');
  assert.equal(get('dataset-split-summary').textContent,'train');
  for(const model of ['google/mobilenet_v2_1.0_224','microsoft/resnet-18']) {
    get('model-preset').value=model;get('model-preset').events.change();
    assert.equal(get('dataset-model-choice').hidden,true,'either ImageNet model is already compatible');
  }
  const validationSplit=get('dataset-split-options').children[1].children[0];
  get('dataset-split-options').children[0].children[0].checked=false;get('dataset-split-options').children[0].children[0].events.change();
  validationSplit.checked=true;validationSplit.events.change();
  get('dataset-subset').value='1000';get('dataset-subset').events.input();
  await get('load-dataset').events.click();await flush();
  assert.deepEqual(datasetRequested,{source:'huggingface',name:'ILSVRC/imagenet-1k',splits:['validation'],subset:'1000',config:'default',column:'',label_column:''});
  get('dataset-preset').value='other';get('dataset-preset').events.change();
  assert.equal(get('dataset-access-note').hidden,true);
  assert.equal(get('dataset-name-fields').hidden,false);
  assert.equal(get('dataset-column').value,'','Other does not inherit a preset-specific image column');
  get('dataset-name').value='org/slow';get('dataset-name').events.change();await flush();
  assert.equal(get('load-dataset').disabled,true,'wait for split metadata before loading');
  get('dataset-name').value='org/photos';get('dataset-name').events.change();await flush();
  resolveOldOptions();await flush();
  assert.equal(get('dataset-split-summary').textContent,'validation','late metadata cannot replace a new dataset; absent train defaults to first available split');
  assert.equal(get('dataset-label-column').children[1].value,'species','unfamiliar scalar label columns are available for explicit selection');
  assert.match(get('dataset-options-status').textContent,/Choose a label column/);
  get('dataset-config').value='default';get('dataset-column').value='picture';get('dataset-label-column').value='species';
  get('dataset-subset').value='20';
  await get('load-dataset').events.click();await flush();
  assert.deepEqual(datasetRequested,{source:'huggingface',name:'org/photos',splits:['validation'],subset:'20',config:'default',column:'picture',label_column:'species'});
  domains.find(button=>button.dataset.task==='summarization').events.click();
  get('summary-source').value='text';get('summary-source').events.change();
  assert.equal(get('dataset-preview').hidden,true,'text workspaces never show a retained image sample');
  assert.equal(get('generate').textContent,'Analyze','text workspaces use the same embedding workflow');
  assert.equal(get('viewer-title').textContent,'Search space');
  assert.equal(get('input-browser').hidden,true);
  get('image-select').replaceChildren();
  const imageRequestsBeforeText=requests.filter(request=>request.url.includes('/items?')).length;
  await get('next-images').events.click();await flush();
  assert.equal(requests.filter(request=>request.url.includes('/items?')).length,imageRequestsBeforeText,'hidden dataset controls cannot page images in a text workspace');
  await get('preview-form').events.submit({preventDefault(){}});await flush();
  assert.equal(submitted.task,'summarization');
  assert.equal(submitted.dataset_id,'','a retained image dataset is not part of text analysis');
  assert.equal(submitted.text,'HELLO World');
  assert.equal(modelRequested,undefined,'text embeddings also avoid the download-only model route');
  assert.equal(get('embedding-wrap').hidden,false);
  assert.equal(get('graph-wrap').hidden,true);
  assert.equal(get('details').hidden,false,'a single text input opens its action view automatically');
  assert.equal(get('original-view').children[0].textContent,'HELLO World');
  assert.equal(get('selected-view').children[0].textContent,'HELLO World');
  assert.match(get('embedding-meta').textContent,/summarization encoder/);
  assert.match(get('embedding-meta').textContent,/original.*2 action variants.*3 total/i);
  assert.match(get('embedding-meta').title,/attention-mask mean pooled/);
  assert.equal(requests.filter(request=>request.url.includes('/items?')).length,imageRequestsBeforeText,'automatic text focus never pages a retained image dataset');
  get('embedding-action').value='0';get('embedding-action').events.change();
  assert.equal(get('selected-view').children[0].textContent,'hello world');
  assert.match(get('selected-action').textContent,/to lowercase/);
  assert.ok(!get('selected-action').textContent.includes('undefined'));
  assert.match(get('embedding-similarity').textContent,/0\.875/);
  get('embedding-action').value='1';get('embedding-action').events.change();
  assert.match(get('selected-action').textContent,/did not change/,'unchanged edits are disclosed in the text inspector');
  assert.match(get('embedding-similarity').textContent,/1\.000/);
  get('embedding-action').value='original';get('embedding-action').events.change();
  assert.equal(get('embedding-similarity').hidden,true);
  assert.equal(get('selected-view').children[0].textContent,'HELLO World');
  get('model-preset').value='other';get('model-preset').events.change();
  assert.equal(get('embedding-wrap').hidden,true,'changing a text model clears its old coordinates');
  assert.equal(get('details').hidden,true);
  assert.equal(get('generate').disabled,true,'all domains require a model');
  get('model-name').value='org/summarizer';get('model-name').events.input();
  await get('preview-form').events.submit({preventDefault(){}});await flush();
  assert.deepEqual(submitted.model,{source:'huggingface',name:'org/summarizer'});
  get('source-text').value='EDITED TEXT';get('source-text').events.input();
  assert.equal(get('embedding-wrap').hidden,true,'editing the source clears old text embeddings');
  await get('preview-form').events.submit({preventDefault(){}});await flush();
  assert.equal(get('original-view').children[0].textContent,'EDITED TEXT');
  removeAction('remove_punctuation');
  assert.equal(get('action-count').textContent,'1 combination');
  assert.equal(get('embedding-wrap').hidden,true,'changing text actions clears old coordinates');
  removeAction('to_lowercase');assert.equal(get('generate').disabled,true,'empty text action selection blocks Analyze');
  addAction('to_lowercase');
  deferNextPreview=true;
  const stalePreview=get('preview-form').events.submit({preventDefault(){}});await flush();
  const embeddingRequestsBeforeSwitch=requests.filter(request=>request.url==='/api/embeddings').length;
  domains.find(button=>button.dataset.task==='generation').events.click();
  resolvePreview();await stalePreview;await flush();
  assert.equal(requests.filter(request=>request.url==='/api/embeddings').length,embeddingRequestsBeforeSwitch,'a preview acknowledged after a task change must not start stale embeddings');
  assert.equal(get('embedding-wrap').hidden,true);
  assert.equal(get('generate').disabled,true,'the next workspace needs a loaded dataset or entered prompt');
  get('generation-source').value='text';get('generation-source').events.change();
  assert.equal(get('generate').disabled,false,'canceling a stale analysis releases the next workspace');
  assert.ok(get('model-preset').children.some(option=>option.value==='segmind/tiny-sd'),'model presets follow the workspace domain');
  assert.equal(get('model-name-fields').hidden,true,'the new domain selects a preset, not a stale custom model');
  for(const key of ['lighting','detail'])removeAction(key);
  chooseValues('style','value-0');
  await get('preview-form').events.submit({preventDefault(){}});await flush();
  assert.equal(submitted.task,'generation');
  assert.deepEqual(submitted.action_space,{style:['photograph'],lighting:[],detail:[]});
  assert.equal(get('original-view').children[0].textContent,'A unique scientist in a corporate office.','the original is a concrete prompt, independent of selected action values');
  assert.match(get('embedding-meta').textContent,/CLIP prompt conditioning/);
  assert.match(get('embedding-meta').textContent,/original.*1 action variant.*2 total/i);
  assert.match(get('model-help').textContent,/Stable Diffusion 1\/2/);
  assert.match(get('source-help').textContent,/Find failure generates images/);
  assert.ok(get('embedding-action').children[1].textContent.includes('photograph'));
  assert.ok(!get('embedding-action').children[1].textContent.includes('rotation'));
  get('embedding-action').value='36';get('embedding-action').events.change();
  assert.equal(get('selected-view').children[0].textContent,'A unique scientist in a corporate office. Style: photograph.');
  assert.equal(get('selected-action').textContent,'Style: photograph');
  assert.match(get('embedding-similarity').textContent,/0\.875/);
  assert.equal(modelRequested,undefined);
  addAction('lighting');chooseValues('lighting','value-4');addAction('detail');chooseValues('detail','value-4');
  await get('preview-form').events.submit({preventDefault(){}});await flush();
  assert.deepEqual(submitted.action_space,{style:['photograph'],lighting:['low-key'],detail:['intricate']},'the request combines the selected group values');
  removeAction('lighting');assert.equal(get('action-count').textContent,'1 combination per prompt');
  await get('preview-form').events.submit({preventDefault(){}});await flush();
  assert.deepEqual(submitted.action_space,{style:['photograph'],lighting:[],detail:['intricate']},'disabled groups send empty lists');
  addAction('lighting');assert.equal(get('action-lighting').value,'value-4','restoring a group keeps its selected value');
  get('model-source').value='local';get('model-source').events.change();
  assert.equal(get('hf-model-fields').hidden,true);
  assert.equal(get('model-name-fields').hidden,false);
  assert.equal(get('model-name').required,true,'local model directories are required for generation embeddings');
  assert.equal(get('generate').disabled,true);
  assert.equal(get('embedding-wrap').hidden,true);
  get('model-name').value='/models/stable-diffusion';get('model-name').events.input();
  assert.equal(get('generate').disabled,false);
  await get('preview-form').events.submit({preventDefault(){}});await flush();
  assert.deepEqual(submitted.model,{source:'local',name:'/models/stable-diffusion'});
  domains.find(button=>button.dataset.task==='summarization').events.click();
  assert.equal(get('source-text').value,'EDITED TEXT','each text domain retains its own input');
  assert.equal(get('action-count').textContent,'1 combination','workspace changes retain the chosen text edits');
  assert.equal(get('embedding-wrap').hidden,true,'switching domains clears the other encoder’s map');
  assert.equal(get('model-name').required,true);
  domains.find(button=>button.dataset.task==='classification').events.click();
  assert.equal(get('embedding-wrap').hidden,true);
  assert.equal(get('input-browser').hidden,false,'switching back retains the image dataset');
  assert.equal(get('dataset-preview').hidden,false,'returning to classification restores its selected sample');
  get('model-name').value='/models/vision';get('model-name').events.input();
  get('analysis-device').value='cuda';get('analysis-device').events.change();
  await get('preview-form').events.submit({preventDefault(){}});await flush();
  assert.equal(embeddingRequested.device,'cuda','the form submits the selected Analyze device');
  get('analysis-device').value='cpu';get('analysis-device').events.change();
  assert.equal(get('embedding-wrap').hidden,true,'switching devices invalidates the old map');
  await get('preview-form').events.submit({preventDefault(){}});await flush();
  assert.equal(embeddingRequested.device,'cpu');
  // The real graph renderer must keep every combination addressable, including the last one.
  let selectedGraphNode;
  let graphSelections=0;
  context.graphTestSvg=new Element();context.onGraphSelect=node=>{selectedGraphNode=node;graphSelections++;};
  context.graphTestData=[];
  for(const rotation of [-10,-5,0,5,10])for(const darkness of [.1,.2,.3,.4,.5])for(const saturation of [.5,.55,.6,.65,.7]){
    const id=context.graphTestData.length;
    context.graphTestData.push({id,action:{rotation,darkness,saturation},image:`/combination-${id}.png`});
  }
  vm.runInContext('const fullGraph=new CombinationGraph(graphTestSvg,onGraphSelect);fullGraph.setData(graphTestData,"classification",actionDescription);',context);
  let allNodes=context.graphTestSvg.querySelectorAll('[data-node]');
  assert.equal(allNodes.length,126,'125 combinations plus the original input are rendered');
  let last=allNodes.find(node=>node.dataset.node==='124');
  last.events.click();
  assert.equal(selectedGraphNode.image,'/combination-124.png','clicking the last node returns its own sample');
  assert.equal(get('graph-space').textContent,'Rotation: 5 · Darkness: 5 · Saturation: 5','all action dimensions are visible above the graph');
  for(const [dimension,label] of [['darkness','Darkness 0.1'],['saturation','Saturation 0.5']]){
    get('graph-group-by').value=dimension;get('graph-group-by').events.change();
    allNodes=context.graphTestSvg.querySelectorAll('[data-node]');
    assert.equal(allNodes.length,126,'regrouping does not remove combinations');
    assert.equal(get('graph-legend').children.length,5);
    assert.equal(get('graph-legend').children[0].children[1].textContent,label);
    last=allNodes.find(node=>node.dataset.node==='124');
    assert.equal(last.attrs['aria-pressed'],'true','regrouping preserves the selected sample');
    last.events.click();
    assert.equal(selectedGraphNode.image,'/combination-124.png');
  }
  const beforeMove=last.attrs.transform,beforePan=context.graphTestSvg.attrs.viewBox;
  const lastEdge=context.graphTestSvg.children[0].children[124];
  const edgeEnd=[lastEdge.attrs.x2,lastEdge.attrs.y2],selectionCount=graphSelections;
  context.graphTestSvg.events.pointerdown({button:0,pointerId:2,clientX:200,clientY:200,target:last});
  context.graphTestSvg.events.pointermove({pointerId:2,clientX:260,clientY:230});
  context.graphTestSvg.events.pointerup({type:'pointerup',pointerId:2});
  assert.notEqual(last.attrs.transform,beforeMove,'dragging a node repositions it');
  assert.equal(context.graphTestSvg.attrs.viewBox,beforePan,'node dragging does not pan the background');
  assert.notDeepEqual([lastEdge.attrs.x2,lastEdge.attrs.y2],edgeEnd,'the connecting edge follows the moved node');
  last.events.click({detail:1});
  assert.equal(graphSelections,selectionCount,'the click emitted after a drag must not select a sample');
  context.graphTestSvg.events.pointerdown({button:0,pointerId:3,clientX:260,clientY:230,target:last});
  context.graphTestSvg.events.pointerup({type:'pointerup',pointerId:3});
  last.events.click({detail:1});
  assert.equal(graphSelections,selectionCount+1,'a pointer click selects exactly once after a drag');
  assert.equal(selectedGraphNode.image,'/combination-124.png');
  const originalNode=allNodes.find(node=>node.dataset.node==='original');
  context.graphTestSvg.events.pointerdown({button:0,pointerId:4,clientX:400,clientY:300,target:originalNode});
  context.graphTestSvg.events.pointermove({pointerId:4,clientX:430,clientY:320});
  context.graphTestSvg.events.pointerup({type:'pointerup',pointerId:4});
  for(const edge of context.graphTestSvg.children[0].children)assert.notEqual(edge.attrs.x1,'0','moving the original updates all edge origins');
  get('graph-search').value='#125';get('graph-search').events.input();
  assert.equal(allNodes.filter(node=>node.dataset.node!=='original'&&node.attrs.display==='inline').length,1);
  assert.equal(last.attrs.display,'inline','all combinations remain searchable');
  allNodes.find(node=>node.dataset.node==='original').events.click();
  assert.equal(selectedGraphNode,null,'the original input remains selectable');
  context.graphTestData=[
    {id:43,action:{id:43,style:'photograph',lighting:'even',detail:'minimal'},text:'Full combination'},
    {id:36,action:{id:36,style:'photograph',lighting:null,detail:null},text:'Style only'}
  ];
  vm.runInContext('fullGraph.setData(graphTestData,"generation",actionDescription);',context);
  allNodes=context.graphTestSvg.querySelectorAll('[data-node]');
  const combination=allNodes.find(node=>node.dataset.node==='43'),styleOnly=allNodes.find(node=>node.dataset.node==='36');
  assert.equal(combination.attrs['aria-label'],'Action 43: Style: photograph · Lighting: even · Detail: minimal');
  assert.equal(styleOnly.attrs['aria-label'],'Action 36: Style: photograph','disabled groups do not appear as null instructions');
  assert.equal(get('graph-legend').children[0].children[1].textContent,'Style: photograph · Lighting: even · Detail: minimal');
  get('graph-search').value='#43';get('graph-search').events.input();
  assert.equal(combination.attrs.display,'inline');assert.equal(styleOnly.attrs.display,'none','generation search uses stable action IDs');
  combination.events.click();assert.equal(selectedGraphNode.action.id,43);
  console.log('PASS: action chips add/remove choices, preserve values, prevent duplicates, and submit the selected transformations');
  console.log('PASS: selected dataset and selected image reach the generated graph');
  console.log('PASS: selected dataset samples preview before Analyze and follow selection, invalidation and workspace changes');
  console.log('PASS: classification loads actual model embedding jobs and selection expands the correct image');
  console.log('PASS: action selection shows original/changed input, all action values and cosine similarity');
  console.log('PASS: all 125 combinations and their sample identities remain available in the graph');
  console.log('PASS: darkness/saturation regrouping preserves all nodes and the selected sample');
  console.log('PASS: dragging nodes moves their edges without accidental selection; pointer clicks still select');
  console.log('PASS: Hugging Face dataset selection ignores delayed pages from the previous folder');
  console.log('PASS: model presets and Other submit the selected ID and follow the workspace domain');
  console.log('PASS: real datasets default to full train; split multiselect, row ranges and automatic columns reach the loader');
  console.log('PASS: dataset metadata ignores stale responses and matching-model selection preserves the loaded dataset');
  console.log('PASS: summarization and generation Analyze display text variants, original baselines, cosine and encoder/PCA metadata');
  console.log('PASS: model, input, action and task edits clear maps; stale previews and retained image datasets cannot affect text workspaces');
})().catch(error=>{console.error(error);process.exitCode=1;});
