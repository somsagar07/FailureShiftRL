'use strict';
// Real app/controller behavior at the DOM and HTTP boundaries; no model downloads.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const elements=new Map();
class Element {
  constructor(){this.value='';this.children=[];this.events={};this.dataset={};this.attrs={};this.style={};this.classList={toggle(){}};}
  set id(value){this._id=value;elements.set(value,this);} get id(){return this._id;}
  replaceChildren(...nodes){this.children=nodes;} append(...nodes){this.children.push(...nodes);}
  addEventListener(name,fn){this.events[name]=fn;} setAttribute(key,value){this.attrs[key]=value;}
  getBoundingClientRect(){return {left:0,top:0,width:800,height:600};}
  getContext(){return new Proxy({measureText:()=>({width:40})},{get:(target,key)=>target[key]||(()=>{})});}
  focus(){} scrollIntoView(){}
}
const get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
const content=node=>[node.textContent||'',...node.children.map(content)].join(' ');
const domains=['classification','summarization','generation'].map(task=>Object.assign(new Element(),{dataset:{task}}));
for(const [id,value] of Object.entries({'model-source':'huggingface','dataset-source':'folder','finetune-epochs':'1','finetune-lr':'0.001','discovery-episodes':'2','discovery-seed':'42','discovery-threshold':'.15','discovery-device':'cpu','discovery-steps':'12','discovery-size':'256'}))get(id).value=value;
const generationActions={style:['photograph','watercolor','pencil drawing','digital illustration','3D render'],lighting:['even','soft','directional','high-key','low-key'],detail:['minimal','simple','moderate','detailed','intricate']};
let preview,postedDataset,postedPreview,slowPrompt,slowOptions;
const calls=[],datasetId='generation-dataset';
const prompt=index=>({index,name:`Prompt ${index+1}`,text:`Two red birds beside a blue vase, scene ${index+1}. ${'Exact source wording. '.repeat(20)}`,checks:[{question:'How many birds are present?',answer:'2',skill:'counting'},{question:'What color is the vase?',answer:'blue'}],source_split:'test',source_index:index});
const metadata=name=>({name,configs:['default'],config:'default',splits:name==='geneval2'?[{name:'test',count:800}]:[{name:'train',count:100},{name:'test',count:1}],text_columns:['Prompt','requirements'],suggested_prompt_column:'Prompt',checks_columns:['requirements'],suggested_checks_column:name==='geneval2'?'requirements':'',image_columns:[],label_columns:[]});
const loaded=()=>({id:datasetId,task:'generation',source:postedDataset.source,status:'ready',name:postedDataset.name,count:101,checked_count:101,missing_checks_count:0,prompt_column:postedDataset.prompt_column,checks_column:postedDataset.checks_column,splits:postedDataset.splits,subset:postedDataset.subset});
const context={console,TextEncoder,document:{getElementById:get,createElement:()=>new Element(),createElementNS:()=>new Element(),querySelectorAll:selector=>selector==='.domain'?domains:[]},setTimeout:()=>1,clearTimeout(){},window:{addEventListener(){},devicePixelRatio:1,localStorage:{getItem:()=>null,setItem(){}}},fetch:async(url,options={})=>{
  const body=options.body?JSON.parse(options.body):undefined;calls.push({url,body});let result;
  if(url==='/api/catalog')result={tasks:{classification:{rotation:[0],darkness:[0],saturation:[1]},summarization:{actions:['to_lowercase']},generation:generationActions},text:'A pasted article.',prompt:'A red bird beside a blue vase.'};
  else if(url.startsWith('/api/datasets/options?')) {
    const name=new URL(url,'http://local').searchParams.get('name');result=metadata(name);
    if(name==='org/slow')return new Promise(resolve=>{slowOptions=()=>resolve({ok:true,json:async()=>metadata(name)});});
  } else if(url==='/api/datasets'){postedDataset=body;result=loaded();}
  else if(url.startsWith(`/api/datasets/${datasetId}/items?`)) {
    const offset=Number(new URL(url,'http://local').searchParams.get('offset'));
    result={offset,count:101,items:Array.from({length:Math.min(100,101-offset)},(_,i)=>({index:offset+i,name:`Prompt ${offset+i+1}`,text_preview:'Short preview',check_count:2}))};
  } else if(url.startsWith(`/api/datasets/${datasetId}/text/`)) {
    const index=Number(url.split('/').at(-1));result=prompt(index);
    if(index===2)return new Promise(resolve=>{slowPrompt=()=>resolve({ok:true,json:async()=>result});});
  } else if(url==='/api/previews') {
    postedPreview=body;result=preview={id:'preview-generation',status:'completed',config:body,dataset:body.dataset_id?loaded():undefined,input_count:body.dataset_id?101:1,action_count:125,original:body.dataset_id?prompt(0):{text:body.text},nodes:[]};
  } else if(url.startsWith('/api/previews/'))result=preview;
  else if(url==='/api/embeddings')result={id:'embedding-generation',task:'generation',status:'completed',feature_dim:64,model:postedPreview.model,projection:{variance:.9,method:'PCA fitted on original prompts'},requirements:postedPreview.dataset_id?{dataset_id:datasetId,total:101,count:101,missing:0,prompt_column:'Prompt',checks_column:'requirements'}:undefined,points:Array.from({length:postedPreview.dataset_id?101:1},(_,index)=>({index,name:`Prompt ${index+1}`,text:postedPreview.dataset_id?'Short preview':postedPreview.text,position:[index,0,0]}))};
  else if(url==='/api/embeddings/embedding-generation/expand')result={id:`expand-${body.input_index}`,status:'completed',input_index:body.input_index,original:postedPreview.dataset_id?prompt(body.input_index):{text:postedPreview.text},nodes:[{id:215,text:`${postedPreview.dataset_id?prompt(body.input_index).text:postedPreview.text} Style: 3D render. Lighting: low-key. Detail: intricate.`,action:{id:215,style:'3D render',lighting:'low-key',detail:'intricate'},similarity:.9,position:[1,1,1]}]};
  else throw Error(`Unexpected request ${url}`);
  return {ok:true,json:async()=>result};
}};
vm.createContext(context);
for(const file of ['graph.js','embedding-view.js','embedding-controller.js','adaptation-controller.js','discovery-controller.js','app.js'])vm.runInContext(fs.readFileSync(path.join(__dirname,'../static',file),'utf8'),context);
async function flush(){for(let i=0;i<40;i++)await Promise.resolve();}
const switchTask=async task=>{domains.find(button=>button.dataset.task===task).events.click();await flush();};
const addAction=key=>{get('action-add').value=key;get('action-add').events.change();};
const chooseValue=(key,value)=>{if(get(`edit-action-${key}`).attrs['aria-expanded']!=='true')get(`edit-action-${key}`).events.click();get(`action-${key}`).value=value;get(`action-${key}`).events.change();};
(async()=>{
  await flush();
  assert.equal(get('generation-source').value,'dataset','generation opens the dataset flow');
  await switchTask('generation');
  assert.equal(get('text-fields').hidden,true);
  assert.deepEqual(get('generation-dataset-preset').children.map(option=>option.value),['geneval2','nateraw/parti-prompts','other']);
  assert.equal(get('generation-dataset-split-summary').textContent,'test');assert.equal(get('generation-dataset-subset').value,'');
  assert.equal(get('generation-dataset-prompt-column').value,'Prompt');assert.equal(get('generation-dataset-checks-column').value,'requirements');
  assert.equal(get('action-chips').children.length,3,'generation uses three editable action groups');assert.equal(get('action-count').textContent,'125 combinations per prompt');
  for(const [key,name] of [['style','Style'],['lighting','Lighting'],['detail','Detail']])assert.equal(get(`action-caption-${key}`).textContent,`${name} (5)`);
  get('remove-action-lighting').events.click();assert.equal(get('action-chips').children.length,2);assert.equal(get('action-count').textContent,'25 combinations per prompt');
  chooseValue('detail','value-3');assert.equal(get('action-count').textContent,'5 combinations per prompt');
  get('remove-action-detail').events.click();assert.equal(get('action-count').textContent,'5 combinations per prompt','style alone retains all five values');
  get('remove-action-style').events.click();assert.equal(get('action-count').textContent,'0 combinations per prompt');assert.equal(get('generate').disabled,true);
  addAction('style');addAction('detail');assert.equal(get('action-detail').value,'value-3','re-adding a group retains its selected value');
  addAction('lighting');assert.equal(get('action-count').textContent,'25 combinations per prompt');
  chooseValue('style','value-1');assert.equal(get('action-count').textContent,'5 combinations per prompt');
  chooseValue('lighting','value-1');assert.equal(get('action-count').textContent,'1 combination per prompt');
  get('reset-actions').events.click();assert.equal(get('action-chips').children.length,3);assert.equal(get('action-count').textContent,'125 combinations per prompt');
  get('generation-dataset-subset').value='101';get('generation-dataset-subset').events.input();
  await get('load-generation-dataset').events.click();await flush();
  assert.deepEqual(postedDataset,{source:'benchmark',task:'generation',name:'geneval2',config:'default',splits:['test'],subset:'101',prompt_column:'Prompt',checks_column:'requirements'});
  assert.equal(get('dataset-preview-image').children[0].textContent,prompt(0).text);
  assert.match(content(get('dataset-preview-requirements')),/How many birds.*2.*What color.*blue/);
  assert.match(get('generation-dataset-checks-status').textContent,/101\s*\/\s*101/);
  assert.equal(calls.some(call=>call.url==='/api/previews'),false);
  get('image-select').value='2';const stale=get('image-select').events.change();await flush();await switchTask('summarization');slowPrompt();await stale;
  assert.equal(get('dataset-preview').hidden,true,'late generation samples cannot replace another tab');
  await switchTask('generation');get('image-select').value='1';await get('image-select').events.change();await flush();
  await get('next-images').events.click();await flush();assert.equal(get('image-select').value,'100');
  assert.equal(get('dataset-preview-image').children[0].textContent,prompt(100).text);
  await get('preview-form').events.submit({preventDefault(){}});await flush();
  assert.equal(postedPreview.dataset_id,datasetId);assert.equal(postedPreview.text,'');assert.deepEqual(postedPreview.action_space,generationActions);
  assert.equal(get('generation-dataset-count').textContent,'101 prompts × 125 combinations per prompt = 12625 combinations.');
  assert.equal(get('details').hidden,true,'dataset analysis begins with all original prompts');
  assert.match(get('embedding-meta').textContent,/101 prompts.*original prompts/);
  get('image-select').value='100';await get('image-select').events.change();await flush();
  assert.equal(get('original-view').children[0].textContent,prompt(100).text);
  assert.match(content(get('original-requirements')),/How many birds.*2/);
  get('embedding-action').value='215';get('embedding-action').events.change();assert.equal(get('selected-action').textContent,'Style: 3D render · Lighting: low-key · Detail: intricate');
  assert.equal(get('selected-title').textContent,'Action 215');
  assert.match(get('embedding-action').children[1].textContent,/Action 215.*Style: 3D render.*Lighting: low-key.*Detail: intricate/);
  get('find-failure').events.click();assert.equal(get('discovery-generation-metric').value,'combined');assert.equal(Number(get('discovery-threshold').value),.5);
  get('generation-source').value='text';get('generation-source').events.change();await flush();
  assert.equal(get('text-fields').hidden,false);assert.equal(get('input-browser').hidden,true);assert.equal(get('source-label').textContent,'Prompt');
  get('source-text').value='A ceramic teapot on a rock.';get('source-text').events.input();
  await get('preview-form').events.submit({preventDefault(){}});await flush();
  assert.equal(postedPreview.text,'A ceramic teapot on a rock.');assert.equal(postedPreview.dataset_id,'');
  get('find-failure').events.click();assert.equal(get('discovery-generation-metric').value,'combined');assert.match(get('discovery-requirements-status').textContent,/visual content review/i);assert.equal(get('discovery-start').disabled,false);
  get('generation-source').value='dataset';get('generation-source').events.change();await flush();
  get('generation-dataset-preset').value='nateraw/parti-prompts';await get('generation-dataset-preset').events.change();await flush();
  assert.equal(get('generation-dataset-checks-column').value,'');assert.equal(get('generation-dataset-split-summary').textContent,'train');
  const split=get('generation-dataset-split-options').children[1].children[0];split.checked=true;split.events.change();
  await get('load-generation-dataset').events.click();await flush();
  assert.equal(postedDataset.source,'huggingface');assert.equal(postedDataset.name,'nateraw/parti-prompts');assert.deepEqual(postedDataset.splits,['train','test']);
  get('generation-dataset-preset').value='other';await get('generation-dataset-preset').events.change();
  get('generation-dataset-name').value='org/slow';const pending=get('generation-dataset-name').events.change();await flush();
  get('generation-dataset-preset').value='geneval2';await get('generation-dataset-preset').events.change();await flush();slowOptions();await pending;
  assert.equal(get('generation-dataset-name').value,'geneval2');assert.equal(get('generation-dataset-split-summary').textContent,'test');
  console.log('PASS: generation groups and Cartesian counts, retained values, presets, full prompts, requirements, pagination, stale responses, overview, free prompts, and evaluation defaults');
})().catch(error=>{console.error(error);process.exitCode=1;});
