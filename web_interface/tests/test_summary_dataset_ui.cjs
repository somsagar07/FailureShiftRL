'use strict';
// Exercise the real app and controllers across the DOM and HTTP boundaries.
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
const domains=['classification','summarization','generation'].map(task=>Object.assign(new Element(),{dataset:{task}}));
for(const [id,value] of Object.entries({'model-source':'huggingface','dataset-source':'folder','finetune-epochs':'1','finetune-lr':'0.001','discovery-episodes':'2','discovery-seed':'42','discovery-threshold':'.15','discovery-device':'cpu','discovery-metric':'reference'}))get(id).value=value;
let preview,postedDataset,postedPreview,slowArticle,slowOptions;
const calls=[],summaryId='summary-dataset';
const article=index=>({index,name:`Article ${index+1}`,text:`Full article ${index+1}. ${'Detailed source text. '.repeat(30)}`,summary:`Reference for article ${index+1}.`});
const metadata=name=>({name,configs:[name==='abisee/cnn_dailymail'?'3.0.0':'default'],config:name==='abisee/cnn_dailymail'?'3.0.0':'default',splits:[{name:'train',count:101},{name:'test',count:12}],image_columns:[],label_columns:[],text_columns:['dialogue','summary'],suggested_text_column:'dialogue',suggested_summary_column:'summary'});
const loaded=()=>({id:summaryId,task:'summarization',source:'huggingface',status:'ready',name:postedDataset.name,count:101,reference_count:100,missing_reference_count:1,text_column:postedDataset.text_column,summary_column:postedDataset.summary_column,splits:postedDataset.splits,subset:postedDataset.subset});
const context={console,TextEncoder,document:{getElementById:get,createElement:()=>new Element(),createElementNS:()=>new Element(),querySelectorAll:selector=>selector==='.domain'?domains:[]},setTimeout:()=>1,clearTimeout(){},window:{addEventListener(){},devicePixelRatio:1,localStorage:{getItem:()=>null,setItem(){}}},fetch:async(url,options={})=>{
  const body=options.body?JSON.parse(options.body):undefined;calls.push({url,body});let result;
  if(url==='/api/catalog')result={tasks:{classification:{rotation:[0],darkness:[0],saturation:[1]},summarization:{actions:['to_lowercase','negate_sentence']},generation:{style:['photograph'],lighting:['even'],detail:['moderate']}},text:'Original pasted text.',prompt:'A scientist in an office.'};
  else if(url.startsWith('/api/datasets/options?')) {
    const name=new URL(url,'http://local').searchParams.get('name');result=metadata(name);
    if(name==='org/slow')return new Promise(resolve=>{slowOptions=()=>resolve({ok:true,json:async()=>metadata(name)});});
  } else if(url==='/api/datasets'){postedDataset=body;result=loaded();}
  else if(url.startsWith(`/api/datasets/${summaryId}/items?`)) {
    const offset=Number(new URL(url,'http://local').searchParams.get('offset'));
    result={offset,count:101,items:Array.from({length:Math.min(100,101-offset)},(_,i)=>({index:offset+i,name:`Article ${offset+i+1}`,text_preview:'Short snippet',has_reference:offset+i!==100}))};
  } else if(url.startsWith(`/api/datasets/${summaryId}/text/`)) {
    const index=Number(url.split('/').at(-1));result=article(index);
    if(index===2)return new Promise(resolve=>{slowArticle=()=>resolve({ok:true,json:async()=>result});});
  } else if(url==='/api/previews') {
    postedPreview=body;result=preview={id:'preview-summary',status:'completed',config:body,dataset:loaded(),input_count:101,action_count:2,original:{...article(0),reference_summary:article(0).summary},nodes:[]};
  } else if(url.startsWith('/api/previews/'))result=preview;
  else if(url==='/api/embeddings')result={id:'embedding-summary',task:'summarization',status:'completed',feature_dim:64,model:postedPreview.model,projection:{variance:.9,method:'PCA fitted on original inputs'},references:{dataset_id:summaryId,total:101,count:100,missing:1,text_column:'dialogue',summary_column:'summary'},points:Array.from({length:101},(_,index)=>({index,name:`Article ${index+1}`,text:'Short snippet',position:[index,0,0]}))};
  else if(url==='/api/embeddings/embedding-summary/expand')result={id:`expand-${body.input_index}`,status:'completed',input_index:body.input_index,original:{...article(body.input_index),reference_summary:article(body.input_index).summary},nodes:[{id:0,text:article(body.input_index).text.toLowerCase(),action:{name:'to_lowercase'},similarity:.9,position:[1,1,1]}]};
  else throw Error(`Unexpected request ${url}`);
  return {ok:true,json:async()=>result};
}};
vm.createContext(context);
for(const file of ['graph.js','embedding-view.js','embedding-controller.js','adaptation-controller.js','discovery-controller.js','app.js'])vm.runInContext(fs.readFileSync(path.join(__dirname,'../static',file),'utf8'),context);
async function flush(){for(let i=0;i<35;i++)await Promise.resolve();}
const switchTask=async task=>{domains.find(button=>button.dataset.task===task).events.click();await flush();};
(async()=>{
  await flush();get('folder-path').value='/retained/classifier';
  await switchTask('summarization');
  assert.equal(get('summary-source').value,'dataset','summarization defaults to the dataset flow');
  assert.equal(get('image-fields').hidden,true);assert.equal(get('text-fields').hidden,true);
  assert.deepEqual(get('summary-dataset-preset').children.map(option=>option.value),['knkarthick/samsum','abisee/cnn_dailymail','other']);
  assert.equal(get('summary-dataset-split-summary').textContent,'train');
  assert.equal(get('summary-dataset-text-column').value,'dialogue');assert.equal(get('summary-dataset-summary-column').value,'summary');
  assert.equal(get('summary-dataset-subset').value,'');assert.equal(get('generate').disabled,true);
  const testSplit=get('summary-dataset-split-options').children[1].children[0];testSplit.checked=true;testSplit.events.change();
  get('summary-dataset-subset').value='[20:30]';get('summary-dataset-subset').events.input();
  await get('load-summary-dataset').events.click();await flush();
  assert.deepEqual(postedDataset,{source:'huggingface',task:'summarization',name:'knkarthick/samsum',config:'default',splits:['train','test'],subset:'[20:30]',text_column:'dialogue',summary_column:'summary'});
  assert.equal(get('dataset-preview').hidden,false);assert.equal(get('dataset-preview-image').children[0].textContent,article(0).text);
  assert.equal(get('dataset-preview-reference-text').textContent,article(0).summary);
  assert.equal(get('dataset-preview-reference').hidden,false);assert.match(get('summary-dataset-reference-status').textContent,/100\s*\/\s*101/);
  assert.equal(calls.some(call=>call.url==='/api/previews'),false,'sample browsing does not start analysis');
  get('image-select').value='1';await get('image-select').events.change();await flush();
  assert.equal(get('dataset-preview-image').children[0].textContent,article(1).text);
  get('image-select').value='2';const oldSample=get('image-select').events.change();await flush();
  // A different workspace can be opened while an article read is pending.
  await switchTask('generation');slowArticle();await oldSample;await flush();
  assert.equal(get('dataset-preview').hidden,true,'a delayed article cannot appear in generation');
  assert.equal(get('summary-fields').hidden,true);assert.equal(get('text-fields').hidden,true);
  await switchTask('classification');assert.equal(get('folder-path').value,'/retained/classifier');
  await switchTask('summarization');
  get('image-select').value='1';await get('image-select').events.change();await flush();
  await get('next-images').events.click();await flush();
  assert.equal(get('image-select').value,'100');assert.equal(get('dataset-preview-image').children[0].textContent,article(100).text);
  assert.equal(get('image-page').textContent,'101–101 of 101');assert.equal(get('next-images').disabled,true);
  await get('preview-form').events.submit({preventDefault(){}});await flush();
  assert.equal(postedPreview.task,'summarization');assert.equal(postedPreview.dataset_id,summaryId);assert.equal(postedPreview.text,'');
  assert.equal(get('dataset-preview').hidden,true);assert.equal(get('details').hidden,true,'dataset Analyze begins with all original points');
  assert.match(get('embedding-meta').textContent,/101 articles/);assert.match(get('embedding-meta').textContent,/original articles/);
  get('image-select').value='100';await get('image-select').events.change();await flush();
  assert.equal(get('original-view').children[0].textContent,article(100).text,'inspection uses the full expansion original');
  assert.equal(get('original-reference-text').textContent,article(100).summary);
  get('embedding-action').value='0';get('embedding-action').events.change();
  assert.equal(get('selected-view').children[0].textContent,article(100).text.toLowerCase());
  get('find-failure').events.click();assert.equal(get('discovery-reference-field').hidden,true);
  assert.match(get('discovery-reference-status').textContent,/100\s*\/\s*101/);assert.equal(get('discovery-start').disabled,true);
  assert.match(get('discovery-reference-status').textContent,/column|dataset/i);
  get('summary-source').value='text';get('summary-source').events.change();await flush();
  assert.equal(get('text-fields').hidden,false);assert.equal(get('input-browser').hidden,true);assert.equal(get('source-text').value,'Original pasted text.');
  get('summary-source').value='dataset';get('summary-source').events.change();await flush();
  assert.equal(get('source-text').required,false);assert.equal(get('input-browser').hidden,false);
  get('summary-dataset-preset').value='abisee/cnn_dailymail';await get('summary-dataset-preset').events.change();await flush();
  assert.equal(get('summary-dataset-config').value,'3.0.0');assert.equal(get('summary-dataset-subset').value,'');
  get('summary-dataset-preset').value='other';await get('summary-dataset-preset').events.change();
  assert.equal(get('summary-dataset-name-fields').hidden,false);
  get('summary-dataset-name').value='org/slow';const staleOptions=get('summary-dataset-name').events.change();await flush();
  get('summary-dataset-preset').value='knkarthick/samsum';await get('summary-dataset-preset').events.change();await flush();slowOptions();await staleOptions;
  assert.equal(get('summary-dataset-name').value,'knkarthick/samsum');assert.equal(get('summary-dataset-split-summary').textContent,'train');
  console.log('PASS: summary dataset presets, split and column payloads, full sample browsing, pagination, stale responses, N-point inspection, references, and workspace isolation');
})().catch(error=>{console.error(error);process.exitCode=1;});
