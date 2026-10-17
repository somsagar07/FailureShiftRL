// Controller tests use the real asynchronous workflow; model API and canvas are boundaries.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
class Element {
  constructor(){this.value='';this.children=[];this.events={};this.hidden=false;this.textContent='';}
  replaceChildren(...children){this.children=children;}
  append(...children){this.children.push(...children);}
  addEventListener(name,handler){this.events[name]=handler;}
}
const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
class View {
  constructor(canvas,events){this.events=events;this.reset();}
  setData(points){this.points=points;}
  focus(index){this.index=index;this.nodes=[];}
  setExpansion(index,nodes){if(index===this.index)this.nodes=nodes;}
  selectAction(id){this.selected=id;}
  clearFocus(){this.index=null;this.nodes=[];}
  reset(){this.points=[];this.clearFocus();}
  fit(){} zoom(){}
}
let inspected,status,busy,error,selectedIndex,resolveExpansion,resolveEmbedding;
const points=[0,1].map(index=>({index,name:`image-${index}.png`,image:`/input-${index}/original.png`,position:[index,0,0]}));
const action=index=>({id:0,action:{rotation:5,darkness:.2,saturation:.5},image:`/input-${index}/action-0.png`,position:[index,.1,0],similarity:.95});
const job={id:'embedding',status:'completed',points,feature_dim:4,feature_method:'pooled features',model:{name:'test/model'},projection:{method:'PCA',variance:.85,rank:3},device:'cuda',device_name:'NVIDIA GeForce RTX 4090',timings_seconds:{preparation:.12,model_loading:.25,embedding:.08,pca:.01,total:.47}};
const expansion=index=>({id:`expand-${index}`,status:'completed',input_index:index,original:points[index],nodes:[action(index)]});
const calls=[];
let deferNextEmbedding=false;
const api=async(url,body)=>{
  calls.push({url,body});
  if(url==='/api/embeddings')return deferNextEmbedding?new Promise(resolve=>{resolveEmbedding=resolve;}):job;
  if(url==='/api/embeddings/embedding/expand'){
    if(body.input_index===0)return new Promise(resolve=>{resolveExpansion=resolve;});
    return expansion(1);
  }
  throw Error('Unexpected API request: '+url);
};
const context={document:{getElementById:get,createElement:()=>new Element()},EmbeddingView:View,console,setTimeout,clearTimeout};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/embedding-controller.js'),'utf8'),context);
context.options={api,onInspect:(...args)=>{inspected=args;},onStatus:value=>{status=value;},onError:value=>{error=value;},onBusy:value=>{busy=value;},onSelectInput:index=>{selectedIndex=index;}};
vm.runInContext('const explorer=new EmbeddingExplorer(options);',context);
const explorer=vm.runInContext('explorer',context);
const preview={id:'preview',config:{task:'classification',model:{name:'test/model'},action_space:{rotation:[5],darkness:[.2],saturation:[.5]}}};
const content=element=>[element.textContent,...element.children.map(content)].join(' ');
(async()=>{
  await explorer.load(preview,'cuda');
  assert.equal(calls[0].body.preview_id,'preview','the map uses the current preview, model and dataset');
  assert.equal(calls[0].body.device,'cuda','Analyze passes the chosen device to the backend');
  assert.equal(get('analysis-performance').hidden,false);
  assert.match(get('analysis-performance-summary').textContent,/CUDA.*RTX 4090/);
  assert.match(content(get('analysis-timings')),/Weight loading.*0\.25/);
  assert.match(content(get('analysis-timings')),/Embeddings.*0\.08/);
  assert.match(content(get('analysis-timings')),/PCA.*0\.01/);
  assert.equal(explorer.view.points.length,2,'all original images appear before expanding any');
  assert.equal(explorer.view.index,null);
  assert.equal(get('embedding-wrap').hidden,false);
  assert.equal(busy,false);
  const old=explorer.focus(0);
  assert.equal(explorer.view.index,0,'the selected image is focused immediately while actions compute');
  assert.equal(inspected[0].image,points[0].image,'the original preview is available immediately');
  for(let i=0;i<10;i++)await Promise.resolve();
  const latest=explorer.focus(1);
  for(let i=0;i<10;i++)await Promise.resolve();
  assert.equal(calls.filter(call=>call.url.endsWith('/expand')).length,1,'wait for the earlier POST acknowledgment so late delivery cannot cancel the latest selection');
  assert.equal(selectedIndex,1);
  resolveExpansion(expansion(0));await old;await latest;
  assert.equal(explorer.view.index,1,'a late expansion must not steal focus');
  assert.equal(explorer.view.nodes[0].image,action(1).image);
  get('embedding-action').value='0';get('embedding-action').events.change();
  assert.equal(inspected[1].image,action(1).image,'the dropdown reaches actions even when points overlap');
  assert.equal(inspected[0].image,points[1].image);
  assert.equal(explorer.view.selected,0);
  explorer.clearFocus();
  assert.equal(explorer.view.index,null);
  assert.equal(explorer.view.nodes.length,0);
  assert.equal(get('embedding-actions').hidden,true);
  assert.equal(inspected[0],null,'clearing focus also clears the old inspector');
  const beforeLive=calls.length;explorer.isDiscoveryActive=()=>true;
  await explorer.focus(0);
  assert.equal(calls.length,beforeLive,'a live discovery selection cannot queue a full action expansion');
  assert.match(get('embedding-focus').textContent,/live discovery/);
  await explorer.focus(1);
  assert.equal(calls.length,beforeLive,'already computed action embeddings stay inspectable without new model calls');
  assert.equal(explorer.view.nodes[0].image,action(1).image);
  explorer.focusMeasured(0,4);assert.equal(calls.length,beforeLive,'measured pair selection never runs inference');
  assert.equal(explorer.view.index,0);assert.equal(inspected[0].image,points[0].image);
  explorer.isDiscoveryActive=()=>false;
  deferNextEmbedding=true;const pending=explorer.load(preview);
  explorer.reset();resolveEmbedding(job);await pending;
  assert.equal(get('embedding-wrap').hidden,true,'a stale load cannot reopen a cleared map');
  assert.equal(get('analysis-performance').hidden,true,'reset hides timings from the old analysis');
  assert.equal(content(get('analysis-timings')).trim(),'','reset clears stale timing values');
  assert.equal(explorer.view.points.length,0);
  deferNextEmbedding=false;
  context.options.api=async()=>({status:'error',error:'This model cannot produce image embeddings.'});
  const originalApi=explorer.api;explorer.api=context.options.api;
  await explorer.load(preview);
  assert.ok(error.includes('cannot produce image embeddings'));
  assert.equal(explorer.view.points.length,0,'model failure never produces fake coordinates');
  assert.equal(busy,false);
  explorer.api=originalApi;
  const cached={...job,device:'cpu',device_name:'CPU',model_reused:true,features_reused:true,timings_seconds:{preparation:0,model_loading:0,embedding:0,pca:0,total:.005}};
  explorer.api=async()=>cached;
  await explorer.load(preview);
  assert.equal(get('analysis-performance').hidden,false,'zero-duration cached stages still have timing information');
  assert.match(get('analysis-performance-summary').textContent,/CPU/);
  assert.match(content(get('analysis-timings')),/reused/i);
  assert.ok(!get('analysis-performance-summary').textContent.includes('4090'),'device changes cannot retain the old GPU label');
  explorer.api=originalApi;
  for(const [task,text,actionText,action,method] of [
    ['summarization','HELLO World!','hello world!',{name:'to_lowercase'},'summarization encoder'],
    ['generation','A unique scientist in a corporate office.','A cool artist in a classroom.',{description:'cool',profession:'artist',setting:'classroom'},'CLIP prompt conditioning']
  ]) {
    const source={index:0,name:task==='generation'?'Baseline prompt':'Source text',text,position:[0,0,0]};
    const variant={id:0,action,text:actionText,unchanged:false,position:[1,0,0],similarity:.875,neighbors:[]};
    const textPreview={id:`preview-${task}`,config:{task,model:{name:`test/${task}`}}};
    const textJob={id:`embedding-${task}`,task,status:'completed',points:[source],feature_dim:768,feature_method:'final encoder token states, attention-mask mean pooled',model:textPreview.config.model,projection:{method:'PCA fitted on original and action variants',variance:1,rank:1,components:1,fit_count:2}};
    const textCalls=[];
    explorer.api=async(url,body)=>{
      textCalls.push({url,body});
      if(url==='/api/embeddings')return textJob;
      if(url===`/api/embeddings/embedding-${task}/expand`)return {id:`expand-${task}`,status:'completed',input_index:0,original:source,nodes:[variant]};
      throw Error(`Unexpected ${task} request: ${url}`);
    };
    selectedIndex='unchanged';
    await explorer.load(textPreview);
    assert.equal(textCalls[0].body.device,'auto','Analyze defaults to automatic device selection');
    assert.equal(get('analysis-performance').hidden,true,'older reports without timings do not reuse a previous timing panel');
    assert.equal(explorer.view.index,0,`${task} automatically focuses the only input`);
    assert.equal(inspected[0].text,text,'the filled baseline or original text is immediately previewed');
    assert.equal(selectedIndex,'unchanged','text input focus must not invoke the image dataset browser');
    assert.equal(textCalls[1].body.input_index,0);
    assert.match(get('embedding-meta').textContent,new RegExp(method));
    assert.match(get('embedding-meta').textContent,/original.*1 action variant.*2 total/i,'the PCA fitting population includes the action variants');
    assert.match(get('embedding-meta').title,/attention-mask mean pooled/,'the precise feature method remains inspectable');
    assert.ok(!get('embedding-meta').textContent.includes('images'));
    assert.equal(get('embedding-action').children[0].textContent,'Original input');
    assert.ok(!get('embedding-action').children[1].textContent.includes('undefined'),'text action labels never read image-only dimensions');
    get('embedding-action').value='0';get('embedding-action').events.change();
    assert.equal(inspected[1].text,actionText);
    assert.equal(inspected[1].similarity,.875,'the inspector receives full-space cosine from the server');
    get('embedding-action').value='original';get('embedding-action').events.change();
    assert.equal(inspected[1],null);
    explorer.clearFocus();
    assert.match(status,/input/i);
  }
  console.log('PASS: dataset overview, focus/expansion, action inspector, collapse, stale responses and model errors');
  console.log('PASS: text domains focus their input, label encoder and PCA population, and preserve text/cosine selection');
  console.log('PASS: selected Analyze device, actual device/timings, cached stages and stale result clearing');
})().catch(err=>{console.error(err);process.exitCode=1;});
