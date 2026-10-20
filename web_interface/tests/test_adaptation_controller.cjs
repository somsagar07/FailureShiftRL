'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
class Element {
  constructor(){this.value='';this.checked=false;this.children=[];this.events={};this.hidden=false;this.disabled=false;this.textContent='';this.attrs={};this.style={};}
  addEventListener(type,fn){this.events[type]=fn;}
  replaceChildren(...nodes){this.children=nodes;}
  append(...nodes){this.children.push(...nodes);}
  setAttribute(key,value){this.attrs[key]=value;}
  scrollIntoView(){}
}
function setup(task='classification') {
  const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
  const context={console,TextEncoder,document:{getElementById:get,createElement:()=>new Element()}};vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/adaptation-controller.js'),'utf8'),context);
  const Controller=vm.runInContext('AdaptationController',context),calls=[],responses=[],rings=[],focuses=[],training=[];
  const explorer={task,job:{id:'emb1'},point:null,view:{setAdaptation:ids=>rings.push([...ids])},focusMeasured:(...args)=>focuses.push(args),describeAction:action=>action.name||`Rotation ${action.rotation}`};
  const discovery={active:false,attached:true,trainingActive:false,job:{id:'dis1',embedding_id:'emb1',task,status:'completed',records:[{input_index:0,action_id:4}]},startTraining:config=>training.push(config)};
  const api=async(url,body)=>{calls.push({url,body});const value=responses.shift();if(value instanceof Error)throw value;if(typeof value==='function')return value();if(!value)throw Error('Unexpected API call');return value;};
  for(const [id,value] of Object.entries({'finetune-epochs':'1','finetune-lr':'0.001','finetune-device':'auto','finetune-batch-size':'auto','finetune-seed':'42','finetune-weight-decay':'0.01','finetune-max-grad-norm':'1'}))get(id).value=value;
  get('finetune-cleanup').checked=true;
  const controller=new Controller({api,explorer,discovery});
  const summary={discovery_id:'dis1',embedding_id:'emb1',task,input_count:task==='classification'?50:1,max_examples:null,
    actions:[{action_id:4,action:{id:4,name:task==='summarization'?'remove_punctuation':'rotation',rotation:5},evaluated_inputs:10,failure_count:8,failure_rate:.8,mean_severity:.9,example_input_index:0},
      {action_id:9,action:{id:9,name:task==='summarization'?'negate_sentence':'darkness'},evaluated_inputs:5,failure_count:3,failure_rate:.6,mean_severity:.7,example_input_index:1},
      {action_id:11,action:{id:11,name:'saturation'},evaluated_inputs:0,failure_count:0,failure_rate:null,mean_severity:null,example_input_index:null}],
    suggested_action_ids:[4],defaults:{include_originals:task!=='generation',source_text:'A trial was successful.',reference_summary:'A trial succeeded.'}};
  return {controller,get,explorer,discovery,calls,responses,rings,focuses,training,summary};
}
const tests=[];const test=(name,fn)=>tests.push({name,fn});
const plain=value=>JSON.parse(JSON.stringify(value));
const content=el=>[el.textContent,...el.children.map(content)].join(' ');
async function open(f){f.responses.push(f.summary);await f.controller.show();}

test('generation datasets require explicit prompt/action target pairs and count only supplied pairs',async()=>{
  const f=setup('generation'),{controller:c,get}=f;
  f.summary.input_count=101;Object.assign(f.summary.defaults,{uses_dataset:true,dataset:{id:'prompts',count:101}});
  await open(f);c.toggle(9);
  assert.equal(get('adapt-data-source').value,'json');assert.equal(get('adapt-targets').children.length,0);
  assert.match(get('adapt-json-help').textContent,/input_index.*action_id.*image_path/);
  assert.match(get('adapt-data-help').textContent,/prompt\/action pairs/);
  assert.match(get('adapt-summary').textContent,/prompt\/action pairs/);
  assert.equal(get('finetune-start').disabled,true,'generated outputs never become targets automatically');
  const set=records=>{get('adapt-data-json').value=JSON.stringify(records);get('adapt-data-json').events.input();};
  set([{action_id:4,image_path:'/data/bird.png'},{input_index:3,action_id:9,image_path:'/data/vase.png'}]);
  assert.equal(get('finetune-start').disabled,true);assert.match(get('adapt-training-blocker').textContent,/input_index/);
  set([{input_index:0,action_id:4,image_path:'/data/bird.png'},{input_index:3,action_id:9,image_path:'/data/vase.png'}]);
  assert.equal(get('finetune-start').disabled,false);assert.equal(c.trainingConfig().training_data.length,2);assert.equal(c.exampleCount,2);
  assert.match(get('adapt-training-count').textContent,/2.*per epoch/);
  set([{input_index:101,action_id:4,image_path:'/data/bird.png'},{input_index:3,action_id:9,image_path:'/data/vase.png'}]);
  assert.equal(get('finetune-start').disabled,true);
  set([{input_index:0,action_id:4,image_path:'/data/bird.png',prompt:'Unrelated scene'},{input_index:3,action_id:9,image_path:'/data/vase.png'}]);
  assert.equal(get('finetune-start').disabled,true);assert.match(get('adapt-training-blocker').textContent,/prompt.*override|override.*prompt/i);
  set([{input_index:0,action_id:4,image_path:'/data/bird.png'},{input_index:0,action_id:4,image_path:'/data/other.png'},{input_index:3,action_id:9,image_path:'/data/vase.png'}]);
  assert.equal(get('finetune-start').disabled,true);assert.match(get('adapt-training-blocker').textContent,/duplicate/i);
});

test('Adapt is gated by a finished run with measurements from this analysis',async()=>{
  const f=setup(),{discovery:d,controller:c,get,calls}=f;
  for(const change of [{job:null},{job:{...d.job,status:'running'}},{job:{...d.job,records:[]}},{attached:false}]){
    const previous={...d};Object.assign(d,change);c.controls();assert.equal(get('adapt').disabled,true);await c.show();assert.equal(calls.length,0);Object.assign(d,previous);
  }
  await open(f);assert.equal(get('adapt').disabled,false);assert.equal(get('adapt-panel').hidden,false);
  assert.deepEqual(f.rings.at(-1),[4]);assert.equal(f.training.length,0,'opening Adapt does not train');
});

test('global action selection adds graph actions, removes suggestions, and survives reopening',async()=>{
  const f=setup(),{controller:c,get}=f;await open(f);
  c.setGraphAction(0,11);get('adapt-toggle-action').events.click();
  assert.deepEqual([...c.selected],[4,11]);assert.deepEqual(f.rings.at(-1),[4,11]);
  c.setGraphAction(27,11);assert.match(get('adapt-toggle-action').textContent,/Remove/);
  get('adapt-toggle-action').events.click();c.toggle(4);
  assert.equal(c.selected.size,0);assert.equal(get('finetune-start').disabled,true);
  c.close();await c.show();assert.equal(c.selected.size,0,'reopening never silently reselects removed actions');
  assert.equal(f.calls.length,1);
});

test('clicking a graph action exposes its global FT add/remove control in every workspace',async()=>{
  for(const task of ['classification','summarization','generation']) {
    const f=setup(task),{controller:c,get}=f;await open(f);
    c.setGraphAction(0,11);assert.equal(get('adapt-graph-action').hidden,false);
    assert.match(get('adapt-graph-toggle').textContent,/Add to FT actions/);
    get('adapt-graph-toggle').events.click();assert.equal(c.selected.has(11),true);
    c.setGraphAction(27,11);assert.match(get('adapt-graph-toggle').textContent,/Remove from FT actions/);
    get('adapt-graph-toggle').events.click();assert.equal(c.selected.has(11),false);
    c.setGraphAction(0,null);assert.equal(get('adapt-graph-action').hidden,true,'original inputs cannot be added as global actions');
    c.setGraphAction(0,4);f.discovery.active=true;c.controls();assert.equal(get('adapt-graph-toggle').disabled,true);
    get('adapt-graph-toggle').events.click();assert.equal(c.selected.has(4),true,'training locks action edits');
    f.discovery.active=false;c.close();assert.equal(get('adapt-graph-action').hidden,true);
  }
});

test('classification emits selected actions across all inputs and clean examples',async()=>{
  const f=setup(),{controller:c,get}=f;await open(f);c.toggle(9);
  const plan=c.trainingConfig();
  assert.deepEqual(plain(plan.action_ids),[4,9]);assert.equal(plan.include_originals,true);
  assert.equal(plan.examples,undefined);assert.equal(plan.training_data,undefined);
  assert.match(get('adapt-training-count').textContent,/150/);
  get('adapt-include-originals').checked=false;get('adapt-include-originals').events.change();
  assert.match(get('adapt-training-count').textContent,/100/);
  await c.start();assert.deepEqual(plain(f.training[0]),{action_ids:[4,9],include_originals:false});
  assert.match(content(get('adapt-selection')),/80.*8.*10/);
});

test('summary dataset training retains per-article references and counts all analyzed inputs',async()=>{
  const f=setup('summarization'),{controller:c,get}=f;
  f.summary.input_count=101;
  Object.assign(f.summary.defaults,{uses_dataset:true,dataset:{id:'articles',count:101,reference_count:101,text_column:'article',summary_column:'highlights'}});
  await open(f);
  assert.equal(get('adapt-summary-field').hidden,true,'one shared reference cannot replace the dataset references');
  assert.match(get('adapt-current-option').textContent,/dataset/i);
  assert.match(get('adapt-data-help').textContent,/101.*articles.*101.*101.*reference/i);
  assert.deepEqual(plain(c.trainingConfig()),{action_ids:[4],include_originals:true});
  assert.match(get('adapt-training-count').textContent,/202.*per epoch/);
  get('adapt-include-originals').checked=false;get('adapt-include-originals').events.change();
  assert.match(get('adapt-training-count').textContent,/101.*per epoch/);
  await c.start();assert.equal(f.training[0].training_data,undefined,'the server streams original article/reference pairs');
});

test('summary datasets require per-article corrected summaries for negation',async()=>{
  const f=setup('summarization'),{controller:c,get}=f;
  f.summary.input_count=2;Object.assign(f.summary.defaults,{uses_dataset:true,dataset:{id:'articles',count:2,reference_count:2}});
  await open(f);c.toggle(9);
  assert.equal(get('finetune-start').disabled,true);
  assert.equal(get('adapt-targets').children.length,0,'dataset negation never creates one global corrected target');
  assert.equal(get('adapt-training-blocker').textContent,'Remove Negate sentence or supply training JSON with corrected summaries for each article.');
  await c.start();assert.equal(f.training.length,0);
  get('adapt-data-source').value='json';get('adapt-data-source').events.change();
  assert.doesNotMatch(get('adapt-data-help').textContent,/analyzed articles/,'JSON training describes its supplied articles');
  get('adapt-data-json').value=JSON.stringify([{text:'First article.',summary:'First reference.',action_summaries:{9:'Corrected first.'}},{text:'Second article.',summary:'Second reference.',action_summaries:{9:'Corrected second.'}}]);
  get('adapt-data-json').events.input();assert.equal(get('finetune-start').disabled,false);
  const config=c.trainingConfig();assert.equal(config.training_data.length,2);assert.equal(config.training_data[1].action_summaries[9],'Corrected second.');
  assert.match(get('adapt-training-count').textContent,/6.*per epoch/);
});

test('full training selections are enabled beyond the old cap with exact epoch counts',async()=>{
  const f=setup(),{controller:c,get}=f;f.summary.input_count=1000;await open(f);c.toggle(9);c.toggle(11);
  assert.equal(get('finetune-start').disabled,false);assert.equal(c.exampleCount,4000);
  assert.match(get('adapt-training-count').textContent,/4,000.*per epoch/);
  f.summary.input_count=50000;get('finetune-epochs').value='3';get('finetune-epochs').events.input();
  assert.equal(c.exampleCount,200000);assert.match(get('adapt-training-count').textContent,/200,000.*per epoch.*600,000.*total/);
  await c.start();assert.equal(f.training.length,1);assert.deepEqual(plain(f.training[0].action_ids),[4,9,11]);
});

test('training JSON accepts more than 2048 records while retaining a byte limit',async()=>{
  const f=setup('summarization'),{controller:c,get}=f;await open(f);
  get('adapt-data-source').value='json';get('adapt-data-source').events.change();
  get('adapt-data-json').value=JSON.stringify(Array.from({length:3000},(_,i)=>({text:`Article ${i}`,summary:`Reference ${i}`})));
  get('adapt-data-json').events.input();assert.equal(get('finetune-start').disabled,false);assert.equal(c.exampleCount,6000);
  assert.throws(()=>c.parseDataset(' '.repeat(20*1024*1024+1)),/20 MB/);
});

test('invalid settings and missing selections explain why Start is disabled',async()=>{
  const f=setup(),{controller:c,get}=f;await open(f);
  for(const [id,invalid,valid,reason] of [['finetune-epochs','0','2',/epochs/i],['finetune-lr','0.2','0.001',/learning rate/i],['finetune-batch-size','1.5','32',/batch/i]]) {
    get(id).value=invalid;(get(id).events.input||get(id).events.change)();
    assert.equal(get('finetune-start').disabled,true);assert.match(get('adapt-training-blocker').textContent,reason);
    await c.start();assert.equal(f.training.length,0);
    get(id).value=valid;(get(id).events.input||get(id).events.change)();assert.equal(get('finetune-start').disabled,false);
  }
  assert.equal(c.trainingSettings().batch_size,32);
  c.toggle(4);assert.match(get('adapt-training-blocker').textContent,/select.*action/i);
});

test('additional training controls validate boundaries before a job can start',async()=>{
  const f=setup(),{controller:c,get}=f;await open(f);
  for(const [id,values,valid,reason] of [
    ['finetune-seed',['','-1','1.5','4294967296'],'4294967295',/seed/i],
    ['finetune-weight-decay',['','-0.1','1.1','Infinity'],'1',/weight decay/i],
    ['finetune-max-grad-norm',['','-1','100.01','Infinity'],'100',/gradient|clipping/i]
  ]) {
    for(const value of values) {
      get(id).value=value;get(id).events.input?.();
      assert.equal(get('finetune-start').disabled,true,`${id} rejects ${value}`);
      assert.match(get('adapt-training-blocker').textContent,reason);
      await c.start();assert.equal(f.training.length,0,'invalid options never start training');
    }
    get(id).value=valid;get(id).events.input();assert.equal(get('finetune-start').disabled,false);
  }
  get('finetune-seed').value='0';get('finetune-weight-decay').value='0';get('finetune-max-grad-norm').value='0';
  get('finetune-cleanup').checked=false;
  assert.deepEqual(plain(c.trainingSettings()),{epochs:1,learning_rate:.001,device:'auto',batch_size:'auto',seed:0,weight_decay:0,max_grad_norm:0,cleanup_after_finetune:false});
  f.discovery.active=true;c.controls();assert.equal(get('adapt-settings').disabled,true,'all options remain in the training lock');
});

test('summarization defaults to analyzed reference and requires a corrected negation target',async()=>{
  const f=setup('summarization'),{controller:c,get}=f;await open(f);
  assert.deepEqual(plain(c.trainingConfig().training_data),[{text:'A trial was successful.',summary:'A trial succeeded.'}]);
  c.toggle(9);assert.equal(get('finetune-start').disabled,true);
  const input=c.targetFields.get(9);input.value='A trial did not succeed.';input.events.input();
  const data=plain(c.trainingConfig().training_data);
  assert.deepEqual(data[0].action_summaries,{'9':'A trial did not succeed.'});assert.equal(get('finetune-start').disabled,false);
  get('adapt-data-source').value='json';get('adapt-data-source').events.change();
  get('adapt-data-json').value=JSON.stringify([{text:'One article',summary:'One summary',action_summaries:{9:'Changed one'}},{text:'Two articles',summary:'Two summaries',action_summaries:{9:'Changed two'}}]);
  get('adapt-data-json').events.input();assert.equal(c.trainingConfig().training_data.length,2);assert.match(get('adapt-training-count').textContent,/6/);
});

test('generation requires reference images for every selected action and supports multiple examples',async()=>{
  const f=setup('generation'),{controller:c,get}=f;await open(f);c.toggle(9);
  assert.equal(get('finetune-start').disabled,true);
  for(const [id,value] of [[4,'/data/a.png\n/data/b.png'],[9,'/data/c.png']]){const field=c.targetFields.get(id);field.value=value;field.events.input();}
  const config=plain(c.trainingConfig());assert.equal(config.include_originals,false);
  assert.deepEqual(config.training_data,[{action_id:4,image_path:'/data/a.png'},{action_id:4,image_path:'/data/b.png'},{action_id:9,image_path:'/data/c.png'}]);
  assert.match(get('adapt-training-count').textContent,/3/);assert.equal(get('finetune-start').disabled,false);
  c.toggle(9);assert.equal(c.trainingConfig().training_data.length,2,'removed actions do not retain training examples');
});

test('generation JSON examples use a real grouped action ID for single prompts and datasets',async()=>{
  for(const dataset of [false,true]) {
    const f=setup('generation'),{controller:c,get}=f;
    f.summary.actions=f.summary.actions.slice(0,2).map((row,index)=>({...row,action_id:index?215:43,
      action:{id:index?215:43,style:index?'3D render':'photograph',lighting:index?'low-key':'even',detail:index?'intricate':'minimal'}}));
    f.summary.suggested_action_ids=[215];
    if(dataset)Object.assign(f.summary.defaults,{uses_dataset:true,dataset:{id:'prompts',count:1}});
    await open(f);
    get('adapt-data-source').value='json';get('adapt-data-source').events.change();
    const example=JSON.parse(get('adapt-data-json').placeholder);
    assert.equal(example[0].action_id,215);
    assert.equal(example[0].input_index,dataset?0:undefined);
    get('adapt-data-json').value=JSON.stringify(example);get('adapt-data-json').events.input();
    assert.equal(get('finetune-start').disabled,false,'the shown example accepts the current selected action');
    assert.equal(c.trainingConfig().training_data[0].action_id,215);
    c.toggle(215);
    assert.equal(JSON.parse(get('adapt-data-json').placeholder)[0].action_id,43,'empty selection still shows an ID from this analysis');
  }
});

test('generation JSON keeps the corpus while removing actions from the training selection',async()=>{
  const f=setup('generation'),{controller:c,get}=f;await open(f);c.toggle(9);
  get('adapt-data-source').value='json';get('adapt-data-source').events.change();
  const corpus=JSON.stringify([{action_id:4,image_path:'/data/a.png'},{action_id:9,image_path:'/data/b.png'}]);
  get('adapt-data-json').value=corpus;get('adapt-data-json').events.input();
  assert.equal(c.trainingConfig().training_data.length,2);c.toggle(9);
  assert.deepEqual(plain(c.trainingConfig().training_data),[{action_id:4,image_path:'/data/a.png'}]);
  assert.equal(get('adapt-data-json').value,corpus);c.toggle(9);assert.equal(c.trainingConfig().training_data.length,2);
});

test('training limits and malformed datasets block requests',async()=>{
  const f=setup('summarization'),{controller:c,get}=f;await open(f);
  get('adapt-data-source').value='json';get('adapt-data-source').events.change();get('adapt-data-json').value='[]';get('adapt-data-json').events.input();
  await c.start();assert.equal(f.training.length,0);assert.match(get('adapt-error').textContent,/training|record|pair/i);
  get('adapt-data-json').value='[{"text":"article","summary":"reference"}]';get('adapt-data-json').events.input();
  f.summary.max_examples=1;assert.throws(()=>c.trainingConfig(),/limit|maximum|at most/i);
});

test('late summary responses and uploads cannot populate a new analysis',async()=>{
  const f=setup(),{controller:c,get}=f;let release;f.responses.push(()=>new Promise(resolve=>{release=resolve;}));
  const opening=c.show();c.reset();f.discovery.attached=false;release(f.summary);await opening;
  assert.equal(c.summary,null);assert.deepEqual(f.rings.at(-1),[]);assert.equal(get('adapt-panel').hidden,true);
  f.discovery.attached=true;await open(f);let read;
  get('adapt-data-file').files=[{size:20,name:'pairs.json',text:()=>new Promise(resolve=>{read=resolve;})}];const uploading=c.upload();c.reset();read('[{"text":"a","summary":"b"}]');await uploading;
  assert.equal(get('adapt-data-json').value,'');assert.equal(c.summary,null);
});

test('selection is locked during training while the training panel remains available after input changes',async()=>{
  const f=setup(),{controller:c,get,discovery:d}=f;await open(f);d.active=true;d.trainingActive=true;c.controls();
  c.toggle(9);assert.deepEqual([...c.selected],[4]);assert.equal(get('adapt-toggle-action').disabled,true);
  assert.equal(get('finetune-start').disabled,true);d.attached=false;c.reset();assert.equal(get('adapt-panel').hidden,false);
});

test('closing while loading does not focus the graph and changing data source ignores a late upload',async()=>{
  const f=setup('summarization'),{controller:c,get}=f;let release;
  f.responses.push(()=>new Promise(resolve=>{release=resolve;}));const showing=c.show();c.close();release(f.summary);await showing;
  assert.equal(c.summary,null);assert.equal(f.focuses.length,0);assert.equal(get('adapt-panel').hidden,true);
  await open(f);get('adapt-data-source').value='json';get('adapt-data-source').events.change();
  let finish;get('adapt-data-file').files=[{size:100,text:()=>new Promise(resolve=>{finish=resolve;})}];const upload=c.upload();
  get('adapt-data-source').value='current';get('adapt-data-source').events.change();finish('[{"text":"late article","summary":"late reference"}]');await upload;
  assert.equal(get('adapt-data-source').value,'current');assert.equal(get('adapt-data-json').value,'');
});

test('loading Adapt preserves an existing graph action and later graph selections',async()=>{
  for(const timing of ['before opening','while loading','clear while loading']) {
    const f=setup(),{controller:c,get}=f;let finish;
    if(timing==='before opening')c.setGraphAction(0,9);
    f.responses.push(()=>new Promise(resolve=>{finish=resolve;}));const showing=c.show();
    if(timing!=='before opening')c.setGraphAction(0,timing==='clear while loading'?null:9);
    finish(f.summary);await showing;
    const expected=timing==='clear while loading'?null:9;
    assert.equal(c.graphAction,expected,`${timing}: the summary must not replace the user's selection`);
    assert.equal(f.focuses.length,0,'suggested FT actions do not steal focus from a newer graph choice');
    assert.equal(get('adapt-graph-action').hidden,expected===null);
    assert.deepEqual([...c.selected],[4],'suggestions still initialize the training selection');
  }
});

(async()=>{let failures=0;for(const {name,fn} of tests){try{await fn();console.log('ok - '+name);}catch(error){failures++;console.error('not ok - '+name+'\n'+error.stack);}}if(failures)process.exitCode=1;else console.log(`${tests.length} adaptation controller tests passed`);})();
