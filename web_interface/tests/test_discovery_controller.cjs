// Real controller against DOM/network/timer boundaries; no model downloads.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
class Element {
  constructor(){this.value='';this.children=[];this.events={};this.hidden=false;this.disabled=false;this.textContent='';}
  addEventListener(type,fn){this.events[type]=fn;}
  replaceChildren(...nodes){this.children=nodes;}
  append(...nodes){this.children.push(...nodes);}
}
function setup(task='classification') {
  const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
  const timers=new Map();let timer=0;
  const context={console,document:{getElementById:get,createElement:()=>new Element()},setTimeout:fn=>{timers.set(++timer,fn);return timer;},clearTimeout:id=>timers.delete(id)};
  vm.createContext(context);vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/discovery-controller.js'),'utf8'),context);
  const Controller=vm.runInContext('DiscoveryController',context);
  const calls=[],responses=[],overlays=[],focuses=[],checkpoints=[];
  const explorer={task,job:null,view:{setDiscovery:job=>overlays.push(job)},focusMeasured:(...args)=>focuses.push(args)};
  const controller=new Controller({explorer,api:async(url,body)=>{calls.push({url,body});const response=responses.shift();if(response instanceof Error)throw response;if(typeof response==='function')return response();if(!response)throw new Error('Unexpected request '+url);return response;},onCheckpoint:value=>checkpoints.push(value)});
  for(const [id,value] of Object.entries({'discovery-episodes':'10','discovery-seed':'42','discovery-threshold':'.15','discovery-device':'cpu','discovery-size':'256','discovery-steps':'12','discovery-metric':'grounded','finetune-epochs':'1'}))get(id).value=value;
  const ready=()=>{explorer.job={id:'emb1'};controller.ready(explorer.job);};
  return {get,controller,calls,responses,overlays,focuses,checkpoints,explorer,timers,ready};
}
const record=(input_index=0,action_id=4)=>({input_index,action_id,visits:1,severity:.8,reward:.4,failure_rate:1,criterion:'confirmed error',metrics:{confidence_drop:.6},output:{class_id:3,label:'dog',confidence:.3}});
const job=(extra={})=>({id:'discovery1',embedding_id:'emb1',task:'classification',status:'running',episode:1,episodes:10,epsilon:.96,loss:.12,reward:.4,mean_reward:.4,unique_evaluations:1,model_calls:2,records:[record()],baseline:[{input_index:0,output:{class_id:2,label:'cat',confidence:.9}}],current:{input_index:0,action_id:4},...extra});
const tests=[];function test(name,fn){tests.push({name,fn});}

test('Analyze gates discovery; labels are explicit and progress/inspection never expand embeddings',async()=>{
  const f=setup(),{controller:c,get,responses,calls}=f;
  assert.equal(get('find-failure').disabled,true);await c.start();assert.equal(calls.length,0);
  f.ready();assert.equal(get('find-failure').disabled,false);
  get('discovery-label-map').value='{"0":"cat"}';responses.push(job());await c.start();
  assert.deepEqual(JSON.parse(JSON.stringify(calls[0].body)),{embedding_id:'emb1',episodes:10,seed:42,device:'cpu',label_map:{0:'cat'}});
  assert.match(get('discovery-metrics').textContent,/Loss 0.120.*epsilon 0.960.*1 measured pairs.*2 selected-model calls/);
  assert.equal(get('discovery-cancel').hidden,false);assert.equal(get('finetune-panel').hidden,true);
  assert.match(get('discovery-output').children[0].textContent,/dog/);assert.match(get('discovery-baseline').children[0].textContent,/cat/);
  assert.equal(f.focuses.length,0,'polling never takes over selection');
  get('discovery-pair').events.change();assert.deepEqual(f.focuses[0],[0,4]);assert.equal(calls.length,1,'selection only highlights an existing point');
  c.inspectPair(0,99);assert.match(get('discovery-components').textContent,/not been evaluated/);assert.equal(get('finetune-add').disabled,true);
  c.receive('discoveries',job({episode:2}),c.version);assert.match(get('discovery-components').textContent,/not been evaluated/,'new polls preserve the selected unvisited action');
});

test('errors permit retry/cancel; terminal cancellation preserves measured results',async()=>{
  const f=setup(),{controller:c,responses,get}=f;f.ready();responses.push(job());await c.start();
  responses.push(new Error('offline'));await c.poll('discoveries',c.version);
  assert.equal(get('discovery-retry').hidden,false);assert.equal(get('discovery-cancel').hidden,false);assert.equal(c.active,true);
  responses.push(job({status:'cancelled',phase:'Cancelled'}));await c.cancel('discoveries');
  assert.equal(c.active,false);assert.equal(get('discovery-cancel').hidden,true);assert.equal(get('finetune-panel').hidden,false);
  assert.equal(c.job.records.length,1);assert.equal(get('discovery-retry').hidden,true);
});

test('explicit reviewed targets, replace/remove examples and actual training/checkpoint workflow',async()=>{
  const f=setup(),{controller:c,responses,calls,get}=f;f.ready();responses.push(job({status:'completed'}));await c.start();
  await c.startTraining();assert.equal(calls.length,1,'no implicit training from discovery completion');
  c.addExample();assert.match(get('discovery-error').textContent,/explicit desired target/);assert.equal(c.examples.length,0);
  get('finetune-target').value='2';c.addExample();assert.equal(c.examples[0].target,2);
  get('finetune-target').value='exact class name';c.addExample();assert.equal(c.examples.length,1);assert.equal(c.examples[0].target,'exact class name');
  get('finetune-examples').children[0].children[1].events.click();assert.equal(c.examples.length,0);
  get('finetune-target').value='cat';c.addExample();responses.push({id:'train1',status:'running',completed:1,total:2,loss:.3});await c.startTraining();
  assert.equal(calls[1].url,'/api/finetunes');assert.deepEqual(JSON.parse(JSON.stringify(calls[1].body.examples)),[{input_index:0,action_id:4,target:'cat'}]);
  assert.equal(get('finetune-cancel').hidden,false);assert.equal(get('finetune-checkpoint').hidden,true);
  responses.push({id:'train1',status:'completed',completed:2,total:2,loss:.2,checkpoint:'/models/reviewed'});await c.poll('finetunes',c.version);
  assert.equal(get('finetune-checkpoint').hidden,false);get('finetune-use').events.click();assert.deepEqual(f.checkpoints,['/models/reviewed']);
});

test('input changes hide stale outputs and leave cancellation available; old polls cannot replace a new run',async()=>{
  const f=setup(),{controller:c,responses,get,explorer,overlays}=f;f.ready();responses.push(job());await c.start();
  c.invalidate();explorer.job=null;
  assert.equal(get('discovery-inspector').hidden,true);assert.equal(get('discovery-cancel').hidden,false);
  responses.push(job({episode:2}));await c.poll('discoveries',c.version);assert.equal(overlays.at(-1),null,'detached job cannot color a changed workspace');
  responses.push(job({status:'cancelled'}));await c.cancel('discoveries');c.invalidate();f.ready();
  let resolve;responses.push(()=>new Promise(r=>{resolve=r;}));const starting=c.start();
  c.invalidate();explorer.job=null;resolve(job());await starting;
  assert.equal(get('discovery-cancel').hidden,false,'a start acknowledged after input changes remains cancellable');
  assert.equal(get('discovery-inspector').hidden,true);
  responses.push(job({status:'cancelled'}));await c.cancel('discoveries');c.invalidate();f.ready();
  const oldVersion=c.version;let finish; c.job=job();responses.push(()=>new Promise(r=>{finish=r;}));const poll=c.poll('discoveries',oldVersion);
  c.job.status='completed';c.invalidate();f.ready();responses.push(job({id:'new',status:'completed'}));await c.start();finish(job({id:'old'}));await poll;
  assert.equal(c.job.id,'new','version guard ignores a late old response');
});

test('domain settings submit explicit metric and real generation options',async()=>{
  for(const task of ['summarization','generation']){
    const f=setup(task),{controller:c,responses,get,calls}=f;f.ready();
    get('discovery-reference').value='Reviewed reference';get('discovery-metric').value='reference';
    responses.push(job({task,status:'completed'}));await c.start();
    if(task==='summarization'){assert.equal(calls[0].body.summary_metric,'reference');assert.equal(calls[0].body.reference_summary,'Reviewed reference');assert.match(get('finetune-target-label').textContent,/Corrected/);}
    else {assert.equal(calls[0].body.generation_steps,12);assert.equal(calls[0].body.generation_size,256);assert.match(get('finetune-target-label').textContent,/server path/);}
    assert.equal(calls[0].body.label_map,undefined);
  }
});

test('training cancellation remains available after form invalidation and failures can retry',async()=>{
  const f=setup(),{controller:c,responses,get}=f;f.ready();responses.push(job({status:'completed'}));await c.start();get('finetune-target').value='cat';c.addExample();
  responses.push({id:'train1',status:'running'});await c.startTraining();c.invalidate();
  assert.equal(get('finetune-cancel').hidden,false);assert.equal(get('finetune-panel').hidden,true);
  responses.push(new Error('temporary failure'));await c.cancel('finetunes');assert.equal(get('finetune-cancel').hidden,false);
  responses.push({id:'train1',status:'cancelled'});await c.cancel('finetunes');assert.equal(c.active,false);
});

function content(element){return [element.textContent,...element.children.map(content)].join(' ');}

test('measured outcomes use readable predictions, summaries, generated images and named scores',async()=>{
  const f=setup(),{controller:c,get,responses}=f;f.ready();responses.push(job({status:'completed'}));await c.start();
  assert.match(content(get('discovery-output')),/Prediction: dog \(class 3\).*Confidence: 30.0%/);
  assert.match(content(get('discovery-components')),/Severity 80.0%.*Target confidence drop 0.600/);
  assert.ok(!content(get('discovery-output')).includes('"class_id"'),'outputs do not expose API JSON');
  c.output('discovery-output',{summary:'First summary paragraph.\n\nSecond paragraph.',quality:.8});
  assert.equal(get('discovery-output').children[0].textContent,'First summary paragraph.');
  assert.equal(get('discovery-output').children[1].textContent,'Second paragraph.');
  c.output('discovery-output',{image:'/discoveries/job/artifacts/image.png',clip_similarity:.37,prompt:'A scientist',seed:42});
  assert.equal(get('discovery-output').children[0].src,'/discoveries/job/artifacts/image.png');
  assert.match(content(get('discovery-output')),/CLIP prompt alignment: 0.370.*Prompt: A scientist.*Noise seed: 42/);
});

test('policy inspector exposes mean, variation and tested-input coverage for measured and unvisited actions',async()=>{
  const f=setup(),{controller:c,get,responses}=f;f.ready();
  const statistics={method:'softmax_q_across_original_inputs',temperature:1,input_count:4,policy_updates:10,std_ddof:1,actions:[
    {action_id:4,mean:.6,std:.02,evaluated_inputs:3},{action_id:9,mean:.2,std:.08,evaluated_inputs:0}
  ]};
  responses.push(job({status:'completed',policy_statistics:statistics}));await c.start();
  assert.match(content(get('discovery-components')),/Mean DQN action preference 0\.600/);
  assert.match(content(get('discovery-components')),/Preference variation.*sample SD.*0\.0200/);
  assert.match(content(get('discovery-components')),/Original inputs.*4/);assert.match(content(get('discovery-components')),/Tested inputs.*3 \/ 4/);
  c.inspectPair(0,9);assert.match(content(get('discovery-components')),/not been evaluated/);
  assert.match(content(get('discovery-components')),/Mean DQN action preference 0\.200/);assert.match(content(get('discovery-components')),/Tested inputs.*0 \/ 4/);
  c.receive('discoveries',job({status:'completed',policy_statistics:{...statistics,input_count:1,actions:[{action_id:9,mean:.2,std:null,evaluated_inputs:1}]}}),c.version);
  assert.match(content(get('discovery-components')),/unavailable.*at least 2 original inputs/i,'one input never reports zero cross-input variation');
  c.receive('discoveries',job({status:'running',policy_statistics:{...statistics,policy_updates:0,actions:[]}}),c.version);
  assert.match(content(get('discovery-components')),/unavailable.*no policy updates/i,'an untrained policy is not presented as confident');
});

test('task training-rate defaults preserve edits, summary references validate locally and classification hides unused threshold',async()=>{
  const f=setup(),{controller:c,get,explorer,calls}=f;f.ready();
  assert.equal(get('finetune-lr').value,'0.001');assert.equal(get('discovery-threshold-field').hidden,true);
  get('finetune-lr').value='0.005';c.configure();c.configure();assert.equal(get('finetune-lr').value,'0.005','opening settings preserves an edit');
  explorer.task='summarization';c.configure();assert.equal(get('finetune-lr').value,'0.00002');assert.equal(get('discovery-threshold-field').hidden,false);
  for(const metric of ['grounded','reference']){get('discovery-metric').value=metric;await c.start();assert.match(get('discovery-error').textContent,/human reference summary/);}
  assert.equal(calls.length,0,'missing reference never starts a server job');
  explorer.task='generation';c.configure();assert.equal(get('finetune-lr').value,'0.0001');
  explorer.task='classification';c.configure();assert.equal(get('finetune-lr').value,'0.005','returning to a task preserves its edited rate');
});

test('an old job completing after a different analysis cannot display or fine-tune its records',async()=>{
  const f=setup(),{controller:c,get,responses,calls,explorer,overlays}=f;f.ready();responses.push(job());await c.start();
  c.invalidate();explorer.job={id:'emb2'};c.ready(explorer.job);
  assert.equal(c.analysisReady,true);assert.equal(c.attached,false);assert.equal(get('discovery-cancel').hidden,false,'old job remains cancellable');
  responses.push(job({status:'completed'}));await c.poll('discoveries',c.version);
  assert.equal(get('discovery-inspector').hidden,true);assert.equal(get('discovery-records').hidden,true);assert.equal(get('finetune-panel').hidden,true);assert.equal(overlays.at(-1),null);
  const before=calls.length;c.selected='0:4';get('finetune-target').value='cat';c.addExample();c.examples=[{input_index:0,action_id:4,target:'cat'}];await c.startTraining();assert.equal(calls.length,before,'mismatched discovery cannot provide training examples');
  c.training={status:'completed',checkpoint:'/old/model'};get('finetune-use').events.click();assert.equal(f.checkpoints.length,0);
  assert.equal(get('discovery-start').disabled,false,'the new analysis can start its own discovery');
  responses.push(job({id:'new',embedding_id:'emb2',status:'completed'}));await c.start();assert.equal(c.attached,true);assert.equal(c.job.id,'new');assert.equal(c.examples.length,0);
});

test('a delayed discovery GET cannot resurrect discovery or cancel fine-tuning polling',async()=>{
  for(const delayedStatus of ['running','completed']) {
    const f=setup(),{controller:c,get,responses,calls,timers}=f;f.ready();responses.push(job());await c.start();
    const discoveryVersion=c.version;let release;
    responses.push(()=>new Promise(resolve=>{release=resolve;}));const delayed=c.poll('discoveries',discoveryVersion);
    responses.push(job({status:'cancelled'}));await c.cancel('discoveries');
    get('finetune-target').value='cat';c.addExample();
    const beforeValidation=c.version,beforeCalls=calls.length;get('finetune-lr').value='0.2';await c.startTraining();
    assert.equal(c.version,beforeValidation,'invalid settings do not advance the polling generation');assert.equal(calls.length,beforeCalls,'out-of-range learning rates are rejected before POST');
    get('finetune-lr').value='0.001';
    responses.push({id:'train1',status:'running',completed:0,total:2});await c.startTraining();
    const trainingTimer=c.timer;
    assert.equal(timers.has(trainingTimer),true,'training has a scheduled status poll');
    release(job({status:delayedStatus}));await delayed;
    assert.equal(c.job.status,'cancelled','old discovery responses cannot resurrect or change the terminal job');
    assert.equal(c.timer,trainingTimer,'late discovery response cannot replace the training timer');
    assert.equal(timers.has(trainingTimer),true,'training polling stays active');
    responses.push({id:'train1',status:'completed',completed:2,total:2,checkpoint:'/models/reviewed'});
    await timers.get(trainingTimer)();
    assert.equal(calls.at(-1).url,'/api/finetunes/train1');assert.equal(c.training.status,'completed');
  }
});

(async()=>{let failures=0;for(const {name,fn} of tests){try{await fn();console.log('ok - '+name);}catch(error){failures++;console.error('not ok - '+name+'\n'+error.stack);}}if(failures)process.exitCode=1;else console.log(`${tests.length} discovery controller tests passed`);})();
