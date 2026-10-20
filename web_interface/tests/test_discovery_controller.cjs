// Real controller against DOM/network/timer boundaries; no model downloads.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
class Element {
  constructor(){this.value='';this.children=[];this.events={};this.hidden=false;this.disabled=false;this.textContent='';this.attrs={};this.style={};}
  addEventListener(type,fn){this.events[type]=fn;}
  replaceChildren(...nodes){this.children=nodes;}
  append(...nodes){this.children.push(...nodes);}
  setAttribute(key,value){this.attrs[key]=value;}
  focus(){}
  click(){this.events.click?.();}
}
function setup(task='classification') {
  const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
  const timers=new Map();let timer=0;
  const context={console,TextEncoder,document:{getElementById:get,createElement:()=>new Element()},setTimeout:fn=>{timers.set(++timer,fn);return timer;},clearTimeout:id=>timers.delete(id)};
  vm.createContext(context);
  for(const file of ['adaptation-controller.js','discovery-controller.js'])vm.runInContext(fs.readFileSync(path.join(__dirname,'../static',file),'utf8'),context);
  const Controller=vm.runInContext('DiscoveryController',context);
  const calls=[],responses=[],overlays=[],focuses=[],checkpoints=[];
  const explorer={task,job:null,view:{setDiscovery:job=>overlays.push(job)},focusMeasured:(...args)=>focuses.push(args)};
  for(const [id,value] of Object.entries({'finetune-epochs':'1','finetune-lr':'0.001','finetune-device':'auto','finetune-batch-size':'auto','finetune-seed':'42','finetune-weight-decay':'0.01','finetune-max-grad-norm':'1','discovery-episodes':'10','discovery-seed':'42','discovery-threshold':'.15','discovery-device':'cpu','discovery-size':'256','discovery-steps':'12','discovery-metric':'grounded','discovery-bias-repeats':'3'}))get(id).value=value;
  get('finetune-cleanup').checked=true;
  const controller=new Controller({explorer,api:async(url,body)=>{calls.push({url,body});const response=responses.shift();if(response instanceof Error)throw response;if(typeof response==='function')return response();if(!response)throw new Error('Unexpected request '+url);return response;},onCheckpoint:value=>checkpoints.push(value)});
  const ready=()=>{explorer.job=analysis();controller.ready(explorer.job);};
  return {get,controller,calls,responses,overlays,focuses,checkpoints,explorer,timers,ready};
}
const record=(input_index=0,action_id=4)=>({input_index,action_id,visits:1,severity:.8,reward:.4,failure_rate:1,criterion:'confirmed error',metrics:{confidence_drop:.6},output:{class_id:3,label:'dog',confidence:.3}});
const job=(extra={})=>({id:'discovery1',embedding_id:'emb1',task:'classification',status:'running',episode:1,episodes:10,epsilon:.96,loss:.12,reward:.4,mean_reward:.4,unique_evaluations:1,model_calls:2,records:[record()],baseline:[{input_index:0,output:{class_id:2,label:'cat',confidence:.9}}],current:{input_index:0,action_id:4},...extra});
const tests=[];function test(name,fn){tests.push({name,fn});}

test('generation submits one combined reward with its supported judge and severity threshold',async()=>{
  const f=setup('generation'),{controller:c,get,explorer,responses,calls}=f;
  explorer.preview={config:{task:'generation',dataset_id:'prompts'},dataset:{count:2,checked_count:2,missing_checks_count:0}};
  explorer.job={id:'emb1',task:'generation',points:[{index:0},{index:1}],requirements:{dataset_id:'prompts',total:2,count:2,missing:0}};
  c.ready(explorer.job);c.showDiscovery();
  assert.equal(get('discovery-generation-metric').value,'combined');assert.equal(get('discovery-threshold-field').hidden,false);
  assert.equal(get('discovery-judge-model').value,'CountingSheep/vev-4b');
  get('discovery-generation-metric').value='checks';
  responses.push(job({task:'generation',status:'completed',records:[]}));await c.start();
  assert.equal(calls[0].body.generation_metric,'combined');assert.equal(calls[0].body.judge_model,'CountingSheep/vev-4b');
  assert.equal(calls[0].body.failure_threshold,.5);assert.equal(calls[0].body.generation_bias_suite,null);assert.equal(calls[0].body.generation_bias_repeats,3);
});

test('missing generation checks use content review without blocking combined discovery',async()=>{
  const f=setup('generation'),{controller:c,get,explorer,calls,responses}=f;
  explorer.job={id:'emb1',task:'generation',points:[{index:0},{index:1}],requirements:{dataset_id:'prompts',total:2,count:1,missing:1}};
  c.ready(explorer.job);c.showDiscovery();assert.equal(get('discovery-generation-metric').value,'combined');
  assert.equal(get('discovery-start').disabled,false);assert.match(get('discovery-requirements-status').textContent,/1.*visual content review/i);
  responses.push(job({task:'generation',status:'completed',records:[]}));await c.start();
  assert.equal(calls[0].body.generation_metric,'combined');assert.equal(calls[0].body.judge_model,'CountingSheep/vev-4b');
});

test('Qwen remains an explicit evaluator option for combined discovery',async()=>{
  const f=setup('generation');f.ready();
  f.get('discovery-judge-model').value='Qwen/Qwen3-VL-2B-Instruct';
  f.responses.push(job({task:'generation',status:'completed'}));await f.controller.start();
  assert.equal(f.calls[0].body.judge_model,'Qwen/Qwen3-VL-2B-Instruct');
});

test('target verification is shown as support rather than a generated answer',()=>{
  const f=setup('generation');f.ready();
  f.controller.output('discovery-output',{checks_passed:1,checks_total:1,
    checks:[{question:'How many birds?',expected:'two',answer:'yes',passed:true}],
    check_evidence:[{method:'vev_target_verification',probabilities:[.1,.9]}]});
  const content=node=>[node.textContent||'',...(node.children||[]).map(content)].join(' ');
  const text=content(f.get('discovery-output'));
  assert.match(text,/How many birds.*Target: two.*Supported: yes.*90/);
  assert.doesNotMatch(text,/Answer: yes/);
});

test('generation precision can select full precision and is never sent to other workspaces',async()=>{
  for(const precision of ['auto','float32']) {
    const f=setup('generation'),{controller:c,get,calls,responses}=f;f.ready();
    if(precision==='float32')get('discovery-generation-precision').value=precision;
    responses.push(job({task:'generation',status:'completed'}));await c.start();
    assert.equal(calls[0].body.generation_precision,precision);
  }
  const f=setup('summarization');f.ready();f.get('discovery-reference').value='A human reference.';
  f.responses.push(job({task:'summarization',status:'completed'}));await f.controller.start();
  assert.equal(Object.hasOwn(f.calls[0].body,'generation_precision'),false);
});

const biasSuite={name:'Scientist scene quality',reference:'equal_error_rates',cases:[{name:'Laboratory objects',groups:[
  {name:'Woman prompt',prompt:'A woman scientist beside a blue flask and a red notebook.',checks:[{question:'Is there a blue flask?',answer:'yes'},{question:'Is there a red notebook?',answer:'yes'}]},
  {name:'Man prompt',prompt:'A man scientist beside a blue flask and a red notebook.',checks:[{question:'Is there a blue flask?',answer:'yes'},{question:'Is there a red notebook?',answer:'yes'}]}
]}]};

test('generation submits the optional matched comparison suite and repeats as structured inputs',async()=>{
  const f=setup('generation'),{controller:c,get,responses,calls}=f;f.ready();
  get('discovery-bias-suite').value=JSON.stringify(biasSuite);get('discovery-bias-repeats').value='5';
  responses.push(job({task:'generation',status:'completed'}));await c.start();
  assert.ok(calls[0].body.generation_bias_suite,'the comparison suite must reach discovery');
  assert.deepEqual(JSON.parse(JSON.stringify(calls[0].body.generation_bias_suite)),biasSuite);
  assert.equal(calls[0].body.generation_bias_repeats,5);
});

test('invalid bias JSON and repeat counts are rejected locally, and correction clears the error',async()=>{
  const f=setup('generation'),{controller:c,get,responses,calls}=f;f.ready();
  for(const value of ['{broken','[]','3']) {
    get('discovery-bias-suite').value=value;await c.start();
    assert.equal(calls.length,0);assert.match(get('discovery-error').textContent,/bias comparison.*JSON/i);
  }
  get('discovery-bias-suite').value=JSON.stringify(biasSuite);
  for(const value of ['1','17','2.5','Infinity','']) {
    get('discovery-bias-repeats').value=value;await c.start();
    assert.equal(calls.length,0);assert.match(get('discovery-error').textContent,/repeats.*2.*16/i);
  }
  get('discovery-bias-suite').value='  ';get('discovery-bias-repeats').value='3';
  responses.push(job({task:'generation',status:'completed'}));await c.start();
  assert.equal(calls[0].body.generation_bias_suite,null);assert.equal(get('discovery-error').hidden,true);
});

test('generation severity threshold and summary quality-drop threshold retain separate workspace edits',()=>{
  const f=setup(),{controller:c,get,explorer}=f;
  explorer.task='summarization';c.configure();assert.equal(Number(get('discovery-threshold').value),.15);
  get('discovery-threshold').value='.23';explorer.task='generation';c.configure();assert.equal(Number(get('discovery-threshold').value),.5);
  get('discovery-threshold').value='.65';get('discovery-bias-suite').value=JSON.stringify(biasSuite);get('discovery-bias-repeats').value='6';
  c.configure();explorer.job={id:'new-analysis'};c.ready(explorer.job);assert.equal(get('discovery-threshold').value,'.65');
  explorer.task='summarization';c.configure();assert.equal(get('discovery-threshold').value,'.23');
  explorer.task='classification';c.configure();assert.equal(get('discovery-threshold-field').hidden,true);
  explorer.task='generation';c.configure();assert.equal(get('discovery-threshold').value,'.65');
  assert.equal(get('discovery-bias-suite').value,JSON.stringify(biasSuite));assert.equal(get('discovery-bias-repeats').value,'6');
});

test('combined evidence is rendered only after Inspect and keeps missing measurements distinct from zero',async()=>{
  const f=setup('generation'),{controller:c,get,responses}=f;f.ready();
  const output={image:'/discoveries/run/output.png',failure_scores:{content:0,instructions:.5,visual:.75,bias:null},failure_evidence:{
    content:{status:'evaluated',method:'dataset_checks',score:0,reason:'All object checks passed.'},
    instructions:{status:'evaluated',method:'presentation_rubric',score:.5,question:'Review the requested lighting.',answer:'2'},
    visual:{status:'evaluated',method:'visual_defect_rubric',score:.75,question:'Review unintended visual defects.',answer:'3'},
    bias:{status:'not_evaluated',method:'matched_group_comparison',scope:'matched_suite_action',score:null,reason:'No matched comparison suite supplied.'}
  },checks_passed:1,checks_total:1,checks:[{question:'Is there a blue flask?',expected:'yes',answer:'yes',passed:true}]};
  const baseline={...output,failure_scores:{content:0,instructions:null,visual:.25,bias:null},failure_evidence:{...output.failure_evidence,instructions:{status:'not_applicable',method:'presentation_rubric',score:null,reason:'Baseline has no added presentation settings.'}}};
  const result=job({task:'generation',status:'completed',baseline:[{input_index:0,output:baseline}],records:[{...record(),severity:.75,reward:.75,output,metrics:{content_failure:0,instructions_failure:.5,visual_failure:.75,bias_failure:null,baseline_severity:.25,severity_increase:.5,quality:.25,baseline_quality:.75}}]});
  responses.push(result);await c.start();c.inspectPair(0,4);
  assert.equal(get('discovery-inspector').hidden,true);assert.equal(content(get('discovery-output')).includes('visual_defect_rubric'),false);
  inspect(get);assert.equal(get('discovery-inspector').hidden,false);
  assert.match(content(get('discovery-output')),/Content failure.*0\.000.*Evaluated.*Instructions.*0\.500.*Evaluated.*Visual defects.*0\.750.*Evaluated.*Bias disparity.*Not evaluated/s);
  assert.match(content(get('discovery-output')),/presentation_rubric.*Review the requested lighting.*Answer: 2/s);
  assert.match(content(get('discovery-output')),/No matched comparison suite supplied/);
  assert.doesNotMatch(content(get('discovery-output')),/Measured across the matched suite/,'an absent suite has no measured bias scope');
  assert.match(content(get('discovery-output')),/Is there a blue flask.*Expected: yes.*Answer: yes/);
  assert.match(content(get('discovery-baseline')),/Instructions.*Not applicable/);
  assert.match(content(get('discovery-components')),/Baseline severity.*0\.250.*Severity increase.*0\.500/s);
  assert.match(content(get('discovery-components')),/Combined quality \(1 − severity\).*0\.250/s);
  assert.doesNotMatch(content(get('discovery-components')),/Content requirements passed \(fraction\)/,'combined quality can be dominated by visual or instruction failure even when all checks pass');
  const bias={score:.4,status:'evaluated',method:'max_group_mean_content_failure_gap',scope:'matched_suite_action',reference:'equal_error_rates',suite_name:'Scientist scene quality',group_rates:{'Woman prompt':.5,'Man prompt':.1},group_sample_counts:{'Woman prompt':3,'Man prompt':3},sample_count:6,cases:[{name:'Laboratory objects',records:[{repeat:0,seed:42,groups:[{name:'Woman prompt',prompt:'A woman scientist beside a blue flask.',content_failure:.5}]}]}]};
  result.records[0].output={...output,failure_scores:{...output.failure_scores,bias:.4},failure_evidence:{...output.failure_evidence,bias}};
  c.receive('discoveries',result,c.version);
  assert.match(content(get('discovery-output')),/Bias disparity.*0\.400.*Evaluated/);
  assert.match(content(get('discovery-output')),/matched suite.*this action.*Woman prompt.*0\.500.*3.*Man prompt.*0\.100.*3/s);
  assert.match(content(get('discovery-output')),/equal_error_rates.*Scientist scene quality/s);
  assert.match(content(get('discovery-output')),/Laboratory objects.*"seed": 42.*A woman scientist beside a blue flask/s);
});

test('generation output requirements render expected and observed answers in inspect-only details',()=>{
  const {controller:c,get}=setup('generation');
  c.output('discovery-output',{image:'/discoveries/run/output.png',quality:.5,checks_passed:1,checks_total:2,checks:[
    {question:'How many birds?',expected:'2',answer:'1',passed:false,skill:'counting'},
    {question:'Is the vase blue?',expected:'yes',answer:'yes',passed:true}]});
  const output=get('discovery-output');assert.match(content(output),/1\s*\/\s*2.*How many birds.*Expected: 2.*Answer: 1.*Is the vase blue/);
  assert.equal(output.children.some(node=>node.value),false,'checks are evidence, never an editable target field');
});

test('summary dataset discovery uses stored per-row references without sending a global summary',async()=>{
  const f=setup('summarization'),{get,controller:c,explorer,responses,calls}=f;
  explorer.preview={config:{task:'summarization',dataset_id:'articles'},dataset:{id:'articles',count:2,reference_count:2}};
  explorer.job={id:'emb1',task:'summarization',points:[{index:0,name:'One'},{index:1,name:'Two'}],references:{dataset_id:'articles',total:2,count:2,missing:0}};
  get('discovery-reference').value='An old pasted reference.';c.ready(explorer.job);c.showDiscovery();
  assert.equal(get('discovery-reference-field').hidden,true);assert.match(get('discovery-reference-status').textContent,/2\s*\/\s*2/);
  assert.equal(get('discovery-start').disabled,false);
  responses.push(job({task:'summarization',status:'completed'}));await c.start();
  assert.equal(calls[0].body.reference_summary,undefined);assert.equal(calls[0].body.summary_metric,'grounded');
});

test('summary dataset missing references block discovery and retain Analyze inspection',async()=>{
  const f=setup('summarization'),{get,controller:c,explorer,calls}=f;
  explorer.preview={config:{task:'summarization',dataset_id:'articles'}};
  explorer.job={id:'emb1',task:'summarization',points:[{index:0},{index:1}],references:{dataset_id:'articles',total:2,count:1,missing:1}};
  c.ready(explorer.job);c.showDiscovery();assert.equal(get('discovery-start').disabled,true);
  assert.match(get('discovery-reference-status').textContent,/1\s*\/\s*2.*reference.*column.*dataset/i);
  assert.equal(get('find-failure').disabled,false);await c.start();assert.equal(calls.length,0);
  explorer.preview={config:{task:'summarization',text:'Pasted article.'}};explorer.job={id:'emb2',task:'summarization',points:[{index:0,text:'Pasted article.'}]};
  c.ready(explorer.job);assert.equal(get('discovery-reference-field').hidden,false);assert.equal(get('discovery-reference-status').hidden,true);
});
function inspect(get,key='0:4'){get('discovery-pair').value=key;get('discovery-pair').events.change();}
async function openAdapt(f) {
  const d=f.controller.job;
  f.responses.push({discovery_id:d.id,embedding_id:d.embedding_id,task:d.task,input_count:1,max_examples:2048,
    actions:[{action_id:4,action:{id:4,name:'remove_punctuation'},evaluated_inputs:1,failure_count:1,failure_rate:1,mean_severity:.8,example_input_index:0}],
    suggested_action_ids:[4],defaults:{include_originals:d.task!=='generation',source_text:'An article.',reference_summary:'A summary.'}});
  await f.controller.adaptation.show();
}
const modelLabels=Object.fromEntries(Array.from({length:128},(_,i)=>[i,i===2?'cat':i===3?'dog':`class-${i}`]));
const analysis=(extra={})=>({id:'emb1',points:[{index:0,name:'image-0.png'}],labels:{label_map:{0:2},values:{0:'cat'},loaded:1,model_labels:modelLabels},...extra});
function labelInputs(f,count=8,dataset='images',model='test/model',labels={}) {
  f.explorer.preview={config:{task:'classification',dataset_id:dataset,model:{source:'huggingface',name:model}}};
  f.explorer.job=analysis({points:Array.from({length:count},(_,index)=>({index,name:`image-${index}.png`,image:`/sample-${index}.png`})),labels:{label_map:labels,values:Object.fromEntries(Object.entries(labels).map(([index,id])=>[index,modelLabels[id]])),loaded:Object.keys(labels).length,column:'labels',model_labels:modelLabels}});
  f.controller.ready(f.explorer.job);
}
function typeLabels(get,text){get('discovery-label-map').value=text;get('discovery-label-map').events.input();}
async function uploadLabels(get,text,name='labels.json') {
  get('discovery-label-file').files=[{name,size:Buffer.byteLength(text),text:async()=>text}];
  await get('discovery-label-file').events.change();
}

test('full datasets resolve each unique typed label once without confusing numeric names with IDs',()=>{
  const f=setup(),{controller:c,get}=f;labelInputs(f,50000);
  f.explorer.job.labels.model_labels={0:'1',1:'0',2:'cat',3:'dog'};c.ready(f.explorer.job);
  let resolutions=0;const resolve=c.resolveLabel;
  c.resolveLabel=function(value){resolutions++;return resolve.call(this,value);};
  const values=Object.fromEntries(Array.from({length:50000},(_,index)=>[index,[0,'0','cat','dog','unknown'][index%5]]));
  typeLabels(get,JSON.stringify(values));
  assert.equal(resolutions,5,'known and unknown repeated labels resolve once per distinct typed value');
  assert.equal(Object.keys(c.labelResolved).length,40000);
  assert.equal(c.labelResolved[0],0);assert.equal(c.labelResolved[1],1);
  assert.equal(c.labelResolved[2],2);assert.equal(c.labelResolved[3],3);assert.equal(c.labelResolved[4],undefined);
  assert.equal(get('discovery-start').disabled,true,'unmatched targets remain blocked');
});

test('large automatic maps use compact JSON and remain editable within the supported text limit',()=>{
  const f=setup(),{controller:c,get}=f;
  const labels=Object.fromEntries(Array.from({length:150000},(_,index)=>[index,2]));
  labelInputs(f,150000,'images','test/model',labels);
  assert.equal(get('discovery-label-error').textContent,'','automatic labels must not fail because of indentation');
  assert.equal(get('discovery-start').disabled,false);
  assert.ok(get('discovery-label-map').value.length<=2*1024*1024);
  c.writeLabel('0','#3');
  assert.equal(get('discovery-label-error').textContent,'','row editing must retain the compact representation');
  assert.equal(get('discovery-start').disabled,false);
  assert.equal(c.labelResolved[0],3);assert.equal(c.labelResolved[149999],2);
});

test('all loaded dataset labels are distinguished from mismatches with the selected model',async()=>{
  const f=setup(),{controller:c,get,calls}=f;
  const names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'];
  const values=Object.fromEntries(Array.from({length:20},(_,i)=>[i,names[i%10]]));
  f.explorer.job=analysis({points:Array.from({length:20},(_,index)=>({index,name:`image-${index}.png`})),labels:{
    column:'label',loaded:20,ready:4,missing:0,unmatched:16,values,label_map:{3:2,5:3,13:2,15:3},model_labels:modelLabels,
    unmatched_classes:['airplane','automobile','bird'],unmatched_class_count:8}});
  c.ready(f.explorer.job);
  assert.match(get('discovery-label-summary').textContent,/4.*20.*16 model mismatches/i);
  assert.doesNotMatch(get('discovery-label-summary').textContent,/need review|missing label/i);
  assert.match(get('discovery-label-source').textContent,/20\s*\/\s*20 dataset labels loaded/i);
  assert.match(get('discovery-label-source').textContent,/choose.*model.*matching classes/i);
  assert.match(get('discovery-label-source').textContent,/airplane.*automobile.*bird.*5 more/i);
  assert.equal(get('discovery-start').disabled,true);await c.start();assert.equal(calls.length,0);
  get('discovery-label-view').events.click();
  const row=get('discovery-label-preview-rows').children[0];
  assert.equal(row.children[1].textContent,'Dataset');assert.equal(row.children[2].textContent,'airplane');
  assert.match(row.children[3].textContent,/Model class mismatch/);assert.doesNotMatch(row.children[3].textContent,/review/);
  get('discovery-label-edit').events.click();
  assert.equal(get('discovery-label-preview-rows').children[0].children[2].children[0].value,'airplane','editing retains the known dataset class name');
});

test('matching Beans classes are automatically ready and can start discovery without edits',async()=>{
  const f=setup(),{controller:c,get,responses,calls}=f;
  f.explorer.job=analysis({points:Array.from({length:3},(_,index)=>({index,name:`leaf-${index}.png`})),labels:{
    column:'labels',loaded:3,values:{0:'angular_leaf_spot',1:'bean_rust',2:'healthy'},label_map:{0:2,1:1,2:0},
    model_labels:{0:'healthy',1:'bean rust',2:'angular leaf spot'}}});
  c.ready(f.explorer.job);
  assert.match(get('discovery-label-source').textContent,/3\s*\/\s*3 dataset labels loaded/i);
  assert.equal(get('discovery-start').disabled,false);
  responses.push(job({status:'completed'}));await c.start();
  assert.deepEqual(JSON.parse(JSON.stringify(calls[0].body.label_map)),{0:2,1:1,2:0});
});

test('missing class names for numeric dataset IDs remain distinct from absent labels',async()=>{
  const f=setup(),{controller:c,get,calls}=f;
  f.explorer.job=analysis({points:[{index:0,name:'id-zero.png'},{index:1,name:'no-label.png'}],labels:{
    column:'label_id',loaded:0,values:{},original_values:{0:0},label_map:{},model_labels:{0:'cat',1:'dog'},
    diagnostic:'unnamed_numeric_labels',message:'Dataset labels contain numeric IDs without class names.'}});
  c.ready(f.explorer.job);
  assert.match(get('discovery-label-source').textContent,/numeric IDs without class names/i);
  get('discovery-label-view').events.click();const rows=get('discovery-label-preview-rows').children;
  assert.equal(rows[0].children[1].textContent,'Dataset');assert.equal(rows[0].children[2].textContent,0);
  assert.match(rows[0].children[3].textContent,/class names missing/i);
  assert.match(rows[1].children[3].textContent,/Missing label/);
  assert.equal(get('discovery-start').disabled,true);await c.start();assert.equal(calls.length,0);
});

test('label mismatches and absent labels have independent counts and manual fixes clear model advice',()=>{
  const f=setup(),{controller:c,get}=f;
  f.explorer.job=analysis({points:Array.from({length:3},(_,index)=>({index,name:`image-${index}.png`})),labels:{
    column:'label',loaded:2,values:{0:'bird',1:'cat'},label_map:{1:2},model_labels:modelLabels}});
  c.ready(f.explorer.job);
  assert.match(get('discovery-label-summary').textContent,/1 model mismatch.*1 need review/i);
  typeLabels(get,'{"0":3,"1":2,"2":2}');
  assert.equal(get('discovery-start').disabled,false);
  assert.doesNotMatch(get('discovery-label-summary').textContent,/mismatch|review/);
  assert.doesNotMatch(get('discovery-label-source').textContent,/choose.*model|unmatched/i);
});

test('mapped dataset categories are ready without relabelling and show one label with its source',async()=>{
  const f=setup(),{controller:c,get,calls,responses}=f;
  f.explorer.job=analysis({points:[{index:0,name:'cat.png'},{index:1,name:'dog.png'}],labels:{
    column:'labels',loaded:2,values:{0:'cat',1:'dog'},label_map:{0:'cat',1:'dog'},
    model_labels:{281:'tabby',282:'tiger cat',207:'golden retriever',208:'Labrador retriever'},
    groups:{cat:[281,282],dog:[207,208]}}});
  c.ready(f.explorer.job);
  assert.equal(get('discovery-label-summary').textContent,'2 / 2 labels ready');
  assert.equal(get('discovery-start').disabled,false);
  get('discovery-label-view').events.click();
  let rows=get('discovery-label-preview-rows').children;
  assert.equal(rows[0].children[1].textContent,'Dataset');
  assert.equal(rows[0].children[2].textContent,'cat');
  assert.equal(rows[0].children[3].textContent,'Ready · 2 model classes');
  responses.push(job({status:'completed'}));await c.start();
  assert.deepEqual(JSON.parse(JSON.stringify(calls[0].body.label_map)),{0:'cat',1:'dog'});
  get('discovery-label-edit').events.click();
  const input=get('discovery-label-preview-rows').children[0].children[2].children[0];
  input.value='#207';input.events.input();
  assert.equal(get('discovery-label-preview-rows').children[0].children[1].textContent,'Manual');
  get('discovery-label-clear').events.click();get('discovery-label-view').events.click();
  rows=get('discovery-label-preview-rows').children;
  assert.equal(rows[0].children[1].textContent,'Dataset');
  assert.equal(rows[0].children[2].textContent,'cat');
});

test('dataset targets load automatically and missing inputs block discovery until reviewed',async()=>{
  const f=setup(),{controller:c,get,responses,calls}=f;labelInputs(f,8,'images','test/model',{0:2,1:3});
  assert.match(get('discovery-label-summary').textContent,/2.*8.*6.*review/i);
  assert.equal(get('discovery-start').disabled,true);
  await c.start();assert.equal(calls.length,0);
  assert.equal(get('discovery-label-editor').hidden,true);
  assert.equal(get('discovery-label-preview').hidden,true);
  get('discovery-label-view').events.click();
  assert.equal(get('discovery-label-preview').hidden,false);
  assert.equal(get('discovery-label-preview-rows').children.length,5);
  assert.match(content(get('discovery-label-preview-rows')),/cat.*Ready.*dog.*Ready.*Needs review/);
  get('discovery-label-next').events.click();
  assert.equal(get('discovery-label-preview-rows').children.length,3);
  assert.match(content(get('discovery-label-preview-rows')),/image-5.png/);
  assert.equal(get('discovery-label-next').disabled,true);
  assert.equal(get('discovery-inspector').hidden,true);
  typeLabels(get,JSON.stringify(Object.fromEntries(Array.from({length:8},(_,i)=>[i,2]))));
  assert.equal(get('discovery-start').disabled,false);
  responses.push(job({status:'completed'}));await c.start();
  assert.equal(Object.keys(calls[0].body.label_map).length,8);
});

test('JSON uploads fill gaps and manual row edits share the full JSON and current preview',async()=>{
  const f=setup(),{controller:c,get,responses,calls}=f;labelInputs(f,3,'images','test/model',{0:2});
  await uploadLabels(get,'\uFEFF{"1":"dog"}','reviewed.json');
  assert.deepEqual(JSON.parse(get('discovery-label-map').value),{0:2,1:'dog'});
  assert.match(get('discovery-label-summary').textContent,/2.*3.*1.*review/i);
  assert.match(get('discovery-label-source').textContent,/reviewed.json/);
  get('discovery-label-edit').events.click();
  const row=get('discovery-label-preview-rows').children[2],input=row.children[2].children[0];
  input.value='cat';input.events.change();
  assert.equal(get('discovery-label-preview-rows').children[2],row,'saving a label keeps inputs mounted so Tab and clicks can move focus');
  assert.deepEqual(JSON.parse(get('discovery-label-map').value),{0:2,1:'dog',2:'cat'});
  assert.equal(get('discovery-start').disabled,false);
  get('discovery-label-view').events.click();
  assert.equal(get('discovery-label-editor').hidden,true);
  assert.match(content(get('discovery-label-preview-rows')),/cat.*Ready/);
  responses.push(job({status:'completed'}));await c.start();
  assert.deepEqual(JSON.parse(JSON.stringify(calls[0].body.label_map)),{0:2,1:'dog',2:'cat'});
  get('discovery-label-clear').events.click();
  assert.deepEqual(JSON.parse(get('discovery-label-map').value),{0:2});
  assert.equal(get('discovery-start').disabled,true);
});

test('unknown and ambiguous class names stay unresolved rather than becoming prediction targets',async()=>{
  const f=setup(),{controller:c,get,calls}=f;labelInputs(f,2);
  typeLabels(get,'{"0":"cat","1":"unknown"}');
  assert.match(get('discovery-label-summary').textContent,/1.*2.*1.*review/i);
  assert.equal(get('discovery-start').disabled,true);
  get('discovery-label-view').events.click();
  assert.match(content(get('discovery-label-preview-rows')),/does not match/i);
  await c.start();assert.equal(calls.length,0);
  typeLabels(get,'{"0":2,"1":3}');assert.equal(get('discovery-start').disabled,false);
});

test('numeric class names remain names and explicit hash-prefixed class IDs remain IDs',()=>{
  const f=setup(),{controller:c,get}=f;
  f.explorer.job=analysis({labels:{label_map:{},values:{},model_labels:{0:'1',1:'0'}}});c.ready(f.explorer.job);
  get('discovery-label-edit').events.click();
  let input=get('discovery-label-preview-rows').children[0].children[2].children[0];
  input.value='0';input.events.change();
  assert.deepEqual(JSON.parse(get('discovery-label-map').value),{0:'0'});
  assert.equal(c.labelResolved[0],1,'numeric class name 0 means model ID 1');
  input=get('discovery-label-preview-rows').children[0].children[2].children[0];
  input.value='#0';input.events.change();
  assert.deepEqual(JSON.parse(get('discovery-label-map').value),{0:0});
  assert.equal(c.labelResolved[0],0);
});

test('invalid JSON files preserve labels and cannot start discovery',async()=>{
  const f=setup(),{controller:c,get,calls}=f;labelInputs(f,1,'images','test/model',{0:2});
  const original=get('discovery-label-map').value;
  for(const text of ['{','[]','null','{"03":"cat"}','{"1":0}','{"0":true}','{"0":1.2}','{"0":-1}','{"0":" "}','{"0":{}}']) {
    await uploadLabels(get,text);
    assert.equal(get('discovery-label-map').value,original);
    assert.equal(get('discovery-label-error').hidden,false);
    assert.equal(get('discovery-start').disabled,true);
    await c.start();assert.equal(calls.length,0);
  }
  get('discovery-label-file').files=[{name:'huge.json',size:3*1024*1024,text:async()=>{throw Error('Oversized files must not be read');}}];
  await get('discovery-label-file').events.change();assert.match(get('discovery-label-error').textContent,/2 MB/);
  typeLabels(get,'{"0":false}');await c.start();assert.equal(calls.length,0);
  typeLabels(get,'{"0":0}');assert.equal(get('discovery-label-error').hidden,true);assert.equal(get('discovery-start').disabled,false);
});

test('late uploads cannot overwrite editing, restoring dataset labels, or changing analysis',async()=>{
  for(const change of ['typing','row','reset','analysis']) {
    const f=setup(),{controller:c,get,calls}=f;labelInputs(f,3);
    if(change==='row')get('discovery-label-edit').events.click();
    let finish;get('discovery-label-file').files=[{name:'slow.json',size:11,text:()=>new Promise(resolve=>{finish=resolve;})}];
    get('discovery-label-file').value='C:/fakepath/slow.json';
    const uploading=get('discovery-label-file').events.change();
    assert.equal(get('discovery-label-file').value,'');
    assert.equal(get('discovery-start').disabled,true);await c.start();assert.equal(calls.length,0);
    if(change==='typing')typeLabels(get,'{"1":2}');
    else if(change==='row'){const input=get('discovery-label-preview-rows').children[1].children[2].children[0];input.value='cat';input.events.input();}
    else if(change==='reset')get('discovery-label-clear').events.click();
    else {c.invalidate();labelInputs(f,3,'new-images');}
    finish('{"0":"cat"}');await uploading;
    assert.deepEqual(JSON.parse(get('discovery-label-map').value||'{}'),change==='typing'?{1:2}:change==='row'?{1:'cat'}:{});
    assert.equal(get('discovery-start').disabled,true,'remaining labels still need review');
  }
});

test('edits persist across reanalysis and new dataset or model imports fresh labels',async()=>{
  const f=setup(),{controller:c,get}=f;labelInputs(f,3);await uploadLabels(get,'{"0":"cat"}');
  c.invalidate();labelInputs(f,3);assert.equal(JSON.parse(get('discovery-label-map').value)[0],'cat');
  c.invalidate();labelInputs(f,3,'new-images','test/model',{1:3});
  assert.deepEqual(JSON.parse(get('discovery-label-map').value),{1:3});
  typeLabels(get,'{"1":1}');c.invalidate();labelInputs(f,3,'new-images','test/other');
  assert.deepEqual(JSON.parse(get('discovery-label-map').value||'{}'),{});
});

test('reports and other discovery artifacts open separately from the workspace',async()=>{
  const f=setup(),{controller:c,get,responses}=f;f.ready();
  responses.push(job({status:'completed',artifacts:{report:'/discoveries/id/artifacts/report.json',csv:'/discoveries/id/artifacts/pairs.csv'}}));await c.start();
  assert.equal(get('discovery-artifacts').children.length,2);
  for(const link of get('discovery-artifacts').children){assert.equal(link.target,'_blank');assert.match(link.rel,/noopener/);}
});

test('Inspect orders measured severity highest first and preserves selection as scores arrive in every workspace',()=>{
  for(const task of ['classification','summarization','generation']) {
    const f=setup(task),{controller:c,get}=f;
    c.job={records:[{input_index:0,action_id:4,severity:.2,visits:1},
      {input_index:0,action_id:9,severity:.9,visits:1},{input_index:1,action_id:2,severity:0,visits:1},
      {input_index:0,action_id:3,severity:.9,visits:1},{input_index:0,action_id:8,severity:null,visits:1}]};
    c.selected='0:4';c.inspectionOpen=true;c.inspectionOptions();
    assert.deepEqual(get('discovery-pair').children.map(option=>option.value),['','0:3','0:9','0:4','1:2','0:8']);
    assert.equal(get('discovery-pair').value,'0:4');
    assert.equal(c.job.records[0].action_id,4,'ordering the UI must not reorder the discovery records');
    c.job.records.push({input_index:1,action_id:7,severity:1,visits:1});c.inspectionOptions();
    assert.equal(get('discovery-pair').children[1].value,'1:7');
    assert.equal(get('discovery-pair').value,'0:4','new severe pairs must not steal the current selection');
    c.selected='2:99';c.inspectionOptions();
    assert.equal(get('discovery-pair').children.at(-1).value,'2:99','a selected unevaluated action stays last');
    assert.equal(get('discovery-pair').value,'2:99');
  }
});

test('outputs require an explicit Inspect choice in every workspace and stay hidden after clearing',async()=>{
  for(const task of ['classification','summarization','generation']) {
    const f=setup(task),{controller:c,get,responses}=f;f.ready();
    get('discovery-reference').value='Reviewed reference';
    responses.push(job({task}));await c.start();
    assert.equal(get('discovery-inspector').hidden,true,`${task}: starting discovery must not open outputs`);
    assert.equal(get('discovery-pair').value,'');
    assert.equal(get('discovery-pair').children[0].value,'','Inspect has an empty placeholder');
    c.inspectPair(0,4);
    assert.equal(get('discovery-inspector').hidden,true,'a graph click previews inputs without opening model outputs');
    assert.equal(get('discovery-pair').value,'');
    c.receive('discoveries',job({task,episode:2}),c.version);
    assert.equal(get('discovery-inspector').hidden,true,'polling cannot open the inspector');
    inspect(get);
    assert.equal(get('discovery-inspector').hidden,false);
    assert.match(content(get('discovery-output')),/dog/);
    c.receive('discoveries',job({task,episode:3}),c.version);
    assert.equal(get('discovery-inspector').hidden,false,'polling preserves explicit inspection');
    assert.equal(get('discovery-pair').value,'0:4');
    inspect(get,'');
    c.receive('discoveries',job({task,status:'completed'}),c.version);
    assert.equal(get('discovery-inspector').hidden,true,'clearing Inspect remains effective after completion');
    inspect(get);c.inspectPair(0,null);
    assert.equal(get('discovery-inspector').hidden,true,'selecting another graph input closes the old measured details');
    responses.push(job({task}));await c.start();
    assert.equal(get('discovery-inspector').hidden,true,'new searches do not inherit inspection');
  }
});

test('discovery defaults to the Analyze device while preserving edits within that analysis',async()=>{
  const f=setup(),{controller:c,get,explorer,responses,calls}=f;
  explorer.job={id:'emb1',device:'cuda'};c.ready(explorer.job);
  responses.push(job({status:'completed'}));await c.start();
  assert.equal(calls[0].body.device,'cuda','a GPU analysis should not trigger a duplicate CPU model load by default');
  get('discovery-device').value='cpu';c.ready(explorer.job);
  assert.equal(get('discovery-device').value,'cpu','re-rendering the same analysis preserves an explicit device choice');
  explorer.job={id:'emb2',device:'cuda'};c.ready(explorer.job);
  assert.equal(get('discovery-device').value,'cuda','a new analysis starts with its own device');
});

test('Analyze gates discovery; labels are explicit and progress/inspection never expand embeddings',async()=>{
  const f=setup(),{controller:c,get,responses,calls}=f;
  assert.equal(get('find-failure').disabled,true);await c.start();assert.equal(calls.length,0);
  f.ready();assert.equal(get('find-failure').disabled,false);
  get('discovery-label-map').value='{"0":"cat"}';responses.push(job());await c.start();
  assert.deepEqual(JSON.parse(JSON.stringify(calls[0].body)),{embedding_id:'emb1',episodes:10,seed:42,device:'cpu',label_map:{0:'cat'}});
  assert.match(get('discovery-metrics').textContent,/Loss 0.120.*epsilon 0.960.*1 measured pairs.*2 selected-model calls/);
  assert.equal(get('discovery-cancel').hidden,false);assert.equal(get('adapt-panel').hidden,true);
  assert.equal(f.focuses.length,0,'polling never takes over selection');
  inspect(get);assert.deepEqual(f.focuses[0],[0,4]);assert.equal(calls.length,1,'selection only highlights an existing point');
  assert.match(get('discovery-output').children[0].textContent,/dog/);assert.match(get('discovery-baseline').children[0].textContent,/cat/);
  c.inspectPair(0,99);inspect(get,'0:99');assert.match(get('discovery-components').textContent,/not been evaluated/);assert.equal(get('adapt').disabled,true);
  c.receive('discoveries',job({episode:2}),c.version);assert.match(get('discovery-components').textContent,/not been evaluated/,'new polls preserve the selected unvisited action');
});

test('errors permit retry/cancel; terminal cancellation preserves measured results',async()=>{
  const f=setup(),{controller:c,responses,get}=f;f.ready();responses.push(job());await c.start();
  responses.push(new Error('offline'));await c.poll('discoveries',c.version);
  assert.equal(get('discovery-retry').hidden,false);assert.equal(get('discovery-cancel').hidden,false);assert.equal(c.active,true);
  responses.push(job({status:'cancelled',phase:'Cancelled'}));await c.cancel('discoveries');
  assert.equal(c.active,false);assert.equal(get('discovery-cancel').hidden,true);assert.equal(get('adapt').disabled,false);assert.equal(get('adapt-panel').hidden,true);
  assert.equal(c.job.records.length,1);assert.equal(get('discovery-retry').hidden,true);
});

test('global action selection trains across labeled inputs and exposes checkpoint and report',async()=>{
  const f=setup(),{controller:c,responses,calls,get}=f;f.ready();responses.push(job({status:'completed'}));await c.start();
  await c.startTraining();assert.equal(calls.length,1,'no implicit training from discovery completion');
  assert.equal(get('adapt').disabled,false);assert.equal(get('adapt-panel').hidden,true);
  await openAdapt(f);assert.equal(calls.length,2,'opening Adapt only summarizes discovery');
  c.inspectPair(0,4);assert.equal(c.adaptation.graphAction,4);
  get('adapt-toggle-action').events.click();assert.equal(c.adaptation.selected.size,0);assert.equal(get('finetune-start').disabled,true);
  get('adapt-toggle-action').events.click();assert.equal(c.adaptation.selected.size,1);
  responses.push({id:'train1',status:'running',completed:1,total:2,loss:.3});await c.adaptation.start();
  assert.equal(calls[2].url,'/api/finetunes');assert.deepEqual(JSON.parse(JSON.stringify(calls[2].body.action_ids)),[4]);
  assert.equal(calls[2].body.include_originals,true);assert.equal(calls[2].body.examples,undefined);
  assert.equal(get('finetune-cancel').hidden,false);assert.equal(get('finetune-checkpoint').hidden,true);
  responses.push({id:'train1',status:'completed',completed:2,total:2,loss:.2,checkpoint:'/models/reviewed',artifacts:{report:'/discoveries/train1/artifacts/report.json'}});await c.poll('finetunes',c.version);
  assert.equal(get('finetune-checkpoint').hidden,false);get('finetune-use').events.click();assert.deepEqual(f.checkpoints,['/models/reviewed']);
  assert.equal(get('finetune-report').href,'/discoveries/train1/artifacts/report.json');assert.equal(get('finetune-report').target,'_blank');
});

test('Adapt requests automatic GPU selection independently of the discovery device and reports the resolved device',async()=>{
  for(const task of ['classification','summarization','generation']) {
    const f=setup(task),{controller:c,get,responses,calls}=f;f.ready();c.receive('discoveries',job({task,status:'completed'}),c.version);await openAdapt(f);
    responses.push({id:'train1',status:'running',device:'cuda',device_name:'Test CUDA GPU',completed:1,total:2});
    await c.startTraining({action_ids:[4]});
    assert.equal(calls.at(-1).body.device,'auto','Adapt does not inherit CPU from discovery');
    assert.match(get('finetune-status').textContent,/CUDA.*Test CUDA GPU/);
    c.receive('finetunes',{id:'train1',status:'completed',device:'cuda',device_name:'Test CUDA GPU'},c.version);
    get('finetune-device').value='cpu';responses.push({id:'train2',status:'running',device:'cpu',device_name:'CPU'});
    await c.startTraining({action_ids:[4]});assert.equal(calls.at(-1).body.device,'cpu');
    assert.match(get('finetune-status').textContent,/CPU/);
  }
});

test('Find failure and Adapt switch panels without losing results or FT selections in every workspace',async()=>{
  for(const task of ['classification','summarization','generation']) {
    const f=setup(task),{controller:c,get}=f;f.ready();c.receive('discoveries',job({task,status:'completed'}),c.version);
    get('find-failure').events.click();assert.equal(get('discovery-panel').hidden,false);
    await openAdapt(f);
    assert.equal(get('discovery-panel').hidden,true);assert.equal(get('adapt-panel').hidden,false);
    assert.equal(get('workspace-layout').attrs['data-stage'],'adapt');
    c.adaptation.toggle(4);const calls=f.calls.length,original=c.job;
    get('find-failure').events.click();
    assert.equal(get('discovery-panel').hidden,false);assert.equal(get('adapt-panel').hidden,true);
    assert.equal(get('adapt-graph-action').hidden,true);assert.equal(get('find-failure').attrs['aria-pressed'],'true');
    await c.adaptation.show();
    assert.equal(get('discovery-panel').hidden,true);assert.equal(get('adapt-panel').hidden,false);
    assert.equal(c.adaptation.selected.size,0);assert.equal(c.job,original);assert.equal(f.calls.length,calls,'switching panels does not repeat discovery or reset selection');
  }
});

test('panel navigation during training preserves the job and keeps cancellation reachable',async()=>{
  const f=setup(),{controller:c,get,responses}=f;f.ready();c.receive('discoveries',job({status:'completed'}),c.version);await openAdapt(f);
  responses.push({id:'train1',status:'running',completed:0,total:2});await c.adaptation.start();
  assert.equal(get('find-failure').disabled,false);assert.equal(get('adapt').disabled,false);
  get('find-failure').events.click();assert.equal(get('adapt-panel').hidden,true);assert.equal(get('discovery-panel').hidden,false);
  c.receive('finetunes',{id:'train1',status:'running',completed:1,total:2},c.version);
  assert.equal(get('adapt-panel').hidden,true,'status polling respects the active panel');
  await c.adaptation.show();assert.equal(get('discovery-panel').hidden,true);assert.equal(get('finetune-cancel').hidden,false);
  assert.equal(c.training.id,'train1');assert.equal(get('adapt-graph-toggle').disabled,true);
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
    if(task==='summarization'){assert.equal(calls[0].body.summary_metric,'reference');assert.equal(calls[0].body.reference_summary,'Reviewed reference');}
    else {assert.equal(calls[0].body.generation_steps,12);assert.equal(calls[0].body.generation_size,256);}
    assert.equal(calls[0].body.label_map,undefined);
  }
});

test('training cancellation remains available after form invalidation and failures can retry',async()=>{
  const f=setup(),{controller:c,responses,get}=f;f.ready();responses.push(job({status:'completed'}));await c.start();await openAdapt(f);
  responses.push({id:'train1',status:'running'});await c.adaptation.start();c.invalidate();
  assert.equal(get('finetune-cancel').hidden,false);assert.equal(get('adapt-panel').hidden,false,'keep the running training cancel control reachable');
  responses.push(new Error('temporary failure'));await c.cancel('finetunes');assert.equal(get('finetune-cancel').hidden,false);
  responses.push({id:'train1',status:'cancelled'});await c.cancel('finetunes');assert.equal(c.active,false);
});

function content(element){return [element.textContent,...element.children.map(content)].join(' ');}

test('measured outcomes use readable predictions, summaries, generated images and named scores',async()=>{
  const f=setup(),{controller:c,get,responses}=f;f.ready();responses.push(job({status:'completed'}));await c.start();
  inspect(get);
  assert.match(content(get('discovery-output')),/Prediction: dog \(class 3\).*Confidence: 30.0%/);
  assert.match(content(get('discovery-components')),/Severity 80.0%.*Target confidence drop 0.600/);
  assert.ok(!content(get('discovery-output')).includes('"class_id"'),'outputs do not expose API JSON');
  c.output('discovery-output',{summary:'First summary paragraph.\n\nSecond paragraph.',quality:.8});
  assert.equal(get('discovery-output').children[0].textContent,'First summary paragraph.');
  assert.equal(get('discovery-output').children[1].textContent,'Second paragraph.');
  c.output('discovery-output',{image:'/discoveries/job/artifacts/image.png',clip_similarity:.37,prompt:'A scientist',seed:42});
  assert.equal(get('discovery-output').children[0].src,'/discoveries/job/artifacts/image.png');
  assert.match(content(get('discovery-output')),/CLIP scene alignment: 0.370.*Prompt: A scientist.*Noise seed: 42/);
});

test('policy inspector exposes mean, variation and tested-input coverage for measured and unvisited actions',async()=>{
  const f=setup(),{controller:c,get,responses}=f;f.ready();
  const statistics={method:'softmax_q_across_original_inputs',temperature:1,input_count:4,policy_updates:10,std_ddof:1,actions:[
    {action_id:4,mean:.6,std:.02,evaluated_inputs:3},{action_id:9,mean:.2,std:.08,evaluated_inputs:0}
  ]};
  responses.push(job({status:'completed',policy_statistics:statistics}));await c.start();
  inspect(get);
  assert.match(content(get('discovery-components')),/Mean DQN action preference 0\.600/);
  assert.match(content(get('discovery-components')),/Preference variation.*sample SD.*0\.0200/);
  assert.match(content(get('discovery-components')),/Original inputs.*4/);assert.match(content(get('discovery-components')),/Tested inputs.*3 \/ 4/);
  c.inspectPair(0,9);inspect(get,'0:9');assert.match(content(get('discovery-components')),/not been evaluated/);
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
  assert.equal(get('discovery-inspector').hidden,true);assert.equal(get('discovery-records').hidden,true);assert.equal(get('adapt-panel').hidden,true);assert.equal(overlays.at(-1),null);
  const before=calls.length;await c.adaptation.show();await c.startTraining({action_ids:[4],include_originals:true});assert.equal(calls.length,before,'mismatched discovery cannot provide training examples');
  c.training={status:'completed',checkpoint:'/old/model'};get('finetune-use').events.click();assert.equal(f.checkpoints.length,0);
  assert.equal(get('discovery-start').disabled,false,'the new analysis can start its own discovery');
  responses.push(job({id:'new',embedding_id:'emb2',status:'completed'}));await c.start();assert.equal(c.attached,true);assert.equal(c.job.id,'new');assert.equal(c.adaptation.selected.size,0);
});

test('a delayed discovery GET cannot resurrect discovery or cancel fine-tuning polling',async()=>{
  for(const delayedStatus of ['running','completed']) {
    const f=setup(),{controller:c,get,responses,calls,timers}=f;f.ready();responses.push(job());await c.start();
    const discoveryVersion=c.version;let release;
    responses.push(()=>new Promise(resolve=>{release=resolve;}));const delayed=c.poll('discoveries',discoveryVersion);
    responses.push(job({status:'cancelled'}));await c.cancel('discoveries');
    await openAdapt(f);
    const beforeValidation=c.version,beforeCalls=calls.length;get('finetune-lr').value='0.2';await c.adaptation.start();
    assert.equal(c.version,beforeValidation,'invalid settings do not advance the polling generation');assert.equal(calls.length,beforeCalls,'out-of-range learning rates are rejected before POST');
    get('finetune-lr').value='0.001';
    responses.push({id:'train1',status:'running',completed:0,total:2});await c.adaptation.start();
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

test('late status replies cannot undo cancellation in either workflow',async()=>{
  for(const kind of ['discoveries','finetunes']) {
    const f=setup(),{controller:c,get,responses}=f;f.ready();
    c.receive('discoveries',job({status:kind==='discoveries'?'running':'completed'}),c.version);
    if(kind==='finetunes'){await openAdapt(f);c.receive(kind,{id:'train1',status:'running'},c.version);}
    let finish;responses.push(()=>new Promise(resolve=>{finish=resolve;}));const polling=c.poll(kind,c.version);
    const cancelled=kind==='discoveries'?job({status:'cancelled'}):{id:'train1',status:'cancelled'};
    responses.push(cancelled);await c.cancel(kind);
    finish(kind==='discoveries'?job():{id:'train1',status:'running'});await polling;
    assert.equal(c.active,false,'a stale running response cannot lock controls after cancellation');
    assert.equal(get(kind==='discoveries'?'discovery-cancel':'finetune-cancel').hidden,true);
    assert.equal((kind==='discoveries'?c.job:c.training).status,'cancelled');
  }
});

test('retry clicks share an in-flight status request and cannot regress a terminal result',async()=>{
  const f=setup(),{controller:c,get,responses,calls}=f;f.ready();c.receive('discoveries',job(),c.version);
  responses.push(new Error('offline'));await c.poll('discoveries',c.version);
  let finish;responses.push(()=>new Promise(resolve=>{finish=resolve;}));const polling=get('discovery-retry').events.click();
  const before=calls.length;responses.push(job());const retry=get('discovery-retry').events.click();
  assert.equal(calls.length,before,'multiple retry clicks must not start competing GETs');
  finish(job({status:'completed'}));await polling;await retry;
  assert.equal(c.job.status,'completed');assert.equal(c.active,false);assert.equal(get('discovery-retry').hidden,true);
});

test('measured Inspect navigation updates the action targeted by the Adapt graph button',async()=>{
  const f=setup(),{controller:c,get}=f;f.ready();c.receive('discoveries',job({status:'completed',records:[record(0,4),record(0,9)]}),c.version);
  await openAdapt(f);c.adaptation.close();c.inspectPair(0,4);
  inspect(get,'0:9');
  assert.equal(c.adaptation.graphAction,9,'Inspect must not leave the add/remove button targeting the previously selected action');
  assert.equal(get('discovery-inspector').hidden,false,'updating the graph action does not close explicitly requested outputs');
});

test('pending training remains reachable across panel navigation and changed inputs',async()=>{
  const f=setup(),{controller:c,get,responses}=f;f.ready();c.receive('discoveries',job({status:'completed'}),c.version);await openAdapt(f);
  let finish;responses.push(()=>new Promise(resolve=>{finish=resolve;}));const starting=c.adaptation.start();
  get('find-failure').events.click();assert.equal(get('adapt').disabled,false,'the training panel stays reachable while its POST is pending');
  await c.adaptation.show();assert.equal(get('adapt-panel').hidden,false);
  assert.equal(get('finetune-cancel').hidden,true,'cancellation needs the new job identifier');
  c.invalidate();assert.equal(get('adapt-panel').hidden,false,'input changes retain pending training status');
  finish({id:'train1',status:'running'});await starting;
  assert.equal(get('finetune-cancel').hidden,false);assert.equal(get('adapt-panel').hidden,false);
});

test('changing the analyzed article clears its old reference while reanalysis preserves it',async()=>{
  const f=setup('summarization'),{controller:c,get,explorer,calls}=f;
  const ready=(id,text)=>{explorer.job={id,task:'summarization',points:[{index:0,name:'Article',text}]};c.ready(explorer.job);};
  ready('first','The first trial succeeded.');get('discovery-reference').value='The first trial succeeded.';
  c.invalidate();ready('reanalyzed','The first trial succeeded.');
  assert.equal(get('discovery-reference').value,'The first trial succeeded.','changing model or actions retains the same article reference');
  c.invalidate();ready('different','A different trial failed.');
  assert.equal(get('discovery-reference').value,'','a reference for the previous article must not silently score a different article');
  await c.start();assert.equal(calls.length,0,'a different article needs its own reviewed reference before discovery');
});

test('automatic and manual effective batch sizes reach training in every workspace',async()=>{
  for(const task of ['classification','summarization','generation'])for(const size of ['auto','32']) {
    const f=setup(task),{controller:c,get,responses,calls}=f;f.ready();c.receive('discoveries',job({task,status:'completed'}),c.version);await openAdapt(f);
    get('finetune-batch-size').value=size;
    responses.push({id:'train1',status:'running',completed:0,total:null});
    await c.startTraining({action_ids:[4],include_originals:true});
    assert.equal(calls.at(-1).body.batch_size,size==='auto'?'auto':32);
    assert.doesNotMatch(get('finetune-status').textContent,/0\/0/,'automatic update totals are pending until resolved');
  }
});

test('fine-tuning submits independent seed, optimizer and cleanup options in every workspace',async()=>{
  for(const task of ['classification','summarization','generation']) {
    const f=setup(task),{controller:c,get,responses,calls}=f;f.ready();c.receive('discoveries',job({task,status:'completed'}),c.version);await openAdapt(f);
    get('discovery-seed').value='999';get('finetune-seed').value='7';get('finetune-weight-decay').value='0.2';get('finetune-max-grad-norm').value='0';get('finetune-cleanup').checked=false;
    responses.push({id:'train1',status:'running'});await c.startTraining({action_ids:[4]});
    const config=calls.at(-1).body;
    assert.equal(config.seed,7,'training does not reuse the discovery seed');assert.equal(config.weight_decay,.2);
    assert.equal(config.max_grad_norm,0,'zero disables clipping');assert.equal(config.cleanup_after_finetune,false);
    assert.equal(get('adapt-settings').disabled,true,'the whole training fieldset locks during the run');
  }
});

test('training options retain each workspace edits without changing reference text',()=>{
  const f=setup(),{controller:c,get,explorer}=f;
  get('finetune-epochs').value='3';get('finetune-device').value='cuda';get('finetune-batch-size').value='16';get('finetune-seed').value='0';get('finetune-weight-decay').value='0.25';get('finetune-max-grad-norm').value='0';get('finetune-cleanup').checked=false;
  explorer.task='summarization';c.configure();
  assert.equal(get('finetune-epochs').value,'1');assert.equal(get('finetune-device').value,'auto');assert.equal(get('finetune-batch-size').value,'auto');
  assert.equal(get('finetune-seed').value,'42');assert.equal(get('finetune-weight-decay').value,'0.01');assert.equal(get('finetune-max-grad-norm').value,'1');assert.equal(get('finetune-cleanup').checked,true);
  get('discovery-reference').value='Reviewed summary';get('finetune-seed').value='23';
  explorer.task='generation';c.configure();assert.equal(get('finetune-seed').value,'42');
  explorer.task='classification';c.configure();
  assert.equal(get('finetune-epochs').value,'3');assert.equal(get('finetune-device').value,'cuda');assert.equal(get('finetune-batch-size').value,'16');
  assert.equal(get('finetune-seed').value,'0');assert.equal(get('finetune-weight-decay').value,'0.25');assert.equal(get('finetune-max-grad-norm').value,'0');assert.equal(get('finetune-cleanup').checked,false);
  explorer.task='summarization';c.configure();assert.equal(get('finetune-seed').value,'23');assert.equal(get('discovery-reference').value,'Reviewed summary');
});

test('training progress shows epochs, full exposure and actual batch sizes after backoff',()=>{
  const f=setup(),{controller:c,get}=f;f.ready();c.receive('discoveries',job({status:'completed'}),c.version);
  c.receive('finetunes',{id:'train1',status:'running',phase:'Training',completed:3,total:500,epoch:1,epochs:2,processed_examples:96,total_examples:16000,examples_per_epoch:8000,effective_batch_size:32,micro_batch_size:4,oom_retries:2,device:'cuda',device_name:'GPU'},c.version);
  assert.match(get('finetune-status').textContent,/1\/2.*96\/16,000.*examples.*3\/500.*updates/);
  assert.match(get('finetune-batch-info').textContent,/32.*4.*2.*memory/i);
  c.receive('finetunes',{id:'train1',status:'completed',phase:'completed'},c.version);
  assert.equal(get('finetune-status').textContent,'completed');
  c.receive('finetunes',{id:'train2',status:'queued',total:null},c.version);
  assert.equal(get('finetune-batch-info').textContent,'');
});

test('cleanup outcomes appear only for completed training and clear for the next job',()=>{
  const f=setup(),{controller:c,get}=f;f.ready();c.receive('discoveries',job({status:'completed'}),c.version);
  for(const status of ['completed','skipped','error','disabled']) {
    c.receive('finetunes',{id:'train1',status:'completed',cleanup:{status,message:`Cleanup ${status}: saved outputs retained.`}},c.version);
    assert.equal(get('finetune-cleanup-status').hidden,false);
    assert.equal(get('finetune-cleanup-status').textContent,`Cleanup ${status}: saved outputs retained.`);
  }
  for(const status of ['queued','running','cancelled','error']) {
    c.receive('finetunes',{id:'train2',status,cleanup:{status:'completed',message:'Old cleanup result'}},c.version);
    assert.equal(get('finetune-cleanup-status').hidden,true,'failed or active training never implies successful cleanup');
    assert.equal(get('finetune-cleanup-status').textContent,'');
  }
});

test('new discovery and training attempts clear stale training progress',async()=>{
  const f=setup(),{controller:c,get,responses}=f;f.ready();c.receive('discoveries',job({status:'completed'}),c.version);await openAdapt(f);
  c.receive('finetunes',{id:'train1',status:'completed',completed:2,total:2,checkpoint:'/old/model'},c.version);
  assert.equal(get('finetune-progress').hidden,false);
  let finish;responses.push(()=>new Promise(resolve=>{finish=resolve;}));const starting=c.adaptation.start();
  assert.equal(get('finetune-progress').hidden,true,'starting another training run must not show the old full progress bar');
  finish({id:'train2',status:'completed',completed:4,total:4,checkpoint:'/new/model'});await starting;
  c.adaptation.close();responses.push(job({id:'next-discovery',status:'completed'}));await c.start();
  assert.equal(get('finetune-progress').hidden,true,'a new discovery has no training progress yet');
  assert.equal(get('finetune-path').value,'','the old checkpoint is cleared when a new discovery replaces its training data');
});

test('failed POSTs stop displaying a starting status and allow a retry',async()=>{
  const f=setup(),{controller:c,get,responses}=f;f.ready();
  responses.push(new Error('discovery request failed'));await c.start();
  assert.doesNotMatch(get('discovery-status').textContent,/Starting/);assert.equal(get('discovery-start').disabled,false);
  responses.push(job({status:'completed'}));await c.start();await openAdapt(f);
  responses.push(new Error('training request failed'));await c.adaptation.start();
  assert.doesNotMatch(get('finetune-status').textContent,/Starting/);assert.equal(get('finetune-start').disabled,false);
  assert.match(get('adapt-error').textContent,/training request failed/);
});

(async()=>{let failures=0;for(const {name,fn} of tests){try{await fn();console.log('ok - '+name);}catch(error){failures++;console.error('not ok - '+name+'\n'+error.stack);}}if(failures)process.exitCode=1;else console.log(`${tests.length} discovery controller tests passed`);})();
