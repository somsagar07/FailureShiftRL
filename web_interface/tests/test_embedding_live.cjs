// Exercise live discovery with the real graph and both controllers. Only the
// browser, timers and model API are external boundaries.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');

class Element {
  constructor(){this.value='';this.children=[];this.events={};this.hidden=false;this.disabled=false;this.textContent='';this.style={};}
  addEventListener(type,fn){this.events[type]=fn;}
  replaceChildren(...children){this.children=children;}
  append(...children){this.children.push(...children);}
  setAttribute(name,value){this[name]=value;}
  getBoundingClientRect(){return {width:260,height:76};}
}
class Canvas extends Element {
  constructor(){
    super();this.box={left:20,top:30,width:800,height:600};this.captured=new Set();
    this.context={setTransform(){},setLineDash(){},clearRect(){},beginPath(){},arc(){},moveTo(){},lineTo(){},fill(){},stroke(){}};
  }
  getContext(){return this.context;}
  getBoundingClientRect(){return {...this.box};}
  setPointerCapture(id){this.captured.add(id);}
  hasPointerCapture(id){return this.captured.has(id);}
  releasePointerCapture(id){this.captured.delete(id);}
  focus(){}
  fire(type,overrides={}){this.events[type]?.({type,button:0,pointerId:1,clientX:0,clientY:0,preventDefault(){},...overrides});}
}
function setup(task='classification') {
  const canvas=new Canvas(),elements=new Map([['embedding-canvas',canvas]]),get=id=>{
    if(!elements.has(id))elements.set(id,new Element());return elements.get(id);
  };
  for(const [id,value] of Object.entries({'finetune-epochs':'1','finetune-lr':'0.001','finetune-device':'auto','finetune-batch-size':'auto','finetune-seed':'42','finetune-weight-decay':'0.01','finetune-max-grad-norm':'1'}))get(id).value=value;
  get('finetune-cleanup').checked=true;
  const timers=new Map();let nextTimer=0;
  const context={console,document:{getElementById:get,createElement:()=>new Element()},
    setTimeout:fn=>{timers.set(++nextTimer,fn);return nextTimer;},clearTimeout:id=>timers.delete(id),
    addEventListener(){},devicePixelRatio:1,matchMedia:()=>({matches:true})};
  context.window=context;vm.createContext(context);
  for(const file of ['embedding-view.js','embedding-controller.js','adaptation-controller.js','discovery-controller.js'])vm.runInContext(fs.readFileSync(path.join(__dirname,'../static',file),'utf8'),context);
  const Explorer=vm.runInContext('EmbeddingExplorer',context),Discovery=vm.runInContext('DiscoveryController',context);
  const calls=[],pending=[],inspected=[],errors=[];
  const api=(url,body)=>{calls.push({url,body});return new Promise(resolve=>pending.push(resolve));};
  const explorer=new Explorer({api,onInspect:(...args)=>inspected.push(args),onStatus(){},onError:error=>errors.push(error),onBusy(){},onSelectInput(){}});
  const source=task==='classification'?{image:'/original.png'}:{text:'Original input'};
  const points=[{index:0,name:'Original input',position:[-1,0,0],...source}];
  if(task==='classification')points.push({index:1,name:'Second input',position:[1,0,0],image:'/second.png'});
  explorer.preview={id:'preview',action_count:3,config:{task,model:{name:'test/model'}}};
  explorer.job={id:'embedding',task,status:'completed',points,labels:{label_map:{0:0,1:0},model_labels:{0:'fixture'}}};explorer.view.setData(points);
  const discovery=new Discovery({api,explorer,onCheckpoint(){}});discovery.ready(explorer.job);
  explorer.onInspectPair=(index,action)=>discovery.inspectPair(index,action);
  explorer.isDiscoveryActive=()=>Boolean(discovery.active);
  const node=(id,position=[0,1,0],index=0)=>({id,action:task==='classification'?{rotation:5}:task==='summarization'?{name:'to_lowercase'}:{profession:'artist'},position,similarity:.93,
    ...(task==='classification'?{image:`/input-${index}/action-${id}.png`}:{text:`Changed input ${id}`,unchanged:false})});
  const expansion=(index=0,nodes=[node(0)])=>({id:`expand-${index}`,input_index:index,status:'completed',nodes,original:points[index]});
  const report=(records=[],extra={})=>({id:'discovery',embedding_id:'embedding',task,status:'running',episode:1,episodes:10,records,baseline:[],...extra});
  const measured=(id,position,index=0)=>({input_index:index,action_id:id,visits:1,severity:.5,reward:.5,node:node(id,position,index)});
  const receive=job=>discovery.receive('discoveries',job,discovery.version);
  return {canvas,get,explorer,discovery,calls,pending,inspected,errors,timers,node,expansion,report,measured,receive};
}
function camera(view){return JSON.parse(JSON.stringify({center:view.center,scale:view.scale,distance:view.distance,fitDistance:view.fitDistance,yaw:view.yaw,pitch:view.pitch,pan:view.pan}));}
function moveCamera(view){view.orbit(.23,-.17);view.zoom(1.35);view.pan={x:31,y:-22};view.render();}
function location(canvas,point){return {clientX:canvas.box.left+point.x,clientY:canvas.box.top+point.y};}
async function flush(){for(let i=0;i<8;i++)await Promise.resolve();}
const tests=[];function test(name,fn){tests.push({name,fn});}

test('Adapt graph clicks offer nearby FT controls that follow the camera in every workspace',async()=>{
  for(const task of ['classification','summarization','generation']) {
    const f=setup(task),{explorer:e,discovery:d,get,canvas}=f;
    const focusing=e.focus(0);await flush();f.pending.shift()(f.expansion(0,[f.node(4),f.node(9,[.2,-.2,0])]));await focusing;
    f.receive(f.report([f.measured(4)],{status:'completed'}));
    const opening=d.adaptation.show();await flush();
    f.pending.shift()({discovery_id:'discovery',task,input_count:e.job.points.length,suggested_action_ids:[],
      actions:[4,9].map(action_id=>({action_id,action:f.node(action_id).action,evaluated_inputs:0})),defaults:{include_originals:false}});
    await opening;
    const node=e.view.projected.find(point=>point.kind==='action'&&point.data.id===9);
    const before=camera(e.view);
    canvas.fire('pointerdown',location(canvas,node));canvas.fire('pointerup',location(canvas,node));
    const menu=get('adapt-graph-action');assert.equal(menu.hidden,false);
    const left=Number.parseFloat(menu.style.left),top=Number.parseFloat(menu.style.top);
    const gap=Math.hypot(Math.max(left-node.x,0,node.x-left-260),Math.max(top-node.y,0,node.y-top-76));
    assert.ok(gap<50,'the FT control opens beside the action, not at the bottom of the graph');
    get('adapt-graph-toggle').events.click();assert.equal(d.adaptation.selected.has(9),true);assert.equal(e.view.adaptationActions.has(9),true);
    get('adapt-graph-toggle').events.click();assert.equal(d.adaptation.selected.has(9),false);assert.deepEqual(camera(e.view),before);
    const previous={...menu.style};e.view.orbit(.15,.2);
    assert.notDeepEqual(menu.style,previous,'the menu follows the selected node when the graph rotates');
    e.view.pan={x:1000,y:1000};e.view.render();
    assert.equal(menu.style.top,'','an offscreen action uses the visible dock instead of an offscreen menu');
    d.showDiscovery();assert.equal(menu.hidden,true,'Find failure does not show FT controls');
  }
});

test('predicted and measured modes remain separate and inspectable in every workspace',async()=>{
  for(const task of ['classification','summarization','generation']) {
    const f=setup(task),{explorer:e,get}=f;
    const focusing=e.focus(0);await flush();
    f.pending.shift()(f.expansion(0,[f.node(4),f.node(9,[.5,.5,0])]));await focusing;
    const landscape={policy_updates:5,input_mean_reward:e.job.points.map(()=>.5),action_ids:[4,9]};
    const report=f.report([f.measured(4)],{status:'completed',predicted_landscape:landscape});
    const before=camera(e.view);f.receive(report);await flush();
    assert.equal(f.calls.at(-1).url,'/api/discoveries/discovery/predictions?input=0','only the focused input needs its action values');
    f.pending.shift()({discovery_id:'discovery',policy_updates:5,input_index:0,actions:[{action_id:4,q_value:.3,preference:.4},{action_id:9,q_value:.7,preference:.6}]});await flush();
    const untested=e.view.projected.find(node=>node.kind==='action'&&node.data.id===9);
    assert.equal(e.view.nodeColor(untested),'rgb(163,83,63)');
    assert.match(get('landscape-scale-label').textContent,/Predicted reward/);
    assert.equal(get('discovery-inspector').hidden,true);
    get('landscape-mode').value='measured';get('landscape-mode').events.change();
    assert.equal(e.view.nodeColor(untested),'#9aa1a9');
    assert.match(get('landscape-scale-label').textContent,/Measured severity/);
    get('landscape-mode').value='predicted';get('landscape-mode').events.change();
    assert.equal(e.view.nodeColor(untested),'rgb(163,83,63)');
    e.inspect(e.view.nodes[1]);get('discovery-pair').value='0:9';get('discovery-pair').events.change();
    const text=element=>[element.textContent,...element.children.map(text)].join(' ');
    assert.match(text(get('discovery-components')),/Predicted reward.*0\.700/);
    assert.match(text(get('discovery-output')),/Not evaluated/,'estimates must not fabricate model outputs');
    f.receive({...report,episode:40});await flush();
    assert.equal(f.calls.length,2,'polling the same checkpoint reuses its predictions');
    assert.deepEqual(camera(e.view),before);
    assert.equal(e.view.selectedAction,9);
  }
});

test('prediction responses cannot cross checkpoint, input, or analysis boundaries',async()=>{
  const f=setup(),{explorer:e}=f;
  const focusing=e.focus(0);await flush();f.pending.shift()(f.expansion());await focusing;
  const report=updates=>f.report([],{predicted_landscape:{policy_updates:updates,input_mean_reward:[.2,.8],action_ids:[0]}});
  f.receive(report(1));await flush();
  assert.equal(f.calls.at(-1).url,'/api/discoveries/discovery/predictions?input=0');
  const old=f.pending.shift();f.receive(report(2));await flush();const current=f.pending.shift();
  const prediction=(updates,input=0)=>({discovery_id:'discovery',policy_updates:updates,input_index:input,actions:[{action_id:0,q_value:.8,preference:1}]});
  old(prediction(1));await flush();assert.equal(e.view.predictions,null,'old checkpoint estimates are discarded');
  e.focusMeasured(1,0);await flush();const other=f.pending.shift();
  current(prediction(2));await flush();assert.equal(e.view.predictions,null,'another input cannot overwrite the current selection');
  other(prediction(2,1));await flush();assert.equal(e.view.predictions.input_index,1);
  f.receive(report(3));await flush();const pending=f.pending.shift();e.reset();pending(prediction(3,1));await flush();
  assert.equal(e.view.predictions,null,'old requests cannot restore a cleared analysis');
});

test('a successful prediction retry clears its own error and preserves unrelated errors',async()=>{
  const f=setup(),{explorer:e,get}=f;
  const focusing=e.focus(0);await flush();f.pending.shift()(f.expansion());await focusing;
  get('form-error').textContent='An unrelated input error';
  const report=f.report([],{predicted_landscape:{policy_updates:2,input_mean_reward:[.2,.8],action_ids:[0]}});
  e.api=async()=>{throw new Error('temporary connection failure');};
  f.receive(report);await flush();
  assert.equal(get('landscape-error').hidden,false);
  assert.match(get('landscape-error').textContent,/temporary connection failure/);
  e.api=async()=>({discovery_id:'discovery',policy_updates:2,input_index:0,actions:[{action_id:0,q_value:.2,preference:1}]});
  f.receive(report);await flush();
  assert.equal(e.view.predictions.policy_updates,2);
  assert.equal(get('landscape-error').hidden,true);
  assert.equal(get('form-error').textContent,'An unrelated input error');
});

test('new and cached action expansions preserve the camera in every domain',async()=>{
  for(const task of ['classification','summarization','generation']) {
    const f=setup(task),{explorer:e,pending}=f;
    const focusing=e.focus(0);await flush();moveCamera(e.view);
    const before=camera(e.view),original=e.view.projected.find(point=>point.kind==='sample');
    pending.shift()(f.expansion(0,[f.node(0,[100,20,0])]));await focusing;
    assert.deepEqual(camera(e.view),before,`${task}: arriving actions cannot refit or recenter the graph`);
    const after=e.view.projected.find(point=>point.kind==='sample');assert.equal(after.x,original.x);assert.equal(after.y,original.y);
    e.inspect(e.expansion.nodes[0]);await e.focus(0);
    assert.deepEqual(camera(e.view),before,`${task}: returning to the original cannot reset zoom, pan or orbit`);
    assert.equal(e.view.selectedAction,null);assert.equal(f.calls.length,1,'cached input selection needs no model request');
    f.get('embedding-fit').events.click();assert.notDeepEqual(camera(e.view),before,'the explicit Fit control still includes new actions');
  }
});

test('parent clicks request full expansions during discovery and polling preserves selection in every domain',async()=>{
  for(const task of ['classification','summarization','generation']) {
    const f=setup(task),{explorer:e,get,canvas,discovery:d}=f;
    f.receive(f.report());const focusing=e.focus(0);await flush();
    assert.equal(f.calls.length,1,`${task}: an active discovery must still expand a clicked parent`);
    assert.equal(f.calls[0].url,'/api/embeddings/embedding/expand');
    f.pending.shift()({...f.expansion(),status:'running',nodes:[],completed:0,total:3});await focusing;
    moveCamera(e.view);const before=camera(e.view);
    f.receive(f.report([f.measured(0,[0,.5,0])]));
    assert.equal(e.view.nodes.length,1,`${task}: completed measurements appear in the focused graph`);
    assert.equal(get('embedding-actions').hidden,false);
    assert.equal(e.expansionCache.has(0),false,'measured subsets are not cached as complete action expansions');
    const plotted=e.view.projected.find(point=>point.kind==='action'),down=location(canvas,plotted);
    canvas.fire('pointerdown',down);
    // A status response supplies new JSON objects while the pointer is held.
    f.receive(f.report([f.measured(0,[0,.5,0]),f.measured(1,[0,-.5,0])],{episode:2,current:{input_index:0,action_id:1}}));
    canvas.fire('pointerup',down);
    assert.equal(e.view.selectedAction,0,`${task}: polling during a click cannot lose the selected action`);
    assert.equal(d.selected,'0:0');assert.equal(f.inspected.at(-1)[1].id,0);
    assert.equal(get('discovery-inspector').hidden,true,'graph clicks keep measured outputs closed');
    assert.equal(get('embedding-action').value,'0');
    f.receive(f.report([f.measured(0,[0,.5,0]),f.measured(1,[0,-.5,0]),f.measured(2,[.5,0,0])],{episode:3,current:{input_index:0,action_id:2}}));
    assert.equal(e.view.selectedAction,0,'new RL actions do not take over graph selection');
    assert.equal(d.selected,'0:0');assert.equal(get('embedding-action').value,'0');
    assert.deepEqual(camera(e.view),before,'measurement updates preserve all camera state');
    assert.equal(f.calls.length,1,'progress and action selection do not start additional expansions');
    get('embedding-action').value='1';get('embedding-action').events.change();
    assert.equal(e.view.selectedAction,1);assert.equal(d.selected,'0:1');assert.equal(f.inspected.at(-1)[1].id,1);
    await e.receiveExpansion(f.expansion(0,[f.node(0),f.node(1),f.node(2)]),e.run,e.selection);
    await e.focus(0);assert.equal(e.view.selectedAction,null);assert.equal(e.view.nodes.length,3);
    f.receive(f.report([f.measured(0,[0,.5,0])],{episode:4}));
    assert.equal(d.selected,null,'polling preserves an explicitly selected original');
    assert.equal(e.view.selectedAction,null);assert.equal(get('embedding-action').value,'original');
  }
});

test('measured-pair navigation reuses nodes across inputs and remains usable for reports without node data',async()=>{
  const f=setup(),{explorer:e,discovery:d,get}=f;
  const focusing=e.focus(0);await flush();f.pending.shift()(f.expansion());await focusing;
  f.receive(f.report([f.measured(1,[1,.5,0],1),{input_index:0,action_id:0,visits:1,severity:.2}]));
  moveCamera(e.view);const before=camera(e.view);
  get('discovery-pair').value='1:1';get('discovery-pair').events.change();
  assert.equal(e.view.focusIndex,1);assert.equal(e.view.selectedAction,1);assert.equal(f.inspected.at(-1)[1].image,'/input-1/action-1.png');
  assert.equal(get('embedding-actions').hidden,false,'measured selection keeps graph actions accessible');
  f.receive(f.report([f.measured(1,[1,.5,0],1),{input_index:0,action_id:0,visits:1,severity:.2},f.measured(2,[.5,-.5,0],0)],{episode:2,current:{input_index:0,action_id:2}}));
  assert.equal(e.view.focusIndex,1,'measurements of other inputs cannot steal focus');assert.equal(e.view.selectedAction,1);assert.equal(d.selected,'1:1');
  get('discovery-pair').value='0:0';get('discovery-pair').events.change();
  assert.equal(e.view.selectedAction,0,'older records without coordinates can still use an existing full expansion');
  assert.equal(f.inspected.at(-1)[1].image,'/input-0/action-0.png');
  assert.deepEqual(camera(e.view),before);assert.equal(f.calls.length,1);
});

test('missing positions stay unplotted and completed partial caches can still expand fully',async()=>{
  const f=setup(),{explorer:e}=f;
  const invalid=f.measured(1,[NaN,0,0]);
  f.receive(f.report([{input_index:0,action_id:0,visits:1,severity:.7},invalid]));const focusing=e.focus(0);await flush();
  assert.equal(e.view.nodes.length,0,'severity records cannot invent graph coordinates');
  f.receive(f.report([f.measured(0,[0,.5,0])],{status:'completed'}));
  assert.equal(e.view.nodes.length,1);
  assert.equal(f.calls.length,1,'a measured subset does not suppress full expansion during search');
  assert.equal(e.view.nodes.length,1,'measured actions stay available while the full expansion runs');
  e.inspect(e.view.nodes[0]);moveCamera(e.view);const before=camera(e.view);
  f.pending.shift()(f.expansion(0,[f.node(0,[0,.5,0]),f.node(1,[0,-.5,0])]));await focusing;
  assert.equal(e.view.nodes.length,2);assert.equal(e.view.selectedAction,0,'completion preserves an action selected during computation');
  assert.deepEqual(camera(e.view),before);
  e.clearFocus();f.receive(f.report([f.measured(2,[.5,0,0])],{status:'completed'}));
  assert.equal(e.view.focusIndex,null,'polls cannot reopen a cleared graph selection');
  assert.equal(e.view.nodes.length,0);
});

test('starting discovery preserves a graph action selected while its request is pending',async()=>{
  const f=setup(),{explorer:e,discovery:d,get}=f;
  const focusing=e.focus(0);await flush();f.pending.shift()(f.expansion(0,[f.node(0),f.node(1,[0,-.5,0])]));await focusing;
  for(const [id,value] of Object.entries({'discovery-episodes':'10','discovery-seed':'42','discovery-device':'cpu'}))get(id).value=value;
  const starting=d.start();e.inspect(e.expansion.nodes[1]);
  f.pending.shift()(f.report([f.measured(0)]));await starting;
  assert.equal(e.view.selectedAction,1);assert.equal(d.selected,'0:1','start acknowledgement preserves the current graph action');
  assert.equal(get('discovery-inspector').hidden,true,'discovery start does not open the graph selection’s outputs');
  f.receive(f.report([f.measured(0)],{episode:2,current:{input_index:0,action_id:0}}));
  assert.equal(e.view.selectedAction,1);assert.equal(d.selected,'0:1');
});

test('partial expansion batches appear during search and survive live measurements without camera changes',async()=>{
  const f=setup(),{explorer:e,get}=f;
  f.receive(f.report());const focusing=e.focus(0);await flush();
  assert.equal(f.calls.length,1);
  f.pending.shift()({...f.expansion(),status:'running',nodes:[],completed:0,total:3});await focusing;
  moveCamera(e.view);const before=camera(e.view);
  const progress={...f.expansion(0,[f.node(0),f.node(1)]),status:'running',completed:2,total:3,phase:'embedding transformations'};
  await e.receiveExpansion(progress,e.run,e.selection);
  assert.equal(e.view.nodes.length,2,'computed action batches are immediately selectable');
  assert.equal(e.expansionCache.has(0),false,'partial batches cannot masquerade as completed expansions');
  e.inspect(e.view.nodes[1]);
  f.receive(f.report([f.measured(0)],{episode:2}));
  assert.equal(e.view.nodes.length,2,'discovery polling cannot replace expanded nodes with a smaller measured subset');
  assert.equal(e.view.selectedAction,1);assert.equal(get('embedding-action').value,'1');
  assert.equal(get('discovery-inspector').hidden,true);
  assert.deepEqual(camera(e.view),before);
  await e.receiveExpansion(f.expansion(0,[f.node(0),f.node(1),f.node(2)]),e.run,e.selection);
  assert.equal(e.view.nodes.length,3);assert.equal(e.view.selectedAction,1);
  assert.equal(e.expansionCache.has(0),true);
  assert.deepEqual(camera(e.view),before);
});

test('returning to a partially expanded parent keeps its cached nodes and selection while batches restart',async()=>{
  const f=setup(),{explorer:e,get}=f;
  f.receive(f.report());const first=e.focus(0);await flush();
  assert.equal(f.calls.length,1);
  f.pending.shift()({...f.expansion(0,[f.node(0),f.node(1),f.node(2)]),status:'running',completed:3,total:4});await first;
  const other=e.focus(1);await flush();f.pending.shift()(f.expansion(1,[f.node(0,[1,.5,0],1)]));await other;
  const returning=e.focus(0);await flush();
  assert.equal(e.view.nodes.length,3,'returning shows the previously computed nodes immediately');
  e.inspect(e.view.nodes[2]);moveCamera(e.view);const before=camera(e.view);
  f.pending.shift()({...f.expansion(0,[f.node(0)]),status:'running',completed:1,total:4});await returning;
  assert.equal(e.view.nodes.length,3,'a restarted prefix cannot discard valid cached coordinates');
  assert.equal(e.view.selectedAction,2);assert.equal(get('embedding-action').value,'2');
  f.receive(f.report([f.measured(0)],{episode:2}));
  assert.equal(e.view.nodes.length,3);assert.equal(e.view.selectedAction,2);
  assert.deepEqual(camera(e.view),before);
  await e.receiveExpansion(f.expansion(0,[f.node(0),f.node(1),f.node(2),f.node(3)]),e.run,e.selection);
  assert.equal(e.view.nodes.length,4);assert.equal(e.view.selectedAction,2);
});

test('explicit action navigation waits for an uncached action through queued expansion batches in every workspace',async()=>{
  for(const task of ['classification','summarization','generation']) {
    const f=setup(task),{explorer:e}=f;
    f.receive(f.report([f.measured(4,[0,.5,0])],{status:'completed'}));
    moveCamera(e.view);const before=camera(e.view);
    assert.equal(typeof e.focusAction,'function','Adapt has an explicit action-loading navigation method');
    const focusing=e.focusAction(0,9);await flush();
    assert.equal(f.calls.length,1,`${task}: missing action 9 requests real action embeddings`);
    assert.equal(f.calls[0].url,'/api/embeddings/embedding/expand');
    assert.equal(f.calls[0].body.input_index,0);
    assert.deepEqual(Array.from(e.view.nodes,node=>node.id),[4],'cached action 4 stays visible without inventing action 9 coordinates');
    assert.equal(f.inspected.at(-1)[0],null,'the pending action must not be labelled Original input');
    f.pending.shift()({...f.expansion(0,[]),status:'queued',completed:0,total:10});await focusing;
    const poll=async job=>{
      const timer=e.expansionTimer,callback=f.timers.get(timer);f.timers.delete(timer);
      assert.equal(typeof callback,'function','queued expansions keep polling');
      const polling=callback();await flush();
      assert.equal(f.calls.at(-1).url,'/api/embeddings/embedding/expansions/expand-0');
      f.pending.shift()(job);await polling;
    };
    await poll({...f.expansion(0,[f.node(4,[0,.5,0])]),status:'running',completed:1,total:10});
    assert.equal(f.inspected.at(-1)[0],null,'an unrelated batch cannot masquerade as the requested action');
    const requested=f.node(9,[.25,-.5,.1]);
    await poll({...f.expansion(0,[f.node(4,[0,.5,0]),requested]),status:'running',completed:2,total:10});
    assert.equal(e.view.selectedAction,9,'the requested action is inspected as soon as its real node arrives');
    assert.equal(f.inspected.at(-1)[1],requested);
    assert.equal(f.get('embedding-action').value,'9');
    assert.deepEqual(e.view.nodes.find(node=>node.id===9).position,[.25,-.5,.1]);
    assert.deepEqual(camera(e.view),before);
    await poll(f.expansion(0,[f.node(4,[0,.5,0]),requested]));
    assert.equal(e.view.selectedAction,9,'completion preserves the requested action');
  }
});

test('explicit action navigation reuses measured, partial, and complete nodes without new requests',async()=>{
  for(const task of ['classification','summarization','generation']) {
    const f=setup(task),{explorer:e}=f;
    const measured=f.measured(4,[0,.5,0]);
    f.receive(f.report([measured],{status:'completed'}));
    moveCamera(e.view);const before=camera(e.view);
    await e.focusAction(0,4);
    assert.equal(f.calls.length,0,`${task}: a measured node is immediately inspectable`);
    assert.equal(f.inspected.at(-1)[1],measured.node);
    const requested=f.node(9,[.5,-.5,0]);
    for(const status of ['running','completed']) {
      await e.receiveExpansion({...f.expansion(0,[measured.node,requested]),status,completed:2,total:2},e.run,e.selection);
      e.clearFocus();
      await e.focusAction(0,9);
      assert.equal(f.calls.length,0,`${task}: ${status} cached nodes need no inference`);
      assert.equal(f.inspected.at(-1)[1],requested);
      assert.equal(e.view.selectedAction,9);
      assert.deepEqual(camera(e.view),before);
    }
  }
});

test('later graph, original, and measured selections supersede a pending explicit action inspection',async()=>{
  for(const selection of ['graph action','original input','measured action']) {
    const f=setup(),{explorer:e,canvas}=f;
    f.receive(f.report([f.measured(4,[0,.5,0])],{status:'completed'}));
    const focusing=e.focusAction(0,9);await flush();
    f.pending.shift()({...f.expansion(0,[f.node(4,[0,.5,0])]),status:'running',completed:1,total:10});await focusing;
    if(selection==='graph action') {
      const at=location(canvas,e.view.projected.find(node=>node.kind==='action'&&node.data.id===4));
      canvas.fire('pointerdown',at);canvas.fire('pointerup',at);
    } else if(selection==='original input') {
      f.get('embedding-action').value='original';f.get('embedding-action').events.change();
    } else e.focusMeasured(0,4);
    const expected=selection==='original input'?null:4,inspections=f.inspected.length;
    assert.equal(e.view.selectedAction,expected);
    moveCamera(e.view);const before=camera(e.view);
    await e.receiveExpansion(f.expansion(0,[f.node(4,[0,.5,0]),f.node(9,[.5,-.5,0])]),e.run,e.selection);
    assert.equal(e.view.selectedAction,expected,`${selection} wins over the late action 9 response`);
    assert.equal(f.inspected.length,inspections,'expansion completion cannot replace the newer inspector');
    assert.deepEqual(camera(e.view),before);
  }
});

test('viewing a cached action supersedes pending inspection without stopping the active expansion',async()=>{
  const f=setup(),{explorer:e}=f;
  f.receive(f.report([f.measured(4,[0,.5,0])],{status:'completed'}));
  const focusing=e.focusAction(0,9);await flush();
  f.pending.shift()({...f.expansion(0,[f.node(4,[0,.5,0])]),status:'running',completed:1,total:10});await focusing;
  await e.focusAction(0,4);
  assert.equal(f.calls.length,1,'viewing the cached action does not restart inference');
  const poll=f.timers.get(e.expansionTimer);
  assert.equal(typeof poll,'function','the active expansion keeps its poll when only the inspected action changes');
  const polling=poll();await flush();
  f.pending.shift()(f.expansion(0,[f.node(4,[0,.5,0]),f.node(9,[.5,-.5,0])]));await polling;
  assert.equal(e.view.nodes.length,2,'the graph still receives the rest of its expansion');
  assert.equal(e.view.selectedAction,4);
  assert.equal(f.inspected.at(-1)[1].id,4);
});

test('clearing or resetting during an action poll prevents stale graph and inspector takeover',async()=>{
  for(const cancel of ['clearFocus','reset']) {
    const f=setup(),{explorer:e}=f;
    f.receive(f.report([f.measured(4,[0,.5,0])],{status:'completed'}));
    const focusing=e.focusAction(0,9);await flush();
    f.pending.shift()({...f.expansion(0,[]),status:'queued',completed:0,total:10});await focusing;
    const polling=f.timers.get(e.expansionTimer)();await flush();
    e[cancel]();const inspections=f.inspected.length;
    f.pending.shift()(f.expansion(0,[f.node(9,[.5,-.5,0])]));await polling;
    assert.equal(e.view.focusIndex,null,`${cancel} prevents an old response from refocusing the graph`);
    assert.equal(e.view.nodes.length,0);
    assert.equal(f.inspected.length,inspections,`${cancel} prevents an old response from reopening the inspector`);
  }
});

test('a later original-input selection supersedes an explicit action request before its acknowledgement',async()=>{
  const f=setup(),{explorer:e}=f;
  f.receive(f.report([f.measured(4,[0,.5,0])],{status:'completed'}));
  const earlier=e.focusAction(0,9);await flush();
  const latest=e.focus(1);await flush();
  assert.equal(f.calls.length,1,'the new input waits for the earlier POST acknowledgement');
  f.pending.shift()(f.expansion(0,[f.node(9,[.5,-.5,0])]));await earlier;await flush();
  assert.equal(f.calls.length,2);
  assert.equal(f.calls[1].body.input_index,1);
  f.pending.shift()(f.expansion(1,[f.node(4,[1,.5,0],1)]));await latest;
  assert.equal(e.view.focusIndex,1);
  assert.equal(e.view.selectedAction,null);
  assert.equal(f.inspected.at(-1)[0].index,1);
  assert.equal(f.inspected.at(-1)[1],null);
  assert.ok(!f.inspected.some(([,node])=>node?.id===9),'the superseded action is never inspected');
});

test('an unavailable requested action reports an error instead of inspecting the original',async()=>{
  const f=setup(),{explorer:e}=f;
  const focusing=e.focusAction(0,9);await flush();
  f.pending.shift()(f.expansion(0,[f.node(4,[0,.5,0])]));await focusing;
  assert.match(f.errors.at(-1),/action.*9.*not available/i);
  assert.equal(f.inspected.at(-1)[0],null);
  const inspections=f.inspected.length;
  f.receive(f.report([f.measured(9,[.5,-.5,0])],{status:'completed'}));
  assert.equal(f.inspected.length,inspections,'later reports cannot resurrect a failed navigation request');
});

test('measured records without a transformed node never preview the original as that action',async()=>{
  for(const task of ['classification','summarization','generation']) {
    const f=setup(task),{explorer:e,get}=f;
    const focusing=e.focus(0);await flush();f.pending.shift()(f.expansion(0,[f.node(4)]));await focusing;
    e.inspect(e.expansion.nodes[0]);
    f.receive(f.report([{input_index:0,action_id:9,visits:1,severity:.7,output:{summary:'Measured output'}}],{status:'completed'}));
    get('discovery-pair').value='0:9';get('discovery-pair').events.change();
    assert.equal(e.action,9);assert.equal(f.inspected.at(-1)[0],null,`${task}: unavailable transformed input must not look like an unchanged original`);
    assert.equal(get('discovery-inspector').hidden,false,'the measured model output remains inspectable');
    assert.equal(f.calls.length,1,'reading historical measurements does not start inference');
  }
});

(async()=>{let failures=0;for(const {name,fn} of tests){try{await fn();console.log('ok - '+name);}catch(error){failures++;console.error('not ok - '+name+'\n'+error.stack);}}if(failures)process.exitCode=1;else console.log(`${tests.length} live embedding integration tests passed`);})();
