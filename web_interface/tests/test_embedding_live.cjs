// Exercise live discovery with the real graph and both controllers. Only the
// browser, timers and model API are external boundaries.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');

class Element {
  constructor(){this.value='';this.children=[];this.events={};this.hidden=false;this.disabled=false;this.textContent='';this.style={};}
  addEventListener(type,fn){this.events[type]=fn;}
  replaceChildren(...children){this.children=children;}
  append(...children){this.children.push(...children);}
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
  const timers=new Map();let nextTimer=0;
  const context={console,document:{getElementById:get,createElement:()=>new Element()},
    setTimeout:fn=>{timers.set(++nextTimer,fn);return nextTimer;},clearTimeout:id=>timers.delete(id),
    addEventListener(){},devicePixelRatio:1,matchMedia:()=>({matches:true})};
  context.window=context;vm.createContext(context);
  for(const file of ['embedding-view.js','embedding-controller.js','discovery-controller.js'])vm.runInContext(fs.readFileSync(path.join(__dirname,'../static',file),'utf8'),context);
  const Explorer=vm.runInContext('EmbeddingExplorer',context),Discovery=vm.runInContext('DiscoveryController',context);
  const calls=[],pending=[],inspected=[];
  const api=(url,body)=>{calls.push({url,body});return new Promise(resolve=>pending.push(resolve));};
  const explorer=new Explorer({api,onInspect:(...args)=>inspected.push(args),onStatus(){},onError(){},onBusy(){},onSelectInput(){}});
  const source=task==='classification'?{image:'/original.png'}:{text:'Original input'};
  const points=[{index:0,name:'Original input',position:[-1,0,0],...source}];
  if(task==='classification')points.push({index:1,name:'Second input',position:[1,0,0],image:'/second.png'});
  explorer.preview={id:'preview',action_count:3,config:{task,model:{name:'test/model'}}};
  explorer.job={id:'embedding',task,status:'completed',points};explorer.view.setData(points);
  const discovery=new Discovery({api,explorer,onCheckpoint(){}});discovery.ready(explorer.job);
  explorer.onInspectPair=(index,action)=>discovery.inspectPair(index,action);
  explorer.isDiscoveryActive=()=>Boolean(discovery.active);
  const node=(id,position=[0,1,0],index=0)=>({id,action:task==='classification'?{rotation:5}:task==='summarization'?{name:'to_lowercase'}:{profession:'artist'},position,similarity:.93,
    ...(task==='classification'?{image:`/input-${index}/action-${id}.png`}:{text:`Changed input ${id}`,unchanged:false})});
  const expansion=(index=0,nodes=[node(0)])=>({id:`expand-${index}`,input_index:index,status:'completed',nodes,original:points[index]});
  const report=(records=[],extra={})=>({id:'discovery',embedding_id:'embedding',task,status:'running',episode:1,episodes:10,records,baseline:[],...extra});
  const measured=(id,position,index=0)=>({input_index:index,action_id:id,visits:1,severity:.5,reward:.5,node:node(id,position,index)});
  const receive=job=>discovery.receive('discoveries',job,discovery.version);
  return {canvas,get,explorer,discovery,calls,pending,inspected,node,expansion,report,measured,receive};
}
function camera(view){return JSON.parse(JSON.stringify({center:view.center,scale:view.scale,distance:view.distance,fitDistance:view.fitDistance,yaw:view.yaw,pitch:view.pitch,pan:view.pan}));}
function moveCamera(view){view.orbit(.23,-.17);view.zoom(1.35);view.pan={x:31,y:-22};view.render();}
function location(canvas,point){return {clientX:canvas.box.left+point.x,clientY:canvas.box.top+point.y};}
async function flush(){for(let i=0;i<8;i++)await Promise.resolve();}
const tests=[];function test(name,fn){tests.push({name,fn});}

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

test('live measured actions become selectable without inference and polling preserves selection in every domain',async()=>{
  for(const task of ['classification','summarization','generation']) {
    const f=setup(task),{explorer:e,get,canvas,discovery:d}=f;
    f.receive(f.report());await e.focus(0);
    assert.match(get('embedding-focus').textContent,/waiting for measured actions/i);
    assert.equal(f.calls.length,0,'an active discovery does not queue an expansion behind inference');
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
    assert.equal(get('embedding-action').value,'0');
    f.receive(f.report([f.measured(0,[0,.5,0]),f.measured(1,[0,-.5,0]),f.measured(2,[.5,0,0])],{episode:3,current:{input_index:0,action_id:2}}));
    assert.equal(e.view.selectedAction,0,'new RL actions do not take over graph selection');
    assert.equal(d.selected,'0:0');assert.equal(get('embedding-action').value,'0');
    assert.deepEqual(camera(e.view),before,'measurement updates preserve all camera state');
    assert.equal(f.calls.length,0,'selection and progress do not request model inference');
    get('embedding-action').value='1';get('embedding-action').events.change();
    assert.equal(e.view.selectedAction,1);assert.equal(d.selected,'0:1');assert.equal(f.inspected.at(-1)[1].id,1);
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
  f.receive(f.report([{input_index:0,action_id:0,visits:1,severity:.7},invalid]));await e.focus(0);
  assert.equal(e.view.nodes.length,0,'severity records cannot invent graph coordinates');
  f.receive(f.report([f.measured(0,[0,.5,0])],{status:'completed'}));
  assert.equal(e.view.nodes.length,1);
  const focusing=e.focus(0);await flush();assert.equal(f.calls.length,1,'a measured subset does not suppress a later full expansion');
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
  assert.match(get('discovery-components').textContent,/not been evaluated/);
  f.receive(f.report([f.measured(0)],{episode:2,current:{input_index:0,action_id:0}}));
  assert.equal(e.view.selectedAction,1);assert.equal(d.selected,'0:1');
});

(async()=>{let failures=0;for(const {name,fn} of tests){try{await fn();console.log('ok - '+name);}catch(error){failures++;console.error('not ok - '+name+'\n'+error.stack);}}if(failures)process.exitCode=1;else console.log(`${tests.length} live embedding integration tests passed`);})();
