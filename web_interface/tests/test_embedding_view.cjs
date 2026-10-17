// Run with Node.js: node tests/test_embedding_view.cjs
// Canvas and DOM are boundaries; exercise the actual renderer and event handlers.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Canvas {
  constructor() {
    this.box = {left: 35, top: 70, width: 800, height: 600};
    this.events = {};
    this.style = {};
    this.captured = new Set();
    this.focused = false;
    this.title = '';
    const ctx = this.context = {
      globalAlpha: 1, circles: [], strokes: [], labels: [],
      clearRect() {this.circles = []; this.strokes = []; this.labels = [];},
      setTransform(...values) {this.transform = values;},
      setLineDash(values) {this.dashes = values.slice();},
      beginPath() {this.path = [];},
      arc(x, y, radius) {this.path.push({x, y, radius});},
      moveTo(x, y) {this.path.push({x, y});},
      lineTo(x, y) {this.path.push({x, y});},
      fill() {this.circles.push(...this.path.map(p => ({...p, color: this.fillStyle, alpha: this.globalAlpha})));},
      stroke() {this.strokes.push({path: this.path.slice(), color: this.strokeStyle, alpha: this.globalAlpha, dashes: this.dashes?.slice()||[]});},
      fillText(text, x, y) {this.labels.push({text, x, y});},
    };
    this.getContext = kind => {assert.equal(kind, '2d'); return ctx;};
  }
  addEventListener(name, handler) {this.events[name] = handler;}
  getBoundingClientRect() {return {...this.box};}
  setPointerCapture(id) {if (this.captureFails) throw new Error('pointer no longer active'); this.captured.add(id);}
  hasPointerCapture(id) {return this.captured.has(id);}
  releasePointerCapture(id) {this.captured.delete(id);}
  focus() {this.focused = true;}
  fire(type, overrides = {}) {
    const event = {type, target: this, button: 0, pointerId: 1, clientX: this.box.left,
      clientY: this.box.top, preventDefault() {this.defaultPrevented = true;}, ...overrides};
    this.events[type]?.(event);
    return event;
  }
}

function setup() {
  const canvas = new Canvas(), events = {}, observed = [];
  const context = {console, clearTimeout(){}, setTimeout(){return 1;}, devicePixelRatio: 2,
    addEventListener(name, handler) {events[name] = handler;},
    ResizeObserver: class {constructor(callback) {this.callback = callback; observed.push(this);} observe() {}},
  };
  context.window = context;
  vm.createContext(context);
  const source = path.join(__dirname, '../static/embedding-view.js');
  if (fs.existsSync(source)) vm.runInContext(fs.readFileSync(source, 'utf8'), context);
  const Renderer = vm.runInContext('typeof EmbeddingView === "function" ? EmbeddingView : null', context);
  assert.equal(typeof Renderer, 'function', 'the local renderer exposes EmbeddingView');
  const selections = [], actions = [];
  let clears = 0;
  const view = new Renderer(canvas, {
    onSelectSample: node => selections.push(node),
    onSelectAction: node => actions.push(node),
    onClear: () => clears++,
  });
  return {view, canvas, context, events, observed, selections, actions, get clears() {return clears;}};
}

function point(index, position) {
  return Object.freeze({index, name: `image-${index}.png`, image: `/original/${index}`, position: Object.freeze(position)});
}
function action(id, position) {
  return Object.freeze({id, action: Object.freeze({rotation: 15, darkness: 0.2, saturation: 0.5}),
    image: `/changed/${id}`, position: Object.freeze(position), similarity: 0.87});
}
function circles(canvas) {return canvas.context.circles;}
function click(canvas, circle) {
  const location = {clientX: canvas.box.left + circle.x, clientY: canvas.box.top + circle.y};
  canvas.fire('pointerdown', location);
  canvas.fire('pointerup', location);
}
function finiteDrawing(canvas) {
  for (const entry of [...circles(canvas), ...canvas.context.strokes.flatMap(line => line.path), ...canvas.context.labels]) {
    assert.ok(Number.isFinite(entry.x) && Number.isFinite(entry.y), 'every projected coordinate stays finite');
    if ('radius' in entry) assert.ok(Number.isFinite(entry.radius) && entry.radius > 0);
  }
}
const tests = [];
function test(name, run) {tests.push({name, run});}

test('perspective makes the near sample larger and draws it last', () => {
  const {view, canvas} = setup();
  view.setData([point(0, [0, 0, 1]), point(1, [0, 0, -1])]);
  const [far, near] = circles(canvas);
  assert.equal(circles(canvas).length, 2);
  assert.ok(near.radius > far.radius, 'depth must affect perspective scale');
  assert.equal(near.x, 400);
  assert.equal(near.y, 300);
  assert.equal(far.x, near.x);
  assert.equal(far.y, near.y);
});

test('focus dims other originals without removing them and clearFocus restores them', () => {
  const {view, canvas, selections, actions} = setup();
  view.setData([point(5, [-1, 0, 0]), point(9, [1, 0, 0])]);
  const before = circles(canvas).map(p => ({...p}));
  view.focus(9);
  assert.equal(circles(canvas).length, 2);
  assert.ok(circles(canvas)[0].alpha < circles(canvas)[1].alpha);
  assert.ok(circles(canvas)[1].radius > before[1].radius);
  view.clearFocus();
  assert.deepEqual(circles(canvas), before);
  assert.equal(selections.length + actions.length, 0, 'mutators must not emit user callbacks');
});

test('expansion connects measured action positions and fit includes distant actions', () => {
  const {view, canvas} = setup();
  const samples = [point(3, [-1, 0, 0]), point(8, [1, 0, 0])];
  const transformed = [action(27, [100, 12, 0])];
  view.setData(samples);
  view.focus(3);
  view.setExpansion(3, transformed);
  view.fit();
  const rendered = circles(canvas);
  assert.equal(rendered.length, 3);
  const actionCircle = rendered.find(p => p.color !== rendered[0].color);
  assert.ok(actionCircle, 'actions have a distinct color from originals');
  for (const p of rendered) assert.ok(p.x > 0 && p.x < 800 && p.y > 0 && p.y < 600, 'Fit must include every measured point');
  assert.ok(canvas.context.strokes.some(line => line.path.length === 2 && line.path.some(p => p.x === actionCircle.x && p.y === actionCircle.y)), 'a line connects the action to its source');
  assert.deepEqual(transformed[0].position, [100, 12, 0], 'rendering never moves measured points');
  finiteDrawing(canvas);
});

test('an old expansion cannot replace the newly focused sample', () => {
  const {view, canvas} = setup();
  view.setData([point(0, [-1, 0, 0]), point(1, [1, 0, 0])]);
  view.focus(0);
  view.setExpansion(0, [action(6, [0, 1, 0])]);
  assert.equal(circles(canvas).length, 3);
  view.focus(1);
  assert.equal(circles(canvas).length, 2);
  view.setExpansion(0, [action(7, [0, -1, 0])]);
  assert.equal(circles(canvas).length, 2, 'stale action nodes stay out of the current selection');
});

test('selectAction changes emphasis and null restores the original', () => {
  const {view, canvas, selections, actions} = setup();
  view.setData([point(0, [-1, 0, 0]), point(1, [1, 0, 0])]);
  view.focus(0);
  view.setExpansion(0, [action(4, [0, 1, 0])]);
  const original = circles(canvas).map(p => ({...p}));
  view.selectAction(4);
  assert.ok(circles(canvas)[2].radius > original[2].radius);
  view.selectAction(null);
  assert.deepEqual(circles(canvas), original);
  assert.equal(selections.length + actions.length, 0);
});

test('coincident and single samples draw finitely; reset clears selection and data', () => {
  const {view, canvas} = setup();
  view.setData([point(2, [4, 4, 4]), point(7, [4, 4, 4])]);
  view.focus(2);
  view.setExpansion(2, [action(8, [4, 4, 4])]);
  view.fit();
  assert.equal(circles(canvas).length, 3);
  assert.ok(circles(canvas).every(p => p.x === 400 && p.y === 300), 'coincident points are never artificially separated');
  finiteDrawing(canvas);
  view.setData([point(10, [0, 0, 0])]);
  assert.equal(circles(canvas).length, 1);
  finiteDrawing(canvas);
  view.reset();
  assert.equal(circles(canvas).length, 0);
  view.fit();
  view.render();
  finiteDrawing(canvas);
});

test('picking uses frontmost depth in CSS coordinates and returns the same sample object', () => {
  const {view, canvas, selections} = setup();
  const near = point(20, [0, 0, 1]), far = point(21, [0, 0, -1]);
  view.setData([near, far]);
  click(canvas, circles(canvas).at(-1));
  assert.equal(selections.length, 1);
  assert.equal(selections[0], near, 'the visible foreground point wins a depth overlap');
  assert.equal(canvas.width, 1600);
  assert.equal(canvas.height, 1200);
  assert.equal(canvas.focused, true);
  assert.equal(canvas.captured.size, 0);
});

test('dimmed originals remain clickable, then actions and the focused original keep their identity', () => {
  const {view, canvas, selections, actions} = setup();
  const first = point(5, [-1, 0, 0]), second = point(8, [1, 0, 0]), altered = action(44, [0, 1, 0]);
  view.setData([first, second]);
  view.focus(5);
  click(canvas, circles(canvas)[1]);
  assert.equal(selections.at(-1), second);
  view.setExpansion(8, [altered]);
  click(canvas, circles(canvas)[2]);
  assert.equal(actions.at(-1), altered);
  assert.equal(actions.length, 1);
  click(canvas, circles(canvas)[1]);
  assert.equal(selections.at(-1), second);
  assert.equal(circles(canvas).length, 3, 'returning to the original preserves its expansion');
  assert.ok(circles(canvas)[1].radius > circles(canvas)[0].radius);
});

test('dragging orbits in 3D without altering measured positions or selecting a sample', () => {
  const {view, canvas, selections} = setup();
  const samples = [point(0, [0, 0, -1]), point(1, [0, 0, 1])];
  view.setData(samples);
  const start = circles(canvas).at(-1);
  const down = {clientX: canvas.box.left + start.x, clientY: canvas.box.top + start.y};
  canvas.fire('pointerdown', down);
  canvas.fire('pointermove', {...down, clientX: down.clientX + 40, clientY: down.clientY + 10});
  const firstMove = circles(canvas).map(p => ({...p}));
  view.render();
  canvas.fire('pointermove', {...down, clientX: down.clientX + 80, clientY: down.clientY + 20});
  assert.notDeepEqual(circles(canvas), firstMove, 'a render during the drag must preserve the active gesture');
  canvas.fire('pointerup', {...down, clientX: down.clientX + 80, clientY: down.clientY + 20});
  assert.ok(circles(canvas).some(p => Math.abs(p.x - start.x) > 20), 'Z separation rotates into screen X separation');
  assert.ok(circles(canvas).some(p => Math.abs(p.y - start.y) > 5), 'pitch rotates Z separation into screen Y separation');
  assert.equal(selections.length, 0, 'camera drags never activate a node');
  assert.deepEqual(samples.map(p => p.position), [[0, 0, -1], [0, 0, 1]]);
  assert.equal(canvas.captured.size, 0);
});

test('Shift and right drags pan both samples equally without changing their depth scale', () => {
  for (const modifier of [{shiftKey: true}, {button: 2}]) {
    const {view, canvas, selections} = setup();
    view.setData([point(0, [-1, 0, 0]), point(1, [1, 0, 0])]);
    const before = circles(canvas).map(p => ({...p}));
    canvas.fire('pointerdown', {clientX: 100, clientY: 100, ...modifier});
    canvas.fire('pointermove', {clientX: 130, clientY: 125});
    canvas.fire('pointerup', {clientX: 130, clientY: 125});
    circles(canvas).forEach((p, i) => {
      assert.equal(p.x - before[i].x, 30);
      assert.equal(p.y - before[i].y, 25);
      assert.equal(p.radius, before[i].radius);
    });
    assert.equal(selections.length, 0);
  }
});

test('small pointer jitter clicks, cancellation never clicks, and capture failures are harmless', () => {
  const {view, canvas, selections} = setup();
  view.setData([point(0, [0, 0, 0])]);
  const location = {clientX: canvas.box.left + 400, clientY: canvas.box.top + 300};
  canvas.fire('pointerdown', location);
  canvas.fire('pointermove', {...location, clientX: location.clientX + 2});
  canvas.fire('pointerup', {...location, clientX: location.clientX + 2});
  assert.equal(selections.length, 1);
  for (const type of ['pointercancel', 'lostpointercapture']) {
    canvas.fire('pointerdown', location);
    canvas.fire(type, location);
    canvas.fire('pointerup', location);
    assert.equal(selections.length, 1, `${type} cannot trigger selection`);
  }
  canvas.captureFails = true;
  assert.doesNotThrow(() => click(canvas, circles(canvas)[0]));
  assert.equal(selections.length, 2, 'capture failure must not disable a valid local click');
});

test('wheel and button zoom change perspective and fit restores all data', () => {
  const {view, canvas} = setup();
  view.setData([point(0, [-1, 0, 0]), point(1, [1, 0, 0])]);
  const before = circles(canvas).map(p => ({...p}));
  view.zoom(1.5);
  assert.ok(circles(canvas)[1].x > before[1].x);
  view.fit();
  assert.deepEqual(circles(canvas), before);
  const wheel = canvas.fire('wheel', {deltaY: -100, deltaMode: 0});
  assert.ok(circles(canvas)[1].x > before[1].x);
  assert.equal(wheel.defaultPrevented, true);
  view.zoom(0);
  view.zoom(NaN);
  view.zoom(Infinity);
  finiteDrawing(canvas);
});

test('resizing updates DPR, projection, and picking without moving data', () => {
  const {view, canvas, context, events, observed, selections} = setup();
  const sample = point(1, [0, 0, 0]);
  view.setData([sample]);
  canvas.box = {left: 100, top: 50, width: 500, height: 300};
  context.devicePixelRatio = 1.5;
  events.resize?.();
  observed.forEach(observer => observer.callback());
  assert.equal(canvas.width, 750);
  assert.equal(canvas.height, 450);
  assert.deepEqual(Array.from(canvas.context.transform), [1.5, 0, 0, 1.5, 0, 0]);
  assert.equal(circles(canvas)[0].x, 250);
  assert.equal(circles(canvas)[0].y, 150);
  click(canvas, circles(canvas)[0]);
  assert.equal(selections.at(-1), sample);
});

test('keyboard orbits, zooms, fits, and Escape clears exactly once', () => {
  const fixture = setup(), {view, canvas, selections, actions} = fixture;
  view.setData([point(0, [0, 0, -1]), point(1, [0, 0, 1])]);
  assert.equal(canvas.tabIndex, 0);
  const before = circles(canvas).map(p => ({...p}));
  assert.equal(canvas.fire('keydown', {key: 'ArrowRight'}).defaultPrevented, true);
  assert.ok(circles(canvas).some(p => p.x !== 400));
  canvas.fire('keydown', {key: 'ArrowLeft'});
  assert.deepEqual(circles(canvas), before);
  canvas.fire('keydown', {key: 'ArrowUp'});
  assert.ok(circles(canvas).some(p => p.y !== 300));
  canvas.fire('keydown', {key: 'ArrowDown'});
  canvas.fire('keydown', {key: '+'});
  assert.ok(circles(canvas)[1].radius > before[1].radius);
  canvas.fire('keydown', {key: '0'});
  assert.deepEqual(circles(canvas), before);
  view.focus(0);
  view.setExpansion(0, [action(3, [1, 0, 0])]);
  assert.equal(canvas.fire('keydown', {key: 'Escape'}).defaultPrevented, true);
  assert.equal(fixture.clears, 1);
  assert.equal(circles(canvas).length, 2);
  assert.equal(circles(canvas)[0].alpha, circles(canvas)[1].alpha);
  assert.equal(selections.length + actions.length, 0);
});

test('hover exposes sample and action information as plain text', () => {
  const {view, canvas} = setup();
  const sample = Object.freeze({...point(0, [-1, 0, 0]), name: '<img src=x onerror=bad()>'});
  view.setData([sample, point(1, [1, 0, 0])]);
  let p = circles(canvas)[0];
  canvas.fire('pointermove', {clientX: canvas.box.left + p.x, clientY: canvas.box.top + p.y});
  assert.ok(canvas.title.includes(sample.name));
  view.focus(0);
  view.setExpansion(0, [action(4, [0, 1, 0])]);
  p = circles(canvas)[2];
  canvas.fire('pointermove', {clientX: canvas.box.left + p.x, clientY: canvas.box.top + p.y});
  assert.ok(canvas.title.includes('rotation') && canvas.title.includes('15'));
  assert.ok(canvas.title.includes('0.87'));
  canvas.fire('pointerleave');
  assert.equal(canvas.title, '');
});

test('points behind the camera are neither drawn nor picked after a close zoom', () => {
  const {view, canvas, selections} = setup();
  const near = point(0, [0, 0, 1]), far = point(1, [0, 0, -1]);
  view.setData([near, far]);
  view.zoom(100);
  assert.equal(circles(canvas).length, 1);
  finiteDrawing(canvas);
  click(canvas, circles(canvas)[0]);
  assert.equal(selections.at(-1), far);
});

test('fit keeps the complete expansion visible in a narrow resized canvas', () => {
  const {view, canvas} = setup();
  canvas.box.width = 240;
  canvas.box.height = 800;
  view.setData([point(0, [-1, 0, -2]), point(1, [1, 1, 0])]);
  view.focus(0);
  view.setExpansion(0, [action(3, [100, 12, 200])]);
  canvas.fire('keydown', {key: 'ArrowRight'});
  view.fit();
  assert.equal(circles(canvas).length, 3);
  for (const p of circles(canvas)) {
    assert.ok(p.x > p.radius && p.x + p.radius < 240);
    assert.ok(p.y > p.radius && p.y + p.radius < 800);
  }
});

test('data replacement cancels an active gesture and releases pointer capture', () => {
  const {view, canvas, selections} = setup();
  view.setData([point(0, [0, 0, 0])]);
  const location = {clientX: canvas.box.left + 400, clientY: canvas.box.top + 300};
  canvas.fire('pointerdown', location);
  view.setData([point(2, [-1, 0, 0]), point(3, [1, 0, 0])]);
  assert.equal(canvas.captured.size, 0, 'a previous dataset cannot retain the pointer');
  const before = circles(canvas).map(p => ({...p}));
  canvas.fire('pointermove', {...location, clientX: location.clientX + 40});
  canvas.fire('pointerup', location);
  assert.deepEqual(circles(canvas), before);
  assert.equal(selections.length, 0);
});

test('pointerup displacement suppresses accidental clicks even without a final move event', () => {
  const {view, canvas, selections} = setup();
  view.setData([point(0, [0, 0, 0])]);
  const location = {clientX: canvas.box.left + 400, clientY: canvas.box.top + 300};
  canvas.fire('pointerdown', location);
  canvas.fire('pointerup', {...location, clientX: location.clientX + 5});
  assert.equal(selections.length, 0);
});

test('discovery colors only measured pairs, aggregates observed originals and preserves coordinates/camera', () => {
  const {view,canvas}=setup();
  const samples=[point(0,[-1,0,0]),point(1,[1,0,0]),point(2,[0,-1,0])];
  view.setData(samples);view.focus(0);view.setExpansion(0,[action(3,[0,1,0]),action(4,[0,2,0])]);
  view.orbit(.2,.1);view.zoom(1.2);
  const before=view.projected.map(node=>({kind:node.kind,x:node.x,y:node.y,id:node.data.id??node.data.index}));
  const records=[{input_index:0,action_id:3,severity:1},{input_index:1,action_id:3,severity:0}];
  view.setDiscovery({status:'running',records,current:{input_index:0,action_id:3}});
  assert.deepEqual(view.projected.map(node=>({kind:node.kind,x:node.x,y:node.y,id:node.data.id??node.data.index})),before,'polls cannot move the camera or points');
  assert.equal(view.severities.get(0),1,'one measured failing action makes this observed aggregate 1, not 1 divided by all actions');
  assert.equal(view.severities.get(1),0);assert.equal(view.severities.has(2),false);
  const colors=view.projected.map((node,index)=>({node,color:circles(canvas)[index].color}));
  assert.equal(colors.find(item=>item.node.kind==='action'&&item.node.data.id===4).color,'#9aa1a9','unvisited action stays gray');
  assert.equal(colors.find(item=>item.node.kind==='action'&&item.node.data.id===3).color,view.severityColor(1));
  assert.ok(canvas.context.strokes.some(stroke=>stroke.color==='#203f72'),'current pair is outlined');
  view.setDiscovery({status:'completed',records:[...records,{input_index:0,action_id:4,severity:0}]});
  assert.equal(view.severities.get(0),.5);
  assert.ok(!canvas.context.strokes.some(stroke=>stroke.color==='#203f72'),'terminal jobs stop the pulse');
  view.setDiscovery(null);assert.equal(view.discovery,null);
});

test('action size ranks cross-input preference variation, preserves ties and distinguishes unknown values',()=>{
  const {view,canvas}=setup();
  view.setData([point(0,[-1,0,0]),point(1,[1,0,0])]);view.focus(0);
  view.setExpansion(0,[action(0,[-.5,.5,0]),action(1,[0,.5,0]),action(2,[.5,.5,0]),action(3,[-.5,-.5,0]),action(4,[.5,-.5,0])]);
  const originalRadius=view.projected.find(node=>node.kind==='sample'&&node.data.index===0).radius;
  const positions=view.projected.map(node=>({kind:node.kind,id:node.data.id??node.data.index,x:node.x,y:node.y}));
  const statistics={method:'softmax_q_across_original_inputs',temperature:1,input_count:4,policy_updates:10,std_ddof:1,actions:[
    {action_id:0,mean:.6,std:.02,evaluated_inputs:3},
    {action_id:1,mean:.5,std:.08,evaluated_inputs:2},
    {action_id:2,mean:.3,std:.20,evaluated_inputs:1},
    {action_id:3,mean:.4,std:.08,evaluated_inputs:2},
    {action_id:4,mean:.1,std:null,evaluated_inputs:0},
  ]};
  view.setDiscovery({status:'completed',records:[{input_index:0,action_id:0,severity:.9},{input_index:0,action_id:2,severity:.1}],policy_statistics:statistics});
  const byId=id=>view.projected.find(node=>node.kind==='action'&&node.data.id===id);
  assert.ok(byId(0).radius>byId(1).radius&&byId(1).radius>byId(2).radius,'higher variation produces smaller actions at the same depth');
  assert.equal(byId(1).radius,byId(3).radius,'equal variation cannot acquire an arbitrary size order');
  assert.ok(byId(4).radius<byId(2).radius,'unknown variation cannot appear maximally certain');
  assert.equal(view.projected.find(node=>node.kind==='sample'&&node.data.index===0).radius,originalRadius,'original nodes remain role markers');
  assert.deepEqual(view.projected.map(node=>({kind:node.kind,id:node.data.id??node.data.index,x:node.x,y:node.y})),positions);
  assert.equal(circles(canvas)[view.projected.indexOf(byId(0))].color,view.severityColor(.9),'size does not change measured-severity color');
  assert.equal(circles(canvas)[view.projected.indexOf(byId(2))].color,view.severityColor(.1));
  assert.ok(canvas.context.strokes.some(stroke=>stroke.dashes.length&&stroke.path.some(p=>p.x===byId(4).x&&p.y===byId(4).y)),'unknown variation has a distinct rim');
  const selectedRadius=byId(2).radius;view.selectAction(2);assert.equal(byId(2).radius,selectedRadius,'selection emphasis cannot override uncertainty size');
  const hovered=byId(0);canvas.fire('pointermove',{clientX:canvas.box.left+hovered.x,clientY:canvas.box.top+hovered.y});
  assert.match(canvas.title,/mean DQN action preference: 0\.600/i);assert.match(canvas.title,/sample SD.*0\.0200/i);
  assert.match(canvas.title,/4 original inputs/i);assert.match(canvas.title,/tested inputs: 3\/4/i);assert.match(canvas.title,/not calibrated/i);
});

test('single-input and untrained policy statistics remain unknown; tied statistics use a neutral size',()=>{
  for(const [input_count,policy_updates,std,reason] of [[1,10,null,/at least 2 original inputs/i],[4,0,0,/no policy updates/i]]) {
    const {view,canvas}=setup();view.setData([point(0,[-1,0,0]),point(1,[1,0,0])]);view.focus(0);view.setExpansion(0,[action(0,[0,.5,0])]);
    view.setDiscovery({status:'completed',records:[],policy_statistics:{input_count,policy_updates,actions:[{action_id:0,mean:.5,std,evaluated_inputs:1}]}});
    const p=view.projected.find(node=>node.kind==='action');canvas.fire('pointermove',{clientX:canvas.box.left+p.x,clientY:canvas.box.top+p.y});
    assert.match(canvas.title,/variation.*unavailable/i);assert.match(canvas.title,reason);
    assert.ok(canvas.context.strokes.some(stroke=>stroke.dashes.length),'undefined variation stays visibly unknown');
  }
  const {view,canvas}=setup();view.setData([point(0,[-1,0,0]),point(1,[1,0,0])]);view.focus(0);view.setExpansion(0,[action(0,[-.5,.5,0]),action(1,[.5,.5,0])]);
  view.setDiscovery({status:'completed',records:[],policy_statistics:{input_count:2,policy_updates:10,actions:[{action_id:0,mean:.4,std:.1,evaluated_inputs:1},{action_id:1,mean:.6,std:.1,evaluated_inputs:1}]}});
  const actions=view.projected.filter(node=>node.kind==='action');assert.equal(actions[0].radius,actions[1].radius);finiteDrawing(canvas);
});

let failures = 0;
for (const {name, run} of tests) {
  try {run(); console.log(`ok - ${name}`);}
  catch (error) {failures++; console.error(`not ok - ${name}\n${error.stack}`);}
}
if (failures) process.exitCode = 1;
else console.log(`${tests.length} embedding renderer tests passed`);
