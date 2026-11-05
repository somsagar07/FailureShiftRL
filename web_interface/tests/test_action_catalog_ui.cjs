// Exercise the browser action editor with an expanded catalog and unchanged defaults.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const defaults={
  classification:{rotation:[-10,-5,0,5,10],darkness:[.1,.2,.3,.4,.5],saturation:[.5,.55,.6,.65,.7]},
  summarization:{actions:['drop_stop_words','remove_punctuation','to_lowercase','negate_sentence','delete_random_character','insert_random_character','change_letter_case','apply_misspellings','repeat_random_character','replace_with_adjacent_character','swap_random_character','delete_random_word','perturb_word_order','repeat_random_word']},
  generation:{style:['photograph','watercolor','pencil drawing','digital illustration','3D render'],lighting:['even','soft','directional','high-key','low-key'],detail:['minimal','simple','moderate','detailed','intricate']}
};
const catalog={tasks:defaults,text:'The trial succeeded.',prompt:'A blue vase.',limits:{max_actions:5000},
  action_catalog:{
    classification:{...defaults.classification,rotation:[...defaults.classification.rotation,-20,20],blur:[.5,1,2,3],contrast:[.2,.4,.6,.8,1,1.2,1.4,1.6,1.8,2],noise:[.05,.1]},
    summarization:{actions:[...defaults.summarization.actions,'normalize_whitespace']},
    generation:{...defaults.generation,style:[...defaults.generation.style,'ink sketch'],camera:['close-up','wide shot']}
  },
  action_metadata:{classification:{blur:{label:'Gaussian blur',description:'Soften image edges.',semantic_risk:'visibility',risk_reason:'Fine details can become unreadable.',optional:true,minimum:0,maximum:10,identity:0,integer:false}},
    summarization:{normalize_whitespace:{label:'Normalize whitespace',description:'Keep words while normalizing spacing.',semantic_risk:'low',optional:true}},
    generation:{camera:{label:'Camera framing',description:'Choose the scene framing.',semantic_risk:'medium',optional:true}}}
};
async function setup(data=catalog) {
  const elements=new Map();
  class Element {
    constructor(){this.value='';this.children=[];this.events={};this.dataset={};this.attrs={};this.style={};this.classList={toggle(){}};this.textContent='';}
    set id(value){this._id=value;elements.set(value,this);}get id(){return this._id;}
    replaceChildren(...children){this.children=children;}append(...children){this.children.push(...children);}
    addEventListener(name,handler){this.events[name]=handler;}setAttribute(key,value){this.attrs[key]=value;}
    getBoundingClientRect(){return {left:0,top:0,width:800,height:600};}
    getContext(){return new Proxy({measureText:()=>({width:40})},{get:(target,key)=>target[key]||(()=>{})});}
    focus(){}querySelectorAll(){return [];}
  }
  const get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
  const domains=['classification','summarization','generation'].map(task=>{const el=new Element();el.dataset.task=task;return el;});
  const requests=[],context={console,document:{getElementById:get,createElement:()=>new Element(),createElementNS:()=>new Element(),querySelectorAll:selector=>selector==='.domain'?domains:[]},
    setTimeout:()=>1,clearTimeout(){},window:{addEventListener(){},localStorage:{getItem:()=>null}},fetch:async(url,options={})=>{
      requests.push({url,body:options.body&&JSON.parse(options.body)});
      const result=url==='/api/catalog'?data:url==='/api/datasets'?{id:'images',status:'ready',count:1}:url.startsWith('/api/datasets/images/items')?{items:[{index:0,name:'image.png'}],count:1,offset:0}:
        url.startsWith('/api/datasets/options?')?{configs:['default'],config:'default',splits:[{name:'train',count:1}],image_columns:['image'],text_columns:['text'],summary_columns:['summary']}:{error:'Preview not executed by this test.'};
      return {ok:url!=='/api/previews',json:async()=>result};
    }};
  get('model-source').value='local';get('dataset-source').value='folder';get('folder-path').value='/images';
  vm.createContext(context);
  for(const file of ['graph.js','embedding-view.js','embedding-controller.js','adaptation-controller.js','discovery-controller.js','app.js'])vm.runInContext(fs.readFileSync(path.join(__dirname,'../static',file),'utf8'),context);
  for(let i=0;i<30;i++)await Promise.resolve();
  assert.equal(get('form-error').textContent,'','the real app must initialize successfully');
  const run=source=>vm.runInContext(source,context),space=()=>JSON.parse(run('JSON.stringify(actionSpace())'));
  const add=key=>{get('action-add').value=key;get('action-add').events.change();};
  const remove=key=>get(`remove-action-${key}`).events.click();
  const choose=(key,value)=>{if(get(`edit-action-${key}`).attrs['aria-expanded']!=='true')get(`edit-action-${key}`).events.click();get(`action-${key}`).value=value;get(`action-${key}`).events.change();};
  const switchTask=task=>domains.find(button=>button.dataset.task===task).events.click();
  return {get,requests,run,space,add,remove,choose,switchTask};
}
const tests=[];const test=(name,fn)=>tests.push({name,fn});
test('optional actions remain opt-in and restoring defaults preserves all three default spaces',async()=>{
  const f=await setup(),{get,space,add,remove,switchTask}=f;
  for(const [task,count,optional,label] of [['classification',125,'blur','Gaussian blur'],['summarization',14,'normalize_whitespace','Normalize whitespace'],['generation',125,'camera','Camera framing']]) {
    switchTask(task);
    assert.match(get('action-count').textContent,new RegExp(`^${count} combination`));
    assert.deepEqual(space(),defaults[task],'available optional choices must not change a default payload');
    assert.ok(get('action-add').children.some(option=>option.value===optional&&option.textContent.includes(label)),'the add menu exposes a readable optional action');
    add(optional);
    assert.ok(task==='summarization'?space().actions.includes(optional):Object.hasOwn(space(),optional));
    remove(optional);
    assert.deepEqual(space(),defaults[task],'removing an optional group must omit it from the request');
    add(optional);get('reset-actions').events.click();
    assert.deepEqual(space(),defaults[task],'Restore defaults removes every optional choice');
  }
});
test('editing defaults exposes expanded catalog values without silently expanding the selection',async()=>{
  const {get,space,choose}=await setup();
  get('edit-action-rotation').events.click();
  assert.deepEqual(space().rotation,[-10,-5,0,5,10]);
  assert.ok(get('action-rotation').children.some(option=>option.textContent==='20°'),'the full catalog is available in the editor');
  choose('rotation','all');assert.deepEqual(space().rotation,[-10,-5,0,5,10,-20,20]);
  choose('rotation','defaults');assert.deepEqual(space().rotation,[-10,-5,0,5,10],'the default subset is a distinct choice');
});
test('the full Cartesian count blocks submission above the limit and allows the exact limit',async()=>{
  const {get,add,remove,requests}=await setup();
  await get('load-dataset').events.click();get('model-name').value='/model';get('model-name').events.input();
  add('blur');add('contrast');
  assert.match(get('action-count').textContent,/^5000 combinations/);assert.equal(get('generate').disabled,false);
  add('noise');assert.match(get('action-count').textContent,/^10000 combinations/);
  assert.equal(get('generate').disabled,true);assert.equal(get('action-error').hidden,false);
  assert.match(get('action-error').textContent,/10,?000.*5,?000/);
  const before=requests.length;await get('preview-form').events.submit({preventDefault(){}});
  assert.equal(requests.length,before,'programmatic submits cannot bypass the action limit');
  remove('noise');await get('preview-form').events.submit({preventDefault(){}});
  const sent=requests.filter(request=>request.url==='/api/previews').at(-1);
  assert.ok(sent,'the exact maximum is allowed');assert.equal(Object.hasOwn(sent.body.action_space,'noise'),false);
});
test('older catalog responses retain their complete default action workflow',async()=>{
  const {get,space}=await setup({tasks:defaults,text:'A trial.',prompt:'A vase.'});
  assert.deepEqual(space(),defaults.classification);assert.equal(get('action-add').disabled,true);
  get('edit-action-rotation').events.click();assert.equal(get('action-rotation').value,'all');
});
(async()=>{let failures=0;for(const {name,fn} of tests){try{await fn();console.log('ok - '+name);}catch(error){failures++;console.error('not ok - '+name+'\n'+error.stack);}}if(failures)process.exitCode=1;else console.log(`${tests.length} expanded action catalog UI tests passed`);})();
