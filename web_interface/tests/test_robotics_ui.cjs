/* Run: node tests/test_robotics_ui.cjs (uses a local Chrome binary, no packages).
 * Also runs in a browser: load before index.html scripts, then call
 * runRoboticsUITests() after DOMContentLoaded. Only fetch is replaced:
 * navigation, forms, uploads, graph rendering, polling and inspection are real.
 */
'use strict';
if(typeof window==='undefined') {
  const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),http=require('node:http'),{execFile}=require('node:child_process');
  const root=path.resolve(__dirname,'..'),profile=fs.mkdtempSync(path.join(os.tmpdir(),'robotics-ui-chrome-'));
  const pixel=Buffer.from('R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==','base64');
  const server=http.createServer((request,response)=>{
    const route=new URL(request.url,'http://localhost').pathname;
    let body,mime;
    if(route==='/') {
      body=fs.readFileSync(path.join(root,'static/index.html'),'utf8').replace('<head>','<head><script src="/test.js"></script>').replace('</body>','<script>addEventListener("DOMContentLoaded",()=>runRoboticsUITests());</script></body>');mime='text/html';
    } else if(route.startsWith('/robotics/')){body=pixel;mime='image/gif';}
    else {
      const filename=route==='/test.js'?__filename:path.join(root,'static',path.basename(route));
      if(!fs.existsSync(filename)){response.writeHead(404);response.end();return;}
      body=fs.readFileSync(filename);mime=filename.endsWith('.css')?'text/css':'text/javascript';
    }
    response.writeHead(200,{'Content-Type':mime});response.end(body);
  });
  server.listen(0,'127.0.0.1',()=>{
    execFile(process.env.CHROME_BIN||'google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--no-first-run','--no-default-browser-check',`--user-data-dir=${profile}`,'--dump-dom','--virtual-time-budget=12000',`http://127.0.0.1:${server.address().port}/`],{timeout:45000,maxBuffer:8*1024*1024},(error,stdout)=>{
      try {
        if(error)throw error;
        const match=stdout.match(/<pre id="robotics-test-result"[^>]*>([\s\S]*?)<\/pre>/);
        if(!match)throw new Error('Browser did not produce test results.');
        const decode=text=>text.replaceAll('&quot;','"').replaceAll('&lt;','<').replaceAll('&gt;','>').replaceAll('&amp;','&');
        const report=JSON.parse(decode(match[1]));
        for(const name of report.passed)console.log('ok - '+name);
        if(report.error)throw new Error(report.error);
        console.log(`${report.passed.length} robotics browser tests passed`);
      } catch(failure) {console.error(failure.stack);process.exitCode=1;}
      finally {server.close();fs.rmSync(profile,{recursive:true,force:true});}
    });
  });
} else {
(() => {
  const requests=[];
  const legacy={tasks:{classification:{rotation:[-5,0,5],darkness:[0,.2,.4],saturation:[.5,1,1.5]},summarization:{actions:['to_lowercase']},generation:{style:['photo'],lighting:['soft'],detail:['simple']}},text:'A source article.',prompt:'A blue vase.'};
  const catalog={tasks:[{id:'Lift',label:'Lift',instruction:'Lift the cube.'},{id:'Stack',label:'Stack',instruction:'Stack the cubes.'},{id:'TwoArmLift',label:'TwoArmLift',instruction:'Lift the object with both arms.'}],robots:['Panda','Sawyer','Baxter'],controllers:['OSC_POSE','JOINT_POSITION'],
    policy_formats:[{id:'robomimic',label:'Robomimic',sources:['local','upload']},{id:'torchscript',label:'TorchScript',sources:['local','upload']},{id:'sb3',label:'Stable-Baselines3',sources:['local','upload']},{id:'lerobot',label:'LeRobot / VLA',sources:['local','upload','huggingface']}],
    action_groups:[{id:'friction_scale',label:'Contact friction',values:[.5,1,1.5],neutral:1,default:true},{id:'mass_scale',label:'Object mass',values:[.5,1,1.5],neutral:1,default:true},{id:'object_x_offset',label:'Object x offset',values:[-.03,0,.03],neutral:0,default:true},{id:'camera_offset',label:'Camera offset',values:[-.1,0,.1],neutral:0,default:false}],
    default_action_space:{friction_scale:[.5,1,1.5],mass_scale:[.5,1,1.5],object_x_offset:[-.03,0,.03]},
    defaults:{environment:{task:'Lift',robots:['Panda'],controller:'OSC_POSE',horizon:300,control_freq:20},episodes_per_variation:2,variation_budget:12,seed:42,device:'auto',record_video:true},
    runtime:{available:true,message:'Test runtime available'},limits:{max_actions:5000,chunk_bytes:8388608}};
  const starters=[{id:'lift',label:'Lift starter',description:'A learned Lift checkpoint.',policy:{format:'torchscript',source:'local',name:'/runs/starter-policies/lift.pt'},environment:{task:'Lift',robots:['Panda'],controller:'OSC_POSE',control_freq:20,horizon:180},mapping:{state_keys:['robot0_eef_pos','object-state']},instruction:'Lift the cube.',evaluation:{successes:3,episodes:5,success_rate:.6},training:{method:'Behavior cloning'}},{id:'pickplace',label:'PickPlaceCan starter',description:'A learned pick-and-place checkpoint.',policy:{format:'torchscript',source:'local',name:'/runs/starter-policies/pickplace.pt'},environment:{task:'PickPlaceCan',robots:['Panda'],controller:'OSC_POSE',control_freq:10,horizon:240},mapping:{state_keys:['object-state','robot0_eef_pos']},instruction:'Place the can into its bin.',evaluation:{successes:1,episodes:5,success_rate:.2},training:{method:'Behavior cloning'}}];
  starters[0].checkpoint_sha256='a'.repeat(64);starters[1].checkpoint_sha256='b'.repeat(64);
  catalog.starter_policies=[starters[0]];
  const baseline={episodes:2,success_rate:.5,failure_rate:.5,errors:0,records:[{seed:42,success:true,steps:40,return:1,image:'/robotics/analysis-1/artifacts/baseline-seed-42.jpg'},{seed:43,success:false,steps:300,return:0}]};
  const nodes=[{id:0,label:'Variation 1',values:{friction_scale:.5,mass_scale:.5,object_x_offset:-.03},position:[-1,-1,-1],status:'untested',episodes:0},{id:1,label:'Variation 2',values:{friction_scale:1,mass_scale:.5,object_x_offset:0},position:[0,-1,0],status:'untested',episodes:0},{id:2,label:'Variation 3',values:{friction_scale:1.5,mass_scale:1.5,object_x_offset:.03},position:[1,1,1],status:'untested',episodes:0},{id:3,label:'Neutral variation',values:{friction_scale:1,mass_scale:1,object_x_offset:0},position:[0,0,0],status:'untested',episodes:0}];
  const embedding={version:1,source:{kind:'policy_hidden',source_layer:'policy.network.2',dimension:64,pooling:'mean'},projection:{method:'PCA',rank:3},points:[{index:0,seed:42,name:'Episode 1 · seed 42',position:[-1,0,0],success:true,steps:40,image:'/robotics/analysis-1/artifacts/baseline-seed-42.jpg'},{index:1,seed:43,name:'Episode 2 · seed 43',position:[1,.5,0],success:false,steps:300}]};
  let predictionsUnavailable=false;
  let analysis,discovery,discoverCount=0,releaseLegacyCatalog,failBaseline=false,includePolicyWarnings=true;
  let environmentJob,environmentCount=0,failEnvironment=false,holdEnvironment=false,starterCatalog=[];
  let holdAnalysisStart=false,releaseAnalysisStart,holdEnvironmentStart=false,releaseEnvironmentStart;
  let delayCancels=false;const pendingCancels=new Map();
  const reply=data=>Promise.resolve({ok:true,json:async()=>structuredClone(data)});
  window.fetch=(url,options={})=>{
    const body=typeof options.body==='string'?JSON.parse(options.body):options.body;
    requests.push({url:String(url),method:options.method||'GET',body});
    if(url==='/api/catalog')return new Promise(resolve=>{releaseLegacyCatalog=()=>resolve({ok:true,json:async()=>structuredClone(legacy)});});
    if(String(url).startsWith('/api/datasets/options?'))return reply({configs:['default'],config:'default',splits:[{name:'train',count:1}],image_columns:['image'],label_columns:['label'],text_columns:['text'],summary_columns:['summary'],prompt_columns:['prompt'],checks_columns:['checks']});
    if(url==='/api/robotics/catalog')return reply(catalog);
    if(url==='/api/robotics/starter-policies')return reply({policies:starterCatalog});
    if(url==='/api/robotics/policies')return reply({id:'upload-1',status:'uploading',chunk_bytes:8388608,files:body.files,total_bytes:body.files.reduce((sum,file)=>sum+file.size,0),uploaded_bytes:0});
    if(String(url).includes('/api/robotics/policies/upload-1/files/'))return reply({id:'upload-1',status:'uploading'});
    if(url==='/api/robotics/policies/upload-1/finish')return reply({id:'upload-1',status:'ready'});
    if(String(url).endsWith('/cancel')&&delayCancels){
      const cancelledJob=structuredClone(String(url).includes('/environment-')?environmentJob:discovery);
      return new Promise((resolve,reject)=>pendingCancels.set(cancelledJob.id,{
        resolve:()=>resolve({ok:true,json:async()=>({...cancelledJob,status:'cancelled',message:'Delayed cancellation complete'})}),
        reject:()=>reject(new Error('Delayed cancellation network error'))
      }));
    }
    if(url==='/api/robotics/runs'&&body.mode==='environment'){
      environmentJob={id:'environment-'+(++environmentCount),mode:'environment',status:'queued',config:body,message:'Loading environment',phase:'preparing',nodes:[],records:[],artifacts:{}};
      if(holdEnvironmentStart)return new Promise(resolve=>{releaseEnvironmentStart=()=>resolve({ok:true,json:async()=>structuredClone(environmentJob)});});
      return reply(environmentJob);
    }
    if(String(url).startsWith('/api/robotics/runs/environment-')){
      if(String(url).endsWith('/cancel'))environmentJob={...environmentJob,status:'cancelled',message:'Environment load cancelled'};
      else if(failEnvironment)environmentJob={...environmentJob,status:'error',message:'Environment load failed',error:'The selected environment could not render.'};
      else if(holdEnvironment)environmentJob={...environmentJob,status:'running',message:'Resetting environment'};
      else environmentJob={...environmentJob,status:'completed',message:'Environment loaded',phase:'completed',environment_info:{...environmentJob.config.environment,action_dimension:7,observation_shapes:{'robot0_eef_pos':[3],'object-state':[14]},cameras:['agentview','robot0_eye_in_hand'],preview_camera:'agentview',seed:environmentJob.config.seed},artifacts:{preview:`/robotics/${environmentJob.id}/artifacts/environment.jpg`}};
      return reply(environmentJob);
    }
    if(url==='/api/robotics/runs'&&body.mode==='analyze'){
      analysis={id:'analysis-1',mode:'analyze',status:'completed',config:body,message:'Baseline complete',phase:'completed',completed:2,total:2,episode:2,step:300,horizon:300,baseline,embedding,nodes,records:[],artifacts:{report:'/robotics/analysis-1/artifacts/report.json'}};
      if(includePolicyWarnings)Object.assign(analysis,{warnings:[' Saved VecNormalize statistics may be needed. ','Saved VecNormalize statistics may be needed.'],policy_info:{warnings:['Saved VecNormalize statistics may be needed.','Controller action units must match the checkpoint.']}});
      if(failBaseline)analysis={...analysis,status:'error',error:'No baseline episode completed.',baseline:{episodes:0,success_rate:null,failure_rate:null,errors:1,records:[{seed:42,error:'Missing camera observation.images.front'}]}};
      if(holdAnalysisStart)return new Promise(resolve=>{releaseAnalysisStart=()=>resolve({ok:true,json:async()=>structuredClone(analysis)});});
      return reply(analysis);
    }
    if(url==='/api/robotics/runs'&&body.mode==='discover'){
      discoverCount++;
      discovery={...analysis,id:'discovery-'+discoverCount,analysis_id:analysis.id,mode:'discover',status:'running',phase:'search',search:{...body.search,episode:0,epsilon:body.search.epsilon_start,model_calls:0,cache_hits:0,updates:0,loss:null},current:{input_index:0,action_id:0},completed:0,total:4,step:0,message:'Evaluating variations',active_action_id:0,artifacts:{preview:'/robotics/discovery-1/artifacts/live.jpg'}};
      return reply(discovery);
    }
    if(String(url).endsWith('/cancel')){discovery={...discovery,status:'cancelled',message:'Cancelled; partial results retained'};return reply(discovery);}
    if(String(url).includes('/predictions?')&&predictionsUnavailable){predictionsUnavailable=false;return reply({input_index:Number(String(url).split('input_index=')[1]),policy_updates:7,available:false,actions:[]});}
    if(String(url).includes('/predictions?'))return reply({input_index:Number(String(url).split('input_index=')[1]),policy_updates:7,actions:[{action_id:0,q_value:.9},{action_id:1,q_value:.2},{action_id:2,q_value:.4},{action_id:3,q_value:.1}]});
    if(String(url).startsWith('/api/robotics/runs/discovery-')){
      if(discoverCount===1)discovery={...discovery,status:'completed',phase:'completed',completed:4,total:4,step:300,search:{...discovery.search,episode:17,epsilon:.4,model_calls:3,cache_hits:14,updates:7,loss:.125},predicted_landscape:{policy_updates:7,input_mean_reward:[.6,.2]},records:[{input_index:0,action_id:0,seed:42,success:false,return:0,steps:300,embedding_warning:'Feature projection unavailable for this episode.'},{input_index:1,action_id:0,seed:43,success:false,return:0,steps:300,embedding:{position:[.5,.3,.1],similarity:.7,feature_samples:20},video:'/robotics/discovery-1/artifacts/variation-0.gif'},{input_index:0,action_id:2,seed:42,error:'Unsupported physical change'},...baseline.records.map(record=>({...record,action_id:3,reused_baseline:true}))],nodes:[{...nodes[0],status:'completed',episodes:2,success_rate:0,failure_rate:1,regressions:1,recoveries:0,video:'/robotics/discovery-1/artifacts/variation-0.gif'},nodes[1],{...nodes[2],status:'error',episodes:0,errors:1},{...nodes[3],status:'completed',episodes:2,success_rate:.5,failure_rate:.5,regressions:0,recoveries:0}],artifacts:{report:'/robotics/discovery-1/artifacts/report.json',preview:'/robotics/discovery-1/artifacts/live.jpg'}};
      return reply(discovery);
    }
    return Promise.reject(new Error('Unexpected test request: '+url));
  };
  const check=(condition,message)=>{if(!condition)throw new Error(message);};
  const get=id=>document.getElementById(id);
  const change=(id,value)=>{get(id).value=value;get(id).dispatchEvent(new Event('change',{bubbles:true}));};
  const flush=async()=>{for(let i=0;i<40;i++)await Promise.resolve();};
  const until=async(predicate,message)=>{for(let i=0;i<150;i++){if(predicate())return;await new Promise(resolve=>setTimeout(resolve,20));}throw new Error(message);};
  window.runRoboticsUITests=async()=>{
    const passed=[];
    try{
      await flush();
      const nav=document.querySelector('.domain[data-task="robotics"]');
      check(nav,'Robosuite must be selectable as a fourth workspace');
      nav.click();await flush();releaseLegacyCatalog();await flush();
      check(!get('robotics-workspace').hidden,'A late legacy catalog must not change the selected workspace');
      document.querySelector('.domain[data-task="classification"]').click();await flush();
      check(get('action-chips').children.length===3&&get('model-preset').children.length>1,'The legacy workspace must initialize after entering robotics early');
      get('model-source').value='local';get('model-name').value='/preserved-model';
      nav.click();await flush();
      check(get('workspace-layout').hidden&&!get('robotics-workspace').hidden,'Robosuite must display its own panel');
      check(get('robotics-action-chips').children.length===3,'Default physical groups must remain enabled');
      check(/27/.test(get('robotics-action-count').textContent),'Default combinations must be counted');
      get('robotics-remove-mass_scale').click();
      check(/9/.test(get('robotics-action-count').textContent),'Removing a group must shrink combinations');
      change('robotics-action-add','mass_scale');
      check(/27/.test(get('robotics-action-count').textContent),'A removed default group must be addable again');
      get('robotics-remove-mass_scale').click();
      change('robotics-action-add','camera_offset');
      check(/27/.test(get('robotics-action-count').textContent),'Optional group must join the Cartesian product');
      get('robotics-reset-actions').click();
      passed.push('navigation and variation editor');

      check(get('robotics-load-environment')&&!get('robotics-load-environment').disabled,'Environment loading must be available before choosing a policy');
      check(get('robotics-load-environment').type==='button'&&get('robotics-analyze').disabled,'The environment action must bypass policy form validation');
      get('robotics-mapping').value='{invalid';get('robotics-episodes').value='';
      get('robotics-load-environment').click();await flush();
      const environmentRequest=requests.find(request=>request.body?.mode==='environment').body;
      check(JSON.stringify(Object.keys(environmentRequest).sort())==='["environment","mode","seed"]','Environment loading must exclude all policy, mapping, and evaluation settings');
      check(environmentRequest.environment.task==='Lift'&&environmentRequest.environment.robots[0]==='Panda'&&environmentRequest.seed===42,'Environment loading must use the configured simulator and seed');
      check(get('robotics-load-environment').disabled&&!get('robotics-cancel').hidden,'A pending environment load must use worker cancellation controls');
      await until(()=>/Environment loaded/.test(get('robotics-status').textContent),'Environment preview did not finish');
      check(!get('robotics-environment-preview').hidden&&get('robotics-environment-image').getBoundingClientRect().width>100,'The loaded environment preview must be visible');
      if(window.innerWidth<=900)check(get('robotics-environment-summary').getBoundingClientRect().top<window.innerHeight,'A completed environment preview must come into view on narrow screens');
      check(get('robotics-environment-image').getAttribute('src').includes('/environment-1/'),'The preview must use the completed environment artifact');
      check(/Lift/.test(get('robotics-environment-details').textContent)&&/Panda/.test(get('robotics-environment-details').textContent)&&/7/.test(get('robotics-environment-details').textContent),'Environment metadata must show the task, robot, and action dimension');
      check(get('robotics-metrics').hidden&&get('robotics-discover').disabled&&!get('robotics-graph').querySelector('[data-node-id]'),'A reset environment must have no failure scores, variations, or enabled discovery');
      check(!/episodes/.test(get('robotics-step-status').textContent),'Environment progress must not invent baseline episodes');
      get('robotics-mapping').value='{}';get('robotics-episodes').value='2';
      change('robotics-seed','43');
      check(!get('robotics-environment-stale').hidden,'Changing the seed must mark the preview as stale');
      change('robotics-seed','42');
      check(get('robotics-environment-stale').hidden,'An exact environment snapshot must recognize restored settings');
      passed.push('policy-free environment loading and preview');

      change('robotics-task','__custom__');get('robotics-custom-task').value='';
      const beforeInvalid=requests.filter(request=>request.body?.mode==='environment').length;
      get('robotics-load-environment').click();await flush();
      check(requests.filter(request=>request.body?.mode==='environment').length===beforeInvalid&&!get('robotics-error').hidden,'Missing custom environment names must fail without submitting');
      change('robotics-task','Lift');failEnvironment=true;
      get('robotics-load-environment').click();await flush();
      await until(()=>/could not render/.test(get('robotics-error').textContent),'Environment errors did not appear');
      check(get('robotics-environment-preview').hidden&&get('robotics-discover').disabled&&get('robotics-metrics').hidden,'Failed loading must not display an earlier environment as newly loaded or produce measurements');
      failEnvironment=false;holdEnvironment=true;change('robotics-policy-source','upload');
      check(!get('robotics-load-environment').disabled&&get('robotics-analyze').disabled,'Missing upload files must only prevent policy evaluation');
      get('robotics-load-environment').click();await flush();
      document.querySelector('.domain[data-task="classification"]').click();await flush();nav.click();await flush();
      get('robotics-cancel').click();await flush();
      check(requests.some(request=>request.url.endsWith('/environment-3/cancel'))&&/cancelled/i.test(get('robotics-status').textContent),'Environment loading must remain cancellable across workspace navigation');
      check(get('robotics-environment-preview').hidden,'Cancelled environment loads must not appear ready');
      holdEnvironment=false;
      passed.push('environment validation, errors and cancellation');

      check([...get('robotics-starter-policy').options].some(option=>option.value==='lift'),'The initial catalog must populate the starter selector without selecting a policy');
      get('robotics-refresh-starters').click();await flush();
      check(/no .*starter/i.test(get('robotics-starter-status').textContent)&&!get('robotics-refresh-starters').disabled,'An empty starter catalog must be explicit and refreshable');
      starterCatalog=starters;get('robotics-refresh-starters').click();await flush();
      const beforeStarter=requests.filter(request=>request.url==='/api/robotics/runs').length;
      change('robotics-starter-policy','pickplace');
      check(get('robotics-task').value==='__custom__'&&get('robotics-custom-task').value==='PickPlaceCan','A starter for an installed custom task must fill its exact task name');
      check(get('robotics-policy-format').value==='torchscript'&&get('robotics-policy-source').value==='local'&&get('robotics-policy-name').value===starters[1].policy.name,'Starter selection must choose its checkpoint format, source, and absolute path');
      check(get('robotics-robot').value==='Panda'&&get('robotics-robots').value==='Panda'&&get('robotics-controller').value==='OSC_POSE'&&get('robotics-frequency').value==='10'&&get('robotics-horizon').value==='240','Starter selection must apply the complete simulator contract');
      check(get('robotics-mapping').value.includes('object-state')&&JSON.parse(get('robotics-mapping').value).state_keys[0]==='object-state'&&get('robotics-instruction').value===starters[1].instruction,'Starter selection must preserve observation order and task instruction');
      check(/1\s*\/\s*5/.test(get('robotics-starter-details').textContent)&&/20%/.test(get('robotics-starter-details').textContent)&&/Behavior cloning/.test(get('robotics-starter-details').textContent),'Starter details must display measured counts and training method');
      check(requests.filter(request=>request.url==='/api/robotics/runs').length===beforeStarter&&get('robotics-discover').disabled&&get('robotics-environment-preview').hidden,'Choosing a starter must not silently load its environment or evaluate it');
      change('robotics-horizon','250');
      check(/edited|changed/i.test(get('robotics-starter-details').textContent),'Edited starter contracts must not imply the recorded performance still matches');
      change('robotics-starter-policy','');change('robotics-task','Lift');change('robotics-frequency','20');change('robotics-horizon','300');
      get('robotics-policy-name').value='';get('robotics-mapping').value='{}';
      passed.push('starter refresh, measured results and exact autofill');

      change('robotics-policy-format','lerobot');change('robotics-policy-source','huggingface');
      get('robotics-policy-name').value='owner/policy';get('robotics-instruction').value='Place the cube precisely.';
      get('robotics-mapping').value='{invalid';
      get('robotics-form').dispatchEvent(new Event('submit',{cancelable:true,bubbles:true}));await flush();
      check(!requests.some(request=>request.body?.mode==='analyze'),'Invalid mapping must block a baseline run');
      check(!get('robotics-error').hidden,'Mapping error must be visible');
      get('robotics-mapping').value='{"observation_map":{"observation.images.front":"agentview_image"},"state_keys":["robot0_joint_pos"]}';
      get('robotics-form').dispatchEvent(new Event('submit',{cancelable:true,bubbles:true}));await flush();
      const submitted=requests.find(request=>request.body?.mode==='analyze').body;
      check(submitted.mode==='analyze'&&submitted.policy.source==='huggingface'&&submitted.policy.format==='lerobot','Analyze must carry the native policy format and source');
      check(submitted.instruction==='Place the cube precisely.','Editable task instruction must reach the adapter');
      check(submitted.mapping.observation_map['observation.images.front']==='agentview_image','Explicit camera mapping must be preserved');
      check(submitted.environment.robots[0]==='Panda'&&submitted.episodes_per_variation===2,'Environment and paired episode settings must reach Analyze');
      check(get('robotics-graph') instanceof HTMLCanvasElement,'Robosuite must reuse the 3D canvas embedding interaction');
      check(robotics.graph.points.length===2&&robotics.graph.nodes.length===0,'Each baseline episode must have one original point, with children initially collapsed');
      check(get('robotics-graph').getBoundingClientRect().height>100,'The populated episode graph must be visible');
      check(/policy|hidden/i.test(get('robotics-coordinate-note').textContent)&&!/environment parameters/i.test(get('robotics-coordinate-note').textContent),'The coordinate note must identify the real feature source');
      check(!get('robotics-discover').disabled,'Completed baseline must enable variation evaluation');
      passed.push('mapping validation and baseline payload');
      check(get('robotics-policy-notes')&&!get('robotics-policy-notes').hidden,'Policy compatibility notes must be surfaced beside policy settings');
      check(!get('robotics-policy-notes').open,'Policy notes must remain compact by default');
      check(get('robotics-policy-notes-list').children.length===2,'Runtime and adapter warnings must be combined and deduplicated');
      check(/VecNormalize/.test(get('robotics-policy-notes-list').textContent)&&/Controller action units/.test(get('robotics-policy-notes-list').textContent),'Both runtime and adapter warning content must remain accessible');
      includePolicyWarnings=false;
      passed.push('deduplicated policy compatibility notes');

      get('robotics-load-environment').click();await flush();
      await until(()=>/Environment loaded/.test(get('robotics-status').textContent),'Environment reload after baseline did not finish');
      check(get('robotics-metrics').hidden&&get('robotics-discover').disabled&&!get('robotics-show-evaluation').hidden,'Environment mode must hide previous failure metrics and offer the preserved evaluation');
      get('robotics-show-evaluation').click();await flush();
      check(get('robotics-baseline-rate').textContent==='50%'&&!get('robotics-metrics').hidden&&!get('robotics-discover').disabled,'Environment loading must preserve the valid baseline and its settings snapshot');
      passed.push('environment preview preserves completed analysis');

      const canvas=get('robotics-graph');
      const graphClick=(kind,id,between)=>{
        const point=robotics.graph.projected.find(item=>item.kind===kind&&(kind==='sample'?item.data.index:item.data.id)===id),box=canvas.getBoundingClientRect();
        check(Boolean(point),`Missing ${kind} ${id}`);
        const options={bubbles:true,button:0,pointerId:7,isPrimary:true,clientX:box.left+point.x,clientY:box.top+point.y};
        canvas.dispatchEvent(new PointerEvent('pointerdown',options));
        if(between)between();
        canvas.dispatchEvent(new PointerEvent('pointerup',options));
      };
      graphClick('sample',0);
      check(robotics.graph.focusIndex===0&&robotics.graph.nodes.length===4,'Clicking an original must expand every action for that episode');
      check(robotics.graph.nodes.every(node=>node.pending&&node.similarity==null),'Untested nodes must use pending layout and no fabricated similarity');
      check(/42/.test(get('robotics-selected-title').textContent)&&Boolean(get('robotics-baseline-media').querySelector('img')),'Selecting an original must inspect its own baseline');
      graphClick('action',0);
      check(/untested/i.test(get('robotics-selected-outcome').textContent)&&!get('robotics-variation-media').querySelector('img,video'),'An untested child must have no claimed outcome or replay');
      const graphBounds=get('robotics-search-space').getBoundingClientRect(),inspectorBounds=get('robotics-inspector').getBoundingClientRect();
      check(innerWidth>1250?inspectorBounds.left>=graphBounds.right:inspectorBounds.top>=graphBounds.bottom,'Inspector must sit to the right on desktop and stack on narrow screens');
      check(get('robotics-finetune').disabled&&get('robotics-finetune').parentElement===get('robotics-discover').parentElement,'Fine-tune must remain disabled beside Find failure');
      check(get('robotics-discover').getBoundingClientRect().top>=graphBounds.bottom,'Discovery buttons must sit below the graph');
      canvas.dispatchEvent(new KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));
      get('robotics-graph-zoom-in').click();
      const camera=()=>JSON.stringify({yaw:robotics.graph.yaw,pitch:robotics.graph.pitch,distance:robotics.graph.distance,pan:robotics.graph.pan,center:robotics.graph.center,scale:robotics.graph.scale});
      const zoomedView=camera();
      const beforeSearch=requests.filter(request=>request.body?.mode==='discover').length;
      get('robotics-discover').click();await flush();
      check(requests.filter(request=>request.body?.mode==='discover').length===beforeSearch&&!get('robotics-search-panel').hidden,'Find failure must open RL settings without starting a job');
      check(get('robotics-search-panel').getBoundingClientRect().top>=get('robotics-search-space').getBoundingClientRect().bottom,'Discovery settings must open below the graph');
      check(get('robotics-search-episodes').value==='200'&&get('robotics-search-rollouts').value==='36','DQN steps and new-rollout budgets must be separate');
      check(get('robotics-search-epsilon-start').value==='1'&&get('robotics-search-epsilon-end').value==='0.1'&&get('robotics-search-learning-rate').value==='0.002','DQN defaults must remain explicit');
      change('robotics-search-episodes','40');change('robotics-search-rollouts','3');change('robotics-search-epsilon-start','0.8');change('robotics-search-epsilon-end','0.2');
      change('robotics-search-learning-rate','0.003');change('robotics-search-batch-size','4');change('robotics-search-replay-capacity','64');change('robotics-search-seed','11');change('robotics-search-criterion','new_failure');
      change('robotics-search-epsilon-end','0.9');get('robotics-start-search').click();await flush();
      check(requests.filter(request=>request.body?.mode==='discover').length===beforeSearch&&!get('robotics-search-error').hidden,'Inverted exploration bounds must fail locally without launching a worker');
      change('robotics-search-epsilon-end','0.2');change('robotics-search-replay-capacity','2');get('robotics-start-search').click();await flush();
      check(requests.filter(request=>request.body?.mode==='discover').length===beforeSearch&&/batch size/i.test(get('robotics-search-error').textContent),'Replay capacity smaller than the batch must fail locally');
      change('robotics-search-replay-capacity','64');get('robotics-start-search').click();await flush();
      const searchRequest=requests.filter(request=>request.body?.mode==='discover').at(-1).body;
      check(searchRequest.analysis_id==='analysis-1'&&JSON.stringify(searchRequest.search)===JSON.stringify({episodes:40,rollout_budget:3,epsilon_start:.8,epsilon_end:.2,learning_rate:.003,batch_size:4,replay_capacity:64,seed:11,criterion:'new_failure'}),'Start discovery must send every edited RL setting');
      check(get('robotics-start-search').disabled,'Starting discovery must lock the search form');
      const initialFrame=get('robotics-live-image').getAttribute('src');
      graphClick('sample',1,()=>robotics.applyJob(structuredClone(discovery)));
      check(robotics.graph.focusIndex===1&&robotics.graph.nodes.length===4,'Clicking an original during a live poll must select it and expand the correct episode');
      graphClick('action',0,()=>robotics.applyJob(structuredClone(discovery)));
      check(robotics.graph.selectedAction===0,'Children must stay clickable while training and polling');
      await until(()=>robotics.job.status==='completed','Discovery results did not appear');
      check(get('robotics-landscape-mode').value==='predicted','Learned Q values must color the landscape automatically after DQN updates');
      await until(()=>robotics.graph.prediction(1,1),'Untested actions must receive estimates for the selected episode');
      const untestedPoint=robotics.graph.projected.find(node=>node.kind==='action'&&node.data.id===1);
      check(robotics.graph.nodeColor(untestedPoint)===robotics.graph.severityColor(robotics.graph.prediction(1,1).q_value),'Untested action fill must reflect its Q value despite pending position');
      check(robotics.graph.nodeColor(robotics.graph.projected.find(node=>node.kind==='sample'&&node.data.index===0))===robotics.graph.severityColor(.6),'Other episode originals must use their mean Q values');
      check(camera()===zoomedView&&robotics.graph.focusIndex===1&&robotics.graph.selectedAction===0,'Polling and measured expansion must preserve camera and selection');
      check(robotics.graph.nodes.find(node=>node.id===0).position[0]===.5&&!robotics.graph.nodes.find(node=>node.id===0).pending,'Evaluated children must use measured coordinates for the selected episode');
      check(robotics.graph.nodes.find(node=>node.id===1).pending,'A measurement on another action cannot populate an untested child');
      check(get('robotics-failure-rate').textContent==='100%','Baseline reuses and rollout errors must not change observed variation failure rates');
      check(/17/.test(get('robotics-search-steps').textContent)&&/3/.test(get('robotics-search-model-calls').textContent)&&/14/.test(get('robotics-search-cache-hits').textContent),'RL steps, simulator rollouts and cache hits must use real search progress');
      check(/0.4/.test(get('robotics-search-epsilon').textContent)&&/0.125/.test(get('robotics-search-loss').textContent)&&/7/.test(get('robotics-search-updates').textContent),'Epsilon, loss and optimizer updates must be shown');
      check(get('robotics-live-image').getAttribute('src')!==initialFrame,'Changing episode progress must refresh an artifact with a stable filename');
      check(/failure/i.test(get('robotics-selected-outcome').textContent)&&/43/.test(get('robotics-selected-title').textContent),'The inspector must report the selected episode/action outcome');
      check(Boolean(get('robotics-variation-media').querySelector('img[src*="variation-0.gif"]')),'A child must show its exact episode replay');
      check(!get('robotics-baseline-media').querySelector('img, video')&&/43/.test(get('robotics-baseline-outcome').textContent),'The selected baseline must use the same seed, even when its replay is unavailable');
      change('robotics-episode','0');change('robotics-inspect','0');
      check(!get('robotics-variation-media').querySelector('img,video'),'A child without media must never borrow another seed’s replay or a node aggregate replay');
      check(/Feature projection unavailable/.test(get('robotics-selected-note').textContent)&&/failure/i.test(get('robotics-selected-outcome').textContent),'A missing projection must explain the warning without discarding the observed task outcome');
      check(Boolean(get('robotics-baseline-media').querySelector('img[src*="baseline-seed-42"]')),'Changing episodes must restore only that episode’s baseline');
      change('robotics-inspect','2');check(/error/i.test(get('robotics-selected-outcome').textContent),'Rollout errors must be distinct from failures');
      change('robotics-inspect','3');check(/reused/i.test(get('robotics-selected-note').textContent),'Neutral baseline reuse must be explicit');
      const pickerIds=[...get('robotics-inspect').options].map(option=>option.value).filter(Boolean);
      check(pickerIds[0]==='0','Measured failures must sort above successful, untested and errored variations');
      predictionsUnavailable=true;change('robotics-landscape-mode','predicted');await flush();
      check(/not available/i.test(get('robotics-landscape-note').textContent),'Publishing an unavailable prediction snapshot must remain explicit');
      const predictionCalls=requests.filter(request=>request.url.includes('/predictions?')).length;
      robotics.applyJob(structuredClone(discovery));await flush();
      check(requests.filter(request=>request.url.includes('/predictions?')).length===predictionCalls+1&&robotics.graph.prediction(0,0)?.q_value===.9,'A later poll must retry a temporarily missing prediction snapshot without requiring a policy update');
      check(robotics.graph.nodeColor(robotics.graph.projected.find(node=>node.kind==='action'&&node.data.id===2))==='#9a72bc','A rollout error must remain visibly distinct when estimated colors are selected');
      check(requests.some(request=>request.url.includes('/predictions?input_index=0'))&&/estimate|estimated/i.test(get('robotics-landscape-note').textContent),'Predicted colors must fetch the selected input and be labeled estimates');
      check(get('robotics-observed-legend')?.hidden&&get('robotics-estimate-legend')&&!get('robotics-estimate-legend').hidden,'Estimated colors must have their own legend without observed-outcome claims');
      change('robotics-landscape-mode','measured');
      check(!get('robotics-observed-legend').hidden&&get('robotics-estimate-legend').hidden,'Switching back to observed colors must restore their legend');
      robotics.applyJob(structuredClone(discovery));await flush();
      check(get('robotics-landscape-mode').value==='measured','Polling must preserve an explicit observed-color choice');
      check(get('robotics-artifacts').querySelector('a')?.target==='_blank','Reports must open in a new tab');
      get('robotics-graph-fit').click();check(camera()!==zoomedView,'Fit must frame the complete current expansion');
      passed.push('episode canvas, live selection, exact replay and DQN controls');

      get('robotics-discover').click();await flush();get('robotics-start-search').click();await flush();get('robotics-cancel').click();await flush();
      check(requests.some(request=>request.url.endsWith('/discovery-2/cancel')),'Cancel must target the active worker');
      check(!get('robotics-status').textContent.toLowerCase().includes('running'),'Cancellation must leave a terminal status');
      document.querySelector('.domain[data-task="classification"]').click();await flush();
      check(!get('workspace-layout').hidden&&get('robotics-workspace').hidden,'Returning to classification must restore its panel');
      check(get('model-name').value==='/preserved-model','Visiting Robosuite must preserve the existing form');
      document.querySelector('.domain[data-task="summarization"]').click();await flush();
      document.querySelector('.domain[data-task="generation"]').click();await flush();
      check(get('domain-title').textContent==='Image generation','Existing domains must still switch normally');
      nav.click();await flush();
      passed.push('cancellation and existing-domain navigation');

      change('robotics-policy-format','torchscript');change('robotics-policy-source','upload');
      const file=new File([new Uint8Array(8388615)],'policy.pt');
      const transfer=new DataTransfer();transfer.items.add(file);get('robotics-policy-file').files=transfer.files;
      get('robotics-policy-file').dispatchEvent(new Event('change',{bubbles:true}));
      await until(()=>/ready/i.test(get('robotics-upload-status').textContent),'Chunked upload did not finish');
      const chunks=requests.filter(request=>request.url.includes('/api/robotics/policies/upload-1/files/'));
      check(chunks.length===2&&chunks[0].body.size===8388608&&chunks[1].body.size===7,'Large uploads must stream bounded chunks');
      check(chunks[1].url.endsWith('offset=8388608'),'Chunk offsets must match the file position');
      passed.push('bounded checkpoint uploads');

      change('robotics-policy-format','sb3');change('robotics-policy-source','upload');
      const checkpointFiles=new DataTransfer();checkpointFiles.items.add(new File(['metadata'],'policy.metadata.json'));checkpointFiles.items.add(new File(['weights'],'policy.zip'));checkpointFiles.items.add(new File(['stats'],'policy.vecnormalize.pkl'));
      get('robotics-policy-file').files=checkpointFiles.files;get('robotics-policy-file').dispatchEvent(new Event('change',{bubbles:true}));
      await until(()=>/ready/i.test(get('robotics-upload-status').textContent),'Checkpoint sidecar upload did not finish');
      const checkpointManifest=requests.filter(request=>request.url==='/api/robotics/policies').at(-1).body;
      check(checkpointManifest.files.length===3&&checkpointManifest.primary==='policy.zip','Optional statistics files must preserve the native primary checkpoint');
      passed.push('checkpoint metadata and normalization sidecars');

      change('robotics-policy-format','lerobot');change('robotics-policy-source','upload');
      const folderFiles=new DataTransfer();
      for(const [name,contents] of [['saved-policy/config.json','{}'],['saved-policy/model.safetensors','weights'],['saved-policy/processors/normalize.json','{}']]) {
        const entry=new File([contents],name.split('/').at(-1));Object.defineProperty(entry,'webkitRelativePath',{value:name});folderFiles.items.add(entry);
      }
      get('robotics-policy-folder').files=folderFiles.files;get('robotics-policy-folder').dispatchEvent(new Event('change',{bubbles:true}));
      await until(()=>/ready/i.test(get('robotics-upload-status').textContent),'Policy folder upload did not finish');
      const folderManifest=requests.filter(request=>request.url==='/api/robotics/policies').at(-1).body;
      check(JSON.stringify(folderManifest.files.map(file=>file.path))==='["config.json","model.safetensors","processors/normalize.json"]','Folder manifests must retain subdirectories and remove the selected root');
      get('robotics-form').dispatchEvent(new Event('submit',{cancelable:true,bubbles:true}));await flush();
      const uploadedRun=requests.filter(request=>request.url==='/api/robotics/runs').at(-1).body;
      check(uploadedRun.policy.upload_id==='upload-1'&&uploadedRun.policy.source==='upload','Analyze must reference the finalized upload');
      check(get('robotics-policy-notes').hidden&&get('robotics-policy-notes-list').children.length===0,'A new job without warnings must clear the previous policy notes');
      get('robotics-instruction').value='A changed instruction';get('robotics-instruction').dispatchEvent(new Event('input',{bubbles:true}));
      check(get('robotics-discover').disabled,'Changing policy inputs must invalidate the matched baseline for discovery');
      change('robotics-task','Stack');check(get('robotics-instruction').value==='Stack the cubes.','Changing the task must suggest its task instruction');
      passed.push('LeRobot directory manifest and stale-baseline guard');
      failBaseline=true;get('robotics-form').dispatchEvent(new Event('submit',{cancelable:true,bubbles:true}));await flush();
      check(!get('robotics-baseline-errors').hidden&&get('robotics-baseline-errors').open&&/Missing camera/.test(get('robotics-baseline-error-list').textContent),'Baseline compatibility errors must be visible without selecting an untested variation');
      check(get('robotics-discover').disabled&&get('robotics-baseline-rate').textContent==='—','An errored baseline must not become a measured failure or enable discovery');
      passed.push('baseline error evidence');
      change('robotics-task','TwoArmLift');
      check(get('robotics-robots').value==='Panda, Panda'&&!get('robotics-robots-fields').hidden,'Two-arm tasks must visibly configure two single-arm robots');
      change('robotics-robot','Sawyer');check(get('robotics-robots').value==='Sawyer, Sawyer','Robot changes must preserve two-arm task compatibility');
      change('robotics-robot','Baxter');check(get('robotics-robots').value==='Baxter','A bimanual Baxter must not be duplicated');
      change('robotics-task','__custom__');get('robotics-robots').value='Panda, Sawyer';change('robotics-controller','JOINT_POSITION');
      check(get('robotics-robots').value==='Panda, Sawyer','Custom task robot configurations must remain manual');
      passed.push('two-arm task robot selection');

      failBaseline=false;change('robotics-starter-policy','lift');
      check(get('robotics-task').value==='Lift'&&get('robotics-robots-fields').hidden&&get('robotics-horizon').value==='180','A starter must replace a previous custom two-robot configuration');
      check(get('robotics-discover').disabled&&!get('robotics-stale').hidden,'Choosing another starter must invalidate the previous policy baseline');
      holdAnalysisStart=true;
      get('robotics-form').dispatchEvent(new Event('submit',{cancelable:true,bubbles:true}));await flush();
      change('robotics-horizon','181');releaseAnalysisStart();await flush();holdAnalysisStart=false;
      check(analysis.config.environment.horizon===180&&get('robotics-discover').disabled&&!get('robotics-stale').hidden,'A delayed baseline response must retain its submission snapshot when settings change');
      holdEnvironmentStart=true;
      get('robotics-load-environment').click();await flush();change('robotics-seed','99');releaseEnvironmentStart();await flush();holdEnvironmentStart=false;
      await until(()=>/Environment loaded/.test(get('robotics-status').textContent),'Delayed environment response did not finish');
      check(!get('robotics-environment-stale').hidden&&/seed 42/.test(get('robotics-environment-details').textContent),'A delayed environment response must report its original seed and remain stale against newer settings');
      get('robotics-show-evaluation').click();await flush();
      check(get('robotics-discover').disabled&&!get('robotics-stale').hidden,'Restoring an evaluation must preserve its stale-settings guard');
      passed.push('starter and asynchronous request snapshot guards');

      change('robotics-starter-policy','lift');
      get('robotics-form').dispatchEvent(new Event('submit',{cancelable:true,bubbles:true}));await flush();
      const priorStarterAnalysis=robotics.analysis,priorStarterRevision=robotics.revision;
      check(!get('robotics-discover').disabled,'The original starter baseline must be available before catalog refresh');
      const updatedMetrics={...starters[0],label:'Lift with updated measurements',description:'Updated measurements for the same checkpoint.',evaluation:{successes:4,episodes:5,success_rate:.8}};
      starterCatalog=[updatedMetrics,starters[1]];get('robotics-refresh-starters').click();await flush();
      check(/4\s*\/\s*5/.test(get('robotics-starter-details').textContent)&&!/3\s*\/\s*5/.test(get('robotics-starter-details').textContent),'Refreshing a selected starter must replace its displayed measured results');
      check(robotics.selectedStarter.evaluation.successes===4&&get('robotics-starter-policy').selectedOptions[0].textContent===updatedMetrics.label,'The selected starter must use the refreshed catalog entry');
      check(robotics.analysis===priorStarterAnalysis&&robotics.revision===priorStarterRevision&&!get('robotics-discover').disabled,'Measurement-only catalog changes must preserve a matching baseline');
      const samePathCheckpoint={...updatedMetrics,checkpoint_sha256:'c'.repeat(64),evaluation:{successes:1,episodes:5,success_rate:.2}};
      starterCatalog=[samePathCheckpoint,starters[1]];get('robotics-refresh-starters').click();await flush();
      check(robotics.analysis===priorStarterAnalysis&&get('robotics-baseline-rate').textContent==='50%'&&get('robotics-discover').disabled,'Replacing checkpoint contents at the same path must retain the old evidence and invalidate its baseline');
      check(!get('robotics-stale').hidden&&/prior|previous|earlier/i.test(get('robotics-stale').textContent),'An updated starter must explicitly identify the preserved evaluation as old');
      check(/1\s*\/\s*5/.test(get('robotics-starter-details').textContent)&&/updated|changed/i.test(get('robotics-starter-details').textContent),'The refreshed checkpoint must show its current results and an update notice');
      check(get('robotics-horizon').value==='180'&&get('robotics-policy-name').value===starters[0].policy.name,'Refreshing checkpoint identity must not silently rewrite form settings');
      change('robotics-horizon','190');change('robotics-instruction','My edited task instruction');
      get('robotics-mapping').value='{"state_keys":["robot0_eef_pos"],"clip_actions":false}';get('robotics-mapping').dispatchEvent(new Event('input',{bubbles:true}));
      get('robotics-form').dispatchEvent(new Event('submit',{cancelable:true,bubbles:true}));await flush();
      const editedAnalysis=robotics.analysis,editedValues=['task','robots','controller','horizon','frequency','instruction','mapping','policy-name'].map(id=>get('robotics-'+id).value);
      const changedContract={...samePathCheckpoint,environment:{...samePathCheckpoint.environment,task:'Stack',horizon:220},mapping:{state_keys:['object-state','robot0_eef_pos']},instruction:'Stack the cubes.',evaluation:{successes:2,episodes:5,success_rate:.4}};
      starterCatalog=[changedContract,starters[1]];get('robotics-refresh-starters').click();await flush();
      check(JSON.stringify(editedValues)===JSON.stringify(['task','robots','controller','horizon','frequency','instruction','mapping','policy-name'].map(id=>get('robotics-'+id).value)),'Refreshed starter contracts must preserve every user-edited field');
      check(robotics.analysis===editedAnalysis&&get('robotics-discover').disabled&&/2\s*\/\s*5/.test(get('robotics-starter-details').textContent),'Changed simulator or mapping contracts must update recorded results and invalidate the prior baseline');
      check(/current|refreshed/i.test(get('robotics-starter-details').textContent)&&/differ|edited|changed/i.test(get('robotics-starter-details').textContent),'Preserved inputs must be distinguished from the refreshed starter configuration');
      get('robotics-form').dispatchEvent(new Event('submit',{cancelable:true,bubbles:true}));await flush();
      const beforeRemovalAnalysis=robotics.analysis;
      check(!get('robotics-discover').disabled,'A fresh baseline must be available before removing its selected starter');
      starterCatalog=[starters[1]];get('robotics-refresh-starters').click();await flush();
      check(get('robotics-starter-policy').value===''&&get('robotics-starter-details').hidden&&get('robotics-starter-details').textContent===''&&/no longer available/i.test(get('robotics-starter-status').textContent),'Removed starters must clear the previous displayed results and explain their absence');
      check(robotics.analysis===beforeRemovalAnalysis&&get('robotics-discover').disabled&&!get('robotics-stale').hidden,'Removing the selected starter must invalidate a previously matching baseline and retain its evidence');
      check(JSON.stringify(editedValues)===JSON.stringify(['task','robots','controller','horizon','frequency','instruction','mapping','policy-name'].map(id=>get('robotics-'+id).value)),'Removing a starter must preserve custom inputs');
      starterCatalog=starters;get('robotics-refresh-starters').click();await flush();change('robotics-starter-policy','lift');
      passed.push('refreshed starter identity, results and custom inputs');

      delayCancels=true;holdEnvironment=true;
      get('robotics-load-environment').click();await flush();
      const pendingCancelId=robotics.job.id,pendingCancel=robotics.cancel();await flush();
      holdEnvironment=false;await robotics.poll(robotics.runGeneration);
      check(robotics.job.status==='completed'&&robotics.busy,'Finishing a poll must not unlock job controls while cancellation is pending');
      const beforePendingStart=requests.filter(request=>request.url==='/api/robotics/runs').length;
      get('robotics-load-environment').click();get('robotics-analyze').click();await flush();
      check(requests.filter(request=>request.url==='/api/robotics/runs').length===beforePendingStart,'Pending cancellation must block new runs after worker completion');
      pendingCancels.get(pendingCancelId).resolve();await pendingCancel;
      check(!robotics.busy&&!robotics.cancelling,'A current cancellation must release the controls when it settles');
      for(const outcome of ['resolve','reject']) {
        holdEnvironment=true;get('robotics-load-environment').click();await flush();
        const oldId=robotics.job.id,oldCancel=robotics.cancel();await flush();
        holdEnvironment=false;await robotics.poll(robotics.runGeneration);
        // Supersede the completed job through the controller entrypoint to test
        // stale callbacks independently of the UI's pending-cancel interlock.
        holdEnvironment=true;await robotics.start(robotics.environmentConfig());
        const latestId=robotics.job.id,latestGeneration=robotics.runGeneration,latestCancel=robotics.cancel();await flush();
        robotics.error('Current job evidence');
        pendingCancels.get(oldId)[outcome]();await oldCancel;
        check(robotics.job.id===latestId&&robotics.runGeneration===latestGeneration,'A delayed old cancellation must not replace a newer job or invalidate its polling generation');
        check(robotics.cancelling&&get('robotics-error').textContent==='Current job evidence','Stale cancellation errors and cleanup must not overwrite newer errors or cancellation state');
        await robotics.poll(latestGeneration);
        check(robotics.job.id===latestId&&robotics.job.status==='running','The new worker must still accept polling after an old cancellation settles');
        pendingCancels.get(latestId).resolve();await latestCancel;
        check(!robotics.cancelling&&!robotics.busy,'The latest cancellation must still settle normally');
      }
      delayCancels=false;holdEnvironment=false;robotics.error();
      passed.push('delayed cancellation and stale response guards');

      catalog.runtime={available:false,message:'Test runtime unavailable'};get('robotics-runtime-retry').click();await flush();
      check(get('robotics-load-environment').disabled&&get('robotics-analyze').disabled,'An unavailable runtime must disable environment loading and evaluation');
      const unavailableCount=requests.filter(request=>request.body?.mode==='environment').length;
      get('robotics-load-environment').click();await flush();
      check(requests.filter(request=>request.body?.mode==='environment').length===unavailableCount,'Disabled environment loading must not submit jobs');
      catalog.runtime={available:true,message:'Test runtime available'};get('robotics-runtime-retry').click();await flush();
      check(!get('robotics-load-environment').disabled,'Refreshing an available runtime must restore environment loading');
      passed.push('runtime availability guards');
      check(document.documentElement.scrollWidth<=window.innerWidth,'The workspace must fit the viewport without horizontal page overflow');
      passed.push('responsive viewport containment');
      window.scrollTo(0,0);
      window.roboticsUITestResult={passed};
    }catch(error){window.roboticsUITestResult={passed,error:error.stack};}
    const output=document.createElement('pre');output.id='robotics-test-result';output.style.whiteSpace='pre-wrap';output.style.overflowWrap='anywhere';output.textContent=JSON.stringify(window.roboticsUITestResult);document.body.append(output);
    return window.roboticsUITestResult;
  };
})();
}
