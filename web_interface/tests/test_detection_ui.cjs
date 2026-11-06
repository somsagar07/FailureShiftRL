/* Run with node web_interface/tests/test_detection_ui.cjs. */
'use strict';
if (typeof window === 'undefined') {
  const fs=require('node:fs'),path=require('node:path'),os=require('node:os'),http=require('node:http'),{execFile}=require('node:child_process');
  const root=path.resolve(__dirname,'..'),profile=fs.mkdtempSync(path.join(os.tmpdir(),'detection-ui-'));
  const server=http.createServer((req,res)=>{
    if(req.url==='/'){const html=fs.readFileSync(path.join(root,'static/index.html'),'utf8'),sidebar=html.match(/<aside id="workspace-sidebar"[\s\S]*?<\/aside>/)[0];res.setHeader('Content-Type','text/html');res.end('<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="/style.css"><link rel="stylesheet" href="/detection.css"><body data-sidebar-collapsed="false">'+sidebar+'<main><section id="detection-workspace"></section></main><script src="/embedding-view.js"></script><script src="/detection-controller.js"></script><script src="/test.js"></script><script>runDetectionTests()</script>');return;}
    const file=req.url==='/test.js'?__filename:path.join(root,'static',path.basename(req.url));
    if(!fs.existsSync(file)){res.writeHead(404);res.end();return;}
    res.setHeader('Content-Type',file.endsWith('.css')?'text/css':'text/javascript');res.end(fs.readFileSync(file));
  });
  server.listen(0,'127.0.0.1',()=>execFile(process.env.CHROME_BIN||'google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage',`--window-size=${process.env.TEST_VIEWPORT||'1920,1080'}`,`--user-data-dir=${profile}`,'--dump-dom','--virtual-time-budget=10000',`http://127.0.0.1:${server.address().port}/`],{timeout:45000},(error,stdout)=>{
    try{if(error)throw error;const match=stdout.match(/<pre id="detection-test-result"[^>]*>([\s\S]*?)<\/pre>/);if(!match)throw Error('No browser result');const report=JSON.parse(match[1].replaceAll('&quot;','"').replaceAll('&lt;','<').replaceAll('&gt;','>').replaceAll('&amp;','&'));console.log(JSON.stringify(report,null,2));if(report.error)process.exitCode=1;}
    catch(failure){console.error(failure);process.exitCode=1;}finally{server.close();fs.rmSync(profile,{recursive:true,force:true});}
  }));
} else window.runDetectionTests=async()=>{
  const passed=[],calls=[];const assert=(v,m)=>{if(!v)throw Error(m);};
  const $=id=>document.getElementById('detection-'+id);
  const api=async(path,body)=>{calls.push({path,body});
    if(path==='/api/datasets')return {id:'data',status:'loading'};
    if(path==='/api/datasets/data')return {id:'data',status:'ready',count:2};
    if(path==='/api/detection/analyze')return {id:'analysis',status:'completed',points:[{index:0,name:'<img src=x>',x:0,y:0},{index:1,name:'second',x:1,y:1}]};
    if(path==='/api/detection/discover')return {id:'search',status:'completed',points:[{index:0,name:'<img src=x>',x:0,y:0},{index:1,name:'second',x:1,y:1}],actions:[{id:1,darkness:.1,saturation:.5}],records:[{input_index:1,action_id:1,severity:.7,reward:.7,baseline:[{label:'cat',score:.9}],output:[{label:'cat',score:.2}]}]};
    if(path==='/api/detection/expand')return {id:'expand',status:'completed',input_index:body.input_index,total:1,completed:1,nodes:[{id:1,position:[.5,.2,0],action:{darkness:.1,saturation:.5}}]};
    if(path==='/api/detection/inspect')return {id:'inspection',status:'completed',action_id:body.action_id,baseline:{detections:[{label:'cat',score:.9}]},output:body.action_id===null?null:{detections:[{label:'cat',score:.2}]},metrics:{severity:.7}};
    throw Error('Unexpected request '+path);
  };
  try{
    const c=new DetectionController({api});c.enter();
    for(const collapsed of ['false','true']){
      document.body.dataset.sidebarCollapsed=collapsed;
      const sidebar=document.getElementById('workspace-sidebar'),inspector=document.querySelector('.detection-inspector');
      assert(getComputedStyle(inspector).position!=='fixed','Inspector must stay in its workspace, not cover the sidebar');
      for(const button of sidebar.querySelectorAll('.domain')){
        const box=button.getBoundingClientRect(),hit=document.elementFromPoint(box.x+box.width/2,box.y+box.height/2);
        assert(hit&&(button===hit||button.contains(hit)),`Workspace ${button.dataset.task} remains visible and clickable`);
      }
      assert(inspector.getBoundingClientRect().x>=sidebar.getBoundingClientRect().right||getComputedStyle(sidebar).position==='static','Inspector stays beside or below navigation');
      assert(document.documentElement.scrollWidth<=innerWidth+1,'Workspace fits the viewport');
    }
    document.body.dataset.sidebarCollapsed='false';passed.push('Sidebar remains accessible with an expanded or collapsed workspace bar');
    $('folder-path').value='/tmp/images';await c.loadDataset();
    assert(calls[0].body.source==='folder'&&calls[0].body.path==='/tmp/images','Folder request');assert(calls.some(c=>c.path==='/api/datasets/data'),'Dataset status polled');
    await c.analyze();const a=calls.find(c=>c.path==='/api/detection/analyze').body;
    assert(a.dataset_id==='data'&&a.model.name==='facebook/detr-resnet-50'&&a.device==='auto','Analysis uses loaded dataset and model');
    assert(!$('find-failure').disabled&&$('discovery-panel').hidden,'Search settings stay collapsed until Find failure is clicked');
    $('find-failure').click();assert(!$('discovery-panel').hidden,'Find failure opens search settings below the graph');
    assert(c.view.points.length===2,'Analysis points available in interactive graph');
    await c.select(1);assert(c.view.nodes.length===1,'Selecting an image expands action nodes');assert($('original').getAttribute('src').includes('/inspection/images/1/original'),'Selected image shows original detections');assert($('details').textContent.includes('cat'),'Predicted labels shown');
    await c.selectAction(1);assert($('transformed').getAttribute('src').includes('/inspection/images/1/action-1'),'Selecting a node displays its transformed image');
    $('saturation').checked=false;await c.discover();const d=calls.find(c=>c.path==='/api/detection/discover').body;
    assert(d.analysis_id==='analysis'&&d.actions.join(',')==='darkness'&&Number.isInteger(d.episodes)&&Number.isInteger(d.seed),'Discovery payload uses selected groups');
    assert($('records').textContent.includes('0.7'),'Measured outcome shown');await c.select(1,c.discovery.records[0]);await new Promise(r=>setTimeout(r,10));
    assert($('transformed').getAttribute('src').includes('/inspection/images/1/action-1'),'Measured record selects its transformation');assert($('details').textContent.includes('cat'),'Box prediction details shown');
    assert($('fine-tune').disabled,'Fine tune remains disabled');passed.push('Dataset, analysis, selection and measured discovery flow');
    $('source').value='huggingface';$('source').dispatchEvent(new Event('change'));
    assert($('dataset-preset').options.length>=4,'Preset datasets and custom choice available');
    await c.loadDataset();let hf=calls.filter(call=>call.path==='/api/datasets').at(-1).body;
    assert(hf.name==='detection-datasets/coco'&&hf.split==='val'&&hf.subset==='20'&&hf.column==='image','COCO preset sets a bounded validation sample');
    $('dataset-preset').value='cvdl/oxford-pets';$('dataset-preset').dispatchEvent(new Event('change'));await c.loadDataset();
    hf=calls.filter(call=>call.path==='/api/datasets').at(-1).body;assert(hf.name==='cvdl/oxford-pets'&&hf.column==='img'&&hf.split==='test','Pets preset chooses image rather than mask');
    $('dataset-preset').value='other';$('dataset-preset').dispatchEvent(new Event('change'));$('dataset-name').value='owner/data';$('split').value='validation';$('subset').value='[0:100]';$('column').value='image';await c.loadDataset();
    hf=calls.filter(call=>call.path==='/api/datasets').at(-1).body;assert(hf.source==='huggingface'&&hf.name==='owner/data'&&hf.split==='validation'&&hf.subset==='[0:100]'&&hf.column==='image','Hugging Face settings passed through');assert(!c.analysis&&$('discover').disabled,'Changing datasets clears old results');passed.push('Hugging Face presets and custom configuration');
    c.analysis={id:'unfinished',points:[{index:0,name:'stale',x:0,y:0}]};c.busy=true;c.applyJob('analysis',{status:'failed',error:'Model unavailable'});assert(!c.analysis&&$('discover').disabled&&$('error').textContent.includes('Model unavailable'),'Failed analysis cannot be used for discovery');passed.push('Failed analysis is cleared and reports its error');
    c.leave();
  }catch(e){passed.push({error:e.message,stack:e.stack});}
  const pre=document.createElement('pre');pre.id='detection-test-result';pre.textContent=JSON.stringify({passed,error:passed.find(x=>x.error)});document.body.append(pre);
};
