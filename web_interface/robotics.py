"""Local robotics jobs. Simulator packages are imported only by owned workers."""
import copy
import html
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from robotics_search import normalize_search

ROOT = Path(__file__).resolve().parent
CHUNK_BYTES = 8 * 1024 * 1024
MAX_ACTIONS = 5000
FORMATS = ('robomimic', 'torchscript', 'sb3', 'lerobot')
CONTROLLERS = ['OSC_POSE', 'OSC_POSITION', 'JOINT_POSITION', 'JOINT_VELOCITY']
GROUPS = [
    dict(id='friction_scale', label='Contact friction', values=[.5, 1, 1.5], neutral=1, default=True,
         description='Scale scene contact friction, including the object and support surface.'),
    dict(id='mass_scale', label='Object mass', values=[.5, 1, 1.5], neutral=1, default=True,
         description='Scale task-object mass and inertia; robot dynamics stay fixed.'),
    dict(id='object_x_offset', label='Object X position', values=[-.03, 0, .03], neutral=0, default=True,
         description='Shift task objects by metres along world X.'),
    dict(id='object_y_offset', label='Object Y position', values=[-.03, 0, .03], neutral=0, default=False,
         description='Shift task objects by metres along world Y.'),
    dict(id='object_yaw', label='Object yaw', values=[-15, 0, 15], neutral=0, default=False,
         description='Rotate task objects in degrees about world Z.'),
    dict(id='gravity_scale', label='Gravity', values=[.8, 1, 1.2], neutral=1, default=False,
         description='Scale the gravity vector.'),
    dict(id='lighting_scale', label='Lighting', values=[.5, 1, 1.5], neutral=1, default=False,
         description='Scale scene lighting; relevant to camera-based policies.'),
    dict(id='camera_offset', label='Camera position', values=[-.03, 0, .03], neutral=0, default=False,
         description='Shift policy cameras along world X, in metres.'),
    dict(id='observation_noise', label='State noise', values=[0, .01, .03], neutral=0, default=False,
         description='Gaussian noise on floating-point state observations in their native units.'),
    dict(id='action_delay', label='Control delay', values=[0, 1, 3], neutral=0, default=False,
         description='Delay policy commands by this many control steps; hold zero initially.'),
]
DEFAULT_SPACE = {g['id']:g['values'] for g in GROUPS if g['default']}
TASK_NAMES = ['Lift','Stack','NutAssembly','NutAssemblySingle','NutAssemblySquare','NutAssemblyRound',
              'PickPlace','PickPlaceSingle','PickPlaceMilk','PickPlaceBread','PickPlaceCereal',
              'PickPlaceCan','Door','Wipe','ToolHang','TwoArmLift','TwoArmHandover','TwoArmPegInHole','TwoArmTransport']
INSTRUCTIONS = {'Lift':'Lift the cube off the table.', 'Stack':'Stack the red cube on the green cube.',
                'Door':'Open the door.', 'Wipe':'Wipe the marked area of the table.',
                'ToolHang':'Assemble the frame and hang the tool on it.'}
ARTIFACT = re.compile(r'(?:live\.jpg|(?:baseline|action-\d+)-seed-\d+\.(?:jpg|mp4|gif)|report\.html|results\.json|dqn-policy\.pt|dqn-history\.jsonl)')


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer from {low} to {high}.')
    return value


def _text(value, name, maximum=1024):
    if not isinstance(value, str) or not value.strip() or len(value)>maximum or any(ord(c)<32 for c in value):
        raise ValueError(f'Enter a valid {name}.')
    return value.strip()


def normalize_environment(env):
    if not isinstance(env, dict): raise ValueError('Environment must be an object.')
    task = _text(env.get('task', 'Lift'), 'Robosuite task', 80)
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', task): raise ValueError('Invalid task name.')
    robots = env.get('robots', ['Panda'])
    if not isinstance(robots, list) or not 1<=len(robots)<=2 or any(
            not isinstance(r,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,79}',r) for r in robots):
        raise ValueError('Select one or two installed robots.')
    controller = env.get('controller', 'OSC_POSE')
    if controller not in CONTROLLERS: raise ValueError('Select an available controller.')
    return copy.deepcopy(dict(task=task, robots=robots, controller=controller,
        control_freq=_integer(env.get('control_freq',20),'Control frequency',1,100),
        horizon=_integer(env.get('horizon',300),'Episode horizon',1,10000)))


def normalize_environment_request(data):
    if not isinstance(data,dict): raise ValueError('Request must be a JSON object.')
    return dict(environment=normalize_environment(data.get('environment',{})),
        seed=_integer(data.get('seed',42),'Seed',0,2**31-1001),mapping={})


def normalize_config(data):
    if not isinstance(data, dict): raise ValueError('Request must be a JSON object.')
    environment=normalize_environment(data.get('environment',{})); task=environment['task']
    space = data.get('action_space', DEFAULT_SPACE)
    if not isinstance(space, dict) or any(key not in {g['id'] for g in GROUPS} for key in space):
        raise ValueError('Choose available environment action groups.')
    normalized = {}
    for group in GROUPS:
        values = space.get(group['id'], [])
        if not isinstance(values, list): raise ValueError('Action values must be lists.')
        if any(type(v) not in (float,int) or not math.isfinite(v) or v not in group['values'] for v in values):
            raise ValueError(f'Choose the listed values for {group["label"]}.')
        if values: normalized[group['id']]=list(dict.fromkeys(values))
    if math.prod(len(v) for v in normalized.values())>MAX_ACTIONS:
        raise ValueError(f'Keep the action space at {MAX_ACTIONS:,} combinations or fewer.')
    policy = data.get('policy')
    if not isinstance(policy,dict) or policy.get('format') not in FORMATS:
        raise ValueError('Select a policy format and checkpoint.')
    source = policy.get('source')
    if source not in ('local','upload','huggingface') or source=='huggingface' and policy['format']!='lerobot':
        raise ValueError('This policy source is not supported by the selected format.')
    policy = copy.deepcopy(policy)
    if source=='upload':
        if not re.fullmatch(r'[0-9a-f]{32}',str(policy.get('upload_id',''))):
            raise ValueError('Upload the policy first.')
    else:
        policy['name']=_text(policy.get('name'), 'policy path or repository',4096)
    if policy.get('revision'):
        policy['revision']=_text(policy['revision'],'repository revision',256)
    mapping = data.get('mapping', {})
    if not isinstance(mapping,dict): raise ValueError('Policy mapping must be a JSON object.')
    try: mapping_size=len(json.dumps(mapping, allow_nan=False))
    except (ValueError,TypeError): raise ValueError('Policy mapping must contain finite JSON values.') from None
    if mapping_size>65536: raise ValueError('Policy mapping is too large.')
    instruction=data.get('instruction',INSTRUCTIONS.get(task, f'Complete the {task} task.'))
    if not isinstance(instruction,str) or len(instruction)>4000: raise ValueError('Instruction must be at most 4,000 characters.')
    device=data.get('device','auto')
    if device not in ('auto','cpu','cuda'): raise ValueError('Select automatic, CPU, or CUDA policy inference.')
    record_video=data.get('record_video',True)
    if type(record_video) is not bool: raise ValueError('Record video must be true or false.')
    return copy.deepcopy(dict(environment=environment,policy=policy,mapping=mapping,instruction=instruction,
        action_space=normalized, episodes_per_variation=_integer(data.get('episodes_per_variation',3),'Episodes',1,1000),
        variation_budget=_integer(data.get('variation_budget',12),'Variation budget',1,MAX_ACTIONS),
        seed=_integer(data.get('seed',42),'Seed',0,2**31-1001),device=device,record_video=record_video))


def variation_nodes(space):
    keys=list(space)
    nodes=[]
    labels={g['id']:g['label'] for g in GROUPS}
    for index, row in enumerate(itertools.product(*(space[k] for k in keys))):
        values=dict(zip(keys,row))
        # A fixed orthogonal projection of normalized parameters. Not policy embeddings.
        position=[0.,0.,0.]
        for axis,(key,value) in enumerate(values.items()):
            lo,hi=min(space[key]),max(space[key]); x=0 if hi==lo else 2*(value-lo)/(hi-lo)-1
            if axis<3: position[axis]=x
            else:
                for dim in range(3): position[dim]+=x*math.sin((axis+1)*(dim+1)*2.39996)*.3
        nodes.append(dict(id=index,label=' · '.join(f'{labels[k]} {v:g}' for k,v in values.items()) or 'Baseline',
            values=values,position=position,status='untested',success_rate=None,failure_rate=None,episodes=0,errors=0))
    return nodes


def summarize(records, baseline=None):
    good=[r for r in records if not r.get('error') and type(r.get('success')) is bool]
    n=len(good); successes=sum(r['success'] for r in good)
    paired={r['seed']:r['success'] for r in (baseline or []) if not r.get('error') and type(r.get('success')) is bool}
    return dict(episodes=n, errors=len(records)-n, success_rate=successes/n if n else None,
        failure_rate=1-successes/n if n else None,
        regressions=sum(paired.get(r['seed']) is True and r['success'] is False for r in good),
        recoveries=sum(paired.get(r['seed']) is False and r['success'] is True for r in good),
        paired_episodes=sum(r['seed'] in paired for r in good))


class RoboticsManager:
    def __init__(self, directory, *, runner=None):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        self.uploads=self.directory/'policies'; self.uploads.mkdir(exist_ok=True)
        self.lock=threading.RLock(); self.jobs={}; self.active=None; self.processes={}; self.events={}; self.threads={}
        self.runner=runner or self._run_worker; self.injected=runner is not None
        for path in self.directory.glob('*/job.json'):
            try:
                job=json.loads(path.read_text())
                if job['status'] in ('queued','running'):
                    job.update(status='interrupted',message='Server stopped. Analyze again before discovery.')
                self.jobs[job['id']]=job
            except (ValueError,KeyError,OSError): continue

    def catalog(self):
        from setup_robotics import runtime_available
        runtime=runtime_available(); runtime.setdefault('command','.venv/bin/python -E setup_robotics.py --render')
        return dict(tasks=[dict(id=t,label=t,instruction=INSTRUCTIONS.get(t,f'Complete the {t} task.')) for t in TASK_NAMES],
            robots=['Panda','Sawyer','IIWA','UR5e','Jaco','Kinova3','Baxter'],controllers=CONTROLLERS,
            policy_formats=[dict(id=f,label={'robomimic':'Robomimic','torchscript':'TorchScript','sb3':'Stable-Baselines3','lerobot':'LeRobot / VLA'}[f],
                sources=['local','upload']+(['huggingface'] if f=='lerobot' else [])) for f in FORMATS],
            action_groups=copy.deepcopy(GROUPS),default_action_space=copy.deepcopy(DEFAULT_SPACE),
            starter_policies=self.starter_policies()['policies'],
            defaults=dict(environment=dict(task='Lift',robots=['Panda'],controller='OSC_POSE',control_freq=20,horizon=300),
                episodes_per_variation=3,variation_budget=12,seed=42,device='auto',record_video=True,search=normalize_search({})),
            runtime=runtime,limits=dict(max_actions=MAX_ACTIONS,chunk_bytes=CHUNK_BYTES))

    def starter_policies(self):
        from robotics_starters import load_starter_catalog
        return load_starter_catalog(self.directory/'starter-policies')

    def _upload(self, uid):
        if not re.fullmatch(r'[0-9a-f]{32}',str(uid)): raise KeyError('Unknown policy upload.')
        return json.loads((self.uploads/uid/'manifest.json').read_text())

    def create_upload(self, data):
        if not isinstance(data,dict) or data.get('format') not in FORMATS: raise ValueError('Select a policy format.')
        files=data.get('files')
        if not isinstance(files,list) or not 1<=len(files)<=10000: raise ValueError('Select policy files (up to 10,000).')
        entries=[]; seen=set()
        for item in files:
            if not isinstance(item,dict): raise ValueError('Invalid file manifest.')
            path=_text(item.get('path'),'relative policy filename',1024)
            parts=path.split('/')
            if path.startswith('/') or '\\' in path or any(p in ('','..','.') for p in parts) or ':' in path:
                raise ValueError('Policy filenames must be safe relative paths.')
            if path in seen or any(path.startswith(s+'/') or s.startswith(path+'/') for s in seen):
                raise ValueError('Policy filenames overlap.')
            seen.add(path)
            entries.append(dict(path=path,size=_integer(item.get('size'),'File size',0,64*1024**3),uploaded=0))
        primary=None
        if data['format']!='lerobot':
            candidates=[e for e in entries if not e['path'].endswith(('.json','.pkl'))]
            if len(candidates)!=1 or not candidates[0]['size']:
                raise ValueError('Select one native checkpoint plus optional metadata or VecNormalize sidecars.')
            primary=candidates[0]['path']; main=PurePosixPath(primary)
            allowed={primary,primary+'.metadata.json',str(main.with_suffix('.metadata.json'))}
            if data['format']=='sb3': allowed.update({str(main.with_suffix('.vecnormalize.pkl')),str(main.parent/'vecnormalize.pkl')})
            if any(e['path'] not in allowed for e in entries):
                raise ValueError('Companion files must be checkpoint.metadata.json, checkpoint.vecnormalize.pkl, or vecnormalize.pkl.')
        total=sum(e['size'] for e in entries)
        if total==0 or total>64*1024**3: raise ValueError('Upload a nonempty checkpoint of at most 64 GiB, or use its server path.')
        if shutil.disk_usage(self.directory).free<total+256*1024**2: raise ValueError('Not enough free disk space; use an existing server path instead.')
        uid=uuid.uuid4().hex; directory=self.uploads/uid; (directory/'files').mkdir(parents=True)
        manifest=dict(id=uid,format=data['format'],status='uploading',chunk_bytes=CHUNK_BYTES,files=entries,uploaded_bytes=0,total_bytes=total)
        if primary: manifest['primary']=primary
        self._save(directory/'manifest.json',manifest)
        return copy.deepcopy(manifest)

    def upload_chunk(self, uid, index, offset, payload):
        with self.lock:
            manifest=self._upload(uid)
            if manifest['status']!='uploading': raise ValueError('This upload is already complete.')
            _integer(index,'File index',0,len(manifest['files'])-1)
            entry=manifest['files'][index]; _integer(offset,'Chunk offset',0,entry['size'])
            if not isinstance(payload,bytes) or not 1<=len(payload)<=CHUNK_BYTES or offset+len(payload)>entry['size']:
                raise ValueError('Invalid policy chunk size.')
            path=self.uploads/uid/'files'/entry['path']
            if offset<entry['uploaded']:
                with path.open('rb') as stream:
                    stream.seek(offset); previous=stream.read(len(payload))
                if offset+len(payload)>entry['uploaded'] or previous!=payload: raise ValueError('Retry bytes differ from the saved chunk.')
                return manifest
            if offset!=entry['uploaded']: raise ValueError(f'Continue at byte {entry["uploaded"]}.')
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('r+b' if path.exists() else 'wb') as stream:
                stream.seek(offset); stream.write(payload); stream.truncate(offset+len(payload))
            entry['uploaded']+=len(payload); manifest['uploaded_bytes']+=len(payload)
            self._save(self.uploads/uid/'manifest.json',manifest)
            return copy.deepcopy(manifest)

    def finish_upload(self, uid):
        with self.lock:
            manifest=self._upload(uid)
            if any(e['uploaded']!=e['size'] for e in manifest['files']): raise ValueError('Finish uploading every file first.')
            if manifest['format']=='lerobot' and not any(e['path']=='config.json' for e in manifest['files']):
                raise ValueError('Select the LeRobot pretrained policy folder containing config.json.')
            for entry in manifest['files']:
                path=self.uploads/uid/'files'/entry['path']
                if entry['size']==0:
                    path.parent.mkdir(parents=True,exist_ok=True); path.touch(exist_ok=True)
                if path.stat().st_size!=entry['size']: raise ValueError('Uploaded file size differs from its manifest.')
            manifest['status']='ready'; self._save(self.uploads/uid/'manifest.json',manifest)
            return copy.deepcopy(manifest)

    def resolve_policy(self, selection):
        selection=copy.deepcopy(selection)
        if selection['source']=='upload':
            manifest=self._upload(selection['upload_id'])
            if manifest['status']!='ready' or manifest['format']!=selection['format']:
                raise ValueError('Finish uploading this policy in the selected format first.')
            path=self.uploads/manifest['id']/'files'
            if selection['format']!='lerobot': path=path/manifest.get('primary',manifest['files'][0]['path'])
            selection.update(source='local',name=str(path.resolve()))
        return selection

    def start(self, data):
        if not isinstance(data,dict): raise ValueError('Request must be a JSON object.')
        mode=data.get('mode','analyze')
        if mode not in ('environment','analyze','discover'): raise ValueError('Choose Load environment, Analyze, or Find failure.')
        with self.lock:
            if self.active is not None: raise RuntimeError('A robotics run is active. Wait or cancel it first.')
            analysis=None
            if mode=='discover':
                analysis=self.get(data.get('analysis_id',''))
                if analysis['mode']!='analyze' or analysis['status']!='completed' or not analysis['baseline']['episodes']:
                    raise ValueError('Complete Analyze with this policy before finding failures.')
                config=copy.deepcopy(analysis['config'])
                if 'search' in data:
                    config['search']=normalize_search(data['search'])
                    if not analysis.get('embedding',{}).get('points'):
                        raise ValueError('Analyze again to capture episode features before DQN discovery.')
            elif mode=='environment':
                config=normalize_environment_request(data)
            else:
                config=normalize_config(data); config['policy']=self.resolve_policy(config['policy'])
            if not self.injected:
                runtime=self.catalog()['runtime']
                if not runtime['available']: raise RuntimeError(runtime['message']+' '+runtime['command'])
            jid=uuid.uuid4().hex; directory=self.directory/jid; directory.mkdir()
            baseline=copy.deepcopy(analysis['baseline']) if analysis else dict(records=[],**summarize([]))
            job=dict(id=jid,mode=mode,status='queued',config=config,message='Preparing simulation.',phase='preparing',
                completed=0,total=config['episodes_per_variation'] if mode=='analyze' else 0,episode=0,step=0,
                horizon=config['environment']['horizon'],baseline=baseline,records=[],
                nodes=[] if mode=='environment' else variation_nodes(config['action_space']),
                active_action_id=None,artifacts={},
                strategy='environment preview' if mode=='environment' else 'paired seeded environment sweep',created_at=time.time())
            if analysis:
                job['analysis_id']=analysis['id']
                if analysis.get('embedding'): job['embedding']=copy.deepcopy(analysis['embedding'])
            if config.get('search'):
                job['strategy']='terminal DQN over paired environment episodes'
                job['search']=dict(config['search'],episode=0,updates=0,model_calls=0,cache_hits=0,loss=None)
            self.jobs[jid]=job; self.active=jid; self.events[jid]=threading.Event()
            self._persist(job)
            thread=threading.Thread(target=self._execute,args=(jid,),daemon=True,name=f'robotics-{jid[:8]}')
            self.threads[jid]=thread; thread.start()
            return copy.deepcopy(job)

    def get(self, jid):
        with self.lock:
            if not isinstance(jid,str) or jid not in self.jobs: raise KeyError('Unknown robotics run.')
            return copy.deepcopy(self.jobs[jid])

    def predictions(self, jid, input_index):
        job=self.get(jid)
        points=job.get('embedding',{}).get('points',[])
        _integer(input_index,'Episode index',0,len(points)-1)
        metadata=job.get('predicted_landscape',{})
        updates=metadata.get('policy_updates',0)
        result=dict(input_index=input_index,policy_updates=updates,actions=[],available=False)
        if not updates or not metadata.get('available'): return result
        filename=metadata.get('values_file','')
        if not re.fullmatch(r'dqn-predictions(?:-\d+)?\.npy',filename):
            raise ValueError('DQN predictions are unavailable for this run.')
        import numpy as np
        try: values=np.load(self.directory/jid/filename,mmap_mode='r',allow_pickle=False)
        except FileNotFoundError: return result  # A newer snapshot replaced this poll's version.
        ids=metadata.get('action_ids',[])
        if (values.shape!=(len(points),len(ids)) or ids!=[node['id'] for node in job['nodes']]
            or metadata.get('seeds')!=[point['seed'] for point in points]):
            raise ValueError('Prediction dimensions do not match this analysis.')
        row=values[input_index]
        if not np.isfinite(row).all(): raise ValueError('DQN returned nonfinite predictions.')
        result.update(available=True,actions=[dict(action_id=action_id,q_value=float(value)) for action_id,value in zip(ids,row)])
        return result

    def _emit(self, jid, event):
        with self.lock:
            job=self.jobs[jid]; kind=event.get('type')
            if kind=='progress':
                for key in ('message','phase','completed','total','episode','step','horizon','active_action_id','policy_info','warnings','environment_info',
                            'search','current','policy_statistics','predicted_landscape'):
                    if key in event: job[key]=event[key]
                active=event.get('active_action_id')
                if type(active) is int and 0<=active<len(job['nodes']): job['nodes'][active]['status']='running'
            elif kind=='embedding':
                embedding=copy.deepcopy(event['embedding'])
                baseline={row['seed']:row for row in job['baseline']['records']}
                for point in embedding['points']:
                    for key in ('image','video'):
                        if point.get(key) and ARTIFACT.fullmatch(point[key]): point[key]=self._url(jid,point[key])
                    if point['seed'] in baseline:
                        baseline[point['seed']].update(input_index=point['index'],embedding=dict(position=point['position'],
                            similarity=point.get('self_similarity',1.),feature_samples=point.get('feature_samples')))
                job['embedding']=embedding
            elif kind=='artifact':
                filename=event['filename']
                if ARTIFACT.fullmatch(filename):
                    url=self._url(jid,filename)
                    job['artifacts'][event['name']]=url+(f'?v={time.time_ns()}' if filename=='live.jpg' else '')
            elif kind=='resolved_policy':
                job['config']['policy']=copy.deepcopy(event['policy'])
            elif kind=='record':
                record=copy.deepcopy(event['record']); action_id=record['action_id']
                for key in ('video','image'):
                    if record.get(key) and ARTIFACT.fullmatch(record[key]): record[key]=self._url(jid,record[key])
                if action_id=='baseline':
                    records=job['baseline']['records']; records.append(record)
                    job['baseline'].update(summarize(records))
                elif type(action_id) is int and 0<=action_id<len(job['nodes']):
                    job['records'].append(record)
                    node=job['nodes'][action_id]; records=[r for r in job['records'] if r['action_id']==action_id]
                    node.update(summarize(records,job['baseline']['records']))
                    if record.get('reused_baseline'): node['reused_baseline']=True
                    node['status']='completed' if node['episodes'] else 'error'
                    for key in ('video','image'):
                        if record.get(key) and (not node.get(key) or not record.get('success')): node[key]=record[key]
                    if record.get('error'): node['error']=record['error']
            self._persist(job)

    def _execute(self, jid):
        event=self.events[jid]
        with self.lock:
            job=self.jobs[jid]; job['status']='running'; snapshot=copy.deepcopy(job)
        try:
            self.runner(snapshot,self.directory/jid,lambda item:self._emit(jid,item),event)
            with self.lock:
                if event.is_set(): job.update(status='cancelled',message='Run cancelled. Partial results retained.')
                elif job['mode']=='environment':
                    if not job.get('environment_info') or not job['artifacts'].get('preview'):
                        raise RuntimeError('The environment preview was not returned by the worker.')
                    job.update(status='completed',phase='completed',message='Environment loaded. Select a policy to Analyze.')
                elif not job['baseline']['episodes']:
                    first=next((r.get('error') for r in job['baseline']['records'] if r.get('error')), 'No rollout result was returned.')
                    raise RuntimeError('No baseline episode completed. '+first)
                else: job.update(status='completed',phase='completed',message='Baseline ready.' if job['mode']=='analyze' else
                    'DQN discovery complete.' if job['config'].get('search') else 'Environment sweep complete.')
        except Exception as exc:
            with self.lock:
                if event.is_set(): job.update(status='cancelled',message='Run cancelled. Partial results retained.')
                else: job.update(status='error',error=str(exc),message=str(exc))
        finally:
            with self.lock:
                job['active_action_id']=None
                job['current']=None
                for node in job['nodes']:
                    if node['status']=='running': node['status']='completed' if node['episodes'] else 'untested'
                try: self._report(job)
                except OSError as exc: job['report_error']=str(exc)
                self._persist(job); self.active=None; self.processes.pop(jid,None)

    def _run_worker(self, job, directory, emit, cancel):
        request=dict(job=job,analysis_directory=str(self.directory/job['analysis_id']) if job.get('analysis_id') else None)
        self._save(directory/'request.json',request)
        env=os.environ.copy(); env['MUJOCO_GL']='egl'; env['PYOPENGL_PLATFORM']='egl'; env['PYTHONUNBUFFERED']='1'
        process=subprocess.Popen([sys.executable,'-E',str(ROOT/'robotics_worker.py'),str(directory/'request.json')],
            cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
        with self.lock: self.processes[job['id']]=process
        if cancel.is_set(): self._stop_process(process)
        tail=[]
        try:
            with (directory/'worker.log').open('w') as log:
                for line in process.stdout:
                    if line.startswith('@robotics '):
                        emit(json.loads(line[len('@robotics '):]))
                    else:
                        log.write(line); log.flush(); tail.append(line.strip()); tail=tail[-12:]
            code=process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
            process.stdout.close()
        if code and not cancel.is_set(): raise RuntimeError('Simulation worker failed: '+' '.join(tail)[-2500:])

    def cancel(self, jid):
        with self.lock:
            job=self.get(jid)
            if job['status'] not in ('queued','running'): return job
            self.events[jid].set(); process=self.processes.get(jid)
            self.jobs[jid]['message']='Cancelling simulation…'
            if process is not None and process.poll() is None:
                self._stop_process(process)
            return self.get(jid)

    @staticmethod
    def _stop_process(process):
        try: process.terminate()
        except ProcessLookupError: return
        def kill_later():
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try: process.kill()
                except ProcessLookupError: pass
        threading.Thread(target=kill_later,daemon=True).start()

    def close(self):
        if self.active:
            jid=self.active; self.cancel(jid); self.threads[jid].join(timeout=8)

    def artifact(self, jid, filename):
        if not re.fullmatch(r'[0-9a-f]{32}',str(jid)) or not ARTIFACT.fullmatch(filename): raise FileNotFoundError('Unknown artifact.')
        if jid not in self.jobs: raise FileNotFoundError('Unknown run.')
        path=self.directory/jid/filename
        if not path.is_file(): raise FileNotFoundError('Artifact not ready.')
        return path

    @staticmethod
    def _url(jid, filename): return f'/robotics/{jid}/artifacts/{filename}'

    @staticmethod
    def _save(path, data):
        temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(data,allow_nan=False)); temp.replace(path)

    def _persist(self, job): self._save(self.directory/job['id']/'job.json',job)

    def _report(self, job):
        directory=self.directory/job['id']
        job['artifacts'].update(report=self._url(job['id'],'report.html'),results=self._url(job['id'],'results.json'))
        self._save(directory/'results.json',job)
        if job['mode']=='environment':
            scene=job['artifacts'].get('preview')
            image=f'<img src="{html.escape(scene,quote=True)}" alt="Loaded environment">' if scene else ''
            (directory/'report.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>FailureShift · Environment preview</title>
<body><h1>Environment preview</h1><p>{html.escape(job['message'])}</p>{image}
<pre>{html.escape(json.dumps(job.get('environment_info',job['config']),indent=2))}</pre>
<p><a href="results.json" target="_blank" rel="noopener">Environment settings</a></p></body></html>''')
            return
        def pct(value): return '—' if value is None else f'{value:.1%}'
        rows=[]
        for node in job['nodes']:
            replay=node.get('video') or node.get('image')
            link=f'<a href="{html.escape(replay,quote=True)}" target="_blank" rel="noopener">Replay</a>' if replay else ''
            status='Baseline reused' if node.get('reused_baseline') else node['status']
            rows.append(f'<tr><td>{html.escape(node["label"])}</td><td>{html.escape(status)}</td>'
                        f'<td>{pct(node["failure_rate"])}</td><td>{node["episodes"]}</td>'
                        f'<td>{node.get("regressions",0)}</td><td>{node.get("errors",0)}</td><td>{link}</td></tr>')
        document=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>FailureShift · Robosuite results</title><body><h1>Robosuite results</h1>
<p>{html.escape(job['config']['environment']['task'])} · {html.escape(job['status'])}</p>
<p>{html.escape(job['message'])}</p>
<p>Baseline success: {pct(job['baseline']['success_rate'])} over {job['baseline']['episodes']} episodes.</p>
<p>{html.escape(job['strategy'])}. Rates include completed episodes only. Errors are reported separately.
The policy stays fixed; untested combinations have no measured failure rate.</p>
<p>{'DQN colors estimate the selected terminal reward; they are not measured failure rates or calibrated probabilities. Cached visits do not add trials.' if job['config'].get('search') else ''}</p>
<details><summary>Representation and search</summary><pre>{html.escape(json.dumps(dict(source=job.get('embedding',{}).get('source'),projection=job.get('embedding',{}).get('projection'),search=job.get('search')),indent=2))}</pre></details>
<table><thead><tr><th>Environment variation</th><th>Status</th><th>Failure rate</th><th>Episodes</th><th>New failures</th><th>Errors</th><th>Media</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table><p><a href="results.json" target="_blank" rel="noopener">Full JSON report</a></p>
<details><summary>Run settings</summary><pre>{html.escape(json.dumps(job['config'],indent=2))}</pre></details></body></html>'''
        (directory/'report.html').write_text(document)
