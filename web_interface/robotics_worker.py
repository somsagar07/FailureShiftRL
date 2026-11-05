"""Isolated Robosuite episodes, policy representations, and failure discovery."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time


def emit(event):
    print('@robotics '+json.dumps(event,allow_nan=False),flush=True)


def fingerprint(selection):
    if selection['source']!='local':
        return {'repository':selection['name'],'revision':selection.get('revision')}
    path=Path(selection['name']).expanduser().resolve()
    if not path.exists(): raise ValueError(f'Policy path does not exist on the server: {path}')
    files=sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    if not path.is_dir():
        for sidecar in [Path(str(path)+'.metadata.json'),path.with_suffix('.metadata.json'),
                        path.with_suffix('.vecnormalize.pkl'),path.parent/'vecnormalize.pkl']:
            if sidecar.is_file(): files.append(sidecar)
    if len(files)>10000: raise ValueError('Policy folder has too many files. Select the pretrained checkpoint folder only.')
    result={}
    for file in files:
        digest=hashlib.sha256()
        with file.open('rb') as stream:
            for chunk in iter(lambda:stream.read(4*1024**2),b''): digest.update(chunk)
        result[str(file.relative_to(path) if path.is_dir() else file.name)]=dict(size=file.stat().st_size,sha256=digest.hexdigest())
    return result


def _parent_guard():
    if sys.platform.startswith('linux'):
        import ctypes
        import signal
        parent=os.getppid()
        ctypes.CDLL(None).prctl(1,signal.SIGTERM)
        if os.getppid()!=parent: raise RuntimeError('The server stopped before this worker started.')


class Rollouts:
    def __init__(self, config, directory, env, policy, cameras, landscape=None):
        self.config=config; self.directory=directory; self.env=env; self.policy=policy; self.cameras=cameras
        self.completed=0; self.total=0; self.media={}; self.landscape=landscape; self.last_features=None
        available=list(env.sim.model.camera_names)
        self.preview_camera=next((name for name in ['agentview',*cameras,'frontview'] if name in available),available[0] if available else None)

    def frame(self):
        import numpy as np
        # MuJoCo's raw framebuffer is bottom-up; policy observations remain untouched.
        pixels=self.env.sim.render(width=256,height=256,camera_name=self.preview_camera)
        return np.ascontiguousarray(pixels[::-1])

    def preview(self, frame):
        from PIL import Image
        path=self.directory/'live.jpg'; temporary=self.directory/'live.tmp.jpg'
        Image.fromarray(frame).save(temporary,quality=82); temporary.replace(path)
        emit(dict(type='artifact',name='preview',filename=path.name))

    def media_for(self, action_id, seed, success, frames, frame, stride):
        from PIL import Image
        # Every point gets its own final frame. Bound videos to the first episode
        # and first failure per condition; never substitute another seed's video.
        previous=self.media.get(action_id)
        retain_video=previous is None or previous is True and not success
        if retain_video: self.media[action_id]=success
        prefix=f'baseline-seed-{seed}' if action_id=='baseline' else f'action-{action_id}-seed-{seed}'
        result={}
        if frame is not None:
            path=self.directory/(prefix+'.jpg'); Image.fromarray(frame).save(path,quality=88); result['image']=path.name
        if retain_video and frames and self.config['record_video']:
            import imageio.v2 as imageio
            path=self.directory/(prefix+'.mp4')
            imageio.mimsave(path,frames,fps=self.config['environment']['control_freq']/stride,
                codec='libx264',quality=6,macro_block_size=16,ffmpeg_log_level='error')
            result['video']=path.name
        return result

    def episode(self, action_id, values, seed, snapshot_dir, episode):
        import numpy as np
        from robotics_env import restore_snapshot,apply_variation,perturb_observation,delayed_action
        from robotics_landscape import EpisodeFeatures
        features=EpisodeFeatures(seed=seed); self.last_features=None; feature_error=None
        started=time.monotonic(); steps=0; reward_sum=0.; frame=None; frames=[]
        horizon=self.config['environment']['horizon']; stride=max(2,math.ceil(horizon/120))
        phase='baseline' if action_id=='baseline' else 'variations'
        record=dict(action_id=action_id,seed=seed,success=None)
        if self.landscape is not None: record['input_index']=self.landscape.indexes[seed]
        emit(dict(type='progress',phase=phase,active_action_id=action_id,episode=episode,step=0,
                  completed=self.completed,total=self.total,message=f'{phase.title()} · episode {episode}'))
        try:
            restore_snapshot(self.env,snapshot_dir,seed)
            apply_variation(self.env,values,cameras=self.cameras)
            self.env._obs_cache={}; observation=self.env._get_observations(force_update=True)
            self.policy.reset(); queue=[]; rng=np.random.default_rng(seed)
            frame=self.frame(); self.preview(frame)
            if self.config['record_video']: frames.append(frame)
            success=False; last_update=0
            for steps in range(1,horizon+1):
                action=self.policy.act(perturb_observation(observation,values.get('observation_noise',0),rng))
                if feature_error is None:
                    try:
                        vector=self.policy.take_features()
                        if vector is not None: features.add(vector)
                    except Exception as exc:
                        feature_error=f'Policy embedding unavailable: {exc}'
                action=delayed_action(action,int(values.get('action_delay',0)),queue)
                observation,reward,done,_=self.env.step(action)
                if not math.isfinite(float(reward)) or not np.isfinite(self.env.sim.data.qpos).all():
                    raise RuntimeError('Simulation returned a nonfinite state or reward.')
                reward_sum+=float(reward)
                checked=self.env._check_success()
                if isinstance(checked,dict): checked=checked.get('task',all(checked.values()))
                success=bool(checked)
                if steps%stride==0 or success or done or steps==horizon:
                    frame=self.frame()
                    if self.config['record_video']: frames.append(frame)
                if time.monotonic()-last_update>.5 or success or done or steps==horizon:
                    self.preview(frame)
                    emit(dict(type='progress',phase=phase,active_action_id=action_id,episode=episode,step=steps,
                        completed=self.completed,total=self.total,message=f'{phase.title()} · episode {episode} · step {steps}/{horizon}'))
                    last_update=time.monotonic()
                if success or done: break
            record['success']=success
            record['severity']=float(not success)
            try: record.update(self.media_for(action_id,seed,success,frames,frame,stride))
            except Exception as exc: record['media_warning']=f'Rollout completed; replay could not be saved: {exc}'
        except Exception as exc:
            record.update(error=f'{type(exc).__name__}: {exc}',success=None)
        record.update(steps=steps,seconds=round(time.monotonic()-started,3))
        record['return']=reward_sum
        if feature_error is None and not record.get('error') and type(record.get('success')) is bool:
            self.last_features=features.finish()
            if self.landscape is not None and self.last_features is not None:
                try:
                    info=self.policy.feature_info(); expected=self.landscape.public['source']
                    if any(info.get(key)!=expected.get(key) for key in ('kind','source_layer','dimension','pooling')):
                        raise ValueError('The policy feature source changed since Analyze.')
                    record['embedding']=self.landscape.project(seed,self.last_features)
                except ValueError as exc: feature_error=str(exc)
        capture_warning=self.policy.feature_info().get('warning')
        if feature_error or capture_warning: record['embedding_warning']=feature_error or capture_warning
        self.completed+=1
        emit(dict(type='record',record=record))
        emit(dict(type='progress',completed=self.completed,total=self.total))
        return record


def preview_environment(config, directory):
    from robotics_env import create_environment,seed_episode
    env=None
    try:
        emit(dict(type='progress',phase='environment',message='Loading environment.'))
        env,cameras=create_environment(config,{})
        seed_episode(config['seed'],env)
        observation=env.reset()
        viewer=Rollouts(config,directory,env,None,cameras)
        viewer.preview(viewer.frame())
        info=dict(config['environment'],seed=config['seed'],action_dimension=int(env.action_dim),
            observation_shapes={key:list(value.shape) if hasattr(value,'shape') else [] for key,value in observation.items()},
            cameras=list(env.sim.model.camera_names),preview_camera=viewer.preview_camera)
        emit(dict(type='progress',phase='completed',environment_info=info,message='Environment loaded.'))
    finally:
        if env is not None: env.close()


def run(request_file):
    _parent_guard()
    from setup_robotics import bootstrap_runtime
    bootstrap_runtime()
    # All numeric/model imports follow overlay initialization.
    from robotics_env import create_environment,seed_episode,save_snapshot
    from robotics import GROUPS
    request_file=Path(request_file); directory=request_file.parent
    request=json.loads(request_file.read_text()); job=request['job']; config=job['config']
    if job['mode']=='environment':
        return preview_environment(config,directory)
    from robotics_policies import read_policy_metadata,load_policy
    selection=config['policy']; seed_episode(config['seed'])
    emit(dict(type='progress',phase='preparing',message='Reading policy configuration.'))
    metadata=read_policy_metadata(selection)
    if metadata.get('resolved_revision'):
        selection['revision']=metadata['resolved_revision']
        emit(dict(type='resolved_policy',policy=selection))
    identity=fingerprint(selection)
    snapshot_dir=directory/'snapshots'
    if job['mode']=='discover':
        analysis_dir=Path(request['analysis_directory']); snapshot_dir=analysis_dir/'snapshots'
        previous=json.loads((analysis_dir/'policy-identity.json').read_text())
        if identity!=previous: raise ValueError('The policy files changed after Analyze. Analyze the checkpoint again before discovery.')
    (directory/'policy-identity.json').write_text(json.dumps(identity))
    env=None; policy=None
    try:
        emit(dict(type='progress',message='Creating Robosuite environment.'))
        env,cameras=create_environment(config,metadata)
        emit(dict(type='progress',message='Loading policy weights.'))
        policy=load_policy(selection,env,device=config['device'],instruction=config['instruction'],mapping=config['mapping'],
            progress=lambda message:emit(dict(type='progress',message=str(message))))
        policy.enable_features()
        emit(dict(type='progress',policy_info=policy.describe(),warnings=metadata.get('warnings',[])))
        rollouts=Rollouts(config,directory,env,policy,cameras)
        if job['mode']=='analyze':
            from robotics_landscape import LandscapeBuilder
            builder=LandscapeBuilder()
            seeds=[config['seed']+index for index in range(config['episodes_per_variation'])]
            rollouts.total=len(seeds)
            for index,seed in enumerate(seeds):
                emit(dict(type='progress',message=f'Preparing initial state {index+1}/{len(seeds)}.'))
                seed_episode(seed,env); env.reset(); save_snapshot(env,snapshot_dir,seed)
                record=rollouts.episode('baseline',{},seed,snapshot_dir,index+1)
                builder.add(record,rollouts.last_features,policy.feature_info())
            landscape=builder.finish()
            if landscape is not None:
                landscape.save(directory)
                emit(dict(type='embedding',embedding=landscape.public))
            else:
                emit(dict(type='progress',warnings=['No policy features were captured. Check the adapter and Analyze again.']))
        elif config.get('search') is not None:
            from robotics_landscape import EpisodeLandscape
            from robotics_search import run_search
            landscape=EpisodeLandscape.load(analysis_dir)
            rollouts.landscape=landscape
            rollouts.total=config['search']['rollout_budget']
            by_seed={row['seed']:row for row in job['baseline']['records'] if type(row.get('success')) is bool and not row.get('error')}
            baseline=[by_seed[seed] for seed in landscape.seeds]
            def evaluate(input_index,action):
                return rollouts.episode(action['id'],action['values'],landscape.seeds[input_index],snapshot_dir,input_index+1)
            run_search(landscape.states,landscape.seeds,job['nodes'],baseline,config['search'],evaluate,emit,directory)
        else:
            baseline=[r for r in job['baseline']['records'] if not r.get('error') and type(r.get('success')) is bool]
            neutral={g['id']:g['neutral'] for g in GROUPS}
            nodes=job['nodes']
            candidates=[]
            for node in nodes:
                if all(value==neutral[key] for key,value in node['values'].items()):
                    for row in baseline:
                        # Artifact URLs still refer to the baseline run. Never fabricate another rollout.
                        reused=dict(row,action_id=node['id'],reused_baseline=True)
                        emit(dict(type='record',record=reused))
                else: candidates.append(node)
            random.Random(config['seed']).shuffle(candidates)
            candidates=candidates[:config['variation_budget']]
            rollouts.total=len(candidates)*len(baseline)
            for node in candidates:
                for index,row in enumerate(baseline):
                    rollouts.episode(node['id'],node['values'],row['seed'],snapshot_dir,index+1)
        emit(dict(type='progress',phase='completed',completed=rollouts.completed,total=rollouts.total,active_action_id=None))
    finally:
        if policy is not None: policy.close()
        if env is not None: env.close()


if __name__=='__main__':
    try: run(sys.argv[1])
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
