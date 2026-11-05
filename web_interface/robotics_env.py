"""Matched Robosuite resets and explicit environment interventions."""
import copy
import json
import math
from pathlib import Path
import random
import re

import numpy as np

from robotics import GROUPS


def environment_kwargs(environment, metadata, mapping):
    """Honor native task settings; reject changes to the trained embodiment."""
    expected=metadata.get('environment',{})
    if expected.get('task') and expected['task']!=environment['task']:
        raise ValueError(f"Checkpoint task is {expected['task']}; select that task instead of {environment['task']}.")
    robots=expected.get('robots')
    if robots and ([robots] if isinstance(robots,str) else robots)!=environment['robots']:
        raise ValueError(f"Checkpoint robots are {robots}; select the same robot configuration.")
    kwargs=copy.deepcopy(expected.get('env_kwargs',{}))
    frequency=expected.get('control_freq',kwargs.get('control_freq'))
    if frequency and frequency!=environment['control_freq']:
        raise ValueError(f"Checkpoint control frequency is {frequency} Hz; select that frequency to preserve action timing.")
    if expected.get('controller') and expected['controller']!=environment['controller']:
        raise ValueError(f"Checkpoint controller is {expected['controller']}; select the trained controller.")
    version=re.match(r'^(\d+)\.(\d+)(?:\.(\d+))?',str(expected.get('env_version','')))
    if version and tuple(int(value or 0) for value in version.groups())<=(1,4,1):
        kwargs.setdefault('lite_physics',False)
    # Serialized kwargs are data, not constructors. Force headless local execution.
    for key in ('env_name','env','render_camera','renderer_config','placement_initializer','seed'):
        kwargs.pop(key,None)
    kwargs.update(env_name=environment['task'],robots=environment['robots'],horizon=environment['horizon'],
        control_freq=environment['control_freq'],has_renderer=False,has_offscreen_renderer=True,
        renderer='mujoco',hard_reset=True,ignore_done=False,reward_shaping=True,use_object_obs=True)
    cameras={}
    for camera in metadata.get('cameras',[]):
        raw=mapping.get('observation_map',{}).get(camera['key'])
        if raw:
            suffix='_depth' if camera.get('depth') else '_image'
            if not raw.endswith(suffix): raise ValueError(f'Camera mapping {raw} must end in {suffix}.')
            name=raw[:-len(suffix)]
        else: name=camera.get('name')
        if not name:
            raise ValueError(f"Map policy camera {camera['key']} to a Robosuite camera, e.g. agentview_image, using observation_map.")
        dimensions=(int(camera['height']),int(camera['width']),bool(camera.get('depth')))
        if any(not 1<=v<=2048 for v in dimensions[:2]): raise ValueError('Camera dimensions must be 1–2048 pixels.')
        if name in cameras and cameras[name][:2]!=dimensions[:2]: raise ValueError('The same camera has incompatible feature resolutions.')
        cameras[name]=(dimensions[0],dimensions[1],dimensions[2] or cameras.get(name,(0,0,False))[2])
    kwargs.update(use_camera_obs=bool(cameras),camera_names=list(cameras) or ['agentview'],
        camera_heights=[v[0] for v in cameras.values()] or 256,camera_widths=[v[1] for v in cameras.values()] or 256,
        camera_depths=[v[2] for v in cameras.values()] or False)
    controller=expected.get('controller_configs') or kwargs.get('controller_configs')
    if controller:
        parts=controller if isinstance(controller,list) else [controller]
        for config in parts:
            kinds=[config.get('type')] if 'body_parts' not in config else [
                part.get('type') for key,part in config['body_parts'].items() if key in ('right','left')]
            if any(kind and kind!=environment['controller'] for kind in kinds):
                raise ValueError(f'Checkpoint controller {kinds} differs from {environment["controller"]}. Select its trained controller.')
        kwargs['controller_configs']=controller
    else: kwargs.pop('controller_configs',None)
    return kwargs


def create_environment(config, metadata):
    import robosuite as suite
    from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
    from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
    kwargs=environment_kwargs(config['environment'],metadata,config['mapping'])
    controllers=kwargs.get('controller_configs')
    if not controllers:
        controllers=suite.load_part_controller_config(default_controller=config['environment']['controller'])
    if not isinstance(controllers,list): controllers=[copy.deepcopy(controllers) for _ in kwargs['robots']]
    if len(controllers)!=len(kwargs['robots']): raise ValueError('Checkpoint controller count does not match selected robots.')
    converted=[]
    for controller,robot in zip(controllers,kwargs['robots']):
        if robot not in REGISTERED_ROBOTS: raise ValueError(f'Robot {robot} is not installed in this Robosuite runtime.')
        arms=getattr(REGISTERED_ROBOTS[robot],'arms',['right'])
        converted.append(refactor_composite_controller_config(controller,robot,arms))
    kwargs['controller_configs']=converted
    env=suite.make(**kwargs)
    seed_episode(config['seed'],env)
    return env,kwargs['camera_names'] if kwargs['use_camera_obs'] else []


def seed_episode(seed, env=None):
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.seed=seed; env.rng=np.random.default_rng(seed)


def save_snapshot(env, directory, seed):
    """Save an episode-start reset, including controller references and task mode."""
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    prefix=directory/f'seed-{seed}'
    prefix.with_suffix('.xml').write_text(env.sim.model.get_xml())
    arrays={'state':env.sim.get_state().flatten()}
    for key in ('ctrl','act','qacc_warmstart'):
        arrays[key]=np.array(getattr(env.sim.data,key),copy=True)
    np.savez_compressed(prefix.with_suffix('.npz'),**arrays)
    context={}
    for key in ('obj_to_use','object_id','nut_id','objects_in_bins','objects_on_pegs'):
        if hasattr(env,key):
            value=getattr(env,key)
            context[key]=value.tolist() if isinstance(value,np.ndarray) else value
    context['observables']={key:[obs.is_enabled(),obs.is_active()] for key,obs in env._observables.items()}
    context['rng']=env.rng.bit_generator.state
    context['ep_meta']=env.get_ep_meta()
    context['controllers']=[{
        name:{key:np.asarray(getattr(controller,key)).tolist()
              for key in ('initial_joint','initial_ref_pos','initial_ref_ori_mat') if hasattr(controller,key)}
        for name,controller in robot.part_controllers.items()
    } for robot in env.robots]
    prefix.with_suffix('.json').write_text(json.dumps(context))


def restore_snapshot(env, directory, seed):
    prefix=Path(directory)/f'seed-{seed}'
    context=json.loads(prefix.with_suffix('.json').read_text())
    if 'controllers' not in context:
        raise ValueError('This saved scene lacks the robot controller reset state. Run Analyze again before finding failures.')
    seed_episode(seed,env)
    env.set_ep_meta(context.get('ep_meta',{}))
    env.reset_from_xml_string(prefix.with_suffix('.xml').read_text())
    for key,value in context.items():
        if key in ('rng','observables','ep_meta','controllers'): continue
        setattr(env,key,np.array(value) if isinstance(getattr(env,key,None),np.ndarray) else value)
    for key,(enabled,active) in context['observables'].items():
        if key in env._observables:
            env._observables[key].set_enabled(enabled); env._observables[key].set_active(active)
    env.rng.bit_generator.state=context['rng']
    with np.load(prefix.with_suffix('.npz'),allow_pickle=False) as arrays:
        env.sim.set_state_from_flattened(arrays['state'])
        for key in ('ctrl','act','qacc_warmstart'): getattr(env.sim.data,key)[:]=arrays[key]
    env.sim.forward(); env.update_state(); env._obs_cache={}
    # XML reset creates controllers at noise-free joints. Restore the original
    # randomized reset references and refresh cached kinematics before action 1.
    if len(context['controllers'])!=len(env.robots):
        raise ValueError('Saved robot controllers do not match this scene. Run Analyze again.')
    for robot,saved in zip(env.robots,context['controllers']):
        if set(saved)!=set(robot.part_controllers):
            raise ValueError('Saved controller parts do not match this scene. Run Analyze again.')
        for name,controller in robot.part_controllers.items():
            controller.update(force=True)
            for key,value in saved[name].items():
                setattr(controller,key,np.array(value,copy=True))
        robot.composite_controller.update_state()
        robot.composite_controller.reset()
    # Re-seed policy-side sampling equally after reset's randomization.
    seed_episode(seed)
    return env._get_observations(force_update=True)


def _object_roots(env):
    result=[]
    for obj in env.model.mujoco_objects:
        # PickPlace includes purely visual goal markers in mujoco_objects.
        # Moving those would change the target along with the manipulated object.
        if getattr(obj,'obj_type',None)=='visual' or hasattr(obj,'contact_geoms') and not obj.contact_geoms:
            continue
        if getattr(env,'single_object_mode',0) and getattr(env,'obj_to_use',None)!=obj.name: continue
        root=env.sim.model.body_name2id(obj.root_body)
        if root>0: result.append(root)
    return list(dict.fromkeys(result))


def _descendants(model, roots):
    result=[]
    for body in range(1,model.nbody):
        parent=body
        while parent>0:
            if parent in roots: result.append(body); break
            parent=int(model.body_parentid[parent])
    return result


def _quat_multiply(a,b):
    w,x,y,z=a; v,i,j,k=b
    return np.array([w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v])


def _recompute(env):
    import mujoco
    # mj_setConst modifies data while recomputing constants: restore the physical state.
    state=env.sim.get_state().flatten().copy()
    mujoco.mj_setConst(env.sim.model._model,env.sim.data._data)
    env.sim.set_state_from_flattened(state)


def apply_variation(env, values, *, cameras=(), recompute=None):
    known={g['id']:g['neutral'] for g in GROUPS}
    if any(key not in known for key in values): raise ValueError('Unknown environment variation.')
    changes={key:value for key,value in values.items() if value!=known[key]}
    model=env.sim.model; roots=_object_roots(env)
    object_changes=set(changes)&{'mass_scale','object_x_offset','object_y_offset','object_yaw'}
    if object_changes and not roots: raise ValueError('This task has no applicable task object. Remove object mass/pose actions.')
    if set(changes)&{'lighting_scale','camera_offset'} and not cameras:
        raise ValueError('This policy has no camera inputs. Remove lighting/camera actions for a state-only policy.')
    if 'friction_scale' in changes: model.geom_friction[:]*=changes['friction_scale']
    if 'mass_scale' in changes:
        bodies=_descendants(model,roots)
        if not bodies or not np.any(model.body_mass[bodies]>0): raise ValueError('No task object mass can be changed in this task.')
        model.body_mass[bodies]*=changes['mass_scale']; model.body_inertia[bodies]*=changes['mass_scale']
        (recompute or _recompute)(env)
    shift=np.array([changes.get('object_x_offset',0),changes.get('object_y_offset',0),0.])
    theta=math.radians(changes.get('object_yaw',0))
    if np.any(shift) or theta:
        for body in roots:
            first=int(model.body_jntadr[body]); count=int(model.body_jntnum[body])
            free=next((j for j in range(first,first+count) if model.jnt_type[j]==0),None)
            if free is not None:
                addr=int(model.jnt_qposadr[free]); env.sim.data.qpos[addr:addr+3]+=shift
                if theta:
                    yaw=[math.cos(theta/2),0,0,math.sin(theta/2)]
                    env.sim.data.qpos[addr+3:addr+7]=_quat_multiply(yaw,env.sim.data.qpos[addr+3:addr+7])
            else:
                parent=int(model.body_parentid[body]); rotation=env.sim.data.xmat[parent].reshape(3,3).T
                model.body_pos[body]+=rotation@shift
                if theta:
                    axis=rotation@np.array([0.,0.,1.]); yaw=np.r_[math.cos(theta/2),axis*math.sin(theta/2)]
                    model.body_quat[body]=_quat_multiply(yaw,model.body_quat[body])
    if 'gravity_scale' in changes: model.opt.gravity[:]*=changes['gravity_scale']
    if 'lighting_scale' in changes:
        for name in ('light_diffuse','light_ambient','light_specular'):
            values_array=getattr(model,name); values_array[:]=np.clip(values_array*changes['lighting_scale'],0,1)
    if 'camera_offset' in changes:
        for name in set(cameras):
            index=model.camera_name2id(name); parent=int(model.cam_bodyid[index])
            model.cam_pos[index]+=env.sim.data.xmat[parent].reshape(3,3).T@np.array([changes['camera_offset'],0,0])
    env.sim.forward()
    return changes


def perturb_observation(observation, sigma, rng):
    if not sigma: return observation
    return {key:(np.asarray(value)+rng.normal(0,sigma,np.asarray(value).shape)
                 if np.asarray(value).ndim<=1 and np.issubdtype(np.asarray(value).dtype,np.floating) else value)
            for key,value in observation.items()}


def delayed_action(action, delay, queue):
    if not delay: return action
    queue.append(np.asarray(action).copy())
    return queue.pop(0) if len(queue)>delay else np.zeros_like(action)
