"""Real direct-reset versus saved-reset control traces; isolated runtime required."""
from pathlib import Path
import json
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from setup_robotics import bootstrap_runtime
bootstrap_runtime()
import numpy as np
from robotics import normalize_config
from robotics_env import create_environment, seed_episode, save_snapshot, restore_snapshot


def main():
    checks=[]
    for controller in ('OSC_POSE','OSC_POSITION','JOINT_POSITION','JOINT_VELOCITY'):
        config=normalize_config({'policy':{'format':'torchscript','source':'local','name':'unused.pt'},
                                 'environment':{'task':'Lift','controller':controller,'horizon':12}})
        env,_=create_environment(config,{})
        try:
            with tempfile.TemporaryDirectory() as directory:
                seed_episode(2000,env); original=env.reset()
                assert env.robots[0].part_controllers['right'].name==controller
                expected_joints=env.robots[0].part_controllers['right'].initial_joint.copy()
                save_snapshot(env,directory,2000)
                actions=np.random.default_rng(100).uniform(-.1,.1,(5,env.action_dim)); actions[:,-1]=-1
                direct=[]
                for action in actions:
                    env.step(action); direct.append(env.sim.data.qpos.copy())
                restored=restore_snapshot(env,directory,2000)
                np.testing.assert_array_equal(restored['robot0_joint_pos'],original['robot0_joint_pos'])
                np.testing.assert_allclose(env.robots[0].part_controllers['right'].initial_joint,
                                           expected_joints,rtol=0,atol=1e-12,err_msg=controller+' initial joints')
                repeated=[]
                for action in actions:
                    env.step(action); repeated.append(env.sim.data.qpos.copy())
                maximum=float(np.max(np.abs(np.asarray(direct)-repeated)))
                # XML serialization rounds model constants; allow numerical noise,
                # far below the milliradian drift caused by stale controller state.
                np.testing.assert_allclose(repeated,direct,rtol=0,atol=1e-7,err_msg=controller+' restored action trace')
                checks.append({'controller':controller,'max_joint_error':maximum,'steps':len(actions)})
                print(json.dumps(checks[-1]),flush=True)
        finally:
            env.close()
    print(json.dumps({'checks':checks}),flush=True)


if __name__=='__main__': main()
