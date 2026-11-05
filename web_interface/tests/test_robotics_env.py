import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from robotics_env import apply_variation, environment_kwargs, perturb_observation, delayed_action, restore_snapshot


def fake_env():
    model=SimpleNamespace(body_mass=np.array([0.,10.,2.,1.]),body_inertia=np.ones((4,3)),
        body_parentid=np.array([0,0,0,2]),nbody=4,geom_friction=np.ones((3,3)),
        body_pos=np.zeros((4,3)),body_quat=np.tile([1.,0,0,0],(4,1)),
        jnt_type=np.array([0]),jnt_qposadr=np.array([0]),body_jntadr=np.array([-1,-1,0,-1]),
        body_jntnum=np.array([0,0,1,0]),opt=SimpleNamespace(gravity=np.array([0.,0.,-9.81])),
        light_diffuse=np.ones((1,3)),light_ambient=np.ones((1,3)),light_specular=np.ones((1,3)),
        cam_pos=np.zeros((1,3)),cam_bodyid=np.array([0]),camera_names=['agentview'])
    model.body_name2id=lambda name: {'cube':2}[name]
    model.camera_name2id=lambda name: {'agentview':0}[name]
    data=SimpleNamespace(qpos=np.array([.1,.2,.3,1.,0.,0.,0.]),xmat=np.tile(np.eye(3).reshape(9),(4,1)))
    sim=SimpleNamespace(model=model,data=data,forward=lambda:None)
    return SimpleNamespace(sim=sim,model=SimpleNamespace(mujoco_objects=[SimpleNamespace(root_body='cube',name='cube')]))


class EnvironmentTests(unittest.TestCase):
    def test_old_snapshot_requires_reanalysis_instead_of_different_controller_state(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'seed-42.json').write_text(json.dumps({}))
            with self.assertRaisesRegex(ValueError,'Run Analyze again'):
                restore_snapshot(None,directory,42)

    def test_native_saved_reset_preserves_action_trace_for_all_four_controllers(self):
        from setup_robotics import runtime_available
        if not runtime_available()['available']:
            self.skipTest('Optional robotics runtime is not installed.')
        app=Path(__file__).resolve().parents[1]
        result=subprocess.run([sys.executable,'-E',str(app/'tests/robotics_reset_case.py')],
                              cwd=app,text=True,capture_output=True,timeout=120)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        report=json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual({row['controller'] for row in report['checks']},
                         {'OSC_POSE','OSC_POSITION','JOINT_POSITION','JOINT_VELOCITY'})

    def test_mass_changes_task_subtree_and_inertia_not_robot(self):
        env=fake_env(); apply_variation(env,{'mass_scale':1.5},recompute=lambda env:None)
        np.testing.assert_allclose(env.sim.model.body_mass,[0,10,3,1.5])
        np.testing.assert_allclose(env.sim.model.body_inertia[1],1)
        np.testing.assert_allclose(env.sim.model.body_inertia[2:],1.5)

    def test_visual_goal_markers_are_not_moved_with_physical_objects(self):
        env=fake_env()
        env.model.mujoco_objects.append(SimpleNamespace(root_body='goal',name='goal',obj_type='visual',contact_geoms=[]))
        env.sim.model.body_name2id=lambda name: {'cube':2,'goal':1}[name]
        before=env.sim.model.body_pos[1].copy()
        apply_variation(env,{'object_x_offset':.03},recompute=lambda env:None)
        np.testing.assert_equal(env.sim.model.body_pos[1],before)

    def test_object_offsets_do_not_change_robot_and_invalid_groups_fail(self):
        env=fake_env(); apply_variation(env,{'object_x_offset':.03,'object_yaw':15},recompute=lambda env:None)
        self.assertAlmostEqual(env.sim.data.qpos[0],.13)
        self.assertAlmostEqual(np.linalg.norm(env.sim.data.qpos[3:7]),1)
        empty=fake_env(); empty.model.mujoco_objects=[]
        with self.assertRaisesRegex(ValueError,'task object'): apply_variation(empty,{'mass_scale':1.5})
        with self.assertRaises(ValueError): apply_variation(env,{'unknown':1})

    def test_visual_variations_require_policy_cameras(self):
        with self.assertRaisesRegex(ValueError,'camera'):
            apply_variation(fake_env(),{'lighting_scale':.5},recompute=lambda env:None)
        env=fake_env(); apply_variation(env,{'camera_offset':.03},cameras=['agentview'],recompute=lambda env:None)
        self.assertAlmostEqual(env.sim.model.cam_pos[0,0],.03)

    def test_metadata_mismatch_is_explicit_and_camera_mapping_is_applied(self):
        env=dict(task='Lift',robots=['Panda'],controller='OSC_POSE',horizon=10,control_freq=20)
        with self.assertRaisesRegex(ValueError,'Stack'):
            environment_kwargs(env,{'environment':{'task':'Stack'}},{})
        with self.assertRaisesRegex(ValueError,'frequency'):
            environment_kwargs(env,{'environment':{'env_kwargs':{'control_freq':10}}},{})
        with self.assertRaisesRegex(ValueError,'frequency'):
            environment_kwargs(env,{'environment':{'control_freq':10}},{})
        with self.assertRaisesRegex(ValueError,'controller'):
            environment_kwargs(env,{'environment':{'controller':'JOINT_POSITION'}},{})
        metadata={'cameras':[dict(key='observation.images.front',name=None,height=224,width=224)]}
        with self.assertRaisesRegex(ValueError,'observation_map'):
            environment_kwargs(env,metadata,{})
        args=environment_kwargs(env,metadata,{'observation_map':{'observation.images.front':'agentview_image'}})
        self.assertEqual(args['camera_names'],['agentview'])
        self.assertEqual(args['camera_heights'],[224])

    def test_noise_is_reproducible_does_not_corrupt_pixels_or_mutate_input(self):
        obs={'robot0_joint_pos':np.zeros(7),'agentview_image':np.zeros((5,5,3),dtype=np.uint8)}
        one=perturb_observation(obs,.1,np.random.default_rng(42))
        two=perturb_observation(obs,.1,np.random.default_rng(42))
        np.testing.assert_equal(one['robot0_joint_pos'],two['robot0_joint_pos'])
        np.testing.assert_equal(obs['robot0_joint_pos'],0)
        np.testing.assert_equal(one['agentview_image'],obs['agentview_image'])
        queue=[]
        np.testing.assert_equal(delayed_action(np.array([1.,2.]),2,queue),[0,0])
        delayed_action(np.array([3.,4.]),2,queue)
        np.testing.assert_equal(delayed_action(np.array([5.,6.]),2,queue),[1,2])

    def test_old_robosuite_checkpoint_uses_legacy_physics_unless_explicit(self):
        env=dict(task='Lift',robots=['Panda'],controller='OSC_POSE',horizon=10,control_freq=20)
        args=environment_kwargs(env,{'environment':{'env_version':'1.4.1'}},{})
        self.assertIs(args.get('lite_physics'),False)
        explicit=environment_kwargs(env,{'environment':{'env_version':'1.4.1','env_kwargs':{'lite_physics':True}}},{})
        self.assertIs(explicit['lite_physics'],True)
        self.assertNotIn('lite_physics',environment_kwargs(env,{},{}))


if __name__=='__main__': unittest.main()
