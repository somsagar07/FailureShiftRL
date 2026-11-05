"""Opt-in real EGL simulation checks. No trained-policy quality claims.

Run: .venv/bin/python -E tests/robotics_smoke.py
Artifacts: runs/verification/robotics-20261101/simulation
"""
from pathlib import Path
import json
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from setup_robotics import bootstrap_runtime
bootstrap_runtime()
import numpy as np
import torch
from robotics import RoboticsManager, normalize_config, GROUPS
from robotics_env import create_environment, seed_episode, save_snapshot, restore_snapshot, apply_variation


def main():
    root=Path(__file__).resolve().parents[1]/'runs/verification/robotics-20261101/simulation'
    root.mkdir(parents=True,exist_ok=True)
    class Zero(torch.nn.Module):
        def forward(self,state): return state[:,:7]*0
    checkpoint=root/'zero-smoke-policy.pt'
    torch.jit.trace(Zero(),torch.zeros(1,7)).save(str(checkpoint))
    config=normalize_config(dict(policy=dict(format='torchscript',source='local',name=str(checkpoint)),
        mapping={'state_keys':['robot0_joint_pos']},environment={'task':'Lift','horizon':8},
        action_space={'mass_scale':[.5,1,1.5],'object_x_offset':[-.03,0,.03]},
        episodes_per_variation=2,variation_budget=2,device='cpu'))
    env,_=create_environment(config,{})
    try:
        seed_episode(42,env); env.reset(); snapshots=root/'snapshots'; save_snapshot(env,snapshots,42)
        initial=restore_snapshot(env,snapshots,42)
        qpos=env.sim.data.qpos.copy(); mass=env.sim.model.body_mass.copy(); friction=env.sim.model.geom_friction.copy()
        rng_state=env.rng.bit_generator.state
        for group in GROUPS:
            restore_snapshot(env,snapshots,42)
            nonneutral=next(v for v in group['values'] if v!=group['neutral'])
            apply_variation(env,{group['id']:nonneutral},cameras=['agentview'])
            env.step(np.zeros(env.action_dim))
        restored=restore_snapshot(env,snapshots,42)
        np.testing.assert_array_equal(env.sim.data.qpos,qpos)
        np.testing.assert_array_equal(env.sim.model.body_mass,mass)
        np.testing.assert_array_equal(env.sim.model.geom_friction,friction)
        np.testing.assert_allclose(restored['robot0_joint_pos'],initial['robot0_joint_pos'],atol=1e-12)
        assert env.rng.bit_generator.state==rng_state
    finally: env.close()
    manager=RoboticsManager(root/'jobs')
    def wait(job):
        deadline=time.monotonic()+180
        while job['status'] in ('queued','running') and time.monotonic()<deadline:
            time.sleep(.3); job=manager.get(job['id'])
        print(job['mode'],job['status'],job.get('error',job['message']),flush=True)
        assert job['status']=='completed',job
        return job
    try:
        baseline=wait(manager.start(config))
        assert baseline['baseline']['episodes']==2 and baseline['baseline']['errors']==0,baseline['baseline']
        discovery=wait(manager.start(dict(mode='discover',analysis_id=baseline['id'])))
        measured=[n for n in discovery['nodes'] if n['episodes']]
        assert len(measured)==3 and all(n['episodes']==2 and n['errors']==0 for n in measured),discovery['nodes']
        assert any(n.get('reused_baseline') for n in measured)
        for job in (baseline,discovery):
            for record in job['baseline']['records']+job['records']:
                assert not record.get('media_warning'),record
                for key in ('image','video'):
                    if record.get(key):
                        parts=record[key].split('/'); assert manager.artifact(parts[2],parts[-1]).stat().st_size>0
        result={'baseline_id':baseline['id'],'discovery_id':discovery['id'],
            'baseline_episodes':baseline['baseline']['episodes'],'measured_variations':len(measured),
            'checks':['all ten intervention groups step real simulation','snapshot state/physics restoration',
                      'real TorchScript baseline and paired sweep','neutral reuse','JPEG and MP4 artifacts'],
            'policy':'untrained zero-action TorchScript fixture; plumbing verification only'}
        (root/'verification.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    finally: manager.close()


if __name__=='__main__': main()
