"""Scripted demonstrators used only for starter-policy training labels.

The saved policies do not import this module. Call bootstrap_runtime before
importing it, as with other numeric robotics modules.
"""
from __future__ import annotations

import numpy as np


class StackTeacher:
    horizon = 220
    state_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
                  "cubeA_pos", "cubeA_quat", "cubeB_pos", "cubeB_quat"]

    def reset(self, observation):
        pass

    def action(self, observation, step):
        target = np.array(observation["cubeA_pos"], copy=True)
        destination = np.asarray(observation["cubeB_pos"])
        action = np.zeros(7, dtype=np.float32)
        if step < 30:
            target[2] += .12
            action[-1] = -1
        elif step < 60:
            target[2] += .005
            action[-1] = -1
        elif step < 78:
            target[2] += .005
            action[-1] = 1
        elif step < 108:
            target[2] = 1.15
            action[-1] = 1
        elif step < 140:
            target = destination.copy()
            target[2] = 1.15
            action[-1] = 1
        elif step < 173:
            target = destination.copy()
            target[2] += .06
            action[-1] = 1
        elif step < 193:
            target = destination.copy()
            target[2] += .06
            action[-1] = -1
        else:
            target = destination.copy()
            target[2] += .20
            action[-1] = -1
        action[:3] = np.clip((target - observation["robot0_eef_pos"]) * 15, -1, 1)
        return action


class PickPlaceCanTeacher:
    horizon = 260
    state_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
                  "Can_pos", "Can_quat"]

    def __init__(self, env):
        self.destination = np.array(env.target_bin_placements[env.object_id], copy=True)

    def reset(self, observation):
        pass

    def action(self, observation, step):
        target = np.array(observation["Can_pos"], copy=True)
        action = np.zeros(7, dtype=np.float32)
        if step < 30:
            target[2] += .12
            action[-1] = -1
        elif step < 60:
            target[2] += .005
            action[-1] = -1
        elif step < 78:
            target[2] += .005
            action[-1] = 1
        elif step < 110:
            target[2] = 1.15
            action[-1] = 1
        elif step < 160:
            target = self.destination.copy()
            target[2] = 1.15
            action[-1] = 1
        elif step < 190:
            target = self.destination.copy()
            target[2] += .10
            action[-1] = 1
        elif step < 210:
            target = self.destination.copy()
            target[2] += .10
            action[-1] = -1
        else:
            target = self.destination.copy()
            target[2] += .30
            action[-1] = -1
        action[:3] = np.clip((target - observation["robot0_eef_pos"]) * 15, -1, 1)
        return action


def make_teacher(task, env):
    if task == "Stack":
        return StackTeacher()
    if task == "PickPlaceCan":
        return PickPlaceCanTeacher(env)
    raise ValueError(f"No scripted training teacher for {task}.")
