"""Motion primitives."""

import numpy as np
from cliport.utils import utils

class LocobotPickPlace():
    """Pick and place primitive."""

    def __init__(self, height=0.2, speed=0.01):
        self.height, self.speed = height, speed

    def __call__(self, env, action):
     # movej, movep, ee, center, pose0, pose1,
     #             navigator, obs_info_fn, turn_fn):]
        substep_obs = []
        pick_pose, place_pose = action['pose0'], action['pose1']
        center = action['center']
        env.motion_planner(pick_pose[0][:2])
        substep_obs.append(env.get_obs_wrapper())

        self.pick(env, pick_pose)
        substep_obs.extend(env.turn_around_center(env.table_center))

        if pick_success:
            env.motion_planner(place_pose[0][:2])
            substep_obs.append(env.get_obs_wrapper())

            self.place(env, place_pose)

            env.turn_to_point(center[:2], tol=np.pi/18)
            success = True

        # Move to prepick pose if pick is not successful.
        else:
            env.ee.release()
            env.movep(prepick_pose)
            env.movej(p='home')
            env.turn_to_point(center[:2], tol=np.pi/18)
            success = False

        return (not success), substep_obs

    def pick(self, env, pick_pose):
        prepick_pos = list(pick_pose[0]);  prepick_pos[2] += self.height
        postpick_pos = list(pick_pose[0]);  postpick_pos[2] += self.height
        prepick_pose = (prepick_pos, pick_pose[1])
        postpick_pose = (postpick_pos, pick_pose[1])

        env.movej(p='action')
        env.movep(prepick_pose, tol=1e-2, speed=1.0)
        env.movep(pick_pose, collision_detector=True, tol=1e-3)

        env.ee.activate()
        env.movep(postpick_pose)

        pick_success = env.ee.check_grasp()
        env.movej(p='home')
        return pick_success

    def place(self, env, place_pose):
        preplace_pos = list(place_pose[0]);  preplace_pos[2] += self.height
        postplace_pos = list(place_pose[0]);  postplace_pos[2] += self.height
        preplace_pose = (preplace_pos, place_pose[1])
        postplace_pose = (postplace_pos, place_pose[1])
        targ_pose = preplace_pose

        env.movej(p='action')
        env.movep(preplace_pose)
        env.movep(place_pose, tol=1e-3)

        env.ee.release()
        env.movep(postplace_pose)

        env.movej(p='home')
