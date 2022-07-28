"""Motion primitives."""

import numpy as np
from cliport.utils import utils

class LocobotPickPlace():
    """Pick and place primitive."""

    def __init__(self, height=0.2, speed=0.01):
        self.height, self.speed = height, speed

    def __call__(self, env, action):
     # movej, movep, ee, center, pose0, pose1,
     #             navigator, obs_info_fn, turn_fn):
        substep_obs = []
        pick_pose, place_pose = action['pose0'], action['pose1']
        center = action['center']

        # print(f'pick pose: {pick_pose}, place_pose: {place_pose}')
        # input()
        prepick_pos = list(pick_pose[0]);  prepick_pos[2] += self.height
        postpick_pos = list(pick_pose[0]);  postpick_pos[2] += self.height
        prepick_pose = (prepick_pos, pick_pose[1])
        postpick_pose = (postpick_pos, pick_pose[1])

        # print(f'prepick: {prepick_pose}, postpick: {postpick_pose}')

        # if navigator is not None:
        env.motion_planner(prepick_pose[0][:2])
        # print(f'Locobot moved to pick position: {success}')

        substep_obs.append(env.get_obs_wrapper())

        env.movej(p='action')
        env.movep(prepick_pose, tol=1e-2, speed=1.0)
        # Move towards pick pose until contact is detected.
        # print(f'Locobot arm at prepick pose: {success}')
        env.movep(pick_pose, collision_detector=True, tol=1e-3)
        # print(f'Locobot arm at pick pose: {success}')

        env.ee.activate()
        env.movep(postpick_pose)
        # print(f'Locobot arm at postpick pose: {success}')

        pick_success = env.ee.check_grasp()
        env.movej(p='home')

        substep_obs.extend(env.turn_around_center(env.table_center))

        # Execute placing primitive if pick is success ful.
        if pick_success:

            preplace_pos = list(place_pose[0]);  preplace_pos[2] += self.height
            postplace_pos = list(place_pose[0]);  postplace_pos[2] += self.height
            preplace_pose = (preplace_pos, place_pose[1])
            postplace_pose = (postplace_pos, place_pose[1])
            targ_pose = preplace_pose

            env.motion_planner(targ_pose[0][:2])
            # print(f'Locobot moved at place pose: {success}')

            substep_obs.append(env.get_obs_wrapper())

            env.movej(p='action')
            env.movep(preplace_pose)
            # print(f'Locobot arm at preplace pose: {success}')

            env.movep(place_pose, tol=1e-3)
            # print(f'Locobot arm at place pose: {success}')

            env.ee.release()
            env.movep(postplace_pose)
            # print(f'Locobot arm at postplace pose: {success}')

            env.movej(p='home')
            env.turn_to_point(center[:2], tol=np.pi/18)
            success = True
            # substep_obs = [obs1, obs2, obs3]
            # for idx, obs in enumerate(substep_obs):
            #     print('Inside primitives:', idx, obs['configs'][3]['position'])

        # Move to prepick pose if pick is not successful.
        else:
            env.ee.release()
            env.movep(prepick_pose)
            env.movej(p='home')
            env.turn_to_point(center[:2], tol=np.pi/18)
            success = False

        return (not success), substep_obs
