"""Motion primitives."""

import numpy as np
from cliport.utils import utils


class PickPlace():
    """Pick and place primitive."""

    def __init__(self, height=0.32, speed=0.01):
        self.height, self.speed = height, speed

    def __call__(self, movej, movep, ee, pose0, pose1):
        """Execute pick and place primitive.

        Args:
          movej: function to move robot joints.
          movep: function to move robot end effector pose.
          ee: robot end effector.
          pose0: SE(3) picking pose.
          pose1: SE(3) placing pose.

        Returns:
          timeout: robot movement timed out if True.
        """

        pick_pose, place_pose = pose0, pose1

        # Execute picking primitive.
        prepick_to_pick = ((0, 0, 0.32), (0, 0, 0, 1))
        postpick_to_pick = ((0, 0, self.height), (0, 0, 0, 1))
        prepick_pose = utils.multiply(pick_pose, prepick_to_pick)
        postpick_pose = utils.multiply(pick_pose, postpick_to_pick)
        timeout = movep(prepick_pose)

        # Move towards pick pose until contact is detected.
        delta = (np.float32([0, 0, -0.001]),
                 utils.eulerXYZ_to_quatXYZW((0, 0, 0)))
        targ_pose = prepick_pose
        while not ee.detect_contact():  # and target_pose[2] > 0:
            targ_pose = utils.multiply(targ_pose, delta)
            timeout |= movep(targ_pose)
            if timeout:
                return True

        # Activate end effector, move up, and check picking success.
        ee.activate()
        timeout |= movep(postpick_pose, self.speed)
        pick_success = ee.check_grasp()

        # Execute placing primitive if pick is successful.
        if pick_success:
            preplace_to_place = ((0, 0, self.height), (0, 0, 0, 1))
            postplace_to_place = ((0, 0, 0.32), (0, 0, 0, 1))
            preplace_pose = utils.multiply(place_pose, preplace_to_place)
            postplace_pose = utils.multiply(place_pose, postplace_to_place)
            targ_pose = preplace_pose
            while not ee.detect_contact():
                targ_pose = utils.multiply(targ_pose, delta)
                timeout |= movep(targ_pose, self.speed)
                if timeout:
                    return True
            ee.release()
            timeout |= movep(postplace_pose)

        # Move to prepick pose if pick is not successful.
        else:
            ee.release()
            timeout |= movep(prepick_pose)

        return timeout, None

class LocobotPickPlace():
    """Pick and place primitive."""

    def __init__(self, height=0.2, speed=0.01):
        self.height, self.speed = height, speed

    def __call__(self, movej, movep, ee, pose0, pose1,
                 navigator=None, obs_info_fn=None):
        pick_pose, place_pose = pose0, pose1

        print(f'pick pose: {pick_pose}, place_pose: {place_pose}')
        # input()

        prepick_pos = list(pick_pose[0]);  prepick_pos[2] += self.height
        postpick_pos = list(pick_pose[0]);  postpick_pos[2] += self.height
        prepick_pose = (prepick_pos, pick_pose[1])
        postpick_pose = (postpick_pos, pick_pose[1])

        print(f'prepick: {prepick_pose}, postpick: {postpick_pose}')

        # if navigator is not None:
        success = navigator(prepick_pose[0][:2])
        print(f'Locobot moved to pick position: {success}')

        obs1, info1 = obs_info_fn()

        success &= movej('action')
        success &= movep(prepick_pose, tol=1e-2, speed=1.0)
        # Move towards pick pose until contact is detected.
        print(f'Locobot arm at prepick pose: {success}')
        success &= movep(pick_pose, collision_detector=True, tol=1e-3)
        print(f'Locobot arm at pick pose: {success}')

        ee.activate()
        success &= movep(postpick_pose)
        print(f'Locobot arm at postpick pose: {success}')

        pick_success = ee.check_grasp()

        # Execute placing primitive if pick is successful.
        if pick_success:

            preplace_pos = list(place_pose[0]);  preplace_pos[2] += self.height
            postplace_pos = list(place_pose[0]);  postplace_pos[2] += self.height
            preplace_pose = (preplace_pos, place_pose[1])
            postplace_pose = (postplace_pos, place_pose[1])
            targ_pose = preplace_pose

            success &= navigator(targ_pose[0][:2])
            print(f'Locobot moved at place pose: {success}')

            obs2, info2 = obs_info_fn()

            success &= movej('action')
            success &= movep(preplace_pose)
            print(f'Locobot arm at preplace pose: {success}')

            success &= movep(place_pose, tol=1e-3)
            print(f'Locobot arm at place pose: {success}')

            ee.release()
            success &= movep(postplace_pose)
            print(f'Locobot arm at postplace pose: {success}')

            success &= movej('home')
            success = True
            additional_info = {'obs': [obs1, obs2],
                               'info': [info1, info2]}
        # Move to prepick pose if pick is not successful.
        else:
            ee.release()
            movep(prepick_pose)
            movej('home')
            success = False
            additional_info = None

        return (not success), additional_info

def push(movej, movep, ee, pose0, pose1):  # pylint: disable=unused-argument
    """Execute pushing primitive.

    Args:
      movej: function to move robot joints.
      movep: function to move robot end effector pose.
      ee: robot end effector.
      pose0: SE(3) starting pose.
      pose1: SE(3) ending pose.

    Returns:
      timeout: robot movement timed out if True.
    """

    # Adjust push start and end positions.
    pos0 = np.float32((pose0[0][0], pose0[0][1], 0.005))
    pos1 = np.float32((pose1[0][0], pose1[0][1], 0.005))
    vec = np.float32(pos1) - np.float32(pos0)
    length = np.linalg.norm(vec)
    vec = vec / length
    pos0 -= vec * 0.02
    pos1 -= vec * 0.05

    # Align spatula against push direction.
    theta = np.arctan2(vec[1], vec[0])
    rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))

    over0 = (pos0[0], pos0[1], 0.31)
    over1 = (pos1[0], pos1[1], 0.31)

    # Execute push.
    timeout = movep((over0, rot))
    timeout |= movep((pos0, rot))
    n_push = np.int32(np.floor(np.linalg.norm(pos1 - pos0) / 0.01))
    for _ in range(n_push):
        target = pos0 + vec * n_push * 0.01
        timeout |= movep((target, rot), speed=0.003)
    timeout |= movep((pos1, rot), speed=0.003)
    timeout |= movep((over1, rot))
    return timeout, None
