"""Put Blocks in COntainer Task."""

import os
import numpy as np
from cliport.tasks.task import Task
from cliport.utils import utils

import random
# import pybullet as p


class PutBlockInContainerUnseenColors(Task):
    """Put Blocks in Bowl base class and task."""

    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "put the {pick} blocks in brown box"
        self.task_completed_desc = "done placing blocks in box."

    def reset(self, env, n_blocks=1, fixed=True):
        super().reset(env)
        if n_blocks is None:
            n_blocks = np.random.randint(1, 5)

        all_color_names = self.get_colors()
        selected_color_names = random.sample(all_color_names, 2)
        colors = [utils.COLORS[cn] for cn in selected_color_names]

        # Add container box.
        zone_size = self.get_random_size(0.05, 0.3, 0.05, 0.3, 0.05, 0.05)
        zone_pose = self.get_random_pose(env, zone_size)
        if fixed:
            zone_size = (0.18, 0.18, 0.05)
            pos = (0.35, -0.4, zone_pose[0][2])
            theta = 0
            rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
            zone_pose = pos, rot
        container_template = 'container/container-template.urdf'
        half = np.float32(zone_size) / 2
        replace = {'DIM': zone_size, 'HALF': half}
        container_urdf = self.fill_template(container_template, replace)
        env.add_object(container_urdf, zone_pose, 'fixed')
        if os.path.exists(container_urdf):
            os.remove(container_urdf)

        # Add blocks.
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = 'stacking/block.urdf'
        object_points = {}
        for _ in range(n_blocks):
            # import pdb; pdb.set_trace()
            block_pose = self.get_random_pose(env, block_size)
            # if fixed:
            # block_pose = (0.45, 0.4, 0.2), (0,0,0,1)
            block_id = env.add_object(block_urdf, block_pose)
            env.pb_client.changeVisualShape(block_id, -1, rgbaColor=colors[0] + [1])
            blocks.append((block_id, (0, None)))
            object_points[block_id] = self.get_box_object_points(block_id)

        poses = self.generate_random_poses(env, zone_pose, zone_size,
                                            block_size, num_objs=n_blocks)
        for i in range(n_blocks):
            b_id = blocks[i][0]
            self.goals.append(([blocks[i]], np.eye(1),
                               [poses[i]], False, True, 'zone',
                               ({b_id: object_points[b_id]},
                                [(zone_pose, zone_size)]), 1/n_blocks))
            self.lang_goals.append(self.lang_template.format(pick=selected_color_names[0]))

        # Only one mistake allowed.
        # self.max_steps = len(blocks) + 1
        self.max_steps = len(blocks)

    def generate_random_poses(self, env, box_pose, box_dims, obj_dim, num_objs = 1):
        coeffs_x = np.array(random.sample(range(-100, 100, 20), num_objs))/100 * 0.20 * box_dims[0]
        coeffs_y = np.array(random.sample(range(-100, 100, 20), num_objs))/100 * 0.20 * box_dims[1]
        obj_poses = []
        for i in range(num_objs):

            pos, ori = box_pose
            _, _, theta = env.pb_client.getEulerFromQuaternion(ori)
            x, y = np.array(pos[:2]) \
                    + coeffs_x[i] * np.array([np.cos(theta), np.sin(theta)]) \
                    + coeffs_y[i] * np.array([np.sin(theta), -np.cos(theta)])
            z = pos[2] + box_dims[2] + obj_dim[2]
            orientation = env.pb_client.getQuaternionFromEuler([0, 0, np.pi*np.random.rand()])
            obj_poses.append(((x, y, z), orientation))

        return obj_poses

    def get_colors(self):
        return utils.TRAIN_COLORS if self.mode == 'train' else utils.EVAL_COLORS

class PutBlockInContainerSeenColors(PutBlockInContainerUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        return utils.TRAIN_COLORS


class PutBlockInContainerFull(PutBlockInContainerUnseenColors):
    def __init__(self):
        super().__init__()

    def get_colors(self):
        all_colors = list(set(utils.TRAIN_COLORS) | set(utils.EVAL_COLORS))
        return all_colors
