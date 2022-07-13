import numpy as np
# import transformations as tf
import pybullet as p
import time
from datetime import datetime
import random
import matplotlib.pyplot as plt
# from pyb_utils.collision import NamedCollisionObject, CollisionDetector
from PIL import Image

class Grasper:
    def __init__(self, env):
        self.env = env
    # def reset(self, object):
    #     self.object = object

    def get_object_pose(self, object_id):
        return self.env.pb_client.getBasePositionAndOrientation(object_id)

    # def compute_destination(self):
    #     pos, _ = self.get_object_pose()
    #     return np.array(pos[:2])

    def compute_pre_pick(self, object_id):
        object_position, object_ori = self.get_object_pose(object_id)

        # theta = tf.euler_from_quaternion(object_ori)[0]
        theta = self.env.pb_client.getEulerFromQuaternion(object_ori)[-1]
        # orientation = tf.quaternion_from_euler(theta + np.pi/2,
        #                                        -np.pi/2,
        #                                        np.pi/2)
        orientation = self.env.pb_client.getQuaternionFromEuler([np.pi/2,
                                                np.pi/2,
                                                theta + np.pi/2])
        pre_pick_z = object_position[2] + 0.2
        position = [*object_position[:2], pre_pick_z]
        return position, orientation

    def compute_pick(self, object_id):
        object_position, object_ori = self.get_object_pose(object_id)
        pick_position = [*object_position[:2], object_position[2] + 0.05]

        return pick_position

    def get_nearest_object(self, objects = None, d_max = 0.6):
        ''' returns nearest objects id which is within d_max distance from
            the end effector.
        '''
        ee_pos, _ = self.env.bot.get_ee_pose()

        dist = []
        # if objects is None:
        #     objects = self.env.object_ids
        for object in objects:
            object_pos, _ = self.env.pb_client.getBasePositionAndOrientation(object)
            d = np.linalg.norm(np.array(object_pos[:2]) - np.array(ee_pos[:2]))
            dist.append(d)
            # print(f'distance from ee {object}: {d}, pos: {object_pos}')
        idx = np.argmin(dist)

        if dist[idx] > d_max:
            return None

        return objects[idx]

    def execute_grasp(self, objects, num_tries = 1):

        self.env.bot.set_camera_grasp_mode()

        success = self.env.bot.move_arm(self.env.bot.actionj)

        for _ in range(num_tries):
            object_id = self.get_nearest_object(objects = objects)
            if object_id is None:
                return False

            pre_pick_pos, pre_pick_ori = self.compute_pre_pick(object_id)
            success = self.env.bot.move_ee(pre_pick_pos, pre_pick_ori, tol = 1e-2)

            # self.env.bot.open_gripper()
            pick_pos = self.compute_pick(object_id)
            success = self.env.bot.move_ee(pick_pos, pre_pick_ori)
            self.env.task.ee.activate()

            success = self.env.bot.move_ee(pre_pick_pos, pre_pick_ori, tol = 1e-2)

            gripper_state = self.env.bot.get_gripper_state()
            if not (gripper_state == 2):
                continue

            self.env.bot.move_arm(self.env.bot.homej)
            # self.env.bot.close_gripper()

            self.env.bot.set_camera_navigation_mode()

            # gripper_state = self.env.bot.get_gripper_state()
            # if gripper_state == 2:
            #     return True

        return True
        return False

    def is_object_grasped(self):
        gripper_state = self.env.bot.get_gripper_state()
        if gripper_state == 2:
            return True
        return False

    def get_nearest_table(self, d_max = 0.6):
        dist = []
        bot_pos, _ = self.env.bot.get_base_pose()
        for table in self.env.table_ids:
            table_pos, _ = self.env.pb_client.getBasePositionAndOrientation(table)
            d = np.linalg.norm(np.array(table_pos[:2]) - np.array(bot_pos[:2]))
            dist.append(d)

        if min(dist) > d_max:
            return None
        else:
            idx = np.argmin(dist)
            return self.env.table_ids[idx]

    def drop_object(self, table = None):
        success = False
        if self.is_object_grasped():

            # self.env.bot.set_camera_grasp_mode()
            object_to_drop = list(self.env.bot.get_bodies_in_gripper())[0]

            # self.env.bot.move_to(drop_pos[:2], tol = 0.4)
            # table = self.get_nearest_table()
            if table is None:
                table = self.get_nearest_table()

            if table is None:
                pos, ori = self.env.bot.get_base_pose()
                # theta = tf.euler_from_quaternion(ori)[0]
                theta = self.env.pb_client.getEulerFromQuaternion(ori)[-1]
                x, y = np.array(pos[:2]) + 0.3*np.array([np.cos(theta), np.sin(theta)])
            else:
                # drop at random location on the table.
                coeff = random.uniform(-0.5, 0.5) * 0.8 * self.env.table_dims[0]
                pos, ori = self.env.pb_client.getBasePositionAndOrientation(table)
                _, _, theta = self.env.pb_client.getEulerFromQuaternion(ori)
                x, y = np.array(pos[:2]) + coeff * np.array([np.cos(theta), np.sin(theta)])

            z = 2 * self.env.table_dims[2] + self.env.object_dims[2]
            drop_pos = [x, y, z]
            drop_ori = self.env.bot.get_ee_pose()[0]

            self.env.bot.move_arm(self.env.bot.actionj)

            drop_pos[2] = drop_pos[2] + 0.15
            success = self.env.bot.move_ee(drop_pos, drop_ori, tol = 1e-2)

            self.env.bot.open_gripper()
            self.env.bot.forward_simulation(20)
            self.env.bot.move_arm(self.env.bot.actionj)
            self.env.bot.close_gripper()
            self.env.bot.move_arm(self.env.bot.homej)

            self.env.forward_simulation(10)
            object_pos = self.env.pb_client.getBasePositionAndOrientation(object_to_drop)[0]
            # print(f'object position: {object_pos}')

            if (self.env.bot.get_gripper_state() == 3) and (object_pos[2] > self.env.table_dims[2]):
                success = True

        # if success:
        #     print('Object dropped successfully!')
        # else:
        #     print('Drop failed!')

        return success


    # def execute_grasp_discrete(self):
    #     self.env.bot.move_arm(self.env.bot.actionj, speed = 0.05)
    #     object_id = self.get_nearest_object()
    #
    #     if object_id is None:
    #         return False
    #
    #     pre_pick_pos, pre_pick_ori = self.compute_pre_pick(object_id)
    #     # print('Pre pick pose', pre_pick_pos, tf.euler_from_quaternion(pre_pick_ori))
    #     # success = self.env.bot.set_ee_pose(position, orientation)
    #     success = self.env.bot.move_ee(pre_pick_pos, pre_pick_ori, tol = 1e-2)
    #     # print('Grapser pre pick pose', self.env.bot.get_ee_pose())
    #
    #     self.env.bot.open_gripper()
    #
    #     pick_pos = self.compute_pick(object_id)
    #     # print('Pick pose', pick_pos, tf.euler_from_quaternion(pre_pick_ori))
    #     # self.env.bot.set_ee_pose(position, orientation)
    #     success = self.env.bot.move_ee(pick_pos, pre_pick_ori)
    #     # print('Grapser pick pose', self.env.bot.get_ee_pose())
    #
    #     self.env.bot.close_gripper()
    #
    #     success = self.env.bot.move_ee(pre_pick_pos, pre_pick_ori, tol = 1e-2)
    #     # print('Post Pick pose', pre_pick_pos, tf.euler_from_quaternion(pre_pick_ori))
    #     # print('Grapser post pick pose', self.env.bot.get_ee_pose())
    #
    #     self.env.bot.move_arm(self.env.bot.homej, speed = 0.05)
    #
    #     self.env.bot.close_gripper()
    #     time.sleep(2)
    #
    #     gripper_state = self.env.bot.get_gripper_state()
    #     if gripper_state == 2:
    #         print(f'Gripper state: {gripper_state}, Object grasped successfully')
    #         return True
    #     else:
    #         print(f'Gripper state: {gripper_state}, Grasp failed')
    #         return False
