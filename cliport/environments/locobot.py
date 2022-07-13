from dataclasses import dataclass

import numpy as np
import pybullet as p
from airobot.utils.common import to_quat
from airobot.utils.common import to_rot_mat
from scipy.spatial.transform import Rotation as R

from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet
from airobot.utils.pb_util import create_pybullet_client
from yacs.config import CfgNode as CN

# from locobot.sim.discrete_env_info import *
from cliport.environments.discrete_env_info import *
from cliport.environments.utils.common import ang_in_mpi_ppi
# import transformations as tf
import time

def _get_default_camera_cfg():
    _C = CN()
    _C.ZNEAR = 0.01
    _C.ZFAR = 10
    _C.WIDTH = 640
    _C.HEIGHT = 480
    _C.FOV = 60
    _ROOT_C = CN()
    _ROOT_C.CAM = CN()
    _ROOT_C.CAM.SIM = _C
    return _ROOT_C.clone()

@dataclass
class Locobot:
    # bot: int

    # def __post_init__(self, env, bot):
    def __init__(self, env, bot):
        self.env = env
        self.bot = bot

        self.wheel_joints = [1, 2]  # Left and right wheels
        self.wheel_default_forward_vel = 40
        self.wheel_default_rotate_vel = 20

        # self.time_long = 80
        # self.time_short = 40
        self.time_long = 100
        self.time_short = 50

        self.time_rotate_long = 60
        self.time_rotate_short = 30

        self.position_controlled = False

        self.distance_forward_long = 0.5
        self.distance_forward_short = 0.3

        self.angle_rotate_long = 0.2
        self.angle_rotate_short = 0.1

        self.arm_joints = [13, 14, 15, 16, 17]  # Arm joints
        # self.gripper_joints = [18, 19]  # Left and right
        self.ee_link = self.arm_joints[-1]  # Link to which ee is attached
        self.ee_tip = self.ee_link + 1
        self.camera_link = 22
        self.camera_motor_joints = [20, 21]

        # some configurations for the joints
        self.homej = np.array([np.pi/2, np.pi/4, -np.pi/4, np.pi/2, 0])  # default config for arm
        self.actionj = np.array([0, 0, -np.pi/12, np.pi/2, 0])

        self.navigation_cam_tilt = NAVIGATION_CAM_TILT
        self.grasping_cam_tilt = GRASPING_CAM_TILT

        # self.homej = np.array([0, 0, 0, 0, 0])  # default config for arm
        self.ee_half_width = 0.025  # half-width of gripper
        self.close_ee_config = np.array([-1e-3, 1e-3])
        self.open_ee_config = self.ee_half_width * np.array([-1, 1])

        # self.arm_kp = np.array([0.08] * 5)
        # self.arm_kd = np.array([0.3] * 5)
        # self.gripper_kp = np.array([0.08] * 2)
        # self.gripper_kd = np.array([0.3] * 2)

        self.arm_kp = np.array([0.03] * 5)
        self.arm_kd = np.array([1.0] * 5)
        self.gripper_kp = np.array([0.08] * 2)
        self.gripper_kd = np.array([0.8] * 2)

        self.camera_motor_kp = np.array([0.03] * 2)
        self.camera_motor_kd = np.array([0.3] * 2)

        self.arm_jnt_max_force = 10
        self.gripper_jnt_max_force = 10
        self.wheel_jnt_max_force = 20

        self.num_joints = self.env.pb_client.getNumJoints(self.bot)
        self.joints_info = [self.env.pb_client.getJointInfo(self.bot, i) for i in range(self.num_joints)]

        camera_offset1 = R.from_rotvec(np.array([0, 1, 0]) * np.pi / 2)
        camera_offset2 = R.from_rotvec(-np.array([0, 0, 1]) * np.pi / 2)
        self.cam_offset = np.dot(camera_offset1.as_matrix(), camera_offset2.as_matrix())

        fp_cam_pos, fp_cam_ori = self.get_locobot_camera_pose()
        # first-person camera
        self.fp_cam = self.create_camera(pos=fp_cam_pos, ori=fp_cam_ori)

    def reset(self):
        self.discrete_steps_taken = 0
        self.step_simulation_calls = 0

        for i in range(len(self.arm_joints)):
            self.env.pb_client.resetJointState(self.bot, self.arm_joints[i], self.homej[i])
        # self.set_camera_grasp_mode()
        self.set_locobot_camera_pan_tilt(0., 0.6)
        # self.forward_simulation()

    def get_fp_images(self):
        # print('entered fp images')
        fp_cam_pos, fp_cam_ori = self.get_locobot_camera_pose()
        # print('obtained camera pose')
        self.fp_cam.set_cam_ext(pos=fp_cam_pos, ori=fp_cam_ori)
        # print('set camera pose done')
        return self.fp_cam.get_images(get_rgb=True,
                                      get_depth=True,
                                      get_seg=True)

    def create_camera(self, pos, ori, cfg=None):
        if cfg is None:
            cfg = _get_default_camera_cfg()
        cam = RGBDCameraPybullet(cfgs=cfg, pb_client=self.env.pb_client)
        cam.set_cam_ext(pos=pos, ori=ori)
        return cam

    def get_camera_config(self):
        _root_c = _get_default_camera_cfg()
        width = _root_c.CAM.SIM.WIDTH
        height = _root_c.CAM.SIM.HEIGHT
        fov = _root_c.CAM.SIM.FOV
        z_near = _root_c.CAM.SIM.ZNEAR
        z_far = _root_c.CAM.SIM.ZFAR

        Cu = width / 2
        Cv = height / 2
        f = width / (2 * np.tan(fov * np.pi / 360))
        intrinsics = (f, 0., Cu, 0., f, Cv, 0., 0., 1.)

        image_size = (height, width)
        fp_cam_pos, fp_cam_ori = self.get_locobot_camera_pose()
        CONFIG = [{
            'image_size': image_size,
            'intrinsics': intrinsics,
            'position': fp_cam_pos,
            'rotation': fp_cam_ori,
            'zrange': (z_near, z_far),
            'noise': False
        }]
        return CONFIG

    def _setup_gripper(self):
        """
        Setup the gripper, pass the robot info from the arm to the gripper.
        Args:
            robot_id (int): robot id in Pybullet.
            jnt_to_id (dict): mapping from the joint name to joint id.
        """

        self.env.pb_client.changeDynamics(self.bot,
                         self.gripper_joints[0],
                         lateralFriction=2.0,
                         spinningFriction=1.0,
                         rollingFriction=1.0)
        self.env.pb_client.changeDynamics(self.bot,
                         self.gripper_joints[1],
                         lateralFriction=2.0,
                         spinningFriction=1.0,
                         rollingFriction=1.0)

    # def _setup_base(self):
    #     """
    #     to remove b3Warning - doesn't work
    #     """
    #     p.changeDynamics(self.bot, -1, mass = 1.0, localInertiaDiagnoal = [1,1,1])

    def get_jpos(self, joints):
        states = self.env.pb_client.getJointStates(self.bot, joints)
        pos = [state[0] for state in states]
        return np.array(pos)

    def get_arm_jpos(self):
        return self.get_jpos(self.arm_joints)

    def get_gripper_jpos(self):
        return self.get_jpos(self.gripper_joints)

    def get_wheel_jpos(self):
        return self.get_jpos(self.wheel_joints)

    def get_base_pose(self):
        pos, ori = self.env.pb_client.getBasePositionAndOrientation(self.bot)
        return np.array(pos), np.array(ori)

    def get_locobot_camera_pose(self):
        info = self.env.pb_client.getLinkState(self.bot, self.camera_link)
        pos = info[4]
        quat = info[5]
        pos = np.array(pos)
        quat = np.array(quat)
        rot_mat = to_rot_mat(quat)
        offset_rot_mat = np.dot(rot_mat, self.cam_offset)
        offset_quat = to_quat(offset_rot_mat)
        return pos, offset_quat

    def set_locobot_camera_pan_tilt(self, pan, tilt, ignore_physics=True):
        if not ignore_physics:
            self.env.pb_client.setJointMotorControlArray(
                bodyIndex=self.bot,
                jointIndices=self.camera_motor_joints,
                controlMode=self.env.pb_client.POSITION_CONTROL,
                targetPositions=[pan, tilt],
                forces=self.arm_jnt_max_force * np.ones(2),
                positionGains=self.camera_motor_kp,
                velocityGains=self.camera_motor_kp)
        else:
            self.env.pb_client.resetJointState(self.bot,
                              self.camera_motor_joints[0],
                              targetValue=pan,
                              targetVelocity=0)
            self.env.pb_client.resetJointState(self.bot,
                              self.camera_motor_joints[1],
                              targetValue=tilt,
                              targetVelocity=0)

    def set_locobot_camera_tilt(self, tilt):
        self.env.pb_client.resetJointState(self.bot,
                          self.camera_motor_joints[1],
                          targetValue=tilt,
                          targetVelocity=0)

    def set_base_vel(self, vels):
        self.env.pb_client.setJointMotorControlArray(bodyIndex=self.bot,
                                    jointIndices=self.wheel_joints,
                                    controlMode=self.env.pb_client.VELOCITY_CONTROL,
                                    targetVelocities=vels,
                                    forces=self.wheel_jnt_max_force * np.ones(2))

    def rotate_to_left(self, vel = None):
        if vel is None:
            vel = self.wheel_default_rotate_vel
        self.set_base_vel([-vel, vel])

    def rotate_to_right(self, vel = None):
        if vel is None:
            vel = self.wheel_default_rotate_vel
        self.set_base_vel([vel, -vel])

    def base_forward(self, vel = None):
        if vel is None:
            vel = self.wheel_default_forward_vel
        self.set_base_vel([vel, vel])

    def base_backward(self, vel = None):
        if vel is None:
            vel = self.wheel_default_forward_vel
        self.set_base_vel([-vel, -vel])

    def stop_base(self):
        self.set_base_vel([0, 0])

    def set_arm_jpos(self, jpos):
        self.env.pb_client.setJointMotorControlArray(
            bodyIndex=self.bot,
            jointIndices=self.arm_joints,
            controlMode=self.env.pb_client.POSITION_CONTROL,
            targetPositions=jpos,
            forces=self.arm_jnt_max_force * np.ones(len(jpos)),
            positionGains=self.arm_kp,
            velocityGains=self.arm_kd)

    def set_gripper_jpos(self, jpos):
        self.env.pb_client.setJointMotorControlArray(
            bodyIndex=self.bot,
            jointIndices=self.gripper_joints,
            controlMode=self.env.pb_client.POSITION_CONTROL,
            targetPositions=jpos,
            forces=self.gripper_jnt_max_force * np.ones(len(jpos)),
            positionGains=self.gripper_kp,
            velocityGains=self.gripper_kd)

    def get_ee_pose(self):
        info = self.env.pb_client.getLinkState(self.bot, self.ee_link)
        pos = info[4]
        quat = info[5]
        return np.array(pos), np.array(quat)

    def set_ee_pose(self, position, orientation):
        joints = self.env.pb_client.calculateInverseKinematics(
            bodyUniqueId=self.bot,
            endEffectorLinkIndex=self.ee_link,
            targetPosition=position,
            targetOrientation=orientation,
            restPoses=self.homej.tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5)
        arm_joints_qs = np.array(joints[2:7])
        self.set_arm_jpos(arm_joints_qs)

    def jpos_from_ee_pose(self, position, orientation):
        joints = self.env.pb_client.calculateInverseKinematics(
            bodyUniqueId=self.bot,
            endEffectorLinkIndex=self.ee_link,
            targetPosition=position,
            targetOrientation=orientation,
            restPoses=self.homej.tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5)
        arm_joints_qs = np.array(joints[2:7])
        arm_joints_qs = ang_in_mpi_ppi(arm_joints_qs)
        if (arm_joints_qs[-1] > np.pi/2):
            arm_joints_qs[-1] -= np.pi
        if (arm_joints_qs[-1] < -np.pi/2):
            arm_joints_qs[-1] += np.pi
        return arm_joints_qs


    # def movep(self, position, orientation, speed = 0.05, tol = 1e-3):
    #     arm_joints_qs = self.jpos_from_ee_pose(position, orientation)
    #     success = self.move_arm(arm_joints_qs, tol = tol)
    #     return success

    # def move_gripper(self, targj, tol = 1e-3, max_steps = 50, t_lim = 5):
    #     success = False

    #     # t0 = time.time()
    #     # while (time.time() - t0) < t_lim:
    #     for i in range(max_steps):
    #         self.set_gripper_jpos(targj)
    #         self.forward_simulation(1)
    #         currj = [self.env.pb_client.getJointState(self.bot, i)[0] for i in self.gripper_joints]
    #         currj = np.array(currj)
    #         diffj = targj - currj
    #         if all(np.abs(diffj) < tol):
    #             success = True
    #             break

    #     return success

    # def forward_simulation(self, nsteps = 1):
    #     # return None
    #     self.env.step_simulation()
    #     # self.step_simulation_calls += nsteps
    #     # for i in range(nsteps):
    #     #     self.env.pb_client.stepSimulation()
