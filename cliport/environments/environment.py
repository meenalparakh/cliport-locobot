"""Environment class."""

import os
import tempfile
import time
import cv2
import imageio

import gym
import numpy as np
from cliport.tasks import cameras
from cliport.utils import pybullet_utils
from cliport.utils import utils

# import pybullet as p

from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet
from airobot.utils.pb_util import create_pybullet_client
from yacs.config import CfgNode as CN
from copy import deepcopy
# import locobot
# from locobot.sim.locobot import Locobot
from cliport.environments.locobot import Locobot
from cliport.environments.utils.common import *

import sys
import random

PLACE_STEP = 0.0003
PLACE_DELTA_THRESHOLD = 0.005

WORKSPACE_URDF_PATH = 'table/table.urdf'
PLANE_URDF_PATH = 'plane/plane.urdf'
LOCOBOT_URDF = 'locobot_description/locobot.urdf'
CUBE_URDF = 'assets/cube/cube.urdf'
FP_CAM_IDX = 1
# NUM_EXPLORATION_IMAGES = 2
IMAGE_PAIRINGS = [[0,1], [2], [3,4], [5]]

## TODO
## (1) fix the image size from camera - currently resizing
##     instead of setting camera config
## (2) hard coded image size - in config - and everywhere
## (3) camera tilt - manage it

class Environment(gym.Env):
    """OpenAI Gym-style environment class."""

    def __init__(self,
                 assets_root,
                 task=None,
                 opengl_render=True,
                 realtime=False,
                 disp=False,
                 shared_memory=False,
                 hz=240,
                 record_cfg=None):
        """Creates OpenAI Gym-style environment with PyBullet.

        Args:
          assets_root: root directory of assets.
          task: the task to use. If None, the user must call set_task for the
            environment to work properly.
          disp: show environment with PyBullet's built-in display viewer.
          shared_memory: run with shared memory.
          hz: PyBullet physics simulation step speed. Set to 480 for deformables.

        Raises:
          RuntimeError: if pybullet cannot load fileIOPlugin.
        """
        # self.n_substeps = n_substesps
        # self.pb_client.setAdditionalSearchPath(locobot.LIB_PATH.joinpath('assets').as_posix())

        self.pix_size = 0.003125
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}
        # self.homej = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.workspace_height = 0.165

        self.agent_cams = cameras.RealSenseD415.CONFIG[1:2]
        # repeat the last comfig until we add a locobot.
        self.agent_cams.append(self.agent_cams[-1])
        self.record_cfg = record_cfg
        self.save_video = False
        self.step_counter = 0

        self.assets_root = assets_root
        self.table_center = [0.5, 0]
        self.num_turns = 2

        color_tuple = [
            gym.spaces.Box(0, 255, config['image_size'] + (3,), dtype=np.uint8)
            for config in self.agent_cams
        ]
        depth_tuple = [
            gym.spaces.Box(0.0, 20.0, config['image_size'], dtype=np.float32)
            for config in self.agent_cams
        ]
        self.observation_space = gym.spaces.Dict({
            'color': gym.spaces.Tuple(color_tuple),
            'depth': gym.spaces.Tuple(depth_tuple),
        })
        self.position_bounds = gym.spaces.Box(
            low=np.array([0.25, -0.5, 0.], dtype=np.float32),
            high=np.array([0.75, 0.5, 0.28], dtype=np.float32),
            shape=(3,),
            dtype=np.float32)
        self.action_space = gym.spaces.Dict({
            'pose0':
                gym.spaces.Tuple(
                    (self.position_bounds,
                     gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32))),
            'pose1':
                gym.spaces.Tuple(
                    (self.position_bounds,
                     gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)))
        })

        self.pb_client = create_pybullet_client(gui=disp,
                                                realtime=realtime,
                                                opengl_render=opengl_render)

        self.pb_client.configureDebugVisualizer(self.pb_client.COV_ENABLE_GUI, 1)
        self.pb_client.setPhysicsEngineParameter(enableFileCaching=0)
        self.pb_client.setAdditionalSearchPath(assets_root)
        self.pb_client.setAdditionalSearchPath(tempfile.gettempdir())
        self.pb_client.setTimeStep(1. / hz)

        if disp:
            target = self.pb_client.getDebugVisualizerCamera()[11]
            self.pb_client.resetDebugVisualizerCamera(
                cameraDistance=1.1,
                cameraYaw=90,
                cameraPitch=-25,
                cameraTargetPosition=target)
        if task:
            self.set_task(task)

    def __del__(self):
        if hasattr(self, 'video_writer'):
            self.video_writer.close()

    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [np.linalg.norm(self.pb_client.getBaseVelocity(i)[0])
             for i in self.obj_ids['rigid']]
        return all(np.array(v) < 5e-3)

    def add_object(self, urdf, pose, category='rigid'):
        """List of (fixed, rigid, or deformable) objects in env."""
        fixed_base = 1 if category == 'fixed' else 0
        obj_id = pybullet_utils.load_urdf(
            self.pb_client,
            os.path.join(self.assets_root, urdf),
            pose[0],
            pose[1],
            useFixedBase=fixed_base)
        if not obj_id is None:
            self.obj_ids[category].append(obj_id)
        return obj_id

    # ---------------------------------------------------------------------------
    # Standard Gym Functions
    # ---------------------------------------------------------------------------

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed

    def add_cube(self, mass, dims, color, pos, ori):
        cube_template = CUBE_URDF
        l, w, h = dims
        r, g, b, a = color
        replace = {'MASS': mass,
                   'LENGTH': l,
                   'WIDTH': w,
                   'HEIGHT': h,
                   'RED': r,
                   'GREEN': g,
                   'BLUE': b,
                   'ALPHA': a}

        urdf = fill_template(cube_template, replace)
        id = self.add_object(urdf, (pos, ori), 'rigid')
        os.remove(urdf)
        return id

    def reset(self):
        """Performs common reset functionality for all supported tasks."""
        if not self.task:
            raise ValueError('environment task must be set. Call set_task or pass '
                             'the task arg in the environment constructor.')
        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}

        self.pb_client.resetSimulation(self.pb_client.RESET_USE_DEFORMABLE_WORLD)
        self.pb_client.setGravity(0, 0, -9.8)

        # Temporarily disable rendering to load scene faster.
        self.pb_client.configureDebugVisualizer(self.pb_client.COV_ENABLE_RENDERING, 0)

        pybullet_utils.load_urdf(self.pb_client,
                                 os.path.join(self.assets_root,
                                              PLANE_URDF_PATH),
                                 [0, 0, -0.001])

        self.workspace = pybullet_utils.load_urdf(
            self.pb_client, os.path.join(self.assets_root, WORKSPACE_URDF_PATH),
                                         [*self.table_center, 0])

        self.ws_edgepts = self.get_ws_edgepts(show=False, margin = 0.25)


        bot_pos = random.choice(self.ws_edgepts[0])
        dx, dy = np.array(self.table_center) - np.array(bot_pos)
        theta = np.arctan2(dy, dx) + np.random.uniform(-np.pi/6, np.pi/6)
        ori = self.pb_client.getQuaternionFromEuler([0, 0, theta])

        self.bot_id = pybullet_utils.load_urdf(self.pb_client,
                                               os.path.join(self.assets_root,
                                                            LOCOBOT_URDF),
                                               [*bot_pos, 0.001], ori)

        self.locobot = Locobot(self, self.bot_id)
        # import pdb; pdb.set_trace()
        self.locobot.reset()

        self.ee = self.task.ee(self.assets_root, self.pb_client, self.bot_id,
                               self.locobot.ee_link, self.obj_ids)
        self.ee_tip = self.locobot.ee_link + 1 #10  # Link ID of suction cup.
        self.ee.release()
        # self.turn_to_point(self.table_center, tol=np.pi/18)
        # Reset task.
        self.task.reset(self)


        self.agent_cams[-1] = self.locobot.get_camera_config(bot_frame=False)
        # Re-enable rendering.
        self.pb_client.configureDebugVisualizer(self.pb_client.COV_ENABLE_RENDERING, 1)

        # self.step_simulation()
        obs, _, _, _ = self.step()
        return obs

    def step(self, action=None):
        """Execute action with specified primitive.
        Args:
          action: action to execute.
        Returns:
          (obs, reward, done, info) tuple containing MDP step data.
        """
        substep_obs = []

        if action is not None:
            timeout, substep_obs = self.task.primitive(self, action)
            for substep in range(len(substep_obs)):
                print('Inside step: substep', substep,
                        substep_obs[substep]['configs'][FP_CAM_IDX]['position'])

            # Exit early if action times out. We still return an observation
            # so that we don't break the Gym API contract.
            if timeout:
                obs = self.get_obs_wrapper()
                # assert False
                return [obs], 0.0, True, self.info

        # Step simulator asynchronously until objects settle.
        while not self.is_static:
            self.step_simulation()

        # Get task rewards.
        reward, info = self.task.reward() if action is not None else (0, {})
        done = self.task.done()

        obs = self.turn_around_center(self.table_center)
        substep_obs.extend(obs)
        # obs = [*substep_obs, *obs]

        info.update(self.info)
        # print(f'type {type(obs)}, {type(info)}')
        return substep_obs, reward, done, info

    def get_obs_wrapper(self):

        while not self.is_static:
            self.step_simulation()

        self.agent_cams[-1] = self.locobot.get_camera_config(bot_frame=False)
        obs = self._get_obs()

        bot_pose = self.locobot.get_base_pose()
        configs = deepcopy(self.agent_cams)
        bot_jpos= self.locobot.get_arm_jpos()
        lang_goal = self.get_lang_goal()

        return {'image': obs,
                'configs': configs,
                'bot_pose': bot_pose,
                'bot_jpos': bot_jpos,
                'lang_goal': lang_goal}

    def step_simulation(self):
        self.pb_client.stepSimulation()
        self.step_counter += 1

        if self.save_video and self.step_counter % 5 == 0:
            self.add_video_frame()

    def render(self, mode='rgb_array'):
        # Render only the color image from the first camera.
        # Only support rgb_array for now.
        if mode != 'rgb_array':
            raise NotImplementedError('Only rgb_array implemented')
        color, _, _ = self.render_camera()
        return color

    def render_camera(self, config, fp=False, image_size=None, shadow=1):
        """Render RGB-D image with specified camera configuration."""

        # if fp:
        #     return self.locobot.get_fp_images()

        if not image_size:
            image_size = config['image_size']

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = self.pb_client.getMatrixFromQuaternion(config['rotation'])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config['position'] + lookdir
        focal_len = config['intrinsics'][0]
        znear, zfar = config['zrange']
        viewm = self.pb_client.computeViewMatrix(config['position'], lookat, updir)
        fovh = (image_size[0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = image_size[1] / image_size[0]
        projm = self.pb_client.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = self.pb_client.getCameraImage(
            width=image_size[1],
            height=image_size[0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=shadow,
            flags=self.pb_client.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=self.pb_client.ER_BULLET_HARDWARE_OPENGL)

        # Get color image.
        color_image_size = (image_size[0], image_size[1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config['noise']:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, image_size))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (image_size[0], image_size[1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = (zfar + znear - (2. * zbuffer - 1.) * (zfar - znear))
        depth = (2. * znear * zfar) / depth
        if config['noise']:
            depth += self._random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm

    @property
    def info(self):
        """Environment info variable with object poses, dimensions, and colors."""

        # Some tasks create and remove zones, so ignore those IDs.
        # removed_ids = []
        # if (isinstance(self.task, tasks.names['cloth-flat-notarget']) or
        #         isinstance(self.task, tasks.names['bag-alone-open'])):
        #   removed_ids.append(self.task.zone_id)

        info = {}  # object id : (position, rotation, dimensions)
        for obj_ids in self.obj_ids.values():
            for obj_id in obj_ids:
                pos, rot = self.pb_client.getBasePositionAndOrientation(obj_id)
                dim = self.pb_client.getVisualShapeData(obj_id)[0][3]
                info[obj_id] = (pos, rot, dim)

        # info['bot_pose'] = self.locobot.get_base_pose()
        # info['cam_configs'] = self.agent_cams
        # info['bot_jpos'] = self.locobot.get_arm_jpos()
        info['lang_goal'] = self.get_lang_goal()
        # print('Bot pose', info['bot_pose'])
        # print('Cam config', info['cam_configs'][3]['position'])
        return info

    def set_task(self, task):
        task.set_assets_root(self.assets_root)
        self.task = task

    def get_lang_goal(self):
        if self.task:
            return self.task.get_lang_goal()
        else:
            raise Exception("No task for was set")

    # ---------------------------------------------------------------------------
    # Robot Movement Functions
    # ---------------------------------------------------------------------------

    def detect_objects_in_ee(self):
        contact_points = self.pb_client.getContactPoints(bodyA = self.bot_id,
                                            linkIndexA = self.locobot.ee_link)
        bodies = set([contact[2] for contact in contact_points])
        bodies = (bodies.difference({self.bot_id})).intersection(self.obj_ids['rigid'])

        return bodies, (len(bodies) > 0)

    def movej(self, targj=None, p=None, tol=1e-2, speed=0.5, max_steps=100,
              collision_detector = False):
        success = False
        collision = False

        if p == 'home':
            targj = self.locobot.homej

        elif p == 'action':
            targj = self.locobot.actionj

        for i in range(max_steps):
            currj = np.array(self.locobot.get_arm_jpos())
            diffj = targj - currj

            if collision_detector:
                collision = self.detect_objects_in_ee()[1]
            if all(np.abs(diffj) < tol) or collision:
                success = True
                break
            delta = diffj
            if np.linalg.norm(delta) > speed:
                delta = speed*diffj/(np.linalg.norm(delta))
            new_targj = currj + delta
            self.locobot.set_arm_jpos(new_targj)
            self.step_simulation()

        return success

    def start_rec(self, video_filename):
        assert self.record_cfg

        # make video directory
        if not os.path.exists(self.record_cfg['save_video_path']):
            os.makedirs(self.record_cfg['save_video_path'])

        # close and save existing writer
        if hasattr(self, 'video_writer'):
            self.video_writer.close()

        # initialize writer
        self.video_writer = imageio.get_writer(os.path.join(self.record_cfg['save_video_path'],
                                                            f"{video_filename}.mp4"),
                                               fps=self.record_cfg['fps'],
                                               format='FFMPEG',
                                               codec='h264',)
        self.pb_client.setRealTimeSimulation(False)
        self.save_video = True

    def end_rec(self):
        if hasattr(self, 'video_writer'):
            self.video_writer.close()

        self.pb_client.setRealTimeSimulation(True)
        self.save_video = False

    def add_video_frame(self):
        # Render frame.
        config = self.agent_cams[0]
        image_size = (self.record_cfg['video_height'], self.record_cfg['video_width'])
        color, depth, _ = self.render_camera(config, image_size, shadow=0)
        color = np.array(color)

        # Add language instruction to video.
        if self.record_cfg['add_text']:
            lang_goal = self.get_lang_goal()
            reward = f"Success: {self.task.get_reward():.3f}"

            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.65
            font_thickness = 1

            # Write language goal.
            lang_textsize = cv2.getTextSize(lang_goal, font, font_scale, font_thickness)[0]
            lang_textX = (image_size[1] - lang_textsize[0]) // 2

            color = cv2.putText(color, lang_goal, org=(lang_textX, 600),
                                fontScale=font_scale,
                                fontFace=font,
                                color=(0, 0, 0),
                                thickness=font_thickness, lineType=cv2.LINE_AA)

            ## Write Reward.
            # reward_textsize = cv2.getTextSize(reward, font, font_scale, font_thickness)[0]
            # reward_textX = (image_size[1] - reward_textsize[0]) // 2
            #
            # color = cv2.putText(color, reward, org=(reward_textX, 634),
            #                     fontScale=font_scale,
            #                     fontFace=font,
            #                     color=(0, 0, 0),
            #                     thickness=font_thickness, lineType=cv2.LINE_AA)

            color = np.array(color)

        self.video_writer.append_data(color)

    def movep(self, pose, speed=0.5, tol=1e-2, collision_detector=False):
        """Move UR5 to target end effector pose."""
        arm_joints_qs = self.locobot.jpos_from_ee_pose(pose[0], pose[1])
        success = self.movej(arm_joints_qs, speed=speed, tol = tol,
                             collision_detector=collision_detector)
        # print('inside movep', success)
        return success

    def rotate_base(self, theta, relative = True, t_lim = 20, tol = 1e-3):
        success = False
        _, init_orientation = self.locobot.get_base_pose()
        init_angle = self.pb_client.getEulerFromQuaternion(init_orientation)[-1]

        if relative:
            target_angle = init_angle + theta
        else:
            target_angle = theta

        t0 = time.time()
        while (time.time() - t0) < t_lim:
            _, curr_orientation = self.locobot.get_base_pose()
            curr_angle = self.pb_client.getEulerFromQuaternion(curr_orientation)[-1]

            diffj = (target_angle - curr_angle)
            diffj = ang_in_mpi_ppi(diffj)

            if np.abs(diffj) < tol:
                self.locobot.stop_base()
                success = True
                break

            vel = self.locobot.wheel_default_rotate_vel
            # if abs(diffj) < 0.75:
            #     vel = 10.0
            if abs(diffj) < 0.25:
                vel = self.locobot.wheel_default_rotate_vel/2
            if diffj > 0:
                self.locobot.rotate_to_left(vel)
            else:
                self.locobot.rotate_to_right(vel)

            self.step_simulation()

        for _ in range(20):
            self.step_simulation()

        return success

    def move_to(self, target_position, t_lim=100,
                tol=0.1,
                direction_error_threshold=np.pi/6,
                skip_starting_rotation=False,
                additional_target_position=None):

        success = False
        found = False

        currj = np.array(self.locobot.get_base_pose()[0][:2])
        dx, dy = target_position - currj
        target_direction = np.arctan2(dy, dx)

        if not skip_starting_rotation:
            self.rotate_base(target_direction, relative = False)

        t0 = time.time()
        while (time.time() - t0) < t_lim:
            currj = np.array(self.locobot.get_base_pose()[0][:2])
            diffj = target_position - currj

            if np.linalg.norm(diffj) < tol:
                self.locobot.stop_base()
                success = True
                break

            if additional_target_position is not None:
                d_ = np.linalg.norm(additional_target_position - currj)
                if d_ < 0.45:
                    self.locobot.stop_base()
                    found = True
                    break

            ##############  Check if the bot deviated ##############
            dx, dy = diffj
            target_direction = np.arctan2(dy, dx)

            current_direction = self.pb_client.getEulerFromQuaternion(
                self.locobot.get_base_pose()[1])[-1]

            sin_direction_error = np.abs(np.sin(target_direction - current_direction))
            if (sin_direction_error > np.sin(direction_error_threshold)):
                self.rotate_base(target_direction, relative = False)

            #########################################################
            norm = np.linalg.norm([dx, dy])
            vel = self.locobot.wheel_default_forward_vel
            if norm < 0.75:
                vel = self.locobot.wheel_default_forward_vel/2
            if norm < 0.2:
                vel = 10.0
            self.locobot.base_forward(vel)
            self.step_simulation()

        for _ in range(20):
            self.step_simulation()

        return success, found

    def set_camera_navigation_mode(self):
        self.locobot.set_locobot_camera_tilt(
                self.locobot.navigation_cam_tilt)
        self.step_simulation()

    def set_camera_grasp_mode(self):
        self.locobot.set_locobot_camera_tilt(
                self.locobot.grasping_cam_tilt)
        self.step_simulation()

    def turn_to_point(self, pos, tol = np.pi/6):
        bot_pos = self.locobot.get_base_pose()[0][:2]
        dx, dy = np.array(pos) - np.array(bot_pos)
        theta = np.arctan2(dy, dx)
        success = self.rotate_base(theta, relative = False, tol = tol)
        return success

    def turn_around_center(self, center):
        obs = []
        bot_pos = self.locobot.get_base_pose()[0]
        dx, dy = np.array(center) - np.array(bot_pos[:2])
        theta = np.arctan2(dy, dx)

        # self.rotate_base(theta-np.pi/6, relative=False)
        # while not self.is_static:
        #     self.step_simulation()

        ori = self.pb_client.getQuaternionFromEuler([0, 0, theta])
        self.pb_client.resetBasePositionAndOrientation(self.bot_id, bot_pos, ori)
        self.locobot.set_locobot_camera_pan_tilt(-np.pi/6, 0.6)

        for i in range(10):
            self.step_simulation()
        obs.append(self.get_obs_wrapper())

        # ori = self.pb_client.getQuaternionFromEuler([0, 0, theta+np.pi/6])
        # self.pb_client.resetBasePositionAndOrientation(self.bot_id, bot_pos, ori)
        self.locobot.set_locobot_camera_pan_tilt(np.pi/6, 0.6)
        for i in range(10):
            self.step_simulation()
        obs.append(self.get_obs_wrapper())

        return obs

    def motion_planner(self, target_pos, tol_dist=0.52, tol_angle=np.pi/6):

        target_idx= None
        for id, pt in enumerate(self.ws_edgepts[0]):
            # print(id, pt)
            d = np.linalg.norm(np.array(pt) - target_pos)
            # print(f'Distance of {target_pos} from {id} ({pt}): {d}, ')
            if d < tol_dist:
                target_idx = id
                print(f'Target index: {target_idx}')
                break
        # assert False
        if target_idx is None:
            import pdb; pdb.set_trace()
            raise RuntimeError(f'No edgepoint is within {tol_dist} of edgepoints.')

        success = self.movej(p='home')
        self.locobot.set_locobot_camera_pan_tilt(0, 0.6)

        print(f'Setting pose to homej: {success}')
        # success &= _move(pt1_idx, target_idx, target_pos)
        edge_pt = np.array(self.ws_edgepts[0][target_idx])
        dx, dy = np.array(target_pos) - np.array(edge_pt)

        eps = min(max(tol_angle*np.random.normal(), -tol_angle), tol_angle)
        theta = np.arctan2(dy, dx) + eps
        ori = self.pb_client.getQuaternionFromEuler([0, 0, theta])
        self.pb_client.resetBasePositionAndOrientation(self.bot_id, [*edge_pt, 0.001], ori)

        while not self.is_static:
            self.step_simulation()

        return success

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------

    def _get_obs(self):
        # Get RGB-D camera image observations.
        obs = {'color': (), 'depth': ()}
        for config in self.agent_cams:
            color, depth, _ = self.render_camera(config)
            obs['color'] += (color,)
            obs['depth'] += (depth,)

        # color, depth, _ = self.render_camera(None, fp=True)
        # obs['color'] += (color,)
        # obs['depth'] += (depth,)
        return obs

    def get_ws_edgepts(self, margin=0.25, show=False):
        xlim = (0.25, 0.75)
        ylim = (-0.5, 0.5)
        diag = margin/np.sqrt(2)
        edge1 = [(xlim[0]-margin, i) for i in np.linspace(*ylim, num=4)][1:-1]
        edge2 = [(i, ylim[1] + margin) for i in np.linspace(*xlim, num=3)][1:-1]
        edge3 = [(xlim[1] + margin, i) for i in np.linspace(*ylim, num=4)][1:-1]
        edge4 = [(i, ylim[0] - margin) for i in np.linspace(*xlim, num=3)][1:-1]
        edge3.reverse(); edge4.reverse()

        diag1 = (xlim[0] - diag, ylim[1] + diag)
        diag2 = (xlim[1] + diag, ylim[1] + diag)
        diag3 = (xlim[1] + diag, ylim[0] - diag)
        diag4 = (xlim[0] - diag, ylim[0] - diag)

        # edgepoints = [diag4, *edge1, diag1, *edge2, diag2, *edge3, diag3, *edge4]
        edgepoints = [*edge1, *edge2, *edge3, *edge4]
        if show:
            for pt in edgepoints:
                self.add_cube(0, [0.01, 0.01, 0.01],
                              (0.8, 0, 0, 1),
                              [*pt, self.workspace_height],
                              (0, 0, 0, 1))
        return edgepoints, None

class EnvironmentNoRotationsWithHeightmap(Environment):
    """Environment that disables any rotations and always passes [0, 0, 0, 1]."""

    def __init__(self,
                 assets_root,
                 task=None,
                 disp=False,
                 shared_memory=False,
                 hz=240):
        super(EnvironmentNoRotationsWithHeightmap,
              self).__init__(assets_root, task, disp, shared_memory, hz)

        heightmap_tuple = [
            gym.spaces.Box(0.0, 20.0, (320, 160, 3), dtype=np.float32),
            gym.spaces.Box(0.0, 20.0, (320, 160), dtype=np.float32),
        ]
        self.observation_space = gym.spaces.Dict({
            'heightmap': gym.spaces.Tuple(heightmap_tuple),
        })
        self.action_space = gym.spaces.Dict({
            'pose0': gym.spaces.Tuple((self.position_bounds,)),
            'pose1': gym.spaces.Tuple((self.position_bounds,))
        })

    def step(self, action=None):
        """Execute action with specified primitive.

        Args:
          action: action to execute.

        Returns:
          (obs, reward, done, info) tuple containing MDP step data.
        """
        if action is not None:
            action = {
                'pose0': (action['pose0'][0], [0., 0., 0., 1.]),
                'pose1': (action['pose1'][0], [0., 0., 0., 1.]),
            }
        return super(EnvironmentNoRotationsWithHeightmap, self).step(action)

    def _get_obs(self):
        obs = {}

        color_depth_obs = {'color': (), 'depth': ()}
        for config in self.agent_cams:
            color, depth, _ = self.render_camera(config)
            color_depth_obs['color'] += (color,)
            color_depth_obs['depth'] += (depth,)
        cmap, hmap = utils.get_fused_heightmap(color_depth_obs, self.agent_cams,
                                               self.task.bounds, pix_size=0.003125)
        obs['heightmap'] = (cmap, hmap)
        return obs
