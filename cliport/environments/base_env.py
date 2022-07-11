import numpy as np
import pybullet as p
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet
from airobot.utils.pb_util import create_pybullet_client
from yacs.config import CfgNode as CN

import locobot
from locobot.sim.locobot import Locobot

import os
import sys
import random
from locobot.utils.common import *

class BaseEnv:
    def __init__(self, gui=True, realtime=False, opengl_render=True, n_substesps=10):
        self.n_substeps = n_substesps
        self.pb_client = create_pybullet_client(gui=gui, realtime=realtime, opengl_render=opengl_render)
        self.pb_client.setAdditionalSearchPath(locobot.LIB_PATH.joinpath('assets').as_posix())

    def reset(self, tables = True, boundary = True, furniture = False, num_tables = 2, table_dims = None):
        self.pb_client.resetSimulation()
        self.pb_client.setGravity(0, 0, -9.8)
        self.pb_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        self.plane_id = self.pb_client.loadURDF('plane/plane.urdf', [0, 0, -0.001])

        # positions = self.get_table_positions_grid(5)
        positions = self.get_table_and_bot_positions_circle(1+num_tables)
        # ori = tf.quaternion_from_euler(np.random.uniform(-np.pi, np.pi), 0, np.pi)
        ori = self.pb_client.getQuaternionFromEuler([0, 0, np.random.uniform(-np.pi, np.pi)])

        self.bot_id = self.pb_client.loadURDF('locobot_description/locobot.urdf',
                                                [positions[0, 0], positions[0, 1], 0.001], ori)

        self.bot = Locobot(self, self.bot_id)
        # self.bot.env = self
        self.bot.reset()
        self.bot.set_locobot_camera_pan_tilt(0., 0.6)
        fp_cam_pos, fp_cam_ori = self.bot.get_locobot_camera_pose()
        # first-person camera
        self.fp_cam = self.create_camera(pos=fp_cam_pos, ori=fp_cam_ori)
        # third-person camera
        # self.tp_cam = self.create_camera(pos=np.array([0, 0, 4.5]), ori=np.array([0.707, -0.707, -0., -0.]))
        self.tp_cam = self.create_camera(pos=np.array([0, 0, 2.5]), ori=np.array([0.707, -0.707, -0., -0.]))

        self.boundary_length = 1.5
        self.num_objects_per_table = 1

        if table_dims is None:
            self.table_dims = [0.10, 0.08, 0.06]
        else:
            self.table_dims = table_dims

        YELLOW = [0.8, 0.8, 0.1]
        self.table_colors = [YELLOW, RED, BLACK, GREEN]
        self.table_colors_names = ['yellow', 'red', 'black', 'green']
        self.object_dims = [0.05, 0.025, 0.025]
        self.table_mass = 5.0
        self.cube_mass = 0.01
        self.table_ids = []
        self.object_ids = []

        if boundary:
            self.add_boundary()

        if tables:
            self.num_tables = num_tables
            self.add_tables_and_objects(positions[1:, :], self.num_tables)

        if furniture:
            self.add_furniture()

        # self.pb_client.configureDebugVisualizer(self.pb_client.COV_ENABLE_RENDERING, 1)
        return None

    def get_fp_images(self):
        # print('entered fp images')
        fp_cam_pos, fp_cam_ori = self.bot.get_locobot_camera_pose()
        # print('obtained camera pose')
        self.fp_cam.set_cam_ext(pos=fp_cam_pos, ori=fp_cam_ori)
        # print('set camera pose done')
        return self.fp_cam.get_images()

    def get_tp_images(self):
        return self.tp_cam.get_images()

    def create_camera(self, pos, ori, cfg=None):
        if cfg is None:
            cfg = self._get_default_camera_cfg()
        cam = RGBDCameraPybullet(cfgs=cfg, pb_client=self.pb_client)
        cam.set_cam_ext(pos=pos, ori=ori)
        return cam

    def forward_simulation(self, nsteps=None):
        # return None
        if nsteps is None:
            nsteps = self.n_substeps
        for i in range(nsteps):
            self.pb_client.stepSimulation()

    def _get_default_camera_cfg(self):
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

    def get_intrinsic_matrix(self):
        '''
        CHECK IF THIS IS CORRECT
        '''
        _root_c = self._get_default_camera_cfg()
        width = _root_c.CAM.SIM.WIDTH
        height = _root_c.CAM.SIM.HEIGHT
        fov = _root_c.CAM.SIM.FOV

        Cu = width / 2
        Cv = height / 2
        f = width / (2 * np.tan(fov * np.pi / 360))
        K = np.array([[f, 0, Cu],
                      [0, f, Cv],
                      [0, 0, 1 ]])
        return K

    def add_boundary(self):
        half_thickness = 0.02
        half_height = 0.2
        half_length = self.boundary_length
        self.pb_client.load_geom(shape_type='box',
                                 size=[half_length, half_thickness, half_height],
                                 mass=0,
                                 rgba=[0.6, 0.4, 0.2, 1],
                                 base_pos=[0, half_length, half_height])
        self.pb_client.load_geom(shape_type='box',
                                 size=[half_thickness, half_length, half_height],
                                 mass=0,
                                 rgba=[0.6, 0.4, 0.2, 1],
                                 base_pos=[half_length, 0, half_height])
        self.pb_client.load_geom(shape_type='box',
                                 size=[half_length, half_thickness, half_height],
                                 mass=0,
                                 rgba=[0.6, 0.4, 0.2, 1],
                                 base_pos=[0, -half_length, half_height])
        self.pb_client.load_geom(shape_type='box',
                                 size=[half_thickness, half_length, half_height],
                                 mass=0,
                                 rgba=[0.6, 0.4, 0.2, 1],
                                 base_pos=[-half_length, 0, half_height])

    def add_table(self, mass, dims, color, pos, ori):

        return self.pb_client.load_geom(shape_type='box',
                                         size = dims,
                                         mass = mass,
                                         rgba = [*color, 1],
                                         base_pos = pos,
                                         base_ori = ori)

    def add_cube(self, mass, dims, color, pos, ori):
        cube_template = 'assets/cube/cube.urdf'
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
        cube_id = self.pb_client.loadURDF(urdf, pos, ori)
        os.remove(urdf)
        return cube_id

    def generate_random_cubes(self, cube_dims, num_cubes = 1, table_id = None, exclude_colors = None):
        coeffs_x = np.array(random.sample(range(-100, 100, 20), num_cubes))/100 * 0.8 * self.table_dims[0]
        coeffs_y = np.array(random.sample(range(-100, 100, 20), num_cubes))/100 * 0.8 * self.table_dims[1]

        cube_ids = []

        for i in range(num_cubes):
            if table_id is None:
                x, y = np.random.uniform(-self.boundary_length, self.boundary_length, 2)
                z = self.object_dims[2]
            else:
                pos, ori = self.pb_client.getBasePositionAndOrientation(table_id)
                _, _, theta = self.pb_client.getEulerFromQuaternion(ori)
                x, y = np.array(pos[:2]) \
                        + coeffs_x[i] * np.array([np.cos(theta), np.sin(theta)]) \
                        + coeffs_y[i] * np.array([np.sin(theta), -np.cos(theta)])
                z = 2*self.table_dims[2] + self.object_dims[2]
            color = generate_color(exclude_colors)
            # orientation = tf.quaternion_from_euler(np.pi*np.random.rand(), 0, 0)
            orientation = self.pb_client.getQuaternionFromEuler([0, 0, np.pi*np.random.rand()])
            cube_id = self.add_cube(self.cube_mass, cube_dims, [*color, 1], [x, y, z], orientation)
            cube_ids.append(cube_id)

        return cube_ids

    def get_table_positions(self, num_tables):
        point_1 = np.random.uniform(-2, 2, 2)
        pts = [point_1]

        for i in range(num_tables - 1):
            found = False
            while not found:
                sample = np.random.uniform(-2, 2, 2)
                found = True
                for pt in pts:
                    d = np.linalg.norm(sample-pt)
                    if d < 0.5:
                        found = False
            pts.append(sample)

        return pts

    def get_table_positions_grid(self, num_pos):

        idxs = random.sample(range(100), k=num_pos)
        def get_pos_from_idx(idx):
            x = -2 + 0.2 + 0.4*(idx//10)
            y = -2 + 0.2 + 0.4*(idx%10)
            return (x, y)

        pts = np.array([get_pos_from_idx(idx) for idx in idxs])
        return pts.reshape((num_pos, 2))

    def get_table_and_bot_positions_circle(self, num_pos):
        idxs = random.sample(range(0, 300, 50), k=num_pos)
        thetas = (np.random.rand() + np.array(idxs) / 300) * 2 * np.pi
        cos_thetas, sin_thetas = np.cos(thetas), np.sin(thetas)
        d_tables = np.random.uniform(0.5, 1.2, num_pos - 1)
        d_min = np.min(d_tables)
        d_bot = np.array([np.random.uniform(0, d_min)])
        ds = np.concatenate((d_bot, d_tables))
        center = np.random.uniform(0, 0.25, 2)
        x, y = center[0] + ds * cos_thetas, center[1] + ds * sin_thetas
        pts = np.concatenate((x, y)).reshape((2, num_pos)).T
        # print(ds, thetas*180/np.pi, pts)
        return pts

    def add_tables_and_objects(self, table_positions, num_tables):

        # colors
        # BLACK = [0.1, 0.1, 0.1, 1]
        # BLUE = [0.3, 0.3, 0.8, 1]
        # RED = [0.8, 0.2, 0.2, 1]

        # positions
        # positions = [[1, 0, self.table_dims[2]],
        #              [0, -1, self.table_dims[2]],
        #              [-1, 0, self.table_dims[2]],
        #              [0, 1, self.table_dims[2]]]

        positions = np.concatenate((table_positions, np.ones((num_tables, 1))*self.table_dims[2]), axis = 1)
        # random.shuffle(positions)
        thetas = np.random.uniform(-np.pi/2, np.pi/2, num_tables)

        # orientations = [tf.quaternion_from_euler(theta, 0, 0) for theta in thetas]
        orientations = [self.pb_client.getQuaternionFromEuler([0, 0, theta]) for theta in thetas]

        # colors = [BLACK, RED]

        table_ids = []
        for i in range(num_tables):
            table_id = self.add_table(self.table_mass,
                                     self.table_dims,
                                     self.table_colors[i],
                                     positions[i],
                                     orientations[i])

            self.table_ids.append(table_id)

        # initial_tables_with_objects = random.sample(self.table_ids, k=np.random.randint(1, num_tables+1))
        # initial_tables_with_objects = [self.table_ids[0]]
        initial_tables_with_objects = self.table_ids[:]

        for table_id in initial_tables_with_objects:
            cube_ids = self.generate_random_cubes(self.object_dims,
                                                  self.num_objects_per_table,
                                                  table_id,
                                                  exclude_colors = self.table_colors[:num_tables])
            self.object_ids.extend(cube_ids)

        self.initial_tables_with_objects = initial_tables_with_objects
        self.forward_simulation(100)

    def add_furniture(self):
        self.furniture = self.pb_client.loadURDF('storage_furniture/mobility.urdf',
                                                 [1.0, 1.0, 0.5],
                                                 # tf.quaternion_from_euler(np.pi, 0, np.pi),
                                                 self.pb_client.getQuaternionFromEuler([0, 0, np.random.uniform(0, np.pi)]),
                                                 globalScaling = 0.5)
        return self.furniture


    def get_object_positions(self):
        print(f'Positions are: ')
        for object in self.object_ids:
            pos, ori = self.pb_client.getBasePositionAndOrientation(object)
            print(f'    object {object}: {pos}')

    def get_objs_on_table(self, table):
        contact_points = self.pb_client.getContactPoints(bodyA = table)
        bodies_in_contact = set([contact[2] for contact in contact_points])
        obs_on_table_set = bodies_in_contact.intersection(set(self.object_ids))
        objs_on_table = list(obs_on_table_set)

        for obj in objs_on_table:
            contact_points = self.pb_client.getContactPoints(bodyA = obj)
            bodies_in_contact = set([contact[2] for contact in contact_points])
            objs_in_contact = bodies_in_contact.intersection(set(self.object_ids))
            for obj_new in objs_in_contact:
                if obj_new not in objs_on_table:
                    objs_on_table.append(obj_new)

        return objs_on_table
