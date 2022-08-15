# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from math import atan2, cos, sin, pi, radians, degrees, sqrt
import numpy as np

import os
import hydra
import numpy as np
import random
import cv2
from cliport import tasks
from cliport.environments.environment import Environment
from cliport.dataset import RavensDataset
import pdb
import glob
from copy import copy
import time
from matplotlib import pyplot as plt
from collections import deque
from d_star.d_star_lite import DStarLite
from d_star.grid import OccupancyGridMap
from cliport.dataset import PIXEL_SIZE
from locobot_policy_eval import get_pose_from_pixel
import copy
from numpy import sign
from cliport.utils.locobot_dyn_model import BicycleSystem
from cliport.utils.controller_utils import V_MIN, V_MAX, W_MIN, W_MAX
# from cliport.utils.pid_controller import run_simulation

# from cliport.utils.controller_utils import (
#     TrajectoryTracker,
#     position_control_init_fn,
#     _get_absolute_pose,
#     SimpleGoalState,
#     check_server_client_link,
# )


# if robot rotates by this much amount, then we considered that its moving
# at that speed
DT = 0.1
MIN_SPEED=-1
MAX_SPEED=1
MIN_W=-1
MAX_W=1
ROT_MOVE_THR = radians(1)
# if robot moves by this much amount(in meter), we asuume its moving at
# this speed
LIN_MOVE_THR = 0.01
# if the error in angle(radian) is less than this value, we consider task
# to be done
ROT_ERR_THR = radians(1)
# if the error in position(meter) is less than this value we consider task
# to be done
LIN_ERR_THR = 0.01
VEL_DELTA = 0.01  # increment in the velocity
# HZ = 20  # freq at which the controller should run
SLEEP_TIME = 0.1  # sleep time after every action

from cliport.utils.controller_utils import TrajectoryTracker, compute_controls_from_xy
import cliport.utils.cubic_spline_planner as cubic_spline_planner
from cliport.utils.pid_controller import get_control_waypoints, State
# def run_simulation(env):
#     traj_tracker.do_simulation(np.array([0,0,0]), plan)
#     traj_tracker.plot_plan_execution('/Users/meenalp/Desktop/simulation.jpeg')

def run_simulation(X, state):
    xyt, _ = compute_controls_from_xy(X, 0, DT)
    trajectory, controls, waypoints_reached, error = \
            get_control_waypoints(xyt, state, dt=DT, error_th=0.2, verbose=True)

    plt.plot(xyt[:,0], xyt[:,1], c='blue', marker='*')
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:,0], trajectory[:,1], c='red', marker='s')
    plt.xlim(-3, 1)
    plt.ylim(-2, 2)
    plt.axis('equal')

    plt.savefig('/Users/meenalp/Desktop/simulation_2.png')


@hydra.main(config_path='./cfg', config_name='data')
def main(cfg):

    env = Environment(
        cfg['assets_root'],
        disp=cfg['disp'],
        boundary=False,
        shared_memory=cfg['shared_memory'],
        hz=480,
        record_cfg=cfg['record']
    )
    task = tasks.names[cfg['task']]()
    task.mode = cfg['mode']
    agent = task.oracle(env, locobot=cfg['locobot'])

    # dataset = RavensDataset(os.path.join('data_dir', '{}-train'.format(task)), cfg,
    #                          store=False, cam_idx=[0], n_demos=0)

    env.set_task(task)
    obs = env.reset()
    info = env.info

    state = State(env, DT)

    radius = 1.0
    center = np.array([-radius, 0])
    thetas = np.linspace(0, 2*np.pi, 25)
    dx, dy = radius*np.cos(thetas), radius*np.sin(thetas)
    X = np.array([dx, dy]).T + center

    # l1_x = np.linspace(0, -1.0, 10)
    # l1_y = np.linspace(0, -1.0, 10)
    #
    # l2_x = np.linspace(-1.0, -2.0, 10)
    # l2_y = np.linspace(-1.0, 0.0, 10)
    #
    # l3_x = np.linspace(-2.0, -1.0, 10)
    # l3_y = np.linspace(0.0, 1.0, 10)
    #
    # x = np.concatenate((l1_x, l2_x, l3_x))
    # y = np.concatenate((l1_y, l2_y, l3_y))
    # X = np.array([x, y]).T

    # ax = np.array([0.0, -0.6, -1.25, -1.00, -1.75, -2.0, -2.5])*2.0
    # ay = np.array([0.0, -0.30, -0.50, 0.65, 0.30, 0.0, 0.0])*2.0
    # x, y, _, _, _ = cubic_spline_planner.calc_spline_course(ax, ay, ds=0.1)
    # X = np.array([x,y]).T

    plt.plot(X[:,0], X[:, 1])
    plt.savefig('/Users/meenalp/Desktop/trajectory.png')

    # x = np.linspace(0, -1.0, 10)
    # # y = np.linspace(0, -0.50, 50)
    # y = np.linspace(0, -1.0, 10)
    # # ax = [0.0, 6.0, 12.5, 10.0, 17.5, 20.0, 25.0]
    # # ay = [0.0, -3.0, -5.0, 6.5, 3.0, 0.0, 0.0]
    # # x, y, _, _, _ = cubic_spline_planner.calc_spline_course(ax, ay, ds=0.1)
    # X = np.array([x,y]).T

    run_simulation(X, state)
    return


    system = BicycleSystem(dt=0.1) #, min_v=V_MIN,
                            # max_v=V_MAX, min_w=W_MIN, max_w=W_MAX)
    traj_tracker = TrajectoryTracker(system, env)
    print("LQR steering control tracking start!!")
    # ax = [0.0, 6.0, 12.5, 10.0, 17.5, 20.0, 25.0]
    # ay = [0.0, -3.0, -5.0, 6.5, 3.0, 0.0, 0.0]
    # x, y, _, _, _ = cubic_spline_planner.calc_spline_course(ax, ay, ds=0.1)
    x = np.linspace(0, -0.50, 50)
    # y = np.linspace(0, -0.50, 50)
    y = np.zeros(50)
    X = np.array([x,y]).T

    goal = np.array([x[-1], y[-1]])
    xyt, us = compute_controls_from_xy(X, 0, 0.1)
    plan = traj_tracker.generate_plan(xyt, us)

    traj_tracker.do_simulation(np.array([0,0,0]), plan)
    traj_tracker.plot_plan_execution('/Users/meenalp/Desktop/simulation.jpeg')


    # traj_tracker.execute_plan(plan, goal)
    # traj_tracker.plot_plan_execution('/Users/meenalp/Desktop/simulation.jpeg')

if __name__ == '__main__':
     main()
