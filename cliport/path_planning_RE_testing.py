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
# from d_star.d_star_lite import DStarLite
# from d_star.grid import OccupancyGridMap
from cliport.dataset import PIXEL_SIZE
from locobot_policy_eval import get_pose_from_pixel
# # import cliport.utils.cubic_spline_planner as cubic_spline_planner
# from cliport.utils.lqr_controller import get_control_waypoints, State
# from cliport.utils.lqr_controller import compute_controls_from_xy

from cliport.environments.navigation import Navigation, Point

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

    dataset = RavensDataset(os.path.join('data_dir', '{}-train'.format(task)), cfg,
                             store=False, cam_idx=[0], n_demos=0)

    env.set_task(task)

    for i in range(cfg['n']):
        obs = env.reset()
        # state = State(env, DT)

        info = env.info

        obs = obs[-env.num_turns:]
        act = agent.act(obs, info)

        navigation_module = Navigation(env)

        goal_xy = Point(*(act['pose0'][0][:2]))

        navigation_module.navigate(goal_xy)

        time.sleep(2)

if __name__ == '__main__':
    main()
