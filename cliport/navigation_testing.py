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
# import cliport.utils.cubic_spline_planner as cubic_spline_planner
from cliport.utils.lqr_controller import get_control_waypoints, State
from cliport.utils.lqr_controller import compute_controls_from_xy


# TODO
# (0) make the robot move out first
# (1) initial pose of the robot - remember it 
# (2) when obtain a new image transform it based on the 
#     original pose of the robot
# (3) keep upgrading the grid
#



def get_occupancy_grid(env, dataset, obs):
    image = dataset.get_image_wrapper(obs, pairings)
    substep = 0

    depth = image[substep][:,:,3]
    occupancy_grid = (depth > 0.05).astype(np.uint8)*255
    occupancy_grid_dilated = dilate_image(occupancy_grid)

    x_dim_, y_dim_ = depth.shape

    downscale_factor=4
    new_size = (x_dim_//downscale_factor, y_dim_//downscale_factor)
    occupancy_grid_dilated = cv2.resize(occupancy_grid_dilated,
                                    new_size, interpolation=cv2.INTER_AREA)
    occupancy_grid_dilated = (occupancy_grid_dilated > 10).astype(np.uint8)*255  
    return occupancy_grid_dilated
    
def get_path(env, dataset, obs, act):

    occupancy_grid_dilated = get_occupancy_grid(env, dataset, obs)
    x_dim, y_dim = occupancy_grid_dilated.shape

    pairings = [[0, 1]]
    p0s, _, _, _ = dataset.transform_pick_place(act, obs, pairings)
    pick_pt = p0s[0]
    pick_pose = get_pose_from_pixel(env, pick_pt, obs[-1]['bot_pose'])
    new_pick_pt = map_pixel(pick_pt, (x_dim_, y_dim_), (x_dim, y_dim))

    goal, _ = get_nearest_pt(new_pick_pt, occupancy_grid_dilated)
    start = (x_dim_//2, int(0.2/PIXEL_SIZE))
    start_original = map_pixel(start, (x_dim_, y_dim_), (x_dim, y_dim))
    start = get_nearest_pt(start, occupancy_grid_dilated)

    start_ = map_pixel(start_, (x_dim, y_dim), (x_dim_, y_dim_))
    start_xy = get_pose_from_pixel(env, start_, obs[-1]['bot_pose'])[0][:2]


    # marked = distance.copy()[..., None]
    # marked = marked.repeat(3, axis=2)
    # marked = cv2.circle(marked, (goal[1], goal[0]), radius=2, color=(0,255,0),
    #                     thickness=-1)
    # marked = cv2.circle(marked, (start[1], start[0]), radius=2, color=(0,0,255),
    #                     thickness=-1)
    # cv2.imwrite(os.path.join('/Users/meenalp/Desktop', 'start_end.jpeg'), marked)

    map = OccupancyGridMap(x_dim, y_dim)
    map.occupancy_grid_map = occupancy_grid_dilated

    solver = DStarLite(map, start, goal, use_map=True)

    path, g, rhs = solver.move_and_replan(robot_position=start)

    path_pos, pixels = get_path_from_pixels(env, obs, path,
                                        (x_dim, y_dim), (x_dim_, y_dim_))

    image = cv2.arrowedLine(color, (pixels[-1][1], pixels[-1][0]),
                                    (pick_pt[1], pick_pt[0]),
                                (255, 255, 0), 1, tipLength = 0.2)
    cv2.imwrite(os.path.join('/Users/meenalp/Desktop', 'marked.jpeg'), color)
    return path_pos, pick_pose


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
        info = env.info

        obs = obs[-env.num_turns:]
        act = agent.act(obs, info)

        path_pos, pick_pose = get_plan(env, dataset, obs, act)
        # path_pos = get_plan(env, image, pick_pt)

        state = State(env, DT)
        X = np.array(path_pos)
        plt.plot(X[:,0], X[:, 1]); plt.axis('equal')
        plt.savefig('/Users/meenalp/Desktop/trajectory.png')

        pick_pos = pick_pose[0][:2]
        dx, dy = np.array(pick_pos) - np.array(path_pos[-1])
        theta = np.arctan2(dy, dx)
        run_simulation(X, state, goal_yaw=theta, )

        time.sleep(2)

if __name__ == '__main__':
    main()
