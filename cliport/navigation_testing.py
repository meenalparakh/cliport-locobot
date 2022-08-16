"""Data collection script."""

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
from cliport.utils.controller_utils import compute_controls_from_xy
# import cliport.utils.cubic_spline_planner as cubic_spline_planner
from cliport.utils.lqr_controller import get_control_waypoints, State

DT = 0.2

def run_simulation(X, state, goal_yaw=None):
    xyt, _ = compute_controls_from_xy(X, 0, DT)
    if goal_yaw is not None:
        goal_x, goal_y = xyt[-1,:2]
        goal = np.array([[goal_x, goal_y, goal_yaw]])
        xyt = np.concatenate((xyt, goal), axis=0)

    trajectory, controls, waypoints_reached, error = \
            get_control_waypoints(xyt, state, dt=DT, error_th=0.2, verbose=True)
            # get_control_waypoints(xyt, state, dt=DT, error_th=0.05, verbose=True)

    plt.plot(xyt[:,0], xyt[:,1], c='blue', marker='*')
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:,0], trajectory[:,1], c='red', marker='s')
    plt.xlim(-3, 1)
    plt.ylim(-2, 2)
    plt.axis('equal')

    plt.savefig('/Users/meenalp/Desktop/simulation_2.png')

def within_bounds(pt, height, width):
    row, col = pt
    r = min(height, max(0, row))
    c = min(width, max(0, col))
    return (int(r), int(c))

def dilate_image(img, margin=0.50):
    pixel_margin = int(margin/PIXEL_SIZE)
    kernel = np.ones((pixel_margin, pixel_margin), np.uint8)
    dilated_image = cv2.dilate(img, kernel, iterations=1)
    return dilated_image

def map_pixel(pt, from_dim, to_dim):
    scale_x = to_dim[0]/from_dim[0]
    scale_y = to_dim[1]/from_dim[1]
    scale = np.array([scale_x, scale_y])
    to_center = np.array([(to_dim[0]-1)//2, (to_dim[1]-1)//2])
    from_center = np.array([(from_dim[0]-1)//2, (from_dim[1]-1)//2])
    new_pt = (np.array(pt) - from_center)*scale + to_center
    # print(pt, to_center, from_center, scale)
    return within_bounds(new_pt, *to_dim)

def get_path_from_pixels(env, obs, pixel_pts, currrent_dim, original_dim):
    pixels = []
    path_pos = []
    for idx in range(0, len(pixel_pts), 5):
        pt = pixel_pts[idx]
        new_pt = map_pixel(pt, currrent_dim, original_dim)
        pixels.append(new_pt)
        path_pt = get_pose_from_pixel(env, new_pt, obs[-1]['bot_pose'])[0][:2]
        path_pos.append(path_pt)
        color = cv2.circle(img=color, center = (new_pt[1], new_pt[0]),
                           radius=2, color=(0,255,0), thickness=-1)
    return path_pos, pixels

def get_nearest_pt(pixel, occupancy_grid):
    r, c = pixel
    if occupancy_grid[r, c] < 0.5:
        return (r, c), 0
    else:
        x_dim, y_dim = occupancy_grid.shape
        cols, rows = np.meshgrid(range(x_dim), range(y_dim))
        distance = np.square(rows - r) + np.square(cols - c)
        distance = distance/np.max(distance)
        distance = np.where(occ_grid > 10, np.ones((x_dim, y_dim)), distance)

        nearest_pt = np.unravel_index(np.argmin(distance), distance.shape)

        cv2.imwrite('/Users/meenalp/Desktop/distance_plot.jpeg', (distance*255).astype(np.uint8))
        return nearest_pt, np.min(distance)

def get_path(env, dataset, obs, act):
    pairings = [[0, 1]]
    image = dataset.get_image_wrapper(obs, pairings)
    p0s, _, _, _ = dataset.transform_pick_place(act, obs, pairings)
    pick_pt = p0s[0]
    pick_pose = get_pose_from_pixel(env, pick_pt, obs[-1]['bot_pose'])

    substep = 0
    color = image[substep][:,:,:3]
    color_fname = f'navi_color.png'
    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    color = cv2.circle(color, (pick_pt[1], pick_pt[0]), radius=2, color=(0,255,0),
                        thickness=-1)
    cv2.imwrite(os.path.join('/Users/meenalp/Desktop', color_fname), color)

    depth = image[substep][:,:,3]
    # depth_fname = f'navi_depth.png'
    # sdepth = depth/np.max(depth)
    # plt.imsave(os.path.join('/Users/meenalp/Desktop', depth_fname), sdepth)

    occupancy_grid = (depth > 0.05).astype(np.uint8)*255
    occupancy_grid_dilated = dilate_image(occupancy_grid)

    x_dim_, y_dim_ = depth.shape

    downscale_factor=4
    new_size = (x_dim_//downscale_factor, y_dim_//downscale_factor)
    occupancy_grid_dilated = cv2.resize(occupancy_grid_dilated,
                                    new_size, interpolation=cv2.INTER_AREA)
    occupancy_grid_dilated = (occupancy_grid_dilated > 10).astype(np.uint8)*255
    x_dim, y_dim = new_size

    cv2.imwrite(os.path.join('/Users/meenalp/Desktop',
                            'occupancy_grid_resized_dilated.jpeg'),
                            occupancy_grid_dilated)

    new_pick_pt = map_pixel(pick_pt, (x_dim_, y_dim_), (x_dim, y_dim))

    color_resized = cv2.resize(color, new_size, interpolation=cv2.INTER_AREA)
    color_resized = cv2.circle(color_resized, (new_pick_pt[1], new_pick_pt[0]),
                                radius=2, color=(0,255,0), thickness=-1)
    cv2.imwrite('/Users/meenalp/Desktop/resized_color.jpeg', color_resized)
    # return

    # nearest_pt, distance = get_nearest_pt(new_pick_pt, occupancy_grid_dilated)

    nearest_pt, _ = get_nearest_pt(new_pick_pt, occupancy_grid_dilated)
    goal = nearest_pt
    start_ = (x_dim_//2, int(0.2/PIXEL_SIZE))
    start = map_pixel(start_, (x_dim_, y_dim_), (x_dim, y_dim))
    start = get_nearest_pt(start, occupancy_grid_dilated)
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
        run_simulation(X, state, goal_yaw=theta)

        time.sleep(2)

if __name__ == '__main__':
    main()
