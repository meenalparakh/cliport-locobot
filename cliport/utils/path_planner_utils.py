import os
import numpy as np
import random
import cv2
from copy import copy
import time
from matplotlib import pyplot as plt
# from d_star.d_star_lite import DStarLite
# from d_star.grid import OccupancyGridMap
# from cliport.dataset import PIXEL_SIZE
from locobot_policy_eval import get_pose_from_pixel
# import cliport.utils.cubic_spline_planner as cubic_spline_planner
# from cliport.utils.lqr_controller import get_control_waypoints, State
# from cliport.utils.lqr_controller import compute_controls_from_xy

def within_bounds(pt, height, width):
    row, col = pt
    r = min(height, max(0, row))
    c = min(width, max(0, col))
    return (int(r), int(c))

# def dilate_image(img, margin=0.50):
#     pixel_margin = int(margin/PIXEL_SIZE)
#     kernel = np.ones((pixel_margin, pixel_margin), np.uint8)
#     dilated_image = cv2.dilate(img, kernel, iterations=1)
#     return dilated_image

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
# 
# def get_nearest_pt(pixel, occupancy_grid):
#     r, c = pixel
#     if occupancy_grid[r, c] < 0.5:
#         return (r, c), 0
#     else:
#         x_dim, y_dim = occupancy_grid.shape
#         cols, rows = np.meshgrid(range(x_dim), range(y_dim))
#         distance = np.square(rows - r) + np.square(cols - c)
#         distance = distance/np.max(distance)
#         distance = np.where(occupancy_grid > 10, np.ones((x_dim, y_dim)), distance)
#
#         nearest_pt = np.unravel_index(np.argmin(distance), distance.shape)
#
#         cv2.imwrite('/Users/meenalp/Desktop/distance_plot.jpeg', (distance*255).astype(np.uint8))
#         return nearest_pt, np.min(distance)
