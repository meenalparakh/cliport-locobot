import numpy as np
from cliport.utils import utils
# from cliport.utils import path_planner_utils
import pybullet as p
## get_image wrapper: similar to dataset
## get_image: to obtain the point cloud wrt inital robot pose
from collections import namedtuple
# from cliport.d_star.grid import OccupancyGridMap
from cliport.environments.d_star.d_star_lite import DStarLite

def pi_2_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

Point = namedtuple('Point', ['x', 'y'])
Pixel = namedtuple('Pixel', ['row', 'col'])


def get_nearest_pt(pixel, occupancy_grid):
    r, c = pixel
    if occupancy_grid[r, c] < 0.5:
        return (r, c), 0
    else:
        x_dim, y_dim = occupancy_grid.shape
        cols, rows = np.meshgrid(range(y_dim), range(x_dim))
        distance = np.square(rows - r) + np.square(cols - c)
        distance = distance/np.max(distance)
        distance = np.where(occupancy_grid > 0.5, np.ones((x_dim, y_dim)), distance)

        nearest_pt = np.unravel_index(np.argmin(distance), distance.shape)

        # cv2.imwrite('/Users/meenalp/Desktop/distance_plot.jpeg', (distance*255).astype(np.uint8))
        return nearest_pt, np.min(distance)

class PlannerWrapper:
    def __init__(self, map):
        self.name = "D Star Lite"
        self.map = map

    def set_start_goal(self, start, goal):
        self.start = start
        self.goal = goal
        self.actual_start = self.get_nearest_point(self.start)
        self.actual_goal = self.get_nearest_point(self.goal)
        self.planner = DStarLite(self.map, self.actual_start, self.actual_goal)
        return self.actual_start, self.actual_goal

    def get_nearest_point(self, pixel):
        new_pixel, dist = get_nearest_pt(pixel, self.map.occupancy_grid_map)
        return new_pixel

    def set_grid(self, grid):
        self.planner.sensed_map = grid

    def plan_path(self, start, init_start=False):

        path, g, rhs = self.planner.move_and_replan(start)
        # add to path the move from start to actual_start
        if init_start:
            path.insert(0, self.start)
        return path
