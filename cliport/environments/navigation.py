import numpy as np
from cliport.utils import utils
from cliport.utils import path_planner_utils
import pybullet as p
## get_image wrapper: similar to dataset
## get_image: to obtain the point cloud wrt inital robot pose
from collections import namedtuple
# from cliport.environments.d_star.grid import OccupancyGridMap
# from cliport.environments.d_star.d_star_lite import DStarLite
from cliport.environments.planner import PlannerWrapper
from cliport.environments.controller import LocobotController
from cliport.environments.SLAM import SLAM
import cv2

def pi_2_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

Point = namedtuple('Point', ['x', 'y'])
Pixel = namedtuple('Pixel', ['row', 'col'])

## Visualization
    # ImageFolder
        # Each image contains
           # A red dot indicating locobot's initial position
           # A green dot inidicating the object
           # A trail of yellow dots showing the waypoints planned
           # A trail of blue dots showing the path of the locobot
           # The last updated map as an occupancg grid

class Navigation:
    def __init__(self, env):
        self.env = env
        self.slam = SLAM(env, env.locobot.get_base_pose())
        self.controller = LocobotController()
        self.controller.set_localization_class(self.slam)

    def navigate(self, goal_xy):

        obs = self.env.get_obs_wrapper()
        start_xy = self.slam.get_location()[0][:2]
        start_xy = Point(*start_xy)

        print('Start and goal xy:', start_xy, goal_xy)
        input('Press any key')

        self.slam.update_map([obs], start_xy, goal_xy)

        goal = self.slam.xy_to_pixel(goal_xy, self.slam.bounds)
        start = self.slam.xy_to_pixel(start_xy, self.slam.bounds)

        # bounds = self.slam.get_grid_size(start_xy, goal_xy)

        planner = PlannerWrapper(self.slam.slam_map)
        actual_start, actual_goal = planner.set_start_goal(start, goal)
        actual_goal_xy = self.slam.pixel_to_xy(actual_goal, self.slam.bounds)

        trajectory = []
        # self.slam.save_map('/Users/meenalp/Desktop/', bounds)

        # self.SLAM.world got updated above

        current = actual_start
        max_steps=50
        for step in range(max_steps):
            print('Inside loop')

            path_pixel = planner.plan_path(current, init_start=(step==0))
            print(path_pixel)
            path_xy = [self.slam.pixel_to_xy(pixel, self.slam.bounds) for pixel in path_pixel]

            controller_output = self.controller.move_on_waypoints(path_xy, backtrack=(step==0))
            obstacle_found, traj, _, _, _ = controller_output

            self.save_map(path_pixel, current, actual_goal,
                          f'/Users/meenalp/Desktop/navigation_testing/{step}.png')

            if not obstacle_found:
                break

            obs = self.env.get_obs_wrapper()
            new_costs = self.slam.update_map([obs], start_xy, goal_xy)

            self.planner.new_edges_and_old_costs = new_costs
            self.planner.set_grid(grid)

            current_state = self.controller.get_current_state()
            current = self.slam.xy_to_pixel(Point(*(current_state[:2])), self.bounds)


            # recompute if still want to go to the goal, or is there a point nearer to the object
            # for now assume that goal remains fixed

    def save_map(self, path_pixels, start, goal, filename):

        occ_grid_full = self.slam.slam_map.occupancy_grid_map

        occ_grid = occ_grid[..., None]
        occ_grid = occ_grid.repeat(3, axis=2)
        #
        # pcd = self.slam.current_pcd
        # bounds = np.array([[, np.max(pcd[:,0])],
        #                    [, np.max(pcd[:,1])],
        #                    [0               , np.inf          ]])
        # corner_pt_1 = (np.min(pcd[:,0]), np.min(pcd[:,1]))
        # # top_left = (current_grid_bounds[1, 0], current_grid_bounds[0, 0])
        # # bottom_right = (current_grid_bounds[1, 1], current_grid_bounds[0, 1])
        #
        for pixel in path_pixels:
            cv2.circle(occ_grid, (pixel[1], pixel[0]), 1, (255, 0, 0), 1)

        # cv2.rectangle(occ_grid, top_left, bottom_right, (255,0,0), 2)

        cv2.circle(image, (start[1], start[0]), 2, (0, 0, 255), 1)
        cv2.circle(image, (goal[1], goal[0]), 2, (0, 255, 0), 1)

        cv2.imwrite(filename, occ_grid)

        return occ_grid
