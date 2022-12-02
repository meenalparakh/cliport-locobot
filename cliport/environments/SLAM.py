import numpy as np
from cliport.utils import utils
from cliport.utils import path_planner_utils
import pybullet as p
## get_image wrapper: similar to dataset
## get_image: to obtain the point cloud wrt inital robot pose
from collections import namedtuple
PIXEL_SIZE = 0.06250

from cliport.environments.d_star.utils import get_movements_4n, get_movements_8n, heuristic, Vertices, Vertex
from typing import Dict, List

OBSTACLE = 1
UNOCCUPIED = 0

def pi_2_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

Point = namedtuple('Point', ['x', 'y'])
Pixel = namedtuple('Pixel', ['row', 'col'])

def get_pcd(obs, init_X):
    pcds = []
    for step in range(len(obs)):
        image_obs = obs[step]['image']
        configs = obs[step]['configs']
        for cam_idx in range(len(image_obs['color'])):
            print(f'Cam idx: {cam_idx}')
            color = image_obs['color'][cam_idx]
            depth = image_obs['depth'][cam_idx]
            config = configs[cam_idx]

            X_WC = utils.get_transformation_matrix((config['position'],
                                                    config['rotation']))
            X_LC = np.linalg.inv(init_X) @ X_WC
            pos, ori = utils.get_pose_from_transformation(X_LC)
            config['position'], config['rotation'] = pos, ori

            intrinsics = np.array(config['intrinsics']).reshape(3, 3)
            xyz = utils.get_pointcloud(depth, intrinsics)
            position = np.array(config['position']).reshape(3, 1)
            rotation = p.getMatrixFromQuaternion(config['rotation'])
            rotation = np.array(rotation).reshape(3, 3)
            transform = np.eye(4)
            transform[:3, :] = np.hstack((rotation, position))
            xyz = utils.transform_pointcloud(xyz, transform)
            H, W, C = xyz.shape
            xyz = xyz.reshape((H*W, 3))
            pcds.append(xyz)
    return pcds

def get_orthographic_view(pcd_, bounds):
    # print("Bounds shape here", bounds.shape)

    width = int(np.round((bounds[0, 1] - bounds[0, 0]) / PIXEL_SIZE))
    height = int(np.round((bounds[1, 1] - bounds[1, 0]) / PIXEL_SIZE))

    # print(f'Size of grid: {height, width}')
    # print(f'Inside get_heightmap: {bounds}, {width, height}')
    heightmap = np.zeros((height, width), dtype=np.float32)

    pcd = np.vstack(pcd_)

    ix = (pcd[Ellipsis, 0] >= bounds[0, 0]) & (pcd[Ellipsis, 0] < bounds[0, 1])
    iy = (pcd[Ellipsis, 1] >= bounds[1, 0]) & (pcd[Ellipsis, 1] < bounds[1, 1])
    iz = (pcd[Ellipsis, 2] >= bounds[2, 0]) & (pcd[Ellipsis, 2] < bounds[2, 1])

    valid = ix & iy & iz
    pcd = pcd[valid]

    # Sort 3D points by z-value, which works with array assignment to simulate
    # z-buffering for rendering the heightmap image.
    iz = np.argsort(pcd[:, -1])
    pcd = pcd[iz]
    px = np.int32(np.floor((pcd[:, 0] - bounds[0, 0]) / PIXEL_SIZE))
    py = np.int32(np.floor((pcd[:, 1] - bounds[1, 0]) / PIXEL_SIZE))
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    heightmap[py, px] = pcd[:, 2] - bounds[2, 0]
    return heightmap


class OccupancyGridMap:
    def __init__(self, map, exploration_setting='8N'):
        """
        set initial values for the map occupancy grid
        |----------> y, column
        |           (x=0,y=2)
        |
        V (x=2, y=0)
        x, row
        :param x_dim: dimension in the x direction
        :param y_dim: dimension in the y direction
        """
        map = map.astype(np.uint8)
        self.x_dim, self.y_dim = map.shape
        # the map extents in units [m]
        self.map_extents = (self.x_dim, self.y_dim)
        # the obstacle map
        self.occupancy_grid_map = map

        # obstacles
        self.visited = {}
        self.exploration_setting = exploration_setting

    def get_map(self):
        """
        :return: return the current occupancy grid map
        """
        return self.occupancy_grid_map

    def set_map(self, new_ogrid):
        """
        :param new_ogrid:
        :return: None
        """
        self.occupancy_grid_map = new_ogrid

    def is_unoccupied(self, pos: (int, int)) -> bool:
        """
        :param pos: cell position we wish to check
        :return: True if cell is occupied with obstacle, False else
        """
        (x, y) = (round(pos[0]), round(pos[1]))  # make sure pos is int
        (row, col) = (x, y)

        # if not self.in_bounds(cell=(x, y)):
        #    raise IndexError("Map index out of bounds")

        return self.occupancy_grid_map[row][col] == UNOCCUPIED

    def in_bounds(self, cell: (int, int)) -> bool:
        """
        Checks if the provided coordinates are within
        the bounds of the grid map
        :param cell: cell position (x,y)
        :return: True if within bounds, False else
        """
        (x, y) = cell
        return 0 <= x < self.x_dim and 0 <= y < self.y_dim

    def filter(self, neighbors: List, avoid_obstacles: bool):
        """
        :param neighbors: list of potential neighbors before filtering
        :param avoid_obstacles: if True, filter out obstacle cells in the list
        :return:
        """
        if avoid_obstacles:
            return [node for node in neighbors if self.in_bounds(node) and self.is_unoccupied(node)]
        return [node for node in neighbors if self.in_bounds(node)]

    def succ(self, vertex: (int, int), avoid_obstacles: bool = False) -> list:
        """
        :param avoid_obstacles:
        :param vertex: vertex you want to find direct successors from
        :return:
        """
        (x, y) = vertex
        movements = get_movements_8n(x=x, y=y)
        # not needed. Just makes aesthetics to the path
        if (x + y) % 2 == 0: movements.reverse()
        filtered_movements = self.filter(neighbors=movements, avoid_obstacles=avoid_obstacles)
        return list(filtered_movements)

def get_pixel_within_bounds(pixel, height, width):
    row, col = pixel
    row = min(height-1, max(row, 0))
    col = min(width-1, max(col, 0))
    return (row, col)


class SLAM:
    def __init__(self, env, init_pose=((0,0,0), (0,0,0,0))):
        self.ground_truth_map = map
        self.slam_map = None

        self.env = env
        self.init_X_WL = utils.get_transformation_matrix(init_pose)
        # self.world =
        self.world = [] # empty point cloud initially

    def reset_grid(self):
        self.slam_map = None

    def update_map(self, obs, start, goal):
        # combine inital world with the new data

        pcd = get_pcd(obs, self.init_X_WL)
        self.world.extend(pcd)

        self.current_pcd = pcd

        self.bounds = self.get_grid_size(start, goal)
        occ_grid = self.get_gridmap(pcd, self.bounds)

        self.current_grid = occ_grid

        if self.slam_map is None:
            self.slam_map = OccupancyGridMap(np.zeros_like(occ_grid))

        vertices = self.update_changed_edge_costs(self.current_grid)
        # if self.slam_map is None:
        #     self.slam_map = OccupancyGridMap(occ_grid)
        # else:
        grid = self.slam_map.occupancy_grid_map
        self.slam_map.occupancy_grid_map = np.logical_or(grid, occ_grid)
    #     return {node: UNOCCUPIED if self.is_unoccupied(pos=node) else OBSTACLE for node in nodes

        return vertices, self.slam_map

    def update_changed_edge_costs(self, occ_grid) -> Vertices:
        vertices = Vertices()
        height, width = occ_grid.shape

        for row in range(height):
            for col in range(width):
                node = (row, col)
                if occ_grid[row, col] == 1:
                    if self.slam_map.is_unoccupied(node):
                        v = Vertex(pos=node)
                        succ = self.slam_map.succ(node)
                        for u in succ:
                            v.add_edge_with_cost(succ=u, cost=self.c(u, v.pos))
                        vertices.add_vertex(v)
                            # self.slam_map.set_obstacle(node)

                else:
                    if not self.slam_map.is_unoccupied(node):
                        v = Vertex(pos=node)
                        succ = self.slam_map.succ(node)
                        for u in succ:
                            v.add_edge_with_cost(succ=u, cost=self.c(u, v.pos))
                        vertices.add_vertex(v)
                            # self.slam_map.remove_obstacle(node)
        return vertices

    def get_grid_size(self, start, goal):

        # print(type(start), type(goal))
        min_x = min(start.x, goal.x) - 2.0
        max_x = max(start.x, goal.x) + 2.0

        min_y = min(start.y, goal.y) - 2.0
        max_y = max(start.y, goal.y) + 2.0

        bounds = np.array([[min_x, max_x], [min_y, max_y], [0, np.inf]])
        self.bounds = bounds
        # print(bounds.shape)
        return bounds

    def get_gridmap(self, pcd, bounds, z_threshold=0.05):
        heightmap = get_orthographic_view(pcd, bounds)
        occ_grid = (heightmap > z_threshold) #.astype(np.uint8)
        return occ_grid

    def pixel_to_xy(self, pixel, bounds):
        x, y, z = utils.pix_to_xyz(pixel, None, bounds, PIXEL_SIZE, skip_height=True)
        return Point(x, y)

    def xy_to_pixel(self, xy, bounds):
        row, col = utils.xyz_to_pix(xy, bounds, PIXEL_SIZE)
        return Pixel(row, col)

    def get_location(self):
        base_pose = self.env.locobot.get_base_pose()
        X_WL = utils.get_transformation_matrix(base_pose)
        X_LLinit = np.linalg.inv(self.init_X_WL) @ X_WL
        return utils.get_pose_from_transformation(X_LLinit)

    def is_near(self, goal_location, threshold=0.20):
        curr_location = self.get_location()[0][:2]
        d = np.linalg.norm(np.array(goal_location) - np.array(curr_location))
        return (d < threshold)

    def obstacle_infront(self):
        return False


    def c(self, u: (int, int), v: (int, int)) -> float:
        """
        calcuclate the cost between nodes
        :param u: from vertex
        :param v: to vertex
        :return: euclidean distance to traverse. inf if obstacle in path
        """
        if not self.slam_map.is_unoccupied(u) or not self.slam_map.is_unoccupied(v):
            return float('inf')
        else:
            return heuristic(u, v)

    # def rescan(self, global_position: (int, int)):
    #     # rescan local area
    #     local_observation = self.ground_truth_map.local_observation(global_position=global_position,
    #                                                                 view_range=self.view_range)
    #
    #     vertices = self.update_changed_edge_costs(local_grid=local_observation)
    #     return vertices, self.slam_map
