import numpy as np
from cliport.utils import utils
from cliport.utils import path_planner_utils
import pybullet as p
## get_image wrapper: similar to dataset
## get_image: to obtain the point cloud wrt inital robot pose
from collections import namedtuple
# from cliport.d_star.grid import OccupancyGridMap
# from cliport.d_star.d_star_lite import DStarLite

def pi_2_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class LocobotController:
    ## TODO: add stopping criteria when it finds some obstacle ahead

    def __init__(self, constants_cfg=None):
        self.x_dim = 3 # x, y, theta
        self.u_dim = 2 # v, w
        self.Q = np.array([[1.0, 0, 0],[0, 1.0, 0], [0, 0, 5.0]])
        self.R = np.array([[0.1, 0.0], [0.0, 0.05]])
        self.error_th = 0.1
        self.base_radius = 0.115
        self.wheel_radius = 0.025
        self.wheel_max_speed = 40
        self.dt = 0.1
        self.wheel_min_threshold = 3.0

        self.state = [0, 0, 0] # x, y, theta

    def set_localization_class(self, localization_class):
        self.slam = localization_class
        self.env = self.slam.env

    def get_current_state(self):
        # it will be wrt to inital pose in slam class
        pos, ori = self.slam.get_location()
        x, y = pos[:2]
        theta = self.env.pb_client.getEulerFromQuaternion(ori)[-1]
        return np.array([x, y, theta])

    def get_AB(self, yaw):
        A = np.eye(3)
        B = np.array([[np.cos(yaw)*self.dt, 0      ],
                      [np.sin(yaw)*self.dt, 0      ],
                      [0,                   self.dt]])
        return A, B

    def command_locobot(self, vels):
        self.env.locobot.set_base_vel(vels)

        num_steps = int(self.dt*self.env.hz)
        for i in range(num_steps):
            self.env.step_simulation()

    def get_wheel_vel_from_controls(self, v, w):
        left_wheel = (v - w*self.base_radius)/self.wheel_radius
        right_wheel = (v + w*self.base_radius)/self.wheel_radius
        vels = [left_wheel, right_wheel]
        # print('Wheel velocity:', vels)
        vels = np.clip(vels, a_min=-self.wheel_max_speed,
                        a_max=self.wheel_max_speed)
        return vels

    def lqr(self, x_error, A, B):
        # x_error = actual_state_x - desired_state_xf
        dt = self.dt
        Q = self.Q
        R = self.R

        N = 100
        P = [None] * (N + 1)
        Qf = Q
        P[N] = Qf
        for i in range(N, 0, -1):
            P[i-1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
                R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)

        K = -np.linalg.pinv(R + B.T @ P[0] @ B) @ B.T @ P[0] @ A
        u_star = K @ x_error #+ np.array([2.0, 0])
        return u_star

    def compute_controls_from_xy(self, xy, theta0, backtrack=False,
                                    flip_theta=False):
        x, y = xy.T
        theta = np.arctan2(y[1:] - y[:-1], x[1:] - x[:-1])
        if flip_theta:
            theta = theta + np.pi
        theta = np.concatenate([[theta0], theta], axis=0)
        if backtrack:
            theta[1] = -theta[1]

        # Unwrap theta as necessary.
        old_theta = theta[0]
        for i in range(theta.shape[0] - 1):
            theta[i + 1] = pi_2_pi(theta[i + 1] - old_theta) + old_theta
            old_theta = theta[i + 1]

        xyt = np.array([x, y, theta]).T
        return xyt

    def move_on_waypoints(self, xy, backtrack=False, verbose=False):

        theta0 = self.get_current_state()[2]
        waypoints = self.compute_controls_from_xy(np.array(xy), theta0, backtrack)
        waypoints = [np.array(i) for i in waypoints]
        trajectory = []
        controls = []
        error = []
        waypoints_reached = []
        actual_state_x = self.get_current_state() #np.array([0,0,0])

        Q = self.Q
        R = self.R

        commanded_wheel_vel = np.inf
        obstacle_found = False

        for idx in range(len(waypoints)):

            desired_state_xf = waypoints[idx]
            state_error_magnitude = 1e10
            waypoints_reached.append(actual_state_x)

            while (state_error_magnitude >= self.error_th):

                if verbose:
                    print(f'Current State = {actual_state_x}')
                    print(f'Desired State = {desired_state_xf}')

                if self.slam.obstacle_infront():
                    print('Obstacle in front, exiting controller, replanning!')
                    obstacle_found = True

                state_error = actual_state_x - desired_state_xf
                state_error[2] = pi_2_pi(state_error[2])

                trajectory.append(actual_state_x)
                state_error_magnitude = np.linalg.norm(state_error)
                error.append(state_error_magnitude)
                # if verbose:
                print(f'State Error Magnitude = {state_error_magnitude}')

                A, B = self.get_AB(actual_state_x[2])
                optimal_control_input = self.lqr(state_error, A, B)
                controls.append(optimal_control_input)
                # if verbose:
                print(f'Control Input = {optimal_control_input}')
                
                v, w = optimal_control_input
                wheel_vels = self.get_wheel_vel_from_controls(v, w)
                wheel_vel_mag = np.linalg.norm(wheel_vels)

                if wheel_vel_mag < self.wheel_min_threshold:
                    break

                self.command_locobot(wheel_vels)
                actual_state_x = self.get_current_state()
                # actual_state_x = state_space_model(A, actual_state_x, B,
                #                                 optimal_control_input)
                if verbose:
                    if state_error_magnitude < error_th:
                        print("\nGoal Has Been Reached Successfully!")

            if obstacle_found:
                break

        return obstacle_found, trajectory, controls, waypoints_reached, error
