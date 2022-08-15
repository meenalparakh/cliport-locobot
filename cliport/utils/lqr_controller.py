import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
from cliport.utils.controller_utils import compute_controls_from_xy

np.set_printoptions(precision=3,suppress=True)
max_linear_velocity = 3.0
max_angular_velocity = 1.5708
WHEEL_MAX_SPEED = 40

BASE_RADIUS = 0.115
WHEEL_RADIUS = 0.025

def getB(yaw, dt):
    B = np.array([[np.cos(yaw)*dt, 0],[np.sin(yaw)*dt, 0],[0, dt]])
    return B

# def state_space_model(A, state_t_minus_1, B, control_input_t_minus_1):
#     control_input_t_minus_1 = np.clip(control_input_t_minus_1,
#                     a_min=[-max_linear_velocity, -max_angular_velocity],
#                     a_max=[max_linear_velocity, max_angular_velocity])
#
#     state_estimate_t = (A @ state_t_minus_1) + (B @ control_input_t_minus_1)
#     return state_estimate_t

def pi_2_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def lqr(x_error, Q, R, A, B, dt):
    # x_error = actual_state_x - desired_state_xf
    N = 100
    P = [None] * (N + 1)
    Qf = Q
    P[N] = Qf
    for i in range(N, 0, -1):
        P[i-1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
            R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)
    # K = [None] * N
    # u = [None] * N
    # for i in range(N):
    #     K[i] = -np.linalg.pinv(R + B.T @ P[i+1] @ B) @ B.T @ P[i+1] @ A
    #     u[i] = K[i] @ x_error

    K = -np.linalg.pinv(R + B.T @ P[0] @ B) @ B.T @ P[0] @ A
    u_star = K @ x_error #+ np.array([2.0, 0])
    return u_star


def get_wheel_vel_from_controls(v, w):
    left_wheel = (v - w*BASE_RADIUS)/WHEEL_RADIUS
    right_wheel = (v + w*BASE_RADIUS)/WHEEL_RADIUS
    vels = [left_wheel, right_wheel]
    print('Wheel velocity:', vels)
    vels = np.clip(vels, a_min=-WHEEL_MAX_SPEED, a_max=WHEEL_MAX_SPEED)
    return vels

class State:
    def __init__(self, env, dt):
        self.env = env
        self.dt = dt

    def get_current_state(self):
        pos, ori = self.env.locobot.get_base_pose()
        x, y = pos[:2]
        theta = self.env.pb_client.getEulerFromQuaternion(ori)[-1]
        return np.array([x, y, theta])

    def command(self, vels):
        self.env.locobot.set_base_vel(vels)

        num_steps = int(self.dt*self.env.hz)
        for i in range(num_steps):
            self.env.step_simulation()

def get_control_waypoints(waypoints, state, dt=0.1, error_th=1, verbose=False):
    waypoints = [np.array(i) for i in waypoints]
    trajectory = []
    controls = []
    error = []
    waypoints_reached = []
    actual_state_x = state.get_current_state() #np.array([0,0,0])

    A = np.eye(3)
    # Q = np.array([[1.0, 0, 0],[0, 1.0, 0], [0, 0, 0.50]])
    Q = np.array([[1.0, 0, 0],[0, 1.0, 0], [0, 0, 5.0]])
    R = np.array([[1.0, 0.0], [0.0, 0.1]])
    commanded_wheel_vel = np.inf

    for idx in range(len(waypoints)):
        desired_state_xf = waypoints[idx]
        state_error_magnitude = 1e10
        waypoints_reached.append(actual_state_x)
        while (state_error_magnitude >= error_th):
            if verbose:
                print(f'Current State = {actual_state_x}')
                print(f'Desired State = {desired_state_xf}')
            state_error = actual_state_x - desired_state_xf
            state_error[2] = pi_2_pi(state_error[2])

            trajectory.append(actual_state_x)
            state_error_magnitude = np.linalg.norm(state_error)
            error.append(state_error_magnitude)
            if verbose:
                print(f'State Error Magnitude = {state_error_magnitude}')
            B = getB(actual_state_x[2], dt)
            optimal_control_input = lqr(state_error,
                                        Q, R, A, B, dt)
            controls.append(optimal_control_input)
            if verbose:
                print(f'Control Input = {optimal_control_input}')
            v, w = optimal_control_input
            wheel_vels = get_wheel_vel_from_controls(v, w)
            wheel_vel_mag = np.linalg.norm(wheel_vels)

            if wheel_vel_mag < 3.0:
                break

            state.command(wheel_vels)
            actual_state_x = state.get_current_state()
            # actual_state_x = state_space_model(A, actual_state_x, B,
            #                                 optimal_control_input)
            if verbose:
                if state_error_magnitude < error_th:
                    print("\nGoal Has Been Reached Successfully!")

    return trajectory, controls, waypoints_reached, error
