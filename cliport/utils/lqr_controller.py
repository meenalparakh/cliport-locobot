import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
import time

np.set_printoptions(precision=3,suppress=True)
max_linear_velocity = 3.0
max_angular_velocity = 1.5708
WHEEL_MAX_SPEED = 40

BASE_RADIUS = 0.115
WHEEL_RADIUS = 0.025
DT = 0.2

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

    

def getB(yaw, dt):
    B = np.array([[np.cos(yaw)*dt, 0],[np.sin(yaw)*dt, 0],[0, dt]])
    return B

def pi_2_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi




def get_wheel_vel_from_controls(v, w):
    left_wheel = (v - w*BASE_RADIUS)/WHEEL_RADIUS
    right_wheel = (v + w*BASE_RADIUS)/WHEEL_RADIUS
    vels = [left_wheel, right_wheel]
    print('Wheel velocity:', vels)
    vels = np.clip(vels, a_min=-WHEEL_MAX_SPEED, a_max=WHEEL_MAX_SPEED)
    return vels

def get_control_waypoints(waypoints, state, error_th=1, verbose=False):
    waypoints = [np.array(i) for i in waypoints]
    trajectory = []
    controls = []
    error = []
    waypoints_reached = []
    actual_state_x = state.get_current_state() #np.array([0,0,0])

    A = np.eye(3)
    # Q = np.array([[1.0, 0, 0],[0, 1.0, 0], [0, 0, 0.50]])
    Q = np.array([[1.0, 0, 0],[0, 1.0, 0], [0, 0, 5.0]])
    # R = np.array([[1.0, 0.0], [0.0, 0.1]])
    R = np.array([[0.1, 0.0], [0.0, 0.05]])
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
            B = getB(actual_state_x[2], DT)
            optimal_control_input = lqr(state_error,
                                        Q, R, A, B, DT)
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

def run_simulation(xyt, state, goal_yaw=None, fig_filename=None):
    if goal_yaw is not None:
        goal_x, goal_y = xyt[-1,:2]
        goal = np.array([[goal_x, goal_y, goal_yaw]])
        xyt = np.concatenate((xyt, goal), axis=0)

    trajectory, controls, waypoints_reached, error = \
            get_control_waypoints(xyt, state, error_th=0.2, verbose=True)
            # get_control_waypoints(xyt, state, dt=DT, error_th=0.05, verbose=True)

    if fig_filename is not None:
        plt.plot(xyt[:,0], xyt[:,1], c='blue', marker='*')
        trajectory = np.array(trajectory)
        plt.plot(trajectory[:,0], trajectory[:,1], c='red', marker='s')
        plt.xlim(-3, 1)
        plt.ylim(-2, 2)
        plt.axis('equal')
        plt.savefig(fig_filename)
