from locobot.sim.base_env import BaseEnv
from locobot.sim.grasper import Grasper
import gym
from gym import spaces
import numpy as np

from locobot.sim.discrete_env_info import *

class LocobotGym(gym.Env):
    metadata = {'render.modes': ['locobot']}

    def __init__(self, gui = False):
        super(LocobotGym, self).__init__()

        self.env = BaseEnv(gui = gui, realtime=False)
        self.grasper = Grasper(self.env)
        self.action_space = spaces.Discrete(NUM_DISCRETE_ACTIONS)
        # Example for using image as input:
        # self.observation_space = spaces.Box(low=0, high=255, shape=
        #                 (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

    def step(self, action):
        # returns obs, reward, done, {}
        # if action == TERMINATE_EPISODE:

        if action == MOVE_FORWARD_ACTION:
            self.env.bot.move_forward_long()

        elif action == MOVE_FORWARD_SHORT_ACTION:
            self.env.bot.move_forward_short()

        elif action == MOVE_BACKWARD_ACTION:
            self.env.bot.move_backward_long()

        elif action == MOVE_BACKWARD_SHORT_ACTION:
            self.env.bot.move_backward_short()

        elif action == TURN_LEFT_ACTION:
            self.env.bot.move_left_short()

        elif action == TURN_LEFT_LONG_ACTION:
            self.env.bot.move_left_long()

        elif action == TURN_RIGHT_ACTION:
            self.env.bot.move_right_short()

        elif action == TURN_RIGHT_LONG_ACTION:
            self.env.bot.move_right_long()

        elif action == GRASP_ACTION:
            self.grasper.execute_grasp(self.env.object_ids)

        elif action == DROP_ACTION:
            self.grasper.drop_object()

        obs = self.get_observation()
        reward = self.get_reward(action)
        done = self.is_done()

        return obs, reward, done, {}

    def get_reward(self, action):
        raise NotImplementedError

    def is_done(self):
        raise NotImplementedError

    # def reset(self, num_tables = 1, table_dims = None):
    #     self.env.reset(num_tables = num_tables, table_dims = table_dims)
    def reset(self, *args, **kwargs):
        self.env.reset(*args, **kwargs)
        obs = self.get_observation()
        return obs

    def render(self, mode='human', close=False):
    # Render the environment to the screen
        pass

    def get_observation(self):
        gripper_state = self.env.bot.get_gripper_state()
        object_grasped = int(gripper_state == 2)
        # object_grasped = np.array([object_grasped], dtype=int)

        tp_rgb, tp_depth = self.env.get_tp_images()
        fp_rgb, fp_depth = self.env.get_fp_images()

        return tp_rgb, tp_depth, fp_rgb, fp_depth, object_grasped
