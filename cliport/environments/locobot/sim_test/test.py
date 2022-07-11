import time

import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

from locobot.sim.base_env import BaseEnv

np.set_printoptions(precision=3, suppress=True)

env = BaseEnv(realtime=True)
env.reset()
bot = env.bot.bot
pos, ori = env.bot.get_ee_pose()
env.bot.base_forward()
time.sleep(2)
env.bot.rotate_to_left()
time.sleep(2)
env.bot.stop_base()

rgb, depth = env.get_fp_images()
plt.imshow(rgb)
plt.show()
rgb, depth = env.get_tp_images()
plt.imshow(rgb)
plt.show()
embed()
