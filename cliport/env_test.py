import os
import hydra
import numpy as np
import random
import cv2
from cliport import tasks
from cliport.environments.environment import Environment
import pdb
import glob
from copy import copy
import time
import pickle

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

    env.set_task(task)

    obs = env.reset()
    with open('/Users/meenalp/Desktop/MEng/segmentation/example.pkl', 'wb') as f:
        pickle.dump(obs, f)

    print('Okay')

if __name__ == '__main__':
    main()
