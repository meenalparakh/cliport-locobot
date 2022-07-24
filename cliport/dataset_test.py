"""Main training script."""

import os
from pathlib import Path

import torch
from cliport import agents
from cliport.dataset import RavensDataset, RavensMultiTaskDataset

import hydra
import pdb
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2

@hydra.main(config_path="./cfg", config_name='train')
def main(cfg):

    # Config
    data_dir = cfg['train']['data_dir']
    task = cfg['train']['task']
    agent_type = cfg['train']['agent']
    n_demos = cfg['train']['n_demos']
    n_val = cfg['train']['n_val']
    name = '{}-{}-{}'.format(task, agent_type, n_demos)

    # Datasets
    dataset_type = cfg['dataset']['type']

    train_ds = RavensDataset(os.path.join(data_dir, '{}-train'.format(task)), cfg,
                            store=False, cam_idx=[0,1], n_demos=n_demos, augment=False)

# data = train_ds.load(0)
    episode, _  = train_ds.load(train_ds.episode_paths[-1][:-10])

    print(f'Episode length: {len(episode)}')
    for step in range(len(episode)):
        sample = episode[step]
        obs, act, reward, info = sample

        datapoint = train_ds.process_sample(sample, augment=False)
        # print(datapoint)
        # print(len(datapoint['img']))
        # print(datapoint['p0'])
        print('Number of substeps:', len(datapoint['img']))
        for substep in range(len(datapoint['img'])):
            plt.imsave(f'/Users/meenalp/Desktop/step_{step}_{substep}.jpeg', obs[substep]['image']['color'][0])
            color = cv2.cvtColor(datapoint['img'][substep][:,:,:3], cv2.COLOR_RGB2BGR)
            depth = datapoint['img'][substep][:,:,3]
            print('Maximum in depth image', np.max(depth))
            depth = depth/np.max(depth)

            height, width = color.shape[:2]
            if datapoint['p0'] is None:
                break
            p0, p1 = datapoint['p0'][substep], datapoint['p1'][substep]
            print('PICK AND PLACE POINTS', p0, p1)
            d0 = max(0, p0[1] - 10), max(0, p0[0] - 10)
            d0_ = min(width, p0[1] + 10), min(height, p0[0] + 10)
            d1 = max(0, p1[1] - 10), max(0, p1[0] - 10)
            d1_ = min(width, p1[1] + 10), min(height, p1[0] + 10)

            cv2.rectangle(color, d0, d0_, (0, 0, 255), 2)
            cv2.rectangle(color, d1, d1_, (0, 255, 0), 2)

            # print(depth)
            cv2.imwrite(f'/Users/meenalp/Desktop/labelled_img_step_{step}_{substep}.png', color)
            plt.imsave(f'/Users/meenalp/Desktop/labelled_img_step_depth_{step}_{substep}.png', depth)


if __name__ == '__main__':
    main()


# python cliport/train.py train.data_dir=$(pwd)/data_check_run train.gpu=0 train.task=stack-block-pyramid-seq-seen-colors train.n_demos=5 train.n_val=2 train.agent=cliport
