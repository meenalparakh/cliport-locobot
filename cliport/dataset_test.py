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
    if 'multi' in dataset_type:
        train_ds = RavensMultiTaskDataset(data_dir, cfg, group=task, mode='train', n_demos=n_demos, augment=True)
        val_ds = RavensMultiTaskDataset(data_dir, cfg, group=task, mode='val', n_demos=n_val, augment=False)
    else:
        train_ds = RavensDataset(os.path.join(data_dir, '{}-train'.format(task)), cfg, n_demos=n_demos, augment=True)
        val_ds = RavensDataset(os.path.join(data_dir, '{}-val'.format(task)), cfg, n_demos=n_val, augment=False)

    # Initialize agent
    # pdb.set_trace()
    sample, goal = train_ds[0]

    image = sample['img']
    cmap = image[:,:,:3]/255.0
    hmap = image[:,:,3]

    # plt.imsave('/Users/meenalp/Desktop/final_image.png', cmap)
    # plt.imsave('/Users/meenalp/Desktop/final_himage.png', hmap)


if __name__ == '__main__':
    main()


# python cliport/train.py train.data_dir=$(pwd)/data_check_run train.gpu=0 train.task=stack-block-pyramid-seq-seen-colors train.n_demos=5 train.n_val=2 train.agent=cliport
