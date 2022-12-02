"""Main training script."""

import os
from pathlib import Path
import numpy as np
import torch
from cliport import agents
from cliport.dataset import RavensDataset, RavensMultiTaskDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pdb
import random
import cv2

def save_batch_images(batch, batch_idx, epoch):
    sample, goal = batch
    imgs = sample[0]
    labels = sample[1]
    for i in range(len(imgs[0])):
        for substep in range(len(imgs)):
            img = imgs[substep][i]
            label = labels[i][substep]
            img = img.permute(1, 2, 0)
            color = np.rint(img[:,:,:3].numpy()*255).astype(np.uint8)
            p = np.unravel_index(torch.argmax(label), label.shape)
            print(f'Data:{i}, substep: {substep}, label: {p}')
            assert (torch.sum(label) > 0.999) and (torch.sum(label) < 1.001)
            height, width = color.shape[:2]

            d0 = max(0, p[1] - 10), max(0, p[0] - 10)
            d0_ = min(width, p[1] + 10), min(height, p[0] + 10)
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

            cv2.rectangle(color, d0, d0_, (0, 0, 255), 1)
            cv2.imwrite(f'/home/gridsan/meenalp/cliport-locobot/images/image_{epoch}_{batch_idx}_{i}_{substep}.jpeg', color)


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
                             store=False, cam_idx=[0], n_demos=n_demos, augment=False)
    val_ds = RavensDataset(os.path.join(data_dir, '{}-val'.format(task)), cfg,
                            store=False, cam_idx=[0], n_demos=n_val, augment=False)

    # test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_cpus, shuffle = False)


    train_loader = DataLoader(train_ds, batch_size=32,
                                num_workers=cfg['train']['num_cpus'],
                                shuffle = True)
    val_loader = DataLoader(val_ds, batch_size=cfg['train']['batch_size'],
                                # num_workers=cfg['train']['num_cpus'],
                                shuffle = False)

    print(f'Train loader length: {len(train_loader)}')

    print(f'Length of dataset: {len(train_ds)}')

    show_crops = False
    if show_crops:
        print(i, train_ds.idx_to_episode_step[i])
        print("shape of crops:", crops.shape)
        for i in range(crops.shape[0]):
            crop = crops[i].permute(1, 2, 0)
            cmap = np.rint(crop[:,:,:3].numpy()*255)
            cmap = cv2.cvtColor(cmap, cv2.COLOR_RGB2BGR)
            hmap = crop[:,:,3].numpy()
            cv2.imwrite(f'/Users/meenalp/Desktop/crop_{i}.png', cmap)
            plt.imsave(f'/Users/meenalp/Desktop/crop_hmap_{i}.png', hmap)

#     i = random.choice(range(len(train_ds)))
#     sample, goal = train_ds[i]
#     imgs = sample[0]
#     labels = sample[1]
#     crops = sample[2]
    
    dir_path = '/home/gridsan/meenalp/cliport-locobot/images'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    save_labelled_images = True
    if save_labelled_images:
        for epoch in range(cfg['train']['max_epochs']):
            for batch_idx, batch in enumerate(train_loader):
                print(f'Batch idx: {batch_idx}')
                save_batch_images(batch, batch_idx, epoch)

if __name__ == '__main__':
    main()
