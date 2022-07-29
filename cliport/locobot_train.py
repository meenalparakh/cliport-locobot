"""Main training script."""

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
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

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

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
                             store=False, cam_idx=[1], n_demos=n_demos, augment=False)
    val_ds = RavensDataset(os.path.join(data_dir, '{}-val'.format(task)), cfg,
                            store=False, cam_idx=[1], n_demos=n_val, augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'],
                                num_workers=cfg['train']['num_cpus'],
                                shuffle = True)
    val_loader = DataLoader(val_ds, batch_size=cfg['train']['batch_size'],
                                num_workers=cfg['train']['num_cpus'],
                                shuffle = False)

    print(f'Length of dataset: {len(train_ds)}')
    print(f'Length of train loader: {len(train_loader)}')

    agent = agents.names[agent_type](name, cfg)
    agent.apply(initialize_weights)

    logs_dir = name
    print(f'Log dir: {name}')
    logger = WandbLogger(project=logs_dir)
    print('logger initialized!.')

    val_checkpoint = ModelCheckpoint(
        filename='min_val_loss',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        dirpath = logs_dir + '/'
    )
    latest_checkpoint = ModelCheckpoint(
        filename = logs_dir + '/latest',
        monitor = 'step',
        mode = 'max',
        every_n_train_steps = 500,
        save_top_k = 1
    )

    callbacks = [val_checkpoint, latest_checkpoint]
    max_epochs = cfg['train']['max_epochs']
    trainer = Trainer(callbacks=callbacks,
                      logger=logger,
                     accelerator=accelerator,
                     devices=num_gpus,
                      precision=16,
                     # check_val_every_n_epoch=val_every_n_epochs,
                      max_epochs=max_epochs)

    print('Starting fitting')

    # print(f'Num samples: {trainer.num_training_samples}')

    trainer.fit(agent, train_loader, val_loader)
    # print(f'Num samples: {trainer.num_training_samples}')

    # for idx, batch in enumerate(train_loader):
    #     agent.training_step(batch, idx)
    print('done')

    # pdb.set_trace()


if __name__ == '__main__':
    main()
