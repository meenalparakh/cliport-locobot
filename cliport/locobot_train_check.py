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
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from cliport.locobot_policy_eval import eval_training_data

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
#     n_test = cfg['train']['n_test']
    name = '{}-{}-{}'.format(task, agent_type, n_demos)

    # Datasets
    dataset_type = cfg['dataset']['type']

    train_ds = RavensDataset(os.path.join(data_dir, '{}-train'.format(task)), cfg, randomize=True,
                             store=False, cam_idx=[0], n_demos=n_demos, augment=True)
    val_ds = RavensDataset(os.path.join(data_dir, '{}-val'.format(task)), cfg,
                            store=False, cam_idx=[0], n_demos=n_val, augment=False)
    test_ds = RavensDataset(os.path.join(data_dir, '{}-test'.format(task)), cfg,
                            store=False, cam_idx=[0], n_demos=50, augment=False)
    
    print(f'Length of training dataset: {len(train_ds)}')
    print(f'Length of validation dataset: {len(val_ds)}')
    print(f'Length of test dataset: {len(test_ds)}')
    
    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'],
                                num_workers=cfg['train']['num_cpus'], shuffle = True)
    val_loader = DataLoader(val_ds, batch_size=cfg['train']['batch_size'],
                                num_workers=cfg['train']['num_cpus'], shuffle = False)
    test_loader = DataLoader(test_ds, batch_size=cfg['train']['batch_size'],
                                num_workers=cfg['train']['num_cpus'], shuffle = False)

#     print(f'Length of train loader: {len(train_loader)}')

    cfg['name'] = name
    agent = agents.names[agent_type](cfg)
    agent.apply(initialize_weights)

    logs_dir = name
    print(f'Log dir: {name}')
#     logger = CSVLogger("CSV logs", name=logs_dir)
    logger = WandbLogger(project=logs_dir, offline=True)
    print('logger initialized!.')

    max_epochs = cfg['train']['max_epochs']

    val_checkpoint = ModelCheckpoint(
        filename='min_val_loss',
        monitor='transport_val_loss',
        mode='min',
        save_top_k=3,
        dirpath=logs_dir + '/'
    )
    latest_checkpoint = ModelCheckpoint(
        filename='latest',
        monitor='step',
        mode='max',
        every_n_epochs=1,
        save_top_k=1,
        dirpath=logs_dir + '/'
    )
    
    callbacks = [val_checkpoint, latest_checkpoint]
    trainer = Trainer(logger=logger,
                      callbacks=callbacks,
                      accelerator='gpu',
                      devices=1,
                      precision=16,
                      log_every_n_steps=10,
                      check_val_every_n_epoch=cfg['train']['inv_val_freq'],
                      max_epochs=cfg['train']['max_epochs'])

    print('Starting fitting')
    if cfg['train']['ckpt_path'] == 'None':
        cfg['train']['ckpt_path'] = None
        
    trainer.fit(agent, train_loader, val_loader, ckpt_path=cfg['train']['ckpt_path'] )
    print('done')
    print(f'Best checkpoint at: {val_checkpoint.best_model_path}')
    print('Evaluating')

#     agent.eval()
#     agent = agent.to('cuda')
#     for batch_idx, batch in enumerate(train_loader):
#         print(f'Batch idx: {batch_idx}')
#         eval_training_data(agent, batch, batch_idx, 0, 
#                            save_dir=os.path.join(data_dir,'{}-eval'.format(task)))
    
    trainer.test(dataloaders=test_loader, ckpt_path = val_checkpoint.best_model_path)

    f = open(logs_dir + '/model.txt', 'w')
    f.write(str(agent))

    f = open(logs_dir + '/model_params.txt', 'a')
    for attr in dir(agent.cfg):
        f.write(f'{attr}: {getattr(agent.cfg, attr)}\n')

    return 

if __name__ == '__main__':
    main()
