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

@hydra.main(config_path="./cfg", config_name='train')
def main(cfg):
    # Logger
    wandb_logger = WandbLogger(name=cfg['tag']) if cfg['train']['log'] else None

    # Checkpoint saver
    hydra_dir = Path(os.getcwd())
    checkpoint_path = os.path.join(cfg['train']['train_dir'], 'checkpoints')
    last_checkpoint_path = os.path.join(checkpoint_path, 'last.ckpt')
    last_checkpoint = last_checkpoint_path if os.path.exists(last_checkpoint_path) and cfg['train']['load_from_last_ckpt'] else None
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg['wandb']['saver']['monitor'],
        filepath=os.path.join(checkpoint_path, 'best'),
        save_top_k=1,
        save_last=True,
    )

    # Trainer
    max_epochs = cfg['train']['n_steps'] // cfg['train']['n_demos']
    trainer = Trainer(
        gpus=cfg['train']['gpu'],
        fast_dev_run=cfg['debug'],
        logger=wandb_logger,
        checkpoint_callback=checkpoint_callback,
        max_epochs=max_epochs,
        automatic_optimization=False,
        check_val_every_n_epoch=max_epochs // 50,
        resume_from_checkpoint=last_checkpoint,
    )

    # Resume epoch and global_steps
    if last_checkpoint:
        print(f"Resuming: {last_checkpoint}")
        last_ckpt = torch.load(last_checkpoint)
        trainer.current_epoch = last_ckpt['epoch']
        trainer.global_step = last_ckpt['global_step']
        del last_ckpt

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
    val_ds = RavensDataset(os.path.join(data_dir, '{}-val'.format(task)), cfg,
                            store=False, cam_idx=[0,1], n_demos=n_val, augment=False)

    # test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_cpus, shuffle = False)

    # agent = agents.names[agent_type](name, cfg, train_ds, val_ds)

    print(f'Length of dataset: {len(train_ds)}')
    print(f'steps description: {train_ds.idx_to_episode_step}')
    print(f'Numsteps: {train_ds.episode_num_steps}')
    print(f'episode numstep: {train_ds.episode_paths[5]}')
    for i in range(len(train_ds)):
        # goal[0]
        sample, goal = train_ds[i]
        imgs = sample[0]
        print("Number of substeps:", len(imgs))
        for substep in range(len(imgs)):
            im = imgs[substep].permute(1, 2, 0)
            cmap = im[:,:,:3].numpy()
            # print(cmap[80,80,0])
            hmap = im[:,:,3].numpy()
            plt.imsave(f'/Users/meenalp/Desktop/image{substep}.png', cmap)
            plt.imsave(f'/Users/meenalp/Desktop/image_depth{substep}.png', hmap)

        # print("Sample:", sample[0][0].shape)
        break

    train_loader = DataLoader(train_ds, batch_size=32,
                                # num_workers=cfg['train']['num_cpus'],
                                shuffle = True)
    val_loader = DataLoader(val_ds, batch_size=cfg['train']['batch_size'],
                                # num_workers=cfg['train']['num_cpus'],
                                shuffle = False)

    # print(f'Train loader length: {len(train_loader)}')
    # for x, y in train_loader:
    #     sample_img = x[0]
    #     print('Image:', np.max(sample_img[0][0,:,:,3].numpy()))

    # for batch in train_loader:
    #     print(f'Batch shape: {batch.shape}')
    # train_ds[0]
    # trainer.fit(agent)

if __name__ == '__main__':
    main()
