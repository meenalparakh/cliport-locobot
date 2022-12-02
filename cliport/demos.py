"""Data collection script."""

import os
import hydra
import numpy as np
import random

from cliport import tasks
from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment
import pdb

@hydra.main(config_path='./cfg', config_name='data')
def main(cfg):
    # Initialize environment and task.
    print('################################################################################')
    env = Environment(
        cfg['assets_root'],
        disp=cfg['disp'],
        shared_memory=cfg['shared_memory'],
        hz=480,
        record_cfg=cfg['record']
    )
    task = tasks.names[cfg['task']]()
    task.mode = cfg['mode']
    record = cfg['record']['save_video']
    save_data = cfg['save_data']

    # Initialize scripted oracle agent and dataset.
    agent = task.oracle(env, locobot=cfg['locobot'])
    data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], task.mode))
    dataset = RavensDataset(data_path, cfg, store=True, n_demos=0, augment=False)
    print(f"Saving to: {data_path}")
    print(f"Mode: {task.mode}")

    # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
    seed = dataset.max_seed
    if seed < 0:
        if task.mode == 'train':
            seed = -2
        elif task.mode == 'val': # NOTE: beware of increasing val set to >100
            seed = -1
        elif task.mode == 'test':
            seed = -1 + 10000
        else:
            raise Exception("Invalid mode. Valid options: train, val, test")

    # Collect training data from oracle demonstrations.
    while dataset.n_episodes < cfg['n']:
        episode, total_reward = [], 0
        seed += 2

        seed = np.random.randint(0, 100)
        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)

        print('Oracle demo: {}/{} | Seed: {}'.format(dataset.n_episodes + 1, cfg['n'], seed))

        env.set_task(task)
        obs = env.reset()
        info = env.info
        reward = 0

        # Unlikely, but a safety check to prevent leaks.
        if task.mode == 'val' and seed > (-1 + 10000):
            raise Exception("!!! Seeds for val set will overlap with the test set !!!")

        # Start video recording (NOTE: super slow)
        if record:
            env.start_rec(f'{dataset.n_episodes+1:06d}')

        # Rollout expert policy
        for _ in range(task.max_steps):
            # pdb.set_trace()
            obs = obs[-env.num_turns:]
            act = agent.act(obs, info)
            lang_goal = info['lang_goal']
            _obs, _reward, _done, _info = env.step(act)
            for substep in range(len(_obs)):
                print('Data collection:', _obs[substep]['configs'][3]['position'])
            # if isinstance(_obs, list):
            episode.append(([*obs, *(_obs[:-env.num_turns])],
                            act,
                            reward,
                            info))
            # else:
            # episode.append((obs, act, reward, info))

            # print('Step', info[1]['bot_pose'], info[1]['cam_configs'])
            obs = _obs
            reward = _reward
            done = _done
            info = _info
            total_reward += reward

            print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
            if done:
                break
        episode.append((obs[-env.num_turns:], None, reward, info))

        # End video recording
        if record:
            env.end_rec()

        # Only save completed demonstrations.
        if save_data and total_reward > 0.99:
            dataset.add(seed, episode)


if __name__ == '__main__':
    main()
