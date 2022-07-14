"""Data collection script."""

import os
import hydra
import numpy as np
import random

# from cliport import tasks
# from cliport.dataset import RavensDataset
# from cliport.environments.environment import Environment
import pdb

import tasks
from cliport.dataset import RavensDataset
from cliport.environments.environment import Environment

@hydra.main(config_path='./cfg', config_name='data')
def main(cfg):
    # Initialize environment and task.
    env = Environment(
        cfg['assets_root'],
        disp=cfg['disp'],
        shared_memory=cfg['shared_memory'],
        hz=480,
        record_cfg=cfg['record']
    )
    # task = tasks.names['stack-block-pyramid-seq-seen-colors']()
    task = tasks.names['put-block-in-container-seen-colors']()
    task.mode = cfg['mode']
    record = cfg['record']['save_video']
    save_data = cfg['save_data']

    # Initialize scripted oracle agent and dataset.
    agent = task.oracle(env, locobot=True)
    data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], task.mode))
    dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
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

    env.set_task(task)
    obs = env.reset()

    input()
    # env.get_workspace_edgepoints(show = True)

    # pdb.set_trace()
    # while True:
    #     env.step_simulation()
    #
    # obj = env.add_cube(0.01, [0.05, 0.1, 0.05], (0.1, 0.5, 0.5, 1),
    #                         [-1, -1, 0.1],
    #                         (0, 0, 0, 1))
    #
    # object_position, object_ori = env.pb_client.getBasePositionAndOrientation(obj)
    # theta = env.pb_client.getEulerFromQuaternion(object_ori)[-1]
    # pre_pick_ori = env.pb_client.getQuaternionFromEuler([np.pi/2, np.pi/2, 0])
    # pre_pick_z = object_position[2] + 0.2
    # pre_pick_pos = [*object_position[:2], pre_pick_z]
    # pick_pos = [*object_position[:2], object_position[2] + 0.01]
    #
    # print('object Position:', pre_pick_pos[:2])
    #
    # env.movej(env.locobot.homej)
    # env.move_to(np.array(pre_pick_pos[:2]), tol = 0.4)
    # env.turn_to_point(pre_pick_pos[:2], tol = np.pi/6)
    # env.movej(env.locobot.actionj)
    #
    # contacts = False
    # while not contacts:
    #     env.movep((pre_pick_pos, pre_pick_ori))
    #     env.movep((pick_pos, pre_pick_ori), collision_detector = True)
    #     env.ee.activate()
    #     env.movep((pre_pick_pos, pre_pick_ori))
    #     contacts = env.ee.detect_contact()
    # print(f'Bodies in contact: {contacts}')
    #
    # env.movej(env.locobot.homej)
    # env.move_to((0, 0), tol=0.4)
    # env.turn_to_point((0, 0), tol=np.pi/6)
    # env.movej(env.locobot.actionj)
    #
    # drop_ori = env.locobot.get_ee_pose()[1]
    # env.movep(((0, 0, 0.2), drop_ori))
    # env.ee.release()
    # [env.step_simulation() for _ in range(10)]
    # env.movej(env.locobot.homej)
    #
    #
    # while True:
    #     env.step_simulation()

    # # Collect training data from oracle demonstrations.
    # while dataset.n_episodes < cfg['n']:
    episode, total_reward = [], 0
    seed += 2

    # Set seeds.
    np.random.seed(seed)
    random.seed(seed)

    print('Oracle demo: {}/{} | Seed: {}'.format(dataset.n_episodes + 1, cfg['n'], seed))

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
        act = agent.act(obs, info)
        episode.append((obs, act, reward, info))
        lang_goal = info['lang_goal']
        obs, reward, done, info = env.step(act)
        total_reward += reward
        print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
        if done:
            break
    episode.append((obs, None, reward, info))

    # End video recording
    if record:
        env.end_rec()

    # Only save completed demonstrations.
    if save_data and total_reward > 0.99:
        dataset.add(seed, episode)


if __name__ == '__main__':
    main()
