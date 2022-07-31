"""Data collection script."""

import os
import hydra
import numpy as np
import random

from cliport import tasks
from cliport.dataset import RavensDataset, FOLDER_PREFIX
from cliport.environments.environment import Environment
import pdb
import glob
from cliport.utils.multiprocessing_utils import UptownFunc
from copy import copy
import pickle
from multiprocessing import cpu_count
import time
from cliport.dataset import BOUNDS, FP_CAM_IDX, PIXEL_SIZE, IMG_SHAPE

# os.environ['NUMEXPR_MAX_THREADS'] = '96'

def get_pose_from_pixel(p, bot_pose):
    x, y, z = utils.pix_to_xyz(p, 0.2, BOUNDS, PIXEL_SIZE, skip_height=True)
    z = 0.2
    X_WL = utils.get_transformation_matrix(bot_pose)
    p_W = X_WL @ np.array([[x], [y], [z], [1.0]])
    pickplace_ori = np.array(env.pb_client.getQuaternionFromEuler(
                                    [np.pi/2, np.pi/2, 0]))
    pose = (p_W[:3,0], pickplace_ori)
    return pose

def save_labelled_img(img, p, location, margin=5):
    height, width = img.shape[:2]
    d0 = max(0, p[1] - margin), max(0, p[0] - margin)
    d0_ = min(width, p[1] + margin), min(height, p[0] + margin)
    color = img[:,:,:3]
    cv2.rectangle(color, d0, d0_, (255, 0, 0), 1)
    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    cv2.imwrite(location, color)

def get_image(obs, cam_config):
    """Stack color and height images image."""
    cmap, hmap = utils.get_fused_heightmap(
        obs, cam_config, BOUNDS, PIXEL_SIZE)
    kernel = np.ones((2,2), np.uint8)
    cmap = cv2.dilate(cmap, kernel, iterations=1)
    hmap = cv2.dilate(hmap, kernel, iterations=1)

    img = np.concatenate((cmap,
                          hmap[Ellipsis, None]), axis=2)

    assert img.shape == IMG_SHAPE, img.shape
    return img

def preprocess_image(img):
    img[:,:,:3] = img[:,:,:3]/255.0
    img = torch.tensor(img).float().permute(2, 0, 1)
    img = img[None, ...]
    return img

def get_image_wrapper(self, obs):
    """Stack color and height images image."""
    # print('Info:', obs[substep]['configs'][0]['position'])
    cam_configs = []
    substep_colors = []
    substep_depths = []
    for lower_substeps in range(len(obs)):
        substep_colors.extend(obs[lower_substeps]['image']['color'][FP_CAM_IDX:FP_CAM_IDX+1])
        substep_depths.extend(obs[lower_substeps]['image']['depth'][FP_CAM_IDX:FP_CAM_IDX+1])
        # for idx in [FP_CAM_IDX]:
        config = obs[lower_substeps]['configs'][FP_CAM_IDX]
        pos, ori = config['position'], config['rotation']
        # print(f'Substep: {substep}, Camera {idx}: position: {pos}, rotation: {ori}')
        # if (idx == self.fp_cam_idx) and (self.img_frame == 'fp'):
        bot_pose = obs[lower_substeps]['bot_pose']
        X_WL = utils.get_transformation_matrix(bot_pose)
        X_WC = utils.get_transformation_matrix((config['position'],
                                                config['rotation']))
        X_LC = np.linalg.inv(X_WL) @ X_WC
        pos, ori = utils.get_pose_from_transformation(X_LC)
        config['position'], config['rotation'] = pos, ori

        cam_configs.append(config)

    substep_obs = {'color': substep_colors, 'depth': substep_depths}
    img = self.get_image(substep_obs, cam_configs)
    return img

def act_pick(obs, agent, save_location=None):
    img = preprocess_image(get_image_wrapper(obs))
    prob_map = agent.attention_layers(img)[0].cpu().numpy()
    p = np.unravel_index(torch.argmax(prob_map), prob_map.shape)
    img = img[0]
    if save_location is not None:
        save_labelled_img(img, p, save_location, margin=5)

    pick_pose = get_pose_from_pixel(p, obs[-1]['bot_pose'])
    return pick_pose

def act_place(obs, agent, save_location=None):
    img = preprocess_image(get_image_wrapper(obs))
    prob_map = agent.transport_layers(img)[0].cpu().numpy()
    p = np.unravel_index(torch.argmax(prob_map), prob_map.shape)
    img = img[0]
    if save_location is not None:
        save_labelled_img(img, p, save_location, margin=5)

    place_pose = get_pose_from_pixel(p, obs[-1]['bot_pose'])
    return place_pose

def policy_evalute(cfg, agent, save_traj=True):

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

    data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], 'eval'))
    print(f"Saving to: {data_path}")
    print(f"Mode: {task.mode}")

    seed = cfg['seed']

    rollout_summary = np.zeros(cfg['n'])
    for rollout_idx in range(cfg['n']):
        episode, total_reward = [], 0
        seed += 2
        np.random.seed(seed)
        random.seed(seed)

        env.set_task(task)
        obs = env.reset()
        info = env.info
        reward = 0
        rollout_dir = os.path.join(data_path, f'{rollout_idx}')
        if not os.path.exists(rollout_dir):
            os.makedirs(rollout_dir)

        for step in range(task.max_steps):
            # pdb.set_trace()
            obs = obs[-env.num_turns:]
            ####################################################################
            pick_pose = act_pick(obs, agent,
                            os.path.join(rollout_dir, f'{step}_explore1.jpeg'))
            env.motion_planner(pick_pose[0][:2])
            obs = [env.get_obs_wrapper()]
            ####################################################################
            pick_pose = act_pick(obs, agent,
                            os.path.join(rollout_dir, f'{step}_pick.jpeg'))
            env.task.pick(pick_pose)
            obs = env.turn_around_center(env.table_center))
            ####################################################################
            place_pose = act_place(obs, agent,
                            os.path.join(rollout_dir, f'{step}_explore2.jpeg'))
            env.motion_planner(place_pose[0][:2])
            obs = [env.get_obs_wrapper()]
            ####################################################################
            place_pose = act_pick(obs, agent,
                            os.path.join(rollout_dir, f'{step}_place.jpeg'))
            env.task.place(pick_pose)
            obs = env.turn_around_center(env.table_center))

            reward, info = self.task.reward()
            done = self.task.done()
            # is_done = check_completion_oracle()
            if done:
                rollout_summary[rollout_idx] = 1
                break

    fname = os.path.join(data_path, 'summary.pkl')
    np.savetxt(fname, rollout_summary)
    return np.sum(rollout_summary), cfg['n']

@hydra.main(config_path='./cfg', config_name='data')
def main(cfg):

    print(f'Number of cpus being used: {cpu_count()}')
    # Initialize environment and task.
    print('################################################################################')
    data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], 'eval'))
    print('Multiprocessing enabled:', cfg['multiprocessing'])

    run = 0
    seed = int(time.time())

    if not cfg['multiprocessing']:
        if not cfg['run_specified']:
            cfg['seed'] = seed

        success_count, total = policy_evalute(cfg)

    else:
        cfg['disp'] = False
        num_trajs_per_process = cfg['trajs_per_process']
        num_processes = cfg['n']//num_trajs_per_process
        num_processes_list = [num_trajs_per_process]*num_processes
        remaining = cfg['n'] - (num_processes * num_trajs_per_process)
        if remaining > 0:
            num_processes_list.append(remaining)

        arguments = []
        for i in range(len(num_processes_list)):
            cfg_ = copy(cfg)
            cfg_['run_id'] = run + i
            cfg_['n'] = num_processes_list[i]
            cfg_['seed'] = seed + 5*i*num_trajs_per_process
            arguments.append(cfg_)

        P = UptownFunc()
        P.parallelise_function(arguments, collect_data)
        # num_trajs_tried = 0
        # for result in results:
        #     num_trajs_tried_ = result
        #     num_trajs_tried += num_trajs_tried_

    num_success, num_tried = 0, 0
    for fname in os.listdir(data_path):
        if 'info' in fname:
            n1, n2 = pickle.load(open(os.path.join(data_path, fname), 'rb'))
            num_success += n1
            num_tried += n2

    print(f'Total successful episodes: {num_success}')
    print(f'Total episodes tried: {num_tried}')
    print(f'Success rate (upper bound): {num_success/num_tried}')

        # if args.remove_old:
    for fname in os.listdir(data_path):
        if 'info' in fname:
            # print(fname)
            os.remove(os.path.join(data_path, fname))
    with open(os.path.join(data_path, 'info_.pkl'), 'wb') as f:
        pickle.dump([num_success, num_tried], f)

    ## check if directory exists, if not create one



    # print(f'Total trajectories tried: {num_trajs_tried}')
    # print('Successful trajectories:', cfg['n'])
    # print('Success rate (approx):', cfg['n']/num_trajs_tried)


if __name__ == '__main__':

    main()
