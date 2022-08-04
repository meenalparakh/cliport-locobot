"""Data collection script."""

import os
import hydra
import numpy as np
import random
import cv2
from cliport import tasks
from cliport.dataset import RavensDataset, FOLDER_PREFIX
from cliport.environments.environment import Environment
from torch.utils.data import DataLoader

import pdb
import glob
from cliport import agents

from cliport.utils.multiprocessing_utils import UptownFunc
from cliport.utils import utils
from copy import copy
import pickle
from multiprocessing import cpu_count
import time
from cliport.dataset import BOUNDS, FP_CAM_IDX, PIXEL_SIZE, IMG_SHAPE
import torch 
import matplotlib.pyplot as plt
import torch.nn.functional as F

# os.environ['NUMEXPR_MAX_THREADS'] = '96'

def get_p_from_logits(logits):
    prob = torch.exp(logits[0])
    prob = prob/torch.sum(prob)
    p = np.unravel_index(torch.argmax(prob).detach().cpu(), prob.shape)
    prob = prob.detach().cpu().numpy()
    return p, prob

def get_bounding_box(p, width, height, margin):
    d0 = max(0, p[1] - margin), max(0, p[0] - margin)
    d0_ = min(width, p[1] + margin), min(height, p[0] + margin)
    return d0, d0_

def eval_training_data(agent, batch, batch_idx, epoch, save_dir=None, device='cuda'):
    sample, _ = batch
    imgs, labels, crops, values = sample
    batch_size = imgs[0].shape[0]
    margin=5
#     print(f'Batch size: {batch_size}')

    assert(len(imgs) == 4)

    explore_img_1 = imgs[0]
    explore_img_1_label = labels[:, 0].reshape((batch_size, -1)).to(device)
#     print('explore label shape sum', explore_img_1_label.shape, explore_img_1_label.sum())
    
    pick_img = imgs[1]
    pick_img_label = labels[:, 1].reshape((batch_size, -1)).to(device)
    
    explore_preds = agent.attn_forward(explore_img_1.to(device))
    pick_preds = agent.attn_forward(pick_img.to(device))
    
    cross_entropy_1 = -explore_img_1_label*F.log_softmax(explore_preds.reshape((batch_size, -1)), dim=1)
    print(f'Batch: {batch_idx}, Loss: {cross_entropy_1.sum()/batch_size}')
    
    for idx in range(batch_size):
        
        true_label = explore_img_1_label[idx].reshape((192, 192)).cpu()
#         print('true label shape, sum', true_label.shape, true_label.sum())
        true_p = np.unravel_index(torch.argmax(true_label), true_label.shape)
        true_label = true_label.cpu().numpy()
           
        logits = explore_preds[idx]
#         print('Logits shape', logits.shape)
        p, prob = get_p_from_logits(logits)
        print('Data idx:', idx, 'ps:', true_p, p)

        img = explore_img_1[idx].permute(1, 2, 0)[:,:,:3]
        img = np.rint(img.detach().cpu().numpy()*255)
        img = img.astype('uint8').copy()
        height, width = img.shape[:2]
        
        d0, d0_ = get_bounding_box(p, width, height, margin)
        d0_true, d0_true_ = get_bounding_box(true_p, width, height, margin)
     
        cv2.rectangle(img, d0, d0_, (255, 0, 0), 1)
        cv2.rectangle(img, d0_true, d0_true_, (0, 255, 0), 1)
        
        fname = f'{epoch}_{batch_idx}_{idx}'
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, fname+'_color.jpeg'), img)
        plt.imsave(os.path.join(save_dir, fname+'_prob_map.jpeg'), prob)
        plt.imsave(os.path.join(save_dir, fname+'_true_prob_map.jpeg'), true_label)    
    
def get_pose_from_pixel(env, p, bot_pose):
    x, y, z = utils.pix_to_xyz(p, 0.2, BOUNDS, PIXEL_SIZE, skip_height=True)
    z = 0.2
    X_WL = utils.get_transformation_matrix(bot_pose)
    p_W = X_WL @ np.array([[x], [y], [z], [1.0]])
    pickplace_ori = np.array(env.pb_client.getQuaternionFromEuler(
                                    [np.pi/2, np.pi/2, 0]))
    pose = (p_W[:3,0], pickplace_ori)
    return pose

def save_labelled_img(img, prob_map, p, dir_path, fname, margin=5):
    original_img = img.permute(1, 2, 0)
    height, width = original_img.shape[:2]
    d0, d0_ = get_bounding_box(p, width, height, margin)

    color = np.rint(original_img[:,:,:3].detach().cpu().numpy()*255)
#     prob_map = prob_map.detach().cpu().numpy()

    cv2.rectangle(color, d0, d0_, (255, 0, 0), 1)
    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(dir_path, fname+'_color.jpeg'), color)
    plt.imsave(os.path.join(dir_path, fname+'_prob_map.jpeg'), prob_map)

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

def preprocess_image(img, device='cuda'):
    img[:,:,:3] = img[:,:,:3]/255.0
    img = torch.tensor(img).float().permute(2, 0, 1)
    img = img[None, ...]
    img = img.to(device)
    return img

def get_image_wrapper(obs):
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
    img = get_image(substep_obs, cam_configs)
    return img

def act_pick(env, obs, agent, save_dir=None, fname=None, device='cuda'):
    img = preprocess_image(get_image_wrapper(obs), device=device)
    logits = agent.attn_forward(img)[0]
    
    p, prob_map = get_p_from_logits(logits)

    img = img[0]
    if save_dir is not None:
        save_labelled_img(img, prob_map, p, save_dir, fname, margin=5)

    pick_pose = get_pose_from_pixel(env, p, obs[-1]['bot_pose'])
    return pick_pose

def act_place(env, obs, agent, save_dir=None, fname=None, device='cuda'):
    img = preprocess_image(get_image_wrapper(obs), device=device)
    logits = agent.transport_forward(img)[0]
    p, prob_map = get_p_from_logits(logits)
    
    img = img[0]
    if save_dir is not None:
        save_labelled_img(img, prob_map, p, save_dir, fname, margin=5)

    place_pose = get_pose_from_pixel(env, p, obs[-1]['bot_pose'])
    return place_pose

def policy_evalute(cfg):

    device='cuda'
    env = Environment(
        cfg['assets_root'],
        disp=cfg['disp'],
        shared_memory=cfg['shared_memory'],
        hz=480,
        record_cfg=cfg['record']
    )
    task = tasks.names[cfg['task']]()
    task.mode = cfg['mode']
    
    model_ckpt = cfg['eval']['model_ckpt']
    cfg['name'] = 'eval_agent'
#     agent = agents.names['locobot'](cfg)
    agent = agents.names['locobot'].load_from_checkpoint(model_ckpt)
    
    agent.eval()
    agent = agent.to(device)
    
    record = cfg['record']['save_video']
    save_data = cfg['save_data']
    # Initialize scripted oracle agent and dataset.

    data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], 'eval'))
    print(f"Saving to: {data_path}")
    print(f"Mode: {task.mode}")
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    seed = cfg['seed']

    rollout_summary = np.zeros(cfg['n'])
    
    print('Agent loaded')
    
    if cfg['eval']['on_train_data']:
        train_ds = RavensDataset(os.path.join(cfg['data_dir'], '{}-train'.format(cfg['task'])), cfg,
                             store=False, cam_idx=[0], n_demos=cfg['n'], augment=False, randomize=False)
        train_loader = DataLoader(train_ds, batch_size=32,
                                num_workers=20,
                                shuffle = True)
        print('Train dataset loaded')
        for epoch in range(cfg['eval']['epochs']):
            for batch_idx, batch in enumerate(train_loader):
                print(f'Batch idx: {batch_idx}')
                eval_training_data(agent, batch, batch_idx, epoch, save_dir=data_path)
        return None, None
    else:
        for rollout_idx in range(cfg['n']):
            print(f'Rollout: {rollout_idx}')
            episode, total_reward = [], 0
            seed += 2
            np.random.seed(seed)
            random.seed(seed)

            env.set_task(task)
            obs = env.reset()
            info = env.info
            reward = 0

            for step in range(task.max_steps):
                # pdb.set_trace()
                obs = obs[-env.num_turns:]
                ####################################################################
                pick_pose = act_pick(env, obs, agent, data_path, f'explore1_{rollout_idx}_{step}', device=device)
                env.motion_planner(pick_pose[0][:2])
                obs = [env.get_obs_wrapper()]
                ####################################################################
                pick_pose = act_pick(env, obs, agent, data_path, f'pick_{rollout_idx}_{step}', device=device)
                env.task.primitive.pick(env, pick_pose)
                obs = env.turn_around_center(env.table_center)
                ####################################################################
                place_pose = act_place(env, obs, agent, data_path, f'explore2_{rollout_idx}_{step}', device=device)
                env.motion_planner(place_pose[0][:2])
                obs = [env.get_obs_wrapper()]
                ####################################################################
                place_pose = act_place(env, obs, agent, data_path, f'place_{rollout_idx}_{step}', device=device)
                env.task.primitive.place(env, place_pose)
                obs = env.turn_around_center(env.table_center)

                reward, info = env.task.reward()
                done = env.task.done()
                # is_done = check_completion_oracle()
                if done:
                    rollout_summary[rollout_idx] = 1
                    break

        fname = os.path.join(data_path, 'summary.pkl')
        np.savetxt(fname, rollout_summary)
        print('Success rate:', np.sum(rollout_summary)/cfg['n'])
        return np.sum(rollout_summary), cfg['n']

@hydra.main(config_path='./cfg', config_name='data')
def main(cfg):

    print(f'Number of cpus being used: {cpu_count()}')
    # Initialize environment and task.
    print('################################################################################')
    data_path = os.path.join(cfg['data_dir'], "{}-{}".format(cfg['task'], 'eval'))
#     print('Multiprocessing enabled:', cfg['multiprocessing'])

    run = 0
    seed = int(time.time())
    
    cfg['mode']='eval'

    if not cfg['multiprocessing'] or cfg['eval']['on_train_data']:
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


#     num_success, num_tried = 0, 0
#     for fname in os.listdir(data_path):
#         if 'info' in fname:
#             n1, n2 = pickle.load(open(os.path.join(data_path, fname), 'rb'))
#             num_success += n1
#             num_tried += n2

#     print(f'Total successful episodes: {num_success}')
#     print(f'Total episodes tried: {num_tried}')
#     print(f'Success rate (upper bound): {num_success/num_tried}')

#         # if args.remove_old:
#     for fname in os.listdir(data_path):
#         if 'info' in fname:
#             # print(fname)
#             os.remove(os.path.join(data_path, fname))
#     with open(os.path.join(data_path, 'info_.pkl'), 'wb') as f:
#         pickle.dump([num_success, num_tried], f)

    ## check if directory exists, if not create one



    # print(f'Total trajectories tried: {num_trajs_tried}')
    # print('Successful trajectories:', cfg['n'])
    # print('Success rate (approx):', cfg['n']/num_trajs_tried)


if __name__ == '__main__':

    main()
