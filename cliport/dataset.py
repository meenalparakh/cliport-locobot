"""Image dataset."""

import os
import pickle
import warnings
import glob

import numpy as np
from torch.utils.data import Dataset
import torch

from cliport import tasks
from cliport.tasks import cameras
from cliport.utils import utils
import cv2
import pdb
import matplotlib.pyplot as plt
from cliport.utils import utils

# See transporter.py, regression.py, dummy.py, task.py, etc.
# PIXEL_SIZE = 0.003125
# CAMERA_CONFIG = cameras.RealSenseD415.CONFIG
# BOUNDS = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
# NUM_SUBSTEPS = 4

# Names as strings, REVERSE-sorted so longer (more specific) names are first.
TASK_NAMES = (tasks.names).keys()
TASK_NAMES = sorted(TASK_NAMES)[::-1]
FOLDER_PREFIX = 'PROCESS'
FP_CAM_IDX = 0
# NUM_EXPLORATION_IMAGES = 2
MAX_SUBSTEPS = 10
IMAGE_PAIRINGS = [[0,1], [2], [3,4], [5]]
BOUNDS = np.array([[0.05, 1.25], [-0.6, 0.6], [0.10, 0.28]])
PIXEL_SIZE = 0.00625
IMG_SHAPE = (192, 192, 4)

class RavensDataset(Dataset):
    """A simple image dataset class."""

    def __init__(self, path, cfg, store,
                 cam_idx=[0], n_demos=0, augment=False,
                 track=False, process_num=0):
        """A simple RGB-D image dataset."""
        self._path = path

        self.cfg = cfg
        self.sample_set = []
        self.max_seed = -1
        self.n_episodes = 0
        self.process_num = process_num
        self.folder_prefix = FOLDER_PREFIX + str(self.process_num).zfill(5)
        self.images = self.cfg['dataset']['images']
        self.cache = self.cfg['dataset']['cache']
        self.n_demos = n_demos
        self.augment = augment

        self.aug_theta_sigma = self.cfg['dataset']['augment']['theta_sigma'] if 'augment' in self.cfg['dataset'] else 60
        # legacy code issue: theta_sigma was newly added
        # self.pix_size = 0.003125
        # self.in_shape = (320, 256, 6)

        self.pix_size = PIXEL_SIZE
        self.in_shape = IMG_SHAPE
        self.cam_idx = cam_idx
        self.fp_cam_idx = FP_CAM_IDX
        if not store:
            self.img_frame = self.cfg['dataset']['img_frame']
            if self.img_frame == 'fp':
                self.cam_idx = [self.fp_cam_idx]
        # self.pix_size = 0.002
        self.depth_scale = 1000.0
        # self.in_shape = (320, 160, 6)
        # self.in_shape = (320, 192, 6)
        self.cam_config = cameras.RealSenseD415.CONFIG
        self.bounds = BOUNDS
        self.crop_size = 40

        self.crop_size = 40
        self.pad_size = int(self.crop_size / 2)
        self.n_rotations = 4
        self.rotator = utils.ImageRotator(self.n_rotations)
        self.padding = np.zeros((3, 2), dtype=int)
        self.padding[:2, :] = self.pad_size
        self.kernel_shape = (self.crop_size, self.crop_size, self.in_shape[2])

        # if not hasattr(self, 'output_dim'):
        #     self.output_dim = 3
        # if not hasattr(self, 'kernel_dim'):
        #     self.kernel_dim = 3

        self._cache = {}

        if store:
            path = os.path.join(self._path, self.folder_prefix)
            os.makedirs(path)

        if self.n_demos > 0:
            self.images = self.cfg['dataset']['images']
            self.cache = self.cfg['dataset']['cache']

            # Check if there sufficient demos in the dataset
            episode_paths = sorted(self.get_episode_paths())
            self.n_episodes = len(episode_paths)
            if self.n_demos > self.n_episodes:
                raise Exception(f"Requested training on {self.n_demos} demos, "
                f"but only {self.n_episodes} demos exist in the dataset path: {self._path}.")

            episodes = np.random.choice(range(self.n_episodes), self.n_demos, False)
            ###
            num_steps = self.get_steps_count(episode_paths)
            self.set(episode_paths, episodes, num_steps)
            ###

    def get_episode_paths(self):
        return glob.glob(self._path + "/" + FOLDER_PREFIX + "*/episode*/label_*.pkl")

    def get_steps_count(self, episode_paths):
        steps = []
        for epi_path in episode_paths:
            n = len(pickle.load(open(epi_path, 'rb')))
            steps.append(n)
        return steps

    def set(self, episode_paths, episodes, num_steps):
        """Limit random samples to specific fixed set."""
        self.episode_paths = episode_paths
        self.episode_num_steps = num_steps
        self.sample_set = episodes
        self.idx_to_episode_step = []
        for episode_id in self.sample_set:
            for step in range(self.episode_num_steps[episode_id]-1):
                self.idx_to_episode_step.append((episode_id, step))

    def add(self, seed, episode):
        """Add an episode to the dataset.

        Args:
          seed: random seed used to initialize the episode.
          episode: list of (obs, act, reward, info) tuples.
        """
        episode_fname = f'episode{self.n_episodes:03d}-{seed}'
        episode_path = os.path.join(self._path, self.folder_prefix, episode_fname)
        os.makedirs(episode_path)

        color_path = os.path.join(episode_path, 'color')
        depth_path = os.path.join(episode_path, 'depth')
        # if not os.path.exists(color_path):
        os.makedirs(color_path)
        os.makedirs(depth_path)

        steps_data = []
        for step in range(len(episode)):
            obs, act, r, i = episode[step]
            print('here', act)
            if step == (len(episode)-1):
                imgs = self.get_image_wrapper(obs, IMAGE_PAIRINGS[:1])
            else:
                imgs = self.get_image_wrapper(obs, IMAGE_PAIRINGS)
            p0s, p1s = None, None
            p0_thetas, p1_thetas = None, None
            perturb_params =  None

            if act:
                p0s, p0_thetas, p1s, p1_thetas = self.transform_pick_place(act, obs)

            steps_data.append((p0s, p0_thetas, p1s, p1_thetas))

            for substep in range(len(imgs)):
                color = imgs[substep][:,:,:3]
                color_fname = f'S{step}-U{substep}.png'
                color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(color_path, color_fname), color)

                depth = imgs[substep][:,:,3]
                depth_fname = f'S{step}-U{substep}.png'
                sdepth = depth * self.depth_scale
                cv2.imwrite(os.path.join(depth_path, depth_fname),
                            sdepth.astype(np.uint16))

        field_fname = 'label_info.pkl'
        field_path = os.path.join(episode_path, field_fname)
        with open(field_path, 'wb') as f:
            pickle.dump(steps_data, f)

        self.n_episodes += 1
        self.max_seed = max(self.max_seed, seed)


    def load(self, episode_path, step_i, step_g, images=True, cache=False):

        def load_image_field(episode_path, step_i, step_g):

            color_dir = os.path.join(episode_path, 'color')
            depth_dir = os.path.join(episode_path, 'depth')

            obs = []
            for step in [step_i, step_g]:
                # pdb.set_trace()
                substep_imgs = []
                for substep in range(MAX_SUBSTEPS):
                    f_check = f'S{step}-U{substep}.png'
                    exists = os.path.exists(os.path.join(color_dir, f_check))
                    if not exists:
                        break

                    f = f'S{step}-U{substep}.png'
                    depth = cv2.imread(os.path.join(depth_dir, f), cv2.IMREAD_UNCHANGED)
                    depth = depth / self.depth_scale
                    color = cv2.imread(os.path.join(color_dir, f))
                    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                    depth = depth[..., None]
                    substep_imgs.append(np.concatenate((color, depth), axis=-1))

                obs.append(substep_imgs)
            return obs

        def load_field(episode_path):

            name = episode_path[episode_path.find(FOLDER_PREFIX):]
            if cache:
                if name in self._cache:
                    if 'label' in self._cache[name]:
                        return self._cache[name]['label']
                else:
                    self._cache[name] = {}

            fname = 'label_info.pkl'
            data = pickle.load(open(os.path.join(episode_path, fname), 'rb'))
            if cache:
                self._cache[name]['label'] = data
            return data

        # Get filename and random seed used to initialize episode.
        seed = None

        episode_data = load_field(episode_path)
        # print('whole episode data', episode_data)
        num_steps = len(episode_data)

        step_i = (step_i + num_steps)%num_steps
        step_g = (step_g + num_steps)%num_steps

        p0s, p0_thetas, p1s, p1_thetas = episode_data[step_i]
        p0s_g, p0_thetas_g, p1s_g, p1_thetas_g = episode_data[step_g]
        # if p0s is None:
        #     print('episode path:', episode_path)
        #     print(step_i, step_g)
        # print('sample', p0s, p0_thetas, p1s, p1_thetas)
        # print('goal', p0s_g, p0_thetas_g, p1s_g, p1_thetas_g)

        imgs, imgs_g = load_image_field(episode_path, step_i, step_g)

        ## None is for language instructions
        i = (imgs, (p0s, p0_thetas), (p1s, p1_thetas), None)
        g = (imgs_g, (p0s_g, p0_thetas_g), (p1s_g, p1_thetas_g), None)

        return i, g

    def get_image(self, obs, cam_config=None):
        """Stack color and height images image."""
        #
        # if cam_config is None:
        #     cam_config = self.cam_config

        # Get color and height maps from RGB-D images.
        cmap, hmap = utils.get_fused_heightmap(
            obs, cam_config, self.bounds, self.pix_size)

        kernel = np.ones((2,2), np.uint8)
        cmap = cv2.dilate(cmap, kernel, iterations=1)
        hmap = cv2.dilate(hmap, kernel, iterations=1)

        img = np.concatenate((cmap,
                              # hmap[Ellipsis, None],
                              # hmap[Ellipsis, None],
                              hmap[Ellipsis, None]), axis=2)
        assert img.shape == self.in_shape, img.shape
        return img

    def get_image_wrapper(self, obs, pairings):
        """Stack color and height images image."""
        images = []

        for higher_substeps in range(len(pairings)):
            # print('Info:', obs[substep]['configs'][0]['position'])
            cam_configs = []
            substep_colors = []
            substep_depths = []
            for lower_substeps in pairings[higher_substeps]:
                # pdb.set_trace()
                # print("inside wrapper: ", higher_substeps, lower_substeps)
                substep_colors.extend(obs[lower_substeps]['image']['color'])
                substep_depths.extend(obs[lower_substeps]['image']['depth'])
                # for idx in self.cam_idx:
                config = obs[lower_substeps]['configs'][0]
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
            images.append(img)
        return images

    def transform_pick_place(self, act, obs):
        pick_pose= act['pose0']
        place_pose = act['pose1']
        center = [*act['center'], 0.2]
        X_W_pick = utils.get_transformation_matrix(pick_pose)
        X_W_place = utils.get_transformation_matrix(place_pose)

        acts = []
        p0s, p0_thetas, p1s, p1_thetas, centers = [], [], [], [], []
        for i in range(len(IMAGE_PAIRINGS)):
            substep_obs = obs[IMAGE_PAIRINGS[i][0]]
            # if self.img_frame == 'fp':
            X_WL = utils.get_transformation_matrix(substep_obs['bot_pose'])
            X_LW = np.linalg.inv(X_WL)
            X_L_pick = X_LW @ X_W_pick
            X_L_place = X_LW @ X_W_place
            p0_xyz, p0_xyzw = utils.get_pose_from_transformation(X_L_pick)
            p1_xyz, p1_xyzw = utils.get_pose_from_transformation(X_L_place)
                # center_xyz = (X_LW[:3,:3] @ np.array(center).reshape((3,1)))[:, 0] \
                #                 + X_LW[:3, 3]
            #
            # else:
            #     p0_xyz, p0_xyzw = pick_pose
            #     p1_xyz, p1_xyzw = place_pose
                # center_xyz = center

            # print(f'    Substep: {i}, pick: {p0_xyz}, place: {p1_xyz}')
            p0 = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
            p0_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p0_xyzw)[2])
            p1 = utils.xyz_to_pix(p1_xyz, self.bounds, self.pix_size)
            p1_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p1_xyzw)[2])
            p1_theta = p1_theta - p0_theta
            p0_theta = 0
            # center = utils.xyz_to_pix(center_xyz, self.bounds, self.pix_size)

            p0s.append(p0); p0_thetas.append(p0_theta)
            p1s.append(p1); p1_thetas.append(p1_theta)
            # centers.append(center)

        return p0s, p0_thetas, p1s, p1_thetas #, centers

    def preprocess_sample(self, input_sample):

        def get_crops(img, pivot):
            margin = self.pad_size

            padded_image = torch.nn.ZeroPad2d(margin)(img)
            pv = np.array(pivot) + margin
            # new_img = img.permute(2, 0, 1)
            new_img = padded_image[None, ...]
            crop = new_img.repeat(self.n_rotations, 1, 1, 1)
            crop = self.rotator(crop, pivot=pv)

            crop = torch.cat(crop, dim=0)
            dims = (pv - margin), (pv + margin)
            crop = crop[:,:,dims[0][0]:dims[1][0], dims[0][1]:dims[1][1]]
            # crop = crop.permute(0, 2, 3, 1)
            return crop

        def within_image_bounds(p):
            h, w = p
            H, W = self.in_shape[:2]
            if (h>0) and (w>0) and (h<H) and (w<W):
                return True
            return False

        img, (p0, p0_theta), (p1, p1_theta), _ = input_sample
        for substep in range(len(img)):
            img[substep][:,:,:3] = img[substep][:,:,:3]/255.0
            img[substep] = torch.tensor(img[substep]).float().permute(2, 0, 1)

        # img[2] = img[2][None, ...] # dimension to make the orientation correlation easier
        # img[3] = img[3][None, ...]

        # print(f'length: {len(img), len(p0)}')
        crops = get_crops(img[1], p0[1])
        img_dims = self.in_shape[:2]
        labels = torch.zeros((4, *img_dims), dtype=torch.uint8)

        for i in range(2):
            p = p0[i]
            if within_image_bounds(p):
                r, c = p
            else:
                print('Warning: Pick point not in the image.')
                print(f'Image shape: {img[i].shape}')
                plt.imsave('/home/gridsan/meenalp/cliport-locobot/no_pick.jpeg',  img[i].permute(1,2,0)[:,:,:3].numpy())
                assert False
                r = max(min(p[0], img_dims[0]), 0)
                c = max(min(p[1], img_dims[1]), 0)
                # labels[i, r, c] = 1.0
            sigma = 1.0
            for row in range(r-1, r+2):
                for col in range(c-1, c+2):
                    if not within_image_bounds((row, col)):
                        continue
                    d_sq = ((row-r)**2 + (col-c)**2)/sigma**2
                    labels[i, row, col] = np.exp(-d_sq)
            labels[i,:,:] = labels[i,:,:]/torch.sum(labels[i,:,:])

        for i in range(2, 4):
            p = p1[i]
            if within_image_bounds(p):
                r, c = p
            else:
                print('Warning: Place point not in the image.')
                print(f'Image shape: {img[i].shape}')

                plt.imsave('/home/gridsan/meenalp/cliport-locobot/no_place.jpeg',  img[i].permute(1,2,0)[:,:,:3].numpy())
                assert False
                r = max(min(p[0], img_dims[0]), 0)
                c = max(min(p[1], img_dims[1]), 0)
                # labels[i, r, c] = 1.0
            sigma = 1.0
            for row in range(r-3, r+4):
                for col in range(c-3, c+4):
                    if not within_image_bounds((row, col)):
                        continue
                    d_sq = ((row-r)**2 + (col-c)**2)/sigma**2
                    labels[i, row, col] = np.exp(-d_sq)
            labels[i,:,:] = labels[i,:,:]/torch.sum(labels[i,:,:])

        # img = img.permute(2, 0, 1)
        return img, labels, crops, (p0, p0_theta, p1, p1_theta)

    def preprocess_goal(self, input_goal):
        img, _, _, _ = input_goal
        for substep in range(len(img)):
            img[substep][:,:,:3] = img[substep][:,:,:3]/255.0
            img[substep] = torch.tensor(img[substep]).float().permute(2, 0, 1)

        return img

    def __len__(self):
        return len(self.idx_to_episode_step)

    def __getitem__(self, idx):
        episode_id, step_id = self.idx_to_episode_step[idx]
        episode_path_full = self.episode_paths[episode_id]
        episode_path = episode_path_full[:episode_path_full.find('label')]
        # print('Episode path:', episode_path)

        # Is the task sequential like stack-block-pyramid-seq?
        is_sequential_task = '-seq' in self._path.split("/")[-1]

        # Return random observation action pair (and goal) from episode.
        step_i = step_id
        step_g = step_i+1 if is_sequential_task else -1

        sample, goal = self.load(episode_path, step_i, step_g,
                                      self.images, self.cache)

        return self.preprocess_sample(sample), self.preprocess_goal(goal)


class RavensMultiTaskDataset(RavensDataset):

    MULTI_TASKS = {
        # all tasks
        'multi-all': {
            'train': [
                'align-box-corner',
                'assembling-kits',
                'block-insertion',
                'manipulating-rope',
                'packing-boxes',
                'palletizing-boxes',
                'place-red-in-green',
                'stack-block-pyramid',
                'sweeping-piles',
                'towers-of-hanoi',
                'align-rope',
                'assembling-kits-seq-unseen-colors',
                'packing-boxes-pairs-unseen-colors',
                'packing-shapes',
                'packing-unseen-google-objects-seq',
                'packing-unseen-google-objects-group',
                'put-block-in-bowl-unseen-colors',
                'stack-block-pyramid-seq-unseen-colors',
                'separating-piles-unseen-colors',
                'towers-of-hanoi-seq-unseen-colors',
            ],
            'val': [
                'align-box-corner',
                'assembling-kits',
                'block-insertion',
                'manipulating-rope',
                'packing-boxes',
                'palletizing-boxes',
                'place-red-in-green',
                'stack-block-pyramid',
                'sweeping-piles',
                'towers-of-hanoi',
                'align-rope',
                'assembling-kits-seq-seen-colors',
                'assembling-kits-seq-unseen-colors',
                'packing-boxes-pairs-seen-colors',
                'packing-boxes-pairs-unseen-colors',
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'packing-unseen-google-objects-seq',
                'packing-seen-google-objects-group',
                'packing-unseen-google-objects-group',
                'put-block-in-bowl-seen-colors',
                'put-block-in-bowl-unseen-colors',
                'stack-block-pyramid-seq-seen-colors',
                'stack-block-pyramid-seq-unseen-colors',
                'separating-piles-seen-colors',
                'separating-piles-unseen-colors',
                'towers-of-hanoi-seq-seen-colors',
                'towers-of-hanoi-seq-unseen-colors',
            ],
            'test': [
                'align-box-corner',
                'assembling-kits',
                'block-insertion',
                'manipulating-rope',
                'packing-boxes',
                'palletizing-boxes',
                'place-red-in-green',
                'stack-block-pyramid',
                'sweeping-piles',
                'towers-of-hanoi',
                'align-rope',
                'assembling-kits-seq-seen-colors',
                'assembling-kits-seq-unseen-colors',
                'packing-boxes-pairs-seen-colors',
                'packing-boxes-pairs-unseen-colors',
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'packing-unseen-google-objects-seq',
                'packing-seen-google-objects-group',
                'packing-unseen-google-objects-group',
                'put-block-in-bowl-seen-colors',
                'put-block-in-bowl-unseen-colors',
                'stack-block-pyramid-seq-seen-colors',
                'stack-block-pyramid-seq-unseen-colors',
                'separating-piles-seen-colors',
                'separating-piles-unseen-colors',
                'towers-of-hanoi-seq-seen-colors',
                'towers-of-hanoi-seq-unseen-colors',
            ],
        },

        # demo-conditioned tasks
        'multi-demo-conditioned': {
            'train': [
                'align-box-corner',
                'assembling-kits',
                'block-insertion',
                'manipulating-rope',
                'packing-boxes',
                'palletizing-boxes',
                'place-red-in-green',
                'stack-block-pyramid',
                'sweeping-piles',
                'towers-of-hanoi',
            ],
            'val': [
                'align-box-corner',
                'assembling-kits',
                'block-insertion',
                'manipulating-rope',
                'packing-boxes',
                'palletizing-boxes',
                'place-red-in-green',
                'stack-block-pyramid',
                'sweeping-piles',
                'towers-of-hanoi',
            ],
            'test': [
                'align-box-corner',
                'assembling-kits',
                'block-insertion',
                'manipulating-rope',
                'packing-boxes',
                'palletizing-boxes',
                'place-red-in-green',
                'stack-block-pyramid',
                'sweeping-piles',
                'towers-of-hanoi',
            ],
        },

        # goal-conditioned tasks
        'multi-language-conditioned': {
            'train': [
                'align-rope',
                'assembling-kits-seq-unseen-colors',
                # unseen here refers to training only seen splits to be consitent with single-task setting
                'packing-boxes-pairs-unseen-colors',
                'packing-shapes',
                'packing-unseen-google-objects-seq',
                'packing-unseen-google-objects-group',
                'put-block-in-bowl-unseen-colors',
                'stack-block-pyramid-seq-unseen-colors',
                'separating-piles-unseen-colors',
                'towers-of-hanoi-seq-unseen-colors',
            ],
            'val': [
                'align-rope',
                'assembling-kits-seq-seen-colors',
                'assembling-kits-seq-unseen-colors',
                'packing-boxes-pairs-seen-colors',
                'packing-boxes-pairs-unseen-colors',
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'packing-unseen-google-objects-seq',
                'packing-seen-google-objects-group',
                'packing-unseen-google-objects-group',
                'put-block-in-bowl-seen-colors',
                'put-block-in-bowl-unseen-colors',
                'stack-block-pyramid-seq-seen-colors',
                'stack-block-pyramid-seq-unseen-colors',
                'separating-piles-seen-colors',
                'separating-piles-unseen-colors',
                'towers-of-hanoi-seq-seen-colors',
                'towers-of-hanoi-seq-unseen-colors',
            ],
            'test': [
                'align-rope',
                'assembling-kits-seq-seen-colors',
                'assembling-kits-seq-unseen-colors',
                'packing-boxes-pairs-seen-colors',
                'packing-boxes-pairs-unseen-colors',
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'packing-unseen-google-objects-seq',
                'packing-seen-google-objects-group',
                'packing-unseen-google-objects-group',
                'put-block-in-bowl-seen-colors',
                'put-block-in-bowl-unseen-colors',
                'stack-block-pyramid-seq-seen-colors',
                'stack-block-pyramid-seq-unseen-colors',
                'separating-piles-seen-colors',
                'separating-piles-unseen-colors',
                'towers-of-hanoi-seq-seen-colors',
                'towers-of-hanoi-seq-unseen-colors',
            ],
        },


        ##### multi-attr tasks
        'multi-attr-align-rope': {
            'train': [
                'assembling-kits-seq-full',
                'packing-boxes-pairs-full',
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'packing-seen-google-objects-group',
                'put-block-in-bowl-full',
                'stack-block-pyramid-seq-full',
                'separating-piles-full',
                'towers-of-hanoi-seq-full',
            ],
            'val': [
                'align-rope',
            ],
            'test': [
                'align-rope',
            ],
            'attr_train_task': None,
        },

        'multi-attr-packing-shapes': {
            'train': [
                'align-rope',
                'assembling-kits-seq-full',
                'packing-boxes-pairs-full',
                'packing-seen-google-objects-seq',
                'packing-seen-google-objects-group',
                'put-block-in-bowl-full',
                'stack-block-pyramid-seq-full',
                'separating-piles-full',
                'towers-of-hanoi-seq-full',
            ],
            'val': [
                'packing-shapes',
            ],
            'test': [
                'packing-shapes',
            ],
            'attr_train_task': None,
        },

        'multi-attr-assembling-kits-seq-unseen-colors': {
            'train': [
                'align-rope',
                'assembling-kits-seq-seen-colors', # seen only
                'packing-boxes-pairs-full',
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'packing-seen-google-objects-group',
                'put-block-in-bowl-full',
                'stack-block-pyramid-seq-full',
                'separating-piles-full',
                'towers-of-hanoi-seq-full',
            ],
            'val': [
                'assembling-kits-seq-unseen-colors',
            ],
            'test': [
                'assembling-kits-seq-unseen-colors',
            ],
            'attr_train_task': 'assembling-kits-seq-seen-colors',
        },

        'multi-attr-packing-boxes-pairs-unseen-colors': {
            'train': [
                'align-rope',
                'assembling-kits-seq-full',
                'packing-boxes-pairs-seen-colors', # seen only
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'packing-seen-google-objects-group',
                'put-block-in-bowl-full',
                'stack-block-pyramid-seq-full',
                'separating-piles-full',
                'towers-of-hanoi-seq-full',
            ],
            'val': [
                'packing-boxes-pairs-unseen-colors',
            ],
            'test': [
                'packing-boxes-pairs-unseen-colors',
            ],
            'attr_train_task': 'packing-boxes-pairs-seen-colors',
        },

        'multi-attr-packing-unseen-google-objects-seq': {
            'train': [
                'align-rope',
                'assembling-kits-seq-full',
                'packing-boxes-pairs-full',
                'packing-shapes',
                'packing-seen-google-objects-group',
                'put-block-in-bowl-full',
                'stack-block-pyramid-seq-full',
                'separating-piles-full',
                'towers-of-hanoi-seq-full',
            ],
            'val': [
                'packing-unseen-google-objects-seq',
            ],
            'test': [
                'packing-unseen-google-objects-seq',
            ],
            'attr_train_task': 'packing-seen-google-objects-group',
        },

        'multi-attr-packing-unseen-google-objects-group': {
            'train': [
                'align-rope',
                'assembling-kits-seq-full',
                'packing-boxes-pairs-full',
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'put-block-in-bowl-full',
                'stack-block-pyramid-seq-full',
                'separating-piles-full',
                'towers-of-hanoi-seq-full',
            ],
            'val': [
                'packing-unseen-google-objects-group',
            ],
            'test': [
                'packing-unseen-google-objects-group',
            ],
            'attr_train_task': 'packing-seen-google-objects-seq',
        },

        'multi-attr-put-block-in-bowl-unseen-colors': {
            'train': [
                'align-rope',
                'assembling-kits-seq-full',
                'packing-boxes-pairs-full',
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'packing-seen-google-objects-group',
                'put-block-in-bowl-seen-colors', # seen only
                'stack-block-pyramid-seq-full',
                'separating-piles-full',
                'towers-of-hanoi-seq-full',
            ],
            'val': [
                'put-block-in-bowl-unseen-colors',
            ],
            'test': [
                'put-block-in-bowl-unseen-colors',
            ],
            'attr_train_task': 'put-block-in-bowl-seen-colors',
        },

        'multi-attr-stack-block-pyramid-seq-unseen-colors': {
            'train': [
                'align-rope',
                'assembling-kits-seq-full',
                'packing-boxes-pairs-full',
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'packing-seen-google-objects-group',
                'put-block-in-bowl-full',
                'stack-block-pyramid-seq-seen-colors', # seen only
                'separating-piles-full',
                'towers-of-hanoi-seq-full',
            ],
            'val': [
                'stack-block-pyramid-seq-unseen-colors',
            ],
            'test': [
                'stack-block-pyramid-seq-unseen-colors',
            ],
            'attr_train_task': 'stack-block-pyramid-seq-seen-colors',
        },

        'multi-attr-separating-piles-unseen-colors': {
            'train': [
                'align-rope',
                'assembling-kits-seq-full',
                'packing-boxes-pairs-full',
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'packing-seen-google-objects-group',
                'put-block-in-bowl-full',
                'stack-block-pyramid-seq-full',
                'separating-piles-seen-colors', # seen only
                'towers-of-hanoi-seq-full',
            ],
            'val': [
                'separating-piles-unseen-colors',
            ],
            'test': [
                'separating-piles-unseen-colors',
            ],
            'attr_train_task': 'separating-piles-seen-colors',
        },

        'multi-attr-towers-of-hanoi-seq-unseen-colors': {
            'train': [
                'align-rope',
                'assembling-kits-seq-full',
                'packing-boxes-pairs-full',
                'packing-shapes',
                'packing-seen-google-objects-seq',
                'packing-seen-google-objects-group',
                'put-block-in-bowl-full',
                'stack-block-pyramid-seq-full',
                'separating-piles-full',
                'towers-of-hanoi-seq-seen-colors', # seen only
            ],
            'val': [
                'towers-of-hanoi-seq-unseen-colors',
            ],
            'test': [
                'towers-of-hanoi-seq-unseen-colors',
            ],
            'attr_train_task': 'towers-of-hanoi-seq-seen-colors',
        },

    }
    def __init__(self, path, cfg, group='multi-all',
                 mode='train', n_demos=100, augment=False):
        """A multi-task dataset."""
        self.root_path = path
        self.mode = mode
        self.tasks = self.MULTI_TASKS[group][mode]
        self.attr_train_task = self.MULTI_TASKS[group]['attr_train_task'] if 'attr_train_task' in self.MULTI_TASKS[group] else None

        self.cfg = cfg
        self.sample_set = {}
        self.max_seed = -1
        self.n_episodes = 0
        self.images = self.cfg['dataset']['images']
        self.cache = self.cfg['dataset']['cache']
        self.n_demos = n_demos
        self.augment = augment

        self.aug_theta_sigma = self.cfg['dataset']['augment']['theta_sigma'] if 'augment' in self.cfg['dataset'] else 60
        # legacy code issue: theta_sigma was newly added
        self.pix_size = 0.003125
        self.in_shape = (320, 160, 6)
        self.cam_config = cameras.RealSenseD415.CONFIG
        # self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
        self.bounds = np.array([[0.25, 1.25], [-0.5, 0.5], [0, 0.28]])

        self.n_episodes = {}
        episodes = {}

        for task in self.tasks:
            task_path = os.path.join(self.root_path, f'{task}-{mode}')
            action_path = os.path.join(task_path, 'action')
            n_episodes = 0
            if os.path.exists(action_path):
                for fname in sorted(os.listdir(action_path)):
                    if '.pkl' in fname:
                        n_episodes += 1
            self.n_episodes[task] = n_episodes

            if n_episodes == 0:
                raise Exception(f"{task}-{mode} has 0 episodes. Remove it from the list in dataset.py")

            # Select random episode depending on the size of the dataset.
            episodes[task] = np.random.choice(range(n_episodes), min(self.n_demos, n_episodes), False)

        if self.n_demos > 0:
            self.images = self.cfg['dataset']['images']
            self.cache = False # TODO(mohit): fix caching for multi-task dataset
            self.set(episodes)

        self._path = None
        self._task = None

    def __len__(self):
        # Average number of episodes across all tasks
        total_episodes = 0
        for _, episode_ids in self.sample_set.items():
            total_episodes += len(episode_ids)
        avg_episodes = total_episodes // len(self.sample_set)
        return avg_episodes

    def __getitem__(self, idx):
        # Choose random task.
        self._task = np.random.choice(self.tasks)
        self._path = os.path.join(self.root_path, f'{self._task}')

        # Choose random episode.
        if len(self.sample_set[self._task]) > 0:
            episode_id = np.random.choice(self.sample_set[self._task])
        else:
            episode_id = np.random.choice(range(self.n_episodes[self._task]))
        episode, _ = self.load(episode_id, self.images, self.cache)

        # Is the task sequential like stack-block-pyramid-seq?
        is_sequential_task = '-seq' in self._path.split("/")[-1]

        # Return observation action pair (and goal) from episode.
        if len(episode) > 1:
            i = np.random.choice(range(len(episode)-1))
            g = i+1 if is_sequential_task else -1
            sample, goal = episode[i], episode[g]
        else:
            sample, goal = episode[0], episode[0]

        # Process sample
        sample = self.process_sample(sample, augment=self.augment)
        goal = self.process_goal(goal, perturb_params=sample['perturb_params'])

        return sample, goal

    def add(self, seed, episode):
        raise Exception("Adding tasks not supported with multi-task dataset")

    def load(self, episode_id, images=True, cache=False):
        if self.attr_train_task is None or self.mode in ['val', 'test']:
            self._task = np.random.choice(self.tasks)
        else:
            all_other_tasks = list(self.tasks)
            all_other_tasks.remove(self.attr_train_task)
            all_tasks = [self.attr_train_task] + all_other_tasks # add seen task in the front

            # 50% chance of sampling the main seen task and 50% chance of sampling any other seen-unseen task
            mult_attr_seen_sample_prob = 0.5
            sampling_probs = [(1-mult_attr_seen_sample_prob) / (len(all_tasks)-1)] * len(all_tasks)
            sampling_probs[0] = mult_attr_seen_sample_prob

            self._task = np.random.choice(all_tasks, p=sampling_probs)

        self._path = os.path.join(self.root_path, f'{self._task}-{self.mode}')
        return super().load(episode_id, images, cache)

    def get_curr_task(self):
        return self._task
