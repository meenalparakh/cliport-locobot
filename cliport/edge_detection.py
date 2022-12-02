import cv2
import numpy as np

import os
import torch
from pathlib import Path
from cliport.dataset import RavensDataset, RavensMultiTaskDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import hydra


def get_edge_and_gradient(image, canny=False, threshold1=100, threshold2=200):
    # image is in bgr format.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('/Users/meenalp/Desktop/grayscale.jpeg', gray)

    # Gaussian Blurring
    blur = cv2.GaussianBlur(gray,(5,5),0)
    # cv2.imwrite('/Users/meenalp/Desktop/blur.jpeg', blur)


    # Apply Sobelx in high output datatype 'float32'
    # and then converting back to 8-bit to prevent overflow
    sobelx_64 = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    absx_64 = np.absolute(sobelx_64)
    sobelx_8u1 = absx_64/absx_64.max()*255
    sobelx_8u = np.uint8(sobelx_8u1)

    # Similarly for Sobely
    sobely_64 = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    absy_64 = np.absolute(sobely_64)
    sobely_8u1 = absy_64/absy_64.max()*255
    sobely_8u = np.uint8(sobely_8u1)

    # From gradients calculate the magnitude and changing
    # it to 8-bit (Optional)
    mag = np.hypot(sobelx_8u, sobely_8u)
    mag = mag/mag.max()*255
    mag = np.uint8(mag)

    theta = np.arctan2(sobely_64, sobelx_64)
    angle = np.rad2deg(theta)
    theta = np.uint8((theta/np.max(theta))*255)

    # if canny:
    out = cv2.Canny(image, 150, 250, L2gradient=True)
    # else:
    #     M, N = mag.shape
    #     Non_max = np.zeros((M,N), dtype= np.uint8)
    #
    #     for i in range(1,M-1):
    #         for j in range(1,N-1):
    #            # Horizontal 0
    #             if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180) or (-22.5 <= angle[i,j] < 0) or (-180 <= angle[i,j] < -157.5):
    #                 b = mag[i, j+1]
    #                 c = mag[i, j-1]
    #             # Diagonal 45
    #             elif (22.5 <= angle[i,j] < 67.5) or (-157.5 <= angle[i,j] < -112.5):
    #                 b = mag[i+1, j+1]
    #                 c = mag[i-1, j-1]
    #             # Vertical 90
    #             elif (67.5 <= angle[i,j] < 112.5) or (-112.5 <= angle[i,j] < -67.5):
    #                 b = mag[i+1, j]
    #                 c = mag[i-1, j]
    #             # Diagonal 135
    #             elif (112.5 <= angle[i,j] < 157.5) or (-67.5 <= angle[i,j] < -22.5):
    #                 b = mag[i+1, j-1]
    #                 c = mag[i-1, j+1]
    #
    #             # Non-max Suppression
    #             if (mag[i,j] >= b) and (mag[i,j] >= c):
    #                 Non_max[i,j] = mag[i,j]
    #             else:
    #                 Non_max[i,j] = 0
    #
    # # cv2.imwrite('/Users/meenalp/Desktop/non_max.jpeg', Non_max)
    #
    #     highThreshold = threshold2
    #     lowThreshold = threshold1
    #
    #     M, N = Non_max.shape
    #     out = np.zeros((M,N), dtype= np.uint8)
    #
    #     # If edge intensity is greater than 'High' it is a sure-edge
    #     # below 'low' threshold, it is a sure non-edge
    #     strong_i, strong_j = np.where(Non_max >= highThreshold)
    #     zeros_i, zeros_j = np.where(Non_max < lowThreshold)
    #
    #     # weak edges
    #     weak_i, weak_j = np.where((Non_max <= highThreshold) & (Non_max >= lowThreshold))
    #
    #     # Set same intensity value for all edge pixels
    #     out[strong_i, strong_j] = 255
    #     out[zeros_i, zeros_j ] = 0
    #     out[weak_i, weak_j] = 75
    #     M, N = out.shape
    #     for i in range(1, M-1):
    #         for j in range(1, N-1):
    #             if (out[i,j] == 75):
    #                 if 255 in [out[i+1, j-1],out[i+1, j],out[i+1, j+1],out[i, j-1],out[i, j+1],out[i-1, j-1],out[i-1, j],out[i-1, j+1]]:
    #                     out[i, j] = 255
    #                 else:
    #                     out[i, j] = 0

    # stop=False

    return out, sobelx_64, sobely_64
# cv2.imwrite('/Users/meenalp/Desktop/edge_theta.jpeg', theta)
#
# cv2.imwrite('/Users/meenalp/Desktop/edge_arrow.jpeg', image)

def save_batch_images(batch, batch_idx, epoch):
    sample, goal = batch
    imgs = sample[0]
    labels = sample[1]
    for i in range(len(imgs[0])):
        for substep in [0]:
            img = imgs[substep][i]
            label = labels[i][substep]
            img = img.permute(1, 2, 0)
            color = np.rint(img[:,:,:3].numpy()*255).astype(np.uint8)
            depth = img[:,:,3].numpy()

            p = np.unravel_index(torch.argmax(label), label.shape)
            print(f'Data:{i}, substep: {substep}, label: {p}')
            assert (torch.sum(label) > 0.999) and (torch.sum(label) < 1.001)
            height, width = color.shape[:2]

            # cv2.imwrite(f'/home/gridsan/meenalp/cliport-locobot/images/image_{epoch}_{batch_idx}_{i}_{substep}.jpeg', color)

            # image = cv2.imread('/Users/meenalp/Desktop/explore1_0_0_color.jpeg')
            image = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            out, sobelx_64, sobely_64 = get_edge_and_gradient(image, canny=True)
            cv2.imwrite(f'/Users/meenalp/Desktop/edge_{epoch}_{batch_idx}_{i}_{substep}.jpeg', out)

            height, width = out.shape

            point_nearby = []
            for row in range(0, height, 2):
                for col in range(0, width, 2):
                    edge_mag = (out[row, col] > 250)
                    distance = np.linalg.norm([row, col] - np.array(p))
                    if edge_mag and (distance < 100) and (depth[row, col] < 0.20) and (distance > 20):
                        start_point = (col, row)
                        dx = sobelx_64[row, col]
                        dy = sobely_64[row, col]
                        grad = -np.array([dx, dy])
                        direction = (20*grad/np.linalg.norm(grad)).astype(int)
                        end_point = [col + direction[0], row + direction[1]]
                        final_end_point = (max(0, min(end_point[0], width)),
                                            max(0, min(end_point[1], height)))

                        # image = cv2.arrowedLine(image, start_point, final_end_point,
                        #                             (255, 255, 0), 1, tipLength = 0.5)
                        point_nearby.append((distance, start_point, final_end_point))


            point_nearby.sort(key=lambda x: x[0])
            top_k_points = point_nearby[:2]
            for point in top_k_points:
                image = cv2.arrowedLine(image, point[1], point[2],
                                            (255, 255, 0), 1, tipLength = 0.5)
                image = cv2.arrowedLine(image, point[2], (p[1], p[0]),
                                            (0, 0, 255), 1, tipLength = 0.2)

            d0 = max(0, p[1] - 10), max(0, p[0] - 10)
            d0_ = min(width, p[1] + 10), min(height, p[0] + 10)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image, d0, d0_, (0, 255, 255), 1)
            # cv2.rectangle(image, (100,110), (120, 130), (0, 255, 255), 3)
            cv2.imwrite(f'/Users/meenalp/Desktop/edge_arrow_{epoch}_{batch_idx}_{i}_{substep}.jpeg', image)


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

    train_loader = DataLoader(train_ds, batch_size=32,
                                num_workers=1,
                                shuffle=False)

    # dir_path = '/Users/meenalp/Desktop/cliport-locobot/images'
    # if not os.path.exists(dir_path):
    #     os.makedirs(dir_path)

    save_labelled_images = True
    if save_labelled_images:
        for epoch in range(cfg['train']['max_epochs']):
            for batch_idx, batch in enumerate(train_loader):
                print(f'Batch idx: {batch_idx}')
                save_batch_images(batch, batch_idx, epoch)


if __name__ == '__main__':
    main()
