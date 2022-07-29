import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from cliport.tasks import cameras
from cliport.utils import utils
from cliport.models.resnet import IdentityBlock, ConvBlock
from cliport.dataset import BOUNDS, PIXEL_SIZE, IMG_SHAPE


def get_renset_layers(input_dim, output_dim, batchnorm):
    layers = nn.Sequential(
        # conv1
        nn.Conv2d(input_dim, 64, stride=1, kernel_size=3, padding=1),
        nn.BatchNorm2d(64) if batchnorm else nn.Identity(),
        nn.ReLU(True),

        # fcn
        ConvBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=batchnorm),
        IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=batchnorm),

        ConvBlock(64, [128, 128, 128], kernel_size=3, stride=2, batchnorm=batchnorm),
        IdentityBlock(128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=batchnorm),

        ConvBlock(128, [256, 256, 256], kernel_size=3, stride=2, batchnorm=batchnorm),
        IdentityBlock(256, [256, 256, 256], kernel_size=3, stride=1, batchnorm=batchnorm),

        ConvBlock(256, [512, 512, 512], kernel_size=3, stride=2, batchnorm=batchnorm),
        IdentityBlock(512, [512, 512, 512], kernel_size=3, stride=1, batchnorm=batchnorm),

        # head
        ConvBlock(512, [256, 256, 256], kernel_size=3, stride=1, batchnorm=batchnorm),
        IdentityBlock(256, [256, 256, 256], kernel_size=3, stride=1, batchnorm=batchnorm),
        nn.UpsamplingBilinear2d(scale_factor=2),

        ConvBlock(256, [128, 128, 128], kernel_size=3, stride=1, batchnorm=batchnorm),
        IdentityBlock(128, [128, 128, 128], kernel_size=3, stride=1, batchnorm=batchnorm),
        nn.UpsamplingBilinear2d(scale_factor=2),

        ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=batchnorm),
        IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=batchnorm),
        nn.UpsamplingBilinear2d(scale_factor=2),

        # conv2
        ConvBlock(64, [16, 16, output_dim], kernel_size=3, stride=1,
                  final_relu=False, batchnorm=batchnorm),
        IdentityBlock(output_dim, [16, 16, output_dim], kernel_size=3, stride=1,
                      final_relu=False, batchnorm=batchnorm),
    )
    return layers

    def forward(self, x):
        out = self.layers(x)
        return out

class LocobotTransporterAgent(LightningModule):
    def __init__(self, name, cfg): #lr, weight_decay, weights = None):
        super().__init__()

        self.name = name
        self.cfg = cfg
        self.name = name
        self.task = cfg['train']['task']

        self.pix_size = PIXEL_SIZE
        self.in_shape = IMG_SHAPE

        self.cam_config = cameras.RealSenseD415.CONFIG[1:2]
        self.bounds = BOUNDS

        batchnorm = self.cfg['train']['batchnorm']
        self.attention_layers = get_renset_layers(4, 1, batchnorm)
        self.transport_layers = get_renset_layers(4, 1, batchnorm)

        self._optimizers = torch.optim.Adam(self.parameters(),
                                             lr=self.cfg['train']['lr'])

    def attn_forward(self, inp_img):
        return self.attention_layers(inp_img)

    def transport_forward(self, inp_img):
        return self.transport_layers(inp_img)

    def configure_optimizers(self):
        return [self._optimizers]

    def cross_entropy_with_logits(self, pred, labels, reduction='mean'):
        # Lucas found that both sum and mean work equally well
        batch_size = pred.shape[0]
        pred = pred.reshape((batch_size, -1))
#         print("prediction shape after flattening:", pred.shape)
        labels = labels.reshape((batch_size, -1))
        log_prob = F.log_softmax(pred, -1)
        x = -(labels * log_prob)
        return x.mean()

    def attn_training_step(self, img, label):
        out = self.attn_forward(img)
        loss = self.cross_entropy_with_logits(out, label)
        return loss

    def transport_training_step(self, img, crop, label, theta):
        out = self.transport_forward(img)
        loss = self.cross_entropy_with_logits(out, label)
        return loss

    def training_step(self, batch, batch_idx):
        sample, _ = batch
        imgs, labels, crops, values = sample
        place_thetas = values[-1]
        # Get training losses.
        # step = self.total_steps + 1
        assert(len(imgs) == 4)
        # if optimizer_idx == 0:
        explore_img_1 = imgs[0]
        explore_img_1_label = labels[:, 0]
        pick_img = imgs[1]
        pick_img_label = labels[:, 1]

        loss0 = self.attn_training_step(explore_img_1, explore_img_1_label)
        loss1 = self.attn_training_step(pick_img, pick_img_label)

        explore_img_2 = imgs[2]
        explore_img_2_label = labels[:, 2]
        place_img = imgs[3]
        place_img_label = labels[:, 3]
        loss2 = self.transport_training_step(explore_img_2, crops,
                                        explore_img_2_label, place_thetas[2])
        loss3 = self.transport_training_step(place_img, crops,
                                        place_img_label, place_thetas[3])

        total_loss = loss0 + loss1 + loss2 + loss3
        self.log('train_loss', total_loss)
#         print(f'Train batch loss: {total_loss}')
        return total_loss

    def validation_step(self, batch, batch_idx):
        # return 1.0
        sample, _ = batch
        imgs, labels, crops, values = sample
        place_thetas = values[-1]

        assert(len(imgs) == 4)
        explore_img_1 = imgs[0]
        explore_img_1_label = labels[:, 0]
        pick_img = imgs[1]
        pick_img_label = labels[:, 1]

        loss0 = self.attn_training_step(explore_img_1, explore_img_1_label)
        loss1 = self.attn_training_step(pick_img, pick_img_label)

        explore_img_2 = imgs[2]
        explore_img_2_label = labels[:, 2]
        place_img = imgs[3]
        place_img_label = labels[:, 3]
        loss2 = self.transport_training_step(explore_img_2, crops,
                                        explore_img_2_label, place_thetas[2])
        loss3 = self.transport_training_step(place_img, crops,
                                        place_img_label, place_thetas[3])

        total_loss = loss0 + loss1 + loss2 + loss3
        self.log('val_loss', total_loss)
#         print(f'Val batch loss: {total_loss}')
        return total_loss
