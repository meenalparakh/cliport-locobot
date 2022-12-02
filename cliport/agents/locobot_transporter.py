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


# class LinearNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(192*192, 32),
#             nn.ReLU(),
#             nn.Linear(32, 192*192)
#             )
        
#     def forward(self, img):
#         batch_size = img.shape[0]
#         img = img[:,0,:,:].reshape((batch_size, -1))
#         out = self.layers(img)
#         out = out.reshape((batch_size, 192, 192))
#         return out

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

#     def forward(self, x):
#         out = self.layers(x)
#         return out

class LocobotTransporterAgent(LightningModule):
    def __init__(self, cfg): #lr, weight_decay, weights = None):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.name = cfg['name']
        self.cfg = cfg
        self.task = cfg['train']['task']

        self.pix_size = PIXEL_SIZE
        self.in_shape = IMG_SHAPE

        self.cam_config = cameras.RealSenseD415.CONFIG[1:2]
        self.bounds = BOUNDS

        batchnorm = self.cfg['train']['batchnorm']
        self.attention_layers = get_renset_layers(4, 1, batchnorm)
        self.transport_layers = get_renset_layers(4, 1, batchnorm)

        self.attn_optimizer = torch.optim.Adam(self.attention_layers.parameters(),
                                             lr=self.cfg['train']['lr'])
        self.transport_optimizer = torch.optim.Adam(self.transport_layers.parameters(),
                                             lr=self.cfg['train']['lr'])
        
        self.attn_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.attn_optimizer, gamma=0.9)
        self.transport_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.transport_optimizer, gamma=0.9)
        
        self._optimizers = [self.attn_optimizer, self.transport_optimizer]
        self._schedulers = [self.attn_scheduler, self.transport_scheduler]

    def attn_forward(self, inp_img):
        return self.attention_layers(inp_img)

    def transport_forward(self, inp_img):
        return self.transport_layers(inp_img)

    def configure_optimizers(self):
        return self._optimizers #, self._schedulers[:1]

    def cross_entropy_with_logits(self, pred, labels):
    
        # Lucas found that both sum and mean work equally well
        batch_size = pred.shape[0]
        
#         for i in range(batch_size):
#             p_true = np.unravel_index(torch.argmax(labels[i]).detach().cpu(), labels[i].shape)            
#             p = np.unravel_index(torch.argmax(pred[i][0]).detach().cpu(), labels[i].shape)
#             print('pixel true, pred:', p_true, p)
            
        pred = pred.reshape((batch_size, -1))
        labels = labels.reshape((batch_size, -1))
        
        assert(int(labels.sum()) == batch_size)
        log_prob = F.log_softmax(pred, dim=1)
        x = -(labels * log_prob)
        return x.sum()/batch_size

    def attn_training_step(self, img, label):
#         print('attn training', img.shape)
        out = self.attn_forward(img)
        loss = self.cross_entropy_with_logits(out, label)
        return loss

    def transport_training_step(self, img, crop, label, theta):
        out = self.transport_forward(img)
        loss = self.cross_entropy_with_logits(out, label)
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        sample, _ = batch
        imgs, labels, crops, values = sample
        place_thetas = values[-1]
        
        # Get training losses.
        # step = self.total_steps + 1
        assert(len(imgs) == 4)
        if optimizer_idx == 0:
            explore_img_1 = imgs[0]
            explore_img_1_label = labels[:, 0]
            pick_img = imgs[1]
            pick_img_label = labels[:, 1]

            loss0 = self.attn_training_step(explore_img_1, explore_img_1_label)
            loss1 = self.attn_training_step(pick_img, pick_img_label)

            attn_loss = loss0 + loss1
            self.log('attn_loss', attn_loss)
            return attn_loss
        
        else:
            explore_img_2 = imgs[2]
            explore_img_2_label = labels[:, 2]
            place_img = imgs[3]
            place_img_label = labels[:, 3]
            loss2 = self.transport_training_step(explore_img_2, crops,
                                            explore_img_2_label, place_thetas[2])
            loss3 = self.transport_training_step(place_img, crops,
                                            place_img_label, place_thetas[3])

            transport_loss = loss2 + loss3
            self.log('transport_loss', transport_loss)
            return transport_loss

    def validation_step(self, batch, batch_idx): 
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

        attn_loss = loss0 + loss1
        self.log('attn_val_loss', attn_loss)
        
        explore_img_2 = imgs[2]
        explore_img_2_label = labels[:, 2]
        place_img = imgs[3]
        place_img_label = labels[:, 3]
        loss2 = self.transport_training_step(explore_img_2, crops,
                                        explore_img_2_label, place_thetas[2])
        loss3 = self.transport_training_step(place_img, crops,
                                        place_img_label, place_thetas[3])

        transport_loss = loss2 + loss3
        self.log('transport_val_loss', transport_loss)
            
#         return attn_loss + transport_loss

    def test_step(self, batch, batch_idx):
        sample, _ = batch
        imgs, labels, crops, values = sample
        place_thetas = values[-1]
                
        explore_img_1 = imgs[0]
        explore_img_1_label = labels[:, 0]
        pick_img = imgs[1]
        pick_img_label = labels[:, 1]

        loss0 = self.attn_training_step(explore_img_1, explore_img_1_label)
        loss1 = self.attn_training_step(pick_img, pick_img_label)

        attn_loss = loss0 + loss1
        self.log('attn_test_loss', attn_loss)
        
        explore_img_2 = imgs[2]
        explore_img_2_label = labels[:, 2]
        place_img = imgs[3]
        place_img_label = labels[:, 3]
        loss2 = self.transport_training_step(explore_img_2, crops,
                                        explore_img_2_label, place_thetas[2])
        loss3 = self.transport_training_step(place_img, crops,
                                        place_img_label, place_thetas[3])

        transport_loss = loss2 + loss3
        self.log('transport_test_loss', transport_loss)
#         return attn_loss + transport_loss
