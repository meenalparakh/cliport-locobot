import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from cliport.tasks import cameras
from cliport.utils import utils
# from cliport.models.core.attention import Attention
# from cliport.models.core.transport import Transport
from cliport.models.core.attention import LocobotAttention
from cliport.models.core.transport import LocobotTransport
from cliport.models.streams.two_stream_attention import TwoStreamAttention
from cliport.models.streams.two_stream_transport import TwoStreamTransport

from cliport.models.streams.two_stream_attention import TwoStreamAttentionLat
from cliport.models.streams.two_stream_transport import TwoStreamTransportLat
from cliport.dataset import BOUNDS


class LocobotTransporterAgentPrimary(LightningModule):
    def __init__(self, name, cfg):
        super().__init__()
        utils.set_seed(0)

        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # this is bad for PL :(
        self.name = name
        self.cfg = cfg
        # self.train_ds = train_ds
        # self.test_ds = test_ds

        self.name = name
        self.task = cfg['train']['task']
        self.total_steps = 0
        self.crop_size = 40
        self.n_rotations = 4 #cfg['train']['n_rotations']

        # self.pix_size = 0.003125
        self.pix_size = 0.00625
        # self.in_shape = (320, 160, 6)
        self.in_shape = (160, 160, 6)
        self.cam_config = cameras.RealSenseD415.CONFIG[1:2]
        # self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
        self.bounds = BOUNDS

        self.val_repeats = cfg['train']['val_repeats']
        self.save_steps = cfg['train']['save_steps']

        # self.bce_loss = torch.nn.BCELoss()
        self._build_model()

        self._optimizers = {
            'attn': torch.optim.Adam(self.attention.parameters(), lr=self.cfg['train']['lr']),
            'trans': torch.optim.Adam(self.transport.parameters(), lr=self.cfg['train']['lr'])
        }
        print("Agent: {}, Logging: {}".format(name, cfg['train']['log']))

    def configure_optimizers(self):
        return [self._optimizers['attn'], self._optimizers['trans']]

    def _build_model(self):
        self.attention = None
        self.transport = None
        raise NotImplementedError()

    def cross_entropy_with_logits(self, pred, labels, reduction='mean'):
        # Lucas found that both sum and mean work equally well
        batch_size = pred.shape[0]
        pred = pred.reshape((batch_size, -1))
        labels = labels.reshape((batch_size, -1))
        log_prob = F.log_softmax(pred, -1)
        # print("inside loss, log prob max:", torch.max(log_prob))
        # print("inside loss labels max:", torch.max(labels))
        # print("inside loss labels min:", torch.min(labels))
        x = -(labels * log_prob)

        # print("inside loss x max:", torch.max(x))
        # print("inside loss x min:", torch.min(x))
        return x.sum()
        # if reduction == 'sum':
        #     return x.sum()
        # elif reduction == 'mean':
        #     return x.mean()
        # else:
        #     raise NotImplementedError()

    def forward(self, sample):
        raise NotImplementedError()

    def attn_training_step(self, img, label):
        out = self.attention(img)
        loss = self.cross_entropy_with_logits(out, label)
        return loss, {}

    def transport_training_step(self, img, crop, label, theta):
        out = self.transport(img, crop)
        loss = self.cross_entropy_with_logits(out, label)
        return loss, {}

    def training_step(self, batch, batch_idx, optimizer_idx):
        # self.attention.train()
        # self.transport.train()

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

            loss0, err0 = self.attn_training_step(explore_img_1, explore_img_1_label)
            loss1, err1 = self.attn_training_step(pick_img, pick_img_label)
            loss2, loss3 = 0, 0

        elif optimizer_idx == 1:
            explore_img_2 = imgs[2]
            explore_img_2_label = labels[:, 2]
            place_img = imgs[3]
            place_img_label = labels[:, 3]
            loss2, err2 = self.transport_training_step(explore_img_2, crops,
                                            explore_img_2_label, place_thetas[2])
            loss3, err3 = self.transport_training_step(place_img, crops,
                                            place_img_label, place_thetas[3])
            loss0, loss1 = 0, 0

        # Rotation pivots
        total_loss = loss0 + loss1 + loss2 + loss3
        self.log('tr/attn/loss1', loss0)
        self.log('tr/attn/loss2', loss1)
        self.log('tr/trans/loss3', loss2)
        self.log('tr/trans/loss4', loss3)

        self.log('tr/loss', total_loss)

        print(f'Total_loss: {total_loss}')
        return total_loss

    def validation_step(self, batch, batch_idx):
        return 0


class LocobotTransporterAgent(LocobotTransporterAgentPrimary):

    def __init__(self, name, cfg):
        super().__init__(name, cfg)

    def _build_model(self):
        stream_fcn = 'plain_resnet'
        self.attention = LocobotAttention(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = LocobotTransport(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
