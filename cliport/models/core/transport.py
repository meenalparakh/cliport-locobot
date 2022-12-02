import numpy as np
import cliport.models as models
from cliport.utils import utils

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transport(nn.Module):

    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        """Transport (a.k.a Place) module."""
        super().__init__()

        self.iters = 0
        self.stream_fcn = stream_fcn
        self.n_rotations = n_rotations
        self.crop_size = crop_size  # crop size must be N*16 (e.g. 96)
        self.preprocess = preprocess
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']

        self.pad_size = int(self.crop_size / 2)
        self.padding = np.zeros((3, 2), dtype=int)
        self.padding[:2, :] = self.pad_size

        in_shape = np.array(in_shape)
        in_shape = tuple(in_shape)
        self.in_shape = in_shape

        # Crop before network (default from Transporters CoRL 2020).
        self.kernel_shape = (self.crop_size, self.crop_size, self.in_shape[2])

        if not hasattr(self, 'output_dim'):
            self.output_dim = 3
        if not hasattr(self, 'kernel_dim'):
            self.kernel_dim = 3

        self.rotator = utils.ImageRotator(self.n_rotations)

        self._build_nets()

    def _build_nets(self):
        stream_one_fcn, _ = self.stream_fcn
        model = models.names[stream_one_fcn]
        self.key_resnet = model(self.in_shape, self.output_dim, self.cfg, self.device)
        self.query_resnet = model(self.kernel_shape, self.kernel_dim, self.cfg, self.device)
        print(f"Transport FCN: {stream_one_fcn}")

    def correlate(self, in0, in1, softmax):
        """Correlate two input tensors."""
        output = F.conv2d(in0, in1, padding=(self.pad_size, self.pad_size))
        output = F.interpolate(output, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
        output = output[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
        if softmax:
            output_shape = output.shape
            output = output.reshape((1, np.prod(output.shape)))
            output = F.softmax(output, dim=-1)
            output = output.reshape(output_shape[1:])
        return output

    def transport(self, in_tensor, crop):
        logits = self.key_resnet(in_tensor)
        kernel = self.query_resnet(crop)
        return logits, kernel

    def forward(self, inp_img, p, softmax=True):
        """Forward pass."""
        img_unprocessed = np.pad(inp_img, self.padding, mode='constant')
        input_data = img_unprocessed
        in_shape = (1,) + input_data.shape
        input_data = input_data.reshape(in_shape) # [B W H D]
        in_tensor = torch.from_numpy(input_data).to(dtype=torch.float, device=self.device)

        # Rotation pivot.
        pv = np.array([p[0], p[1]]) + self.pad_size

        # Crop before network (default from Transporters CoRL 2020).
        hcrop = self.pad_size
        in_tensor = in_tensor.permute(0, 3, 1, 2) # [B D W H]

        crop = in_tensor.repeat(self.n_rotations, 1, 1, 1)
        crop = self.rotator(crop, pivot=pv)
        crop = torch.cat(crop, dim=0)
        crop = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]

        logits, kernel = self.transport(in_tensor, crop)

        # TODO(Mohit): Crop after network. Broken for now.
        # in_tensor = in_tensor.permute(0, 3, 1, 2)
        # logits, crop = self.transport(in_tensor)
        # crop = crop.repeat(self.n_rotations, 1, 1, 1)
        # crop = self.rotator(crop, pivot=pv)
        # crop = torch.cat(crop, dim=0)

        # kernel = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]
        # kernel = crop[:, :, p[0]:(p[0] + self.crop_size), p[1]:(p[1] + self.crop_size)]

        return self.correlate(logits, kernel, softmax)

class LocobotTransport(nn.Module):

    def __init__(self, stream_fcn, in_shape,
                    n_rotations, crop_size,
                    preprocess, cfg, device):
        """Transport (a.k.a Place) module."""
        super().__init__()

        self.iters = 0
        self.stream_fcn = stream_fcn
        self.n_rotations = n_rotations
        self.crop_size = crop_size  # crop size must be N*16 (e.g. 96)
        self.preprocess = preprocess
        self.cfg = cfg
        self.device = device
        self.batchnorm = self.cfg['train']['batchnorm']

        in_shape = np.array(in_shape)
        in_shape = tuple(in_shape)
        self.in_shape = in_shape

        self.crop_size = crop_size
        self.pad_size = int(self.crop_size / 2)
        self.padding = np.zeros((3, 2), dtype=int)
        self.padding[:2, :] = self.pad_size

        # Crop before network (default from Transporters CoRL 2020).
        self.kernel_shape = (self.crop_size, self.crop_size, self.in_shape[2])
        self.output_dim = 1
        self.kernel_dim = 1
        # if not hasattr(self, 'output_dim'):
        #     self.output_dim = 3
        # if not hasattr(self, 'kernel_dim'):
        #     self.kernel_dim = 3

        self._build_nets()

    def _build_nets(self):
        stream_one_fcn, _ = self.stream_fcn
        model = models.names[stream_one_fcn]
        self.key_resnet = model(self.in_shape, self.output_dim, self.cfg,
                                self.device, self.preprocess)
        self.query_resnet = model(self.kernel_shape, self.kernel_dim, self.cfg,
                                    self.device, self.preprocess)
        print(f"Transport FCN: {stream_one_fcn}")

    def correlate(self, in0, in1):
        """Correlate two input tensors."""
        output = F.conv2d(in0, in1, padding=(self.pad_size, self.pad_size))
        output = F.interpolate(output, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
        output = output[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]
        return output

    def transport(self, in_tensor, crop):
        logits = self.key_resnet(in_tensor)
        # print('inside transport, img output shape', logits.shape)
        # print('inside transport, crop shape', crop.shape)
        batch_size, num_rotations, num_channels, height, width = crop.shape
        crop = crop.reshape((batch_size*num_rotations, num_channels, height, width))
        kernel = self.query_resnet(crop)
        # print('inside transport kernel shape,', kernel.shape)
        kernel = kernel.reshape((batch_size, num_rotations, -1, height, width))
        print('inside transport kernel shape after,', kernel.shape)
        return logits, kernel

    def forward(self, inp_img, crop, softmax=True):
        """Forward pass."""
        # padded_image shape: (B, C, H+crop_size, W+crop_size)
        print(f'In transport.py, shape of image: {inp_img.shape}')
        # margin = crop.shape[-1]//2
        # padded_image = torch.nn.ZeroPad2d(margin)(inp_img)
        # padded_image = padded_image[:, None, ...]
        # print(f'In transport.py, shape of padded image: {padded_image.shape}')

        # logits, kernel = self.transport(inp_img, crop)
        # correlations = []
        # for rot_idx in range(kernel.shape[1]):
        #     output = self.correlate(logits, kernel[:, rot_idx, ...])
        #     correlations.append(output)
        #     print(f'Correlation output shape: {output.shape}')

        # pdb.set_trace()
        # print()
        # correlations = torch.cat()
        #
        # if softmax:
        #     output_shape = output.shape
        #     output = output.reshape((1, np.prod(output.shape)))
        #     output = F.softmax(output, dim=-1)
        #     output = output.reshape(output_shape[1:])

        batch_size, num_channels, height, width = inp_img.shape
        output = self.key_resnet(inp_img)
        # output = output.reshape((batch_size, -1))
        # output = F.softmax(output, dim = -1)
        # output = output.reshape((batch_size, height, width))
        print(f'Inside tranport output shape: {output.shape}')
        return output
