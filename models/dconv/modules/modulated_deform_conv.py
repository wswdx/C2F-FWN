#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import init
from torch.nn.modules.utils import _pair

from functions.modulated_deform_conv_func import ModulatedDeformConvFunction
import time

class ModulatedDeformConv(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True):
        super(ModulatedDeformConv, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels//groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, offset, mask):
        assert 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
            offset.shape[1]
        assert self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
            mask.shape[1]
        return ModulatedDeformConvFunction.apply(input, offset, mask,
                                                   self.weight,
                                                   self.bias,
                                                   self.stride,
                                                   self.padding,
                                                   self.dilation,
                                                   self.groups,
                                                   self.deformable_groups,
                                                   self.im2col_step)

_ModulatedDeformConv = ModulatedDeformConvFunction.apply

def biinter(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)

def get_grid(batchsize, rows, cols, gpu_id=0, dtype=torch.float32):
    hor = torch.linspace(-1.0, 1.0, cols)
    hor.requires_grad = False
    hor = hor.view(1, 1, 1, cols)
    hor = hor.expand(batchsize, 1, rows, cols)
    ver = torch.linspace(-1.0, 1.0, rows)
    ver.requires_grad = False
    ver = ver.view(1, 1, rows, 1)
    ver = ver.expand(batchsize, 1, rows, cols)

    hor.requires_grad = False
    ver.requires_grad = False

    if dtype == torch.float16:
        hor = hor.half()
        ver = ver.half()
    return hor.cuda(gpu_id), ver.cuda(gpu_id)

    t_grid = torch.cat([hor, ver], 1)

class ModulatedDeformConvPack(ModulatedDeformConv):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, lr_mult=0.1):
        super(ModulatedDeformConvPack, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, groups, deformable_groups, im2col_step, bias)

        out_channels = self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset = nn.Conv2d(self.in_channels,
                                          out_channels,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.conv_offset.lr_mult = lr_mult
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, input_list):
        input, input_LO, output_LO = input_list
        strideh = stridew = self.stride[0]
        padh = padw = self.padding[0]
        HK = self.kernel_size[0]
        WK = self.kernel_size[1]
        batchsize, label_nc, HI, WI = input_LO.size()
        out = self.conv_offset(input)
        _, _, HO, WO = out.size()
        o1, o2 = torch.chunk(out, 2, dim=1)
        offset = torch.cat((o1, o2), dim=1)

        sample_location_x_0, sample_location_y_0 = get_grid(batchsize, HO, WO, gpu_id=input.get_device(), dtype=input.dtype)
        sample_location_x_0 = sample_location_x_0 * stridew - padw
        sample_location_y_0 = sample_location_y_0 * strideh - padh

        # here we enable layout-constrained sampling
        for hk in range(HK):
            for wk in range(WK):
                sample_location_x_i = sample_location_x_0 + (wk + offset[:, 2*(hk*WK+wk)+1]) / ((WI - 1.0) / 2.0)
                sample_location_y_i = sample_location_y_0 + (hk + offset[:, 2*(hk*WK+wk)]) / ((HI - 1.0) / 2.0)
                sample_location_i = torch.cat([sample_location_x_i, sample_location_y_i], 1)
                if hk==0 and wk==0:
                    sample_location = sample_location_i
                else:
                    sample_location = torch.cat([sample_location, sample_location_i], 1)

        sample_location = sample_location.permute(0, 2, 3, 1).contiguous().view(-1, HO, WO, 2)
        input_LO = input_LO.repeat(1, WK*HK, 1, 1).view(-1, label_nc, HI, WI)

        sample_LO = torch.nn.functional.grid_sample(input_LO, sample_location, mode='bilinear', padding_mode='border')
        sample_LO = sample_LO * output_LO.repeat(1, WK*HK, 1, 1).view(-1, label_nc, HO, WO)

        mask = torch.sum(sample_LO, dim=1, keepdim=True).view(batchsize, WK*HK, HO, WO)

        output = ModulatedDeformConvFunction.apply(input, offset, mask, 
                                                self.weight, 
                                                self.bias, 
                                                self.stride, 
                                                self.padding, 
                                                self.dilation, 
                                                self.groups,
                                                self.deformable_groups,
                                                self.im2col_step)
        return output

