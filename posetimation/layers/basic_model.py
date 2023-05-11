#!/usr/bin/python
# -*- coding:utf8 -*-

import torch.nn as nn
import torch
from torch.nn import functional as F
# from thirdparty.deform_conv import ModulatedDeformConv
from DCNv2_latest_torch1_11.dcn_v2 import DCNv2 as ModulatedDeformConv

BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=groups)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, groups=groups)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

        self.act_fun = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, groups=groups)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_fun(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act_fun(out)

        return out


class Bottleneck(nn.Module):
    """
    From HRNet
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interpolate = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class ChainOfBasicBlocks(nn.Module):
    def __init__(self, input_channel, ouput_channel, kernel_height, kernel_width, dilation, num_blocks, groups=1):
        super(ChainOfBasicBlocks, self).__init__()
        stride = 1
        downsample = nn.Sequential(nn.Conv2d(input_channel, ouput_channel, kernel_size=1, stride=stride, bias=False, groups=groups),
                                   nn.BatchNorm2d(ouput_channel, momentum=BN_MOMENTUM))
        layers = []
        layers.append(BasicBlock(input_channel, ouput_channel, stride, downsample, groups))

        for i in range(1, num_blocks):
            layers.append(BasicBlock(ouput_channel, ouput_channel, stride, downsample=None, groups=groups))

        # return nn.Sequential(*layers)
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class DeformableCONV(nn.Module):
    def __init__(self, num_joints, k, dilation):
        super(DeformableCONV, self).__init__()

        self.deform_conv = modulated_deform_conv(num_joints, k, k, dilation, num_joints).cuda()

    def forward(self, x, offsets, mask):
        return self.deform_conv(x, offsets, mask)


def modulated_deform_conv(n_channels, kernel_height, kernel_width, deformable_dilation, deformable_groups):
    conv_offset2d = ModulatedDeformConv(
        n_channels,
        n_channels,
        (kernel_height, kernel_width),
        stride=1,
        padding=int(kernel_height / 2) * deformable_dilation,
        dilation=deformable_dilation,
        deformable_groups=deformable_groups
    )
    return conv_offset2d


class dcn_v2(nn.Module):
    """
    可变形卷积v2
    可实现输入输出通道不同
    """
    def __init__(self, in_channels, out_channels, k, dilation, deformable_groups):
        super(dcn_v2, self).__init__()

        self.deform_conv = ModulatedDeformConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(k, k),
            stride=1,
            padding=int(k / 2) * dilation,
            dilation=dilation,
            deformable_groups=deformable_groups).cuda()

    def forward(self, x, offsets, mask):
        return self.deform_conv(x, offsets, mask)


if __name__ == '__main__':
    input = torch.rand((4,17*3,96,72)).cuda()
    dcn_1 = dcn_v2(in_channels=17*3,
                   out_channels=17,
                   k=3,
                   dilation=1,
                   deformable_groups=17   # 注意，这里是组数，而不是几个分为一组
                   )
    # 通道数据 分组数 * 2*kh*hw
    o = torch.rand((4, 17 * 18, 96, 72)).cuda()
    # 通道数据 分组数 * 1*kh*hw
    m = torch.rand((4, 17 * 9, 96, 72)).cuda()
    out = dcn_1(input,o,m)
    print(out.size())
