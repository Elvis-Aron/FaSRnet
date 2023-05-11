import torch
from torch import nn as nn
import numpy as np
from torch.nn import functional as F
# 可变形卷积
from DCNv2_latest_torch1_11.dcn_v2 import DCNv2, dcn_v2_conv
# 可视化相似度
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')




'''
将各个关键点的热度图进行对齐
所以应该使用分组卷积

级联时候，应该是对应位置进行级联

'''


class DCN_3(DCNv2):
    """
    将偏移和权重 用分组卷积进行生成
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=1,
            deformable_groups=1,
    ):
        super(DCN_3, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, deformable_groups
        )

        # channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=deformable_groups,
            bias=True,
        )
        self.conv_mask = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=deformable_groups,
            bias=True,
        )

        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()
        self.conv_mask.weight.data.zero_()
        self.conv_mask.bias.data.zero_()

    def forward(self, input1, input2):
        offset = self.conv_offset(input2)
        mask = torch.sigmoid(self.conv_mask(input2))
        return dcn_v2_conv(
            input1,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.deformable_groups,
        )


class DCN_2(DCNv2):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=1,
            deformable_groups=1,
    ):
        super(DCN_2, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, deformable_groups
        )

        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            channels_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input, offset):
        out = self.conv_offset_mask(offset)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return dcn_v2_conv(
            input,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.deformable_groups,
        )

class PCDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.
    特征对齐的模块

    Ref:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
                        输入特征通道数
        deformable_groups (int): Deformable groups. Defaults: 8.
                        可变形卷积分组数
    """

    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCDAlignment, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size

        # 下采样1
        self.nbr_down_1 = nn.Sequential(
            nn.Conv2d(in_channels=num_feat,out_channels=num_feat, kernel_size=(3, 3), stride=2, padding=1,
                    groups=deformable_groups),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=(3, 3), stride=1, padding=1,
                      groups=deformable_groups),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        # 下采样2
        self.nbr_down_2 = nn.Sequential(
            # nn.MaxPool2d(2),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=(3, 3), stride=2, padding=1,
                      groups=deformable_groups),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=(3, 3), stride=1, padding=1,
                      groups=deformable_groups),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        # 下采样1
        self.ref_down_1 = nn.Sequential(
            # cnn.MaxPool2d(2),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=(3, 3), stride=2, padding=1,
                      groups=deformable_groups),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=(3, 3), stride=1, padding=1,
                      groups=deformable_groups),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        # 下采样2
        self.ref_down_2 = nn.Sequential(
            # nn.MaxPool2d(2),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=(3, 3), stride=2, padding=1,
                      groups=deformable_groups),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=(3, 3), stride=1, padding=1,
                      groups=deformable_groups),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        dilation_list = [5, 5, 3, 1]
        # Pyramids  特征金字塔
        for i in range(3, 0, -1):
            level = f'l{i}'
            # 第一层是输入特征两个变一个
            self.offset_conv1[level] = nn.Conv2d(in_channels=num_feat * 2,
                                                 out_channels=num_feat,
                                                 kernel_size=(3, 3),
                                                 stride=1,
                                                 padding=1,
                                                 groups=deformable_groups)

            if i == 3:
                # 如果是最后一层
                self.offset_conv2[level] = nn.Conv2d(in_channels=num_feat,
                                                     out_channels=num_feat,
                                                     kernel_size=(3, 3),
                                                     stride=1,
                                                     padding=1,
                                                     groups=deformable_groups)

            else:
                self.offset_conv2[level] = nn.Conv2d(in_channels=num_feat * 2,
                                                     out_channels=num_feat,
                                                     kernel_size=(3, 3),
                                                     stride=1,
                                                     padding=1,
                                                     groups=deformable_groups)

                self.offset_conv3[level] = nn.Conv2d(in_channels=num_feat,
                                                     out_channels=num_feat,
                                                     kernel_size=(3, 3),
                                                     stride=1,
                                                     padding=1,
                                                     groups=deformable_groups)

            self.dcn_pack[level] = DCN_2(
                num_feat,
                num_feat,
                3,
                stride=1,
                padding=int(3 / 2) * dilation_list[i],
                dilation=dilation_list[i],
                deformable_groups=deformable_groups)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(in_channels=num_feat * 2,
                                                  out_channels=num_feat,
                                                  kernel_size=(3, 3),
                                                  stride=1,
                                                  padding=1,
                                                  groups=deformable_groups)

        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(in_channels=num_feat * 2,
                                          out_channels=num_feat,
                                          kernel_size=(3, 3),
                                          stride=1,
                                          padding=1,
                                          groups=deformable_groups)
        self.cas_offset_conv2 = nn.Conv2d(in_channels=num_feat,
                                          out_channels=num_feat,
                                          kernel_size=(3, 3),
                                          stride=1,
                                          padding=1,
                                          groups=deformable_groups)
        self.cas_dcnpack = DCN_2(
            num_feat,
            num_feat,
            3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=deformable_groups)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat, ref_feat):
        """Align neighboring frame features to the reference frame features.

        Args:
            nbr_feat e shape (b, c, h, w).
                相邻特征
            ref_feat  shape (b, c, h, w).
                当前帧
        Returns:
            Tensor: Aligned features.
        """
        nbr_feat_l1 = self.nbr_down_1(nbr_feat)
        nbr_feat_l2 = self.nbr_down_2(nbr_feat_l1)
        ref_feat_l1 = self.ref_down_1(ref_feat)
        ref_feat_l2 = self.ref_down_2(ref_feat_l1)

        ref_feat_l = [ref_feat, ref_feat_l1, ref_feat_l2]
        nbr_feat_l = [nbr_feat, nbr_feat_l1, nbr_feat_l2]

        # =========Pyramids  金字塔 对齐
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            # offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            # 按对应位置进行连接
            # 2*[B*k*h*w] -> b*2k*h*w
            offset = self.cat_alignment(nbr_feat_l[i - 1],
                                        ref_feat_l[i - 1])
            # b*2k*h*w  -> b*k*h*w
            offset = self.lrelu(self.offset_conv1[level](offset))  # 两个特征变成一个
            if i == 3:  # 如果是最后一层 偏移就是本身
                offset = self.lrelu(self.offset_conv2[level](offset))  # 输出一个特征
            else:
                # 如果是下面两层，先把上采样的偏移连接了
                # 2*[B*k*h*w] -> b*2k*h*w
                offset = self.lrelu(self.offset_conv2[level](self.cat_alignment(offset,
                                                                                upsampled_offset)
                                                             ))
                # b*2k*h*w  -> b*k*h*w
                offset = self.lrelu(self.offset_conv3[level](offset))
            # 计算的偏移 和 邻接帧做 对齐
            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
                # 将特征和上采样的特征 连接
                feat = self.feat_conv[level](self.cat_alignment(feat,
                                                                upsampled_feat
                                                                ))
            if i > 1:
                feat = self.lrelu(feat)  # 特征激活

            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.  对偏移进行上采样，也应该放大偏移的幅度
                # upsampled_offset = self.upsample(offset) * 2  # 偏移上采样
                upsampled_offset = self.upsample(offset) * 2  # 偏移上采样
                upsampled_feat = self.upsample(feat)  # 特征上采样

        # Cascading
        # 将特征和 当前特征  对应连接
        # 2*[B*k*h*w] -> b*2k*h*w
        offset = self.cat_alignment(feat, ref_feat_l[0])
        offset = self.lrelu(
            self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat

    def cat_alignment(self, nbr_f, ref_f):
        """
        改进连接方式，将对应关键点的位置依次连接
        而不是 直接连接在后面 便于分组
        nbr_f 邻近特征
        ref_f 可行的特征
        """
        _, k, _, _ = nbr_f.size()
        # b*k*h*w -> k*[b*1*h*w]
        nbr_f_list, ref_f_list = nbr_f.split(1, dim=1), ref_f.split(1, dim=1)

        # 特征对齐的列表
        cat_alignment_list = []
        for joint_index in range(k):
            # 依次 是各个关键点 邻接特征 和 可行特征  k*[B*2*h*w]
            cat_alignment_list.append(torch.cat([nbr_f_list[joint_index],
                                                 ref_f_list[joint_index]],
                                                dim=1))

        # k*[B*2*h*w] -> b*2k*h*w
        return torch.cat(cat_alignment_list, dim=1)


class PCDAlignment_2(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.
    特征对齐的模块 没有进行下采样的步骤

    Ref:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
                        输入特征通道数
        deformable_groups (int): Deformable groups. Defaults: 8.
                        可变形卷积分组数
    """

    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCDAlignment_2, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size

        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        # Pyramids  特征金字塔
        for i in range(3, 0, -1):
            level = f'l{i}'
            # 第一层是输入特征两个变一个
            self.offset_conv1[level] = nn.Conv2d(in_channels=num_feat * 2,
                                                 out_channels=num_feat,
                                                 kernel_size=(3, 3),
                                                 stride=1,
                                                 padding=1,
                                                 groups=deformable_groups)

            if i == 3:
                # 如果是最后一层
                self.offset_conv2[level] = nn.Conv2d(in_channels=num_feat,
                                                     out_channels=num_feat,
                                                     kernel_size=(3, 3),
                                                     stride=1,
                                                     padding=1,
                                                     groups=deformable_groups)

            else:
                self.offset_conv2[level] = nn.Conv2d(in_channels=num_feat * 2,
                                                     out_channels=num_feat,
                                                     kernel_size=(3, 3),
                                                     stride=1,
                                                     padding=1,
                                                     groups=deformable_groups)

                self.offset_conv3[level] = nn.Conv2d(in_channels=num_feat,
                                                     out_channels=num_feat,
                                                     kernel_size=(3, 3),
                                                     stride=1,
                                                     padding=1,
                                                     groups=deformable_groups)

            self.dcn_pack[level] = DCN_2(
                num_feat,
                num_feat,
                3,
                stride=1,
                padding=1,
                deformable_groups=deformable_groups)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(in_channels=num_feat * 2,
                                                  out_channels=num_feat,
                                                  kernel_size=(3, 3),
                                                  stride=1,
                                                  padding=1,
                                                  groups=deformable_groups)

        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(in_channels=num_feat * 2,
                                          out_channels=num_feat,
                                          kernel_size=(3, 3),
                                          stride=1,
                                          padding=1,
                                          groups=deformable_groups)
        self.cas_offset_conv2 = nn.Conv2d(in_channels=num_feat,
                                          out_channels=num_feat,
                                          kernel_size=(3, 3),
                                          stride=1,
                                          padding=1,
                                          groups=deformable_groups)
        self.cas_dcnpack = DCN_2(
            num_feat,
            num_feat,
            3,
            stride=1,
            padding=1,
            deformable_groups=deformable_groups)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        """Align neighboring frame features to the reference frame features.

        Args:
            nbr_feat e shape (b, c, h, w).
                相邻特征
            ref_feat  shape (b, c, h, w).
                当前帧
        Returns:
            Tensor: Aligned features.
        """

        # =========Pyramids  金字塔 对齐
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            # offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            # 按对应位置进行连接
            # 2*[B*k*h*w] -> b*2k*h*w
            offset = self.cat_alignment(nbr_feat_l[i - 1],
                                        ref_feat_l[i - 1])
            # b*2k*h*w  -> b*k*h*w
            offset = self.lrelu(self.offset_conv1[level](offset))  # 两个特征变成一个
            if i == 3:  # 如果是最后一层 偏移就是偏见
                offset = self.lrelu(self.offset_conv2[level](offset))  # 输出一个特征
            else:
                # 如果是下面两层，先把上采样的偏移连接了
                # 2*[B*k*h*w] -> b*2k*h*w
                offset = self.lrelu(self.offset_conv2[level](self.cat_alignment(offset,
                                                                                upsampled_offset)
                                                             ))
                # b*2k*h*w  -> b*k*h*w
                offset = self.lrelu(self.offset_conv3[level](offset))
            # 计算的偏移 和 邻接帧做 对齐
            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
                # 将特征和上采样的特征 连接
                feat = self.feat_conv[level](self.cat_alignment(feat,
                                                                upsampled_feat
                                                                ))
            if i > 1:
                feat = self.lrelu(feat)  # 特征激活

            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.  对偏移进行上采样，也应该放大偏移的幅度
                upsampled_offset = self.upsample(offset) * 2  # 偏移上采样
                upsampled_feat = self.upsample(feat)  # 特征上采样

        # Cascading
        # 将特征和 当前特征  对应连接
        # 2*[B*k*h*w] -> b*2k*h*w
        offset = self.cat_alignment(feat, ref_feat_l[0])
        offset = self.lrelu(
            self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat

    def cat_alignment(self, nbr_f, ref_f):
        """
        改进连接方式，将对应关键点的位置依次连接
        而不是 直接连接在后面 便于分组
        nbr_f 邻近特征
        ref_f 可行的特征
        """
        _, k, _, _ = nbr_f.size()
        # b*k*h*w -> k*[b*1*h*w]
        nbr_f_list, ref_f_list = nbr_f.split(1, dim=1), ref_f.split(1, dim=1)

        # 特征对齐的列表
        cat_alignment_list = []
        for joint_index in range(k):
            # 依次 是各个关键点 邻接特征 和 可行特征  k*[B*2*h*w]
            cat_alignment_list.append(torch.cat([nbr_f_list[joint_index],
                                                 ref_f_list[joint_index]],
                                                dim=1))

        # k*[B*2*h*w] -> b*2k*h*w
        return torch.cat(cat_alignment_list, dim=1)

# 不进行下采样，使用3个膨胀率不同的分支
class PCDAlignment_3(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.
    特征对齐的模块

    Ref:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
                        输入特征通道数
        deformable_groups (int): Deformable groups. Defaults: 8.
                        可变形卷积分组数
    """

    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCDAlignment_3, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size


        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        dilation_list = [1, 3, 5]
        # Pyramids  特征金字塔
        for i in range(3, 0, -1):
            level = f'l{i}'
            # 第一层是输入特征两个变一个
            self.offset_conv1[level] = nn.Conv2d(in_channels=num_feat * 2,
                                                 out_channels=num_feat,
                                                 kernel_size=(3, 3),
                                                 stride=1,
                                                 padding=1,
                                                 groups=deformable_groups)

            if i == 3:
                # 如果是最后一层
                self.offset_conv2[level] = nn.Conv2d(in_channels=num_feat,
                                                     out_channels=num_feat,
                                                     kernel_size=(3, 3),
                                                     stride=1,
                                                     padding=1,
                                                     groups=deformable_groups)

            else:
                self.offset_conv2[level] = nn.Conv2d(in_channels=num_feat * 2,
                                                     out_channels=num_feat,
                                                     kernel_size=(3, 3),
                                                     stride=1,
                                                     padding=1,
                                                     groups=deformable_groups)

                self.offset_conv3[level] = nn.Conv2d(in_channels=num_feat,
                                                     out_channels=num_feat,
                                                     kernel_size=(3, 3),
                                                     stride=1,
                                                     padding=1,
                                                     groups=deformable_groups)

            self.dcn_pack[level] = DCN_2(
                num_feat,
                num_feat,
                3,
                stride=1,
                padding=int(3 / 2) * dilation_list[i-1],
                dilation=dilation_list[i-1],
                deformable_groups=deformable_groups)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(in_channels=num_feat * 2,
                                                  out_channels=num_feat,
                                                  kernel_size=(3, 3),
                                                  stride=1,
                                                  padding=1,
                                                  groups=deformable_groups)

        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(in_channels=num_feat * 2,
                                          out_channels=num_feat,
                                          kernel_size=(3, 3),
                                          stride=1,
                                          padding=1,
                                          groups=deformable_groups)
        self.cas_offset_conv2 = nn.Conv2d(in_channels=num_feat,
                                          out_channels=num_feat,
                                          kernel_size=(3, 3),
                                          stride=1,
                                          padding=1,
                                          groups=deformable_groups)
        self.cas_dcnpack = DCN_2(
            num_feat,
            num_feat,
            3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=deformable_groups)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat, ref_feat):
        """Align neighboring frame features to the reference frame features.

        Args:
            nbr_feat e shape (b, c, h, w).
                相邻特征
            ref_feat  shape (b, c, h, w).
                当前帧
        Returns:
            Tensor: Aligned features.
        """

        # =========Pyramids  金字塔 对齐
        temp_offset, temp_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            # offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            # 按对应位置进行连接
            # 2*[B*k*h*w] -> b*2k*h*w
            offset = self.cat_alignment(nbr_feat,
                                        ref_feat)
            # b*2k*h*w  -> b*k*h*w
            offset = self.lrelu(self.offset_conv1[level](offset))  # 两个特征变成一个
            if i == 3:  # 如果是最后一层 偏移就是本身
                offset = self.lrelu(self.offset_conv2[level](offset))  # 输出一个特征
            else:
                # 如果是下面两层，先把特征连接
                # 2*[B*k*h*w] -> b*2k*h*w
                offset = self.lrelu(self.offset_conv2[level](self.cat_alignment(offset,
                                                                                temp_offset)
                                                             ))
                # b*2k*h*w  -> b*k*h*w
                offset = self.lrelu(self.offset_conv3[level](offset))
            # 计算的偏移 和 邻接帧做 对齐
            feat = self.dcn_pack[level](nbr_feat, offset)
            if i < 3:
                # 将特征和上采样的特征 连接
                feat = self.feat_conv[level](self.cat_alignment(feat,
                                                                temp_feat
                                                                ))
            if i > 1:
                feat = self.lrelu(feat)  # 特征激活

            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.  对偏移进行上采样，也应该放大偏移的幅度
                # upsampled_offset = self.upsample(offset) * 2  # 偏移上采样
                temp_offset = offset  # 偏移上采样
                temp_feat = feat  # 特征上采样

        # Cascading
        # 将特征和 当前特征  对应连接
        # 2*[B*k*h*w] -> b*2k*h*w
        offset = self.cat_alignment(feat, ref_feat)
        offset = self.lrelu(
            self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat

    def cat_alignment(self, nbr_f, ref_f):
        """
        改进连接方式，将对应关键点的位置依次连接
        而不是 直接连接在后面 便于分组
        nbr_f 邻近特征
        ref_f 可行的特征
        """
        _, k, _, _ = nbr_f.size()
        # b*k*h*w -> k*[b*1*h*w]
        nbr_f_list, ref_f_list = nbr_f.split(1, dim=1), ref_f.split(1, dim=1)

        # 特征对齐的列表
        cat_alignment_list = []
        for joint_index in range(k):
            # 依次 是各个关键点 邻接特征 和 可行特征  k*[B*2*h*w]
            cat_alignment_list.append(torch.cat([nbr_f_list[joint_index],
                                                 ref_f_list[joint_index]],
                                                dim=1))

        # k*[B*2*h*w] -> b*2k*h*w
        return torch.cat(cat_alignment_list, dim=1)

class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module.
        时空注意力(TSA)融合模块
    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
        super(TSAFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, t, c, h, w = aligned_feat.size()
        # temporal attention 卷积提一下特征
        # 中间帧的特征
        embedding_ref = self.temporal_attn1(
            aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        # 所有帧的特征
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]  # 第i帧的特征 # b*c*h*w
            # 相邻特征与 当前特征相乘 计算相似权重
            # 只是得到相同位置与相同位置之间的相似度
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
        # 得到几个图片之间的相关性 并变换形状
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)

        # temp = corr_prob.detach()
        # torch.save(temp.to(torch.device('cpu')), "myTensor.pth")

        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().reshape(b, -1, h, w)  # (b, t*c, h, w)
        # 所有特征乘以概率（相似度）
        aligned_feat = aligned_feat.reshape(b, -1, h, w) * corr_prob

        # fusion 是直接各个特征进行 混合了
        # b*kk*h*w ->  b*kk*h*w
        feat = self.lrelu(self.feat_fusion(aligned_feat))

        # spatial attention
        # 多帧的通道变成一个帧的通道
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        # 特征池化
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        # 下采样，再计算注意力
        attn = self.lrelu(
            self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        # pyramid levels
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(
            self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(
            self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat


if __name__ == '__main__':
    a = PCDAlignment(num_feat=48, deformable_groups=1)

    total = sum([param.nelement() for param in a.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))

    b = TSAFusion(num_feat=17, num_frame=3, center_frame_idx=1)

    total = sum([param.nelement() for param in b.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    input_b = torch.rand((4, 3, 17, 96, 72))
    out_b = b(input_b)
    print(f'out_b.size {out_b.size()}')

    nbr_f = torch.rand((4, 48, 96, 72))
    ref_f = torch.rand((4, 48, 96, 72))
    out = a(nbr_f, ref_f)
    print(f'out.size {out.size()}')

    c = PCDAlignment_3(num_feat=48, deformable_groups=1)
    total = sum([param.nelement() for param in c.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    out = c(nbr_f, ref_f)
    print(f'out.size {out.size()}')

    # f1 = torch.rand((4, 17, 96, 72))
    # f2 = torch.rand((4, 17, 96, 72))
    # f3 = torch.rand((4, 17, 96, 72))
    # f = [f1,f2,f3]
    #
    # aligned_feat = torch.stack(f, dim=1)
    # print(f'aligned_feat.size {aligned_feat.size()}')

    # _, c, _, _ = nbr_f.size()
    # # b*k*h*w -> k*[b*1*h*w]
    # nbr_f_list, ref_f_list = nbr_f.split(1, dim=1), ref_f.split(1, dim=1)
    #
    # # 特征对齐的列表
    # cat_alignment_list = []
    # for joint_index in range(c):
    #     # 依次 是各个关键点 邻接特征 和 可行特征  k*[B*2*h*w]
    #     cat_alignment_list.append(torch.cat([nbr_f_list[joint_index],
    #                                          ref_f_list[joint_index]],
    #                                         dim=1))

    # k*[B*2*h*w] -> b*2k*h*w

    # x1 = conv_l2_2(conv_l2_1(input_))
    # print(f'x1.size {x1.size()}')
    # x2 = conv_l3_2(conv_l3_1(x1))
    # print(f'x1.size {x2.size()}')


def tensor2im(input_image, imtype=np.uint8):
    """"
    Parameters:
        input_image (tensor) --  ÊäÈëµÄtensor£¬Î¬¶ÈÎªCHW£¬×¢ÒâÕâÀïÃ»ÓÐbatch sizeµÄÎ¬¶È
        imtype (type)        --  ×ª»»ºóµÄnumpyµÄÊý¾ÝÀàÐÍ
    """
    mean = [0.485, 0.456, 0.406] # dataLoaderÖÐÉèÖÃµÄmean²ÎÊý£¬ÐèÒª´ÓdataloaderÖÐ¿½±´¹ýÀ´
    std = [0.229, 0.224, 0.225]  # dataLoaderÖÐÉèÖÃµÄstd²ÎÊý£¬ÐèÒª´ÓdataloaderÖÐ¿½±´¹ýÀ´
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor): # Èç¹û´«ÈëµÄÍ¼Æ¬ÀàÐÍÎªtorch.Tensor£¬Ôò¶ÁÈ¡ÆäÊý¾Ý½øÐÐÏÂÃæµÄ´¦Àí
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)): # ·´±ê×¼»¯£¬³ËÒÔ·½²î£¬¼ÓÉÏ¾ùÖµ
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255 #·´ToTensor(),´Ó[0,1]×ªÎª[0,255]
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # ´Ó(channels, height, width)±äÎª(height, width, channels)
    else:  # Èç¹û´«ÈëµÄÊÇnumpyÊý×é,Ôò²»×ö´¦Àí
        image_numpy = input_image
    return image_numpy.astype(imtype)