#!/usr/bin/python
# -*- coding:utf8 -*-


import os
import torch
import torch.nn as nn
import logging
from collections import OrderedDict

from ..base import BaseModel


# from thirdparty.deform_conv import DeformConv, ModulatedDeformConv

from DCNv2_latest_torch1_11.dcn_v2 import DCNv2 as ModulatedDeformConv
from posetimation.layers import DeformableCONV, dcn_v2
from posetimation.layers import CHAIN_RSB_BLOCKS
from ..backbones.hrnet import HRNet
from utils.common import TRAIN_PHASE
from utils.utils_registry import MODEL_REGISTRY
# 他注意力
from posetimation.layers.att_test4 import NONLocalBlock2D
# 金字塔特征对齐模块 时空融合模块
from posetimation.layers.pcda import PCDAlignment_3,TSAFusion


class net2(nn.Module):
    """
    热度图级别的特征 校准
    """
    def __init__(self, cfg, phase):
        super(net2, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.use_rectifier = cfg.MODEL.USE_RECTIFIER
        self.use_margin = cfg.MODEL.USE_MARGIN
        self.use_group = cfg.MODEL.USE_GROUP
        self.deformable_conv_dilations = cfg.MODEL.DEFORMABLE_CONV.DILATION
        self.deformable_aggregation_type = cfg.MODEL.DEFORMABLE_CONV.AGGREGATION_TYPE
        ####


        self.pretrained = cfg.MODEL.PRETRAINED

        k = 3

        prf_inner_ch = cfg.MODEL.PRF_INNER_CH
        prf_basicblock_num = cfg.MODEL.PRF_BASICBLOCK_NUM

        ptm_inner_ch = cfg.MODEL.PTM_INNER_CH
        ptm_basicblock_num = cfg.MODEL.PTM_BASICBLOCK_NUM

        prf_ptm_combine_inner_ch = cfg.MODEL.PRF_PTM_COMBINE_INNER_CH
        prf_ptm_combine_basicblock_num = cfg.MODEL.PRF_PTM_COMBINE_BASICBLOCK_NUM
        hyper_parameters = OrderedDict({
            "k": k,
            "prf_basicblock_num": prf_basicblock_num,
            "prf_inner_ch": prf_inner_ch,
            "ptm_basicblock_num": ptm_basicblock_num,
            "ptm_inner_ch": ptm_inner_ch,
            "prf_ptm_combine_basicblock_num": prf_ptm_combine_basicblock_num,
            "prf_ptm_combine_inner_ch": prf_ptm_combine_inner_ch,
        }
        )
        self.logger.info("###### MODEL {} Hyper Parameters ##########".format(self.__class__.__name__))
        self.logger.info(hyper_parameters)

        # assert self.use_prf and self.use_ptm and self.use_pcn and self.use_margin and self.use_margin and self.use_group

        ####### PRF #######
        diff_temporal_fuse_input_channels = self.num_joints * 4
        self.diff_temporal_fuse = CHAIN_RSB_BLOCKS(diff_temporal_fuse_input_channels, prf_inner_ch, prf_basicblock_num,
                                                   )

        # self.diff_temporal_fuse = ChainOfBasicBlocks(diff_temporal_fuse_input_channels, prf_inner_ch, 1, 1, 2,
        #                                              prf_basicblock_num, groups=self.num_joints)

        ####### PTM #######
        if ptm_basicblock_num > 0:

            self.support_temporal_fuse = CHAIN_RSB_BLOCKS(self.num_joints * 3, ptm_inner_ch, ptm_basicblock_num,
                                                          )

            # self.support_temporal_fuse = ChainOfBasicBlocks(self.num_joints * 3, ptm_inner_ch, 1, 1, 2,
            #                                                 ptm_basicblock_num, groups=self.num_joints)
        else:
            self.support_temporal_fuse = nn.Conv2d(self.num_joints * 3, ptm_inner_ch, kernel_size=3, padding=1,
                                                   groups=self.num_joints)

        prf_ptm_combine_ch = prf_inner_ch + ptm_inner_ch

        self.offset_mask_combine_conv = CHAIN_RSB_BLOCKS(prf_ptm_combine_ch, prf_ptm_combine_inner_ch,
                                                         prf_ptm_combine_basicblock_num)
        # self.offset_mask_combine_conv = ChainOfBasicBlocks(prf_ptm_combine_ch, prf_ptm_combine_inner_ch, 1, 1, 2,
        #                                                    prf_ptm_combine_basicblock_num)

        ####### PCN #######
        self.offsets_list, self.masks_list, self.modulated_deform_conv_list = [], [], []
        for d_index, dilation in enumerate(self.deformable_conv_dilations):
            # offsets
            offset_layers, mask_layers = [], []
            offset_layers.append(self._offset_conv(prf_ptm_combine_inner_ch, k, k, dilation, self.num_joints).cuda())
            mask_layers.append(self._mask_conv(prf_ptm_combine_inner_ch, k, k, dilation, self.num_joints).cuda())
            self.offsets_list.append(nn.Sequential(*offset_layers))
            self.masks_list.append(nn.Sequential(*mask_layers))
            self.modulated_deform_conv_list.append(DeformableCONV(self.num_joints, k, dilation))

        self.offsets_list = nn.ModuleList(self.offsets_list)
        self.masks_list = nn.ModuleList(self.masks_list)
        self.modulated_deform_conv_list = nn.ModuleList(self.modulated_deform_conv_list)

    def _offset_conv(self, nc, kh, kw, dd, dg):
        conv = nn.Conv2d(nc, dg * 2 * kh * kw, kernel_size=(3, 3), stride=(1, 1), dilation=(dd, dd),
                         padding=(1 * dd, 1 * dd), bias=False)
        return conv

    def _mask_conv(self, nc, kh, kw, dd, dg):
        conv = nn.Conv2d(nc, dg * 1 * kh * kw, kernel_size=(3, 3), stride=(1, 1), dilation=(dd, dd),
                         padding=(1 * dd, 1 * dd), bias=False)
        return conv

    def forward(self, current_rough_heatmaps,
                previous_rough_heatmaps,
                next_rough_heatmaps,margin):

        # Difference A and Difference B
        diff_A = current_rough_heatmaps - previous_rough_heatmaps
        diff_B = current_rough_heatmaps - next_rough_heatmaps

        # default use_margin,use_group
        interval = torch.sum(margin, dim=1, keepdim=True)  # interval = n-p
        margin = torch.div(margin.float(), interval.float())  # margin -> (c-p)/(n-p), (n-c)/(n-p)
        prev_weight, next_weight = margin[:, 1], margin[:, 0]  # previous frame weight - (n-c)/(n-p) , next frame weight - (c-p)/(n-p)
        diff_shape = diff_A.shape
        prev_weight = prev_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        next_weight = next_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        extend_shape = [1, 1]  # batch, channel
        extend_shape.extend(list(diff_shape[2:]))
        prev_weight, next_weight = prev_weight.repeat(extend_shape), next_weight.repeat(extend_shape)

        diff_A_list, diff_B_list = diff_A.split(1, dim=1), diff_B.split(1, dim=1)
        temp_diff_fuse_list = []
        for joint_index in range(self.num_joints):
            temp_diff_fuse_list.append(diff_A_list[joint_index])
            temp_diff_fuse_list.append(diff_A_list[joint_index] * prev_weight)
            temp_diff_fuse_list.append(diff_B_list[joint_index])
            temp_diff_fuse_list.append(diff_B_list[joint_index] * next_weight)

        dif_heatmaps = torch.cat(temp_diff_fuse_list, dim=1)
        dif_heatmaps = self.diff_temporal_fuse(dif_heatmaps)

        current_rough_heatmaps_list = current_rough_heatmaps.split(1, dim=1)
        previous_rough_heatmaps_list = previous_rough_heatmaps.split(1, dim=1)
        next_rough_heatmaps_list = next_rough_heatmaps.split(1, dim=1)
        temp_support_fuse_list = []
        for joint_index in range(self.num_joints):
            temp_support_fuse_list.append(current_rough_heatmaps_list[joint_index])
            temp_support_fuse_list.append(previous_rough_heatmaps_list[joint_index] * prev_weight)
            temp_support_fuse_list.append(next_rough_heatmaps_list[joint_index] * next_weight)

        support_heatmaps = torch.cat(temp_support_fuse_list, dim=1)
        support_heatmaps = self.support_temporal_fuse(support_heatmaps).cuda()

        prf_ptm_combine_featuremaps = self.offset_mask_combine_conv(torch.cat([dif_heatmaps, support_heatmaps], dim=1))

        warped_heatmaps_list = []
        for d_index, dilation in enumerate(self.deformable_conv_dilations):
            offsets = self.offsets_list[d_index](prf_ptm_combine_featuremaps)
            masks = self.masks_list[d_index](prf_ptm_combine_featuremaps)

            warped_heatmaps = self.modulated_deform_conv_list[d_index](support_heatmaps, offsets, masks)

            warped_heatmaps_list.append(warped_heatmaps)

        if self.deformable_aggregation_type == "weighted_sum":

            warper_weight = 1 / len(self.deformable_conv_dilations)
            output_heatmaps = warper_weight * warped_heatmaps_list[0]
            for warper_heatmaps in warped_heatmaps_list[1:]:
                output_heatmaps += warper_weight * warper_heatmaps

        else:
            output_heatmaps = self.deformable_aggregation_conv(torch.cat(warped_heatmaps_list, dim=1))
            # elif self.deformable_aggregation_type == "conv":


        return output_heatmaps



@MODEL_REGISTRY.register()
class FaSRnet(BaseModel):

    def __init__(self, cfg, phase, **kwargs):
        super(FaSRnet, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

        self.is_train = True if phase == TRAIN_PHASE else False
        self.cfg = cfg
        self.use_double_loss = cfg.MODEL.USE_DOUBLE_LOSS

        self.freeze_hrnet_weights = cfg.MODEL.FREEZE_HRNET_WEIGHTS
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.use_rectifier = cfg.MODEL.USE_RECTIFIER
        self.deformable_conv_dilations = cfg.MODEL.DEFORMABLE_CONV.DILATION
        self.deformable_aggregation_type = cfg.MODEL.DEFORMABLE_CONV.AGGREGATION_TYPE
        ####
        self.rough_pose_estimation_net = HRNet(cfg, phase)

        self.pretrained = cfg.MODEL.PRETRAINED
        if cfg.MODEL.USE_PCDA:  # 使用帧对齐
            self.pcda_p = PCDAlignment_3(num_feat=cfg.MODEL.USE_F_C, deformable_groups=1)
            self.pcda_n = PCDAlignment_3(num_feat=cfg.MODEL.USE_F_C, deformable_groups=1)

        self.TSA_Fusion = TSAFusion(num_feat=cfg.MODEL.USE_F_C, num_frame=3, center_frame_idx=1)

        self.final_layer = nn.Conv2d(
            in_channels=48,
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=cfg.MODEL.EXTRA.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if cfg.MODEL.EXTRA.FINAL_CONV_KERNEL == 3 else 0
        )

        self.hm_improve = net2(cfg, phase)

    def _offset_conv(self, nc, kh, kw, dd, dg):
        conv = nn.Conv2d(nc, dg * 2 * kh * kw, kernel_size=(3, 3),
                         stride=(1, 1), dilation=(dd, dd), padding=(1 * dd, 1 * dd), bias=False)
        return conv

    def _mask_conv(self, nc, kh, kw, dd, dg):
        conv = nn.Conv2d(nc, dg * 1 * kh * kw, kernel_size=(3, 3),
                         stride=(1, 1), dilation=(dd, dd), padding=(1 * dd, 1 * dd), bias=False)
        return conv

    # def forward(self, x, margin, debug=False, vis=False):
    def forward(self, x, **kwargs):
        num_color_channels = 3
        assert "margin" in kwargs
        margin = kwargs["margin"]
        if not x.is_cuda or not margin.is_cuda:
            x.cuda()
            margin.cuda()

        if not self.use_rectifier:
            target_image = x[:, 0:num_color_channels, :, :]
            rough_x = self.rough_pose_estimation_net(target_image)
            return rough_x

        # current / previous / next

            # current / previous / next
            # 热度图和特征
        rough_heatmaps, rough_feature = self.rough_pose_estimation_net(
                torch.cat(x.split(num_color_channels, dim=1), 0))

        true_batch_size = int(rough_heatmaps.shape[0] / 3)
        # rough heatmaps in sequence frames
        # 热度图
        current_rough_heatmaps, previous_rough_heatmaps, next_rough_heatmaps = rough_heatmaps.split(true_batch_size,
                                                                                                        dim=0)
        # 特征
        current_rough_feature, previous_rough_feature, next_rough_feature = rough_feature.split(true_batch_size,
                                                                                                    dim=0)

        if self.cfg.MODEL.USE_PCDA:  # 使用帧对齐
            previous_rough_feature = self.pcda_p(current_rough_feature, previous_rough_feature)
            next_rough_feature = self.pcda_n(current_rough_feature, next_rough_feature)

        improve_feature = torch.stack([previous_rough_feature, current_rough_feature, next_rough_feature], dim=1)
        improve_feature = self.TSA_Fusion(improve_feature)
        output_heatmaps = self.final_layer(improve_feature)
        output_heatmaps1 = self.hm_improve(output_heatmaps,
                                              previous_rough_heatmaps,
                                              next_rough_heatmaps,
                                              margin)

        if self.use_double_loss:
            # 第一次修正和第二次修正
            #
            if self.cfg.DEBUG.VIS_TENSORBOARD_2:  # 可视化输入图像和热度图

                return [current_rough_heatmaps, previous_rough_heatmaps, next_rough_heatmaps],\
                       output_heatmaps, output_heatmaps1
            elif self.cfg.DEBUG.DEBUG:
                return current_rough_heatmaps, previous_rough_heatmaps, next_rough_heatmaps, output_heatmaps1
            else:
                return output_heatmaps, output_heatmaps1
        else:
            return output_heatmaps1

    def data_preparation(self, rough_heatmaps, margin):
        """
        rough_heatmaps 为三个热度图
        margin 加权信息
        将综合的按关键点个数进行拆分
        """
        # =====================数据分割======================
        true_batch_size = int(rough_heatmaps.shape[0] / 3)
        # rough heatmaps in sequence frames
        current_rough_heatmaps, previous_rough_heatmaps, next_rough_heatmaps = rough_heatmaps.split(true_batch_size,
                                                                                                    dim=0)

        if self.cfg.MODEL.USE_PCDA:  # 使用帧对齐
            previous_rough_heatmaps = self.pcda_p(current_rough_heatmaps, previous_rough_heatmaps)
            next_rough_heatmaps = self.pcda_n(current_rough_heatmaps, next_rough_heatmaps)


        # Difference A and Difference B
        # 差分
        diff_A = current_rough_heatmaps - previous_rough_heatmaps  # B*k*h*w
        diff_B = current_rough_heatmaps - next_rough_heatmaps  # B*k*h*w
        # 乘法


        # ========================权重准备===================================
        # default use_margin,use_group
        # B*1
        interval = torch.sum(margin, dim=1, keepdim=True)  # interval = n-p
        # B*2
        margin = torch.div(margin.float(), interval.float())  # margin -> (c-p)/(n-p), (n-c)/(n-p)
        #  previous frame weight - (n-c)/(n-p) , next frame weight - (c-p)/(n-p)
        prev_weight, next_weight = margin[:, 1], margin[:,0]
        diff_shape = diff_A.shape
        prev_weight = prev_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        next_weight = next_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        extend_shape = [1, 1]  # batch, channel
        extend_shape.extend(list(diff_shape[2:]))
        prev_weight, next_weight = prev_weight.repeat(extend_shape), next_weight.repeat(extend_shape)
        # ==================================================================
        # ==========================差分部分数据准备==========================
        #  B*k*h*w -> k [B*1*h*w]
        diff_A_list, diff_B_list = diff_A.split(1, dim=1), diff_B.split(1, dim=1)

        # mul_A_list, mul_B_list = mul_A.split(1, dim=1), mul_B.split(1, dim=1)
        current_rough_heatmaps_list = current_rough_heatmaps.split(1, dim=1)
        previous_rough_heatmaps_list = previous_rough_heatmaps.split(1, dim=1)
        next_rough_heatmaps_list = next_rough_heatmaps.split(1, dim=1)

        support_fuse_list = []
        diff_fuse_list = []
        for joint_index in range(self.num_joints):
            # 依次 是各个关键点 的差分 和 加权差分  17*[B*4*h*w]
            diff_fuse_list.append(torch.cat([diff_A_list[joint_index],
                                             diff_A_list[joint_index]* prev_weight,
                                             diff_B_list[joint_index],
                                             diff_B_list[joint_index]* next_weight],
                                             dim=1))
            # 原本的关键点热度图 17*[B*3*h*w]
            support_fuse_list.append(torch.cat([current_rough_heatmaps_list[joint_index],
                                                previous_rough_heatmaps_list[joint_index] * prev_weight,
                                                next_rough_heatmaps_list[joint_index] * next_weight],
                                               dim=1))

        return diff_fuse_list, support_fuse_list

    def data_preparation_final(self, warped_heatmaps_list):
        # 将最后的结果按照对应关键点对齐
        temp_out_list = []
        for joint_index in range(self.num_joints):
            temp_out_list.append(torch.cat([temp[joint_index] for temp in warped_heatmaps_list], dim=1)
                            )

        # B*(k*u)*h*w  u位膨胀卷积个数
        return torch.cat(temp_out_list, dim=1)



    def data_preparation_att(self, rough_heatmaps, r_f):
        """
        给前后帧热度图 加权
        rough_heatmaps 为三个热度图
        margin 加权信息
        将综合的按关键点个数进行拆分
        r_f 浅层的特征
        """
        # =====================数据分割======================
        true_batch_size = int(rough_heatmaps.shape[0] / 3)
        # rough heatmaps in sequence frames
        current_rough_heatmaps, previous_rough_heatmaps, next_rough_heatmaps = rough_heatmaps.split(true_batch_size,
                                                                                                    dim=0)
        current_rough_f, previous_rough_f, next_rough_f = r_f.split(true_batch_size,dim=0)
        # 对前后热度图 进行加权
        previous_rough_heatmaps = self.att_1(previous_rough_heatmaps, previous_rough_f)
        next_rough_heatmaps = self.att_2(next_rough_heatmaps, next_rough_f)

        # Difference A and Difference B
        # 差
        diff_A = current_rough_heatmaps - previous_rough_heatmaps  # B*k*h*w
        diff_B = current_rough_heatmaps - next_rough_heatmaps  # B*k*h*w
        # 乘法
        mul_A = current_rough_heatmaps * previous_rough_heatmaps  # B*k*h*w
        mul_B = current_rough_heatmaps * next_rough_heatmaps  # B*k*h*w

        # ========================权重准备===================================
        # # default use_margin,use_group
        # # B*1
        # interval = torch.sum(margin, dim=1, keepdim=True)  # interval = n-p
        # # B*2
        # margin = torch.div(margin.float(), interval.float())  # margin -> (c-p)/(n-p), (n-c)/(n-p)
        # #  previous frame weight - (n-c)/(n-p) , next frame weight - (c-p)/(n-p)
        # prev_weight, next_weight = margin[:, 1], margin[:,0]
        # diff_shape = diff_A.shape
        # prev_weight = prev_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # next_weight = next_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # extend_shape = [1, 1]  # batch, channel
        # extend_shape.extend(list(diff_shape[2:]))
        # prev_weight, next_weight = prev_weight.repeat(extend_shape), next_weight.repeat(extend_shape)
        # ==================================================================
        # ==========================差分部分数据准备==========================
        #  B*k*h*w -> k [B*1*h*w]
        diff_A_list, diff_B_list = diff_A.split(1, dim=1), diff_B.split(1, dim=1)
        mul_A_list, mul_B_list = mul_A.split(1, dim=1), mul_B.split(1, dim=1)
        current_rough_heatmaps_list = current_rough_heatmaps.split(1, dim=1)
        previous_rough_heatmaps_list = previous_rough_heatmaps.split(1, dim=1)
        next_rough_heatmaps_list = next_rough_heatmaps.split(1, dim=1)

        support_fuse_list = []
        diff_fuse_list = []
        for joint_index in range(self.num_joints):
            # 依次 是各个关键点 的差分 和 加权差分  17*[B*4*h*w]
            diff_fuse_list.append(torch.cat([diff_A_list[joint_index],
                                             mul_A_list[joint_index],
                                             diff_B_list[joint_index],
                                             mul_B_list[joint_index]],
                                            dim=1))
            # 原本的关键点热度图 17*[B*3*h*w]
            support_fuse_list.append(torch.cat([current_rough_heatmaps_list[joint_index],
                                                previous_rough_heatmaps_list[joint_index],
                                                next_rough_heatmaps_list[joint_index]],
                                               dim=1))

        return diff_fuse_list, support_fuse_list

    def init_weights(self):
        logger = logging.getLogger(__name__)
        ## init_weights
        rough_pose_estimation_name_set = set()
        for module_name, module in self.named_modules():
            # rough_pose_estimation_net 单独判断一下
            if module_name.split('.')[0] == "rough_pose_estimation_net":
                rough_pose_estimation_name_set.add(module_name)
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.001)
                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, std=0.001)

                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)

            # elif isinstance(module, DeformConv):
            #     filler = torch.zeros([module.weight.size(0), module.weight.size(1), module.weight.size(2), module.weight.size(3)],
            #                          dtype=torch.float32, device=module.weight.device)
            #     for k in range(module.weight.size(0)):
            #         filler[k, k, int(module.weight.size(2) / 2), int(module.weight.size(3) / 2)] = 1.0
            #     module.weight = torch.nn.Parameter(filler)
            #     # module.weight.requires_grad = True

            elif isinstance(module, ModulatedDeformConv):
                filler = torch.zeros([module.weight.size(0), module.weight.size(1), module.weight.size(2), module.weight.size(3)],
                                     dtype=torch.float32, device=module.weight.device)
                for k in range(module.weight.size(0)):
                    filler[k, k, int(module.weight.size(2) / 2), int(module.weight.size(3) / 2)] = 1.0
                module.weight = torch.nn.Parameter(filler)
                # module.weight.requires_grad = True
            else:
                for name, _ in module.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(module.bias, 0)
                    if name in ['weights']:
                        nn.init.normal_(module.weight, std=0.001)

        if os.path.isfile(self.pretrained):
            pretrained_state_dict = torch.load(self.pretrained)
            if 'state_dict' in pretrained_state_dict.keys():
                pretrained_state_dict = pretrained_state_dict['state_dict']
            logger.info('=> loading pretrained model {}'.format(self.pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                        or self.pretrained_layers[0] is '*':
                    layer_name = name.split('.')[0]
                    if layer_name in rough_pose_estimation_name_set:
                        need_init_state_dict[name] = m
                    else:
                        # 为了适应原本hrnet得预训练网络
                        new_layer_name = "rough_pose_estimation_net.{}".format(layer_name)
                        if new_layer_name in rough_pose_estimation_name_set:
                            parameter_name = "rough_pose_estimation_net.{}".format(name)
                            need_init_state_dict[parameter_name] = m
            # TODO pretrained from posewarper not test
            self.load_state_dict(need_init_state_dict, strict=False)
        elif self.pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(self.pretrained))

        # rough_pose_estimation
        if self.freeze_hrnet_weights:
            self.rough_pose_estimation_net.freeze_weight()

    @classmethod
    def get_model_hyper_parameters(cls, args, cfg):
        prf_inner_ch = cfg.MODEL.PRF_INNER_CH
        prf_basicblock_num = cfg.MODEL.PRF_BASICBLOCK_NUM
        ptm_inner_ch = cfg.MODEL.PTM_INNER_CH
        ptm_basicblock_num = cfg.MODEL.PTM_BASICBLOCK_NUM
        prf_ptm_combine_inner_ch = cfg.MODEL.PRF_PTM_COMBINE_INNER_CH
        prf_ptm_combine_basicblock_num = cfg.MODEL.PRF_PTM_COMBINE_BASICBLOCK_NUM
        if "DILATION" in cfg.MODEL.DEFORMABLE_CONV:
            dilation = cfg.MODEL.DEFORMABLE_CONV.DILATION
            dilation_str = ",".join(map(str, dilation))
        else:
            dilation_str = ""
        hyper_parameters_setting = "chPRF_{}_nPRF_{}_chPTM_{}_nPTM_{}_chComb_{}_nComb_{}_D_{}".format(
            prf_inner_ch, prf_basicblock_num, ptm_inner_ch, ptm_basicblock_num, prf_ptm_combine_inner_ch, prf_ptm_combine_basicblock_num,
            dilation_str)

        return hyper_parameters_setting

    @classmethod
    def get_net(cls, cfg, phase, **kwargs):
        model = FaSRnet(cfg, phase, **kwargs)
        return model
