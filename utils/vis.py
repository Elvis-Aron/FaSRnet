# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torchvision
import cv2
import torch

from datasets.process import get_max_preds
from utils.utils_image import read_image, save_image

# f = True
f = True


def tensor_rgb2gbr(batch_image):
    # 颜色通道转换
    if f:
        RGB_batch_tensor = batch_image.split(1, dim=1)
        batch_image = torch.cat([RGB_batch_tensor[2], RGB_batch_tensor[1], RGB_batch_tensor[0]], dim=1)

    return batch_image


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2
                                 , normalize=True
                                 ):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    # 将图片生成网格
    if normalize:
        batch_image = batch_image.clone()
        min_ = float(batch_image.min())
        max_ = float(batch_image.max())

        batch_image.add_(-min_).div_(max_ - min_ + 1e-5)

    batch_image = tensor_rgb2gbr(batch_image)

    grid = torchvision.utils.make_grid(batch_image,
                                       nrow,  # 每一行显示的图像数.
                                       padding,  # 图片间 的间隔
                                       )
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    # print(nmaps)
    xmaps = min(nrow, nmaps)  # 列数目
    ymaps = int(math.ceil(float(nmaps) / xmaps))  # 行数目
    # 宽 高
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]  # 热度图
            joints_vis = batch_joints_vis[k]  # 可视情况

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    # 根据给定的圆心和半径等画圆
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    save_image(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_image = tensor_rgb2gbr(batch_image)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)
    # 网格图像
    grid_image = np.zeros((batch_size * heatmap_height,
                           (num_joints + 1) * heatmap_width,
                           3),
                          dtype=np.uint8)
    # 热度图坐标 和 预测值
    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        # 图像 逆 归一化
        image = batch_image[i].mul(255) \
            .clamp(0, 255) \
            .byte() \
            .permute(1, 2, 0) \
            .cpu().numpy()
        # 热度图
        heatmaps = batch_heatmaps[i].mul(255) \
            .clamp(0, 255) \
            .byte() \
            .cpu().numpy()
        # 改变图像大小
        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            # 在缩放图上 画出关键点
            cv2.circle(resized_image,  # 画出坐标点
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]  # 热度图
            # 热度图转换为颜色图
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            # 重叠后的图像
            masked_image = colored_heatmap * 0.7 + resized_image * 0.3
            # 根据给定的圆心和半径等画圆
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j + 1)
            width_end = heatmap_width * (j + 2)
            # 在网格上添加图像
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3
        # 开头放上 原图加识别的 关键点
        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    save_image(file_name, grid_image)


def save_debug_images(config, input, meta, target, joints_pred, output,
                      prefix):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input[1], meta['joints'], meta['joints_vis'],
            '{}/{}.jpg'.format(prefix[1], prefix[0])
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input[1], joints_pred, meta['joints_vis'],
            '{}/{}.jpg'.format(prefix[2], prefix[0])
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input[1], target,
            '{}/{}.jpg'.format(prefix[3], prefix[0])
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input[1], output,
            '{}/{}.jpg'.format(prefix[4], prefix[0])
        )
    if config.DEBUG.SAVE_PCN_IMAGES:
        save_batch_p_c_n_image(input,
                               '{}/{}.jpg'.format(prefix[5], prefix[0])
                               )


def save_debug_images2(config, input, meta, target, joints_pred, output,
                      prefix):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input, meta['joints'], meta['joints_vis'],
            '{}/{}.jpg'.format(prefix[1], prefix[0])
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'],
            '{}/{}.jpg'.format(prefix[2], prefix[0])
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target,
            '{}/{}.jpg'.format(prefix[3], prefix[0])
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, output,
            '{}/{}.jpg'.format(prefix[4], prefix[0])
        )
    # if config.DEBUG.SAVE_PCN_IMAGES:
    #     save_batch_p_c_n_image(input,
    #                            '{}/{}.jpg'.format(prefix[5], prefix[0])
    #                            )


# 同时显示3个热度图
def save_debug_images3(config, input, meta, target, joints_pred, output,
                      prefix):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input[1], meta['joints'], meta['joints_vis'],
            '{}/{}.jpg'.format(prefix[1], prefix[0])
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input[1], joints_pred, meta['joints_vis'],
            '{}/{}.jpg'.format(prefix[2], prefix[0])
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input[1], target,
            '{}/{}.jpg'.format(prefix[3], prefix[0])
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input[1], output[3],
            '{}/{}.jpg'.format(prefix[4], prefix[0])
        )
    if config.DEBUG.SAVE_P_HEATMAPS_PRED:
        save_batch_heatmaps(
            input[0], output[0],
            '{}/{}.jpg'.format(prefix[5], prefix[0])
        )
    if config.DEBUG.SAVE_C_HEATMAPS_PRED:
        save_batch_heatmaps(
            input[1], output[1],
            '{}/{}.jpg'.format(prefix[6], prefix[0])
            )
    if config.DEBUG.SAVE_N_HEATMAPS_PRED:
        save_batch_heatmaps(
            input[2], output[2],
            '{}/{}.jpg'.format(prefix[7], prefix[0])
            )
    if config.DEBUG.SAVE_PCN_HEATMAPS_PRED:
        # 三个热度叠加
        save_batch_heatmaps(
            input[1], (output[0]+output[1]+output[2]),
            '{}/{}.jpg'.format(prefix[8], prefix[0])
            )

    if config.DEBUG.SAVE_PCN_IMAGES:
        save_batch_p_c_n_image(input,
                               '{}/{}.jpg'.format(prefix[9], prefix[0])
                               )



# 保存三帧相邻的图像
def save_batch_p_c_n_image(batch_image,
                           file_name, normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    # 将图片生成网格  将三帧图像连接

    temp_list = []
    for batch_tensor in batch_image:
        if normalize:
            batch_tensor = batch_tensor.clone()
            min = float(batch_tensor.min())
            max = float(batch_tensor.max())
            batch_tensor.add_(-min).div_(max - min + 1e-5)

        # 将颜色通道由rgb转换为gbr
        RGB_batch_tensor = batch_tensor.split(1, dim=1)
        temp_list.append(torch.cat([RGB_batch_tensor[2], RGB_batch_tensor[1], RGB_batch_tensor[0]], dim=1))

    nrow = batch_image[0].size()[0]  # 多少列
    grid = torchvision.utils.make_grid(torch.cat(temp_list, dim=0),
                                       nrow,  # 每一行显示的图像数.
                                       2,  # 图片间 的间隔
                                       )
    #
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()

    save_image(file_name, ndarr)
