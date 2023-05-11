#!/usr/bin/python
# -*- coding:utf8 -*-
import time
import torch
import numpy as np
import os
import os.path as osp
import logging
from torchvision.utils import make_grid

from tensorboardX import SummaryWriter
from .evaludate import accuracy, pck_accuracy, pck_accuracy_origin_image
from engine.core import CORE_FUNCTION_REGISTRY, BaseFunction, AverageMeter

# from engine.evaludate import accuracy
from engine.defaults import VAL_PHASE, TEST_PHASE, TRAIN_PHASE
from datasets.process import get_final_preds, get_max_preds
from datasets.transforms import reverse_transforms

from utils.utils_bbox import cs2box
from utils.utils_folder import create_folder
from utils.utils_image_tensor import tensor2im
from utils.vis import save_debug_images, save_debug_images2, save_debug_images3

from tabulate import tabulate



@CORE_FUNCTION_REGISTRY.register()
class CommonFunction(BaseFunction):

    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.output_dir = cfg.OUTPUT_DIR

        if "criterion" in kwargs.keys():
            self.criterion = kwargs["criterion"]
        if "tb_log_dir" in kwargs.keys():
            self.tb_log_dir = kwargs["tb_log_dir"]
        if "writer_dict" in kwargs.keys():
            self.writer_dict = kwargs["writer_dict"]

        self.PE_Name = kwargs.get("PE_Name", "DCPOSE")
        ##
        self.max_iter_num = 0
        self.dataloader_iter = None
        self.tb_writer = None
        self.global_steps = 0
        self.DataSetName = str(self.cfg.DATASET.NAME).upper()

    def train(self, model, epoch, optimizer, dataloader, tb_writer_dict, **kwargs):
        self.tb_writer = tb_writer_dict["writer"]
        self.global_steps = tb_writer_dict["global_steps"]
        logger = logging.getLogger(__name__)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        # switch to train mode
        model.train()

        self.max_iter_num = len(dataloader)
        self.dataloader_iter = iter(dataloader)
        end = time.time()
        logger.info(f'PE_Name :{self.PE_Name}')
        for iter_step in range(self.max_iter_num):
            input_x, input_sup_A, input_sup_B, target_heatmaps, target_heatmaps_weight, meta = next(self.dataloader_iter)
            self._before_train_iter(input_x)

            data_time.update(time.time() - end)

            target_heatmaps = target_heatmaps.cuda(non_blocking=True)
            target_heatmaps_weight = target_heatmaps_weight.cuda(non_blocking=True)
            if self.PE_Name == "HRNET":
                outputs = model(input_x.cuda())
            elif self.PE_Name == "DCPOSE":
                margin_left, margin_right = meta["margin_left"], meta["margin_right"]
                margin = torch.stack([margin_left, margin_right], dim=1).cuda()
                concat_input = torch.cat((input_x, input_sup_A, input_sup_B), 1).cuda()
                outputs = model(concat_input, margin=margin)

            else:
                outputs = model(input_x.cuda())

            if isinstance(outputs, list) or isinstance(outputs, tuple):
                if self.cfg.DEBUG.VIS_TENSORBOARD_2:
                    pred_heatmaps = outputs[1]
                    # 第一个损失是特征层面的，乘以其特征权重
                    loss = self.cfg.LOSS.L1_WEIGHT * self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)
                    for pred_heatmaps in outputs[2:]:
                        # 第二个损失是热度图层面的
                        loss += (1-self.cfg.LOSS.L1_WEIGHT)*self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)
                else:
                    pred_heatmaps = outputs[0]
                    loss = self.cfg.LOSS.L1_WEIGHT *self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)
                    for pred_heatmaps in outputs[1:]:
                        loss += (1-self.cfg.LOSS.L1_WEIGHT)*self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)
            else:
                pred_heatmaps = outputs
                loss = self.criterion(pred_heatmaps, target_heatmaps, target_heatmaps_weight)

            # compute gradient and do update step
            if self.cfg.MODEL.USE_RECTIFIER:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()



            # measure accuracy and record loss
            losses.update(loss.item(), input_x.size(0))

            _, avg_acc, cnt, _ = accuracy(pred_heatmaps.detach().cpu().numpy(), target_heatmaps.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if iter_step % self.cfg.PRINT_FREQ == 0 or iter_step >= self.max_iter_num - 1:

                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(epoch, iter_step, self.max_iter_num, batch_time=batch_time,
                                                                        speed=input_x.size(0) / batch_time.val,
                                                                        data_time=data_time, loss=losses, acc=acc)

                logger.info(msg)

            # For Tensorboard
            self.tb_writer.add_scalar('train_loss', losses.val, self.global_steps)
            self.tb_writer.add_scalar('train_acc', acc.val, self.global_steps)
            self.global_steps += 1

        tb_writer_dict["global_steps"] = self.global_steps

    def eval(self, model, dataloader, tb_writer_dict, **kwargs):

        logger = logging.getLogger(__name__)

        self.tb_writer = tb_writer_dict["writer"]
        self.global_steps = tb_writer_dict["global_steps"]

        batch_time, data_time, losses, acc = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()


        phase = kwargs.get("phase", VAL_PHASE)
        epoch = kwargs.get("epoch", "specified_model")
        # switch to evaluate mode
        model.eval()

        self.max_iter_num = len(dataloader)
        self.dataloader_iter = iter(dataloader)
        dataset = dataloader.dataset
        # prepare data fro validate
        num_samples = len(dataset)
        all_preds = np.zeros((num_samples, self.cfg.MODEL.NUM_JOINTS, 3), dtype=np.float)
        all_boxes = np.zeros((num_samples, 6))
        image_path = []
        filenames = []
        filenames_map = {}
        filenames_counter = 0
        imgnums = []
        idx = 0
        acc_threshold = 0.7

        ###
        result_output_dir, vis_output_dir = self.vis_setup(logger, phase, epoch)
        ###

        with torch.no_grad():
            end = time.time()
            num_batch = len(dataloader)
            for iter_step in range(self.max_iter_num):
                input_x, input_sup_A, input_sup_B, target_heatmaps, target_heatmaps_weight, meta = next(self.dataloader_iter)
                if phase == VAL_PHASE:
                    self._before_val_iter(input_x)

                data_time.update(time.time() - end)
                # prepare model input
                margin_left = meta["margin_left"]
                margin_right = meta["margin_right"]
                margin = torch.stack([margin_left, margin_right], dim=1).cuda()
                target_heatmaps = target_heatmaps.cuda(non_blocking=True)

                # outputs = model(concat_input, margin)
                if self.PE_Name == 'DCPOSE':
                    concat_input = torch.cat((input_x, input_sup_A, input_sup_B), 1).cuda()
                    outputs = model(concat_input, margin=margin)
                else:
                    outputs = model(input_x.cuda())

                if phase == VAL_PHASE:
                    if isinstance(model, torch.nn.DataParallel):
                        vis_dict = getattr(model.module, "vis_dict", None)
                    else:
                        vis_dict = getattr(model, "vis_dict", None)

                    if vis_dict:
                        self._running_val_iter(vis_dict=vis_dict, model_input=[input_x, input_sup_A, input_sup_B])

                # 输入图像、中间热度图、最终热度图可视化 tensorboard 的
                if phase == VAL_PHASE:
                    if self.PE_Name == 'DCPOSE':
                        if self.cfg.DEBUG.VIS_TENSORBOARD_2:
                            self._running_val_iter2(
                                input_x, input_sup_A, input_sup_B, target_heatmaps,
                                outputs[0], outputs[-1]
                            )
                    else:
                        pass



                if isinstance(outputs, list) or isinstance(outputs, tuple):
                    pred_heatmaps = outputs[-1]
                else:
                    pred_heatmaps = outputs


                _, avg_acc, cnt, pred = accuracy(pred_heatmaps.detach().cpu().numpy(),
                                                 target_heatmaps.detach().cpu().numpy())

                acc.update(avg_acc, cnt)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if iter_step % self.cfg.PRINT_FREQ == 0 or iter_step >= (num_batch - 1):
                    msg = '{}: [{}/{}]\t' \
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(phase, iter_step, num_batch, batch_time=batch_time,
                                                                          data_time=data_time, acc=acc)
                    logger.info(msg)

                if self.cfg.DEBUG.DEBUG:  # 直接保存图片的
                    if self.PE_Name == 'DCPOSE':
                        prefix = [iter_step,# 0
                                  osp.join(vis_output_dir, 'val/gt_joints_with_images'), # 1
                                  osp.join(vis_output_dir, 'val/pred_joints_with_images'), # 2
                                  osp.join(vis_output_dir, 'val/gt_hm_with_images'), # 3
                                  osp.join(vis_output_dir, 'val/pred_hm_with_images'), # 4
                                  # 三个粗糙的热度图
                                  osp.join(vis_output_dir, 'val/pred_p_hm_with_images'),# 5
                                  osp.join(vis_output_dir, 'val/pred_c_hm_with_images'), # 6
                                  osp.join(vis_output_dir, 'val/pred_n_hm_with_images'), # 7
                                  # 聚合的粗糙的热度图
                                  osp.join(vis_output_dir, 'val/pred_pcn_hm_with_images'), # 8
                                  osp.join(vis_output_dir, 'val/pcn_images'), # 9
                                ]
                        save_debug_images3(self.cfg,
                                          # 前          当前        后
                                          [input_sup_A, input_x, input_sup_B],  # 这里顺序影响
                                          # [input_x, input_sup_A, input_sup_B],
                                          meta,  # 关键点坐标位置
                                          target_heatmaps,
                                          pred * 4,
                                          outputs,
                                          prefix)
                    else:
                        prefix = [iter_step,
                                  osp.join(vis_output_dir, 'val/gt_joints_with_images'),
                                  osp.join(vis_output_dir, 'val/pred_joints_with_images'),
                                  osp.join(vis_output_dir, 'val/gt_hm_with_images'),
                                  osp.join(vis_output_dir, 'val/pred_hm_with_images'),
                                  # osp.join(vis_output_dir, 'val/pcn_images'),
                                  ]
                        save_debug_images2(self.cfg,
                                          # 当前
                                          input_x,  # 这里顺序影响
                                          # [input_x, input_sup_A, input_sup_B],
                                          meta,  # 关键点坐标位置
                                          target_heatmaps,
                                          pred * 4,
                                          outputs,
                                          prefix)




                #### for eval ####
                for ff in range(len(meta['image'])):
                    cur_nm = meta['image'][ff]
                    if not cur_nm in filenames_map:
                        filenames_map[cur_nm] = [filenames_counter]
                    else:
                        filenames_map[cur_nm].append(filenames_counter)
                    filenames_counter += 1

                center = meta['center'].numpy()
                scale = meta['scale'].numpy()
                score = meta['score'].numpy()
                num_images = input_x.size(0)

                preds, maxvals = get_final_preds(pred_heatmaps.clone().cpu().numpy(), center, scale)
                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[idx:idx + num_images, :, 2:3] = maxvals
                all_boxes[idx:idx + num_images, 0:2] = center[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = scale[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(scale * 200, 1)
                all_boxes[idx:idx + num_images, 5] = score
                image_path.extend(meta['image'])
                idx += num_images

                # tensorboard writ
                self.global_steps += 1
                #
                self._after_val_iter(meta["image"], preds, maxvals, vis_output_dir, center, scale)


        logger.info('########################################')
        logger.info('{}'.format(self.cfg.EXPERIMENT_NAME))

        name_values, perf_indicator = dataset.evaluate(self.cfg, all_preds, result_output_dir, all_boxes,
                                                       filenames_map, filenames, imgnums)

        model_name = self.cfg.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                self._print_name_value(name_value, model_name)
        else:
            self._print_name_value(name_values, model_name)

        tb_writer_dict["global_steps"] = self.global_steps

    def _before_train_iter(self, batch_x):
        if not self.cfg.DEBUG.VIS_TENSORBOARD:
            return

        show_image_num = min(6, len(batch_x))
        batch_x = batch_x[:show_image_num]
        label_name = "train_{}_x".format(self.global_steps)
        save_image = []
        for x in batch_x:
            x = tensor2im(x)
            save_image.append(x)
        save_image = np.stack(save_image, axis=0)
        self.tb_writer.add_images(label_name, save_image, global_step=self.global_steps, dataformats="NHWC")

    def _before_val_iter(self, batch_x):
        if not self.cfg.DEBUG.VIS_TENSORBOARD:
            return

        show_image_num = min(6, len(batch_x))
        batch_x = batch_x[:show_image_num]
        label_name = "val_{}_x".format(self.global_steps)
        save_image = []
        for x in batch_x:
            x = tensor2im(x)
            save_image.append(x)
        save_image = np.stack(save_image, axis=0)
        self.tb_writer.add_images(label_name, save_image, global_step=self.global_steps, dataformats="NHWC")

    def _running_val_iter(self, **kwargs):
        if not self.cfg.DEBUG.VIS_TENSORBOARD:
            return

        vis_dict = kwargs.get("vis_dict")
        #
        show_image_num = min(3, len(vis_dict["current_x"]))
        current_x = vis_dict["current_x"][0:show_image_num]  # [N,3,384,288]
        previous_x = vis_dict["previous_x"][0:show_image_num]  # [N,3,384,288]
        next_x = vis_dict["next_x"][0:show_image_num]  # [N,3,384,288]
        current_rough_heatmaps = vis_dict["current_rough_heatmaps"][0:show_image_num]  # [N,17,96,72]
        previous_rough_heatmaps = vis_dict["previous_rough_heatmaps"][0:show_image_num]  # [N,17,96,72]
        next_rough_heatmaps = vis_dict["next_rough_heatmaps"][0:show_image_num]  # [N,17,96,72]
        diff_A = vis_dict["diff_A"][0:show_image_num]  # [N,17,96,72]
        diff_B = vis_dict["diff_B"][0:show_image_num]  # [N,17,96,72]
        diff_heatmaps = vis_dict["diff_heatmaps"][0:show_image_num]  # [N,34,96,72]
        support_heatmaps = vis_dict["support_heatmaps"][0:show_image_num]  # [N,17,96,72]
        prf_ptm_combine_featuremaps = vis_dict["prf_ptm_combine_featuremaps"][0:show_image_num]  # [N,96,96,72]
        warped_heatmaps_list = [warped_heatmaps[0:show_image_num] for warped_heatmaps in vis_dict["warped_heatmaps_list"]]  # [N,17,96,72]
        output_heatmaps = vis_dict["output_heatmaps"][0:show_image_num]  # [N,17,96,72]
        # 三个输入图片
        show_three_input_image = make_grid(reverse_transforms(torch.cat([previous_x, current_x, next_x], dim=0)), nrow=show_image_num)
        self.tb_writer.add_image('01_three_input_image', show_three_input_image, global_step=self.global_steps)

        # show 2.
        three_rough_heatmaps_channels = []
        current_rough_heatmap_channels = current_rough_heatmaps.split(1, dim=1) # [N*1*96*72] *17
        previous_rough_heatmap_channels = previous_rough_heatmaps.split(1, dim=1)
        next_rough_heatmap_channels = next_rough_heatmaps.split(1, dim=1)
        num_channel = current_rough_heatmaps.shape[1]
        for i in range(num_channel):
            three_rough_heatmaps_channels.append(current_rough_heatmap_channels[i])
            three_rough_heatmaps_channels.append(previous_rough_heatmap_channels[i])
            three_rough_heatmaps_channels.append(next_rough_heatmap_channels[i])
        # 三帧热度图
        three_heatmaps_tensor = torch.clamp_min(torch.cat(three_rough_heatmaps_channels, dim=0), 0)
        three_heatmaps_image = make_grid(three_heatmaps_tensor, nrow=show_image_num)
        self.tb_writer.add_image('02_three_heatmaps_image', three_heatmaps_image, global_step=self.global_steps)

        # show 3.
        two_diff_channels = []
        diff_A_channels = diff_A.split(1, dim=1)
        diff_B_channels = diff_B.split(1, dim=1)
        num_channel = current_rough_heatmaps.shape[1]
        for i in range(num_channel):
            two_diff_channels.append(diff_A_channels[i])
            two_diff_channels.append(diff_B_channels[i])
        # 两帧差分的图像
        two_diff_channels_tensor = torch.clamp_min(torch.cat(two_diff_channels, dim=0), 0)
        two_diff_image = make_grid(two_diff_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('03_two_diff_image', two_diff_image, global_step=self.global_steps)

        # show4.  差分热度图通道图片
        diff_heatmaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(diff_heatmaps, 1, dim=1), dim=0), 0)
        diff_heatmaps_channels_image = make_grid(diff_heatmaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('04_diff_heatmaps_channels_image', diff_heatmaps_channels_image,
                                 global_step=self.global_steps)

        # show5.  融合后的热度图
        support_heatmaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(support_heatmaps, 1, dim=1), dim=0), 0)
        support_heatmaps_channels_image = make_grid(support_heatmaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('05_support_heatmaps_channels_image', support_heatmaps_channels_image,
                                 global_step=self.global_steps)

        # show6.  最终差分 热度图融合
        prf_ptm_combine_featuremaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(prf_ptm_combine_featuremaps, 1, dim=1), dim=0),
                                                                      0)
        prf_ptm_combine_featuremaps_channels_image = make_grid(prf_ptm_combine_featuremaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('06_prf_ptm_combine_featuremaps_channels_image',
                                 prf_ptm_combine_featuremaps_channels_image, global_step=self.global_steps)

        # show7.
        warped_heatmaps_1_channels_tensor = torch.clamp_min(torch.cat(torch.split(warped_heatmaps_list[0], 1, dim=1), dim=0), 0)
        warped_heatmaps_1_channels_image = make_grid(warped_heatmaps_1_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('07_warped_heatmaps_1_channels_image', warped_heatmaps_1_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_2_channels_tensor = torch.clamp_min(torch.cat(torch.split(warped_heatmaps_list[1], 1, dim=1), dim=0), 0)
        warped_heatmaps_2_channels_image = make_grid(warped_heatmaps_2_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('08_warped_heatmaps_2_channels_image', warped_heatmaps_2_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_3_channels_tensor = torch.clamp_min(torch.cat(torch.split(warped_heatmaps_list[2], 1, dim=1), dim=0), 0)
        warped_heatmaps_3_channels_image = make_grid(warped_heatmaps_3_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('09_warped_heatmaps_3_channels_image', warped_heatmaps_3_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_4_channels_tensor = torch.clamp_min(torch.cat(torch.split(warped_heatmaps_list[3], 1, dim=1), dim=0), 0)
        warped_heatmaps_4_channels_image = make_grid(warped_heatmaps_4_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('10_warped_heatmaps_4_channels_image', warped_heatmaps_4_channels_image,
                                 global_step=self.global_steps)
        warped_heatmaps_5_channels_tensor = torch.clamp_min(torch.cat(torch.split(warped_heatmaps_list[4], 1, dim=1), dim=0), 0)
        warped_heatmaps_5_channels_image = make_grid(warped_heatmaps_5_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('11_warped_heatmaps_5_channels_image', warped_heatmaps_5_channels_image,
                                 global_step=self.global_steps)

        # show8. 输出热度图
        output_heatmaps_channels_tensor = torch.clamp_min(torch.cat(torch.split(output_heatmaps, 1, dim=1), dim=0), 0)
        output_heatmaps_channels_image = make_grid(output_heatmaps_channels_tensor, nrow=show_image_num)
        self.tb_writer.add_image('12_output_heatmaps_channels_image', output_heatmaps_channels_image,
                                 global_step=self.global_steps)

    #  查看输入三帧图像、中间热度图、和最终热度图
    def _running_val_iter2(self,
                           input_x, input_sup_A, input_sup_B, target_heatmaps,
                           outputs_0, outputs_1
                           ):
        #
        show_image_num = 3
        # 图片经过逆归一化
        current_x = self.reverse_transforms(input_x[0:show_image_num])  # [N,3,384,288]
        previous_x = self.reverse_transforms(input_sup_A[0:show_image_num])  # [N,3,384,288]
        next_x = self.reverse_transforms(input_sup_B[0:show_image_num])  # [N,3,384,288]
        current_rough_heatmaps = outputs_0[0][0:show_image_num]  # [N,17,96,72]
        previous_rough_heatmaps = outputs_0[1][0:show_image_num]  # [N,17,96,72]
        next_rough_heatmaps = outputs_0[2][0:show_image_num]  # [N,17,96,72]

        t_heatmaps = target_heatmaps[0:show_image_num]  # [N,17,96,72]  热度图真值
        output_heatmaps = outputs_1[0:show_image_num]  # [N,17,96,72]   输出热度图
        # 三个输入图片
        show_three_input_image = make_grid(torch.cat([previous_x, current_x, next_x], dim=0),
                                           nrow=show_image_num)
        self.tb_writer.add_image('00_show_three_input_image', show_three_input_image, global_step=self.global_steps)
        # self.tb_writer.add_image('00_current_input_image', current_x, global_step=self.global_steps)
        # self.tb_writer.add_image('01_previous_image', previous_x, global_step=self.global_steps)
        # self.tb_writer.add_image('02_next_image', next_x, global_step=self.global_steps)

        # show 2.
        three_rough_heatmaps_channels = []
        current_rough_heatmap_channels = current_rough_heatmaps.split(1, dim=1)  # [N*1*96*72] *17
        previous_rough_heatmap_channels = previous_rough_heatmaps.split(1, dim=1)
        next_rough_heatmap_channels = next_rough_heatmaps.split(1, dim=1)
        num_channel = current_rough_heatmaps.shape[1]
        for i in range(num_channel):
            three_rough_heatmaps_channels.append(current_rough_heatmap_channels[i])
            three_rough_heatmaps_channels.append(previous_rough_heatmap_channels[i])
            three_rough_heatmaps_channels.append(next_rough_heatmap_channels[i])
        # 三帧热度图
        three_heatmaps_tensor = torch.clamp_min(torch.cat(three_rough_heatmaps_channels, dim=0), 0)
        three_heatmaps_image = make_grid(three_heatmaps_tensor, nrow=show_image_num)
        self.tb_writer.add_image('02_three_heatmaps_image', three_heatmaps_image, global_step=self.global_steps)

        # show5.  热度图真值
        t_heatmaps_tensor = torch.clamp_min(torch.cat(torch.split(t_heatmaps, 1, dim=1), dim=0), 0)
        # nrow = 每一行显示的图像数  17*3 的图片
        t_heatmaps_image = make_grid(t_heatmaps_tensor, nrow=show_image_num)
        self.tb_writer.add_image('03_t_heatmaps_image', t_heatmaps_image,
                                 global_step=self.global_steps)

        # show8. 输出热度图
        output_heatmaps_tensor = torch.clamp_min(torch.cat(torch.split(output_heatmaps, 1, dim=1), dim=0), 0)
        output_heatmaps_image = make_grid(output_heatmaps_tensor, nrow=show_image_num)
        self.tb_writer.add_image('04_output_heatmaps_image', output_heatmaps_image,
                                 global_step=self.global_steps)

    def _after_val_iter(self, image, preds_joints, preds_confidence, vis_output_dir, center, scale):
        cfg = self.cfg
        # prepare data
        coords = np.concatenate([preds_joints, preds_confidence], axis=-1)
        bboxes = []
        for index in range(len(center)):  # 先得到多个检测框
            xyxy_bbox = cs2box(center[index], scale[index], pattern="xyxy")
            bboxes.append(xyxy_bbox)

        if cfg.DEBUG.VIS_SKELETON or cfg.DEBUG.VIS_BBOX:  # 根据检测框画图
            from .vis_helper import draw_skeleton_in_origin_image
            draw_skeleton_in_origin_image(image, coords, bboxes, vis_output_dir, vis_skeleton=cfg.DEBUG.VIS_SKELETON,
                                          vis_bbox=cfg.DEBUG.VIS_BBOX)

    def vis_setup(self, logger, phase, epoch):
        if phase == TEST_PHASE:
            prefix_dir = "test"
        elif phase == TRAIN_PHASE:
            prefix_dir = "train"
        elif phase == VAL_PHASE:
            prefix_dir = "validate"
        else:
            prefix_dir = "inference"

        if isinstance(epoch, int):
            epoch = "model_{}".format(str(epoch))

        output_dir_base = osp.join(self.output_dir, epoch, prefix_dir, "use_gt_box" if self.cfg.VAL.USE_GT_BBOX else "use_precomputed_box")
        vis_output_dir = osp.join(output_dir_base, "vis")
        result_output_dir = osp.join(output_dir_base, "prediction_result")
        create_folder(vis_output_dir)
        create_folder(result_output_dir)
        logger.info("=> Vis Output Dir : {}".format(vis_output_dir))
        logger.info("=> Result Output Dir : {}".format(result_output_dir))

        if phase == VAL_PHASE:
            tensorboard_log_dir = osp.join(self.output_dir, epoch, prefix_dir, "tensorboard")
            self.tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)

        if self.cfg.DEBUG.VIS_SKELETON:
            logger.info("=> VIS_SKELETON")
        if self.cfg.DEBUG.VIS_BBOX:
            logger.info("=> VIS_BBOX")
        return result_output_dir, vis_output_dir

    def reverse_transforms(self, batch_tensor: torch.Tensor):
        """
        tensor
        将图片从归一化中解除
        注意这个适用于Tensorboard 版本，亮度设置为1 ，对比度设置为100
        保存成图片需要乘以255
        """
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if batch_tensor.shape[1] == 1:  # grayscale to RGB
            batch_tensor = batch_tensor.repeat((1, 3, 1, 1))

        for i in range(len(mean)):
            batch_tensor[:, i, :, :] = batch_tensor[:, i, :, :] * std[i] + mean[i]
        # =========tensorboard不需要下面操作=============
        # batch_tensor = batch_tensor * 255
        # RGB -> BGR
        # RGB_batch_tensor = batch_tensor.split(1, dim=1)
        # batch_tensor = torch.cat([RGB_batch_tensor[2], RGB_batch_tensor[1], RGB_batch_tensor[0]], dim=1)

        return batch_tensor

# # 将图片 和 热度图 结合进行可视化
# def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
#                         normalize=True):
#     '''
#     batch_image: [batch_size, channel, height, width]
#     batch_heatmaps: ['batch_size, num_joints, height, width]
#     file_name: saved file name
#     '''
#     if normalize:
#         batch_image = batch_image.clone()
#         min = float(batch_image.min())
#         max = float(batch_image.max())
#
#         batch_image.add_(-min).div_(max - min + 1e-5)
#
#     batch_size = batch_heatmaps.size(0)
#     num_joints = batch_heatmaps.size(1)
#     heatmap_height = batch_heatmaps.size(2)
#     heatmap_width = batch_heatmaps.size(3)
#     # 网格图像
#     grid_image = np.zeros((batch_size*heatmap_height,
#                            (num_joints+1)*heatmap_width,
#                            3),
#                           dtype=np.uint8)
#     # 热度图坐标 和 预测值
#     preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())
#
#     for i in range(batch_size):
#         # 图像 逆 归一化
#         image = batch_image[i].mul(255)\
#                               .clamp(0, 255)\
#                               .byte()\
#                               .permute(1, 2, 0)\
#                               .cpu().numpy()
#         # 热度图
#         heatmaps = batch_heatmaps[i].mul(255)\
#                                     .clamp(0, 255)\
#                                     .byte()\
#                                     .cpu().numpy()
#         # 改变图像大小
#         resized_image = cv2.resize(image,
#                                    (int(heatmap_width), int(heatmap_height)))
#
#         height_begin = heatmap_height * i
#         height_end = heatmap_height * (i + 1)
#         for j in range(num_joints):
#             # 在缩放图上 画出关键点
#             cv2.circle(resized_image,  # 画出坐标点
#                        (int(preds[i][j][0]), int(preds[i][j][1])),
#                        1, [0, 0, 255], 1)
#             heatmap = heatmaps[j, :, :]  # 热度图
#             # 热度图转换为颜色图
#             colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#             # 重叠后的图像
#             masked_image = colored_heatmap*0.7 + resized_image*0.3
#             # 根据给定的圆心和半径等画圆
#             cv2.circle(masked_image,
#                        (int(preds[i][j][0]), int(preds[i][j][1])),
#                        1, [0, 0, 255], 1)
#
#             width_begin = heatmap_width * (j+1)
#             width_end = heatmap_width * (j+2)
#             # 在网格上添加图像
#             grid_image[height_begin:height_end, width_begin:width_end, :] = \
#                 masked_image
#             # grid_image[height_begin:height_end, width_begin:width_end, :] = \
#             #     colored_heatmap*0.7 + resized_image*0.3
#         # 开头放上 原图加识别的 关键点
#         grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image
#
#     cv2.imwrite(file_name, grid_image)