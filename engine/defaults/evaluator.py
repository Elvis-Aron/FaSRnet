#!/usr/bin/python
# -*- coding:utf8 -*-
import logging
import torch
import torch.nn
import os.path as osp
from tensorboardX import SummaryWriter

from .base import BaseExecutor
from datasets import build_eval_loader
from posetimation.zoo import build_model
from .checkpoints import get_all_checkpoints, get_latest_checkpoint
from engine.core import build_core_function
from engine.defaults import VAL_PHASE, TEST_PHASE


class DefaultEvaluator(BaseExecutor):

    def exec(self):
        self.eval()

    def __init__(self, cfg, output_folders: dict, phase=TEST_PHASE, **kwargs):
        super().__init__(cfg, output_folders, phase, **kwargs)

        cfg = self.cfg
        self.PE_Name = kwargs.get("PE_Name")
        self.phase = phase
        self.dataloader = build_eval_loader(cfg, phase)
        self.model = build_model(cfg, phase=phase)
        self.dataset = self.dataloader.dataset
        self.GPUS = cfg.GPUS
        self.test_eval_f = False  # 测试组合模型的精度

        self.output = cfg.OUTPUT_DIR

        self.eval_from_checkpoint_id = kwargs.get("eval_from_checkpoint_id", -1)
        self.evaluate_model_state_files = []
        if not self.test_eval_f:
            self.list_evaluate_model_files(cfg, phase)
        self.core_function = build_core_function(cfg, phase=phase, **kwargs)
        # self.core_function = build_core_function(cfg, PE_name=kwargs.get("PE_Name", "DcPose"), phase=phase,**kwargs)

        self.tb_writer_dict = {"writer": SummaryWriter(self.tb_save_folder),
                               "global_steps": 0}

    def list_evaluate_model_files(self, cfg, phase):
        subCfgNode = cfg.VAL if phase == VAL_PHASE else cfg.TEST
        if subCfgNode.MODEL_FILE:
            if subCfgNode.MODEL_FILE[0] == '.':
                model_state_file = osp.abspath(osp.join(self.checkpoints_save_folder, subCfgNode.MODEL_FILE))
            else:
                model_state_file = osp.join(self.cfg.ROOT_DIR, subCfgNode.MODEL_FILE)

            # model_state_file = osp.abspath(osp.join(cfg.ROOT_DIR, subCfgNode.MODEL_FILE))
            self.evaluate_model_state_files.append(model_state_file)
        else:
            if self.eval_from_checkpoint_id == -1:
                model_state_file = get_latest_checkpoint(self.checkpoints_save_folder)
                self.evaluate_model_state_files.append(model_state_file)
            else:
                candidate_model_files = get_all_checkpoints(self.checkpoints_save_folder)
                for model_file in candidate_model_files:
                    model_file_epoch_num = int(osp.basename(model_file).split("_")[1])
                    if model_file_epoch_num >= self.eval_from_checkpoint_id:
                        self.evaluate_model_state_files.append(model_file)

    def eval(self):

        if self.test_eval_f :
            checkpoints_file = '/home/xu/data/program/DCPose-im/output/PE/DcPose/DCPose_Network_Model_RSN_im8_2/PoseTrack17/chPRF_34_nPRF_2_chPTM_17_nPTM_1_chComb_64_nComb_2_D_3,6,9,12,15/checkpoints/epoch_31_state.pth'
            checkpoints_file_dc = '/home/xu/data/program/DCpose_file/DcPose_supp_files/pretrained_models/Dc_d_1.pth'

            checkpoints_file = torch.load(checkpoints_file)
            checkpoints_file_dc = torch.load(checkpoints_file_dc)
            if 'state_dict' in checkpoints_file.keys():
                checkpoints_file = checkpoints_file['state_dict']
            if 'state_dict' in checkpoints_file.keys():
                checkpoints_file_dc = checkpoints_file_dc['state_dict']

            self.model.load_state_dict(checkpoints_file, strict=False)
            self.model.hm_improve.load_state_dict(checkpoints_file_dc)
            self.model.cuda()
            self.core_function.eval(model=self.model, dataloader=self.dataloader,
                                    tb_writer_dict=self.tb_writer_dict, epoch=19,
                                    phase=self.phase)

        elif len(self.evaluate_model_state_files) == 0:
            logger = logging.getLogger(__name__)
            logger.error("=> No model state file available for evaluation")

        else:
            for model_checkpoint_file in self.evaluate_model_state_files:
                model, epoch = self.model_load(model_checkpoint_file)
                self.core_function.eval(model=model, dataloader=self.dataloader, tb_writer_dict=self.tb_writer_dict, epoch=epoch,
                                        phase=self.phase)


    def model_load(self, checkpoints_file):
        logger = logging.getLogger(__name__)
        logger.info("=> loading checkpoints from {}".format(checkpoints_file))
        checkpoint_dict = torch.load(checkpoints_file)
        # epoch = checkpoint_dict['begin_epoch']
        epoch = checkpoint_dict.get("begin_epoch", "0")
        model = self.model

        if "state_dict" in checkpoint_dict:
            model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint_dict['state_dict'].items()}

        else:
            model_state_dict = checkpoint_dict

        if self.PE_Name == 'MSRA':
            model_state_dict = {k.replace('rough_pose_estimation_net.', ''): v for k, v in model_state_dict.items()}
        if self.cfg.MODEL.NAME == 'PoseWarper':
            model.load_state_dict(model_state_dict, strict=False)
        else:
            model.load_state_dict(model_state_dict)
        if len(self.GPUS) > 1:
            model = torch.nn.DataParallel(model.cuda())
        else:
            model = model.cuda()
        return model, epoch
