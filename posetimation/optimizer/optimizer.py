#!/usr/bin/python
# -*- coding:utf8 -*-
import torch.optim as optimizer_zoo
import logging


def build_optimizer(cfg, model):
    optimizer = None
    logger = logging.getLogger(__name__)
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optimizer_zoo.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )

    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optimizer_zoo.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.LR
        )
        logger.info(
            "=> Use learn rate :  {}".format(cfg.TRAIN.LR))

    return optimizer
