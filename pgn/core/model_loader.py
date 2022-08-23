#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2021/2/1 16:16
# @File     : model_loader.py

"""
from pgn.model.faster_rcnn_vgg16 import FasterRCNNVGG16
from pgn.core.trainer import FasterRCNNTrainer
from pgn.utils import misc
from pgn.utils.logger import DEFAULT_LOGGER


def build_pgn_trainer(cfg):

    DEFAULT_LOGGER.info('start constructing model...')
    faster_rcnn = FasterRCNNVGG16(ratios=cfg.MODEL.PGN_ANCHOR_RATIOS, anchor_scales=cfg.MODEL.PGN_ANCHOR_SCALES)
    DEFAULT_LOGGER.info('finish constructing model.')

    device = misc.get_target_device()
    trainer = FasterRCNNTrainer(cfg, faster_rcnn).to(device)
    trainer.load(cfg.MODEL.WEIGHTS)

    return trainer
