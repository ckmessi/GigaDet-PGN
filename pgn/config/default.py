#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2021/1/30 15:19
# @File     : default.py

"""

from .cfg_node import CfgNode as CN

_C = CN()

_C.INFO = CN()
_C.INFO.UID = "default_uid"
_C.INFO.LOG_LEVEL = "INFO"

# model
_C.MODEL = CN()
_C.MODEL.WEIGHTS = ""

_C.MODEL.PGN_ANCHOR_RATIOS = [1]
_C.MODEL.PGN_ANCHOR_SCALES = [1, 2, 4, 8]

_C.MODEL.ANCHOR = CN()
_C.MODEL.ANCHOR.SUB_SAMPLE = True
# the strategy to calculate include count: `IOU`, `Include`, and `IncludeIgnoreSmall`
_C.MODEL.ANCHOR.INCLUDE_STRATEGY = 'IOU'


# dataset
_C.DATASETS = CN()
_C.DATASETS.DATASET_TYPE = 'panda'
_C.DATASETS.DATASET_ROOT = '/path/to/PANDA/PANDA_IMAGE'

# dataset-PANDA
_C.DATASETS.PANDA = CN()
_C.DATASETS.PANDA.PERSON_KEY = "visible body"
_C.DATASETS.PANDA.REDUCE_INPUT = True


# solver
_C.SOLVER = CN()
# sigma for l1_smooth_loss
_C.SOLVER.PGN_SIGMA = 3.
_C.SOLVER.ROI_SIGMA = 1.

_C.SOLVER.OPTIMIZER = 'SGD'
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.LR_DECAY = 0.1
_C.SOLVER.BASE_LR = 1e-3
_C.SOLVER.STEPS = [9, 13]

_C.SOLVER.MAX_EPOCH = 14
_C.SOLVER.CHECKPOINT_PERIOD = 5


# visdom
_C.VISDOM = CN()
_C.VISDOM.ENABLED = False
_C.VISDOM.ENV = 'visdom_default'


# output
_C.OUTPUT = CN()
_C.OUTPUT.ROOT_DIR = 'outputs/'
