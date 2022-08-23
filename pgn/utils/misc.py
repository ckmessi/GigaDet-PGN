#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/9/9 14:34
# @File     : misc.py

"""
import time
import torch
import numpy as np

from pgn.datasets.voc import preprocess
from pgn.utils.logger import DEFAULT_LOGGER


def get_target_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device


def get_synchronized_time():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def reverse_to_original_numpy(img):
    return img.numpy().transpose((2, 0, 1))

def convert_to_pascal_voc_batch_format(imgs, labels, imgs_origin, min_size=600, max_size=1000, testing=False):

    sizes = torch.tensor([[imgs.shape[1]], [imgs.shape[2]]])
    img = np.array([preprocess(reverse_to_original_numpy(img), min_size, max_size) for img in imgs])
    img = torch.from_numpy(img)

    bbox_ratio = labels[:, 2:6]
    label_ = labels[:, 1].int()
    label_ = label_.unsqueeze(0)
    # TODO： fix this
    scale = torch.Tensor([1.0 for img in imgs])

    h = img.shape[2]
    w = img.shape[3]
    bbox_ = bbox_ratio.clone()
    # [left, top, right, bottom] convert to [ymin, xmin, ymax, xmax]
    bbox_[:, 0] = bbox_ratio[:, 1] * h
    bbox_[:, 2] = bbox_ratio[:, 3] * h
    bbox_[:, 1] = bbox_ratio[:, 0] * w
    bbox_[:, 3] = bbox_ratio[:, 2] * w
    bbox_ = bbox_.unsqueeze(0)

    if not testing:
        return (img, bbox_, label_, scale)
    else:
        gt_difficults_ = torch.Tensor([[0 for label in labels] for img in imgs])
        return (img, sizes, bbox_, label_, gt_difficults_)


def get_train_batch_elements(elements_ensemble, dataset_type):
    if dataset_type == 'PANDA':
        return convert_to_pascal_voc_batch_format(*elements_ensemble)
    else:
        return elements_ensemble


def get_test_batch_elements(elements_ensemble, dataset_type):
    if dataset_type == 'PANDA':
        return convert_to_pascal_voc_batch_format(*elements_ensemble, testing=True)
    else:
        return elements_ensemble
