#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/9/10 15:32
# @File     : faster_rcnn.py

"""
import torch

class FasterRCNN(torch.nn.Module):

    def __init__(self, extractor, pgn, head):
        super(FasterRCNN, self).__init__()

        self.extractor = extractor
        self.pgn = pgn
        self.head = head

    @property
    def n_class(self):
        # Total number of classes including the background.
        return self.head.n_class


    def get_optimizer(self, cfg):
        """
        return optimizer, It could be overwritten if you want to specify
        special optimizer
        """
        lr = cfg.SOLVER.BASE_LR
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.SOLVER.WEIGHT_DECAY}]
        if cfg.SOLVER.OPTIMIZER == 'Adam':
            self.optimizer = torch.optim.Adam(params)
        else:
            self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer
