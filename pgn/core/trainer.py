from __future__ import absolute_import
import os
from collections import namedtuple
import time

import torch

from pgn.utils.creator_tool import AnchorTargetCreator, AnchorIncludeCreator

from torch import nn
from pgn.utils import array_tool
from pgn.utils import misc
from pgn.utils.logger import DEFAULT_LOGGER

from torchnet.meter import AverageValueMeter

LossTuple = namedtuple('LossTuple',
                       [
                        'patch_count_loss',
                        'total_loss'
                        ])


class FasterRCNNTrainer(nn.Module):

    LEGACY_KEYS_TO_DROP = ["rpn.score", "rpn.loc"]
    LEGACY_KEYS_TO_REPLACE = ["rpn."]

    def __init__(self, cfg, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.pgn_sigma = cfg.SOLVER.PGN_SIGMA

        # target creator create gt_bbox gt_label etc as training targets. 
        self.anchor_target_creator = AnchorTargetCreator()
        self.anchor_count_creator = AnchorIncludeCreator(cfg, subsample=cfg.MODEL.ANCHOR.SUB_SAMPLE)

        self.optimizer = self.faster_rcnn.get_optimizer(cfg)

        # indicators for training status
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

        self.device = misc.get_target_device()

    def forward(self, imgs, bboxes, labels, scale):
        if bboxes.shape[0] != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)
        anchor, patch_counts = self.faster_rcnn.pgn(features, img_size, scale)


        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        patch_count = patch_counts[0]

        # ------------------ Count losses -------------------#
        gt_pgn_count, gt_pgn_count_label = self.anchor_count_creator(
            array_tool.tonumpy(bbox),
            anchor,
            img_size)
        gt_pgn_count_label = array_tool.totensor(gt_pgn_count_label).long()
        gt_pgn_count = array_tool.totensor(gt_pgn_count)
        patch_count_loss = _fast_rcnn_count_loss(
            patch_count,
            gt_pgn_count,
            gt_pgn_count_label.data,
            self.pgn_sigma)

        losses = [patch_count_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses

    def save_checkpoint_content(self, save_dir, epoch, save_path=None, save_optimizer=False, **kwargs):
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['other_info'] = kwargs

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            save_path = "model_{:04d}".format(epoch)
            time_str = time.strftime('%Y%m%d_%H%M%S')
            save_path += f"_{time_str}"
            for k, v in kwargs.items():
                save_path += '_%s' % v
            save_path += ".pth"


        save_path = os.path.join(save_dir, save_path)

        DEFAULT_LOGGER.info(f"saving to checkpoints: {save_path}")
        torch.save(save_dict, save_path)
        return save_path

    def load(self, path, load_optimizer=True):
        if not path:
            return

        DEFAULT_LOGGER.info(f'load pretrained model from {path}')

        device = misc.get_target_device()
        state_dict = torch.load(path, map_location=torch.device(device))
        if 'model' in state_dict:
            state_dict['model'] = self.process_legacy_model_state_dict(state_dict['model'])
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    @staticmethod
    def process_legacy_model_state_dict(state_dict):
        """
        to be compatible with legacy model, replace some key in state_dict
        :param state_dict:
        :return:
        """
        for key in list(state_dict.keys()):
            # drop
            drop = False
            for legacy_key_to_drop in FasterRCNNTrainer.LEGACY_KEYS_TO_DROP:
                if key.startswith(legacy_key_to_drop):
                    del state_dict[key]
                    drop = True
            if drop:
                continue

            # replace
            for legacy_key_to_replace in FasterRCNNTrainer.LEGACY_KEYS_TO_REPLACE:
                if key.startswith(legacy_key_to_replace):
                    new_key = key.replace("rpn.", "pgn.")
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

        return state_dict

    def update_meters(self, losses):
        loss_d = {k: array_tool.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()

    def get_average_meter_data(self):
        return {k: v.mean for k, v in self.meters.items()}

    def adjust_learning_rate(self, cfg, current_epoch):
        if current_epoch in cfg.SOLVER.STEPS:
            DEFAULT_LOGGER.info(f"[Train] Execute learning rate adjust.")
            self.faster_rcnn.scale_lr(cfg.SOLVER.LR_DECAY)

    def output_step_info(self, cfg, progress_bar, epoch):
        epoch_progress = '%12s' % ('%g/%g' % (epoch, cfg.SOLVER.MAX_EPOCH))
        mem_info = '%12s' % ('%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0))  # (GB)
        loss_info = self.get_average_meter_data()
        patch_count_loss_str = '%12.4g' % loss_info['patch_count_loss']
        desc = epoch_progress + mem_info + patch_count_loss_str
        progress_bar.set_description(desc)
        return desc

    def output_eval_info(self, eval_result, title="eval_info"):
        learning_rate = self.faster_rcnn.optimizer.param_groups[0]['lr']
        # make sure the recall_dict contains certain key like `16`, `32`, ..., `128`
        recall_dict = eval_result['recall_dict']
        # title
        DEFAULT_LOGGER.info("")
        DEFAULT_LOGGER.info(f"Evaluation: {title}")
        DEFAULT_LOGGER.info("  -----------------------------------------------------")
        DEFAULT_LOGGER.info("  lr    | r@16  | r@32  | r@48  | r@64  | r@128 | r@512")
        DEFAULT_LOGGER.info("  -----------------------------------------------------")
        DEFAULT_LOGGER.info("  {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f}".format(learning_rate, recall_dict[16], recall_dict[32], recall_dict[48], recall_dict[64], recall_dict[128], recall_dict[512]))
        DEFAULT_LOGGER.info("  -----------------------------------------------------")

    def save_checkpoint(self, cfg, eval_result, epoch, metric_key='map'):

        checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        if (epoch % checkpoint_period == 0) or (epoch == cfg.SOLVER.MAX_EPOCH):
            save_dir = cfg.OUTPUT.ROOT_DIR
            self.save_checkpoint_content(save_dir, epoch=epoch, map=eval_result[metric_key])


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()



def _fast_rcnn_count_loss(pred_count, gt_count, gt_label, sigma):
    device = misc.get_target_device()
    in_weight = torch.zeros(gt_count.shape).to(device)
    in_weight[(gt_label > -1).view(-1, 1).expand_as(in_weight).to(device)] = 1
    count_loss = _smooth_l1_loss(pred_count, gt_count, in_weight.detach(), sigma)
    count_loss /= ((gt_label > -1).sum().float())  # ignore gt_label==-1 for count_loss
    return count_loss
