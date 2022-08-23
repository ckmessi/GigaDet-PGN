#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2021/2/1 12:05
# @File     : eval_single.py.py

"""
import json

import cv2
import os
import numpy as np
import torch

from pgn.core import model_loader
from pgn.datasets.panda import convert_to_label_ltrb_list, PandaImageAndLabelDataset
from pgn.core.detect import detect_single
from pgn.schema.pgn_detection import PGNPatch
from pgn.utils import eval_tool
from pgn.utils import misc, visualizer, parser
from pgn.utils.logger import DEFAULT_LOGGER


def parse_args():
    current_parser = parser.default_argument_parser()
    current_parser.add_argument('--load_path', type=str)
    current_parser.add_argument('--image_path', type=str)
    current_parser.add_argument('--load_panda_annotation', action='store_true', default=False)
    current_parser.add_argument('--display', action='store_true', default=False)
    current_parser.add_argument('--save_path', type=str, default=False)
    args = current_parser.parse_args()
    return args


def get_labels(cfg, image_path, annotation_file_path: str):
    if not os.path.exists(annotation_file_path):
        msg = f'[PANDA] Invalid annotation file path: {annotation_file_path}'
        DEFAULT_LOGGER.error(msg)
        raise ValueError(msg)

    category_name_index_dict = {
        'person': 0
    }
    file_name = os.path.basename(image_path)
    dir_name = os.path.dirname(image_path)
    scene_name = os.path.basename(dir_name)
    image_name = scene_name + "/" + file_name

    with open(annotation_file_path, 'r') as annotation_file:
        annotation_dict = json.load(annotation_file)
        image_annotation = annotation_dict[image_name]
        objects_list = image_annotation['objects list']
        label_list_ltrb = convert_to_label_ltrb_list(objects_list, category_name_index_dict, target_key=cfg.DATASETS.PANDA.PERSON_KEY)
        labels = torch.from_numpy(np.array(label_list_ltrb))
        number_labels = len(labels)
        labels_out = torch.zeros((number_labels, 6))
        if number_labels:
            labels_out[:, 1:] = labels
    return labels_out


def fetch_label_bboxes(cfg, image_path, image_origin):
    panda_root = cfg.DATASETS.DATASET_ROOT
    annotation_file_path = os.path.join(panda_root, PandaImageAndLabelDataset.test_annotation_file_path)
    labels_out = get_labels(cfg, image_path, annotation_file_path)

    def build_gt_bbox_list(image_origin, labels_out):
        origin_height, origin_width = image_origin.shape[0], image_origin.shape[1]
        whwh = torch.Tensor([origin_width, origin_height, origin_width, origin_height])
        index_in_batch = 0
        labels = labels_out[labels_out[:, 0] == index_in_batch, 1:]
        label_boxes = labels[:, 1:] * whwh
        # label_boxes = `N x [category, left, top, right, bottom]`
        return label_boxes

    label_bboxes = build_gt_bbox_list(image_origin, labels_out)

    # since label_bboxes is [left, top, right, bottom] format
    label_bboxes = label_bboxes[:, [1, 0, 3, 2]]
    return label_bboxes


def resize_anchor(anchor_array, origin_shape, target_shape):
    target_h, target_w = target_shape
    origin_h, origin_w = origin_shape
    remapped_anchor_list = [
        [
            int(anchor[0] * origin_w / target_w),
            int(anchor[1] * origin_h / target_h),
            int(anchor[2] * origin_w / target_w),
            int(anchor[3] * origin_h / target_h),
        ] for anchor in anchor_array
    ]
    return np.array(remapped_anchor_list)


def eval_single_image(cfg, image_path, load_panda_annotation, display, save_path=None):

    # build trainer
    pgn_trainer = model_loader.build_pgn_trainer(cfg)
    origin_img = cv2.imread(image_path)
    device = misc.get_target_device()

    # forward
    anchor_top_k, pgn_count_top_k = detect_single(pgn_trainer, origin_img, device, only_pgn_count=False, top_k=128)
    DEFAULT_LOGGER.info(f"[Eval] Finish detect_single.")
    DEFAULT_LOGGER.info(f"[Eval] anchor_top_k.shape is {anchor_top_k.shape}")
    DEFAULT_LOGGER.info(f"[Eval] pgn_count_top_k.shape is {pgn_count_top_k.shape}")

    # eval
    if load_panda_annotation:
        # get `anchor_top_k` and `gt_bboxes` both in {`origin`} size
        # now `anchor_top_k` is in original size
        label_bboxes = fetch_label_bboxes(cfg, image_path, origin_img)
        recall_dict = eval_tool.calc_bboxes_recall_in_patches([anchor_top_k], [label_bboxes])
        DEFAULT_LOGGER.info(f"recall_dict: {recall_dict}")
        DEFAULT_LOGGER.info(f"recall@64={recall_dict[64]}")

    if display or save_path:
        DEFAULT_LOGGER.info(f"start to visualize anchor_top_k")
        anchor_top_k_patch = [PGNPatch(anchor[1], anchor[0], anchor[3], anchor[2], 0, pgn_count_top_k[i]) for (i, anchor) in enumerate(anchor_top_k)]
        #gt_bboxes = [BaseDetection(label_bbox[1], label_bbox[0], label_bbox[3], label_bbox[2], 0, 1.0) for label_bbox in label_bboxes]
        visualizer.visualize_patches_result(origin_img, [], anchor_top_k_patch, 1080, display=display, save_path=save_path)


if __name__ == '__main__':

    # read args and cfg
    args = parse_args()
    cfg = parser.setup_cfg(args)
    image_path = args.image_path
    load_panda_annotation = args.load_panda_annotation
    display = args.display
    save_path = args.save_path

    # set current model path
    if args.load_path:
        cfg.defrost()
        cfg.MODEL.WEIGHTS = args.load_path

    # execute eval
    eval_single_image(cfg, image_path, load_panda_annotation, display, save_path)
