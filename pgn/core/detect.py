#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/9/9 16:07
# @File     : detect.py

"""
import cv2
import time
import torch
import os
import numpy as np
import torchvision

from pgn.core import model_loader
from pgn.utils.creator_tool import _get_inside_index
from pgn.datasets.voc import preprocess

from pgn.utils import vis_tool
from pgn.utils import misc, parser
from pgn.utils.logger import DEFAULT_LOGGER


def get_top_k_pgn_count_anchor(img, anchor, patch_count, top_k=10, pgn_patch_iou_threshold=0.2):
    # remove anchor outside the bbox
    H = img.shape[1]
    W = img.shape[2]
    t0 = time.time()
    inside_index = _get_inside_index(anchor, H, W)
    anchor = anchor[inside_index]
    patch_count = patch_count[inside_index]
    DEFAULT_LOGGER.debug(f"[PGN->get_top_k] get_inside_index cost {time.time() - t0}")
    # get pgn_count_filtered
    pgn_count_filter = patch_count > 1
    pgn_count_filter_flatten = torch.flatten(pgn_count_filter)
    png_count_filtered = patch_count[pgn_count_filter_flatten]
    DEFAULT_LOGGER.debug(f"[PGN->get_top_k] get png_count_filtered cost {time.time() - t0}")

    # get anchor filtered
    anchor_filtered = anchor[pgn_count_filter_flatten.cpu().numpy()]
    DEFAULT_LOGGER.debug(f"[PGN->get_top_k] get anchor_filtered cost {time.time() - t0}")

    # get sorted pgn_count_filtered
    pgn_count_flatten = torch.flatten(png_count_filtered)
    pgn_count_flatten_argsort = pgn_count_flatten.argsort(descending=True)
    pgn_count_flatten_sorted = pgn_count_flatten[pgn_count_flatten_argsort].detach().cpu()
    anchor_filtered_sorted = anchor_filtered[pgn_count_flatten_argsort.cpu().numpy()]
    DEFAULT_LOGGER.debug(f"[PGN->get_top_k] get anchor_filtered_sorted cost {time.time() - t0}")

    # apply nms
    anchor_filtered_sorted_tensor = torch.from_numpy(anchor_filtered_sorted)
    DEFAULT_LOGGER.debug(f"[PGN->get_top_k] convert anchor_filtered_sorted to tensor cost {time.time() - t0}")
    keep = torchvision.ops.nms(anchor_filtered_sorted_tensor, pgn_count_flatten_sorted, pgn_patch_iou_threshold)
    DEFAULT_LOGGER.debug(f"[PGN->get_top_k] apply nms cost {time.time() - t0}")

    # import ipdb;ipdb.set_trace()
    # keep = cp.asnumpy(keep)
    anchor_filtered_sorted = anchor_filtered_sorted[keep]
    # NOTE: if keep is a single item array, such as [0] rather than [0, 1]
    # the anchor_filtered_sorted will be automatically squeezed to 1-dimention array
    # [[1,2,3,4],[2,2,3,4],[3,2,3,4]][0,1] =>[[1,2,3,4],[2,2,3,4]]
    # [[1,2,3,4],[2,2,3,4],[3,2,3,4]][0] =>[1,2,3,4]
    if len(anchor_filtered_sorted.shape) == 1:
        anchor_filtered_sorted = np.expand_dims(anchor_filtered_sorted, 0)


    anchor_top_k = anchor_filtered_sorted[0:top_k]
    pgn_count_top_k = pgn_count_flatten_sorted[0:min(top_k, len(anchor_top_k))]
    DEFAULT_LOGGER.debug(f"[PGN->get_top_k] get final top_k {time.time() - t0}")
    return anchor_top_k, pgn_count_top_k


def visualize_pgn_count(origin_img, origin_anchor_top_k, pgn_count_top_k, target_w=1080):
    origin_h, origin_w = origin_img.shape[0], origin_img.shape[1]
    target_h = int(target_w * (float(origin_h) / origin_w))
    img_to_vis = cv2.resize(origin_img, (target_w, target_h))

    anchor_top_k = map_anchor_into_original_image_area(origin_anchor_top_k, (target_h, target_w), (origin_h, origin_w))

    img_to_vis = img_to_vis.transpose(2, 0, 1)
    fig = vis_tool.vis_bbox(img_to_vis, anchor_top_k, score=pgn_count_top_k)
    img_to_vis = vis_tool.fig4vis(fig)
    img_to_vis = img_to_vis.transpose(1, 2, 0)  # 1, 2, 0
    cv2.imshow("", img_to_vis)
    cv2.waitKey(0)


def detect_single(trainer, origin_img, device=None, display=False, top_k=10, pgn_patch_iou_threshold=0.2, **kwargs):
    t0 = time.time()
    if device is None:
        device = misc.get_target_device()
    if origin_img is None:
        raise ValueError(f'imgs is None')

    origin_h, origin_w = origin_img.shape[0], origin_img.shape[1]
    if display:
        # NOTE: bring large time cost
        img = origin_img.copy()
    else:
        img = origin_img

    t1 = time.time()
    DEFAULT_LOGGER.debug(f'[PGN] time cost for image copy: {t1 - t0}')
    img = img.transpose((2, 0, 1))
    img = preprocess(img)
    t2 = time.time()
    DEFAULT_LOGGER.debug(f'[PGN] time cost for image preprocess: {t2 - t1}')
    imgs = np.expand_dims(img, axis=0)
    imgs = torch.Tensor(imgs)
    imgs = imgs.to(device).float()
    _, _, H, W = imgs.shape
    img_size = (H, W)
    features = trainer.faster_rcnn.extractor(imgs)
    t3 = time.time()

    anchor, pgn_counts = trainer.faster_rcnn.pgn(features, img_size, scale=1)

    t4 = misc.get_synchronized_time()
    DEFAULT_LOGGER.debug(f'[PGN] forward time: {t3 - t2}s for feature and {t4 - t3} for pgn.')
    pgn_count = pgn_counts[0]
    anchor_top_k, pgn_count_top_k = get_top_k_pgn_count_anchor(img, anchor, pgn_count, top_k=top_k, pgn_patch_iou_threshold=pgn_patch_iou_threshold)

    t5 = time.time()
    DEFAULT_LOGGER.debug(f'[PGN] forward time: {t5 - t4}s for get_top_k ')
    anchor_top_k = map_anchor_into_original_image_area(anchor_top_k, (origin_h, origin_w), img_size)
    t6 = time.time()
    DEFAULT_LOGGER.debug(f'[PGN] forward time: {t6 - t5}s for remap ')

    return anchor_top_k, pgn_count_top_k



def map_anchor_into_original_image_area(anchor_list, origin_shape, img_shape):
    resize_h, resize_w = img_shape
    origin_h, origin_w = origin_shape
    remapped_anchor_list = [
        [
            int(anchor[0] * origin_w / resize_w),
            int(anchor[1] * origin_h / resize_h),
            int(anchor[2] * origin_w / resize_w),
            int(anchor[3] * origin_h / resize_h),
        ] for anchor in anchor_list
    ]
    return np.array(remapped_anchor_list)


def detect(pgn_trainer, image_folder, display=False, **kwargs):
    device = misc.get_target_device()
    image_name_list = os.listdir(image_folder)
    for image_name in image_name_list:
        image_path = os.path.join(image_folder, image_name)
        # inference
        img = cv2.imread(image_path)
        anchor_top_k, pgn_count_top_k = detect_single(pgn_trainer, img, device, only_pgn_count=False, **kwargs)
        DEFAULT_LOGGER.info(f'anchor top k is {anchor_top_k}')
        if display:
            visualize_pgn_count(img, anchor_top_k, pgn_count_top_k)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--image_folder', type=str)
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--config-file', type=str)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = parser.setup_cfg(args)

    if args.load_path:
        cfg.defrost()
        cfg.MODEL.WEIGHTS = args.load_path

    pgn_trainer = model_loader.build_pgn_trainer(cfg)


    detect(pgn_trainer, args.image_folder, display=args.display, top_k=32)
