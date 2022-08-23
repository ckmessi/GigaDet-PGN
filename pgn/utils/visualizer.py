#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2021/2/1 20:55
# @File     : visualizer.py

"""

from typing import List

import cv2

from pgn.schema.pgn_detection import PGNPatch, BaseDetection

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (225, 255, 255)


def draw_patch_info(image, patch: PGNPatch, color=COLOR_GREEN, thickness=None, draw_label=True):
    bbox = patch.bbox
    p1 = (int(bbox.left), int(bbox.top))
    p2 = (int(bbox.right), int(bbox.bottom))
    thickness = thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    # thickness = 5
    cv2.rectangle(image, p1, p2, color, thickness=thickness, lineType=cv2.LINE_AA)
    if draw_label:
        score = '%.2f' % patch.confidence
        # label = '%g' % patch.category
        label = ""
        text = score + " " + label
        font_thickness = max(thickness - 1, 1)
        text_size = cv2.getTextSize(text, 0, fontScale=thickness / 3, thickness=font_thickness)[0]
        p2 = p1[0] + text_size[0], p1[1] - text_size[1] - 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)
        p3 = (p1[0], p1[1] - 2)
        cv2.putText(image, text, p3, 0, thickness / 3, COLOR_WHITE, thickness=font_thickness, lineType=cv2.LINE_AA)
    return image


def draw_patches_result_to_image(image, patches: List[PGNPatch], color=COLOR_GREEN, draw_label=True):
    for patch in patches:
        image = draw_patch_info(image, patch, color, draw_label=draw_label)
    return image


def visualize_patches_result(image, gt_bboxes_list: List[BaseDetection], patches_list: List[PGNPatch], target_size=None, draw_label=True, display=False, save_path=None):
    # draw gt bounding boxes using green rectangle
    image = draw_patches_result_to_image(image, gt_bboxes_list, COLOR_GREEN, draw_label=draw_label)
    # draw predict bounding boxes using red rectangle
    image = draw_patches_result_to_image(image, patches_list, COLOR_RED, draw_label=draw_label)

    if target_size:
        ratio = target_size / max(image.shape[0], image.shape[1])
        interp = cv2.INTER_AREA
        image = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)), interpolation=interp)

    if display:
        cv2.namedWindow('patches_result', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('patches_result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save_path:
        cv2.imwrite(save_path, image)
