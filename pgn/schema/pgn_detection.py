#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2021/2/1 20:57
# @File     : pgn_detection.py

"""


class BoundingBox:
    left: int
    right: int
    top: int
    bottom: int

    def __init__(self, left, top, right, bottom):
        self.left = int(left)
        self.top = int(top)
        self.right = int(right)
        self.bottom = int(bottom)


class BaseDetection:
    bbox: BoundingBox
    category: int
    confidence: float

    def __init__(self, left, top, right, bottom, category, confidence):
        self.bbox = BoundingBox(left, top, right, bottom)
        self.category = int(category)
        self.confidence = confidence



class PGNPatch(BaseDetection):

    def __init__(self, left, top, right, bottom, category, confidence):
        self.bbox = BoundingBox(left, top, right, bottom)
        self.category = int(category)
        self.confidence = confidence

    def __str__(self):
        return f'[PGNPatch]: left={self.bbox.left}, top={self.bbox.top}, right={self.bbox.right}, bottom={self.bbox.bottom}'
