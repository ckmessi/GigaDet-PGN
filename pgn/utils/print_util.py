#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/9/16 20:56
# @File     : print_util.py

"""
import sys, os

# Disable
def block_stdout():
    sys.stdout = open(os.devnull, 'w')

# Restore
def reset_stdout():
    sys.stdout = sys.__stdout__
