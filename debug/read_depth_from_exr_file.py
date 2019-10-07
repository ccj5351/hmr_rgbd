# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: read_depth_from_exr_file.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 10-06-2019
# @last modified: Mon 10 Jun 2019 06:18:44 PM EDT

import cv2
dep = cv2.imread("0.exr",-1) # "-1" means any depth or channel;
