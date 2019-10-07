# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: read_check_points.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 19-06-2019
# @last modified: Wed 19 Jun 2019 01:26:03 PM EDT

# see https://stackoverflow.com/questions/38218174/how-do-i-find-the-variable-names-and-values-that-are-saved-in-a-checkpoint;


import os
# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

# print all tensors in checkpoint file
# The Tensorflow versioin is 1.7.0
chkp.print_tensors_in_checkpoint_file(
        file_name = "models/resnet_v2_50.ckpt",
        tensor_name = "",
        all_tensors = False,
        all_tensor_names = True
        )
