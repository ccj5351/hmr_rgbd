# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: converTF2PyTorch.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 20-10-2019
# @last modified: Sun 20 Oct 2019 09:35:56 PM EDT

import os
from pprint import pprint
import tensorflow as tf

if __name__ == "__main__":
    if 0:
        # Path to our TensorFlow checkpoint
        model_path = "/home/ccj/hmr-rgbd/models/resnet_v2_50.ckpt"  
        tf_vars = tf.train.list_variables(model_path)
        pprint(tf_vars)
        #Variables are stored as Numpy arrays that you can load 
        #with tf.train.load_variable(name);

    if 1:
        from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
        # Path to our TensorFlow checkpoint

        model_path = "/home/ccj/hmr-rgbd/models/hmr_pretrained_model/model.ckpt-667589"
        tf_vars = tf.train.list_variables(model_path)
        pprint(tf_vars)
        #Variables are stored as Numpy arrays that you can load 
        #with tf.train.load_variable(name);
        
        """ this method also works well """
        #print_tensors_in_checkpoint_file(model_path, all_tensors=True, tensor_name='')
