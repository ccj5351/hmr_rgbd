# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: converTF2PyTorch.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 20-10-2019
# @last modified: Sun 20 Oct 2019 11:41:29 PM EDT

import os
from pprint import pprint
import tensorflow as tf



from os import makedirs
from os.path import exists
from os.path import join as pjoin
import numpy as np



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
        #pprint(tf_vars) # print the variables
        """ a tuple : (variable_name_str, variable_dimension_list) """
        resnet_dict = {}
        var_names = []
        for (varName, varDim) in tf_vars:
            if varName[0:12] == 'resnet_v2_50':
                print ('variable {} : dim = {}'.format(varName, varDim))
                """
                Variables are stored as Numpy arrays that you can load 
                with tf.train.load_variable(name)
                """
                resnet_dict[varName] = tf.train.load_variable(model_path, varName)
                var_names.append(varName)
        resnet_dict['resnet_v2_50_names'] = var_names
        npy_file = '/home/ccj/hmr-rgbd/results/saved_weights/'
        if not exists(npy_file):
            makedirs(npy_file)
        npy_file = pjoin(npy_file, 'hmr_pre_trained_resnet_v2_50.npy')

        #np.save(npy_file, resnet_dict)
        #print ('saved %s' % npy_file)
        """ this method also works well """
        #print_tensors_in_checkpoint_file(model_path, all_tensors=True, tensor_name='')
        
        """ load dictionary """ 
        resnet_dict2 = np.load(npy_file).item()
        print ('loaded var_names : ', resnet_dict2['resnet_v2_50_names'])
        tmp_name = 'resnet_v2_50/block4/unit_3/bottleneck_v2/conv2/weights'
        # check the variable dimensionality
        # print should be : [3, 3, 512, 512];
        print (resnet_dict2[tmp_name].shape)

