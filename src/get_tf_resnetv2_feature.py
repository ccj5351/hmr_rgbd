# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: tf_vs_pt_ResnetV2_50.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 30-10-2019
# @last modified: Wed 30 Oct 2019 07:33:42 PM EDT

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import cv2
import src.pfmutil as pfm
import json

def normalize_rgb_via_mean_std(img):
    assert (np.shape(img)[2] == 3)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    r = (r - np.mean(r)) / np.std(r)
    g = (g-np.mean(g)) / np.std(g)
    b = (b-np.mean(b)) / np.std(b)
    return np.stack([r,g,b], axis = 2)

def Encoder_resnet(x, is_training=True, weight_decay=0.001, reuse=False):
    """
    Resnet v2-50
    Assumes input is [batch, height_in, width_in, channels]!!
    Input:
    - x: N x H x W x 3
    - weight_decay: float
    - reuse: bool->True if test

    Returns:
    - net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.
    """

    #from tensorflow.contrib.slim.python.slim.nets import resnet_v2
    from src import resnet_v2
    with tf.name_scope("Encoder_resnet", [x]):
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            net, end_points, debug_dict = resnet_v2.resnet_v2_50(
                x,
                num_classes=None,
                is_training=is_training,
                reuse=reuse,
                scope='resnet_v2_50',
                isFetchDictForDebug = True
                )
            net = tf.squeeze(net, axis=[1, 2])
    variables = tf.contrib.framework.get_variables('resnet_v2_50')
    return net, variables, debug_dict

def save_to_json(dict_to_save, param_path):
        tmp_dict_to_save = {}
        for k in dict_to_save:
            tmp_dict_to_save[k] = dict_to_save[k].tolist()
        with open(param_path, 'w') as fp:
            json.dump(tmp_dict_to_save, fp, indent=4, sort_keys=True)

def run_model(x, is_training=True,weight_decay=0.001, pretrained_model_path = None):
    sess = tf.Session()
    img_resnet_vars_list = {}
    img_feat, E_var, debug_dict = Encoder_resnet(x, is_training, weight_decay)
    #print ('E_var = ', E_var)
    #load resnet_v2_50 part
    for var in E_var:
        if 'resnet_v2_50' in var.name:
            """ given: 
                Variable name in checkpoint file:  'resnet_v2_50/block1/unit_1/bottleneck_v2/conv1/BatchNorm/beta';
                variable name in this session: 
                    'resnet_v2_50/block1/unit_1/bottleneck_v2/conv1/BatchNorm/beta:0'
                    So we have to extract the checkpoint variable name from the session Variable name;    
            """
            key_tmp = var.name.split(":")[0]
            img_resnet_vars_list[key_tmp] = var
    #print ('img_resnet_vars_list = ', img_resnet_vars_list)
    pre_train_saver_img = tf.train.Saver(img_resnet_vars_list)

    # load pre_train
    pre_train_saver_img.restore(sess, pretrained_model_path)
    name_list = [
            ('resnet_v2_50/conv1/weights', (7, 7, 3, 64)),
            ('resnet_v2_50/conv1/biases', (64,)),
            ]

    fetch_dict = {'y': img_feat}
    for key,shape in name_list:
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            fetch_dict[key] = tf.get_variable(key, shape)
    
    """ for debugging dict from resnet_v2_50 """
    for k,v in debug_dict.items():
        fetch_dict[k] = v
    fetch_dict = sess.run(fetch_dict)
    return fetch_dict


if __name__ == "__main__":
    img = cv2.imread('/home/ccj/hmr-rgbd/data/coco1.png')[:,:,::-1]
    img = normalize_rgb_via_mean_std(img)
    H, W, C = img.shape[:]
    N = 1
    #pfm.show(img)
    #x_pl = tf.placeholder(tf.float32, shape = [N,H,W,C])
    x = tf.constant(np.expand_dims(img, axis=0), dtype=tf.float32)
    pretrained_model_path = "/home/ccj/hmr-rgbd/models/hmr_pretrained_model/model.ckpt-667589"
    weight_decay = 0.001
    is_training = False
    fetch_dict = run_model(x, is_training, weight_decay, pretrained_model_path)
    for k, v in fetch_dict.items():
        if 'x_' in k:
            print ("[***] ", k , "has ", v.shape)
            n,h,w,c = v.shape
            for c_idx in [50]:
                if c_idx < c:
                    print ('channel slice c = %d\n' %c_idx, v[0, 0: min(7,h), 0:min(7,w), c_idx])
    
    y = np.reshape(fetch_dict['y'], [-1])
    print ('y shape = ', y.shape)
    print ('y = ', y[100:110])
    print ('-'*20   )
    #save_to_json(fetch_dict, '/home/ccj/hmr-rgbd/results/resnet-v2-50-hmr-tf-fetch.json')
