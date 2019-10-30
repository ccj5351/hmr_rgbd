# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: get_pt_resnetv2_feature.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 30-10-2019
# @last modified: Wed 30 Oct 2019 02:55:57 AM EDT

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import src.pfmutil as pfm
import json
import sys

import torch
from pytorch_src.ResnetV2 import resnet_v2_50


def normalize_rgb_via_mean_std(img):
    assert (np.shape(img)[2] == 3)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    r = (r - np.mean(r)) / np.std(r)
    g = (g-np.mean(g)) / np.std(g)
    b = (b-np.mean(b)) / np.std(b)
    return np.stack([r,g,b], axis = 2)


def save_to_json(dict_to_save, param_path):
        tmp_dict_to_save = {}
        for k in dict_to_save:
            tmp_dict_to_save[k] = dict_to_save[k].tolist()
        with open(param_path, 'w') as fp:
            json.dump(tmp_dict_to_save, fp, indent=4, sort_keys=True)

def run_model(x, pretrained_model_path = None):
    # get a new model
    my_model = resnet_v2_50().cuda()
    # load the weights
    my_model.load_state_dict(torch.load(pretrained_model_path))
    print ('Loading %s' % pretrained_model_path)
    x = x.cuda()
    y = my_model(x) 
    return y



if __name__ == "__main__":
    img = cv2.imread('/home/ccj/hmr-rgbd/data/coco1.png')[:,:,::-1]
    img = normalize_rgb_via_mean_std(img)
    H, W, C = img.shape[:]
    N = 1
    #pfm.show(img)
    #sys.exit()
    if not torch.cuda.is_available():
        raise Exception("No GPU found, please run without cuda")
    
    x = torch.from_numpy(np.expand_dims(img, axis=0)).float()
    x = x.permute(0,3,1, 2)
    pretrained_model_path = '/home/ccj/hmr-rgbd/results/saved_weights/hmr_pre_trained_resnet_v2_50.pt'
    #with torch.set_grad_enabled(False):
    y = run_model(x, pretrained_model_path)
    y = y.detach().cpu().numpy()
    y = np.reshape(y, [-1])
    print ('y shape = ', y.shape)
    save_to_json({'y_tf': y}, '/home/ccj/hmr-rgbd/results/resnet-v2-50-hmr-pt-feature.json')