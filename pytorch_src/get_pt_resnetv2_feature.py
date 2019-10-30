# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: get_pt_resnetv2_feature.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 30-10-2019
# @last modified: Wed 30 Oct 2019 04:51:17 PM EDT

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
from pytorch_src.ResnetV2 import get_tf2pt_key_map_dict, map_tf_dictKeys_2PyTorch_dictKeys

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
    my_model = resnet_v2_50(isFetchDictForDebug = True).cuda()
    # load the weights
    my_model.load_state_dict(torch.load(pretrained_model_path))
    print ('Loading %s' % pretrained_model_path)
    my_model.eval()
    x = x.cuda()
    y, debug_dict = my_model(x)
    
    map_dict = get_tf2pt_key_map_dict()
    tf_name_list = [
            ('resnet_v2_50/conv1/weights', (7, 7, 3, 64)),
            ('resnet_v2_50/conv1/biases', (64,)),
            ]
    my_model_state_dict = my_model.state_dict()
    #with torch.set_grad_enabled(False):
    fetch_dict = {'y': y.view(-1).detach().cpu().numpy()}
    for tf_key, tf_shape in tf_name_list:
        pt_key = map_tf_dictKeys_2PyTorch_dictKeys(map_dict, tf_key)
        var = my_model_state_dict[pt_key]
        pt_size = var.size()
        if len(pt_size) == 4:
            pt_size = (pt_size[2], pt_size[3], pt_size[1], pt_size[0])
            var = var.permute(2,3,1,0)
        assert tf_shape == pt_size
        fetch_dict[pt_key] = var.detach().cpu().numpy()
    
    """ for debugging dict from resnet_v2_50 """
    for k,v in debug_dict.items():
        # change [N,C,H,W] to [N, H, W, C]
        if k not in [ 'x_layer1', 'x_layer2', 'x_layer3']:
            print ("[***] ", k , "has ", v.shape)
            fetch_dict[k] = v.permute(0,2,3,1).detach().cpu().numpy()
            n,h,w,c = fetch_dict[k].shape
            print ("[***] after permute ", k , "has ", fetch_dict[k].shape)
            for c_idx in [50]:
                if c_idx < c:
                    print ('channel slice c = %d\n' %c_idx, fetch_dict[k][0, 0: min(5,h), 0:min(5,w), c_idx])
    return fetch_dict

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
    x = x.permute(0,3,1,2)
    pretrained_model_path = '/home/ccj/hmr-rgbd/results/saved_weights/hmr_pre_trained_resnet_v2_50.pt'
    #with torch.set_grad_enabled(False):
    fetch_dict = run_model(x, pretrained_model_path)
    y = np.reshape(fetch_dict['y'], [-1])
    print ('y shape = ', y.shape)
    print ('y = ', y[100:105])
    #save_to_json(fetch_dict, '/home/ccj/hmr-rgbd/results/resnet-v2-50-hmr-pt-fetch.json')
