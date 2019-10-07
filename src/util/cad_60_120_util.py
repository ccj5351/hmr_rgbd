# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: cad_60_120_util.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 31-07-2019
# @last modified: Wed 31 Jul 2019 06:05:02 PM EDT

"""
Utiliy function:

@get_cad_2_lsp_idx
@xPixelFromCoords
@yPixelFromCoords
@getPixelValuesFromCoords
@read_cad_txt_annotation

"""

import numpy as np

# > see http://pr.cs.cornell.edu/humanactivities/data.php#format
""" seems there exists left-right swap when projectio from 3d joints to 2d joints;
    but when you want to extract 14 lsp 3d joints from 15 cad60/120 joints, no left-right swap;
"""
def get_cad_2_lsp_idx(is2DJoints):
    # cad 60/120 datasets has 15 joints:
    # 
    
    cad_names_15 = [
        'Head', # 1
        'Neck', # 2
        'Torso', # 3
        'L Shoulder', # 4
        'L Elbow', # 5
        'R Shoulder', # 6
        'R Elbow', # 7
        'L Hip', # 8
        'L Knee', # 9
        'R Hip', # 10
        'R Knee', # 11
        'L Wrist', # 12, L Hand
        'R Wrist', # 13, R Hand
        'L Ankle', # 14, L Foot
        'R Ankle', # 15, R Foot
        ]
    
    cad_names_swapped_15 = [
        'Head', # 1
        'Neck', # 2
        'Torso', # 3
        'R Shoulder', # 6
        'R Elbow', # 7
        'L Shoulder', # 4
        'L Elbow', # 5
        'R Hip', # 10
        'R Knee', # 11
        'L Hip', # 8
        'L Knee', # 9
        'R Wrist', # 13, R Hand
        'L Wrist', # 12, L Hand
        'R Ankle', # 15, R Foot
        'L Ankle', # 14, L Foot
        ]
    #NOTE: when plotting the 2d joints, I find the joints are left-right swapped.
    # So here we "CORRECT" it so that 'right' is right, 'left' is left.
    # And we will follow this cretia across all the datasets, like, surreal, cad, h3.6m, etc;  
    want_names = [
        'R Ankle', # 0
        'R Knee', # 1
        'R Hip', #2
        'L Hip', #3
        'L Knee', #4
        'L Ankle', #5
        'R Wrist', #6
        'R Elbow', #7
        'R Shoulder', #8
        'L Shoulder', #9
        'L Elbow', #10
        'L Wrist', #11
        'Neck', #12
        'Head' #13
    ]

    # cad_to_lsp_idx = [15, 11, 10, 8, 9, 14, 13, 7, 6, 4, 5, 12, 2, 1] - 1
    # so, cad_to_lsp_idx is [14, 10, 9, 7, 8, 13, 12, 6, 5, 3, 4, 11, 1, 0] ;
    
    cad_to_lsp_idx_swap = [cad_names_swapped_15.index(j) for j in want_names]
    cad_to_lsp_idx = [cad_names_15.index(j) for j in want_names]
    if is2DJoints:
        return cad_to_lsp_idx_swap
    else:
        return cad_to_lsp_idx

# > see https://github.com/achalddave/activity_detection/commit/c6174a173b859d9b7fafe9535e8ce1f0f254735a;
# > Projection formulas as described in 
# > https://groups.google.com/forum/#!msg/unitykinect/1ZFCHO9PpjA/1KdxUTdq90gJ.
# > Given (x,y,z) coordinates, converts that point into its x pixel number in the 2D image.
def xPixelFromCoords(x,y,z):
    kRealWorldXtoZ = 1.122133
    kResX = 640
    fCoeffX = kResX / kRealWorldXtoZ
    return int(fCoeffX * x / z) + (kResX / 2)

# Given (x,y,z) coordinates, converts that point into its y pixel number in the 2D image.
def yPixelFromCoords(x, y, z):
    kRealWorldYtoZ = 0.84176
    kResY = 480
    fCoeffY = kResY / kRealWorldYtoZ
    return int(kResY / 2) - (fCoeffY * y / z)

""" convert the above formulat to matrx multiplication format """
# M: matrix M, actually it's the production of intrisic K and extrisic E;
# here M is hard-coded as:
# cad-120: imgH = 480, imgW = 640;
# cad-60:  imgH = 240, imgW = 320;
# M = [ imgW/1.12,  0,           imgW/2
#        0,       -imgH/0.84,    imgH/2
#        0,        0,            1]
# * args: 
#       * pts: input 3D joints, 3 x N
#       * M: projection matrix, 3 x 3      
def getPixelValuesFromCoords(pts, M):
    proj_coords = np.dot(M, pts)
    proj_coords /= proj_coords[2]
    return proj_coords[:2] # 2 x N


# >> see: cad60/120 data format:
# Skeleton data consists of 15 joints. 
# There are 11 joints that have both joint orientation and joint position. 
# And, 4 joints that only have joint position. Each row follows the following format.
# Frame#, ORI(1),P(1),ORI(2),P(2),...,P(11),J(11),P(12),...,P(15)
def read_cad_txt_annotation(anno_fname):

    with open(anno_fname) as f:
        my_lines = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    annos = [ x.strip() for x in my_lines if 'END' not in x ]
    #print (annos[:1])
    # joints3d : 3 x joints_Num (i.e., 15 ) x frames_num 
    all_joints3d = np.zeros([3, 15, len(annos)], dtype = np.float32)
    
    """ all the indices """
    frame_idx = 0
    cad_joints_x_idx = [11] # the first joint_x;
    #There are 11 joints that have both joint orientation and joint position;
    for i in range(1, 11): # the remaining 10 joints of the 11 joints
        cad_joints_x_idx.append(11+ 14*i) # common difference = 14
    # 4 joints that only have joint position;
    for j in range(1, 5):
        cad_joints_x_idx.append(j*4 + 151) # 151 : the last joint_x from last step;

    #print ("[***] cad_joints_x_idx = {}".format(cad_joints_x_idx))

    all_fnames = []
    #********************************
    # For example: anno_fname = '/home/hmr/datasets/cad-60-120/cad-60/Person1/0512174930.txt'
    #********************************
    img_dir = anno_fname[
        anno_fname.rfind('cad-60-120') + 11 : -4]

    for i in range(0, len(annos)):
        annos_list = annos[i].split(',')
        #print (annos_list)
        for j in range(0, 15):
            tmp_x_idx = cad_joints_x_idx[j]
            all_joints3d[:, j, i] = [ float(tmp) for tmp in annos_list[tmp_x_idx: tmp_x_idx+3] ]
        img_name = '%s/RGB_%s.png' % (img_dir, annos_list[frame_idx])
        #print ("[**] extracted img_name = %s" % img_name)
        all_fnames.append(img_name)
    
    return all_joints3d, all_fnames 