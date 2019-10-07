# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: cad60_120_visualization.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 07-10-2019
# @last modified: Mon 07 Oct 2019 05:47:10 PM EDT

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib; matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import sys,json,os,cv2
from os import makedirs
from os.path import join, exists
from time import time

import numpy as np
import glob

import src.pfmutil as pfm

from src.util.cad_60_120_util import (get_cad_2_lsp_idx,
        getPixelValuesFromCoords,
        read_cad_txt_annotation
        )

#from src.datasets.cad60_120_to_tfrecords import (
#    parse_people_via_joints,
#)


def _joints_is_in_image(pt, imgH = 240, imgW = 320):
    x = pt[0]
    y = pt[1]
    return x * (imgW - x) > 0 and y * (imgH - y)


def parse_people_via_joints(lsp_joints2d, imgH, imgW):
    '''
    Parses people from joints 2d gt annotation.
    Input:
      joints is 3 x 14 (lsp order), i.e., (x,y,vis) x l4
    '''
    people = []
    # joints is 3 x 14 (lsp order)
    # All kps are visible.
    min_pt = np.min(lsp_joints2d[:2, :], axis=1)
    max_pt = np.max(lsp_joints2d[:2, :], axis=1)

    success = _joints_is_in_image(min_pt, imgH, imgW) and _joints_is_in_image(max_pt, imgH, imgW)
    if success:
        person_height = np.linalg.norm(max_pt - min_pt)
        center = (min_pt + max_pt) / 2.
        scale = 150. / person_height
        people.append((scale, min_pt, max_pt, center))
    return people


cad_small_subjects = [
    #'cad-60-small-1-img-person1', # 1
    #'cad-60-small-1-img-person2', # 1
    #'cad-60-small-2-imgs-person2' # 2
    #'cad-120-small-1-img-sub1', # 1
    #'cad-120-small-1-img-sub3', # 1
    #'cad-120-small-1-img-sub4', # 1
    'cad-60-small-5-imgs-person1', # 5
    #'cad-60-small-3-diff-imgs-person1', # 3
    ]

cad_eval_subjects = [ 
        'cad-120/Subject4_rgbd_images',
        ]

cad_subjects = [ 
        'cad-60/Person1', 'cad-60/Person2', 'cad-60/Person3', 'cad-60/Person4',
        'cad-120/Subject1_rgbd_images', 'cad-120/Subject3_rgbd_images',
        'cad-120/Subject4_rgbd_images','cad-120/Subject5_rgbd_images',
        ]

cad_subjects_dict = {
    # cad 60
    'cad-60/Person1': {'gender': 'm', 'sampleNum': 19851, 'sampleNum_mpjpe_pa_thre70': 5980},
    'cad-60/Person2': {'gender': 'f', 'sampleNum': 18675, 'sampleNum_mpjpe_pa_thre70': 3953},
    'cad-60/Person3': {'gender': 'f', 'sampleNum': 19465, 'sampleNum_mpjpe_Pa_thre70': 3048},
    'cad-60/Person4': {'gender': 'm', 'sampleNum': 22321, 'sampleNum_mpjpe_pa_thre70': 6721},
    # cad 120
    'cad-120/Subject1_rgbd_images': {'gender': 'f', 'sampleNum': 15227, 
                                     'sampleNum_mpjpe_pa_thre70': 2883},
    'cad-120/Subject3_rgbd_images': {'gender': 'f', 'sampleNum': 17277, 
                                     'sampleNum_mpjpe_pa_thre70': 2358},
    'cad-120/Subject4_rgbd_images': {'gender': 'm', 'sampleNum': 13281, 
                                     'sampleNum_mpjpe_pa_thre70': 354},
    'cad-120/Subject5_rgbd_images': {'gender': 'm', 'sampleNum': 19348, 
                                     'sampleNum_mpjpe_pa_thre70': 1732}, 
    # cad60/120: Total = 145445 images; Total sampleNum_mpjpe_thre70 = 27029 images;
    
    # small dataset, for debugging
    'cad-60-small-1-img-person1': {'gender': 'm', 'sampleNum': 1, },
    'cad-60-small-1-img-person2': {'gender': 'f', 'sampleNum': 1, },
    'cad-60-small-2-imgs-person2': {'gender': 'f', 'sampleNum': 2, },
    'cad-120-small-1-img-sub1' : {'gender': 'f', 'sampleNum': 1, },
    'cad-120-small-1-img-sub3' : {'gender': 'f', 'sampleNum': 1, },
    'cad-120-small-1-img-sub4' : {'gender': 'm', 'sampleNum': 1, },
    'cad-60-small-5-imgs-person1': {'gender': 'm', 'sampleNum': 5, },
}



def visualize_cad_gt3D_2D(
        lsp_joints2d,
        image_name,
        isSavepfm, 
        out_dir, 
    ):

    """
    visualization projected 2D joints from 3D GT joints of CAD-60&120;
    args:
        lsp_joints2d: projected 2D joints, in shape [2,14]
    Return: # of persons parsed;
    """
    print ('image_name is %s' %image_name)
    image = cv2.imread(image_name)
    imgH, imgW = image.shape[:2]
    people = parse_people_via_joints(lsp_joints2d, imgH, imgW)

    if len(people) == 0:
        print ("found 0 person!!!")
        return 0
    
    for scale, min_pt_orig, max_pt_orig, center_orig in people:
        cv2.rectangle(image, (int(min_pt_orig[0]), int(min_pt_orig[1])), 
                             (int(max_pt_orig[0]), int(max_pt_orig[1])), 
                             (255,0,0), 2)
        print ("[**] orig min_pt is {}, max_pt is {}, center is then {}".format(
            min_pt_orig, max_pt_orig, center_orig))
        
        cv2.circle(image, (int (min_pt_orig[0]), int(min_pt_orig[1])), radius = 2, color = (0,0,255), thickness = 1)
        cv2.circle(image, ( int(max_pt_orig[0]), int(max_pt_orig[1])), radius = 2, color = (0,255,0), thickness = 1)

        lsp_names = [
               'R Ankle', 'R Knee', 'R Hip', 'L Hip', 'L Knee', 'L Ankle', 'R Wrist',
               'R Elbow', 'R Shoulder', 'L Shoulder', 'L Elbow', 'L Wrist', 'Neck', 'Head']
        # left joints using red color 
        left_joints = [3,4,5,9,10,11]
        # right joints using green color 
        right_joints = [0,1,2,6,7,8]
        # middle joints using blue color 
        middle_joints = [12,13]
        
        for nk in left_joints:
            x = int(lsp_joints2d[0,nk])
            y = int(lsp_joints2d[1,nk])
            # cv2: in BGR color order
            cv2.circle(image, (x,y), radius = 2, color = (0,0,255), thickness = 2)
            print ('in RGBD : joint {} {}, has y = {}, x = {}'.format(
                nk, lsp_names[nk], y, x))

        if 1:
            for nk in right_joints:
                x = int(lsp_joints2d[0,nk])
                y = int(lsp_joints2d[1,nk])
                cv2.circle(image, (x, y), radius = 2, color = (0,255,0), thickness = 2)
                print ('in RGBD : joint {} {}, has y = {}, x = {}'.format(
                    nk, lsp_names[nk], y, x))
        for nk in middle_joints:
            cv2.circle(image,( int(lsp_joints2d[0,nk]), int(lsp_joints2d[1, nk])), 
            radius = 2, color = (255,0,0), thickness = 2)

        tmp_name = image_name.split("/")[-3] + '-' + image_name.split("/")[-2] + '-' + image_name.split("/")[-1]
        tmp_name_img = join(out_dir, "{}_img_orig.pfm".format(tmp_name[:-4]))
        pfm.save(tmp_name_img, image[:,:,(2,1,0)].astype(np.float32))
        print ('saved %s' % tmp_name_img)

    # Finally return how many were written.
    return len(people)


def main(args):
    
    print('Saving results to %s' % args.output_directory)
    if not exists(args.output_directory):
        makedirs(args.output_directory)
    
    print ("[***] task_type_cad = {}".format(args.task_type_cad))
    
    
    if args.task_type_cad == 'visualize_joints_annotation_from_cad_gt':
        cad_to_lsp_idx_2d = get_cad_2_lsp_idx(is2DJoints = True)
        #cad_to_lsp_idx_3d = get_cad_2_lsp_idx(is2DJoints = False)

        # here M is hard-coded as:
        M_cad_60 =  np.array([[320.0/1.12,   .0,       320.0/2.0], 
                              [     .0,   -240.0/0.84, 240.0/2.0],
                              [.0,           .0,             1.0]])
        
        M_cad_120 =  np.array([[640.0/1.12,   .0,       640.0/2.0], 
                              [     .0,   -480.0/0.84, 480.0/2.0],
                              [.0,           .0,             1.0]])
        
        cam_cad_120 = np.array([(640.0/1.12 + 480.0/0.84)*0.5, 640.0/2.0, 480.0/2.0])
        cam_cad_60 = np.array([(320.0/1.12 + 240.0/0.84)*0.5, 320.0/2.0, 240.0/2.0])
        
        
        for subject in cad_small_subjects:
        #for subject in cad_eval_subjects:
        #for subject in cad_subjects:
            
            img_dir = join(args.img_directory, subject)
            out_dir = join(args.output_directory, subject)
            if not exists(out_dir):
                makedirs(out_dir)
                print ("mkdir %s" % out_dir)
    
            # 3 x 15*N ==> 3 x 15 x N;
            if 'cad-60' in subject:
                M = M_cad_60
                cam = cam_cad_60
                print ('[**] using M from cad-60')
            elif 'cad-120' in subject:
                M = M_cad_120
                cam = cam_cad_120
                print ('[**] using M from cad-120')
            
            
            print ("[***] img_dir = {}".format(img_dir))
            
            if 'cad-60' in subject:
                all_anno_txt_files = [ f for f in glob.glob(join(img_dir, "*.txt"))]
                print (all_anno_txt_files)
                all_anno_txt_files = [ f for f in glob.glob(join(img_dir, "*.txt")) if os.path.isdir( 
                    f[0:-4])]
            elif 'cad-120' in subject:
                all_anno_txt_files = [ f for f in glob.glob(join(img_dir, "*/*.txt")) ]
                
            print ('[***] anno_fnames {}'.format(all_anno_txt_files))
            img_num_sum = 0 
            for anno_fname in all_anno_txt_files:
                print ('[***] anno_fname {}'.format(anno_fname))
                # joints3d : 3 x joints_Num (i.e., 15 ) x frames_num
                joints3d, img_fnames = read_cad_txt_annotation(anno_fname)
                img_num = len(img_fnames) # all image included in this current '.txt' file;
                img_num_sum += img_num
                """ projection from 3D joints to 2d joints """
                #NOTE:
                # 15: cad 60/120 datasets has 15 joints;
                # 14: lsp has 14 joints;
                cad_joints2d = np.reshape(getPixelValuesFromCoords(
                    np.reshape(joints3d, [3, 15*img_num]), M), [2, 15, img_num])
                lsp_joints2d = np.ones([3, 14, img_num], dtype = np.float32) # (0:x, 1: y, 2: vis=1)
                lsp_joints2d[:2,:, :] = cad_joints2d[:, cad_to_lsp_idx_2d, :] #:2 means x, y
                for i in range(0, img_num):
                    image_name = join(args.img_directory, img_fnames[i]) 
                    visualize_cad_gt3D_2D(
                            lsp_joints2d[:2,:,i], 
                            image_name, 
                            isSavepfm =True,
                            out_dir = out_dir
                            )


            
            

    else:
        print("ERROR!!! args.task_type_cad is %s. It should be 'joints_annotation_from_cad_gt' or 'joints_annotation_from_densepose' or 'visualize_joints_annotation_from_cad_gt'"
                    % args.task_type_cad)


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type_cad', dest = 'task_type_cad', type = str, default = 'visualize_joints_annotation_from_cad_gt')
    parser.add_argument('--img_directory', dest = 'img_directory', type = str, default = '/usr/local/ccjData/datasets/cad-60-120/')
    parser.add_argument('--output_directory', dest = 'output_directory', type = str, default = './results/cad_60_tmp')
    args = parser.parse_args()
    
    main(args)
    
    
