# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: cp_cad120_gt_joints2d.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 30-07-2019
# @last modified: Wed 07 Aug 2019 11:15:22 AM EDT


import os,sys,fnmatch,random
import os.path as osp
from os.path import join
import numpy as np
from glob import glob


if __name__ == "__main__":
    
    if 0: # copy annotation txt files to rgb images dir;
        cad_120_root = '/mnt/interns/changjiang/PatientPositioning/Datasets/cad-60-120/cad-120'
        rgbs = ['Subject1_rgbd_images', 'Subject3_rgbd_images', 'Subject4_rgbd_images', 'Subject5_rgbd_images' ]
        annos = ['Subject1_annotations', 'Subject3_annotations', 'Subject4_annotations', 'Subject5_annotations' ]

        for i in range(0, len(rgbs)):
        #for i in range(0, 1):
            one_level_dirs = [ d for d in os.listdir(join(cad_120_root, rgbs[i])) if osp.isdir(join(cad_120_root,rgbs[i], d)) ]
            #print (one_level_dirs)	
            #sys.exit()
            cur_dir = join(cad_120_root, rgbs[i])
            cur_anno_dir = join(cad_120_root, annos[i])
            
            for dir_n1 in one_level_dirs:
                two_level_dirs = os.listdir(os.path.join(cur_dir, dir_n1))
                for dir_n2 in two_level_dirs:
                    src = join(cur_anno_dir, dir_n1, dir_n2 + '.txt')
                    dst = join(cur_dir, dir_n1)
                    cmd = 'cp {} {}'.format(src, dst)
                    print (cmd)
                    os.system(cmd)
    if 0: # copy some cad60 rgb images to smpl obj mesh dir;
        cad_root = '/mnt/interns/changjiang/PatientPositioning/Datasets/cad-60-120/cad-60'
        mesh_root = '/mnt/interns/changjiang/PatientPositioning/Datasets/cad-60-120/cad_60_120_smpl_model_samples/cad-060-via-model_cpt495000'
        mesh_files = [ t for t in os.listdir(mesh_root) if '_mesh.obj' in t]
        for m in mesh_files:
            src = join(cad_root, m.split('-')[0], m.split('-')[1],  m.split('-')[2][0:-24] + '.png')
            dst = join(mesh_root, m[0:-24] + '.png')
            cmd = 'cp {} {}'.format(src, dst)
            print (cmd)
            os.system(cmd)
    
    if 1: # copy some cad120 rgb images to smpl obj mesh dir;
        cad_root = '/mnt/interns/changjiang/PatientPositioning/Datasets/cad-60-120/cad-120'
        mesh_root = '/mnt/interns/changjiang/PatientPositioning/Datasets/cad-60-120/cad_60_120_smpl_model_samples/cad-120-via-model_cpt495000'
        subs = [1,3,4,5]
        for s in subs:
            cur_sub = 'Subject%d_rgbd_images' % s

            mesh_files = [ t for t in os.listdir(join(mesh_root, cur_sub)) if '_mesh.obj' in t]
            for m in mesh_files:
                src = join(cad_root, cur_sub, m.split('-')[0], m.split('-')[1],  m.split('-')[2][0:-24] + '.png')
                dst = join(mesh_root, cur_sub, m[0:-24] + '.png')
                cmd = 'cp {} {}'.format(src, dst)
                print (cmd)
                os.system(cmd)
