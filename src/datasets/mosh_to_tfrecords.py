# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: mosh_to_tfrecords.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 01-08-2019
# @last modified: Thu 01 Aug 2019 10:27:25 AM EDT

"""
# added by Changjiang Cai:
Convert MoSh/data/mosh_gen/mosh_joints_annot.h5 to TFRecords.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import matplotlib; matplotlib.use('Agg') 
#import matplotlib.pyplot as plt

import sys
#import json,os,cv2
#from os import makedirs,listdir
from os.path import join, exists,isfile,isdir
#from time import time

import numpy as np
#import math

import tensorflow as tf
import h5py

from .common import convert_to_example_smpl_pose_lsp_joints3d_pairs

from absl import flags

""" flags """
flags.DEFINE_string('pathToMoSh', 'datasets/MoSh/data/mosh_gen', 'path to mosh_joints_annot.h5')

def _get_config():
    config = flags.FLAGS
    config(sys.argv)
    return config

def load_smpl_joints3d_pair_h5(pathToMoSh):
    fp=h5py.File(join(pathToMoSh,"mosh_joints_annot.h5"),"r")
    print (fp.keys())
    smplShapeParams=np.array(fp["shape"]) # (5988408, 10)
    smplPoseParams=np.array(fp["pose"]) # (5988408, 72)
    smplJoints=np.array(fp["joints"]) # (5988408, 24, 3)
    #num_mosh_examples = 5988408
    num_mosh_examples = smplShapeParams.shape[0]
    return num_mosh_examples, smplShapeParams, smplPoseParams, smplJoints


""" add the pair <pose/shape, joints3d from smpl layer> to tfrecord """
def add_smpl_joints3d_pair_to_tfrecord(
        frame_fname, 
        lsp_joints3d_smpl, 
        pose, 
        shape, 
        #gender_vect = np.array([.0,.0,1.0]).astype(np.float32), # means 'nuetral'
        root = np.array([.0, .0, .0]).astype(np.float32), 
        tf_writer = None
        ):
    example = convert_to_example_smpl_pose_lsp_joints3d_pairs(frame_fname,
            lsp_joints3d_smpl, root, pose, shape)
    
    tf_writer.write(example.SerializeToString())

    # Finally return how many were written.
    return 1 # 1 person added


def process_mosh(out_dir, num_imgs, num_shards):
    
    if not exists(out_dir):
        makedirs(out_dir)
        print ("mkdir %s" % out_dir)
    out_path = join(out_dir, '%06d.tfrecord')
    
    img_idx = 0
    # Count on shards:w
    fidx = 0
    isDisplay = False
    while img_idx < num_imgs:
        tf_filename = out_path % fidx
        print('Starting tfrecord file %s' % tf_filename)
        with tf.python_io.TFRecordWriter(tf_filename) as writer:
            # Count on total ppl in each shard
            num_ppl = 0
            while img_idx < num_imgs and num_ppl < num_shards:
                if img_idx % 50 == 0:
                    print('Reading img %d/%d' % (img_idx, num_imgs))
                frame_fname = ''
                cur_num_ppl = add_smpl_joints3d_pair_to_tfrecord(
                    frame_fname, 
                    lsp_joints3d_smpl, 
                    pose, 
                    shape, 
                    #gender_vect = np.array([.0,.0,1.0]).astype(np.float32), # means 'nuetral'
                    root = np.array([.0, .0, .0]).astype(np.float32), 
                    tf_writer = None)

                num_ppl += cur_num_ppl
                img_idx += cur_num_ppl
        fidx += 1



def main():
    config = _get_config()
    
    #print('Saving results to %s' % config.output_directory)
    
    #if not exists(FLAGS.output_directory):
    #    makedirs(FLAGS.output_directory)
    
    _, smplShapeParams, smplPoseParams, smplJoints = load_smpl_joints3d_pair_h5(config.pathToMoSh)
    print ('shape: smplShapeParams, smplPoseParams, smplJoints = {}, {}, {}'.format(
        smplShapeParams.shape, smplPoseParams.shape, smplJoints.shape))
     

if __name__ == '__main__':
    main()