# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: test_tf_smpl_py_smpl.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 10-07-2019
# @last modified: Wed 10 Jul 2019 02:08:02 AM EDT

import matplotlib.pyplot as plt
from src.smpl.smpl_webuser.serialization import load_model
import numpy as np
from src.util import surreal_in_extrinc as surreal_util
import cv2

import tensorflow as tf
from src.tf_smpl.batch_smpl import SMPL

gt_poses = np.array([
        1.5320075750350952, 
        0.14028765261173248, 
        -2.6789824962615967, 
        0.22448143362998962, 
        0.06187523901462555, 
        0.07241860777139664, 
        0.14330032467842102, 
        -0.08564785122871399, 
        -0.04867864400148392, 
        0.09102527052164078, 
        -0.019132258370518684, 
        -0.016731375828385353, 
        -0.007750052027404308, 
        -0.15625706315040588, 
        -0.08045776933431625, 
        0.17839816212654114, 
        0.06272178143262863, 
        0.02093743160367012, 
        -0.0238698311150074, 
        -0.02493360824882984, 
        0.040423426777124405, 
        -0.16155442595481873, 
        0.1616159975528717, 
        0.11372498422861099, 
        -0.22674991190433502, 
        -0.11870049685239792, 
        0.05428570136427879, 
        0.025431444868445396, 
        -0.02710077539086342, 
        0.017528710886836052, 
        -0.06586545705795288, 
        0.14883504807949066, 
        -0.19256959855556488, 
        -0.010483330115675926, 
        0.08493982255458832, 
        0.0026347716338932514, 
        0.08071184903383255, 
        0.13719549775123596, 
        -0.03732795640826225, 
        -0.054000210016965866, 
        -0.15808750689029694, 
        -0.22886215150356293, 
        0.06829798221588135, 
        0.11018916964530945, 
        0.4616956412792206, 
        0.4153347909450531, 
        0.02039397694170475, 
        -0.01858796924352646, 
        -0.12124170362949371, 
        -0.7454847097396851, 
        -0.5246301889419556, 
        0.1557874083518982, 
        0.12163950502872467, 
        0.8464822173118591, 
        0.04726871848106384, 
        -1.9905153512954712, 
        0.9930220246315002, 
        0.15261900424957275, 
        0.3822908401489258, 
        -0.1001385748386383, 
        -0.3046434819698334, 
        -0.17382748425006866, 
        0.8061538934707642, 
        -0.012420877814292908, 
        -0.011445901356637478, 
        -0.2758418917655945, 
        0.25503501296043396, 
        0.008094564080238342, 
        -0.1027793362736702, 
        0.09171544015407562, 
        0.08430416882038116, 
        0.061581265181303024
    ])
    
gt_shape = np.array([
        -2.0501492023468018, 
        -0.31580671668052673, 
        -0.48987817764282227, 
        0.9297548532485962, 
        0.045968618243932724, 
        -0.17658251523971558, 
        -1.4542434215545654, 
        0.8400909900665283, 
        -1.3031471967697144, 
        0.4580021798610687
    ])

extrinsic = np.array([[ 0.,0.,-1., -1.929299], 
    [0.,1., 0., 0.98388302], [-1., 0., 0.,  6.74562216]])
intrinsic = np.array([[600.0, 0., 160.,], [0., 600., 120.], [0., 0., 1.]])

rgb = cv2.imread("./data/c0001_frame_000.jpg")
rgb = rgb[:,:, [2,1,0]]

pkl_fname = './models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
    
m = load_model( pkl_fname)

print ("[***] loading model %s" % pkl_fname)


m.pose[:] =  gt_poses
m.betas[:] = gt_shape
smpl_joints3D = m.J_transformed.r
batch_size=1
#surreal_util.draw_joints2D(
#                rgb,
#        #        # 24 joints
#                surreal_util.project_vertices(smpl_joints3D, intrinsic, extrinsic),
#                None, m.kintree_table, color = 'b')

#smpl_model_path = "./models/basicModel_m_lbs_10_207_0_v1.0.0.pkl,neutral_smpl_with_cocoplus_reg.pkl"
smpl_model_path = "./models/neutral_smpl_with_cocoplus_reg.pkl,"
shapes_gt_pl = tf.placeholder(tf.float32, shape = [batch_size, 10])
poses_gt_pl = tf.placeholder( tf.float32, shape = [batch_size, 72])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    smpl = SMPL(smpl_model_path, joint_type='lsp')
    verts_gt, Js_gt, _, J_transformed_gt = smpl(shapes_gt_pl, poses_gt_pl, 
             get_skin = True, trans = None, idx = 1)
    
    #verts, Js, _, J_transformed = smpl(shapes_gt_pl, self.poses_gt_pl, 
    #         get_skin = True, trans = None, idx = 1)
    feed_dict = {
            shapes_gt_pl : np.expand_dims(gt_shape, 0),
            poses_gt_pl: np.expand_dims(gt_poses,0),
            }
    J_transformed_np = sess.run(J_transformed_gt,  feed_dict)
    print ("J_transformed_np shape ", J_transformed_np.shape)





