# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: pose_perceptron.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 22-07-2019
# @last modified: Mon 22 Jul 2019 06:17:15 PM EDT


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib; matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


import tensorflow as tf
from absl import flags
import os
from .data_loader import DataLoader
from .data_loader import num_examples
import os.path as osp
from datetime import datetime
import sys
from time import time
import numpy as np


from .tf_smpl.batch_smpl import SMPL

from src.util.smpl_webuser.serialization import load_model

from .tf_smpl.batch_lbs import batch_rodrigues
import cv2
from src.util.surreal_in_extrinc import get_lsp_idx_from_smpl_joints
from src.datasets.mosh_to_tfrecords import load_smpl_joints3d_pair_h5
from src.util.cad_60_120_util import get_cad_2_lsp_idx, read_cad_txt_annotation
from src.smpl_perceptron import (One_hidden_layer_joints3d_2_SmplRegressor, 
                                Three_hidden_layer_joints3d2LspSmplRegressor,
                                Seven_hidden_layer_joints3d2LspSmplRegressor)

from .config import get_config

""" flags """
# > see : https://github.com/abseil/abseil-py/issues/36;
#_conflict_flag_names = ['smpl_model_path', 'result_dir', 
#                        'datasets', 'data_dir', 'split',
#                        'pretrained_model_path', 'batch_size',
#                        'log_dir', 'model_dir', 'epoch', 'joint_type',
#                        ]

#for name in list(flags.FLAGS):
#for name in _conflict_flag_names:
#    if name in list(flags.FLAGS):
#        delattr(flags.FLAGS, name)

#MY_FLAGS_posePercep = flags.FlagValues()  # This is a separate "blank" instance different from the "global" flags.FLAGS instance.

"""
flags.DEFINE_string('smpl_model_path', 
        './models/basicModel_f_lbs_10_207_0_v1.0.0.pkl,basicModel_m_lbs_10_207_0_v1.0.0.pkl,neutral_smpl_with_cocoplus_reg.pkl', 
        'path to the neurtral smpl model')

flags.DEFINE_list('datasets', ['surreal_smpl_joints3d_pair'], 'datasets to use for training')
flags.DEFINE_string('data_dir', 'datasets/tf_datasets/', 'Where to save training models')
flags.DEFINE_string('split', 'train', 'train or valid to filter the tfrecord files whose names starting with this splt flag')
flags.DEFINE_string('pretrained_model_path', None,
                    'if not None, fine-tunes from this ckpt')
flags.DEFINE_integer('batch_size', 8,
                     'Input image size to the network after preprocessing')
flags.DEFINE_integer('save_models_step', 4000, 'frequency of model saving')
flags.DEFINE_integer('run_validation_step', 200, 'frequency of running validation')
flags.DEFINE_string('log_dir', 'logs', 'Where to save training models')
flags.DEFINE_string('model_dir', None, 'Where model will be saved -- filled automatically')
flags.DEFINE_string('method_name', 'joints3d_to_smpl_regressor', 'some unique name to tell from others')

flags.DEFINE_string('smpl_regressor_model_type', '7_hidden_layer_perceptron', 
                    'could be 1_, 3_, or 7_hidden_layer_perceptron')

flags.DEFINE_string('result_dir', './results/tmp', 'Where results will be saved')
flags.DEFINE_integer('epoch', 100, '# of epochs to train')

# Hyper parameters:
flags.DEFINE_float('lr', 0.001, 'learning rate')
flags.DEFINE_float('wd', 0.0001, 'weight decay')

flags.DEFINE_float('weight_pose', 1.0, 'weight loss of pose')
flags.DEFINE_float('weight_shape', 1.0, 'weight loss of shape')
flags.DEFINE_float('weight_joints3d', 100.0, 'weight loss of joints3d')

# Network Parameters
flags.DEFINE_integer('num_hidden_1', 256*2, '1st layer number of neurons')
flags.DEFINE_integer('num_hidden_2', 256*2, '1st layer number of neurons')
flags.DEFINE_integer('num_input', 24*3, 'data input (24 3d joints shape: 24*3)')
flags.DEFINE_integer('num_output', 23*3 + 10, 
             'output of smpl poses for 23 joints, i.e, excluding the root joint, plus the shape')

# has smpl_layer in this network or not
flags.DEFINE_boolean( 'has_smpl_layer', True, 'if set, network has smpl_layer')
#flags.DEFINE_boolean( 'is_training', True, 'if set, network training, otherwise for evaluation or testing')
#flags.DEFINE_boolean( 'is_testing', False, 'if set, network testing')

flags.DEFINE_boolean( 'isPoseToRotation', True, 'if set, change pose axis-angle to rotation matrx when calculating loss')

#joint types:
# * lsp = 14 joints;
# cocoplus = 14 lsp joints + 5 face points = 19 joints;
# smpl: = 24 joints;
flags.DEFINE_string(
    'joint_type', 'smpl',
    'could be cocoplus (19 keypoints) or lsp 14 keypoints, or smpl 24 joints, all of them are returned by SMPL layer')

flags.DEFINE_string(
    'task_type', 'train', "could be 'train', 'evaluation', or 'test' ")

flags.DEFINE_string(
    'toy_data_type', 'cad-120-small-1', "including: cad-120-small-1, mosh-hd5-small-1-x, cad-any-1-sample, etc")
"""


""" utility functions """
def compute_L2_loss(params_pred, params_gt):
    with tf.name_scope("L2_loss", [params_pred, params_gt]):
        res = tf.losses.mean_squared_error(params_gt, params_pred) * 0.5
    return res

def compute_L1_loss(params_pred, params_gt):
    with tf.name_scope("L1_loss", [params_pred, params_gt]):
        res = tf.losses.absolute_difference(params_gt, params_pred)
    return res

def _get_config(my_flag_instance):
    config = my_flag_instance.FLAGS
    config(sys.argv)
    time_str = datetime.now().strftime("%b%d_%H%M")
    config.model_dir = osp.join(config.log_dir, config.method_name + "_"+time_str)
    return config

def _make_dir(src):
     if not osp.exists(src):
        os.makedirs(src)
        print ('makedirs %s' % src)

# Single Layer Perceptron
class SMPLTrainer(object):
    def __init__ (self, sess, config, data_loader, valid_data_loader = None):
        # Optimizer, learning rate
        self.sess = sess
        self.is_first = True
        self.config = config
        self.isAxisAnglePose2Rotation = config.isPoseToRotation
        #self.is_testing = config.is_testing
        #self.is_training = config.is_training
        self.task_type = config.task_type
        if self.task_type not in ['train', 'evaluation', 'test']:
            print('BAD!! Unknown task type: %s, it must be either "train",  "evaluation" or "test"' 
                   % self.task_type)
            import ipdb
            ipdb.set_trace()
        
        self.has_smpl_layer = config.has_smpl_layer
        self.result_dir = config.result_dir
        self.smpl_regressor_model_type = config.smpl_regressor_model_type
        if self.smpl_regressor_model_type == '1_hidden_layer_perceptron':
            self.perceptron = One_layer_joints3d_2_SmplRegressor(self.config)
            print ("[***] smpl_regressor_model_type : 1_hidden_layer_perceptron")
        
        elif self.smpl_regressor_model_type == '3_hidden_layer_perceptron':
            self.perceptron = Three_hidden_layer_joints3d2LspSmplRegressor()
            print ("[***] smpl_regressor_model_type : 3_hidden_layer_perceptron")
        
        elif self.smpl_regressor_model_type == '7_hidden_layer_perceptron':
            self.perceptron = Seven_hidden_layer_joints3d2LspSmplRegressor()
            print ("[***] smpl_regressor_model_type : 7_hidden_layer_perceptron")
        
        self.joint_type = config.joint_type
        if self.joint_type == 'lsp':
            self.joint_num = 14
        elif self.joint_type == 'smpl':
            self.joint_num = 24
        elif self.joint_type == 'cocoplus':
            self.joint_num == 19
        print ("[***] joint_type = %s, joint_num = %d"  % (self.joint_type, self.joint_num))

        _make_dir(self.result_dir)

        self.weight_pose = config.weight_pose
        self.weight_shape = config.weight_shape
        self.weight_joints3d = config.weight_joints3d
        self.smpl_model_path = config.smpl_model_path
        if valid_data_loader is not None:
            self.hasValidation = True
        else:
            self.hasValidation = False

        self.pretrained_model_path = config.pretrained_model_path
        self.model_dir = config.model_dir

        self.lr = config.lr
        # Weight decay
        self.wd = config.wd
        self.batch_size = config.batch_size
        self.save_models_step = config.save_models_step
        #self.run_validation_step = config.run_validation_step

        print ("[****] Batch size = %d" % self.batch_size)
        
        #NOTE:
        
        if self.task_type in  ['train']: # training
            num_images = num_examples(config.datasets) # number of images;
            self.num_itr_per_epoch = num_images / self.batch_size
            self.max_epoch = config.epoch
            print ("[**] has total num_images for training = %d, num_itr_per_epoch = %d" %(
                   num_images, self.num_itr_per_epoch))
        
        if self.task_type in  ['train', 'evaluation']: # training or evaluation
            self.img_name_loader = data_loader['img_name'] 
            self.smpl_shape_loader = data_loader['shape']
            self.smpl_gender_loader = data_loader['gender']
            self.smpl_pose_loader = data_loader['pose']
            self.joints_3d_loader = data_loader['joints3d_from_smpl']
            #self.fname = data_loader['fname']
            if self.hasValidation:
                self.valid_smpl_pose_loader = valid_data_loader['pose']
                self.valid_smpl_shape_loader = valid_data_loader['shape']
                self.valid_smpl_gender_loader = valid_data_loader['gender']
                self.valid_joints_3d_loader = valid_data_loader['joints3d_from_smpl']

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.build_train_model()

            self.saver = tf.train.Saver()
            print ("[ *** ] self.saver Done !")

        if self.task_type in  ['test']: # testing, we use tf.placeholder for data feeding;
            print ("[**] build_test_model() ...")
            self.joints_3d_pl = tf.placeholder(tf.float32, shape= [self.batch_size, self.joint_num*3])
            self.gender_pl = tf.placeholder(tf.float32, shape= [self.batch_size, 3])
            self.build_test_model()

        if self.has_smpl_layer:
            self.smpl = SMPL(
                extract_female_male_neutral_model_paths(self.smpl_model_path, False), 
                joint_type = self.joint_type
            )

    # build test model for inference;
    def build_test_model(self):
        self.smpl_pred = self.perceptron.build_model(self.joints_3d_pl)
        if self.smpl_regressor_model_type in ['3_hidden_layer_perceptron', 
            '7_hidden_layer_perceptron']:
            # model output 24 joints, but here we extract the last 23 joints pose only;
            self.smpl_pred = self.smpl_pred[:, 3:]
        
        self.pose_pred = self.smpl_pred[:, 0:23*3] # pose of 23 joints;
        self.shape_pred = self.smpl_pred[:, 23*3:23*3+10] # shape;

        if self.has_smpl_layer:
            self.smpl = SMPL(
                extract_female_male_neutral_model_paths(self.smpl_model_path, False), 
                joint_type = self.joint_type
            )

            rest_root_pose = tf.reshape(tf.constant([np.pi, .0, .0], tf.float32), [1, 3])
            batch_pose_with_rest_root = tf.concat(
               [tf.tile(rest_root_pose, [self.batch_size, 1]), 
                self.pose_pred], axis = 1)
            self.pred_joints3d, _ = self.smpl(
                beta = self.shape_pred, 
                theta = batch_pose_with_rest_root,
                get_skin=False, 
                gender = self.gender_pl)
        

    
    # load existing models 
    def load(self, checkpoint_dir):
        if checkpoint_dir is None or checkpoint_dir == '':
            print ("checkpoint_dir is None or empty string !!! ")
            return False
        
        print(" [*] Reading checkpoint %s" %checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            counter=int(ckpt.model_checkpoint_path.split('-')[-1].split('.')[0])
            print("Restored step: " + str( counter))
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print("Fine-tuning from %s" % checkpoint_dir)
            return True
        else:
            return False
    

    # Construct model, and define loss and optimizer
    def build_train_model(self):
        self.smpl_pred = self.perceptron.build_model(tf.reshape(self.joints_3d_loader, [-1, self.joint_num*3]))
        self.pose_pred = self.smpl_pred[:, 0:23*3] # the first section is pose_23;
        self.shape_pred = self.smpl_pred[:, 23*3:23*3 + 10] # the second section is shape;
        
        if self.has_smpl_layer:
            batch_pose_with_root_from_gt = tf.concat(
                [self.smpl_pose_loader[:,0:3], 
                self.pose_pred[:,:]], axis = 1)
            #verts, Js, _ , _ = self.smpl(shapes, poses, get_skin=True, gender = gt_gender)
            _, self.Js, self.Rs_pred, _ = self.smpl(
                beta = self.shape_pred, 
                theta = batch_pose_with_root_from_gt,
                get_skin=True, 
                gender = self.smpl_gender_loader)
             
            self.loss_l2_joints3d = compute_L2_loss(
                tf.reshape(self.Js, [-1, self.joint_num*3]),
                tf.reshape(self.joints_3d_loader, [-1, self.joint_num*3])
                )
            
            self.loss_l1_joints3d = compute_L1_loss(
                tf.reshape(self.Js, [-1, self.joint_num*3]),
                tf.reshape(self.joints_3d_loader, [-1, self.joint_num*3])
                )

        """ using rotation matrix or axis-angle pose for 'pose'-loss calculation """
        if not self.isAxisAnglePose2Rotation:
            Rs_gt = tf.reshape(batch_rodrigues(self.smpl_pose_loader), [-1, 24*3*3])
            # Rs_pred also use the same root_gt as Rs_gt;
            self.loss_l2_pose = compute_L2_loss(tf.reshape(self.Rs_pred, [-1, 24*3*3]), Rs_gt)
            self.loss_l1_pose = compute_L1_loss(tf.reshape(self.Rs_pred, [-1, 24*3*3]), Rs_gt)
        else:
            self.loss_l2_pose = compute_L2_loss(self.pose_pred, self.smpl_pose_loader[:, 3:])
            self.loss_l1_pose = compute_L1_loss(self.pose_pred, self.smpl_pose_loader[:, 3:])


        self.loss_l2_shape = compute_L2_loss(self.shape_pred, self.smpl_shape_loader)
        self.loss_l1_shape = compute_L1_loss(self.shape_pred, self.smpl_shape_loader)
        
        

        self.loss_l2 = self.weight_pose*self.loss_l2_pose + self.weight_shape * self.loss_l2_shape
        self.loss_l1 = self.weight_pose*self.loss_l1_pose + self.weight_shape * self.loss_l1_shape

        if self.has_smpl_layer:
            self.loss_l2 += self.weight_joints3d* self.loss_l2_joints3d
            self.loss_l1 += self.weight_joints3d* self.loss_l1_joints3d
        """ using L2 loss for optimization """
        self.loss = self.loss_l2

        """ forward operation on validataion data """ 
        if self.is_training and self.hasValidation:
            smpl_pred_valid = p.build_multilayer_perceptron(tf.reshape(self.valid_joints_3d_loader, [-1, 24*3]))
            pose_pred_valid = smpl_pred_valid[:, 0:23*3] # the first section is pose_23;
            shape_pred_valid = smpl_pred_valid[:, 23*3:23*3 + 10] # the second section is shape;
            if self.has_smpl_layer:
                batch_pose_valid_with_root_from_gt = tf.concat(
                    [self.valid_smpl_pose_loader[:,0:3], 
                     pose_pred_valid[:,:]], axis = 1)
                #verts, Js, _ , _ = self.smpl(shapes, poses, get_skin=True, gender = gt_gender)
                _, Js_valid, self.Rs_pred_valid, _ = self.smpl(
                    beta = shape_pred_valid, 
                    theta = batch_pose_valid_with_root_from_gt,
                    get_skin=True, 
                    gender = self.valid_smpl_gender_loader)
                
                self.loss_l2_joints3d_valid = compute_L2_loss(
                    tf.reshape(Js_valid, [-1, self.joint_num*3]),
                    tf.reshape(self.valid_joints_3d_loader, [-1, self.joint_num*3])
                    )
                
                self.loss_l1_joints3d_valid = compute_L1_loss(
                    tf.reshape(Js_valid, [-1, self.joint_num*3]),
                    tf.reshape(self.valid_joints_3d_loader, [-1, self.joint_num*3])
                    )

            if not self.isAxisAnglePose2Rotation:
                Rs_gt_valid = tf.reshape(batch_rodrigues(self.valid_smpl_pose_loader), [-1, 24*3*3])
                # Rs_pred also use the same root_gt as Rs_gt;
                self.loss_l2_pose_valid = compute_L2_loss(tf.reshape(self.Rs_pred_valid, [-1, 24*3*3]), Rs_gt_valid)
                self.loss_l1_pose_valid = compute_L1_loss(tf.reshape(self.Rs_pred_valid, [-1, 24*3*3]), Rs_gt_valid)
            else:
                self.loss_l2_pose_valid = compute_L2_loss(pose_pred_valid, self.valid_smpl_pose_loader[:, 3:])
                self.loss_l1_pose_valid = compute_L1_loss(pose_pred_valid, self.valid_smpl_pose_loader[:, 3:])
            
            self.loss_l2_shape_valid = compute_L2_loss(shape_pred_valid, self.valid_smpl_shape_loader)
            self.loss_l1_shape_valid = compute_L1_loss(shape_pred_valid, self.valid_smpl_shape_loader)
            
            self.loss_l2_valid = self.weight_pose*self.loss_l2_pose_valid + self.weight_shape * self.loss_l2_shape_valid
            self.loss_l1_valid = self.weight_pose*self.loss_l1_pose_valid + self.weight_shape * self.loss_l1_shape_valid

            if self.has_smpl_layer:
                self.loss_l2_valid += self.weight_joints3d* self.loss_l2_joints3d_valid
                self.loss_l1_valid += self.weight_joints3d* self.loss_l1_joints3d_valid
            
            self.loss_valid = self.loss_l2_valid


        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_opt = self.optimizer.minimize(self.loss, global_step = self.global_step)
        if self.is_training:
            self.setup_summaries()
        print('Done initializing trainer!')

    def setup_summaries(self):
        # Prepare Summary
        always_report = [
            tf.summary.scalar("loss/lossL2", self.loss_l2),
            tf.summary.scalar("loss/lossL1", self.loss_l1),
            # seperate L2 loss
            tf.summary.scalar("loss/lossL2_pose", self.loss_l2_pose),
            tf.summary.scalar("loss/lossL2_shape", self.loss_l2_shape),
            tf.summary.scalar("loss/lossL2_joints3d", self.loss_l2_joints3d),
            # seperate L1 loss
            #tf.summary.scalar("loss/lossL1_pose", self.loss_l1_pose),
            #tf.summary.scalar("loss/lossL1_shape", self.loss_l1_shape),
            #tf.summary.scalar("loss/lossL1_joints3d", self.loss_l1_joints3d),
            # total loss used during training
            #tf.summary.scalar("loss/loss", self.loss),
        ]
        self.summary_op_always = tf.summary.merge(always_report)
        
        if self.hasValidation:
            # for validation data
            always_report_valid = [
                tf.summary.scalar("loss/valid_lossL2", self.loss_l2_valid),
                tf.summary.scalar("loss/valid_lossL1", self.loss_l1_valid),
                # seperate L2 loss
                tf.summary.scalar("loss/valid_lossL2_pose", self.loss_l2_pose_valid),
                tf.summary.scalar("loss/valid_lossL2_shape", self.loss_l2_shape_valid),
                tf.summary.scalar("loss/valid_lossL2_joints3d", self.loss_l2_joints3d_valid),
                # seperate L1 loss
                #tf.summary.scalar("loss/valid_lossL1_pose", self.loss_l1_pose_valid),
                #tf.summary.scalar("loss/valid_lossL1_shape", self.loss_l1_shape_valid),
                #tf.summary.scalar("loss/valid_lossL1_joints3d", self.loss_l1_joints3d_valid),
                # total loss used during training
                #tf.summary.scalar("loss/valid_loss", self.loss_valid)
            ]
            self.summary_op_always_valid = tf.summary.merge(always_report_valid)


    def train(self):
        # Initializing the variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)
        
        if self.load(self.pretrained_model_path):
            self.model_dir = self.pretrained_model_path
            print(" [*] Load SUCCESS! And setting model_dir to be this dir %s" % self.model_dir)
        else:      
            print(" Load failed...neglected")
            print(" Start Training...")
            _make_dir(self.model_dir)

        plot_train_dir = osp.join(self.model_dir, 'plot_train')
        plot_valid_dir = osp.join(self.model_dir, 'plot_valid')
        for tmp_dir in [plot_train_dir, plot_valid_dir]:
            _make_dir(tmp_dir)
        
        self.summary_writer_train = tf.summary.FileWriter(plot_train_dir)
        if self.hasValidation:      
            self.summary_writer_valid = tf.summary.FileWriter(plot_valid_dir)        

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners( sess = self.sess, coord=coord)


        fetch_dict = {
            "summary": self.summary_op_always,
            "summary_valid": self.summary_op_always_valid,
            "train_opt": self.train_opt,
            "step": self.global_step,
            "loss": self.loss,
            "loss_valid": self.loss_valid,
            }
        
        # training cycle
        while True:
            t0 = time()
            result = self.sess.run(fetch_dict)
            t1 = time()
            loss = result['loss']
            loss_valid = result['loss_valid']
            step = result['step']
            epoch = float(step) / self.num_itr_per_epoch
            #print("itr %d/(epoch %.1f): time %g, loss: %.4f, loss_valid: %.4f" %
            #            (step, epoch, t1 - t0, loss, loss_valid))
            print("itr %6d/(epoch %4.1f): time %.6f, loss: %.4f, loss_valid: %.4f" %
                        (step, epoch, t1 - t0, loss, loss_valid))
            if step >= 3500:
            #if step >= 0:
                self.summary_writer_train.add_summary(result['summary'], global_step=result['step'])
                self.summary_writer_valid.add_summary(result['summary_valid'], global_step=result['step'])
            
            if epoch > self.max_epoch:
                break
            step += 1

            if step % self.save_models_step == 0:
               self.saver.save(self.sess,  osp.join(self.model_dir, "model.ckpt"), global_step = step)

        # the last iteration done:       
        self.saver.save(self.sess,  osp.join(self.model_dir, "model.ckpt"), global_step = step)
        coord.request_stop()
        coord.join(threads)
        print('Finish training on %s' % self.model_dir)


    def evaluate(self):
        step = 0
        #assert (self.is_training == False)
        assert (self.task_type == "evaluation")
        assert (self.batch_size == 1)

        # Initializing the variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)
        
        if self.load(self.config.pretrained_model_path):
            self.model_dir = self.pretrained_model_path
            print(" [*] Load SUCCESS! And setting model_dir to be this dir %s" % self.model_dir)
        else:      
            print(" Load failed... Just used initialized weights for testing")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners( sess = self.sess, coord=coord)

        fetch_dict = {
            "loss": self.loss,
            "loss_pose": self.loss_l2_pose,
            "loss_shape": self.loss_l2_shape,
            "gt_gender": self.smpl_gender_loader,
            "gt_pose": self.smpl_pose_loader,
            "gt_shape": self.smpl_shape_loader,
            "pred_pose": self.pose_pred,
            "pred_shape": self.shape_pred,
            "input_joints3d": self.joints_3d_loader,
            "img_name" : self.img_name_loader,
            }
        if self.has_smpl_layer:
            fetch_dict.update({
                "loss_joints3d": self.loss_l2_joints3d,
                "pred_joints3d": self.Js,
                })
        
        # training cycle
        all_pred_pose = []
        all_pred_shape = []
        all_pred_joints = []
        all_gt_pose = []
        all_gt_shape = []
        all_gt_joints = []

        while step < 5:
            t0 = time()
            result = self.sess.run(fetch_dict)
            t1 = time()
            
            loss = result['loss']
            loss_pose = result['loss_pose']
            loss_shape = result['loss_shape']

            gt_pose = np.reshape(result['gt_pose'][0][:], [self.joint_num,3]) # [0] means batch_size = 1;
            gt_shape = result['gt_shape'][0] # [0] means batch_size = 1;
            gt_gender = result['gt_gender'][0] # [0] means batch_size = 1;
            pred_pose = np.reshape(result['pred_pose'][0], [-1,3]) # [0] means batch_size = 1;
            pred_shape = result['pred_shape'][0] # [0] means batch_size = 1;
            img_name = result['img_name'][0] # [0] means batch_size = 1;
            if self.has_smpl_layer:
                pred_joints3d = np.reshape(result['pred_joints3d'][0], [self.joint_num,3]) # [0] means batch_size = 1;
                loss_joints3d = result['loss_joints3d']
            else:
                loss_joints3d = -1.0 
                pred_joints3d = -1.0* np.ones([24,3])

            input_joints3d = np.reshape(result['input_joints3d'][0], [self.joint_num,3]) # [0] means batch_size = 1;
            """ NOTE: already verified the loss returned by tensorflow is the same as the one calculated by hand via numpy"""
            #loss_by_hand = np.sum((gt_pose - pred_pose)*(gt_pose - pred_pose)) / (2.0 * 23*3)
            #print("itr %6d : time %.6f, loss: %.4f, loss_by_hand: %.4f" % (step, t1 - t0, loss, loss_by_hand))
            print("************************************************************************************************************")
            print("itr %6d : time %.6f, loss: %.4f, loss_shape: %.4f, loss_pose: %.4f, loss_joints3d: %.4f" % 
                        (step, t1 - t0, loss, loss_shape, loss_pose, loss_joints3d))

            print("input joints3d (x,y,z)    |    pred joints3d (x,y,z)   |    pred pose_23    |    gt pose_24   |")
            for i in range(0, 24):
                if i == 0:
                    print(" idx = %2d, (%2.3f, %2.3f, %2.3f) \t (%2.3f, %2.3f, %2.3f) \t (%s, %s, %s) \t (%2.3f, %2.3f, %2.3f)" %(i,
                        input_joints3d[i][0], input_joints3d[i][1], input_joints3d[i][2],
                        pred_joints3d[i][0], pred_joints3d[i][1], pred_joints3d[i][2],
                        'N/A', 'N/A', 'N/A',
                        gt_pose[i][0], gt_pose[i][1], gt_pose[i][2],
                        ))
                else:
                    print(" idx = %2d, (%2.3f, %2.3f, %2.3f) \t (%2.3f, %2.3f, %2.3f) \t (%2.3f, %2.3f, %2.3f) \t (%2.3f, %2.3f, %2.3f)" %(i,
                        input_joints3d[i][0], input_joints3d[i][1], input_joints3d[i][2],
                        pred_joints3d[i][0], pred_joints3d[i][1], pred_joints3d[i][2],
                        pred_pose[i-1][0], pred_pose[i-1][1], pred_pose[i-1][2],
                        gt_pose[i][0], gt_pose[i][1], gt_pose[i][2],
                        ))
            
            print("\n[**] shape gt: {}".format(gt_shape))
            print("[**] shape pred: {}".format(pred_shape))
            

            """ save smpl models to mesh file """

            print ('[***] image name : %s' % img_name)
            
            outmesh_path = osp.join(self.result_dir, 'pred_img_%s.obj' % img_name)
            save_to_mesh_file(self.smpl_model_path, gt_gender, 
                              np.concatenate(
                                  [np.reshape(gt_pose[0][:], [-1]), 
                                   np.reshape(pred_pose, [-1])],
                                   axis = 0), 
                              pred_shape, outmesh_path)

            outmesh_path = osp.join(self.result_dir, 'gt_img_%s.obj' % img_name)
            save_to_mesh_file(self.smpl_model_path, gt_gender, 
                              np.reshape(gt_pose, [-1]), gt_shape, outmesh_path)
            
            
            #NOTE:The way we do left/right swap affect the GT smpl ???
            #NOTE: have to be solved SOON!!!
            if False and img_name == '06_07_c0002_frame_010':
                outmesh_path = osp.join(self.result_dir, 'gt_no_swap_img_%s.obj' % img_name)
                tmp_pose = np.array([ 
                    1.5694498, 0.7472164, 0.7963166, # hips
                    -0.6114371, -0.0301569, 0.1145555, # leftUpLeg
                    -0.2869069, 0.0208557, -0.1562156,  # rightUpLeg
                    0.5182559, 0.0643953, 0.1288075, # spine
                    0.2423580, 0.0853673, -0.0724941, # leftLeg
                    0.7831452, 0.2150790, -0.0131519, # rightLeg
                    -0.0032498, 0.0635217, 0.0348864, # spine1
                    0.2006642, 0.2139296, -0.0613069, #leftFoot
                    -0.3186781, 0.0141717, -0.1568169, # rightFoot
                    0.0128728, 0.0685575, 0.0480169, # spine2
                    -0.1354543, 0.0848213, 0.1167145, # leftToeBase
                    -0.0381632, -0.2613489, 0.2968476, # rightToeBase
                    -0.0492332, 0.0647589, -0.0030432, # neck
                    0.0270300, 0.0456309, -0.4468385, # leftShoulder
                    0.1471884, 0.2596276, 0.3748693, # rightShoulder
                    -0.0597905, 0.0594899, -0.0203924, # head
                    0.1904113, -0.0866473, -0.9173592, # leftArm
                    -0.0585544, 0.3967225, 0.5021892, # rightArm

                    0.3859173, -0.7930598, 0.3880045, # leftForeArm
                    0.2750341, 0.4291943, -0.2827638, # rightForeArm
                    -0.0611419, -0.1078939, 0.1185737, # leftHand
                    0.3663123, 0.0470825, -0.2186135, # rightHand
                     -0.1414059, 0.0025787, -0.1025999, # leftHandIndex1
                     -0.1270814, 0.0706553, 0.0798347 # rightHandIndex1
                    ])
                save_to_mesh_file(self.smpl_model_path, gt_gender, np.reshape(tmp_pose, [-1]), gt_shape, outmesh_path)


            all_pred_pose.append(pred_pose) # in shape : (23*3,)
            all_pred_shape.append(pred_shape) # in shape : (10,)
            all_pred_joints.append(pred_joints3d) # 24 x 3
            all_gt_pose.append(gt_pose) # 24 x 3
            all_gt_shape.append(gt_shape) # 10
            all_gt_joints.append(input_joints3d) # 24 x 3;
            step += 1

        coord.request_stop()
        coord.join(threads)
        print('Testing done!!!')

        """ draw 3d joints """
        if False:
            save_fig_result = osp.join(self.result_dir, 'joints3d_with_cycle_loss.png')
            draw_joints_3d_scatter(all_gt_joints, all_pred_joints, save_fig_result)

    def pred_smpl_param(self, joints3d, gender_vect):

        # Initializing the variables
        if self.is_first:
            print (" [***] Initializing the variables !!!")
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            self.sess.run(init_op)
            self.is_first = False
        
        feed_dict = {
            self.joints_3d_pl: joints3d,
            self.gender_pl : gender_vect,
        }

        fetch_dict = {
            "pred_pose": self.pose_pred,
            "pred_shape": self.shape_pred,
            'pred_joints3d': self.pred_joints3d,
            }
        
        result = self.sess.run(fetch_dict, feed_dict)
        pred_pose = result['pred_pose'] # in shape N x 24*3;
        pred_shape = result['pred_shape'] # in shape N x 10;
        pred_joints3d = result['pred_joints3d'] # in shape N x 14 (or 19, 24) x 3
        return pred_pose, pred_shape, pred_joints3d

    """ for testing: no GT provided """
    def test(self, joints3d, gender_vect, img_name):
        assert (self.task_type == 'test')
        assert (self.batch_size == 1)

        # Initializing the variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init_op)
        
        if self.load(self.config.pretrained_model_path):
            self.model_dir = self.pretrained_model_path
            print(" [*] Load SUCCESS! And setting model_dir to be this dir %s" % self.model_dir)
        else:      
            print(" Load failed... Just used initialized weights for testing")
        
        feed_dict = {
            self.joints_3d_pl: joints3d,
            self.gender_pl : gender_vect,
        }

        fetch_dict = {
            "pred_pose": self.pose_pred,
            "pred_shape": self.shape_pred,
            'weight_1': self.perceptron.weights['h1'],
            'weight_2': self.perceptron.weights['h2'],
            'weight_3': self.perceptron.weights['h3'],
            'weight_out': self.perceptron.weights['out'],
            'biases_b1': self.perceptron.biases['b1'],
            'biases_b2': self.perceptron.biases['b2'],
            'biases_b3': self.perceptron.biases['b3'],
            'biases_out': self.perceptron.biases['out'],
            }
        
        result = self.sess.run(fetch_dict, feed_dict)
        #_save_dict_to_json(result, osp.join(self.result_dir, 'debug_network_%s.json' % img_name))
        pred_pose = np.reshape(result['pred_pose'][0], [23,3]) # [0] means batch_size = 1;
        pred_shape = result['pred_shape'][0] # [0] means batch_size = 1;
        input_joints3d = np.reshape(joints3d[0], [self.joint_num,3]) # [0] means batch_size = 1;
        
        """ save smpl models to mesh file """
        img_name = img_name.replace( '/', '-')

        print ('[***] image name : %s' % img_name)
        outmesh_path = osp.join(self.result_dir, 'pred_img_%s.obj' % img_name)
        pred_joints3d =  save_to_mesh_file(
            self.smpl_model_path, 
            gender_vect[0], 
            np.concatenate( [np.array([np.pi, .0, .0]), np.reshape(pred_pose, [-1])], axis = 0), 
            pred_shape, outmesh_path)

        assert (pred_joints3d.shape[0] == 24)
        pred_joints3d = pred_joints3d[get_lsp_idx_from_smpl_joints(),:]
        print("input joints3d (x,y,z)    |    pred joints3d (x,y,z)   |    pred pose_23    |")
        for i in range(0, self.joint_num):
            if i == 0:
                print(" idx = %2d, (%2.3f, %2.3f, %2.3f) \t (%2.3f, %2.3f, %2.3f) \t (%s, %s, %s)" %(
                    i, input_joints3d[i][0], input_joints3d[i][1], input_joints3d[i][2],
                    pred_joints3d[i][0], pred_joints3d[i][1], pred_joints3d[i][2],
                    'N/A', 'N/A', 'N/A',
                        )) 
            else:
                print(" idx = %2d, (%2.3f, %2.3f, %2.3f) \t (%2.3f, %2.3f, %2.3f) \t (%2.3f, %2.3f, %2.3f)" %(
                    i, input_joints3d[i][0], input_joints3d[i][1], input_joints3d[i][2],
                    pred_joints3d[i][0], pred_joints3d[i][1], pred_joints3d[i][2],
                    pred_pose[i-1][0], pred_pose[i-1][1], pred_pose[i-1][2],
                    )) 
        print ('[**] smpl_shape = {}'.format(pred_shape))
        print('Testing done!!!')


#********************************************************************************************        
#********************************************************************************************        
#********************************************************************************************        

def _save_dict_to_json(dict_to_save, json_fname):
    tmp_dict = {}
    for k in dict_to_save:
        tmp_dict[k] = dict_to_save[k].tolist()
        with open(json_fname, 'w') as fp:
            json.dump(tmp_dict, fp, sort_keys=True)


def draw_lsp_skeleton(joints3D, save_fig_result, with_numbers=True):
    from mpl_toolkits.mplot3d import Axes3D
    #fig = plt.figure()
    #ax = Axes3D(fig)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = []
    left_right_mid = ['r', 'g', 'b']
    kintree_colors = [1, 1, 1, 
                      0, 0, 0, 
                      1, 1, 1, 
                      0, 0, 0, 
                      2, 2]
    assert (joints3D.shape[0] == 14 and joints3D.shape[1] == 3)
    
    for c in kintree_colors:
        colors += left_right_mid[c]
    
    # For each 14 joint
    lsp_names = [
        'R Ankle', #0
        'R Knee', # 1
        'R Hip', #2
        'L Hip', #3
        'L Knee',#4
        'L Ankle', #5
        'R Wrist', #6
        'R Elbow', #7
        'R Shoulder', #8
        'L Shoulder', #9
        'L Elbow', #10
        'L Wrist', #11
        'Neck',#12
        'Head'#13
    ]
    kintree_table = np.array(
        # child, parent
        [
            [0, 1],
            [1, 2],
            [2, 8],
            [3, 9],
            [4, 3],
            [5, 4],
            [6, 7],
            [7, 8],
            [8, 12],
            [9, 12],
            [10, 9],
            [11, 10],
            [12, 13],
            [13, 12],
        ]
        )
    for i in range(0, kintree_table.shape[0]): # exlcuding "head" joint
        child_id = kintree_table[i][0]
        parent_id = kintree_table[i][1]
        #print ("drawing joint %s" % lsp_names[child_id])
        ax.plot([joints3D[child_id, 0], joints3D[parent_id, 0]],
                [joints3D[child_id, 1], joints3D[parent_id, 1]],
                [joints3D[child_id, 2], joints3D[parent_id, 2]],
                color=colors[i], linestyle='-', linewidth=2, marker='o', markersize=5)
        if with_numbers:
            ax.text(joints3D[child_id, 0], joints3D[child_id, 1], joints3D[child_id, 2], child_id)
    
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    #ax.set_xlim(-0.7, 0.7)
    #ax.set_ylim(-0.7, 0.7)
    #ax.set_zlim(-0.7, 0.7)
    
    # 'elev' stores the elevation angle in the z plane. 
    # 'azim' stores the azimuth angle in the x,y plane.
    ax.view_init(azim=-90, elev=100)
    #ax.view_init(azim=-90, elev=-100)
    plt.title('LSP 14 Joints3D')
    #fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    if save_fig_result:
        plt.savefig(save_fig_result, bbox_inches='tight', pad_inches=0)
        print ("saved figure at %s ..." % save_fig_result)


""" draw 3d joints """
def draw_joints_3d_scatter(all_gt_joints, all_pred_joints,save_fig_result):
    from mpl_toolkits.mplot3d import Axes3D
    # Scatter plot with groups
    inputs = (all_gt_joints, all_pred_joints)
    colors = ("red", "blue")
    groups = ("gt_joints3d", "pred_joints3d")
    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)

    for data, color, group in zip(inputs, colors, groups):
        x_vals = [ j[:,0] for j in data]
        y_vals = [ j[:,1] for j in data]
        z_vals = [ j[:,2] for j in data]
        # Plot the values
        ax.scatter(
            x_vals, y_vals, z_vals, c = color, label = group, edgecolors='none')
    
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.title('SMPL Regressor Joints3D')
    plt.legend(loc=2)
    dockerEnv = True
    if not dockerEnv:
        plt.show()
    else:
        plt.savefig(save_fig_result)
        print ("saved %s ..." % save_fig_result)

# save the smpl models to mesh files;
def save_to_mesh_file(smpl_model_path, gender_vect, pose, shape, outmesh_path):
    assert (pose.shape[0] == 24*3) 
    assert (shape.shape[0] == 10) 
    models_paths = extract_female_male_neutral_model_paths(smpl_model_path) # in f, m, n order;
    gender_list = ['f', 'm', 'n']
    #print ("[???] gender_vect = {}".format(gender_vect))
    for i in range(0,3):
        if int(gender_vect[i]) == 1:
            gender = gender_list[i]
    print ("gender_vect = {}, ===> gender : {}".format(gender_vect,gender))

    if gender == 'f': # f
        print ('loading model : %s' % models_paths[0])
        m = load_model(models_paths[0])
    elif gender == 'm':  # m
        print ('loading model : %s' % models_paths[1])
        m = load_model(models_paths[1])
    elif gender == 'n':
        print ('loading model : %s' % models_paths[2])
        m = load_model(models_paths[2])
    
    m.pose[:] = pose
    m.betas[:] = shape
    # Set model translation
    #m.trans[:] = trans[:]

    ## Write to an .obj file
    with open( outmesh_path, 'w') as fp:
        for v in m.r:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

        for f in m.f+1: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

    ## Print message
    print ('..Output mesh saved to: %s' % outmesh_path)
    return m.J_transformed.r


def extract_female_male_neutral_model_paths(smpl_model_path, isDisplay = False):
        #print ("??? ", smpl_model_path) 
        smpl_female_model_path = smpl_model_path.split(',')[0]
        posit = smpl_model_path.rfind('/')
        smpl_male_model_path = smpl_model_path[:posit] + '/'+ smpl_model_path.split(',')[1]
        smpl_neutral_model_path = smpl_model_path[:posit] + '/' + smpl_model_path.split(',')[2]
        if isDisplay:
            print ("female model: {}, male model: {}, neutral model: {}".format(
            smpl_female_model_path, 
            smpl_male_model_path,
            smpl_neutral_model_path))
        return [smpl_female_model_path, smpl_male_model_path, smpl_neutral_model_path]

def get_one_batched_cad_120_toy_example():
        # values are in milimeters
        Joints3d_x = np.array([-511.9, -445.3, -458.5, -288.4, -354.3, -378.2, -416.3, -510.2, -433.4, -178.5, -158.1, -217.7, -306.0, -265.2])
        Joints3d_y = np.array([-790.3, -471.4, -89.1, -104.1, -473.1, -839.2, -212.9, 20.8, 273.2, 250.7, 5.1, -237.3,  261.9, 445.3])
        Joints3d_z = np.array([2075.1, 1899.1, 1944.1, 1993.6, 2082.0, 2131.3, 1833.3, 1834.4, 1808.4, 1882.5, 1968.5, 1935.0, 1845.4, 1771.2])
        Joints3d_x = np.reshape(Joints3d_x, [14, 1])
        Joints3d_y = np.reshape(Joints3d_y, [14, 1])
        Joints3d_z = np.reshape(Joints3d_z, [14, 1])
        joints3d = np.hstack([Joints3d_x,Joints3d_y,Joints3d_z])/ 1000.0 # change to meters
        print (joints3d.shape)
        lsp_joints_names = [
        'R Ankle', 'R Knee', 'R Hip', 'L Hip', 'L Knee', 'L Ankle', 'R Wrist',
        'R Elbow', 'R Shoulder', 'L Shoulder', 'L Elbow', 'L Wrist', 'Neck',
        'Head']
        root = 0.5*(joints3d[lsp_joints_names.index('R Hip'), :] + joints3d[lsp_joints_names.index('L Hip'), :])
        print ("[**] root = (R_Hip + L_Hip)/2 = ", root)
        joints3d -= root
        #with np.printoptions(precision=3, suppress=True):
        #    print ('[**] joints3d after root extracted : {}'.format(joints3d*1000.0))
        np.set_printoptions(precision=4, suppress=True)
        print ('[**] joints3d with root extracted :\n{}'.format(joints3d))
        joints3d = np.reshape(np.expand_dims(joints3d, axis = 0), [1, 14*3])
        img_name = 'cad-120-small-1-img-sub1-arranging_objects-0510175411-RGB_1'
        gender_vect = np.array([1.0, .0, .0]) # female; size (3,)
        gender_vect = np.expand_dims(gender_vect, axis = 0) # female, size (1,3)
        return  joints3d, gender_vect, img_name

""" This function is adapted from the code in src/util/data_utils.py """
def reflect_joints3d(joints):
    """
    Assumes input is 14 x 3 (the LSP skeleton subset of H3.6M)
    """
    assert (joints.shape[0] == 14)

    swap_inds = np.array([5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13])
    joints_ref = joints[swap_inds, :]
    flip_mat = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32)
    joints_ref = np.dot(flip_mat, joints_ref.T).T
    # Assumes all joints3d are mean subtracted
    joints_ref = joints_ref - np.mean(joints_ref, axis=0)
    return joints_ref


def get_one_batched_cad_toy_example(cad_anno_txt_fname, sample_idx = 0, gender = 'n', 
        swap = True):
        
        if gender == 'f':
            gender_vect = np.array([1.0, .0, .0]) # female
        elif gender == 'm':
            gender_vect = np.array([.0, 1.0, .0]) # male
        else:
            gender_vect = np.array([.0, .0, 1.0]) # nuetral

        gender_vect = np.expand_dims(gender_vect, axis = 0) # batched
        
        # joints3d : 3 x joints_Num (i.e., 15 ) x frames_num
        all_joints3d, all_fnames = read_cad_txt_annotation(cad_anno_txt_fname)
        cad2lspIdx_3d = get_cad_2_lsp_idx(is2DJoints = False)
        #cad2lspIdx_3d = get_cad_2_lsp_idx(is2DJoints = True)
        img_name = all_fnames[sample_idx]
        torso_idx_cad = 2
        torso = all_joints3d[:, torso_idx_cad, sample_idx] * 0.001
        # values are in milimeters
        joints3d = all_joints3d[:, cad2lspIdx_3d, sample_idx].T * 0.001 
        #print (torso.shape)
        #print (joints3d.shape)
        lsp_joints_names = [
        'R Ankle', 'R Knee', 'R Hip', 'L Hip', 'L Knee', 'L Ankle', 'R Wrist',
        'R Elbow', 'R Shoulder', 'L Shoulder', 'L Elbow', 'L Wrist', 'Neck',
        'Head']
        
        #NOTE: using the middle point of L hip and R hip is not accuracy, 
        # so instead we use the torso provided by CAD dataset itself;
        #root = 0.5*(joints3d[lsp_joints_names.index('R Hip'), :] + joints3d[lsp_joints_names.index('L Hip'), :])
        #print ("[**] root = (R_Hip + L_Hip)/2 = ", root)
        #joints3d_minus_root = joints3d - root
        joints3d_minus_root = joints3d - torso
        #print ('[**] torso:\n{}'.format(torso))
        #print ('[**] joints3d before root extracted :\n{}'.format(joints3d))
        #print ('[**] joints3d after root extracted :\n{}'.format(joints3d_minus_root))
        if swap:
            print ("[***] doing reflect_joints3d(..) to joints3d_minus_root and joints3d ")
            joints3d_minus_root = reflect_joints3d(joints3d_minus_root)
            joints3d = reflect_joints3d(joints3d)
            #print ('[**] joints3d, swapped :\n{}'.format(joints3d))
            #print ('[**] joints3d with root extracted, swapped :\n{}'.format(joints3d_minus_root))
        np.set_printoptions(precision=6, suppress=True)
        joints3d_minus_root = np.reshape(np.expand_dims(joints3d_minus_root, axis = 0), [1, 14*3])
        return  joints3d_minus_root, gender_vect, img_name, joints3d

def get_one_bached_mosh_toy_example(sample_idx = 0, joint_type = 'lsp', gender = 'n'):
    if gender == 'f':
        gender_vect = np.array([1.0, .0, .0]) # female
    elif gender == 'm':
        gender_vect = np.array([.0, 1.0, .0]) # male
    else:
        gender_vect = np.array([.0, .0, 1.0]) # nuetral

    gender_vect = np.expand_dims(gender_vect, axis = 0) # batched
    img_name = 'mosh_img_%07d' % sample_idx

    # in shape [N, 10], [N, 72], and [N, 24, 3]
    _, smplShapeParams, smplPoseParams, smplJoints = load_smpl_joints3d_pair_h5("datasets/MoSh/data/mosh_gen")
    shape = np.expand_dims(smplShapeParams[sample_idx, :], axis = 0)
    pose = np.expand_dims(smplPoseParams[sample_idx, :], axis = 0)
    joints3d = np.expand_dims(smplJoints[sample_idx, :,:], axis = 0)
    joints3d -= joints3d[:,0,:]
    if joint_type == 'lsp':
        lsp_idx = get_lsp_idx_from_smpl_joints()
        joints3d = np.reshape(joints3d[:, lsp_idx, :], [1, 14*3])
    return shape, pose, joints3d, gender_vect, img_name


def main(config):
    
    if config.task_type == 'train':
        # Load data on CPU
        print ("[*****] training !!!")
        with tf.device("/cpu:0"):
            # training data loader
            print ("[***] training batch_size = {}".format(config.batch_size))
            train_loader = DataLoader(config, light_weight = True)
            train_data_pair_loader = train_loader.loader_smpl_joints3d_pair()
        
            # update config for validation data loading while network training; 
            config.split = 'valid'
            #ratio_train_2_valid = (35+ 55.6 + 26.3) / (10+16.4+7.23)
            ratio_train_2_valid = 1
            config.batch_size = int(config.batch_size / ratio_train_2_valid)
            print ("[***] validation batch_size = {}".format(config.batch_size))
            valid_loader = DataLoader(config, light_weight = True)
            valid_data_pair_loader = valid_loader.loader_smpl_joints3d_pair()
        
        with tf.Session() as sess:
            trainer = SMPLTrainer(sess, config, train_data_pair_loader, valid_data_pair_loader)
            trainer.train()
    
    elif config.task_type == 'evaluation':
        # Load data on CPU
        print ("[*****] evaluation !!!")
        with tf.device("/cpu:0"):
            # update config 
            #config.split = 'valid' # all the validation data
            config.split = 'valid_5_samples' # just filte the small 5 samples;
            config.batch_size = 1
            print ("[***] evaluation: batch_size = {}".format(config.batch_size))
            valid_loader = DataLoader(config, light_weight = True)
            valid_data_pair_loader = valid_loader.loader_smpl_joints3d_pair()
        
        with tf.Session() as sess:
            trainer = SMPLTrainer(sess, config, valid_data_pair_loader)
            trainer.evaluate()

    elif config.task_type == 'test':
        # Load data on CPU
        print ("[*****] testing !!!")
        # update config 
        config.batch_size = 1
        print ("[***] test: batch_size = {}".format(config.batch_size))
        
        with tf.Session() as sess:
            trainer = SMPLTrainer(sess, config, None)
            # different toy examples for testing;
            if config.toy_data_type == 'mosh-hd5-small-1-x':
                shape_gt, pose_gt, joints3d, gender_vect, \
                     img_name = get_one_bached_mosh_toy_example(sample_idx = 10000, joint_type = 'lsp', gender = 'n')
                # set rest pose;
                pose_gt[0,0] = np.pi
                pose_gt[0,1] = .0
                pose_gt[0,2] = .0
                save_to_mesh_file(trainer.smpl_model_path, gender_vect[0], pose_gt[0], shape_gt[0], 
                    osp.join(trainer.result_dir, 'gt_img_%s.obj' % img_name))
                trainer.test(joints3d, gender_vect, img_name)
            elif config.toy_data_type == 'cad-any-1-sample':
                cad_anno_to_try = [
                    ('cad-60-small-1-img-person1/0512164529.txt', 0, 'm'), # RGB_1.png
                    ('cad-60-small-5-imgs-person1/0512164529.txt', 0, 'm'), # RGB_1.png
                    ('cad-120/Subject1_rgbd_images/making_cereal/1204142500.txt', 0,'f'), # RGB_1.png
                    ('cad-60/Person2/0510163513.txt', 146, 'f'), # RGB_147.png
                    ('cad-60/Person4/0512152416.txt', 558, 'm'), # RGB_559.png
                    ('cad-120/Subject4_rgbd_images/picking_objects/0510172851.txt', 154, 'm'), # RGB_155.png
                    ('cad-120/Subject3_rgbd_images/taking_food/0510144139.txt', 452, 'f'), # RGB_453.png

                    ('cad-120-small-1-img-sub1/arranging_objects/0510175411.txt', 0, 'f'), #RGB_1.png
                    ('cad-120/Subject1_rgbd_images/picking_objects/0510175829.txt', 158, 'f'), #RGB_159.png
                    ]
                
                for i in range(7, 8):
                    joints3d, gender_vect, img_name, joints3d_with_root = get_one_batched_cad_toy_example(
                        cad_anno_txt_fname = './datasets/cad-60-120/' + cad_anno_to_try[i][0],
                        sample_idx = cad_anno_to_try[i][1], 
                        gender = cad_anno_to_try[i][2],
                        swap = True)
                    np.set_printoptions(precision=4, suppress=True)
                    print (img_name)
                    #np.save(osp.join(trainer.result_dir, 'joints3d_%s' % img_name.replace('/', '-')), joints3d_with_root)
                    
                    save_fig_result = osp.join(trainer.result_dir, 'lsp_14joints3d_%s' % img_name.replace('/', '-'))
                    #draw_lsp_skeleton(np.reshape(joints3d[0], [-1, 3]), save_fig_result, with_numbers=True)
                    draw_lsp_skeleton(joints3d_with_root, save_fig_result, with_numbers=True)
                    trainer.test(joints3d, gender_vect, img_name)

            elif config.toy_data_type == 'cad-120-small-1':
                joints3d, gender_vect, img_name = get_one_batched_cad_120_toy_example()
                trainer.test(joints3d, gender_vect, img_name)
            
            #trainer.test(joints3d, gender_vect, img_name)

if __name__ == '__main__':
    
    #NOTE: axis-angle : (pi, 0, 0) <==> R_(x, pi);
    #src = np.array([np.pi, .0, .0])
    #Rs, _ = cv2.Rodrigues(src)
    #print (Rs)
    #sys.exit()
    config = get_config()
    main()