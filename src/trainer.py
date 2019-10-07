"""
HMR trainer.
From an image input, trained a model that outputs 85D latent vector
consisting of [cam (3 - [scale, tx, ty]), pose (72), shape (10)]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .data_loader import num_examples

from .ops import (keypoint_l1_loss, 
            compute_3d_loss, 
            align_by_pelvis,
            get_batched_cam_trans,
            get_proj_vert2d
            )

from .models import Discriminator_separable_rotations, get_encoder_fn_separate

from .tf_smpl.batch_lbs import batch_rodrigues
from .tf_smpl.batch_smpl import SMPL
from .tf_smpl.projection import batch_orth_proj_idrot

from tensorflow.python.ops import control_flow_ops

from time import time
import tensorflow as tf
import numpy as np

from os.path import join, dirname, exists
from os import makedirs
import deepdish as dd

# For drawing
from .util import renderer as vis_util
import json

from src.load_data_4_inference import extract_14_joints
from src.util import surreal_in_extrinc as surreal_util

import cv2



class HMRTrainer(object):
    def __init__(self, config, data_loader, mocap_loader):
        """
        Args:
          config
          if no 3D label is available,
              data_loader is a dict
          else
              data_loader is a dict
        mocap_loader is a tuple (pose, shape)
        """
        # Config + path
        self.config = config
        self.model_dir = config.model_dir
        self.data_format = config.data_format
        
        self.load_path = config.load_path
        # extract female, male  and neutral model paths;
        self.smpl_model_path = config.smpl_model_path
        self.smpl_female_model_path = config.smpl_model_path.split(',')[0]
        posit = config.smpl_model_path.rfind('/')
        self.smpl_male_model_path = config.smpl_model_path[:posit] + '/'+ config.smpl_model_path.split(',')[1]
        self.smpl_neutral_model_path = config.smpl_model_path[:posit] + '/' + config.smpl_model_path.split(',')[2]
        print ("female model: {}, male model: {}, neutral model: {}".format(
            self.smpl_female_model_path, self.smpl_male_model_path, self.smpl_neutral_model_path))

        self.pretrained_model_path = config.pretrained_model_path
        self.encoder_only = config.encoder_only
        print("[***] encoder_only = {}".format(self.encoder_only))
        self.use_3d_label = config.use_3d_label
        
        self.has_depth_loss = config.has_depth_loss
        
        #NOTE: added by CCJ: with or without the face points (usually 5 face points)
        """ joint types: """
        # * lsp = 14 joints;
        # cocoplus = 14 lsp joints + 5 face points = 19 joints;
        # smpl: = 24 joints;
        self.joint_type = config.joint_type.lower()
        print("[***] joint_type = {}".format(self.joint_type))
        if self.joint_type == 'lsp':
            self.joint_num = 14
        elif self.joint_type == 'cocoplus':
            self.joint_num = 19
        elif self.joint_type == 'smpl':
            self.joint_num = 24
        else:
            print('BAD!! Unknown joint type: %s, it must be either "cocoplus" or "lsp"' % self.joint_type)
            import ipdb
            ipdb.set_trace()
        print("[***] joint_num = {}".format(self.joint_num))

        # Data size
        self.img_size = config.img_size
        self.num_stage = config.num_stage
        self.batch_size = config.batch_size
        print ("[****] Batch size = %d" % self.batch_size)
        self.max_epoch = config.epoch

        self.num_cam = 3 # parameters :  [scale, tx, ty]

        self.num_theta = 72  # 24 * 3
        self.total_params = self.num_theta + self.num_cam + 10 # 10: for parameter shape;

        # Data
        num_images = num_examples(config.datasets) # number of images;
        num_mocap = num_examples(config.mocap_datasets) # number of mocap images;

        self.num_itr_per_epoch = num_images / self.batch_size
        self.num_mocap_itr_per_epoch = num_mocap / self.batch_size
        print ("[**] Total num_images for training = %d, num_itr_per_epoch = %d, num_mocap_itr_per_epoch = %d" %(
            num_images, self.num_itr_per_epoch, self.num_mocap_itr_per_epoch))

        # First make sure data_format is right
        if self.data_format == 'NCHW':
            # B x H x W x 3 --> B x 3 x H x W
            data_loader['image'] = tf.transpose(data_loader['image'], [0, 3, 1, 2])
            data_loader['depth'] = tf.transpose(data_loader['depth'], [0, 3, 1, 2])

        self.image_loader = data_loader['image']
        self.depth_loader = data_loader['depth']
        # reshapt (N,) tp N x 1
        self.depth_max_loader = tf.reshape(data_loader['depth_max'], [-1, 1]) # added for depth loss on Aug 13 2019;
        print ("??? depth_max_loader shape ", self.depth_max_loader.shape)
        self.kp_loader = data_loader['label']
        self.cam_loader = data_loader['cam']
        #NOTE:added by CCJ;
        self.fname = data_loader['fname']
        self.step_50_to_save_json = 50
        if self.use_3d_label:
            self.poseshape_loader = data_loader['label3d']
            # image_loader[3] is N x 2, first column is 3D_joints gt existence,
            # second column is 3D_smpl gt existence
            self.has_gt3d_joints = data_loader['has3d'][:, 0]
            self.has_gt3d_smpl = data_loader['has3d'][:, 1]

        self.pose_loader = mocap_loader[0]
        self.shape_loader = mocap_loader[1]

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.log_img_step = config.log_img_step

        # For visualization:
        num2show = np.minimum(6, self.batch_size)
        """ comments added by CCJ: images to show """
        # e.g., batch_size = 8, num2show = 6,
        # then, to take the first 3 (i.e., num2show/2) and 
        # last 3 (i.e, num2show/2) along the batch dim;
        # Take half from front & back
        self.show_these = tf.constant(
            np.hstack(
                [np.arange(num2show / 2), self.batch_size - np.arange(3) - 1]),
            tf.int32)

        # Model spec
        self.model_type = config.model_type
        self.keypoint_loss = keypoint_l1_loss

        # Optimizer, learning rate
        self.e_lr = config.e_lr
        self.d_lr = config.d_lr
        # Weight decay
        self.e_wd = config.e_wd
        self.d_wd = config.d_wd

        # different weights
        self.e_loss_weight = config.e_loss_weight
        self.d_loss_weight = config.d_loss_weight

        #NOTE: added by CCJ for debugging 3d loss;
        #self.e_3d_weight = config.e_3d_weight
        self.e_weight_depth = config.e_weight_depth
        self.e_3d_weight_js3d = config.e_3d_weight_js3d
        self.e_3d_weight_smpl = config.e_3d_weight_smpl
        self.e_3d_weight = max(self.e_3d_weight_js3d, self.e_3d_weight_smpl)

        
        self.optimizer = tf.train.AdamOptimizer

        # Instantiate SMPL
        #self.smpl = SMPL(self.smpl_model_path, joint_type = 'cocoplus') # joint_type = 'lsp' or 'cocoplus'

        self.smpl = SMPL(
            [self.smpl_female_model_path, self.smpl_male_model_path, self.smpl_neutral_model_path],
            joint_type = self.joint_type,
            ) # joint_type = 'lsp' or 'cocoplus'
        
        self.E_var = []
        self.smpl_dict_to_save = {}
        self.build_model()
        print ("[***] self.build_model() done !")

        # Logging
        init_fn = None
        if self.use_pretrained():
            # Make custom init_fn
            print("Fine-tuning from %s" % self.pretrained_model_path)
            #print ("[???] self.E_var : ", self.E_var)
            
            # see https://stackoverflow.com/questions/45606298/how-to-restore-weights-with-different-names-but-same-shapes-tensorflow?rq=1;
            # var1 is the variable you want ot restore
            img_resnet_vars_list = {}
            dep_resnet_vars_list = {}

            if 'resnet_v2_50' in self.pretrained_model_path:
                # for resnet_v2_50_img branch;
                for var in self.E_var:
                    if 'resnet_v2_50_img' in var.name:
                        """ given: 
                            Variable name in checkpoint file:  'resnet_v2_50/block1/unit_1/bottleneck_v2/conv1/BatchNorm/beta';
                            variable name in this session: 
                            'Encoder_resnet_v2/resnet_v2_50_img/resnet_v2_50/block1/unit_1/bottleneck_v2/conv1/BatchNorm/beta:0'
                            So we have to extract the checkpoint variable name from the session Variable name;    
                        """
                        key_tmp = var.name.split('resnet_v2_50_img/')[1].split(":")[0]
                        img_resnet_vars_list[key_tmp] = var
                    elif 'resnet_v2_50_dep' in var.name:
                        key_tmp = var.name.split('resnet_v2_50_dep/')[1].split(":")[0]
                        dep_resnet_vars_list[key_tmp] = var

                #print ("img_resnet_vars_list : ".format(img_resnet_vars_list))
                #print ("dep_resnet_vars_list : ".format(dep_resnet_vars_list))
                #resnet_vars = [
                #    var for var in self.E_var if 
                #    'resnet_v2_50' in var.name
                #]
                #self.pre_train_saver = tf.train.Saver(resnet_vars)
                self.pre_train_saver_img  = tf.train.Saver(img_resnet_vars_list)
                self.pre_train_saver_dep  = tf.train.Saver(dep_resnet_vars_list)
                print ("[ *** ] self.pre_train_saver_img and _dep Done !")

            elif 'pose-tensorflow' in self.pretrained_model_path:
                resnet_vars = [
                    var for var in self.E_var if 'resnet_v1_101' in var.name
                ]
                self.pre_train_saver = tf.train.Saver(resnet_vars)
            else:
                self.pre_train_saver = tf.train.Saver()

            def load_pretrain(sess):
                #self.pre_train_saver.restore(sess, self.pretrained_model_path)
                # added by CCJ;
                self.pre_train_saver_img.restore(sess, self.pretrained_model_path)
                self.pre_train_saver_dep.restore(sess, self.pretrained_model_path)

            init_fn = load_pretrain

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=4)
        print ("[ *** ] self.saver Done !")
        self.summary_writer = tf.summary.FileWriter(self.model_dir)
        # tf.train.Supervisor: A training helper that checkpoints models and 
        # computes summaries.
        self.sv = tf.train.Supervisor(
            logdir=self.model_dir,
            global_step=self.global_step,
            saver=self.saver,
            summary_writer=self.summary_writer,
            init_fn=init_fn)

        print ("[ *** ] self.sv Done !")
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess_config = tf.ConfigProto(
            allow_soft_placement=False,
            log_device_placement=False,
            gpu_options=gpu_options)

    def use_pretrained(self):
        """
        Returns true only if:
          1. model_type is "resnet"
          2. pretrained_model_path is not None
          3. model_dir is NOT empty, meaning we're picking up from previous
             so use this pretrained model.
        """
        if ('resnet' in self.model_type) and (self.pretrained_model_path is not None):
            # Check is model_dir is empty
            if self.pretrained_model_path is '':
                return False
            
            import os
            if os.listdir(self.model_dir) == []:
                return True
        return False

    def load_mean_param(self):
        mean = np.zeros((1, self.total_params))
        # Initialize scale at 0.9
        mean[0, 0] = 0.9
        mean_path = join( dirname(self.smpl_model_path), 'neutral_smpl_mean_params.h5')
        mean_vals = dd.io.load(mean_path)

        mean_pose = mean_vals['pose']
        # Ignore the global rotation.
        mean_pose[:3] = 0.
        mean_shape = mean_vals['shape']

        # This initializes the global pose to be up-right when projected
        mean_pose[0] = np.pi

        mean[0, 3:] = np.hstack((mean_pose, mean_shape))
        mean = tf.constant(mean, tf.float32)
        self.mean_var = tf.Variable(
            mean, name="mean_param", dtype=tf.float32, trainable=True)
        self.E_var.append(self.mean_var)
        init_mean = tf.tile(self.mean_var, [self.batch_size, 1])
        return init_mean
    
    """ 
    #############
    added by CCJ:
    change the model for incorpoating depth map as input_img
    #############
    """
    #NOTE: change the model for depth map as input;
    def build_model(self):
        img_enc_fn, threed_enc_fn = get_encoder_fn_separate(self.model_type)
        # Extract image features.
        self.img_dep_feat, self.E_var = img_enc_fn(
            x = self.image_loader, depth = self.depth_loader, weight_decay=self.e_wd, 
            reuse=False
            #reuse=tf.AUTO_REUSE
            )
        
        loss_kps = []
        loss_depths = []
        if self.use_3d_label:
            loss_3d_joints, loss_3d_params = [], []
        
        # For discriminator
        fake_rotations, fake_shapes = [], []
        # Start loop
        # 85D tensor; 
        # comments added by CCJ: here 85 = 3 (for camera) + 24*3 (for poses of 24 joints) + 10 (for shape); 
        theta_prev = self.load_mean_param()

        # For visualizations
        self.all_verts = []
        self.all_verts_gt = [] # added by CCJ;
        self.all_pred_kps = []
        
        self.all_gt_kps = [] # added by CCJ;
        self.all_gt_3djoints = [] # added by CCJ;
        self.all_pred_3djoints = [] # added by CCJ;
       
        self.all_pred_cams = []
        self.all_delta_thetas = []
        self.all_theta_prev = []
        
        if self.has_depth_loss:
            self.all_pred_depths = [] # predicted depth;
            self.all_proj_vert2ds = [] # projected vertices to the image plane;

        # ***********
        # NOTE: rendering ground truth smpl model;
        # ***********
        # just let it work only for surreal ???
        gt_gender = self.poseshape_loader[:, 340:343]
        gt_verts, gt_Js, gt_Rs, _  = self.smpl(
                self.poseshape_loader[:, 216:226], #gt_shapes,
                self.poseshape_loader[:, 268:340], #gt_poses, 268+72 = 340
                get_skin=True,
                gender = gt_gender
                )
        
        #gt_joints = gt_Js[:, :14, :]
        """ Main IEF loop (Iterative Error Feedback) """
        # Main IEF loop
        for i in np.arange(self.num_stage):
            print('Iteration %d' % i)
            # ---- Compute outputs
            #NOTE:
            #state = tf.concat([self.img_feat, theta_prev], 1)
            state = tf.concat([self.img_dep_feat, theta_prev], 1)

            if i == 0:
                delta_theta, threeD_var = threed_enc_fn(
                    state,
                    num_output=self.total_params,
                    reuse=False)
                self.E_var.extend(threeD_var)
            else:
                delta_theta, _ = threed_enc_fn(
                    state, num_output=self.total_params, reuse=True)

            # Compute new theta
            theta_here = theta_prev + delta_theta
            # cam = N x 3, pose N x self.num_theta, shape: N x 10
            cams = theta_here[:, :self.num_cam]
            poses = theta_here[:, self.num_cam:(self.num_cam + self.num_theta)]
            shapes = theta_here[:, (self.num_cam + self.num_theta):]
            
            # Rs_wglobal is Nx24x3x3 rotation matrices of poses
            verts, Js, pred_Rs, _ = self.smpl(shapes, poses, get_skin=True, gender = gt_gender)

            pred_kp = batch_orth_proj_idrot(Js, cams, name='proj2d_stage%d' % i)

            # --- Compute losses:
            loss_kps.append(self.e_loss_weight * self.keypoint_loss( self.kp_loader[:,0:self.joint_num], pred_kp))
            pred_Rs = tf.reshape(pred_Rs, [-1, 24, 9])
            if self.use_3d_label:
                #NOTE: updated by CCJ for 3d_loss
                #loss_poseshape, loss_joints = self.get_3d_loss(pred_Rs, shapes, Js)
                print ("[***] using get_smpl_new_loss()")
                #loss_poseshape, loss_joints = self.get_smpl_loss(pred_Rs, shapes, Js, poses)
                loss_poseshape, loss_joints = self.get_smpl_new_loss(Js, shapes,  poses)
                
                loss_3d_params.append(loss_poseshape)
                loss_3d_joints.append(loss_joints)
            
            if self.has_depth_loss:
                # proj_vert2d: N x 6890 x 2
                # pred_depth:  N x 6890
                loss_dep, proj_vert2d, pred_depth = self.get_depth_loss(verts, cams)
                #NOTE: used to debug
                loss_dep, _, pred_depth = self.get_depth_loss(verts, cams)
                #proj_vert2d = batch_orth_proj_idrot(verts, cams, name='proj_vert2d_stage%d' % i) 
                #proj_vert2d = tf.cast(((proj_vert2d + 1.0) * 0.5) * self.img_size, tf.int32) # now in pixel unit;
                loss_depths.append(self.e_weight_depth*loss_dep)
                # Save things for visualiations:
                self.all_pred_depths.append(tf.gather(pred_depth, self.show_these))
                self.all_proj_vert2ds.append(tf.gather(proj_vert2d, self.show_these))

            
            # Save pred_rotations for Discriminator
            fake_rotations.append(pred_Rs[:, 1:, :])
            fake_shapes.append(shapes)

            # Save things for visualiations:
            self.all_verts.append(tf.gather(verts, self.show_these))
            self.all_verts_gt.append(tf.gather(gt_verts, self.show_these))

            self.all_pred_kps.append(tf.gather(pred_kp, self.show_these))
            self.all_pred_cams.append(tf.gather(cams, self.show_these))
            

            # Finally update to end iteration.
            theta_prev = theta_here

        if not self.encoder_only:
            self.setup_discriminator(fake_rotations, fake_shapes)

        # Gather losses.
        with tf.name_scope("gather_e_loss"):
            # Just the last loss.
            self.e_loss_kp = loss_kps[-1]

            if self.encoder_only:
                self.e_loss = self.e_loss_kp # already multiplied by self.e_loss_weight;
            else:
                self.e_loss = self.d_loss_weight * self.e_loss_disc + self.e_loss_kp

            if self.use_3d_label:
                self.e_loss_3d = loss_3d_params[-1]
                self.e_loss_3d_joints = loss_3d_joints[-1]

                self.e_loss += (self.e_loss_3d + self.e_loss_3d_joints)
            # added for depth loss, on Aug 14 2019;
            if self.has_depth_loss:
                self.e_loss_depth = loss_depths[-1]
                self.e_loss += self.e_loss_depth


        if not self.encoder_only:
            with tf.name_scope("gather_d_loss"):
                self.d_loss = self.d_loss_weight * (
                    self.d_loss_real + self.d_loss_fake)

        # For visualizations, only save selected few into:
        # B x T x ...
        self.all_verts = tf.stack(self.all_verts, axis=1)
        self.all_verts_gt = tf.stack(self.all_verts_gt, axis=1)
        self.all_pred_kps = tf.stack(self.all_pred_kps, axis=1)
        self.all_pred_cams = tf.stack(self.all_pred_cams, axis=1)
        self.show_imgs = tf.gather(self.image_loader, self.show_these)
        #NOTE: added by CCJ;
        self.show_deps = tf.gather(self.depth_loader, self.show_these)
        self.show_kps = tf.gather(self.kp_loader, self.show_these)
        
        
        # added by CCJ for depth loss
        if self.has_depth_loss:
            self.all_pred_depths = tf.stack(self.all_pred_depths, axis=1)
            self.all_proj_vert2ds = tf.stack(self.all_proj_vert2ds, axis=1)
            self.show_depths_max = tf.gather(self.depth_max_loader, self.show_these)
        

        # Don't forget to update batchnorm's moving means.
        print('collecting batch norm moving means!!')
        bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if bn_ops:
            self.e_loss = control_flow_ops.with_dependencies(
                [tf.group(*bn_ops)], self.e_loss)

        # Setup optimizer
        print('Setting up optimizer..')
        d_optimizer = self.optimizer(self.d_lr)
        e_optimizer = self.optimizer(self.e_lr)

        self.e_opt = e_optimizer.minimize(
            self.e_loss, global_step=self.global_step, 
            var_list=self.E_var)
        if not self.encoder_only:
            self.d_opt = d_optimizer.minimize(self.d_loss, 
                            var_list=self.D_var)
        
        self.setup_summaries(loss_kps, loss_depths)
        print('Done initializing trainer!')

    def setup_summaries(self, loss_kps, loss_depths):
        # Prepare Summary
        always_report = [
            tf.summary.scalar("loss/e_loss_kp_noscale",
                              self.e_loss_kp / self.e_loss_weight),
            tf.summary.scalar("loss/e_loss", self.e_loss),
        ]
        if self.encoder_only:
            print('ENCODER ONLY!!!')
        else:
            always_report.extend([
                tf.summary.scalar("loss/d_loss", self.d_loss),
                tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
                tf.summary.scalar("loss/d_loss_real", self.d_loss_real),
                tf.summary.scalar("loss/e_loss_disc",
                                  self.e_loss_disc / self.d_loss_weight),
            ])
        # loss at each stage.
        for i in np.arange(self.num_stage):
            name_here = "loss/e_losses_noscale/kp_loss_stage%d" % i
            always_report.append(
                tf.summary.scalar(name_here, loss_kps[i] / self.e_loss_weight))
        
        if self.has_depth_loss: 
            for i in np.arange(self.num_stage):
                name_here = "loss/e_losses_noscale/depth_loss_stage%d" % i
                always_report.append(
                    tf.summary.scalar(name_here, loss_depths[i] / self.e_weight_depth))
        
        if self.use_3d_label:
            always_report.append(
                #tf.summary.scalar("loss/e_loss_3d_params_noscale",self.e_loss_3d / self.e_3d_weight))
                tf.summary.scalar("loss/e_loss_3d_params_noscale",self.e_loss_3d / self.e_3d_weight_smpl))
            always_report.append(
                tf.summary.scalar("loss/e_loss_3d_joints_noscale",self.e_loss_3d_joints / self.e_3d_weight_js3d))

        if not self.encoder_only:
            summary_occ = []
            # Report D output for each joint.
            smpl_names = [
                'Left_Hip', 'Right_Hip', 'Waist', 'Left_Knee', 'Right_Knee',
                'Upper_Waist', 'Left_Ankle', 'Right_Ankle', 'Chest',
                'Left_Toe', 'Right_Toe', 'Base_Neck', 'Left_Shoulder',
                'Right_Shoulder', 'Upper_Neck', 'Left_Arm', 'Right_Arm',
                'Left_Elbow', 'Right_Elbow', 'Left_Wrist', 'Right_Wrist',
                'Left_Finger', 'Right_Finger'
            ]
            # d_out is 25 (or 24), last bit is shape, first 24 is pose
            # 23(relpose) + 1(jointpose) + 1(shape) => 25
            d_out_pose = self.d_out[:, :24]
            for i, name in enumerate(smpl_names):
                summary_occ.append(
                    tf.summary.histogram("d_out/%s" % name, d_out_pose[i]))
            summary_occ.append(
                tf.summary.histogram("d_out/all_joints", d_out_pose[23]))
            summary_occ.append(
                tf.summary.histogram("d_out/beta", self.d_out[:, 24]))

            self.summary_op_occ = tf.summary.merge(
                summary_occ, collections=['occasional'])
        self.summary_op_always = tf.summary.merge(always_report)

    def setup_discriminator(self, fake_rotations, fake_shapes):
        # Compute the rotation matrices of "real" pose, which are from mocap_loader;
        # These guys are in 24 x 3.
        real_rotations = batch_rodrigues(tf.reshape(self.pose_loader, [-1, 3]))
        real_rotations = tf.reshape(real_rotations, [-1, 24, 9])
        # Ignoring global rotation. N x 23*9
        # The # of real rotation is B*num_stage so it's balanced.
        real_rotations = real_rotations[:, 1:, :]
        all_fake_rotations = tf.reshape(
            tf.concat(fake_rotations, 0),
            [self.batch_size * self.num_stage, -1, 9])
        comb_rotations = tf.concat(
            [real_rotations, all_fake_rotations], 0, name="combined_pose")

        comb_rotations = tf.expand_dims(comb_rotations, 2)
        all_fake_shapes = tf.concat(fake_shapes, 0)
        comb_shapes = tf.concat(
            [self.shape_loader, all_fake_shapes], 0, name="combined_shape")

        disc_input = {
            'weight_decay': self.d_wd,
            'shapes': comb_shapes,
            'poses': comb_rotations
        }

        self.d_out, self.D_var = Discriminator_separable_rotations(
            **disc_input)
        
        # comment added by ccj:
        # evenly split to 2 smaller tensors, along the dimension axis = 0 by default; 
        self.d_out_real, self.d_out_fake = tf.split(self.d_out, 2)
        # Compute losses:
        with tf.name_scope("comp_d_loss"):
            self.d_loss_real = tf.reduce_mean(
                tf.reduce_sum((self.d_out_real - 1)**2, axis=1))
            self.d_loss_fake = tf.reduce_mean(
                tf.reduce_sum((self.d_out_fake)**2, axis=1))
            # Encoder loss
            self.e_loss_disc = tf.reduce_mean(
                tf.reduce_sum((self.d_out_fake - 1)**2, axis=1))

    # added by CCJ: NOTE: not used yet ??? 
    """ parameters:
                points : of shape [batch_size N, 24, 3]
                intrinsic_trans: of shape [3, 3] matrix;
                extrinsic_trans: of shape [4, 3] matrix;
                """
    def batch_project_vertices(points, intrinsic_trans, extrinsic_trans):
        num_batch = points.shape[0].value
        home_coords = tf.concat([points, tf.ones([num_batch, 24, 1])], 2) # N X 24 X 4;
        home_coords = tf.reshape(home_coords, [-1, 4]) # N*24 x 4;
        res = tf.matmul(home_coords, tf.matmul(extrinsic_trans, intrinsic_trans)) # N * 24 x 3;
        res = tf.reshape(res, [num_batch, 24, 3])
        return res


    """ added by CCJ on Jul 16, 2019:
        a new 3d loss for 3d joints between gt_js and pred_js, 
        where gt_js = smpl(gt_pose, gt_shape) and 
        pred_js = smpl(pred_pose, pred_shape) 
    """
    def get_smpl_new_loss(self, Js, shapes, poses):
        """
        J_transformed is N x 24 x 3 joint location
        Shape is N x 10
        Js is N x 19 x 3 joints

        Ground truth:
        self.poseshape_loader is a long vector of:
           relative rotation (24*9)
           shape (10)
           3D joints (14*3)
        """
        pred_params = tf.concat([shapes, poses[:, 3:]], 1, name="pred_params")
        # 24*9+10 = 226
        gt_shapes = self.poseshape_loader[:, 216:226]
        POSE_STR_IDX = 268
        POSE_END_IDX = POSE_STR_IDX + 24*3
        gt_poses = self.poseshape_loader[:, POSE_STR_IDX + 3 : POSE_END_IDX]
        gt_params = tf.concat([gt_shapes, gt_poses], 1, name="gt_params")

        #added by CCJ:
        # NOTE: pose seems to be right /left swap, so we do not use pose for loss;
        # NOTE: UPDATED: we ignore the first 3 values in pose for the ROOT joint;
        loss_poseshape = self.e_3d_weight_smpl * compute_3d_loss(pred_params, gt_params,self.has_gt3d_smpl)
        
        # 14*3 = 42
        # in camera system:
        gt_joints = self.poseshape_loader[:, 226:226 + 14*3]
        gt_joints = tf.reshape(gt_joints, [self.batch_size, 14, 3])

        self.smpl_dict_to_save['image']= self.image_loader[0,:,:,:]
        self.smpl_dict_to_save['depth']= self.depth_loader[0,:,:,:]
        self.smpl_dict_to_save['gt_shapes'] = gt_shapes[0,:]
        #self.smpl_dict_to_save['gt_poses'] = tf.reshape(gt_poses[0,:], [-1])
        self.smpl_dict_to_save['gt_joints3d'] = gt_joints[0, :, :]

        self.smpl_dict_to_save['pred_shapes'] = shapes[0,:]
        #self.smpl_dict_to_save['pred_rots']  = tf.reshape( Rs[0, :], [24,9]) # 24 x 3*3;
        self.smpl_dict_to_save['pred_poses']  = tf.reshape( poses[0,:], [-1]) # 24 x 3*3;

        pred_joints = Js[:,:14,:]
        self.smpl_dict_to_save['pred_joints3d'] = pred_joints 
        
        self.smpl_dict_to_save['fname'] = self.fname
        #***********************************************************
        #***********************************************************
        
        # Align the joints by pelvis.
        gt_joints = align_by_pelvis(gt_joints)
        gt_joints = tf.reshape(gt_joints, [self.batch_size, -1])
        
        # Align the joints by pelvis.
        pred_joints = align_by_pelvis(pred_joints)
        pred_joints = tf.reshape(pred_joints, [self.batch_size, -1])

        #added by CCJ:
        loss_joints = self.e_3d_weight_js3d * compute_3d_loss( pred_joints, gt_joints, self.has_gt3d_joints)
        
        return loss_poseshape, loss_joints

    # added by CCJ for incorporating the depth loss;
    # so we make this cam_ro_render opreation inside this fucntion and 
    # make some possible change here and only here!!!
    def get_cam_for_render(self, cam):
        """ 
        cam : [s, tx, ty], 
        return:
           cam_trans: [tx, ty, tz], translation of smpl coordinate frame 
                       w.r.t camera coordinate frame ??
           cam_for_render: camera for rendering the image and 
                       projected vertices from SMPL layer;
        """
        # Fix a flength so i can render this with persp correct scale
        f = 5. 
        tz = f / cam[0]
        cam_for_render = 0.5 * self.img_size * np.array([f, 1, 1])
        cam_trans = np.array([cam[1], cam[2], tz])
        return cam_trans, cam_for_render, f


    def get_depth_loss(self, verts, cams, f = 5.0, is_sigmoid = True):
        """
        verts : N x 6890 x 3, where N is batch_size;
        cams : N x 3, where 3 = S, tx, ty;
        """
        # proj_vert2d: N x 6890 x 2;
        # pred_depth : N x 6890
        proj_vert2d, pred_depth, num_vert = get_proj_vert2d(verts, cams, f, self.img_size)

        # GT depth
        gt_depth = tf.squeeze(self.depth_loader, axis = -1) # N x H x W x 1  ==> N x H X W;
        shape_dep = gt_depth.shape
        # undo scale [-1, 1] gt depth to [0, dep_max];
        gt_depth = self.depth_max_loader * tf.reshape((gt_depth + 1.0)*0.5, [self.batch_size,-1]) # N x -1
        gt_depth = tf.reshape(gt_depth, shape_dep) # N x H x W
        
        # indices along batch dimension
        batch_idxs = tf.reshape(tf.range(0, self.batch_size), [-1, 1, 1]) # N x 1 x 1
        batch_idxs = tf.tile(batch_idxs, [1, num_vert, 1]) # N x 6890 x 1
        
        # > see https://riptutorial.com/tensorflow/example/29069/how-to-use-tf-gather-nd for details;
        # to access elements of gt_depth which is a rank-3 tensor, i.e., 3 = (batch_idx, H_idx, W_idx)
        # the innermost dimension of index_to_pick must have length 3;
        # 6890 x N x 3, here 3 = (batch_idx, H_idx, W_idx)

        #NOTE: updated on Aug 20, 2019!!!
        #proj_vert2d: x, y, i.e, x -> imgW, y -> imgH, so if you want to 
        # get the index of (h_idx, w_idx), you have to change the order (x,y) to (y,x)
        index_to_pick = tf.concat([batch_idxs, proj_vert2d[:,:,::-1]], axis = 2) # N x 6890 x 3 
        gt_depth_picked = tf.gather_nd(gt_depth, index_to_pick) # N x 6890
        print ('[???] gt_depth_picked.shape = ', gt_depth_picked.shape)
        
        # get the loss
        # f(x) = (1-exp(alpha *x))/(1 + exp(alpha *x )) 
        #      = 1.0 - 2.0*(1 / (1 + exp(alpha*x)))
        #      = 1.0 - 2.0* sigmoid(-alpha*x)
        # where, sigmoid(x) = 1 / (1 + exp(-x)) 
        #alpha = 20.0
        alpha = 1.0
        diff = tf.abs(gt_depth_picked - pred_depth)
        if not is_sigmoid: 
            return tf.reduce_mean(diff), proj_vert2d, pred_depth
        else:
            sig_val = 1.0 - 2.0*tf.sigmoid( -diff*alpha)
            return tf.reduce_mean(sig_val), proj_vert2d, pred_depth


    """ added by CCJ on Jul 26, 2019:
        a new 3d loss for 3d joints between gt_js and pred_js, 
        where gt_js = smpl(gt_pose, gt_shape) and 
        pred_js = smpl(pred_pose, pred_shape) 
    """
    def get_smpl_loss(self, Rs, shape, Js, pose):
        """
        Rs is N x 24 x 3*3 rotation matrices of pose
        Shape is N x 10
        Js is N x 19 x 3 joints

        Ground truth:
        self.poseshape_loader is a long vector of:
           relative rotation (24*9)
           shape (10)
           3D joints (14*3)
        """
        Rs = tf.reshape(Rs, [self.batch_size, -1])
        params_pred = tf.concat([Rs, shape], 1, name="prep_params_pred")
        # 24*9+10 = 226
        gt_params = self.poseshape_loader[:, :226]

        #looss_poseshape = self.e_3d_weight * compute_3d_loss(params_pred, gt_params, self.has_gt3d_smpl)
        #added by CCJ:
        loss_poseshape = self.e_3d_weight_smpl * compute_3d_loss(params_pred, gt_params, self.has_gt3d_smpl)
        
        # 14*3 = 42
        gt_shapes = self.poseshape_loader[:, 216:226]
        gt_poses = self.poseshape_loader[:, 268:]
        gt_joints,_ = self.smpl(gt_shapes, gt_poses, get_skin=False, trans = None, idx = 5)
        gt_joints = gt_joints[:, :14, :]
        

        pred_joints = Js[:, :14, :]
        #NOTE: for debugging;
        #gt_joints = tf.Print(gt_joints, [gt_joints[0,0:3,:], pred_joints[0,0:3,:]], 
        #        "few joints gt, and pred joints: ")
        #***********************************************************
        #***********************************************************
        self.smpl_dict_to_save['image']= self.image_loader[0,:,:,:]
        self.smpl_dict_to_save['depth']= self.depth_loader[0,:,:,:]
        
        self.smpl_dict_to_save['gt_shapes'] = gt_shapes[0,:]
        self.smpl_dict_to_save['gt_poses'] = tf.reshape(gt_poses[0,:], [-1])
        self.smpl_dict_to_save['gt_rots'] = tf.reshape(self.poseshape_loader[0,:216], [24, 9]) # 24 x 3*3;
        self.smpl_dict_to_save['gt_joints3d'] = gt_joints[0,:,:]

        self.smpl_dict_to_save['pred_shapes'] = shape[0,:]
        self.smpl_dict_to_save['pred_rots']  = tf.reshape( Rs[0, :], [24,9]) # 24 x 3*3;
        self.smpl_dict_to_save['pred_poses']  = tf.reshape( pose[0,:], [-1]) # 24 x 3*3;
        self.smpl_dict_to_save['pred_joints3d']= pred_joints[0, :, :]
        self.smpl_dict_to_save['fname'] = self.fname
        #***********************************************************
        #***********************************************************
        
        # Align the joints by pelvis.
        gt_joints = align_by_pelvis(gt_joints)
        gt_joints = tf.reshape(gt_joints, [self.batch_size, -1])
        
        # Align the joints by pelvis.
        pred_joints = align_by_pelvis(pred_joints)
        pred_joints = tf.reshape(pred_joints, [self.batch_size, -1])

        #loss_joints = self.e_3d_weight * compute_3d_loss( pred_joints, gt_joints, self.has_gt3d_joints)
        #added by CCJ:
        loss_joints = self.e_3d_weight_js3d * compute_3d_loss( pred_joints, gt_joints, self.has_gt3d_joints)
        
        return loss_poseshape, loss_joints


    def save_to_json(self, global_step, sess):
        param_path = join(self.model_dir, "smpl_iter%06d.json" % global_step)
        dict_to_save = {}
        for k in self.smpl_dict_to_save:
            dict_to_save[k] = sess.run(self.smpl_dict_to_save[k]).tolist() if k is not "fname" else sess.run(self.smpl_dict_to_save[k])
        print ("dict_to_save = {}".format(dict_to_save['fname']))
        with open(param_path, 'w') as fp:
            json.dump(dict_to_save, fp, indent=4, sort_keys=True)


    def get_3d_loss(self, Rs, shape, Js):
        """
        Rs is N x 24 x 3*3 rotation matrices of pose
        Shape is N x 10
        Js is N x 19 x 3 joints

        Ground truth:
        self.poseshape_loader is a long vector of:
           relative rotation (24*9)
           shape (10)
           3D joints (14*3)
        """
        Rs = tf.reshape(Rs, [self.batch_size, -1])
        params_pred = tf.concat([Rs, shape], 1, name="prep_params_pred")
        # 24*9+10 = 226
        gt_params = self.poseshape_loader[:, :226]
        loss_poseshape = self.e_3d_weight * compute_3d_loss(
            params_pred, gt_params, self.has_gt3d_smpl)
        # 14*3 = 42
        gt_joints = self.poseshape_loader[:, 226:268]
        pred_joints = Js[:, :14, :]
        gt_joints = tf.reshape(gt_joints, [self.batch_size, 14, 3])
        
        #NOTE: for debugging;
        #gt_joints = tf.Print(gt_joints, [gt_joints[0,0:3,:], pred_joints[0,0:3,:]], 
        #        "few joints gt, and pred joints: ")
        
        # Align the joints by pelvis.
        gt_joints = align_by_pelvis(gt_joints)
        gt_joints = tf.reshape(gt_joints, [self.batch_size, -1])
        
        # Align the joints by pelvis.
        pred_joints = align_by_pelvis(pred_joints)
        pred_joints = tf.reshape(pred_joints, [self.batch_size, -1])

        loss_joints = self.e_3d_weight * compute_3d_loss(
            pred_joints, gt_joints, self.has_gt3d_joints)
        return loss_poseshape, loss_joints



    #def visualize_img(self, img, gt_kp, vert, pred_kp, cam, renderer):
    #NOTE: updated by CCJ on July 1st, 2019;
    def visualize_img(self, img, gt_kp, vert, pred_kp, cam, renderer, 
                      gt_vert = None, gt_cam = None, 
                      # newly added on Aug 20, 2019
                      pred_depth = None, # (6890,)
                      proj_vert2d = None, # (6890, 2)
                      depth_max = None # (1,)
                      ):
        """
        Overlays gt_kp and pred_kp on img.
        Draws vert with text.
        Renderer is an instance of SMPLRenderer.
        """
        gt_kp = gt_kp[0:self.joint_num,:]
        gt_vis = gt_kp[:, 2].astype(bool)
        loss = np.sum((gt_kp[gt_vis, :2] - pred_kp[gt_vis])**2)
        debug_text = {"sc": cam[0], "tx": cam[1], "ty": cam[2], "kpl": loss}
        
        # Fix a flength so i can render this with persp correct scale
        #f = 5. 
        #tz = f / cam[0]
        #cam_for_render = 0.5 * self.img_size * np.array([f, 1, 1])
        #cam_trans = np.array([cam[1], cam[2], tz])
        cam_trans, cam_for_render, f = self.get_cam_for_render(cam)
        
        # Undo pre-processing.
        input_img = (img + 1) * 0.5 # rescale to [0, 1]
        rend_img = renderer(vert + cam_trans, cam_for_render, img=input_img)
        rend_img = vis_util.draw_text(rend_img, debug_text)
        

        #gt_rendering
        if gt_vert is not None:
            debug_text_gt = {"sc_gt": gt_cam[0], "tx_gt": gt_cam[1], "ty_gt": gt_cam[2], "kpl": loss}
            cam_t_gt = np.array([gt_cam[1], gt_cam[2], f/ gt_cam[0]])
            rend_img_gt = renderer(gt_vert + cam_t_gt, cam_for_render, img=input_img)
            rend_img_gt = vis_util.draw_text(rend_img_gt, debug_text_gt)

        # Draw skeleton
        gt_joint = ((gt_kp[:, :2] + 1) * 0.5) * self.img_size
        pred_joint = ((pred_kp + 1) * 0.5) * self.img_size
        img_with_gt = vis_util.draw_skeleton(
            input_img, gt_joint, draw_edges=False, vis=gt_vis)
        skel_img = vis_util.draw_skeleton(img_with_gt, pred_joint)
        
        # newly added for depth rendering;
        if self.has_depth_loss:
            rend_dep = renderer.depth_render(
                depth_max, 
                vert + cam_trans,
                cam_for_render,
                img_size = [self.img_size, self.img_size]
                )
            # change it to color
            rend_dep = cv2.cvtColor(rend_dep, cv2.COLOR_GRAY2RGB)
            # a while line bourdary for visualization only 
            rend_dep[:, self.img_size-3:self.img_size] = (255, 255, 255)
            rend_dep[self.img_size-3:self.img_size, :] = (255, 255, 255)
            
            rend_dep_wigh_gt = vis_util.draw_skeleton(
                rend_dep, gt_joint, draw_edges=False, vis=gt_vis)
            
            
            skel_dep = vis_util.draw_skeleton(rend_dep_wigh_gt, pred_joint)
            
            myproj_dep = np.zeros((self.img_size, self.img_size, 2), dtype= np.float32)
            # pred_depth : (6890,)
            # proj_vert2d : (6890, 2)
            
            #print ("[???] shapes = {}, {}, {}, {}, {}".format(
            #    skel_img.shape, 
            #    rend_img.shape,
            #    skel_dep.shape,
            #    myproj_dep.shape,
            #    pred_depth.shape))

            for i in range(0, pred_depth.shape[0]):
                x,y = proj_vert2d[i]
                x = min(x, self.img_size - 1)
                y = min(y, self.img_size - 1)
                #print ("??? x,y = {}, {}".format(x, y))
                myproj_dep[y, x, 0] += pred_depth[i]
                myproj_dep[y, x, 1] += 1
            nums = myproj_dep[:,:,1]
            nums [nums < 1.0] = 1.0 
            #print ("??? nums.shape = {}".format(nums.shape))

            myproj_dep = myproj_dep[:,:, 0]/ nums
            myproj_dep /= depth_max
            myproj_dep *= 255.0
            myproj_dep = myproj_dep.astype(np.uint8) 
            myproj_dep = cv2.cvtColor(myproj_dep, cv2.COLOR_GRAY2RGB)
            # a while line bourdary for visualization only 
            myproj_dep[:, self.img_size-3:self.img_size] = (255, 255, 255)
            myproj_dep[self.img_size-3:self.img_size, :] = (255, 255, 255)
            #print ("[???] myproj_dep shape = {}".format(myproj_dep.shape))
            # (H,W) -> (H, W, C)

        to_combined = [skel_img, rend_img/ 255.,]

        if gt_vert is not None:
            to_combined.append(rend_img_gt / 255.)
        if self.has_depth_loss:
            to_combined.append( skel_dep)
            to_combined.append( myproj_dep)
        
        #print ("[???] shapes = {}, {}, {}, {}".format(
        #    skel_img.shape, 
        #    rend_img.shape,
        #    skel_dep.shape,
        #    myproj_dep.shape))

        combined = np.hstack(to_combined)        
        
        #if gt_vert is not None:
        #    combined = np.hstack([skel_img, rend_img / 255., rend_img_gt / 255. ])
        #    if 
        #else:
        #    combined = np.hstack([skel_img, rend_img / 255.])

        # import matplotlib.pyplot as plt
        # plt.ion()
        # plt.imshow(skel_img)
        # import ipdb; ipdb.set_trace()
        return combined

    def draw_results(self, result):
        from StringIO import StringIO
        import matplotlib.pyplot as plt

        # This is B x H x W x 3
        imgs = result["input_img"]
        deps = result["input_dep"]
        #print ("[**] type deps and shape : ", type(deps), deps.shape)
        # B x 14 x 3
        gt_kps = result["gt_kp"]
        if self.data_format == 'NCHW':
            imgs = np.transpose(imgs, [0, 2, 3, 1])
            deps = np.transpose(deps, [0, 2, 3, 1])
        # This is B x T x 6890 x 3
        # NOTE: Here T is the Regressor iteration, e.g, T = 3;
        est_verts = result["e_verts"]
        gt_verts = result["gt_verts"]
        # B x T x 19 x 2
        joints = result["joints"]
        # B x T x 3
        cams = result["cam"]
        
        if self.has_depth_loss:
            pred_depths = result['pred_depths'] # B x T x 6890
            proj_vert2ds = result['proj_vert2ds'] # B x T x 6890 x 2
            show_depths_max = result['show_depths_max'] # B x 1
        else:
            # just set dummy variables for the following enumerate();
            pred_depths = np.zeros((deps.shape[0]))
            proj_vert2ds = np.zeros((deps.shape[0]))
            show_depths_max = np.zeros((deps.shape[0]))
        
        img_summaries = []

        for img_id, (img, dep, gt_kp, verts, gtverts, joints, cams, pred_depths, proj_vert2ds, dep_max ) in enumerate(
                zip(imgs, deps, gt_kps, est_verts, gt_verts, joints, cams, pred_depths, proj_vert2ds, show_depths_max)):
            
            # verts, joints, cams are a list of len T.
            all_rend_imgs = []
            for vert, gt_vert, joint, cam, pred_depth, proj_vert2d in zip(verts, 
                gtverts, joints, cams, pred_depths, proj_vert2ds): # T length
                
                rend_img = self.visualize_img(img, gt_kp, vert, joint, cam, self.renderer, 
                        gt_vert, cam, 
                        # newly added on Aug 20, 2019
                        pred_depth, proj_vert2d, dep_max
                        )
                all_rend_imgs.append(rend_img)
            combined = np.vstack(all_rend_imgs)
            #print ("[**] type combined  and shape : ", type(combined), combined.shape)

            sio = StringIO()
            plt.imsave(sio, combined, format='png')
            vis_sum = tf.Summary.Image(
                encoded_image_string=sio.getvalue(),
                height=combined.shape[0],
                width=combined.shape[1])
            img_summaries.append(
                tf.Summary.Value(tag="vis_images/%d" % img_id, image=vis_sum))
            #NOTE:
            sio2= StringIO()
            #NOTE: problem was that an array of shape (nx,ny,1) is still 
            # considered a 3D array, 
            # and must be squeezed or sliced into a 2D array.
            plt.imsave(sio2, np.squeeze(dep, -1), format = 'png')
            #print ("[**] type dep and shape : ", type(dep), dep.shape)
            img_summaries.append(
                    tf.Summary.Value(tag="vis_depths/%d" % img_id, 
                        image= tf.Summary.Image(
                            encoded_image_string=sio2.getvalue(),
                            height = dep.shape[0], 
                            width=dep.shape[1])
                    )
                )

        img_summary = tf.Summary(value=img_summaries)
        self.summary_writer.add_summary(
            img_summary, global_step=result['step'])

    

    def train(self):
        # For rendering!
        self.renderer = vis_util.SMPLRenderer(
            img_size=self.img_size,
            face_path=self.config.smpl_face_path)

        step = 0

        with self.sv.managed_session(config=self.sess_config) as sess:
            while not self.sv.should_stop():
                fetch_dict = {
                    "summary": self.summary_op_always,
                    "step": self.global_step,
                    "e_loss": self.e_loss,
                    # The meat
                    "e_opt": self.e_opt,
                    "loss_kp": self.e_loss_kp,
                    #added by CCJ for debugging;
                    #'joints3d_gt': self.poseshape_loader[0, 226:226 + 14*3],
                    #'joints2d_gt': self.kp_loader[0,:]
                }
                if not self.encoder_only:
                    fetch_dict.update({
                        # For D:
                        "d_opt": self.d_opt,
                        "d_loss": self.d_loss,
                        "loss_disc": self.e_loss_disc,
                    })
                if self.use_3d_label:
                    fetch_dict.update({
                        "loss_3d_params": self.e_loss_3d,
                        "loss_3d_joints": self.e_loss_3d_joints
                    })

                if self.has_depth_loss:
                    fetch_dict.update({
                        "loss_depth": self.e_loss_depth,
                    })

                if step % self.log_img_step == 0:
                    fetch_dict.update({
                        "input_img": self.show_imgs,
                        # added by CCJ:
                        "input_dep": self.show_deps,
                        "gt_kp": self.show_kps,
                        "e_verts": self.all_verts,
                        "gt_verts": self.all_verts_gt,
                        "joints": self.all_pred_kps,
                        "cam": self.all_pred_cams,
                    })
                    if not self.encoder_only:
                        fetch_dict.update({
                            "summary_occasional":
                            self.summary_op_occ
                        })
                    
                    if self.has_depth_loss:
                        fetch_dict.update({
                            "pred_depths": self.all_pred_depths,
                            "proj_vert2ds": self.all_proj_vert2ds,
                            "show_depths_max": self.show_depths_max,
                        })

                t0 = time()
                result = sess.run(fetch_dict)
                t1 = time()

                self.summary_writer.add_summary(
                    result['summary'], global_step=result['step'])

                e_loss = result['e_loss']
                step = result['step']

                #NOTE:added by CCJ
                if False:
                    if step in [
                            self.step_50_to_save_json, 4*self.step_50_to_save_json, 
                            50*self.step_50_to_save_json, 100*self.step_50_to_save_json, 
                            150*self.step_50_to_save_json, 200*self.step_50_to_save_json,
                            self.max_epoch*self.num_itr_per_epoch - 100,
                            self.max_epoch*self.num_itr_per_epoch - 10,
                            ]:
                        self.save_to_json(step, sess)
                    

                epoch = float(step) / self.num_itr_per_epoch
                if self.encoder_only:
                    print("itr %d/(epoch %.1f): time %g, Enc_loss: %.4f" %
                          (step, epoch, t1 - t0, e_loss))
                    #print("gt_joints3d = {}".format(result['joints3d_gt']))
                    #print("gt_joints2d = {}".format(result['joints2d_gt']))

                else:
                    d_loss = result['d_loss']
                    print(
                        "itr %d/(epoch %.1f): time %g, Enc_loss: %.4f, Disc_loss: %.4f"
                        % (step, epoch, t1 - t0, e_loss, d_loss))

                if step % self.log_img_step == 0:
                    if not self.encoder_only:
                        self.summary_writer.add_summary(
                            result['summary_occasional'],
                            global_step=result['step'])
                    self.draw_results(result)

                self.summary_writer.flush()
                if epoch > self.max_epoch:
                    self.sv.request_stop()

                step += 1

        print('Finish training on %s' % self.model_dir)
