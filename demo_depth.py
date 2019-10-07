"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

""" added by CCJ:
    > see problem : _tkinter.TclError: no display name and no $DISPLAY environment variable, at https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable
    You can solve it by adding these two lines in the VERY beginning of your *.py script.
    It should in the very beginning of the code. This is important.
"""

from os.path import join, exists
import matplotlib
matplotlib.use('Agg')

import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
from src.load_data_4_inference import data_loader_for_inference as load
from src.load_data_4_inference import extract_14_joints
from src.util import surreal_in_extrinc as surreal_util

from src.RunModelDepth import RunModelV2
from src.RunModel import RunModel as RunModelV1
#added by ccj;
import src.pfmutil as pfm
import cv2
import os

from src.config import get_config
from src.pose_perceptron import (get_one_batched_cad_toy_example,
                 draw_lsp_skeleton, save_to_mesh_file)
from src.benchmark.eval_util import align_by_pelvis,compute_error_one_sample
from datetime import datetime

import src.pfmutil as pfm
import deepdish as dd

#flags.DEFINE_string('depth_fname', 'data/cad-60-small/dep-scale-RGB_20.pfm', 'depth image to run')
#flags.DEFINE_string('image_fname', 'data/cad-60-small/img-scale-RGB_20.pfm', 'depth image to run')
#flags.DEFINE_integer('gender', 2, 'femael :0, male : 1, neutral = 2')

#flags.DEFINE_string('info_fname', 'data/im1963.jpg', 'info file run')
#flags.DEFINE_string('result_dir', 'data/im1963.jpg', 'results dir to save files')
#flags.DEFINE_integer('t_beg', 0, 'frame begin idx')
#flags.DEFINE_integer('t_end', 1, 'frame end idx')
#flags.DEFINE_string('json_path', None, 'If specified, uses the openpose output to crop the image.')


def visualize(img, proc_param, joints, verts, cam, save_fig_result):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])

    import matplotlib.pyplot as plt
    # plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)

    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(skel_img)
    
    plt.title('joint projection')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(234)

    plt.imshow(rend_img)
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(rend_img_vp1)
    
    plt.title('diff vp')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(rend_img_vp2)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    
    """ 
    > see https://hub.docker.com/r/dawars/hmr/
    Matplotlib cannot open a window in docker (by default), 
    therefore it needs to replaced by saving the figures 
    instead: In the demo.py change plt.show() to plt.savefig("figure.png")
    """
    # added by CCJ;
    dockerEnv = True
    if not dockerEnv:
        plt.show()
    else:
        plt.savefig(save_fig_result)
        print ("saved %s ..." % save_fig_result)
    # import ipdb
    # ipdb.set_trace()
    return cam_for_render, joints_orig



def visualize_joints2d_3kinds(img, joints1, joints2, joints3, save_fig_result):
    """
    Renders the result in original image coordinate frame.
    """

    # Render results
    skel_img1 = vis_util.draw_skeleton(img, joints1)
    skel_img2 = vis_util.draw_skeleton(img, joints2)
    skel_img3 = vis_util.draw_skeleton(img, joints3)

    import matplotlib.pyplot as plt
    # plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(311)
    plt.imshow(skel_img1)
    plt.title('joints2d_gt')
    plt.axis('off')
    
    plt.subplot(312)
    plt.title('joints 3d smpl ext/intric projection')
    plt.imshow(skel_img2)
    plt.axis('off')
    
    plt.subplot(313)
    plt.imshow(skel_img3)
    plt.title('joints 3d tf smpl ext/intric projection')
    plt.axis('off')

    plt.draw()
    
    """ 
    > see https://hub.docker.com/r/dawars/hmr/
    Matplotlib cannot open a window in docker (by default), 
    therefore it needs to replaced by saving the figures 
    instead: In the demo.py change plt.show() to plt.savefig("figure.png")
    """
    # added by CCJ;
    dockerEnv = True
    if not dockerEnv:
        plt.show()
    else:
        plt.savefig(save_fig_result)
        print ("saved %s ..." % save_fig_result)
    # import ipdb
    # ipdb.set_trace()



def preprocess_image(img, depth, json_path=None, joints2d_gt=None, cam_gt=None):

    #img = io.imread(img_path)
    #if img.shape[2] == 4:
    #    img = img[:, :, :3]
    #if depth_path is not None:
    #    if ".pfm" in depth_path:
    #        dep = pfm.load_pfm(depth_path)
    #    else:
    #        dep = io.imread(depth_path)
    #else:
    #    dep = np.zeros(img.size, dtype = np.float32)
    
    if img.shape[2] == 4:
        img = img[:, :, :3]
    depth = np.reshape(depth, [depth.shape[0], depth.shape[1], 1])
    img_orig = img
    img = np.concatenate([img, depth], -1)

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            #print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)
    if joints2d_gt is not None:
        crop, proc_param, joints2d_gt_scaled,cam_gt_scaled = img_util.scale_and_crop_with_gt(
            img, scale, center, config.img_size, joints2d_gt, cam_gt)

    else:
        joints2d_gt_scaled = None
        cam_gt_scaled = None
        crop, proc_param  = img_util.scale_and_crop(img, scale, center, config.img_size)

    # Normalize image to [-1, 1]
    crop_img = crop[:,:, 0:3]
    crop_depth = np.reshape(crop[:,:,3], [crop.shape[0], crop.shape[1], 1])
    crop_img = 2 * ((crop_img / 255.) - 0.5)
    depth_max = np.max(crop_depth)
    crop_depth = 2.0*(crop_depth / depth_max - 0.5)
    return crop_img, crop_depth, proc_param, img_orig, joints2d_gt_scaled, cam_gt_scaled


def infer_surreal_debug_with_info_print(args, json_path=None):
    sess = tf.Session()
    """ new model with depth"""
    model = RunModelV2(config, sess=sess, has_smpl_gt = True)
    """ original hmr model w/o depth"""
    #model = RunModelV1(config, sess=sess)

    # loading surreal dataset 
    all_images, all_depths, all_gt_poses, all_gt_shapes, all_gt_genders, all_gt_trans, \
            all_gt_joints2d, all_gt_joints3d, all_gt_smpl_verts, \
            all_gt_smpl_joints3d, all_gt_smpl_joints3d_proj, gt_cam, \
            intrinsic, extrinsic, kintree_table = load( args,
            isSave = False, isNormalized = False)
    
    for i in range(0,len(all_images)):
        img = all_images[i]
        dep = all_depths[i]
        gt_pose = all_gt_poses[i]
        gt_shape = all_gt_shapes[i]
        gt_gender = all_gt_genders[i]
        gt_trans = all_gt_trans[i]
        gt_joints2d = all_gt_joints2d[i]
        gt_joints3d = all_gt_joints3d[i]
        
        gt_cam[0] = gt_cam[0] / 500.
        print ("gt_cam = {}".format(gt_cam))
        input_img, input_dep, proc_param, img_orig, gt_joints2d, gt_cam = preprocess_image(img, dep, json_path, 
                gt_joints2d, gt_cam)
        print ("After preprocess_image : gt_cam = {}".format(gt_cam))
        # Add batch dimension: 1 x D x D x 3
        input_img = np.expand_dims(input_img, 0)
        input_dep = np.expand_dims(input_dep, 0)
        
        gt_pose = np.expand_dims(gt_pose, 0)
        gt_shape = np.expand_dims(gt_shape, 0)
        gt_gender = np.expand_dims(gt_gender, 0)
        gt_cam = np.expand_dims(gt_cam, 0)
        print ("[**??**] input_img shape = {}, input_dep shape = {}".format(input_img.shape, input_dep.shape))

        # Theta is the 85D vector holding [camera, pose, shape]
        # where camera is 3D [s, tx, ty]
        # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
        # shape is 10D shape coefficients of SMPL

        
        joints, verts,  cams, joints3d, joints_gt, verts_gt, joints3d_gt, \
                joints3d_gt_24, theta, beta, joints3d_24 = model.predict(
                        input_img, input_dep, gt_pose, gt_shape, gt_cam, gt_gender, get_theta = True)
        print ("[***] prediction done !")
        

        if not exists(args.result_dir):
            os.makedirs(args.result_dir)
            print ("makedirs %s" % args.result_dir)
        tmp_name = args.info_fname.split('/')[-1][:-9] + "frame_%03d" % i
        
        #*********************
        """ 1)predicted """
        #*********************
        save_fig_result = join(args.result_dir, tmp_name + '_fig_predict.png')
        cam_for_render, joints_orig = visualize(img_orig, proc_param, joints[0], verts[0], cams[0], save_fig_result)
        
        #*********************
        """ 2) GT via tf.smpl() model """
        #*********************
        save_fig_gt = join(args.result_dir, tmp_name + '_fig_gt.png')
        # joints_gt shape : [batch_size, 14, 2];
        # joints3d_gt shape : [batch_size, 14, 3];
        # verts_gt shape: [batch_size, 6890, 3];
        #cam_for_render_gt, joints_gt_orig = visualize(img_orig, proc_param, joints_gt[0], verts_gt, cams[0], save_fig_gt)
        cam_for_render_gt, joints_gt_orig = visualize(img_orig, proc_param, joints_gt[0], verts_gt, cams[0], save_fig_gt)
        print ('shape: joints_gt = {}, joints3d_gt = {}, verts_gt = {}'.format(
                joints_gt.shape, joints3d_gt.shape, verts_gt.shape))
         
        #*********************
        """ 3) GT via the regular python smpl model according to `smpl_relations.py` in SURREAL dataset; """
        #*********************
        gt_smpl_surreal_verts = all_gt_smpl_verts[i] # shape [6890, 3]
        gt_smpl_surreal_joints3d = all_gt_smpl_joints3d[i] # in shape [14, 3];
        print ('shape: gt_joints2d = {}, gt_joints3d = {}, gt_smpl_surreal_verts = {}, gt_smpl_surreal_joints3d = {}'.format(
            gt_joints2d.shape, gt_joints3d.shape, gt_smpl_surreal_verts.shape, 
            gt_smpl_surreal_joints3d.shape))
        save_fig_gt_surreal = join(args.result_dir, tmp_name + '_fig_gt_surreal.png')
        # gt_joints3d: in shape [3, 14];
        # gt_joints2d: in shape [2, 14], so we use ".T" to make it be [14, 2], required by visualize() func;
        #visualize(img_orig, proc_param, gt_joints2d.T, gt_smpl_surreal_verts, gt_cam[0], save_fig_gt_surreal)
        cam_for_render_gt_python, gt_joints2d_orig = visualize(img_orig, proc_param, 
                gt_joints2d.T, gt_smpl_surreal_verts, cams[0], save_fig_gt_surreal)
       
        """ save parameters to json files """
        json_fname =  join(args.result_dir, tmp_name + '_params.json')  
        dict_to_save = {}
        dict_to_save['fname'] = tmp_name
        dict_to_save['joints2d_pd_tfSMPL'] = joints[0] # [0] is along batch_size dimension, here we just set batch_size = 1;
        dict_to_save['joints3d_pd_tfSMPL'] = joints3d[0] # [0] is along batch_size dimension, here we just set batch_size = 1;
        dict_to_save['cam_pd_orig'] = cam_for_render
        dict_to_save['cam_pd'] = cams[0]
        
        #dict_to_save['joints2d_gt_tfSMPL_ortho_proj'] = joints_gt[0] # [0] is along batch_size dimension, here we just set batch_size = 1;
        dict_to_save['joints2d_gt_tfSMPL_ortho_proj'] = joints_gt_orig 
        dict_to_save['joints3d_gt_tfSMPL'] = joints3d_gt[0] # [0] is along batch_size dimension, here we just set batch_size = 1;
        
        
        # applying trans to smpl joints;
        trans = gt_trans
        print ('transformed root_joint3d = {}'.format( joints3d_gt_24[0][0,:] + trans))
        """ regressored 14 joints from tf.smpl() layer;"""
        #dict_to_save['joints2d_gt_tfSMPL_exint_proj'] = surreal_util.project_vertices(
        #        joints3d_gt[0] + trans, intrinsic, extrinsic) # in shape [14, 3]
        
        #print ("[*****????] root_pos shape = {}, gt_joints3d shape = {}, trans = {}".format(root_pos.shape, gt_joints3d.shape, trans))
        #print ("[????] intrinsic in tf smpl  = {}".format(intrinsic))
        #print ("[????] extrinsic in tf smpl  = {}".format(extrinsic))
        
        #"""""""""""""""""
        #NOTE: debugging
        print ("[**] python parents = ", kintree_table[0])
        #surreal_util.draw_joints2D(
        #        img_orig,
                # 24 joints 
        #        surreal_util.project_vertices(joints3d_gt_24[0] + trans, intrinsic, extrinsic),
        #        None, kintree_table, color = 'b')

        #print ("smpl_joints3D shape = ", smpl_joints3D.shape) # in shape [24, 3]

        joints3d_gt_tfsmpl = extract_14_joints(joints3d_gt_24[0] + trans,
                    surreal_util.get_lsp_idx_from_smpl_joints()) 
        print ("[***] joints3d_gt_tfsmpl shape = ", joints3d_gt_tfsmpl.shape)
        dict_to_save['joints2d_gt_tfSMPL_exint_proj'] = surreal_util.project_vertices(
                joints3d_gt_tfsmpl, intrinsic, extrinsic) # in shape [14, 3]
        
        dict_to_save['cam_gt_orig'] = cam_for_render_gt
        dict_to_save['cam_gt'] = gt_cam[0]
        print (" dict_to_save['cam_gt'] = {}".format( dict_to_save['cam_gt']))

        #dict_to_save['joints2d_gt_surreal'] = gt_joints2d.T # in shape [14, 2] 
        dict_to_save['joints2d_gt_surreal'] = gt_joints2d_orig # in shape [14, 2] 
        dict_to_save['joints3d_gt_surreal'] = gt_joints3d.T # in shape [14, 3]
        dict_to_save['joints2d_gt_pySMPL_exint_proj'] = all_gt_smpl_joints3d_proj[i] # in shape [14, 2]
        dict_to_save['joints3d_gt_pySMPL_surreal'] = gt_smpl_surreal_joints3d # in shape [14, 3]
        #print ("dict_to_save['joints3d_gt_pySMPL_surreal'] = {}".format( dict_to_save['joints3d_gt_pySMPL_surreal']))

        dict_to_save['pose_pd'] = theta[0]
        dict_to_save['pose_gt'] =  gt_pose[0]
        dict_to_save['shape_pd'] = beta[0]
        dict_to_save['shape_gt'] = gt_shape[0]

        save_to_json(dict_to_save, json_fname)
        
        def get_cam_joints3d(extrinsic, src):
            assert src.shape[0] == 3 and src.shape[1] == 14
            homo = np.concatenate((src,np.ones([1,14]).astype(np.float32)),axis = 0)
            #print ("home = ", homo)
            return np.dot(extrinsic, homo).T
        
        """ (1) 2d joints in image coordinate system """ 
        print ("\njoint idx, gt(x,y) |  GT python smpl joints3d_proj(x,y) | GT tf.smpl joints3d_proj(x,y) | pred tf.smpl joint2d(x, y)")
        lsp_joints = [ 'R ankle', 'R knee',  'R hip', 'L hip', 'L knee', 'L ankle',
                       'R Wrist', 'R Elbow', 'R shoulder','L shoulder','L Elbow',
                       'L Wrist','Neck top','Head top',]
        for j in range(0, 14):
            print ("%02d  %10s  (%4.2f, %4.2f)   (%4.2f, %4.2f)   (%4.2f, %4.2f)  (%4.2f, %4.2f)" % (
                j, lsp_joints[j], 
                gt_joints2d_orig[j,0], 
                gt_joints2d_orig[j,1], 
                all_gt_smpl_joints3d_proj[i][j,0],
                all_gt_smpl_joints3d_proj[i][j,1],
                dict_to_save['joints2d_gt_tfSMPL_exint_proj'][j,0],
                dict_to_save['joints2d_gt_tfSMPL_exint_proj'][j,1],
                joints_orig[j,0],
                joints_orig[j,1],
                ))
        
        """ (2) 3d joints in camera coordinate system """ 
        print ("\n\njoint idx, joints3d_cam_gt | GT python smpl joints3d_cam | GT tf.smpl joints3d_cam | pred tf.smpl | pred_tf.smpl - joints3d_cam_gt")
        tmp_home_pad = np.array([1.0] * 14)
        pred_joints_14 = extract_14_joints(joints3d_24[0],surreal_util. get_lsp_idx_from_smpl_joints())
        gt_joints3d_cam = get_cam_joints3d(extrinsic, gt_joints3d.T)
        for j in range(0, 14):
            print ("%02d  %10s  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)" % (
                j, lsp_joints[j], 
                gt_joints3d_cam[j,0],
                gt_joints3d_cam[j,1],
                gt_joints3d_cam[j,2],

                get_cam_joints3d(extrinsic, gt_smpl_surreal_joints3d.T)[j,0],
                get_cam_joints3d(extrinsic, gt_smpl_surreal_joints3d.T)[j,1],
                get_cam_joints3d(extrinsic, gt_smpl_surreal_joints3d.T)[j,2],
                
                get_cam_joints3d(extrinsic, joints3d_gt_tfsmpl.T)[j,0],
                get_cam_joints3d(extrinsic, joints3d_gt_tfsmpl.T)[j,1],
                get_cam_joints3d(extrinsic, joints3d_gt_tfsmpl.T)[j,2],

                pred_joints_14[j,0],
                pred_joints_14[j,1],
                pred_joints_14[j,2],

                pred_joints_14[j,0] - gt_joints3d_cam[j,0],
                pred_joints_14[j,1] - gt_joints3d_cam[j,1],
                pred_joints_14[j,2] - gt_joints3d_cam[j,2],
                ))

        """ (3) 3d joints in world coordinate system """ 
        print ("\n\njoint idx, joints3d_world_gt | python smpl joints3d_world | tf.smpl joints3d_world")
        for j in range(0, 14):
            print ("%02d  %10s  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)  (%4.2f, %4.2f, %4.2f)" 
                    %(
                j, lsp_joints[j],
                gt_joints3d[j,0], 
                gt_joints3d[j,1], 
                gt_joints3d[j,2],

                gt_smpl_surreal_joints3d[j,0],
                gt_smpl_surreal_joints3d[j,1],
                gt_smpl_surreal_joints3d[j,2],
                
                joints3d_gt_tfsmpl[j,0],
                joints3d_gt_tfsmpl[j,1],
                joints3d_gt_tfsmpl[j,2]))
            
        save_fig_all_joints2d = join(args.result_dir, tmp_name + '_fig_joints2d_all.png')
        #visualize_joints2d_3kinds(all_images[i], gt_joints2d_orig, all_gt_smpl_joints3d_proj[i], 
        #        dict_to_save['joints2d_gt_tfSMPL_exint_proj'],save_fig_all_joints2d)
        visualize_joints2d_3kinds(all_images[i], all_gt_joints2d[i], 
                all_gt_smpl_joints3d_proj[i], 
                dict_to_save['joints2d_gt_tfSMPL_exint_proj'],save_fig_all_joints2d)



def save_to_json(dict_to_save, json_fname):
    import json
    new_dict = {}
    for k in dict_to_save:
        new_dict[k] = dict_to_save[k].tolist() if k is not "fname" else dict_to_save[k]
    with open(json_fname, 'w') as fp:
        json.dump(new_dict, fp, indent = 4, sort_keys = True)
    print ("Saved json file : %s ..." % json_fname)




""" simple infer 1 image (already scaled to image_size = 224 around) """
def simple_infer_1_img(image_path, 
                       depth_path, 
                       gender = 'n', 
                       json_path = None, 
                       result_dir = '', 
                       datatype = 'cad120',
                       has_joints3dgt = False,
                       smpl_model_path = None,
                       take_home_message = None, # some note about this experiment;
                       isSaveResults = False
                       ):
    sess = tf.Session()
    """ new model with depth"""
    model = RunModelV2(config, sess=sess)
    """ original model w/o depth"""
    #model = RunModelV1(config, sess=sess)
    

    # read image and depth; and normalized and rescaled to [-1.0, 1.0];
    if '.pfm' in image_path:
        img = pfm.load(image_path).astype(np.float32)
    else:
        img = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)
    if depth_path is not None:
        dep = pfm.load(depth_path).astype(np.float32)
    else:
        dep = np.zeros([img.size[0], img.size[1]], dtype = np.float32)

    input_img, input_dep, proc_param, img_orig, _, _ = preprocess_image(img, dep, json_path, None, None)
    # Add batch dimension: 1 x D x D x 3
    input_img = np.expand_dims(input_img, 0)
    input_dep = np.expand_dims(input_dep, 0)
    
     
    #*********************
    """ 1) predicted """
    #*********************
    if gender == 'f':
        gender_vect = np.array([1.0, .0, .0]).astype(np.float32) # gender vector - female
    elif gender == 'm':
        gender_vect = np.array([.0, 1.0, .0]).astype(np.float32) # gender vector - male
    elif gender == 'n':
        gender_vect = np.array([.0, .0, 1.0]).astype(np.float32) # gender vector - neutral
    gt_gender = np.expand_dims(gender_vect, 0)
    joints, verts, cams, joints3d, _, _, _, _, theta, beta, joints3d_24 = model.predict(
        input_img, input_dep, None, None, None, gt_gender, get_theta = True)
    print ("[***] prediction done !")
    
    #*********************
    """ 2) evaluation """
    #*********************
    gender_cad_dict = {
        'cad-120/Subject1_rgbd_images': 'f',
        'cad-120/Subject3_rgbd_images': 'f',
        'cad-120/Subject4_rgbd_images': 'm',
        'cad-120/Subject5_rgbd_images': 'm',
        'cad-60/Person1' : 'm',
        'cad-60/Person2' : 'f',
        'cad-60/Person3' : 'f',
        'cad-60/Person4' : 'm',
    }
    if has_joints3dgt:
        print ("[***] evaluating MPJPE and MPJPE PA !")
        if datatype == 'cad120':
            # e.g., image_path = 'cad-120-small/Subject4_rgbd_images/taking_medicine-1130145737-RGB_114_img_extracted.pfm'
            cur_sub = image_path.split('/')[-2]
            gender_cad = gender_cad_dict['cad-120/' + cur_sub]
            tmp_name = image_path.split('/')[-1]
            cad_anno_txt_fname = 'cad-120/' + cur_sub + '/' + tmp_name.split('-')[0] + '/'+ tmp_name.split('-')[1] + '.txt'
            sample_idx = int(tmp_name.split('-')[2].split('_')[1]) - 1
        elif datatype == 'cad60':
            # e.g., image_path = 'cad-60-small/Person4-0512152943-RGB_1041_dep_extracted.pfm'
            tmp_name = image_path.split('/')[-1]
            cur_sub = tmp_name.split('-')[0]
            gender_cad = gender_cad_dict['cad-60/'+cur_sub]
            cad_anno_txt_fname = 'cad-60/' + cur_sub + '/' + tmp_name.split('-')[1] + '.txt'
            sample_idx = int(tmp_name.split('-')[2].split('_')[1]) - 1

        print ('gender_cad = %s, sample_idx = %d' % (gender_cad, sample_idx))

        joints3d_gt_with_torso_subtracted, _, img_name, joints3d_with_root = get_one_batched_cad_toy_example(
            cad_anno_txt_fname = './datasets/cad-60-120/' + cad_anno_txt_fname,
            sample_idx = sample_idx, 
            gender = gender_cad,
            swap = True)
        

        #print ("??? joints3d shape = {}".format(joints3d.shape))
        joints3d_aligned = align_by_pelvis(joints3d[0], get_pelvis = False)
        #print ("??? joints3d_aligned shape = {}, joints3d_gt_with_torso_subtracted shape = {}".format(
        #    joints3d_aligned.shape,
        #    joints3d_gt_with_torso_subtracted[0].shape
        #))
        # Convert to mm!
        err, err_pa = compute_error_one_sample(
                    np.reshape(joints3d_gt_with_torso_subtracted[0]*1000.0, [14,3]), # gt joints3d;
                    joints3d_aligned * 1000.0) # regressed joints3d via SMPL regressor;
        print ("[!!!] Processing {}, has mpjpe error = {} mm, mpjpe_pa error = {} mm".format(img_name, err, err_pa))

    
    
    #*********************
    """ 3) save results """
    #*********************
    """ save as csv file, Excel file format """
    timeStamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    csv_file = './results/experiments_eval_log.csv'   
    #messg = timeStamp + ',{:>7s},kt15,bad-3.0-noc,{:>6.3f},bad-3.0-all,{:>6.3f},fileDir={}\n'.format(
    #    eval_type[0], post_score[4]*100.0, post_score[11]*100.0, args.result_dir)
    messg = timeStamp + ',image,{},mpjpe,{:.3f},mpjpe_pa,{:.3f},fileDir={},note,{}\n'.format(
        img_name, err, err_pa, result_dir, take_home_message)
    with open( csv_file, 'a') as fwrite:
        fwrite.write(messg)
    
    
    if isSaveResults: 
        if not exists(result_dir):
            os.makedirs(result_dir)
            print ("makedirs %s" % result_dir)
        print ('[***] input image :', img_name)
        save_fig_result = join(result_dir, 'lsp_14joints3d_%s' % img_name.replace('/', '-'))
        draw_lsp_skeleton(joints3d_with_root, save_fig_result, with_numbers=True)

        short_nm = image_path.split('/')[-1][:-4]
        save_fig_result = join(result_dir, short_nm + '_infer.png')
        cam_for_render, joints_orig = visualize(img_orig/255.0, proc_param, joints[0], verts[0], cams[0], save_fig_result)
        tmp_name = join(result_dir,  short_nm + '_imgOrig.pfm')
        pfm.save(tmp_name, img_orig)
        tmp_name = join(result_dir,  short_nm + '_imgUsed.pfm')
        pfm.save(tmp_name, input_img[0])
        tmp_name = join(result_dir,  short_nm + '_depUsed.pfm')
        pfm.save(tmp_name, input_dep[0])

        # Write to an .obj file
        outmesh_path = join(result_dir,  short_nm + '_mesh_mpjpepa_%.2fmm.obj' % err_pa)
        save_to_mesh_file(smpl_model_path, 
                gender_vect, 
                np.concatenate([np.array([np.pi, .0, .0]), np.reshape(theta[0,3:72], [-1])], axis = 0), 
                beta[0], outmesh_path)



""" to infer and evaluate 1 image """
def infer_eval_1_img(
                model_type,
                img,
                dep,
                gender_vect,
                joints3d_gt_with_torso_subtracted,
                model = None, 
                result_dir = '', 
                img_name = '', 
                smpl_model_path = None,
                isSaveResults = False,
                meter_to_mm = 1.0 # could be 1000.0 or 1.0, according to the unit you used; 
                       ):
    
    #*********************
    """ 1) prediction """
    #*********************
    # Add batch dimension: 1 x D x D x 3
    input_img, input_dep, proc_param, img_orig, _, _ = preprocess_image(img, dep, None, None, None)
    
    input_img = np.expand_dims(input_img, 0)
    input_dep = np.expand_dims(input_dep, 0)
    gt_gender = np.expand_dims(gender_vect, 0)
    
    if model_type == 'hmr':
        joints, verts, cams, joints3d, _, _, _, _, theta, beta, _ = model.predict( 
            input_img, None, None, None, get_theta=True)

    else:
        joints, verts, cams, joints3d, _, _, _, _, theta, beta, joints3d_24 = model.predict(
                   input_img, input_dep, None, None, None, gt_gender, get_theta = True)
    
    #*********************
    """ 2) evaluation """
    #*********************
    #print ("[***] evaluating MPJPE and MPJPE PA !")
    joints3d_aligned = align_by_pelvis(joints3d[0], get_pelvis = False)
    #joints3d_aligned, pelvis = align_by_pelvis(joints3d[0], get_pelvis = True)
    
    # Convert to mm!
    err, err_pa = compute_error_one_sample(
                np.reshape(joints3d_gt_with_torso_subtracted* meter_to_mm, [14,3]), # gt joints3d;
                joints3d_aligned * meter_to_mm) # regressed joints3d via SMPL regressor;
    #print ("[!!!] Processing {}, has mpjpe error = {} mm, mpjpe_pa error = {} mm".format(img_name, err, err_pa))

    #*********************
    """ 3) save results """
    #*********************
    if isSaveResults: 
        if not exists(result_dir):
            os.makedirs(result_dir)
            print ("makedirs %s" % result_dir)
        
        print ('[***] input image :', img_name)
        save_fig_result = join(result_dir, 'lsp_14joints3d_%s' % img_name.replace('/', '-'))
        draw_lsp_skeleton(joints3d[0], save_fig_result, with_numbers=True)

        short_nm = img_name.split('/')[-1]
        save_fig_result = join(result_dir, short_nm + '_infer.png')
        cam_for_render, joints_orig = visualize(img_orig/255.0, proc_param, joints[0], verts[0], cams[0], save_fig_result)
        tmp_name = join(result_dir,  short_nm + '_imgOrig.pfm')
        pfm.save(tmp_name, img_orig)
        tmp_name = join(result_dir,  short_nm + '_imgUsed.pfm')
        pfm.save(tmp_name, input_img[0])
        tmp_name = join(result_dir,  short_nm + '_depUsed.pfm')
        pfm.save(tmp_name, input_dep[0])

        # Write to an .obj file
        outmesh_path = join(result_dir,  short_nm + '_mesh_mpjpepa_%.2fmm.obj' % err_pa)
        save_to_mesh_file(smpl_model_path, 
                gender_vect, 
                np.concatenate([np.array([np.pi, .0, .0]), np.reshape(theta[0,3:72], [-1])], axis = 0), 
                beta[0], outmesh_path)

    return err, err_pa


""" infer and evaluate: MPJPE and MPJPE_PA """
def infer_evaluate_images(
        model_type, 
        h5_filename, 
        result_dir = '', 
        datatype = 'cad120',
        smpl_model_path = None,
        take_home_message = None # some note about this experiment;
                       ):
    
    sess = tf.Session()
    if model_type == 'hmr':
        """ original hmr model w/o depth"""
        print ('[***] original hmr model !!!')
        model = RunModelV1(config, sess=sess)
    else:
        """ new model with depth """
        print ('[***] our hmr model !!!')
        model = RunModelV2(config, sess=sess)
    
    h5_examples = dd.io.load(h5_filename)
    #if 'cad' in data_type:
    img_num = len(h5_examples)

    errs, err_pas, img_names = [], [], []
    if 'cad' in datatype:
        meter_to_mm = 1000.0 # cad-60/120, in meters, change it to mm via *1000.0;
    elif 'surreal' in datatype:
        meter_to_mm = 1.0 # surreal, already in mm;
    print ("[***] meter_to_mm coefficient = ", meter_to_mm)
    for i in range(img_num):
        example_id = h5_examples[i].keys()[0]
        example = h5_examples[i][example_id]
        img_rgb = example['image_img_rgb']
        dep = example['depth_data']
        gender_vect = example['smpl_gender']
        joints3d_gt_with_torso_subtracted = example['mosh_gt3d']
        img_name = example['image_filename']
        img_names.append(img_name)
        
        #print ('[***] dep shape = {}'.format(dep.shape))
        #print ('[***] img shape = {}'.format(img_rgb.shape))
        #print ("[***] img_name = {}".format(img_name))
        #print(gender_vect, joints3d_gt_with_torso_subtracted, img_name)

        """ save some random results """ 
        if i in [0, 200, 500, 800, 1000, 1500, 2000, 2500]:
            isSaveResults = True
        else:
            isSaveResults = False
        err, err_pa = infer_eval_1_img(
                model_type,
                img_rgb, 
                dep, 
                gender_vect,
                joints3d_gt_with_torso_subtracted,
                model, 
                result_dir, 
                img_name, 
                smpl_model_path,
                isSaveResults, 
                meter_to_mm)

        if i % 50 == 0:
            print ("[***] loading {}/{}: id = {}, img_name = {}, mpjpe = {:.3f}, mpjpe_pa = {:.3f}".format(i+1, img_num, 
                   example_id, img_name, err, err_pa))
        errs.append(err)
        err_pas.append(err_pa)

    #*********************
    """ 3) save as csv file, Excel file format """
    #*********************
    timeStamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    csv_file = './results/log_files_important/experiments_eval_log.csv'   
    avg_err = .0
    avg_err_pa = .0
    with open( csv_file, 'a') as fwrite:
        img_num = len(err_pas)
        for i in range(0, img_num):
            err = errs[i]
            err_pa = err_pas[i]
            avg_err += err
            avg_err_pa += err_pa
            #messg = timeStamp + ',image,{},mpjpe,{:.3f},mpjpe_pa,{:.3f},fileDir={},note,{}\n'.format(
            #    img_names[i], errs[i], err_pas[i], result_dir, take_home_message)
            #fwrite.write(messg)
        avg_err /= img_num
        avg_err_pa /= img_num
        messg = timeStamp + ',image_num,{},avg_mpjpe,{:.3f},mpjpe_pa,{:.3f},fileDir={},note,{}\n'.format(
            img_num, avg_err, avg_err_pa, result_dir, take_home_message)
        fwrite.write(messg)
        print (messg)
    

#*******************************
#******Done !!! ****************
#*******************************
if __name__ == '__main__':
    
    config = get_config()
    print ("[****] load_path = %s" % config.load_path)
    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)
    if 0:
        infer_surreal_debug_with_info_print(args = config, json_path = None)
    if 0:
        take_home_message = 'model_dir={}'.format(config.load_path)
        simple_infer_1_img(
            image_path = config.image_fname, 
            depth_path = config.depth_fname, 
            gender = config.gender, 
            json_path = None, 
            result_dir = config.result_dir,
            datatype = config.data_type, # or 'cad60', 'cad120'
            has_joints3dgt = True,
            smpl_model_path = config.smpl_model_path,
            isSaveResults = False,
            take_home_message = take_home_message
            )
    if 1:
        take_home_message = 'model_dir={}'.format(config.load_path)
        print ('[***] datatype = {}'.format(config.data_type))
        infer_evaluate_images(
        #h5_filename = './results/eval-mpjpepa-thred-90mm-5-samples.h5'
            model_type = config.eval_model_type,
            h5_filename = config.h5_filename, 
            result_dir = config.result_dir, 
            #datatype = config.data_type,
            datatype = 'surreal',
            smpl_model_path = config.smpl_model_path,
            take_home_message = take_home_message, # some note about this experiment;
            )