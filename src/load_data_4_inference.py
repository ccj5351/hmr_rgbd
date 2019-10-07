# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: load_data_4_inference.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 03-07-2019
# @last modified: Wed 17 Jul 2019 01:01:04 PM EDT

import sys
import src.pfmutil as pfm
from src.util import surreal_in_extrinc as surreal_util

import numpy as np
import math

import os
import scipy.io as sio
import argparse
import cv2
from src.smpl.smpl_webuser.serialization import load_model


def extract_14_joints(joint3d24, lsp_idx):
    assert joint3d24.shape[0] == 24 and joint3d24.shape[1] == 3
    joints3d = np.zeros((len(lsp_idx), 3))
    for i, jid in enumerate(lsp_idx):
        joints3d[i, :] = joint3d24[jid, :]
    return joints3d


def _load_anno_mat(fname):
    res = sio.loadmat(fname, struct_as_record=False, squeeze_me=True)
    return res

def data_loader_for_inference(args = None, isSave = False, isNormalized = True):
    
    depth_dict = _load_anno_mat(args.depth_fname)
    info = _load_anno_mat(args.info_fname)

    ## Load SMPL model (here we load the female model)
    ## Make sure path is correct
    # <========= LOAD SMPL MODEL BASED ON GENDER
    if info['gender'][0] == 0:  # f
        pkl_fname = './models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    elif info['gender'][0] == 1:  # m
        pkl_fname = './models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
    else:
        pkl_fname = './models/neutral_smpl_with_cocoplus_reg.pkl' 
    m = load_model( pkl_fname)
    print ("[***] loading model %s" % pkl_fname)
    
    #print ("[****] initial pose = {}, shape = {}, trans = {}, J_regressor shape = {}, J_regressor[0,:] = {}".format(
    #    m.pose[:6], m.betas[:], m.trans[:], m.J_regressor, m.J_regressor.shape))
    
    root_pos = m.J_transformed.r[0]

    zrot = info['zrot']
    print ("info['zrot'].shape = {}".format(zrot.shape))
    zrot = zrot[0]  # body rotation in euler angles
    RzBody = np.array(((math.cos(zrot), -math.sin(zrot), 0),
                       (math.sin(zrot), math.cos(zrot), 0),
                       (0, 0, 1)))
    
    # add gender info: it could be female, male, or neutral;
    gender = info['gender'][0]
    if gender == 0:
        gender_vect = np.array([1.0, .0, .0]).astype(np.float32) # gender vector - female
    elif gender == 1:
        gender_vect = np.array([.0, 1.0, .0]).astype(np.float32) # gender vector - male
    else:
        gender_vect = np.array([.0, .0, 1.0]).astype(np.float32) # gender vector - neutral


    intrinsic, cam = surreal_util.get_intrinsic()
    extrinsic = surreal_util.get_extrinsic(np.reshape(info['camLoc'], [3, -1]))

    all_gt_poses = []
    all_gt_shapes = []
    all_gt_genders = [] # added gender info;
    all_images = []
    all_depths = []
    all_gt_joints2d = []
    all_gt_joints3d = []
    all_gt_trans = [] # added for translation between: info['joints3D'][:,:,t].T[0] - root_pos

    all_gt_smpl_verts = []
    all_gt_smpl_joints3d = []
    all_gt_smpl_joints3d_proj = []
    gt_cam = cam
    
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
        print ("makedirs %s" % args.result_dir)


    for t in range(args.t_beg, args.t_end):
        # depth:  in meters;
        dep = depth_dict['depth_%d'% (t+1)] # the depth_1, index is 1-based, defiend by the dataset itselt;
        dep = surreal_util.normalizeDepth(dep, isNormalized)
        # Read frame of the video
        rgb = surreal_util.get_frame(args.info_fname[:-9] + ".mp4", t)
        
        gt_poses = info['pose'][:,t]
        gt_shape = info['shape'][:,t]
        gt_poses[0:3] = surreal_util.rotateBody(RzBody, gt_poses[0:3])
       
        gt_joints2d = info['joints2D'][:,:,t] # (x,y,vis) x 24;
        gt_joints3d = info['joints3D'][:,:,t] # (x,y,z) x 24;

        #NOTE: surreal joints are left/right swapped, compared with the smpl joint order; 
        gt_joints2d_swap = surreal_util.swap_right_left_joints(gt_joints2d)
        gt_joints3d_swap = surreal_util.swap_right_left_joints(gt_joints3d)
        # read 14 lsp joints;
        gt_joints2d, gt_joints3d = surreal_util.read_joints(gt_joints2d_swap, gt_joints3d_swap, extrinsic) # 3 x 14;

        #NOTE:still, surreal dataset has right/left swapped pose compared with official smpl;
        gt_poses = surreal_util.swap_right_left_pose(gt_poses)
        
        # all the frames of the same subject should have the same gender, so just repeat it;
        all_gt_genders.append(gender_vect)

        all_gt_joints2d.append(gt_joints2d[:2,:]) # (x,y,vis) x 24;
        all_gt_joints3d.append(gt_joints3d/ 1000.0 ) # change back to meters;
        all_gt_poses.append(gt_poses)
        all_gt_shapes.append(gt_shape)
        all_images.append(rgb)
        all_depths.append(dep)
        
        m.pose[:] =  gt_poses
        m.betas[:] = gt_shape
        # Set model translation
        """ actually this m.trans[:] can also be added to the smpl vertices/joints 
            here or after posing/shaping, because of the translation is independent
            of shaping/posing!!!
        """
        m.trans[:] = info['joints3D'][:,:,t].T[0] - root_pos  
        all_gt_trans.append(info['joints3D'][:,:,t].T[0] - root_pos)
        
        print ("[****] updated smpl pose = {}, shape = {}, trans = {}".format(m.pose[:6], m.betas[:], m.trans[:]))
        print ("[****] m.trans[:] = {}".format(info['joints3D'][:,:,t].T[0] - root_pos))
        print ("[****] initial, root_joint_3D = {}, root_joint_from_pos = {}".format(
            info['joints3D'][:,:,t].T[0], 
            root_pos))
        smpl_vertices = m.r
        smpl_joints3D = m.J_transformed.r
        print ("[****] after posing, root_joint_from_pose = {}".format(smpl_joints3D[0]))
        
        #"""""""""""""""""
        #NOTE: debugging
        #surreal_util.draw_joints2D(
        #        rgb,
        #        # 24 joints 
        #        surreal_util.project_vertices(smpl_joints3D, intrinsic, extrinsic),
        #        None, m.kintree_table, color = 'b')

        #print ("smpl_joints3D shape = ", smpl_joints3D.shape) # in shape [24, 3]

        
        smpl_joints3D = extract_14_joints(smpl_joints3D, 
                surreal_util.get_lsp_idx_from_smpl_joints()) # now in shape [14, 3] 
        #print ("smpl_joints3D shape = ", smpl_joints3D.shape) # now in shape [14, 3]
        all_gt_smpl_verts.append(smpl_vertices)
        all_gt_smpl_joints3d.append(smpl_joints3D)

        #print ("intrinsic shape = {}, extrinsic shape = {}".format(intrinsic.shape, extrinsic.shape))
        #print ("[????] intrinsic in py smpl  = {}".format(intrinsic))
        #print ("[????] extrinsic in py smpl  = {}".format(extrinsic))
        proj_smpl_joints3D =  surreal_util.project_vertices(smpl_joints3D, intrinsic, extrinsic)
        all_gt_smpl_joints3d_proj.append(proj_smpl_joints3D)

        
        if isSave:
            ## Write to an .obj file
            tmp_name = args.info_fname.split('/')[-1][:-9] + "frame_%03d" % t
            outmesh_path = os.path.join(args.result_dir, tmp_name + '_smpl.obj')
            outimg_path = os.path.join(args.result_dir, tmp_name + "_rgb.jpg")
            outdep_path = os.path.join(args.result_dir, tmp_name + "_depth.pfm")

            cv2.imwrite(outimg_path, rgb[:, :, [2, 1, 0]])
            pfm.save(outdep_path, dep.astype(np.float32))

            with open( outmesh_path, 'w') as fp:
                for v in m.r:
                    fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

                for f in m.f+1: # Faces are 1-based, not 0-based in obj files
                    fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
            
            ## Print message
            print ('..Output mesh saved to: ', outmesh_path)
            print ('..Output depth saved to: ', outdep_path)
            print ('..Output rgb image saved to: ', outimg_path)
    
    return all_images, all_depths, all_gt_poses, all_gt_shapes, \
            all_gt_genders, all_gt_trans,\
            all_gt_joints2d, all_gt_joints3d, \
            all_gt_smpl_verts, all_gt_smpl_joints3d, \
            all_gt_smpl_joints3d_proj, gt_cam, intrinsic, extrinsic, m.kintree_table



def no_smpl_gt_data_loader_for_inference(args = None, isSave = False, isNormalized = True, gender = 2):
    
    depth_dict = _load_anno_mat(args.depth_fname)
    info = _load_anno_mat(args.info_fname)

    
    # add gender info: it could be female, male, or neutral;
    gender = info['gender'][0]
    if gender == 0:
        gender_vect = np.array([1.0, .0, .0]).astype(np.float32) # gender vector - female
    elif gender == 1:
        gender_vect = np.array([.0, 1.0, .0]).astype(np.float32) # gender vector - male
    else:
        gender_vect = np.array([.0, .0, 1.0]).astype(np.float32) # gender vector - neutral

    all_gt_genders = [] # added gender info;
    all_images = []
    all_depths = []
    all_gt_joints2d = []
    
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
        print ("makedirs %s" % args.result_dir)


    for t in range(args.t_beg, args.t_end):
        # depth:  in meters;
        dep = depth_dict['depth_%d'% (t+1)] # the depth_1, index is 1-based, defiend by the dataset itselt;
        dep = surreal_util.normalizeDepth(dep, isNormalized)
        # Read frame of the video
        rgb = surreal_util.get_frame(args.info_fname[:-9] + ".mp4", t)
        
        gt_poses = info['pose'][:,t]
        gt_shape = info['shape'][:,t]
        gt_poses[0:3] = surreal_util.rotateBody(RzBody, gt_poses[0:3])
       
        gt_joints2d = info['joints2D'][:,:,t] # (x,y,vis) x 24;
        gt_joints3d = info['joints3D'][:,:,t] # (x,y,z) x 24;

        #NOTE: surreal joints are left/right swapped, compared with the smpl joint order; 
        gt_joints2d_swap = surreal_util.swap_right_left_joints(gt_joints2d)
        gt_joints3d_swap = surreal_util.swap_right_left_joints(gt_joints3d)
        # read 14 lsp joints;
        gt_joints2d, gt_joints3d = surreal_util.read_joints(gt_joints2d_swap, gt_joints3d_swap, extrinsic) # 3 x 14;

        #NOTE:still, surreal dataset has right/left swapped pose compared with official smpl;
        gt_poses = surreal_util.swap_right_left_pose(gt_poses)
        
        # all the frames of the same subject should have the same gender, so just repeat it;
        all_gt_genders.append(gender_vect)

        all_gt_joints2d.append(gt_joints2d[:2,:]) # (x,y,vis) x 24;
        all_gt_joints3d.append(gt_joints3d/ 1000.0 ) # change back to meters;
        all_gt_poses.append(gt_poses)
        all_gt_shapes.append(gt_shape)
        all_images.append(rgb)
        all_depths.append(dep)
        
        m.pose[:] =  gt_poses
        m.betas[:] = gt_shape
        # Set model translation
        """ actually this m.trans[:] can also be added to the smpl vertices/joints 
            here or after posing/shaping, because of the translation is independent
            of shaping/posing!!!
        """
        m.trans[:] = info['joints3D'][:,:,t].T[0] - root_pos  
        all_gt_trans.append(info['joints3D'][:,:,t].T[0] - root_pos)
        
        print ("[****] updated smpl pose = {}, shape = {}, trans = {}".format(m.pose[:6], m.betas[:], m.trans[:]))
        print ("[****] m.trans[:] = {}".format(info['joints3D'][:,:,t].T[0] - root_pos))
        print ("[****] initial, root_joint_3D = {}, root_joint_from_pos = {}".format(
            info['joints3D'][:,:,t].T[0], 
            root_pos))
        smpl_vertices = m.r
        smpl_joints3D = m.J_transformed.r
        print ("[****] after posing, root_joint_from_pose = {}".format(smpl_joints3D[0]))
        
        #"""""""""""""""""
        #NOTE: debugging
        #surreal_util.draw_joints2D(
        #        rgb,
        #        # 24 joints 
        #        surreal_util.project_vertices(smpl_joints3D, intrinsic, extrinsic),
        #        None, m.kintree_table, color = 'b')

        #print ("smpl_joints3D shape = ", smpl_joints3D.shape) # in shape [24, 3]

        
        smpl_joints3D = extract_14_joints(smpl_joints3D, 
                surreal_util.get_lsp_idx_from_smpl_joints()) # now in shape [14, 3] 
        #print ("smpl_joints3D shape = ", smpl_joints3D.shape) # now in shape [14, 3]
        all_gt_smpl_verts.append(smpl_vertices)
        all_gt_smpl_joints3d.append(smpl_joints3D)

        #print ("intrinsic shape = {}, extrinsic shape = {}".format(intrinsic.shape, extrinsic.shape))
        #print ("[????] intrinsic in py smpl  = {}".format(intrinsic))
        #print ("[????] extrinsic in py smpl  = {}".format(extrinsic))
        proj_smpl_joints3D =  surreal_util.project_vertices(smpl_joints3D, intrinsic, extrinsic)
        all_gt_smpl_joints3d_proj.append(proj_smpl_joints3D)

        
        if isSave:
            ## Write to an .obj file
            tmp_name = args.info_fname.split('/')[-1][:-9] + "frame_%03d" % t
            outmesh_path = os.path.join(args.result_dir, tmp_name + '_smpl.obj')
            outimg_path = os.path.join(args.result_dir, tmp_name + "_rgb.jpg")
            outdep_path = os.path.join(args.result_dir, tmp_name + "_depth.pfm")

            cv2.imwrite(outimg_path, rgb[:, :, [2, 1, 0]])
            pfm.save(outdep_path, dep.astype(np.float32))

            with open( outmesh_path, 'w') as fp:
                for v in m.r:
                    fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

                for f in m.f+1: # Faces are 1-based, not 0-based in obj files
                    fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
            
            ## Print message
            print ('..Output mesh saved to: ', outmesh_path)
            print ('..Output depth saved to: ', outdep_path)
            print ('..Output rgb image saved to: ', outimg_path)
    
    return all_images, all_depths, all_gt_poses, all_gt_shapes, \
            all_gt_genders, all_gt_trans,\
            all_gt_joints2d, all_gt_joints3d, \
            all_gt_smpl_verts, all_gt_smpl_joints3d, \
            all_gt_smpl_joints3d_proj, gt_cam, intrinsic, extrinsic, m.kintree_table


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='get depth as pfm files')
    parser.add_argument('--depth_fname', type=str,
                        help='Path to the *_depth.mat file')

    parser.add_argument('--info_fname', type=str,
                        help='Path to the *_info.mat file')

    parser.add_argument('--result_dir', type=str,
                        help='Path to output files')

    parser.add_argument('--t_beg', type=int, default=0,
                        help='Frame number (default 0)')
    
    parser.add_argument('--t_end', type=int, default=1,
                        help='Frame number (default 1)')
    
    args = parser.parse_args()
    
    print('depth_fname: {}'.format(args.depth_fname))
    print('info_fname: {}'.format(args.info_fname))
    print('result_dir: {}'.format(args.result_dir))

    print('t_beg: {}'.format(args.t_beg))
    print('t_end: {}'.format(args.t_end))
    _,_,_,_,_,_ = data_loader_for_inference(args = args, isSave = True, isNormalized = True)
