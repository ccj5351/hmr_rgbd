# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: surreal_in_extrinc.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 03-07-2019
# @last modified: Thu 18 Jul 2019 04:47:09 PM EDT
import numpy as np
import sys,os
import scipy.io as sio
import cv2
import math
import transforms3d


#NOTE: added by CCJ:
# Intrinsic camera matrix
# > see https://github.com/gulvarol/surreal/blob/8af8ae195e6b4bb39a0fb64524a15a434ea620f6/datageneration/misc/3Dto2D/getIntrinsicBlender.m 

def get_intrinsic():
    """ 
    Returns:
       cam: (3,), [f, px, py] intrinsic camera parameters.
       e.g.
          [ 600     0   160  ;
              0   600   120  ;
              0     0     1 ];
    """
    res_x_px         = 320.0 # *scn.render.resolution_x
    res_y_px         = 240.0 # *scn.render.resolution_y
    f_mm             = 60.0  # *cam_ob.data.lens
    sensor_w_mm      = 32.0  # *cam_ob.data.sensor_width
    sensor_h_mm = sensor_w_mm * res_y_px / res_x_px # *cam_ob.data.sensor_height (function of others)

    scale = 1.  # *scn.render.resolution_percentage/100
    skew  = .0  # only use rectangular pixels
    pixel_aspect_ratio = 1. 

    fx_px = f_mm * res_x_px * scale / sensor_w_mm 
    fy_px = f_mm * res_y_px * scale * pixel_aspect_ratio / sensor_h_mm 
    
    # Center of the image
    u = res_x_px * scale / 2. 
    v = res_y_px * scale / 2. 
    
    # Intrinsic camera matrix
    intrinsic = [fx_px, skew, u, .0, fy_px, v, .0, .0, 1.]
    K = np.array([np.float(cont) for cont in intrinsic]).reshape(3, 3)
    #""" 
    #K = [ fx_px  skew   u  
    #      0      fy_px  v  
    #      0      0      1 ]
    #"""
    cam = np.zeros(3,dtype=np.float32)
    #print ("K = {}, {}, cam = {}, {}".format(K, K.shape, cam, cam.shape))
    cam[0] =  0.5 * (K[0, 0] + K[1, 1])
    cam[1] = K[0, 2]
    cam[2] = K[1, 2]
    return K, cam

#"""
#def get_surreal_to_lsp_joints_idx():
#    surreal_to_lsp_joints_idx = [
#         7,  # 7:  rightFoot ->  0: R ankle
#         4,  # 4:  rightLeg -> 1: R knee
#         1,  # 1:  rightUpLeg -> 2: R hip
#         2,  # 2:  leftUpLeg -> 3: L hip
#         5,  # 5:  leftLeg -> 4: L knee
#         8,  # 8:  leftFoot -> 5: L ankle
#         20, # 20: rightHand -> 6: R Wrist
#         18, # 18: rightForeArm-> 7: R Elbow
#         13, # 13: rightShoulder -> 8: R shoulder
#         14, # 14: leftShoulder -> 9: L shoulder
#         19, # 19: leftForeArm -> 10: L Elbow
#         21, # 21: leftHand -> 11: L Wrist
#         12, # 12  neck -> 12: # Neck top
#         15, # 15: head -> 13: # Head top
#    ]
    
#    return surreal_to_lsp_joints_idx
#"""

def get_lsp_idx_from_smpl_joints():
    # Mapping from SMPL 24 joints to LSP joints (0:13). In this roder:
    _COMMON_JOINT_IDS = [
         8,  # 8:  rightFoot ->  0: R ankle
         5,  # 5:  rightLeg -> 1: R knee
         2,  # 2:  rightUpLeg -> 2: R hip
         1,  # 1:  leftUpLeg -> 3: L hip
         4,  # 4:  leftLeg -> 4: L knee
         7,  # 7:  leftFoot -> 5: L ankle
         21, # 21: rightHand -> 6: R Wrist
         19, # 19: rightForeArm-> 7: R Elbow
         14, # 14: rightShoulder -> 8: R shoulder
         13, # 13: leftShoulder -> 9: L shoulder
         18, # 18: leftForeArm -> 10: L Elbow
         20, # 20: leftHand -> 11: L Wrist
         12, # 12  neck -> 12: # Neck top
         15, # 15: head -> 13: # Head top
    ]
    
    return _COMMON_JOINT_IDS



def get_smpl_joint_names():
    smpl_joint_names =  [
            'hips', # 0
            'leftUpLeg', # 1
            'rightUpLeg', # 2
            'spine', # 3
            'leftLeg', # 4
            'rightLeg', # 5
            'spine1', # 6
            'leftFoot',# 7
            'rightFoot',# 8
            'spine2', #9
            'leftToeBase',# 10
            'rightToeBase',# 11
            'neck', # 12
            'leftShoulder',# 13
            'rightShoulder', # 14
            'head', # 15
            'leftArm', # 16
            'rightArm', # 17
            'leftForeArm', # 18
            'rightForeArm',# 19
            'leftHand', # 20
            'rightHand', # 21
            'leftHandIndex1', # 22
            'rightHandIndex1' # 23
            ]
    assert len(smpl_joint_names == 24)
    #NOTE:
    """ it seems that surreal dataset has left/right swapped joints, 
        but the smpl joint names provided here is correct.
        What you have to do is to swap the joints and pose/shape of surreal dataset to
        satisfy this smpl joints order shown above !!!
    """
    return smpl_joint_names


# extract 14 lsp joints from surreal;
# return : 
    # joints2d: 3 x 14 
    # joints3d * 1000. # NOTE: used unites : millimeter 
def read_joints(currFrameJoints2D, currFrameJoints3D, Extrinsic):
    #currFrameJoints2D = currInfoDict['joints2D']
    #currFrameJoints3D = currFrameInfoDict['joints3D'] 

    """
    Reads joints in the common joint order.

    Returns:
      joints2d: 3 x |common joints|, e.g., 3 x 14;
      joints3d: |common joints| x 3, e.g., 14 x 3;

    """
    # Mapping from SMPL 24 joints to LSP joints (0:13). In this roder:
    _COMMON_JOINT_IDS = get_lsp_idx_from_smpl_joints()
    # Go over each common joint ids
    # 2d joints is 3 x 14 (lsp order)
    joints2d = np.zeros((3, len(_COMMON_JOINT_IDS)))
    # 3d joints is 3 x 14 (lsp order)
    
    #NOTE:
    #joints3d = np.zeros((3, len(_COMMON_JOINT_IDS)))
    # updated the returned joints3d shape from [3,14] to [14, 3], 
    # now it is consistent with that in tfrecord example parsing;
    joints3d = np.zeros((len(_COMMON_JOINT_IDS), 3))
    for i, jid in enumerate(_COMMON_JOINT_IDS):
        # 2d joints is 3 x 14 (lsp order)
        joints2d[0, i] =  currFrameJoints2D[0,jid] # x
        joints2d[1, i] =  currFrameJoints2D[1,jid] # y
        #NOTE: currently we just set this value as 1. But this value is not provided in the SURREAL dataset.
        joints2d[2, i] =  1 # visible
       
        
        #NOTE:???
        isWorldCoord = True
        #isWorldCoord = False
        if isWorldCoord:
            joints3d[i,:] = currFrameJoints3D[:,jid] # x,y,z: in real world meters;
        else:
            # 3d joints, 3 x 14
            # convert value from the world coordinate to camera coordinate;
            x,y,z = currFrameJoints3D[:,jid] # x,y,z: in real world meters;
            p_c = np.matmul(Extrinsic, np.reshape(np.array([x,y,z,1.0]), [4, -1]))
            joints3d[i,:] =  p_c[:, 0] # x,y,z, camera-coordinate in meters;

    return joints2d, joints3d * 1000. # NOTE: used unites : millimeter 
    #return joints2d, joints3d  # NOTE: used unites : meter 


def normalizeDepth(depthImg, isNormalized = True):
        loc = depthImg!=float(1e10)
        normDepth = depthImg
        if isNormalized:
            normDepth[loc] = (depthImg[loc] - np.min(depthImg[loc]))/(np.max(depthImg[loc])-np.min(depthImg[loc]))
        normDepth[~loc]= .0
        return normDepth
    

# added by CCJ:
# copied from https://github.com/gulvarol/surreal/blob/master/datageneration/misc/smpl_relations/smpl_relations.py
def get_frame(filevideo, t=0):
    cap = cv2.VideoCapture(filevideo)
    cap.set(propId=1, value=t)
    ret, frame = cap.read()
    frame = frame[:, :, [2, 1, 0]]
    return frame

# added by CCJ:
# copied from https://github.com/gulvarol/surreal/blob/master/datageneration/misc/smpl_relations/smpl_relations.py
def rotateBody(RzBody, pelvisRotVec):
    angle = np.linalg.norm(pelvisRotVec)
    Rpelvis = transforms3d.axangles.axangle2mat(pelvisRotVec / angle, angle)
    globRotMat = np.dot(RzBody, Rpelvis)
    R90 = transforms3d.euler.euler2mat(np.pi / 2, 0, 0)
    globRotAx, globRotAngle = transforms3d.axangles.mat2axangle(np.dot(R90, globRotMat))
    globRotVec = globRotAx * globRotAngle
    return globRotVec


# added by CCJ:
# copied from https://github.com/gulvarol/surreal/blob/master/datageneration/misc/smpl_relations/smpl_relations.py
# args:
#       points:     3D points in shape [24, 3];
#       intrinsic : camera intrinsic matrix, in shape [3, 3] matrix;
#       extrinsic : camera extrinsic matrix, in shape [3, 4] matrix;
def project_vertices(points, intrinsic, extrinsic):
    homo_coords = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1).transpose()
    proj_coords = np.dot(intrinsic, np.dot(extrinsic, homo_coords))
    proj_coords = proj_coords / proj_coords[2]
    proj_coords = proj_coords[:2].transpose()
    return proj_coords # in shape [24, 2]



# > see https://github.com/gulvarol/surreal/blob/8af8ae195e6b4bb39a0fb64524a15a434ea620f6/datageneration/misc/3Dto2D/getExtrinsicBlender.m
def get_extrinsic(T):
    """ T: camera location """ 
    R_world2bcam = np.array([[.0, .0, 1.],[.0, -1., .0],[-1., .0, .0]]).transpose()
    T_world2bcam = -1.0 * np.matmul(R_world2bcam, T)
    #print R_world2bcam, T_world2bcam
    # Following is needed to convert Blender camera to computer vision camera
    R_bcam2cv = np.array([[1.,.0,.0], [.0,-1.,.0], [.0,.0,-1.]])
    
    #Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = np.dot(R_bcam2cv, R_world2bcam)
    T_world2cv = np.dot(R_bcam2cv, T_world2bcam)
    # Put into 3x4 matrix
    RT = np.concatenate([R_world2cv, T_world2cv], axis=1)
    return RT



def draw_joints2D(img, joints2D, ax=None, kintree_table=None, with_text=True, color='g'):
    import matplotlib.pyplot as plt
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(img)
    for i in range(1, kintree_table.shape[1]):
        j1 = kintree_table[0][i]
        j2 = kintree_table[1][i]
        ax.plot([joints2D[j1, 0], joints2D[j2, 0]],
                [joints2D[j1, 1], joints2D[j2, 1]],
                color=color, linestyle='-', linewidth=2, marker='o', markersize=5)
        if with_text:
            ax.text(joints2D[j2, 0],
                    joints2D[j2, 1],
                    s=  get_smpl_joint_names()[j2],
                    color=color,
                    fontsize=8)
        print ("idx %d, joint %s, (x,y) = (%d,%d)" % (j2,  get_smpl_joint_names()[j2], joints2D[j2,0], joints2D[j2,1]))
        plt.savefig('/home/hmr/results/surreal_debug/smpl_joints24_proj.png')

#*************************************************
#NOTE: swap the left/right joints poses, due to 
# the left/right inconsitency which exists in the 
# surreal dataset itself;
# This function is copied and revised from src/util/data_utils.py;
#*************************************************
def swap_right_left_pose(pose):
    """
    Input is a 72-Dim vector.
    Global rotation (first 3) is left alone.
    """
    
    """
    # How I got the indices:
    joints_names = ['hips', 'leftUpLeg', 'rightUpLeg', 'spine', 'leftLeg', 'rightLeg', 'spine1',
                    'leftFoot', 'rightFoot', 'spine2',  'leftToeBase', 'rightToeBase',  'neck',
                    'leftShoulder', 'rightShoulder', 'head', 'leftArm', 'rightArm', 'leftForeArm',
                    'rightForeArm', 'leftHand', 'rightHand', 'leftHandIndex1', 'rightHandIndex1' ]
    right = [11, 8, 5, 2, 14, 17, 19, 21, 23] # right joints;
    left = [10, 7, 4, 1, 13, 16, 18, 20, 22] # left joints;
    new_map = {}
    for r_id, l_id in zip(right, left):
        for axis in range(0, 3):
            rind = r_id * 3 + axis
            lind = l_id * 3 + axis
            new_map[rind] = lind
            new_map[lind] = rind
    asis = [id for id in np.arange(0, 24) if id not in right + left]
    for a_id in asis:
        for axis in range(0, 3):
            aind = a_id * 3 + axis
            new_map[aind] = aind
    swap_inds = np.array([new_map[k] for k in sorted(new_map.keys())])
    """
    swap_inds = np.array([
            0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13, 14, 
            18, 19, 20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33, 34, 35, 30, 31, 32,
            36, 37, 38, 42, 43, 44, 39, 40, 41, 45, 46, 47, 51, 52, 53, 48, 49,
            50, 57, 58, 59, 54, 55, 56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66,
            67, 68
        ], np.int32)

    # sign_flip = np.tile([1, -1, -1], (24)) (with the first 3 kept)
    sign_flip = np.array([   
                #""" sing_flip all the joints with the first joint kept"""
                #1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1,
                #-1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1,
                #-1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1,
                #1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1,
                #-1, 1, -1, -1
                
                #""" sign_flip all the left/right joints with the others (e.g., spin, neck, etc) kept """
                1, 1, 1, # hip
                1, -1, -1, 1, -1, -1, 
                1, 1, 1, # spine
                1, -1, -1, 1, -1, -1, 
                1, 1, 1, # spine1
                1, -1, -1, 1, -1, -1, 
                1, 1, 1, # spine 2
                1, -1, -1, 1, -1, -1, 
                #1, 1, 1, #NOTE: neck, flip or not ???
                1, -1, -1, #NOTE: neck, flip or not ???
                1, -1, -1, 1, -1, -1, 
                #1, 1, 1, #NOTE: head flip or not??
                1, -1, -1, #NOTE: head flip or not??
                1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1
            ], dtype=pose.dtype)

    new_pose = np.take(pose, swap_inds) * sign_flip

    return new_pose

#"""
def reflect_lsp_14_joints3d(joints):
    # Assumes input is 14 x 3 (the LSP skeleton subset of H3.6M)
    swap_inds = np.array([5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13], np.int32)
    joints_ref = np.take(joints, swap_inds, axis = 0)
    flip_mat = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    joints_ref = np.matmul(flip_mat, joints_ref.T).T
    # Assumes all joints3d are mean subtracted
    joints_ref = joints_ref - np.mean(joints_ref, axis=0)
    return joints_ref
#"""

def swap_right_left_joints(joints_2d_or_3d):
    assert joints_2d_or_3d.shape[1] == 24
    # Assumes input is 3 x 24 (the SMPL joints)
    swap_inds = np.array([0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11,10,12,14,13, 15, 17,16, 19,18,21,20,23,22], np.int32)
    joints_swap = np.take(joints_2d_or_3d, swap_inds, axis = 1)
    return joints_swap