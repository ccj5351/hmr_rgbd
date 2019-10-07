# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: load_surreal.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 12-06-2019
# @last modified: Sat 06 Jul 2019 09:42:37 PM EDT

import scipy.io as sio
import sys,json,os,cv2
import pfmutil as pfm
import numpy as np
from os import makedirs, listdir, path
#from mathutils import Matrix, Vector, Quaternion, Euler

def load_anno_mat(fname):
    res = sio.loadmat(fname, struct_as_record=False, squeeze_me=True)
    return res


# > see https://github.com/gulvarol/surreal/blob/master/datageneration/misc/smpl_relations/smpl_relations.py
def joint_names():
    return ['hips', # 0
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

def load_anno_mat(fname):
     res = sio.loadmat(fname, struct_as_record=False, squeeze_me=True)
     return res

def draw_text(img,  bottomLeftCornerOfText, text):
    # Write some Text
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    #bottomLeftCornerOfText = (10,500)
    fontScale              = 0.4
    fontColor              = (255,255,255)
    lineType               = 2
    cv2.putText(img, text, bottomLeftCornerOfText, font, fontScale, fontColor,lineType)

# > see https://github.com/gulvarol/surreal/blob/8af8ae195e6b4bb39a0fb64524a15a434ea620f6/datageneration/misc/3Dto2D/getExtrinsicBlender.m
def getExtrinsicBlender(T):
    """ T: camera location """ 
    R_world2bcam = np.array([[.0, .0, 1.],[.0, -1., .0],[-1., .0, .0]]).transpose()
    T_world2bcam = -1.0 * np.matmul(R_world2bcam, T)
    #print R_world2bcam, T_world2bcam
    # Following is needed to convert Blender camera to computer vision camera
    R_bcam2cv = np.array([[1.,.0,.0], [.0,-1.,.0], [.0,.0,-1.]])
    
    #Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = np.matmul(R_bcam2cv, R_world2bcam)
    T_world2cv = np.matmul(R_bcam2cv, T_world2bcam)
    # Put into 3x4 matrix
    return np.hstack((R_world2cv, T_world2cv))



#NOTE: added by CCJ:
# Intrinsic camera matrix
# > see https://github.com/gulvarol/surreal/blob/8af8ae195e6b4bb39a0fb64524a15a434ea620f6/datageneration/misc/3Dto2D/getIntrinsicBlender.m 
def getIntrinsicBlender():
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
    """ 
    K = [ fx_px  skew   u  
          0      fy_px  v  
          0      0      1 ]
    """
    cam = np.zeros(3,dtype=np.float32)
    cam[0] =  0.5 * (K[0, 0] + K[1, 1])
    cam[1] = K[0, 2]
    cam[2] = K[1, 2]
    return K, cam

def rotation_from_euler(phi, theta, psi):
    from math import sin,cos
    R_z_phi = np.array([[cos(phi), -sin(phi), .0], [sin(phi), cos(phi), .0], [.0, .0, 1.]])
    R_y_theta =  np.array([[cos(theta), .0, sin(theta)], [.0, 1., .0], [-sin(theta), .0, cos(theta)]])
    R_z_psi = np.array([[cos(psi), -sin(psi), .0],[sin(psi), cos(psi), .0],[.0, .0, 1.]])
    R = np.matmul(R_z_phi, np.matmul(R_y_theta, R_z_psi))
    return R

def _extract_frames_from_video(src_video, isDisplay = False):
    img_dict = {}
    vidcap = cv2.VideoCapture(src_video)
    success, image = vidcap.read()
    count = 0
    tmp_dir = src_video[:-4]
    if not path.exists(tmp_dir):
        makedirs(tmp_dir)
    while success:
        key =  "frame_%03d" % count
        print (" ??? processing %s" % key)
        cv2.imwrite(os.path.join(tmp_dir, key + ".jpg"), image)
        img_dict[key] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #if count % 500 == 0 and isDisplay:
        #    print ('Read a new frame: %3d' % count, success)
        count += 1
        success, image = vidcap.read()
    return img_dict, count


if __name__ == "__main__":

    if 0:
        fname = "/mnt/interns/changjiang/PatientPositioning/Datasets/surreal/cmu/train/run0/01_01/01_01_c0001_depth.mat"
        dataroot = "/mnt/interns/changjiang/PatientPositioning/Datasets/surreal/small-data/depth"
        dep = load_anno_mat(fname)
        #print ("type : ", type(dep), "keys : %s" % sorted(dep.keys()))
        #for i in range(1,101):
        for i in range(0,4):
            dep_key_name = 'depth_%d' % (i+1)
            dep_name = 'depth_%03d' % i
            depth = dep[dep_key_name]
            depth[depth == 1e+10] = np.inf
            #depth[depth == 1e+10] = 0
            cv2.imwrite(dataroot + '/' + dep_name + ".png", depth.astype(np.uint16))
            pfm.save(dataroot + '/' + dep_name + ".pfm", 1000.0*depth.astype(np.float32))
        #dep_names = ['depth_1', 'depth_10', 'depth_100', 'depth_11', 'depth_12', 'depth_13', 'depth_14', 'depth_15', 'depth_16', 'depth_17', 'depth_18', 'depth_19', 'depth_2', 'depth_20', 'depth_21', 'depth_22', 'depth_23', 'depth_24', 'depth_25', 'depth_26', 'depth_27', 'depth_28', 'depth_29', 'depth_3', 'depth_30', 'depth_31', 'depth_32', 'depth_33', 'depth_34', 'depth_35', 'depth_36', 'depth_37', 'depth_38', 'depth_39', 'depth_4', 'depth_40', 'depth_41', 'depth_42', 'depth_43', 'depth_44', 'depth_45', 'depth_46', 'depth_47', 'depth_48', 'depth_49', 'depth_5', 'depth_50', 'depth_51', 'depth_52', 'depth_53', 'depth_54', 'depth_55', 'depth_56', 'depth_57', 'depth_58', 'depth_59', 'depth_6', 'depth_60', 'depth_61', 'depth_62', 'depth_63', 'depth_64', 'depth_65', 'depth_66', 'depth_67', 'depth_68', 'depth_69', 'depth_7', 'depth_70', 'depth_71', 'depth_72', 'depth_73', 'depth_74', 'depth_75', 'depth_76', 'depth_77', 'depth_78', 'depth_79', 'depth_8', 'depth_80', 'depth_81', 'depth_82', 'depth_83', 'depth_84', 'depth_85', 'depth_86', 'depth_87', 'depth_88', 'depth_89', 'depth_9', 'depth_90', 'depth_91', 'depth_92', 'depth_93', 'depth_94', 'depth_95', 'depth_96', 'depth_97', 'depth_98', 'depth_99']
        #print ("depth images : %d" % len(dep_names))

    if 0:
        video_fname = "/mnt/interns/changjiang/PatientPositioning/Datasets/surreal/cmu/train/run0/01_01/01_01_c0001.mp4"
        dataroot = "/mnt/interns/changjiang/PatientPositioning/Datasets/surreal/small-data/images"
        vidcap = cv2.VideoCapture(video_fname)
        info_fname = "/mnt/interns/changjiang/PatientPositioning/Datasets/surreal/cmu/train/run0/01_01/01_01_c0001_info.mat"
        joints2D = load_anno_mat(info_fname)["joints2D"]
        jnames =  joint_names()
        success, image = vidcap.read()
        for i in range(0, 4):
            key_name = 'frame_%03d' % i
            joints = joints2D[:,:,i]
            for j in range(0, 14):
                 cv2.circle(image,(int(joints[0,j]), int (joints[1,j])), radius = 1, color = (0,0,255), thickness = 2)
                 diff = -40 if j in [0, 2, 5, 8, 11, 14, 19, 21,] else 40
                 if j == 8:
                     diff = -20 
                 if j == 7:
                     diff = 20
                 if j == 16:
                     diff = 80
                 if j == 17:
                     diff = -80
                 if j == 22:
                     diff = 80
                 if j == 23:
                     diff = -80

                 draw_text(image,  (int(joints[0,j] + diff),  int(joints[1,j])), "%d" % j)
                 cv2.line(image, (int(joints[0,j]), int (joints[1,j])), (int(joints[0,j] + diff),  int(joints[1,j])), (255,0,0), 1)
            if i == 0:
                print ("saving %s" % dataroot + '/' + key_name + ".jpg")
                cv2.imwrite(dataroot + '/' + key_name + ".jpg", image)
            success, image = vidcap.read()


        #dep_names = ['depth_1', 'depth_10', 'depth_100', 'depth_11', 'depth_12', 'depth_13', 'depth_14', 'depth_15', 'depth_16', 'depth_17', 'depth_18', 'depth_19', 'depth_2', 'depth_20', 'depth_21', 'depth_22', 'depth_23', 'depth_24', 'depth_25', 'depth_26', 'depth_27', 'depth_28', 'depth_29', 'depth_3', 'depth_30', 'depth_31', 'depth_32', 'depth_33', 'depth_34', 'depth_35', 'depth_36', 'depth_37', 'depth_38', 'depth_39', 'depth_4', 'depth_40', 'depth_41', 'depth_42', 'depth_43', 'depth_44', 'depth_45', 'depth_46', 'depth_47', 'depth_48', 'depth_49', 'depth_5', 'depth_50', 'depth_51', 'depth_52', 'depth_53', 'depth_54', 'depth_55', 'depth_56', 'depth_57', 'depth_58', 'depth_59', 'depth_6', 'depth_60', 'depth_61', 'depth_62', 'depth_63', 'depth_64', 'depth_65', 'depth_66', 'depth_67', 'depth_68', 'depth_69', 'depth_7', 'depth_70', 'depth_71', 'depth_72', 'depth_73', 'depth_74', 'depth_75', 'depth_76', 'depth_77', 'depth_78', 'depth_79', 'depth_8', 'depth_80', 'depth_81', 'depth_82', 'depth_83', 'depth_84', 'depth_85', 'depth_86', 'depth_87', 'depth_88', 'depth_89', 'depth_9', 'depth_90', 'depth_91', 'depth_92', 'depth_93', 'depth_94', 'depth_95', 'depth_96', 'depth_97', 'depth_98', 'depth_99']
        #print ("depth images : %d" % len(dep_names))
    if 0:
        
        tmp_dir = "/mnt/interns/changjiang/PatientPositioning/Datasets/surreal/cmu-small/samples/01_01/"
        sequences = [s for s in listdir(tmp_dir) if path.isfile(path.join(tmp_dir,s)) and ".mp4" in s]
        for s in sequences:
            src_video = path.join(tmp_dir, s)
            print src_video
            _,_ = _extract_frames_from_video(src_video, isDisplay = False)
        sys.exit()

    if 1:
        fname = "/mnt/interns/changjiang/PatientPositioning/Datasets/surreal/cmu/train/run0/01_01/01_01_c0001_depth.mat"
        dataroot = "/mnt/interns/changjiang/PatientPositioning/Datasets/surreal/small-data/depth"
        dep = load_anno_mat(fname)
        
        info_fname = "/mnt/interns/changjiang/PatientPositioning/Datasets/surreal/cmu/train/run0/01_01/01_01_c0001_info.mat"
        anno = load_anno_mat(info_fname)
        joints2D = anno["joints2D"]
        joints3D = anno["joints3D"]
        zrot = anno["zrot"]
        jnames =  joint_names()
        T = np.reshape(anno["camLoc"], [3,-1])
        E = getExtrinsicBlender(T)
        K,_ = getIntrinsicBlender()
        
        #tmp = Quaternion(Euler((0, 0, zrot), 'XYZ'))
        #print ("type : ", type(dep), "keys : %s" % sorted(dep.keys()))

        #for i in range(1,101):
        for i in range(0,1):
            dep_key_name = 'depth_%d' % (i+1)
            dep_name = 'depth_%03d' % i
            depth = dep[dep_key_name]
            depth[depth == 1e+10] = np.inf
            joints = joints2D[:,:,i]
            joints3d = joints3D[:,:,i]
            tmp = rotation_from_euler(0, 0, zrot[i])

            print ("T = {}\n E = {}\ntmp = {}".format(T, E, tmp))
            for j in range(0, 1):
                c = int(joints[0,j])
                r = int(joints[1,j])
                d = depth[r,c]
                x,y,z = joints3d[:, j]
                #x, y, z = 0.,0.,0.
                #print x,y,z
                p_c = np.matmul(E, np.reshape(np.array([x,y,z,1.0]), [4, -1]))
                #print p_c
                #print K
                p_i = np.matmul(K, p_c)
                s = 1.0 / p_i[2,0]
                p_i *= s

                print 'frame %03d, joint %2d : %13s' %(i, j, jnames[j])
                print ('p=({},{},{})\nd = {}\np_c = {}\np_i = {}\njoint2d = ({},{})'.format(x,y,z,d, p_c, p_i, c,r))
                tmp_inv = np.linalg.inv(tmp)
                print tmp_inv
                p_tmp = np.matmul(tmp_inv, np.reshape(np.array([x,y,z]), [3, -1]))
                p_tmp2 = np.matmul(tmp, np.reshape(np.array([x,y,z]), [3, -1]))
                print ('p_tmp = {}, p_tmp2 = {}'.format(p_tmp, p_tmp2))


            pfm.save(dataroot + '/' + dep_name + ".pfm", depth.astype(np.float32))
