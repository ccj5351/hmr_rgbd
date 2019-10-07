"""
# added by Changjiang Cai:
Convert SURREAL to TFRecords.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib; matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import sys,json,os,cv2
from os import makedirs,listdir
from os.path import join, exists,isfile,isdir
from time import time

import numpy as np
import math
import transforms3d

import tensorflow as tf

from .common import (convert_to_example_wmosh_wdepth, convert_to_h5_wmosh_wdepth,
                     ImageCoder, resize_img,
                     convert_to_example_smpl_pose_joints3d_pairs)

import src.pfmutil as pfm
import scipy.io as sio

from termcolor import colored

from src.util.smpl_webuser.serialization import load_model

import deepdish as dd

tf.app.flags.DEFINE_string('img_directory', '/datasets/cad-60/','image data directory')

tf.app.flags.DEFINE_string('task_type', 'add_to_tfrecord', 
    'could be add_to_tfrecord or add_smpl_joints3d_pair')

tf.app.flags.DEFINE_string(
    'output_directory', '/datasets/tf_datasets/cad-60/', 'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 100,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 1000,
                            'Number of shards in validation TFRecord files.')
"""
# save to h5 file or save to tfrecord files:
* Usually, for network training: we chose to write tfrecord files for tfrecord data loading;
* for evaulation and testing, we use h5 file for numpy data loading, 
* which will be further fed to tf.placeholder for evluation and or testing;
"""
tf.app.flags.DEFINE_boolean( 'is_save_to_h5files', False, 'if set, save to h5 files for data loading')

FLAGS = tf.app.flags.FLAGS

SMALL_DATA = False

from src.util.surreal_in_extrinc import (read_joints,rotateBody,
        get_intrinsic,get_extrinsic,
        swap_right_left_pose, 
        swap_right_left_joints,
        reflect_lsp_14_joints3d
        )


surreal_small_subjects = [
        'cmu-small/samples/01_01_100_samples', # 100 frames in total;
        #'cmu-small/samples/01_01_100_samples/1-samples': 1, 
        #'cmu-small/samples/01_01_100_samples/5-samples': 5, 
        #'cmu-small/samples/01_01_100_samples/20-samples': 20, 
        #'cmu-small/samples/01_01_100_samples/100-samples': 100, 
        ]

#NOTE: leave those below for evaluation;
surreal_eval_subjects = [ 
        #'cmu/val/run0', # 15235
        #'cmu/val/run1', # 16176
        'cmu/val/run2', # 13080
]

surreal_subjects = [
        'cmu/train/run0', # 1605030;
        'cmu/train/run1', # 2540380;
        'cmu/train/run2', # 1196680;

        #'cmu/val/run0', # 15235;
        #'cmu/val/run1', # 16176;
        #'cmu/val/run2', # 13080;
        
        #'cmu/test/run0', # 362214;
        #'cmu/test/run1', # 556454;
        #'cmu/test/run2', # 275994;
        ]

# abtained by running function count_images_num() for each dir;
surreal_num_imgs_dict = {
        #Total = 6581243;
        'cmu/train/run0' : 1605030,
        'cmu/train/run1' : 2540380,
        'cmu/train/run2' : 1196680,

        'cmu/val/run0' : 15235,
        'cmu/val/run1' : 16176,
        'cmu/val/run2' : 13080,
        
        'cmu/test/run0': 362214,
        'cmu/test/run1': 556454,
        'cmu/test/run2': 275994,
        
        # for debugging 
        'cmu-small/samples/01_01_100_samples' : 100, # 100 frames in total;
        'cmu-small/samples/01_01_100_samples/1-samples': 1, 
        'cmu-small/samples/01_01_100_samples/5-samples': 5, 
        'cmu-small/samples/01_01_100_samples/20-samples': 20, 
        'cmu-small/samples/01_01_100_samples/100-samples': 100, 
}


def load_anno_mat(fname):
    res = sio.loadmat(fname, struct_as_record=False, squeeze_me=True)
    return res



def _extract_frames_from_video(src_video, isDisplay = False):
    img_dict = {}
    vidcap = cv2.VideoCapture(src_video)
    success, image = vidcap.read()
    count = 0
    while success:
        key =  "frame_%03d" % count
        #print (" ??? processing %s" % key)
        #cv2.imwrite("/home/hmr/datasets/surreal/cmu-small/train/run0/05_20/05_20_c0002_" + key + ".jpg", image)
        img_dict[key] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #if count % 500 == 0 and isDisplay:
        #    print ('Read a new frame: %3d' % count, success)
        count += 1
        success, image = vidcap.read()
    return img_dict, count

# > see https://github.com/gulvarol/surreal for data format:
#""" """
#SURREAL/data/
#""" """
#------------- cmu/  # using MoCap from CMU dataset
#-------------------- train/
#---------------------------- run2/ #70% overlap
#------------------------------------  <sequenceName>/ #e.g. 01_01
def load_sequence_data_and_anno_mat(video_fname, depth_fname, info_fname, isDisplay = None):
    num_imgs  = 0
    # sequence name : 01_01_c0001.mp4
    dep_dict  = load_anno_mat(depth_fname)
    info_dict = load_anno_mat(info_fname)
    img_dict, num_imgs  = _extract_frames_from_video(video_fname, isDisplay)
    return img_dict, dep_dict, info_dict, num_imgs

#NOTE: detect there exists person in this image frame or not;
#For example: there is no person in the frame 38 extracted out of 
# 'surreal/cmu/train/run0/02_03/02_03_c0001.mp4'
# surreal dataset has image size 320 X 240;
def _joints_is_in_image(joint, imgH = 240, imgW = 320):
    x = joint[0]
    y = joint[1]
    return x * (imgW - x) > 0 and y * (imgH - y)

def parse_joints_and_scale_image(
        image, 
        depth, 
        joints, # 2d joints, in size 3 x 14 (lsp order)
        cam, # (3,)  # Flength, px, py (principal point)
        save_scaled_img_dir = None
        ):
    # joints is 3 x 14 (lsp order)
    # All kps are visible.
    min_pt = np.min(joints[:2, :], axis=1)
    max_pt = np.max(joints[:2, :], axis=1)


    success = _joints_is_in_image(min_pt) and _joints_is_in_image(max_pt)
    if not success:
        return None, None, None, None, None, None, None, None, None

    person_height = np.linalg.norm(max_pt - min_pt)
    center = (min_pt + max_pt) / 2.
    scale = 150. / person_height

    image_scaled, scale_factors = resize_img(image, scale)
    if save_scaled_img_dir is not None:
        cv2.imwrite(save_scaled_img_dir, image_scaled[:,:, [2,1,0]])
    depth_scaled, _ = resize_img(depth, scale)
    height, width = image_scaled.shape[:2]
    joints_scaled = np.copy(joints)
    joints_scaled[0,:] *= scale_factors[0] # x
    joints_scaled[1,:] *= scale_factors[1] # y
    center_scaled = np.round(center * scale_factors).astype(np.int)


    # Crop 300x300 around the center
    margin = 150
    start_pt = np.maximum(center_scaled - margin, 0).astype(int)
    end_pt = (center_scaled + margin).astype(int)
    end_pt[0] = min(end_pt[0], width)
    end_pt[1] = min(end_pt[1], height)
    image_scaled = image_scaled[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
    depth_scaled = depth_scaled[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0]]

    # Update others too.
    joints_scaled[0,:] -= start_pt[0]
    joints_scaled[1,:] -= start_pt[1]
    center_scaled -= start_pt
    height, width = image_scaled.shape[:2]
    
    # scale camera: Flength, px, py
    cam_scaled = np.copy(cam)
    cam_scaled[0] *= scale
    cam_scaled[1] *= scale_factors[0]
    cam_scaled[2] *= scale_factors[1]
    # Update principal point:
    cam_scaled[1] -= start_pt[0]
    cam_scaled[2] -= start_pt[1]
    
    return image_scaled, depth_scaled, joints_scaled, height, width, cam_scaled, start_pt, scale_factors, center_scaled

""" add the pair <pose/shape, joints3d from smpl layer> to tfrecord """
def add_smpl_joints3d_pair_to_tfrecord_file(
        frame_fname, joints2d, joints3d_smpl, pose, shape, 
        gender_vect = np.array([.0,.0,1.0]).astype(np.float32), # means 'nuetral'
        trans = np.array([.0, .0, .0]).astype(np.float32), 
        tf_writer = None
        ):

    # joints is 3 x 24 (lsp order)
    # All kps are visible.
    min_pt = np.min(joints2d[:2, :], axis=1)
    max_pt = np.max(joints2d[:2, :], axis=1)
    #NOTE: detect there exists person in this image frame or not;
    success = _joints_is_in_image(min_pt) and _joints_is_in_image(max_pt)
    if not success:
        #print ("[**] probabily no person apper in this frame {} ...".format(frame_fname))
        return 1

    #print ('smpl pose and joints 3d pair ...') 
    example = convert_to_example_smpl_pose_joints3d_pairs(frame_fname,
            joints3d_smpl, trans, pose, shape, gender_vect)
    
    tf_writer.write(example.SerializeToString())

    # Finally return how many were written.
    return 1 # 1 person added

def add_to_tfrecord_file(frame_fname, image, depth, joints2d,  joints3d, pose, shape, 
                     writer,
                     cam, 
                     is_save_to_h5files = False,
                     coder = None, 
                     isSavepfm = False,
                     out_dir = '',
                     gender_vect = np.array([.0,.0,1.0]).astype(np.float32) # means 'nuetral'
                     ):

    #NOTE: this cv2 i/o makes the tfrecord files slow !!! ???
    image_scaled, depth_scaled, joints_scaled, height, width, cam_scaled, \
    start_pt, scale_factors, center_scaled = parse_joints_and_scale_image(image,depth,joints2d, cam) 
    
    if image_scaled is None:
        #print ("[**] probabily no person apper in this frame. with depth shape : ", depth.shape, "image shape : ", image.shape, "depth_scaled shape :", depth_scaled.shape, "image_scaled shape : ", image_scaled.shape, 'frame_fname : ', frame_fname)
        return 1, None
    
    #NOTE: saved the values in millimeter as uint16 type;
    depth_scaled *= 1000
    #depth_scaled[depth_scaled >=65535 ] = 65535 # uint16 max valueose, shape, scale_factors, start_pt, cam_scaled)
    depth_scaled[depth_scaled >=65535 ] = 0
    depth_scaled = depth_scaled.astype(np.uint16)

    #print ("[**] scalded depth shape : ", depth_scaled.shape, "scaled image shape : ", image_scaled.shape)
    if isSavepfm:
        #e.g., frame_fname = ./datasets/surreal/cmu/val/run0/03_01/03_01_c0008_frame_000;
        print ("our_dir = ", out_dir)
        
        tmp_name = frame_fname.split("/")[-1] 
        tmp_name_img = join(out_dir, "img-scale-{}.pfm".format(tmp_name))
        pfm.save(tmp_name_img, image_scaled.astype(np.float32))
        print (" save pfm : %s" %tmp_name_img)

        tmp_name_dep = join(out_dir, "dep-scale-{}.pfm".format(tmp_name))
        pfm.save(tmp_name_dep, depth_scaled.astype(np.float32))
        print (" save pfm : %s" % tmp_name_dep)
        isSavepfm = False

    # Encode image to jpeg format:
    image_data_scaled = coder.encode_jpeg(image_scaled)
    # NOTE: Encode depth to raw or string or byte format:
    depth_data_scaled = depth_scaled.tobytes()
    
    #NOTE: here we have make sure that  joints3d is in shape [14, 3];
    # which is required by tfrecord example write and parse;
    assert (joints3d.shape[1] == 3)
    if is_save_to_h5files:
        tmp_img_name = frame_fname
        example = convert_to_h5_wmosh_wdepth(
                        depth_scaled.astype(np.float32), 
                        image_scaled.astype(np.float32), 
                        tmp_img_name, height, width, joints_scaled, center_scaled, 
                        joints3d, pose, shape, scale_factors, 
                        start_pt, cam_scaled, gender_vect)

    else:
        example = convert_to_example_wmosh_wdepth(depth_data_scaled, image_data_scaled, frame_fname,
            height, width, joints_scaled, center_scaled, joints3d, pose, 
            shape, scale_factors, start_pt, cam_scaled, gender_vect )
        writer.write(example.SerializeToString())

    # Finally return how many were written.
    return 1, example # 1 person added

def count_images_num(img_dir # e.g., == "Datasets/surreal/cmu/test/run0/"
        ):

    onlydirs = [d for d in listdir(img_dir) if isdir( join(img_dir, d))]
    img_num = 0
    for d in onlydirs:
        cur_dir = join(img_dir, d)
        sequences = [s for s in listdir(cur_dir) if isfile(join(cur_dir,s)) and "_info.mat" in s]
        s_num = len(sequences)
        #print ("cur_dir %s contains %d sequences" % (cur_dir, s_num))
        for s in sequences:
            info_dict = load_anno_mat(join(cur_dir, s))
            frame_num = info_dict['joints2D'].shape[2] if len(info_dict['joints2D'].shape) == 3 else 1
            img_num += frame_num
    print ("Total image number %d" % img_num)
    return img_num




""" the structure of the dir tree """
#-----subject: cmu/train/run2
#----------subdir:  <sequenceName>/ #e.g. 01_01
#---------------sequences:  <sequenceName>/ #e.g. 01_01_c%04d.mp4
#---------------------------frames extracted from the sequence video:  #e.g. 100 frames in the video 01_01_c%04d.mp4;
def process_subject_dir( ppl_num_prev, num_imgs, writer, img_dir, subdirs, 
        task_type = 'add_to_tfrecord', # 'add_to_tfrecord' or 'add_smpl_joints3d_pair'
        coder = None, 
        isSavepfm = False, 
        out_dir = '',
        is_save_to_h5files = False,
        h5_flags_list = None,
        isDisplay = False):
    # list all the sub_dirs under this current img_dir;
    #sub_dirs = [d for d in listdir(img_dir) if isdir( join(img_dir, d))]
    
    ppl_num = 0
    #ppl_num_valid = 0
    ppl_num += ppl_num_prev
    
    #onlydirs = [d for d in listdir(img_dir) if isdir( join(img_dir, d))]
                
    _, cam = get_intrinsic()
    
    if is_save_to_h5files:
        h5_example_curdir = []
    else:
        h5_example_curdir = None
    
    for d in subdirs: # for each sequence, e.g., d = 
        ppl_num_valid = 0
        cur_dir = join(img_dir, d)
        sequences = [s for s in listdir(cur_dir) if isfile(join(cur_dir,s)) and ".mp4" in s]
        for s in sequences: # for each video in the sequence
            s_name = s.split(".")[0]
            #print ("s_name = %s" % s_name)
            video_fname = join(cur_dir, s_name + ".mp4" )
            depth_fname = join(cur_dir, s_name + "_depth.mat")
            info_fname = join(cur_dir, s_name + "_info.mat")
             
            img_dict, dep_dict, info_dict, frame_num = load_sequence_data_and_anno_mat(video_fname, depth_fname, info_fname, isDisplay)
            #print ("[***] video_fname = %s, has frame_num = %d" % (video_fname, frame_num))
            # added by CCJ for projecting world-coordinate values to camera-coordinate values;
            E = get_extrinsic(np.reshape(info_dict['camLoc'], [3,-1]))

            #frame_num = info_dict['joints2D'].shape[2] if len(info_dict['joints2D'].shape) == 3 else 1

            
            #do_transform =  False
            do_transform =  True # already test, this value must be TRUE;
            if do_transform:
                    # <========= LOAD SMPL MODEL BASED ON GENDER
                if info_dict['gender'][0] == 0:  # f
                    m = load_model('/home/hmr/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
                elif info_dict['gender'][0] == 1:  # m
                    m = load_model('/home/hmr/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl')

                root_pos = m.J_transformed.r[0]
                zrot = info_dict['zrot']
                zrot = zrot[0]  # body rotation in euler angles
                RzBody = np.array(((math.cos(zrot), -math.sin(zrot), 0),
                    (math.sin(zrot), math.cos(zrot), 0),
                    (0, 0, 1)))

            #gender_list = ['female', 'male', 'neutral']
            gender = info_dict['gender'][0]
            if gender == 0:
                gender_vect = np.array([1.0, .0, .0]).astype(np.float32) # gender vector - female
            elif gender == 1:
                gender_vect = np.array([.0, 1.0, .0]).astype(np.float32) # gender vector - male
            else:
                gender_vect = np.array([.0, .0, 1.0]).astype(np.float32) # gender vector - neutral

            for i in range(0, frame_num): # for each frame
                #NOTE:for debugging;
                #SMALL_DATA = False
                #SMALL_DATA = True
                if SMALL_DATA:
                    if not exists(join(cur_dir, "100-samples", s_name[6:] + "_frame_%03d.jpg" % i)):
                    #if not exists(join(cur_dir, "5-samples", s_name[6:] + "_frame_%03d.jpg" % i)):
                    #if not exists(join(cur_dir, "1-samples", s_name[6:] + "_frame_%03d.jpg" % i)):
                    #if not exists(join(cur_dir, "20-samples", s_name[6:] + "_frame_%03d.jpg" % i)):
                        #print (join(cur_dir, "100-samples", s_name[6:] + "_frame_%03d.jpg" % i))
                        ppl_num += 1
                        continue
                
                #print ("[***]", s_name + "/frame_%03d.jpg" % i )
                currFrameJoints2D = info_dict['joints2D'][:,:, i] if frame_num > 1 else info_dict['joints2D']
                currFrameJoints3D = info_dict['joints3D'][:,:, i] if frame_num > 1 else info_dict['joints3D']
                currFrameJoints2D_swap = swap_right_left_joints(currFrameJoints2D)
                currFrameJoints3D_swap = swap_right_left_joints(currFrameJoints3D)
                # return joints2d: 3 x 14 shape ; 
                # joints3d: 14x 3 shape;
                joints2d, joints3d = read_joints(currFrameJoints2D_swap, currFrameJoints3D_swap, E)
                image = img_dict['frame_%03d' % i] # this frame name is defiend by my own code, I prefer the zero-based index;
                
                depth = dep_dict['depth_%d' % (i+1)] # the depth_1, index is 1-based, defiend by the dataset itselt;
                # pose : 72 x T, where T is the number of frames;
                pose =   np.reshape(info_dict['pose'][:, i], [-1]) # pose: 1D, (72,) 
                
                #NOTE: 
                pose = swap_right_left_pose(pose)
                #pose[0:3] = .0,.0,.0
                # shape : 10 x T, where T is the number of frames;
                shape =  np.reshape(info_dict['shape'][:, i], [-1]) # shape: 1D, (10,)
              
                #******************************
                #NOTE: ******************
                #******************************
                if do_transform:
                    pose[0:3] = rotateBody(RzBody, pose[0:3])

                #frame_fname = s_name + '_frame_%03d' %i # e.g, ==  03_01_c0008_frame_000;
                frame_fname = join(cur_dir, s_name + '_frame_%03d' %i) # e.g, ==  ./datasets/surreal/cmu/val/run0/03_01/03_01_c0008_frame_000;
                #pfm.save("/home/hmr/datasets/surreal/cmu-small/train/run0/05_20/05_20_c0002_depth_%03d"%i + ".pfm", depth.astype(np.float32))
                
                #""" generate the data pair of smpl pose and joints3d inferred by this smpl pose"""
                if task_type == 'add_smpl_joints3d_pair':
                    m.pose[:] = pose
                    m.betas[:] = shape
                    trans = joints3d[0] - root_pos
                    #m.trans[:] = trans
                    # here we do not apply the trans to smpl, instead we keep this trans value if need in future;
                    joints3d_smpl = m.J_transformed.r
                    ppl_num += add_smpl_joints3d_pair_to_tfrecord_file(
                            frame_fname, joints2d, joints3d_smpl, pose, shape,
                            gender_vect, trans, writer)
                #""" generate the full surreal data for network training """
                else:
                #elif task_type == 'add_to_tfrecord':
                    tmp_ppl, example = add_to_tfrecord_file(
                        frame_fname, image, depth, joints2d, 
                        joints3d, pose, shape, writer, 
                        cam, 
                        is_save_to_h5files,
                        coder, isSavepfm, 
                        out_dir,
                        gender_vect)

                    ppl_num += tmp_ppl
                    
                    if is_save_to_h5files:
                        #print ("[???] ppl_num = ",ppl_num_valid)
                        if example is not None and h5_flags_list[ppl_num_valid] == 1:
                            h5_example_curdir.append(example)

                ppl_num_valid += 1
                if ppl_num % 5000 == 0:
                    print('Done img %d/%d' % (ppl_num, num_imgs))
                isSavepfm = False # just save pfm files once
        print ("d = %s, ppl_num_valid = %d" %(d, ppl_num_valid))
    
    return ppl_num, h5_example_curdir


def process_surreal(img_dir, num_imgs, coder, out_path, out_dir, num_shards, 
    task_type = 'add_to_tfrecord', 
    is_save_to_h5files = False,
    h5_flags_list = None
    ):

    # list all the sub_dirs under this current img_dir;
    sub_dirs = [d for d in listdir(img_dir) if isdir( join(img_dir, d))]
    
    img_idx = 0
    # Count on shards:w
    fidx = 0
    isDisplay = False
    i_start = 0
    i_end = 0
    step_dirs = 10
        
    h5_examples_shuff_list = []
    
    
    while img_idx < num_imgs:
        
        tf_filename = out_path % fidx
        print('Starting tfrecord file %s' % tf_filename)
        with tf.python_io.TFRecordWriter(tf_filename) as writer:
            # Count on total ppl in each shard
            num_ppl = 0
            while img_idx < num_imgs and num_ppl < num_shards:
                if img_idx % 50 == 0:
                    print('Reading img %d/%d' % (img_idx, num_imgs))
                    isDisplay = True
                else:
                    isDisplay=False
                if img_idx in [0, 50,100,1000, 5000, 11000]:
                    isSavepfm = True
                else:
                    isSavepfm = False
                i_end = i_start + step_dirs

                cur_num_ppl, h5_example_curdir = process_subject_dir(img_idx, num_imgs, writer, img_dir, 
                        sub_dirs[i_start:i_end], 
                        task_type,
                        coder, isSavepfm, out_dir,
                        is_save_to_h5files,
                        h5_flags_list,
                        isDisplay)
                i_start = i_start + step_dirs
                num_ppl += cur_num_ppl
                img_idx += cur_num_ppl
                print ("img_idx = %d, num_imgs = %d" %(img_idx, num_imgs))
                if is_save_to_h5files:
                    h5_examples_shuff_list.extend(h5_example_curdir)
        fidx += 1

    return h5_examples_shuff_list

def main(unused_argv):
    print('Saving results to %s' % FLAGS.output_directory)
    
    if not exists(FLAGS.output_directory):
        makedirs(FLAGS.output_directory)
    
    if 0:
        """ counting image number """
        img_nums = []
        sub_names = []
        sums = 0
        #for i in range(0, 9):
        for i in range(0,1):
            subject = surreal_subjects[i]
            img_dir = join(FLAGS.img_directory, subject)
            out_dir = join(FLAGS.output_directory, subject)
            if not exists(out_dir):
                makedirs(out_dir)
                print ("mkdir %s" % out_dir)
            tmp_num = count_images_num(img_dir)
            img_nums.append(tmp_num)
            sub_names.append(subject)
            sums +=  tmp_num
            #print ("processing %s , has %d images " % (subject, tmp_num))

        print ('subjetcs: {}'.format(sub_names))
        print ('img_nums: {}, Total = {}'.format(img_nums, sums))
        sys.exit()
    
    """ counting image number """
    coder = ImageCoder()
    is_train = True 
    
    for subject in surreal_eval_subjects:
    #for subject in surreal_subjects:
    #for subject in surreal_small_subjects:
        if 'cmu-small' in subject: 
            SMALL_DATA = True
        else:
            SMALL_DATA = False

        img_dir = join(FLAGS.img_directory, subject)
        out_dir = join(FLAGS.output_directory, subject)
        num_imgs = surreal_num_imgs_dict[subject]
        print ("subject = %s, num_imgs = %d" % (subject, num_imgs))

        if not exists(out_dir):
            makedirs(out_dir)
            print ("mkdir %s" % out_dir)
        if is_train:
            out_path = join(out_dir, 'train_%06d.tfrecord')
        else:
            out_path = join(out_dir, 'test_%06d.tfrecord')
            print('Not implemented for test data')
            exit(1)
        
        """ the structure of the dir tree """
        #-----subject: cmu/train/run2
        #----------subdir:  <sequenceName>/ #e.g. 01_01
        #---------------sequences:  <sequenceName>/ #e.g. 01_01_c%04d.mp4
        #---------------------------frames extracted from the sequence video:  #e.g. 100 frames in the video 01_01_c%04d.mp4;
        print ('task type = %s' % FLAGS.task_type)
        
        if FLAGS.is_save_to_h5files:
            # Here we select 3000 samples
            shuffle_id = np.random.permutation(num_imgs)
            tmp_n = min(num_imgs, 6000)
            shuffle_id = shuffle_id[0:tmp_n]
            h5_flags_list = np.zeros((num_imgs), dtype = np.int)
            h5_flags_list[shuffle_id] = 1 # array of length num_imgs, with 0 or 1 values as flag to save or not;
        else:
            h5_flags_list = None
        h5_examples_shuff_list = process_surreal(img_dir, num_imgs, coder, 
                        out_path, out_dir, FLAGS.train_shards, 
                        FLAGS.task_type, FLAGS.is_save_to_h5files, 
                        h5_flags_list)
        if FLAGS.is_save_to_h5files:
            new_h5_examples = []
            num_tmp = min(len(h5_examples_shuff_list), 1000)
            for i in range(0, num_tmp):
                tmp_dict = {'example_%d' % (i) : h5_examples_shuff_list[i]}
                #tmp_dict = {'example_%d' % (i) : h5_example_list[shuffle_id[i]]}
                new_h5_examples.append(tmp_dict)
            
            h5_filename = join(out_dir, 'eval-mpjpepa-%d-samples.h5' % num_tmp)
            dd.io.save(h5_filename, new_h5_examples, compression = None)
            print ("saved h5_examples_shuff_list to {}".format(h5_filename))

        


if __name__ == '__main__':
    tf.app.run()