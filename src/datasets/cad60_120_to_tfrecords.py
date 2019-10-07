"""
# added by Changjiang Cai:
Convert CAD60/120 to TFRecords.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib; matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import sys,json,os,cv2
from os import makedirs
from os.path import join, exists
from time import time

import numpy as np

import tensorflow as tf

from .common import (convert_to_example_wdepth, 
                    convert_to_example_wmosh_wdepth,
                    convert_to_h5_wmosh_wdepth,
                    ImageCoder, resize_img)
import src.pfmutil as pfm
import glob
from src.util.cad_60_120_util import get_cad_2_lsp_idx,getPixelValuesFromCoords,read_cad_txt_annotation
from src.pose_perceptron import SMPLTrainer,get_config, save_to_mesh_file
from src.benchmark.eval_util import compute_error_one_sample

import deepdish as dd

cad_small_subjects = [
    #'cad-60-small-1-img-person1', # 1
    #'cad-60-small-1-img-person2', # 1
    #'cad-60-small-2-imgs-person2' # 2
    #'cad-120-small-1-img-sub1', # 1
    #'cad-120-small-1-img-sub3', # 1
    #'cad-120-small-1-img-sub4', # 1
    'cad-60-small-5-imgs-person1', # 5
    #'cad-60-small-3-diff-imgs-person1', # 3
    ]

cad_eval_subjects = [ 
        'cad-120/Subject4_rgbd_images',
        ]

cad_subjects = [ 
        'cad-60/Person1', 'cad-60/Person2', 'cad-60/Person3', 'cad-60/Person4',
        'cad-120/Subject1_rgbd_images', 'cad-120/Subject3_rgbd_images',
        'cad-120/Subject4_rgbd_images','cad-120/Subject5_rgbd_images',
        ]

cad_subjects_dict = {
    # cad 60
    'cad-60/Person1': {'gender': 'm', 'sampleNum': 19851, 'sampleNum_mpjpe_pa_thre70': 5980},
    'cad-60/Person2': {'gender': 'f', 'sampleNum': 18675, 'sampleNum_mpjpe_pa_thre70': 3953},
    'cad-60/Person3': {'gender': 'f', 'sampleNum': 19465, 'sampleNum_mpjpe_Pa_thre70': 3048},
    'cad-60/Person4': {'gender': 'm', 'sampleNum': 22321, 'sampleNum_mpjpe_pa_thre70': 6721},
    # cad 120
    'cad-120/Subject1_rgbd_images': {'gender': 'f', 'sampleNum': 15227, 
                                     'sampleNum_mpjpe_pa_thre70': 2883},
    'cad-120/Subject3_rgbd_images': {'gender': 'f', 'sampleNum': 17277, 
                                     'sampleNum_mpjpe_pa_thre70': 2358},
    'cad-120/Subject4_rgbd_images': {'gender': 'm', 'sampleNum': 13281, 
                                     'sampleNum_mpjpe_pa_thre70': 354},
    'cad-120/Subject5_rgbd_images': {'gender': 'm', 'sampleNum': 19348, 
                                     'sampleNum_mpjpe_pa_thre70': 1732}, 
    # cad60/120: Total = 145445 images; Total sampleNum_mpjpe_thre70 = 27029 images;
    
    # small dataset, for debugging
    'cad-60-small-1-img-person1': {'gender': 'm', 'sampleNum': 1, },
    'cad-60-small-1-img-person2': {'gender': 'f', 'sampleNum': 1, },
    'cad-60-small-2-imgs-person2': {'gender': 'f', 'sampleNum': 2, },
    'cad-120-small-1-img-sub1' : {'gender': 'f', 'sampleNum': 1, },
    'cad-120-small-1-img-sub3' : {'gender': 'f', 'sampleNum': 1, },
    'cad-120-small-1-img-sub4' : {'gender': 'm', 'sampleNum': 1, },
    'cad-60-small-5-imgs-person1': {'gender': 'm', 'sampleNum': 5, },
}



def _get_cad_60_img_size():
    return 240, 320

def _get_cad_120_img_size():
    return 480, 640

# load the json files for 2d joints and bbox;
def load_anno_json(img_directory):
    annoDicts = {}
    num_imgs = 0
    for sub in cad_subjects:
        json_anno_file = join(img_directory, sub, 'densePoseAnnos.json')
        with open(json_anno_file, "r") as f:
            data=json.load(f)
        annoDicts[sub] = data
        num_imgs += len(data['images'])
        print (len(data['images']))
    print ("Total image number is %d" %num_imgs)
    #return annoDicts, num_imgs
    return annoDicts



#NOTE: extratct 14 lsp joints from the joints detected by DensePose.
    #joints_by_densepose = [
    #    "nose", "left_eye", "right_eye", "left_ear", "right_ear", 
    #    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
    #    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", 
    #    "right_knee", "left_ankle", "right_ankle"] # 17 joints;

def read_joints_detected_by_densepose(currAnnoDict):
    """
    Reads joints in the common joint order.

    Returns:
      joints: 3 x |common joints|
    """
    # Mapping from DENSEPOSE joints to LSP joints (0:13). In this roder:
    _COMMON_JOINT_IDS = [
        16,  # R ankle
        14,  # R knee
        12,  # R hip
        11,  # L hip
        13,  # L knee
        15,  # L ankle
        10,  # R Wrist
        8,  # R Elbow
        6,  # R shoulder
        5,  # L shoulder
        7,  # L Elbow
        9,  # L Wrist
        #N/A,  # Neck top
        #N/A,  # Head top
    ]

    # Go over each common joint ids
    joints = np.zeros((3, len(_COMMON_JOINT_IDS)+2 )) # +2 : for missing neck and head joints;
    for i, jid in enumerate(_COMMON_JOINT_IDS):
        joints[0, i] =  currAnnoDict["dp_x"][jid] # x
        joints[1, i] =  currAnnoDict["dp_y"][jid] # y
        joints[2, i] =  1 # visible
    
    return joints

def _joints_is_in_image(pt, imgH = 240, imgW = 320):
    x = pt[0]
    y = pt[1]
    return x * (imgW - x) > 0 and y * (imgH - y)


def parse_people_via_joints(lsp_joints2d, imgH, imgW):
    '''
    Parses people from joints 2d gt annotation.
    Input:
      joints is 3 x 14 (lsp order), i.e., (x,y,vis) x l4
    '''
    people = []
    # joints is 3 x 14 (lsp order)
    # All kps are visible.
    min_pt = np.min(lsp_joints2d[:2, :], axis=1)
    max_pt = np.max(lsp_joints2d[:2, :], axis=1)

    success = _joints_is_in_image(min_pt, imgH, imgW) and _joints_is_in_image(max_pt, imgH, imgW)
    if success:
        person_height = np.linalg.norm(max_pt - min_pt)
        center = (min_pt + max_pt) / 2.
        scale = 150. / person_height
        people.append((scale, min_pt, max_pt, center))
    return people


def parse_people_via_bbox(currAnnoDict):
    '''
    Parses people from rect annotation.
    Assumes input is train data.
    Input:
      img_dir: str
      anno_info: annolist[img_id] obj
      single_persons: rect id idx for "single" people

    Returns:
      people - list of annotated single-people in this image.
      Its Entries are tuple (label, img_scale, obj_pos)
    '''
    people = []
    joints = read_joints_detected_by_densepose(currAnnoDict)
    # Compute the scale using the bounding box so the person in the bbox is around 150px.
    currBbox_w=currAnnoDict["bbox"][2]
    currBbox_h=currAnnoDict["bbox"][3]
    # using the bounding box diagonal length
    person_height = np.linalg.norm(np.array(currAnnoDict['bbox'][2:4]))
    scale = 150. / person_height
    pos = currAnnoDict['bbox'][0:2]
    print ("[**] bounding boxs position is {}, size is {}. scale is {}".format(
        np.array(currAnnoDict['bbox'][0:2]), 
        np.array(currAnnoDict['bbox'][2:4]), 
        scale
        ))
    people.append((joints, scale, pos, currBbox_h, currBbox_w))
    return people


def add_to_tfrecord(anno_dict, image_dict, img_dir, writer, isDisplay = False, coder = None, 
                    isSavepfm = False, 
                    out_dir = '', isDebug = False):
    """
    Add each "single person" in this image.
    anno - the entire annotation file.

    Returns: The number of people added.
    """
    people = parse_people_via_bbox(anno_dict)
    if len(people) == 0:
        return 0
    # Add each people to tf record
    img_name = image_dict['file_name']
    depth_name = img_name.replace('RGB_', 'Depth_')
    image_path = join(img_dir, img_name)
    depth_path = join(img_dir, depth_name)

    #NOTE: this cv2 i/o makes the tfrecord files slow !!! ???
    #image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    #depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    #assert (isinstance(depth, np.ndarray))
    #pfm.save("/home/hmr/datasets/tf_datasets/cad_60_120/cad-60-small/dep-cv2-ori.pfm", 
    #          cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32))


    with tf.gfile.FastGFile(image_path, 'rb') as f:
        image_data = f.read()

    #image = sess.run( tf.image.decode_png(image_data, channels = 3, dtype=tf.uint8))
    image = coder.decode_png(image_data)

    with tf.gfile.FastGFile(depth_path, 'rb') as f:
        depth_data = f.read()
    
    #NOTE: ????
    #depth = sess.run(tf.image.decode_png(depth_data, channels = 0, dtype=tf.uint16)).astype(np.float32)
    depth = coder.decode_png_uint16(depth_data)
    #pfm.save("/home/hmr/datasets/tf_datasets/cad_60_120/cad-60-small/dep-ori.pfm", depth.astype(np.float32))


    for joints, scale, pos, bbox_h, bbox_w in people:
        
        # for debugging only, becuase 
        # the cv.circle and cv.rectangle will chagne the image, 
        # which will also be saved to tfrecord files
        if isDebug:
            pt1 = (int (pos[0]), int(pos[1]))
            pt2 = (int(pos[0]+ bbox_w), int(pos[1] + bbox_h))
            print ("pt1 = {}, pt2 = {}".format(pt1, pt2))
            cv2.rectangle(image, pt1, pt2, (255,0,0), 2)
            visible = joints[2, :].astype(bool)
            min_pt = np.min(joints[:2, visible], axis=1)
            max_pt = np.max(joints[:2, visible], axis=1)
            center = (min_pt + max_pt) / 2.
            print ("[**] orig min_pt is {}, max_pt is {}, center is then {}".format(min_pt, max_pt, center))
            
            cv2.circle(image, (int (min_pt[0]), int(min_pt[1])), radius = 2, color = (0,0,255), thickness = 1)
            cv2.circle(image, ( int(max_pt[0]), int(max_pt[1])), radius = 2, color = (0,255,0), thickness = 1)
            
            for nk in range(joints.shape[1]):
                cv2.circle(image,( int(joints[0,nk]), int (joints[1, nk])), radius = 2, color = (0,0,255), thickness = 2)

            
            #tmp_name = image_path.split("/")[-2] + '-' + image_path.split("/")[-1]
            tmp_name = image_name.split("/")[-3] + '-' + image_name.split("/")[-2] + '-' + image_name.split("/")[-1]
            #print ("????? ", tmp_name)
            tmp_name_img = join(out_dir, "{}_img_orig.pfm".format(tmp_name[:-4]))
            pfm.save(tmp_name_img, image.astype(np.float32))
            print (" saved pfm : %s" %tmp_name_img)
            tmp_name_dep = join(out_dir, "{}_dep_orig.pfm".format(tmp_name[:-4]))
            pfm.save(tmp_name_dep, depth.astype(np.float32))
            print (" saved pfm : %s" %tmp_name_dep)

        # Scale image:
        image_scaled, scale_factors = resize_img(image, scale)
        depth_scaled, _ = resize_img(depth, scale)
        if len(depth_scaled.shape) == 2:
            depth_scaled = np.expand_dims(depth_scaled, axis = -1)
            #print ("[??] expand_dims depth_scaled shape : ", depth_scaled.shape)
        height, width = image_scaled.shape[:2]
        if isDisplay:
            print ("loading img = %s, depth = %s, resized img h = %d, w = %d" % (image_path, depth_path,height, width))
        
        joints_scaled = np.copy(joints)
        joints_scaled[0, :] *= scale_factors[0]
        joints_scaled[1, :] *= scale_factors[1]

        if isSavepfm:
            #tmp_name = image_path.split("/")[-2] + '-' + image_path.split("/")[-1]
            tmp_name = image_name.split("/")[-3] + '-' + image_name.split("/")[-2] + '-' + image_name.split("/")[-1]
            tmp_name_img = join(out_dir, "{}_img_scaled.pfm".format(tmp_name[:-4]))
            print (" save pfm : %s" %tmp_name_img)
            pfm.save(tmp_name_img, image_scaled.astype(np.float32))
            tmp_name_dep = join(out_dir, "{}_dep_scaled.pfm".format(tmp_name[:-4]))
            pfm.save(tmp_name_dep, depth_scaled.astype(np.float32))


        visible = joints[2, :].astype(bool)
        min_pt = np.min(joints_scaled[:2, visible], axis=1)
        max_pt = np.max(joints_scaled[:2, visible], axis=1)
        center = (min_pt + max_pt) / 2.
        #print ("[**] scaled min_pt is {}, max_pt is {}, center is then {}".format(min_pt, max_pt, center))

        ## Crop 224 x 224 around this image..
        margin = 112
        start_pt = np.maximum(center - margin, 0).astype(int)
        end_pt = (center + margin).astype(int)
        if start_pt[0] == 0:
            end_pt[0] = min(2*margin, width)
        else:
            end_pt[0] = min(end_pt[0], width)
        if start_pt[1] == 0:
            end_pt[1] = min(2*margin, height)
        else:
            end_pt[1] = min(end_pt[1], height)
        #end_pt[0] = min(end_pt[0], width)
        #end_pt[1] = min(end_pt[1], height)
        print ("[**] w,h: end_pt[0] = %d, end_pt[1] = %d" % (end_pt[0], end_pt[1]))
        image_scaled = image_scaled[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
        depth_scaled = depth_scaled[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
        # Update others oo.
        joints_scaled[0, :] -= start_pt[0]
        joints_scaled[1, :] -= start_pt[1]
        center -= start_pt
        height, width = image_scaled.shape[:2]
        print ("[**] ", image_scaled.shape, depth_scaled.shape)
        if isSavepfm:
            #tmp_name = image_path.split("/")[-2] + '-' + image_path.split("/")[-1]
            tmp_name = image_name.split("/")[-3] + '-' + image_name.split("/")[-2] + '-' + image_name.split("/")[-1]
            tmp_name_img = join(out_dir, "{}_img_extracted.pfm".format(tmp_name[:-4]))
            pfm.save(tmp_name_img, image_scaled.astype(np.float32))
            print (" saved pfm : %s" %tmp_name_img)
            tmp_name_dep = join(out_dir, "{}_dep_extracted.pfm".format(tmp_name[:-4]))
            pfm.save(tmp_name_dep, depth_scaled.astype(np.float32))
            print (" saved pfm : %s" %tmp_name_dep)
        #sys.exit()
        # Encode image to jpeg format:
        #image_data_scaled = sess.run(tf.image.encode_jpeg(image_scaled, format = 'rgb'))
        image_data_scaled = coder.encode_jpeg(image_scaled)
        # Encode depth to raw or string or byte format:
        # NOTE: convert depth to raw data bytes in the array.
        depth_data_scaled = depth_scaled.tobytes()
        #image_data_scaled = cv2.imencode('png', image_scaled, CV_IMWRITE_PNG_COMPRESSION = 0)
        #depth_data_scaled = cv2.imencode('png', depth_scaled, CV_IMWRITE_PNG_COMPRESSION = 0)
        example = convert_to_example_wdepth(depth_data_scaled, image_data_scaled, image_path, height, width, joints_scaled, center)
        writer.write(example.SerializeToString())

    # Finally return how many were written.
    return len(people)


def process_cad(num_imgs, annos_dict, images_dict, img_dir, out_path, out_dir, num_shards):

    coder = ImageCoder()
    img_idx = 0
    # Count on shards:w

    fidx = 0
    num_ppl = 0
    isDisplay = False

    while img_idx < num_imgs:
        
        tf_filename = out_path % fidx
        print('Starting tfrecord file %s' % tf_filename)
        with tf.python_io.TFRecordWriter(tf_filename) as writer:
            # Count on total ppl in each shard
            num_ppl = 0
            while img_idx < num_imgs and num_ppl < num_shards:
                if img_idx % 500 == 0:
                    print('Reading img %d/%d' % (img_idx, num_imgs))
                    isDisplay = True
                else:
                    isDisplay=False
                if img_idx in [0, 1, 25, 1000, 5000, 11000]:
                    isSavepfm = True
                else:
                    isSavepfm = False
                num_ppl += add_to_tfrecord(annos_dict[img_idx], images_dict[img_idx], img_dir, writer, isDisplay, coder, isSavepfm, out_dir)
                img_idx += 1 
        fidx += 1


def process_cad_gt(num_imgs, 
                   all_lsp_joints2d, # 3 x 14 x N
                   all_img_fnames, 
                   img_dir, 
                   out_path, 
                   out_dir, 
                   num_shards,
                   all_lsp_joints3d = None, # N x (14*3)
                   all_lsp_joints3d_pred = None, # N x 14 x 3, this is the regressed joints3d via SMPL regressor;
                   all_pred_pose = None, # N x (23*3)
                   all_pred_shape = None, # N x 10
                   gender_vect = None, # (3,)
                   cam = None, #(3,)
                   fwrite = None,
                   is_save_to_h5files = False, 
                   mpjpe_pa_err_thre = 70
                   ):

    coder = ImageCoder()
    img_idx = 0
    # Count on shards:w

    fidx = 0
    num_ppl = 0
    isDisplay = False
    assert(gender_vect.ndim == 1 and gender_vect.shape[0] == 3)
    assert (cam.ndim == 1 and cam.shape[0] == 3)

    if is_save_to_h5files:
        h5_example_list = []
        h5_filename = join(out_dir, 'eval-mpjpepa-thred-%dmm.h5' % int(mpjpe_pa_err_thre))
        print ('[????] Starting h5_filename file %s' % h5_filename)

    while img_idx < num_imgs:
        
        tf_filename = out_path % fidx
        print('[***] Starting tfrecord file %s' % tf_filename)
        
        with tf.python_io.TFRecordWriter(tf_filename) as writer:
            # Count on total people in each shard
            num_ppl = 0
            while img_idx < num_imgs and num_ppl < num_shards:
                if img_idx % 500 == 0:
                    print('Reading img %d/%d' % (img_idx, num_imgs))
                    isDisplay = True
                else:
                    isDisplay=False
                if img_idx in [0, 200, 1000, 5000, 11000]:
                    isSavepfm = True
                else:
                    isSavepfm = False
                if all_lsp_joints3d is None:
                    num_ppl += add_cad_gt_to_tfrecord(
                        all_lsp_joints2d[:,:,img_idx],
                        join(img_dir, all_img_fnames[img_idx]), 
                        writer, coder, isSavepfm, out_dir)
                else:
                    
                    tmp_ppl, example = add_cad_gt_with_smpl_to_tfrecord(
                        all_lsp_joints2d[:,:,img_idx], 
                        np.reshape(all_lsp_joints3d[img_idx, :], [14,3]), # N x (14*3)
                        all_lsp_joints3d_pred[img_idx, :, :], # N x 14 x 3
                        all_pred_pose[img_idx, :], # N x (23*3)
                        all_pred_shape[img_idx, :], # N x 10
                        cam, # (3, )
                        gender_vect, #(3,)
                        join(img_dir, all_img_fnames[img_idx]), 
                        writer, 
                        is_save_to_h5files, 
                        coder,
                        isSavepfm, 
                        out_dir, 
                        mpjpe_pa_err_thre,
                        )

                    num_ppl += tmp_ppl
                    
                    if is_save_to_h5files and example is not None:
                        h5_example_list.append(example)
                    
                img_idx += 1 
            
            print ("[***] {} has num_ppl = {}".format(tf_filename, num_ppl))
            if fwrite is not None:
                fwrite.write("[***] {} has num_ppl = {}\n".format(tf_filename, num_ppl))
        
        fidx += 1
            
    if is_save_to_h5files:
        #NOTE:3000 is OK, if using 4000 there will be error:  value too large to convert to int when saving to h5 file ??
        tmp_n = min(len(h5_example_list), 3000)
        h5_examples_shuff_list = []
        print ("h5_example_list has {} examples".format(tmp_n))
        shuffle_id = np.random.permutation(tmp_n)

        for i in range(0, tmp_n):
            tmp_dict = {'example_%d' % (shuffle_id[i]) : h5_example_list[shuffle_id[i]]}
            #tmp_dict = {'example_%d' % (i) : h5_example_list[shuffle_id[i]]}
            h5_examples_shuff_list.append(tmp_dict)
        h5_filename = h5_filename[:-3] + "-%d-samples" % tmp_n + '.h5'
        dd.io.save(h5_filename, h5_examples_shuff_list, compression = None)
        print ("saved h5_examples_shuff_list to {}".format(h5_filename))



def add_cad_gt_with_smpl_to_tfrecord(
        lsp_joints2d, 
        lsp_joints3d_with_torso_subtracted, 
        lsp_joints3d_with_torso_subtracted_pred, 
        smpl_pose, 
        smpl_shape, 
        cam, # (3,)  # Flength, px, py (principal point)
        gender_vect, # (3,) 
        image_name,
        writer,
        is_save_to_h5files, 
        coder,
        isSavepfm, 
        out_dir, 
        mpjpe_pa_err_thre = 70.0 , # e.g., = 70.0mm, reconstruction error threshold in mm;
        mpjpe_err_thre = 2000.0 , # threshold in mm;
        isDebug = False,
        smpl_model_path = './models/basicModel_f_lbs_10_207_0_v1.0.0.pkl,basicModel_m_lbs_10_207_0_v1.0.0.pkl,neutral_smpl_with_cocoplus_reg.pkl'
    ):

    """
    Add each "single person" in this image.
    anno - the entire annotation file.

    Returns: The number of people added.
    """
    # Add each people to tf record
    #print ('[***] loading %s' % image_name)
    
    """ computer the error, if larger than threshold, return 0 """
    # Convert to mm!
    err, err_pa = compute_error_one_sample(lsp_joints3d_with_torso_subtracted * 1000.0, # gt joints3d;
                    lsp_joints3d_with_torso_subtracted_pred * 1000.0) # regressed joints3d via SMPL regressor;
    #print ("[!!!] Processing {}, has mpjpe error = {} mm, mpjpe_pa error = {} mm".format(image_name, err, err_pa))
    if err > mpjpe_err_thre or err_pa > mpjpe_pa_err_thre:
        return 0, None
    
    depth_name = image_name.replace('RGB_', 'Depth_')
    with tf.gfile.FastGFile(image_name, 'rb') as f:
        image_data = f.read()
    image = coder.decode_png(image_data)
    with tf.gfile.FastGFile(depth_name, 'rb') as f:
        depth_data = f.read()
    depth = coder.decode_png_uint16(depth_data)
    imgH, imgW = image.shape[:2]

    people = parse_people_via_joints(lsp_joints2d, imgH, imgW)

    if len(people) == 0:
        return 0, None
    
    for scale, min_pt_orig, max_pt_orig, center_orig in people:
        # for debugging:
        # for debugging only, becuase 
        # the cv.circle and cv.rectangle will chagne the image, 
        # which will also be saved to tfrecord files
        if isDebug:
            cv2.rectangle(image, (int(min_pt_orig[0]), int(min_pt_orig[1])), 
                                 (int(max_pt_orig[0]), int(max_pt_orig[1])), 
                                 (255,0,0), 2)
            print ("[**] orig min_pt is {}, max_pt is {}, center is then {}".format(
                min_pt_orig, max_pt_orig, center_orig))
            
            cv2.circle(image, (int (min_pt_orig[0]), int(min_pt_orig[1])), radius = 2, color = (0,0,255), thickness = 1)
            cv2.circle(image, ( int(max_pt_orig[0]), int(max_pt_orig[1])), radius = 2, color = (0,255,0), thickness = 1)

            lsp_names = [
                   'R Ankle', 'R Knee', 'R Hip', 'L Hip', 'L Knee', 'L Ankle', 'R Wrist',
                   'R Elbow', 'R Shoulder', 'L Shoulder', 'L Elbow', 'L Wrist', 'Neck', 'Head']
            # left joints using blue color 
            left_joints = [3,4,5,9,10,11]
            # right joints using green color 
            right_joints = [0,1,2,6,7,8]
            # middle joints using red color 
            middle_joints = [12,13]
            
            for nk in left_joints:
                x = int(lsp_joints2d[0,nk])
                y = int(lsp_joints2d[1,nk])
                cv2.circle(image, (x,y), radius = 2, color = (0,0,255), thickness = 2)
                print ('in RGBD : joint {} {}, has r = {}, c = {}, z = {}'.format(
                    nk, lsp_names[nk], y, x, depth[y,x,0]))

            
            for nk in right_joints:
                x = int(lsp_joints2d[0,nk])
                y = int(lsp_joints2d[1,nk])
                cv2.circle(image, (x, y), radius = 2, color = (0,255,0), thickness = 2)
                print ('in RGBD : joint {} {}, has r = {}, c = {}, z = {}'.format(
                    nk, lsp_names[nk], y, x, depth[y,x,0]))
            for nk in middle_joints:
                cv2.circle(image,( int(lsp_joints2d[0,nk]), int(lsp_joints2d[1, nk])), 
                radius = 2, color = (255,0,0), thickness = 2)

            tmp_name = image_name.split("/")[-3] + '-' + image_name.split("/")[-2] + '-' + image_name.split("/")[-1]
            tmp_name_img = join(out_dir, "{}_img_orig.pfm".format(tmp_name[:-4]))
            pfm.save(tmp_name_img, image.astype(np.float32))
            tmp_name_dep = join(out_dir, "{}_dep_orig.pfm".format(tmp_name[:-4]))
            pfm.save(tmp_name_dep, depth.astype(np.float32))

        # Scale image:
        image_scaled, scale_factors = resize_img(image, scale)
        depth_scaled, _ = resize_img(depth, scale)
        if len(depth_scaled.shape) == 2:
            depth_scaled = np.expand_dims(depth_scaled, axis = -1)
        
        height, width = image_scaled.shape[:2]
        #print ("loading img = %s, depth = %s, img h = %d, w = %d, resized img h = %d, w = %d, scale = %f" % (
        #    image_name, depth_name, imgH, imgW, height, width, scale))
        
        joints_scaled = np.copy(lsp_joints2d)
        joints_scaled[0, :] *= scale_factors[0]
        joints_scaled[1, :] *= scale_factors[1]
        
        if isSavepfm:
            tmp_name = image_name.split("/")[-3] + '-' + image_name.split("/")[-2] + '-' + image_name.split("/")[-1]
            print ("????? ", tmp_name)
            tmp_name_img = join(out_dir, "{}_img_scaled.pfm".format(tmp_name[:-4]))
            print (" save pfm : %s" %tmp_name_img)
            pfm.save(tmp_name_img, image_scaled.astype(np.float32))
            tmp_name_dep = join(out_dir, "{}_dep_scaled.pfm".format(tmp_name[:-4]))
            pfm.save(tmp_name_dep, depth_scaled.astype(np.float32))

        visible = lsp_joints2d[2, :].astype(bool)
        min_pt = np.min(joints_scaled[:2, visible], axis=1)
        max_pt = np.max(joints_scaled[:2, visible], axis=1)
        center = (min_pt + max_pt) / 2.
        #print ("[**] scaled min_pt is {}, max_pt is {}, center is then {}".format(min_pt, max_pt, center))

        ## Crop 300x300 around this image..
        margin = 150
        ## Crop 224 x 224 around this image..
        #margin = 112
        start_pt = np.maximum(center - margin, 0).astype(int)
        end_pt = (center + margin).astype(int)
        if start_pt[0] == 0:
            end_pt[0] = min(2*margin, width)
        else:
            end_pt[0] = min(end_pt[0], width)
        if start_pt[1] == 0:
            end_pt[1] = min(2*margin, height)
        else:
            end_pt[1] = min(end_pt[1], height)
        #print ("[**] scaled, end_pt[0] = %d, end_pt[1] = %d" % (end_pt[0], end_pt[1]))
        image_scaled = image_scaled[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
        depth_scaled = depth_scaled[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
        # Update others too.
        joints_scaled[0, :] -= start_pt[0]
        joints_scaled[1, :] -= start_pt[1]
        center -= start_pt
        height, width = image_scaled.shape[:2]
        #print ("[**] ", image_scaled.shape, depth_scaled.shape)

        # scale camera: Flength, px, py
        cam_scaled = np.copy(cam)
        cam_scaled[0] *= scale
        cam_scaled[1] *= scale_factors[0]
        cam_scaled[2] *= scale_factors[1]
        # Update principal point:
        cam_scaled[1] -= start_pt[0]
        cam_scaled[2] -= start_pt[1]
        
        if isSavepfm:
            #tmp_name = image_name.split("/")[-2] + '-' + image_name.split("/")[-1]
            tmp_name = image_name.split("/")[-3] + '-' + image_name.split("/")[-2] + '-' + image_name.split("/")[-1]
            tmp_name_img = join(out_dir, "{}_img_extracted.pfm".format(tmp_name[:-4]))
            print (" save pfm : %s" %tmp_name_img)
            pfm.save(tmp_name_img, image_scaled.astype(np.float32))
            tmp_name_dep = join(out_dir, "{}_dep_extracted.pfm".format(tmp_name[:-4]))
            pfm.save(tmp_name_dep, depth_scaled.astype(np.float32))

        # Encode image to jpeg format:
        image_data_scaled = coder.encode_jpeg(image_scaled)
        # Encode depth to raw or string or byte format:
        # NOTE: convert depth to raw data bytes in the array.
        depth_data_scaled = depth_scaled.tobytes()

        assert (lsp_joints3d_with_torso_subtracted.shape[1] == 3)
        if smpl_pose.shape[0] == 69: # i.e., 23 * 3 = 69
            smpl_pose_new = np.zeros(72)
            smpl_pose_new[0] = np.pi
            #smpl_pose_new[1] = .0
            #smpl_pose_new[2] = .0
            smpl_pose_new[3:72] = smpl_pose
            smpl_pose = smpl_pose_new
        if isSavepfm:
            tmp_name = image_name.split("/")[-3] + '-' + image_name.split("/")[-2] + '-' + image_name.split("/")[-1]
            tmp_name_mesh = join(out_dir, "{}_smpl_regressed_mesh.obj".format(tmp_name[:-4]))
            
            np.set_printoptions(precision=4, suppress=True)
            #print ('[**] joints3d after root extracted :\n{}'.format(lsp_joints3d_with_torso_subtracted))
            #print ('[**] smpl_pose = {}'.format(smpl_pose))
            #print ('[**] smpl_shape = {}'.format(smpl_shape))
            #print ('[**] gender_vect = {}'.format(gender_vect))
            save_to_mesh_file(smpl_model_path, gender_vect, smpl_pose, 
                              smpl_shape, tmp_name_mesh)

        if is_save_to_h5files:
            tmp_name = image_name.split("/")[-3] + '-' + image_name.split("/")[-2] + '-' + image_name.split("/")[-1]
            tmp_img_name = tmp_name[:-4]
            example = convert_to_h5_wmosh_wdepth(
                           depth_scaled.astype(np.float32), 
                           image_scaled.astype(np.float32), 
                           tmp_img_name, height, width, joints_scaled, center, 
                           lsp_joints3d_with_torso_subtracted, 
                           smpl_pose, smpl_shape, scale_factors, 
                           start_pt, cam_scaled, gender_vect)
            
            #dd.io.save(h5filename, example, compression = None)
        
        else:
            #assert (smpl_pose.shape[0] == 72 and smpl_shape.shape[0] == 10 )
            example = convert_to_example_wmosh_wdepth(depth_data_scaled, image_data_scaled, 
                            image_name, height, width, joints_scaled, center, 
                            lsp_joints3d_with_torso_subtracted, 
                            smpl_pose, smpl_shape, scale_factors, 
                            start_pt, cam_scaled, gender_vect )
            
            writer.write(example.SerializeToString())

    # Finally return how many were written.
    return len(people), example



def add_cad_gt_to_tfrecord(lsp_joints2d, image_name, writer, coder = None,
                           isSavepfm = False, out_dir = '', isDebug = True
                           ):

    """
    Add each "single person" in this image.
    anno - the entire annotation file.

    Returns: The number of people added.
    """
    # Add each people to tf record
    #print ('[***] loading %s' % image_name)
    

    depth_name = image_name.replace('RGB_', 'Depth_')
    with tf.gfile.FastGFile(image_name, 'rb') as f:
        image_data = f.read()
    image = coder.decode_png(image_data)
    with tf.gfile.FastGFile(depth_name, 'rb') as f:
        depth_data = f.read()
    depth = coder.decode_png_uint16(depth_data)
    imgH, imgW = image.shape[:2]

    people = parse_people_via_joints(lsp_joints2d, imgH, imgW)

    if len(people) == 0:
        return 0
    
    for scale, min_pt_orig, max_pt_orig, center_orig in people:
        # for debugging:
        # for debugging only, becuase 
        # the cv.circle and cv.rectangle will chagne the image, 
        # which will also be saved to tfrecord files
        if isDebug:
            cv2.rectangle(image, (int(min_pt_orig[0]), int(min_pt_orig[1])), 
                                 (int(max_pt_orig[0]), int(max_pt_orig[1])), 
                                 (255,0,0), 2)
            print ("[**] orig min_pt is {}, max_pt is {}, center is then {}".format(
                min_pt_orig, max_pt_orig, center_orig))
            
            cv2.circle(image, (int (min_pt_orig[0]), int(min_pt_orig[1])), radius = 2, color = (0,0,255), thickness = 1)
            cv2.circle(image, ( int(max_pt_orig[0]), int(max_pt_orig[1])), radius = 2, color = (0,255,0), thickness = 1)

            lsp_names = [
                   'R Ankle', 'R Knee', 'R Hip', 'L Hip', 'L Knee', 'L Ankle', 'R Wrist',
                   'R Elbow', 'R Shoulder', 'L Shoulder', 'L Elbow', 'L Wrist', 'Neck', 'Head']
            # left joints using blue color 
            left_joints = [3,4,5,9,10,11]
            # right joints using green color 
            right_joints = [0,1,2,6,7,8]
            # middle joints using red color 
            middle_joints = [12,13]
            
            for nk in left_joints:
                x = int(lsp_joints2d[0,nk])
                y = int(lsp_joints2d[1,nk])
                cv2.circle(image, (x,y), radius = 2, color = (0,0,255), thickness = 2)
                print ('in RGBD : joint {} {}, has r = {}, c = {}, z = {}'.format(
                    nk, lsp_names[nk], y, x, depth[y,x,0]))

            
            for nk in right_joints:
                x = int(lsp_joints2d[0,nk])
                y = int(lsp_joints2d[1,nk])
                cv2.circle(image, (x, y), radius = 2, color = (0,255,0), thickness = 2)
                print ('in RGBD : joint {} {}, has r = {}, c = {}, z = {}'.format(
                    nk, lsp_names[nk], y, x, depth[y,x,0]))
            for nk in middle_joints:
                cv2.circle(image,( int(lsp_joints2d[0,nk]), int(lsp_joints2d[1, nk])), 
                radius = 2, color = (255,0,0), thickness = 2)

            tmp_name = image_name.split("/")[-3] + '-' + image_name.split("/")[-2] + '-' + image_name.split("/")[-1]
            tmp_name_img = join(out_dir, "{}_img_orig.pfm".format(tmp_name[:-4]))
            pfm.save(tmp_name_img, image.astype(np.float32))
            tmp_name_dep = join(out_dir, "{}_dep_orig.pfm".format(tmp_name[:-4]))
            pfm.save(tmp_name_dep, depth.astype(np.float32))

        # Scale image:
        image_scaled, scale_factors = resize_img(image, scale)
        depth_scaled, _ = resize_img(depth, scale)
        if len(depth_scaled.shape) == 2:
            depth_scaled = np.expand_dims(depth_scaled, axis = -1)
        
        height, width = image_scaled.shape[:2]
        #print ("loading img = %s, depth = %s, img h = %d, w = %d, resized img h = %d, w = %d, scale = %f" % (
        #    image_name, depth_name, imgH, imgW, height, width, scale))
        
        joints_scaled = np.copy(lsp_joints2d)
        joints_scaled[0, :] *= scale_factors[0]
        joints_scaled[1, :] *= scale_factors[1]
        
        if isSavepfm:
            tmp_name = image_name.split("/")[-3] + '-' + image_name.split("/")[-2] + '-' + image_name.split("/")[-1]
            print ("????? ", tmp_name)
            tmp_name_img = join(out_dir, "{}_img_scaled.pfm".format(tmp_name[:-4]))
            print (" save pfm : %s" %tmp_name_img)
            pfm.save(tmp_name_img, image_scaled.astype(np.float32))
            tmp_name_dep = join(out_dir, "{}_dep_scaled.pfm".format(tmp_name[:-4]))
            pfm.save(tmp_name_dep, depth_scaled.astype(np.float32))

        visible = lsp_joints2d[2, :].astype(bool)
        min_pt = np.min(joints_scaled[:2, visible], axis=1)
        max_pt = np.max(joints_scaled[:2, visible], axis=1)
        center = (min_pt + max_pt) / 2.
        #print ("[**] scaled min_pt is {}, max_pt is {}, center is then {}".format(min_pt, max_pt, center))

        ## Crop 300x300 around this image..
        #margin = 150
        ## Crop 224 x 224 around this image..
        margin = 112
        start_pt = np.maximum(center - margin, 0).astype(int)
        end_pt = (center + margin).astype(int)
        if start_pt[0] == 0:
            end_pt[0] = min(2*margin, width)
        else:
            end_pt[0] = min(end_pt[0], width)
        if start_pt[1] == 0:
            end_pt[1] = min(2*margin, height)
        else:
            end_pt[1] = min(end_pt[1], height)
        #print ("[**] scaled, end_pt[0] = %d, end_pt[1] = %d" % (end_pt[0], end_pt[1]))
        image_scaled = image_scaled[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
        depth_scaled = depth_scaled[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
        # Update others oo.
        joints_scaled[0, :] -= start_pt[0]
        joints_scaled[1, :] -= start_pt[1]
        center -= start_pt
        height, width = image_scaled.shape[:2]
        #print ("[**] ", image_scaled.shape, depth_scaled.shape)
        if isSavepfm:
            #tmp_name = image_name.split("/")[-2] + '-' + image_name.split("/")[-1]
            tmp_name = image_name.split("/")[-3] + '-' + image_name.split("/")[-2] + '-' + image_name.split("/")[-1]
            tmp_name_img = join(out_dir, "{}_img_extracted.pfm".format(tmp_name[:-4]))
            print (" save pfm : %s" %tmp_name_img)
            pfm.save(tmp_name_img, image_scaled.astype(np.float32))
            tmp_name_dep = join(out_dir, "{}_dep_extracted.pfm".format(tmp_name[:-4]))
            pfm.save(tmp_name_dep, depth_scaled.astype(np.float32))

        # Encode image to jpeg format:
        image_data_scaled = coder.encode_jpeg(image_scaled)
        # Encode depth to raw or string or byte format:
        # NOTE: convert depth to raw data bytes in the array.
        depth_data_scaled = depth_scaled.tobytes()
        example = convert_to_example_wdepth(depth_data_scaled, image_data_scaled, 
                                image_name, height, width, joints_scaled, center)
        writer.write(example.SerializeToString())

    # Finally return how many were written.
    return len(people)


def main(unused_argv):
    
    print('Saving results to %s' % FLAGS.output_directory)
    if not exists(FLAGS.output_directory):
        makedirs(FLAGS.output_directory)
    is_train = True 
    
    print ("[***] task_type_cad = {}".format(FLAGS.task_type_cad))
    
    if FLAGS.task_type_cad == 'joints_annotation_from_densepose':
        annos = load_anno_json(FLAGS.img_directory)
        for subject in cad_small_subjects:
        #for subject in cad_subjects:
            img_dir = join(FLAGS.img_directory, subject)
            out_dir = join(FLAGS.output_directory, subject)
            if not exists(out_dir):
                makedirs(out_dir)
                print ("mkdir %s" % out_dir)
            annos_dict  = annos[subject]["annotations"]
            images_dict = annos[subject]["images"]
            num_imgs = cad_subjects_dict[subject]['sampleNum']
            if is_train:
                out_path = join(out_dir, 'train_%03d.tfrecord')
            else:
                out_path = join(out_dir, 'test_%03d.tfrecord')
                print('Not implemented for test data')
                exit(1)
            
            print ("processing %s" % subject)
            process_cad(num_imgs, annos_dict, images_dict, img_dir, out_path, out_dir, FLAGS.train_shards)
    
    elif FLAGS.task_type_cad == 'joints_annotation_from_cad_gt':
        cad_to_lsp_idx_2d = get_cad_2_lsp_idx(is2DJoints = True)
        cad_to_lsp_idx_3d = get_cad_2_lsp_idx(is2DJoints = False)

        # here M is hard-coded as:
        M_cad_60 =  np.array([[320.0/1.12,   .0,       320.0/2.0], 
                              [     .0,   -240.0/0.84, 240.0/2.0],
                              [.0,           .0,             1.0]])
        
        M_cad_120 =  np.array([[640.0/1.12,   .0,       640.0/2.0], 
                              [     .0,   -480.0/0.84, 480.0/2.0],
                              [.0,           .0,             1.0]])
        
        cam_cad_120 = np.array([(640.0/1.12 + 480.0/0.84)*0.5, 640.0/2.0, 480.0/2.0])
        cam_cad_60 = np.array([(320.0/1.12 + 240.0/0.84)*0.5, 320.0/2.0, 240.0/2.0])

        config = get_config()
        #print ("[***] config = {}".format(config))
        config.task_type = 'test'
        config.smpl_regressor_model_type = '7_hidden_layer_perceptron'
        config.joint_type = 'lsp'
        batch_size = 200
        #batch_size = 1
        config.batch_size = batch_size

        sess = tf.Session()
        smpl_regressor = SMPLTrainer(sess, config, data_loader = None)
        #joint_num = smpl_regressor.joint_num
        # Initializing the variables
        #init_op = tf.group(tf.global_variables_initializer(), 
        #                    tf.local_variables_initializer())
        #sess.run(init_op)
        
        save_to_h5file = FLAGS.is_save_to_h5files
        mpjpe_pa_thred = 90.0

        if save_to_h5file:
            fwrite = open('./results/log_files_important/cad60-120-mpjpepa-%.1f-h5.txt' % (mpjpe_pa_thred), 'a')
        else:
            fwrite = open('./results/log_files_important/cad60-120-mpjpepa-%.1f-tfrecord.txt' % (mpjpe_pa_thred), 'a')
        
        #for subject in cad_small_subjects:
        for subject in cad_eval_subjects:
        #for subject in cad_subjects:
            gender = cad_subjects_dict[subject]['gender']
            if gender == 'f':
                gender_vect = np.array([1.0, .0, .0]) # female
            elif gender == 'm':
                gender_vect = np.array([.0, 1.0, .0]) # male
            else:
                gender_vect = np.array([.0, .0, 1.0]) # nuetral
            #(3,) --> (1, 3)
            gender_vect = np.expand_dims(gender_vect, axis = 0) # batched
            
            img_dir = join(FLAGS.img_directory, subject)
            out_dir = join(FLAGS.output_directory, subject)
            if not exists(out_dir):
                makedirs(out_dir)
                print ("mkdir %s" % out_dir)
    
            # 3 x 15*N ==> 3 x 15 x N;
            if 'cad-60' in subject:
                M = M_cad_60
                cam = cam_cad_60
                print ('[**] using M from cad-60')
            elif 'cad-120' in subject:
                M = M_cad_120
                cam = cam_cad_120
                print ('[**] using M from cad-120')
            
            if is_train:
                out_path = join(out_dir, 'train_%03d.tfrecord')
            else:
                out_path = join(out_dir, 'test_%03d.tfrecord')
                print('Not implemented for test data')
                exit(1)
            
            print ("[***] img_dir = {}".format(img_dir))
            
            if 'cad-60' in subject:
                all_anno_txt_files = [ f for f in glob.glob(join(img_dir, "*.txt")) if os.path.isdir( f[0:-4])]
            elif 'cad-120' in subject:
                all_anno_txt_files = [ f for f in glob.glob(join(img_dir, "*/*.txt")) ]
                
            #print ('[***] anno_fnames {}'.format(all_anno_txt_files))
            all_lsp_joints2d = []
            all_img_fnames = []
            img_num_sum = 0 # all image included all the '.txt' files in the current subject;
            all_pred_pose = []
            all_pred_shape = []
            all_joints3d = [] # gt input joints3d with preprocession;
            all_joints3d_pred = [] # regressed joints3d via smpl layer
            for anno_fname in all_anno_txt_files:
                #print ('[***] anno_fname {}'.format(anno_fname))
                # joints3d : 3 x joints_Num (i.e., 15 ) x frames_num
                joints3d, img_fnames = read_cad_txt_annotation(anno_fname)
                all_img_fnames += img_fnames
                img_num = len(img_fnames) # all image included in this current '.txt' file;
                img_num_sum += img_num
                cad_joints2d = np.reshape(getPixelValuesFromCoords(
                    np.reshape(joints3d, [3, 15*img_num]), M), [2, 15, img_num])
                lsp_joints2d = np.ones([3, 14, img_num], dtype = np.float32) # (x,y,vis=1)
                lsp_joints2d[:2,:, :] = cad_joints2d[:, cad_to_lsp_idx_2d, :]
                all_lsp_joints2d.append(lsp_joints2d)

                # for smpl annotation, using the smpl regressor learned offline
                # 3 x joints_Num (i.e., 15 ) x frames_num
                # values are in milimeters, so change it to meters;
                joints3d *= 0.001
                torso_idx_cad = 2
                torso = joints3d[:,torso_idx_cad:torso_idx_cad+1, :] # 3 x 1 x N
                # get lsp 14 joints3d:
                joints3d = joints3d[:, cad_to_lsp_idx_3d, :] # 3 x 14 x N 
                #print ('[**] joints3d before root extracted :\n{}'.format(joints3d))
                joints3d = joints3d - torso # torso subtracted;
                #print ('[**] joints3d after root extracted :\n{}'.format(joints3d))
                joints3d = np.transpose(joints3d, (2, 1, 0)) # N x 14 x 3
                
                # swap the left/right joints3d
                print ("[***] doing reflect_joints3d(..) to joints3d")
                swap_inds = np.array([5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13])
                joints3d = joints3d[:,swap_inds,:] # N x 14 x 3
                # let joints3d (x,y,z) -> (-x, y, z)
                joints3d[:,:,0] *= -1.0
                # Assumes all joints3d are mean subtracted
                joints3d = joints3d - np.mean(joints3d, axis=1, keepdims=True)

                joints3d = np.reshape(joints3d, [img_num, 14*3]) # N x (14*3)
                num_iter_batch = int(np.ceil(img_num / batch_size)) 
                
                last_batch_num = img_num % batch_size
                gender_vect_tiled = np.tile(gender_vect, [batch_size, 1])
                print ("num_iter_batch = {}, last_batch_num = {}".format(num_iter_batch, last_batch_num))

                if num_iter_batch == 1:
                    if last_batch_num == 0:
                        idx_begin = 0
                        idx_end = batch_size
                        tmp_pose, tmp_shape, tmp_joints3d_pred = smpl_regressor.pred_smpl_param(
                            joints3d[idx_begin:idx_end, :], gender_vect_tiled)
                        all_pred_pose.append(tmp_pose)
                        all_pred_shape.append(tmp_shape)
                        all_joints3d_pred.append(tmp_joints3d_pred)# N x 14 x 3
                    else:
                        idx_begin = 0
                        idx_end = last_batch_num
                        itr = 0
                        joints3d_last_batch = np.ones([batch_size, 14*3]).astype(np.float32)
                        joints3d_last_batch[idx_begin:idx_end, :] = joints3d[itr*batch_size:min((itr+1)*batch_size, img_num), :]
                        tmp_pose, tmp_shape, tmp_joints3d_pred = smpl_regressor.pred_smpl_param(
                            joints3d_last_batch, gender_vect_tiled)
                        all_pred_pose.append(tmp_pose[idx_begin:idx_end,:])
                        all_pred_shape.append(tmp_shape[idx_begin:idx_end,:])
                        all_joints3d_pred.append(tmp_joints3d_pred[idx_begin:idx_end,:, :]) # N x 14 x 3
                
                elif num_iter_batch > 1:
                    if last_batch_num == 0:
                        iter_end = num_iter_batch
                    else:
                        iter_end = num_iter_batch - 1
                    for itr in range(0, iter_end):
                        idx_begin = itr*batch_size
                        idx_end = (itr + 1)*batch_size
                        tmp_pose, tmp_shape, tmp_joints3d_pred = smpl_regressor.pred_smpl_param(
                            joints3d[idx_begin:idx_end, :], gender_vect_tiled)
                        all_pred_pose.append(tmp_pose)
                        all_pred_shape.append(tmp_shape)
                        all_joints3d_pred.append(tmp_joints3d_pred)# N x 14 x 3
                    # the last batch
                    # padding the last few samples (e.g., 3) to 1 batch_size (e.g., 6)
                    if last_batch_num > 0:
                        itr = num_iter_batch - 1
                        idx_begin = 0
                        idx_end = last_batch_num
                        print (idx_begin, idx_end)
                        joints3d_last_batch = np.ones([batch_size, 14*3]).astype(np.float32)
                        joints3d_last_batch[idx_begin:idx_end, :] = joints3d[itr*batch_size:min((itr+1)*batch_size, img_num), :]
                        tmp_pose, tmp_shape, tmp_joints3d_pred = smpl_regressor.pred_smpl_param(
                            joints3d_last_batch, gender_vect_tiled)
                        all_pred_pose.append(tmp_pose[idx_begin:idx_end,:])
                        all_pred_shape.append(tmp_shape[idx_begin:idx_end,:])
                        all_joints3d_pred.append(tmp_joints3d_pred[idx_begin:idx_end,:, :]) # N x 14 x 3

                # joints3d 
                all_joints3d.append(joints3d)



            all_lsp_joints2d = np.concatenate(all_lsp_joints2d, axis = 2) # 3 x 14 x N
            all_joints3d = np.concatenate(all_joints3d, axis = 0) # N x (14*3)
            all_pred_pose = np.concatenate(all_pred_pose, axis = 0) # N x (23*3) 
            all_pred_shape = np.concatenate(all_pred_shape, axis = 0) # N x 10
            all_joints3d_pred = np.concatenate(all_joints3d_pred, axis = 0) # N x 14 x 3
            assert (img_num_sum == all_lsp_joints2d.shape[2])
            assert (img_num_sum == all_joints3d.shape[0])
            assert (img_num_sum == all_joints3d_pred.shape[0])
            assert (img_num_sum == all_pred_pose.shape[0])
            assert (img_num_sum == all_pred_shape.shape[0])

            print ("[***] processing %s" % subject)
            print ('[***] all_img_fnames {}'.format(all_img_fnames[-3:-1]))
            
            process_cad_gt(
                img_num_sum, all_lsp_joints2d, all_img_fnames, 
                FLAGS.img_directory, out_path, out_dir, FLAGS.train_shards,
                all_joints3d, # N x (14*3)
                all_joints3d_pred, # N x 14 x 3
                all_pred_pose, # N x (23*3)
                all_pred_shape, # N x 10
                gender_vect[0,:], # (3,)
                cam, # (3, )
                fwrite,
                save_to_h5file,
                mpjpe_pa_thred)

    else:
        print("ERROR!!! FLAGS.task_type_cad is %s. It should be 'joints_annotation_from_cad_gt' or 'joints_annotation_from_densepose'"
                    % FLAGS.task_type_cad)


if __name__ == '__main__':
    
    tf.app.flags.DEFINE_string('img_directory', '/datasets/cad-60/','image data directory')
    tf.app.flags.DEFINE_string('output_directory', '/datasets/tf_datasets/cad-60/', 'Output data directory')
    tf.app.flags.DEFINE_integer('train_shards', 500, 'Number of shards in training TFRecord files.')
    tf.app.flags.DEFINE_integer('validation_shards', 500, 'Number of shards in validation TFRecord files.')
    tf.app.flags.DEFINE_string('task_type_cad', 'joints_annotation_from_densepose', 'task type, could be joints_annotation_from_densepose or joints_annotation_from_cad_gt')
    """
    # save to h5 file or save to tfrecord files:
    * Usually, for network training: we chose to write tfrecord files for tfrecord data loading;
    * for evaulation and testing, we use h5 file for numpy data loading, 
    * which will be further fed to tf.placeholder for evluation and or testing;
    """
    tf.app.flags.DEFINE_boolean( 'is_save_to_h5files', False, 'if set, save to h5 files for data loading')
    
    FLAGS = tf.app.flags.FLAGS
    
    tf.app.run()