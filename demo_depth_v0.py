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

import src.config
from src.RunModelDepth import RunModelV2
#added by ccj;
import src.pfmutil as pfm

flags.DEFINE_string('depth_fname', 'data/im1963.jpg', 'depth image to run')
flags.DEFINE_string('info_fname', 'data/im1963.jpg', 'info file run')
flags.DEFINE_string('result_dir', 'data/im1963.jpg', 'results dir to save files')
flags.DEFINE_integer('t_beg', 0, 'frame begin idx')
flags.DEFINE_integer('t_end', 1, 'frame end idx')
flags.DEFINE_string('json_path', None, 'If specified, uses the openpose output to crop the image.')


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
    return cam_for_render


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
            print('Resizing so the max image size is %d..' % config.img_size)
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
    
    return crop_img, crop_depth, proc_param, img_orig, joints2d_gt_scaled, cam_gt_scaled

def main(args, json_path=None):
    sess = tf.Session()
    model = RunModelV2(config, sess=sess)
    
    all_images, all_depths, all_gt_poses, all_gt_shapes, \
            all_gt_joints2d, all_gt_joints3d, all_gt_smpl_verts, \
            all_gt_smpl_joints3d, all_gt_smpl_joints3d_proj, gt_cam = load( args,
            isSave = False, isNormalized = False)
    
    for i in range(0,len(all_images)):
        img = all_images[i]
        dep = all_depths[i]
        gt_pose = all_gt_poses[i]
        gt_shape = all_gt_shapes[i]
        gt_joints2d = all_gt_joints2d[i]
        gt_joints3d = all_gt_joints3d[i]
        

        input_img, input_dep, proc_param, img_orig, gt_joints2d, gt_cam = preprocess_image(img, dep, json_path, gt_joints2d, gt_cam)
        # Add batch dimension: 1 x D x D x 3
        input_img = np.expand_dims(input_img, 0)
        input_dep = np.expand_dims(input_dep, 0)
        
        gt_pose = np.expand_dims(gt_pose, 0)
        gt_shape = np.expand_dims(gt_shape, 0)
        gt_cam = np.expand_dims(gt_cam, 0)
        print ("[**??**] input_img shape = {}, input_dep shape = {}".format(input_img.shape, input_dep.shape))

        # Theta is the 85D vector holding [camera, pose, shape]
        # where camera is 3D [s, tx, ty]
        # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
        # shape is 10D shape coefficients of SMPL
        joints, verts,  cams, joints3d, joints_gt, verts_gt, joints3d_gt, theta = model.predict(
                        input_img, input_dep, gt_pose, gt_shape, gt_cam, get_theta = True)
        print ("[***] prediction done !")
        

        if not exists(args.result_dir):
            os.makedirs(args.result_dir)
            print ("makedirs %s" % args.result_dir)
        tmp_name = args.info_fname.split('/')[-1][:-9] + "frame_%03d" % i
        
        #*********************
        """ 1)predicted """
        #*********************
        save_fig_result = join(args.result_dir, tmp_name + '_fig_predict.png')
        cam_for_render = visualize(img_orig, proc_param, joints[0], verts[0], cams[0], save_fig_result)
        
        #*********************
        """ 2) GT via tf.smpl() model """
        #*********************
        save_fig_gt = join(args.result_dir, tmp_name + '_fig_gt.png')
        # joints_gt shape : [batch_size, 14, 2];
        # joints3d_gt shape : [batch_size, 14, 3];
        # verts_gt shape: [batch_size, 6890, 3];
        cam_for_render_gt = visualize(img_orig, proc_param, joints_gt[0], verts_gt, gt_cam[0], save_fig_gt)
        #_ = visualize(img_orig, proc_param, joints_gt[0], verts_gt, cams[0], save_fig_gt)
        print ('shape: joints_gt = {}, joints3d_gt = {}, verts_gt = {}'.format(
                joints_gt.shape, joints3d_gt.shape, verts_gt.shape))
        
        
        #*********************
        """ 3) GT via the regular python smpl model according to `smpl_relations.py` in SURREAL dataset; """
        #*********************
        gt_smpl_surreal_verts = all_gt_smpl_verts[i] # shape [6890, 3]
        gt_smpl_surreal_joints3d = all_gt_smpl_joints3d[i] # in shape [14, 3];
        print ('shape: gt_joints2d = {}, gt_joints3d = {}, gt_smpl_surreal_verts = {}, \
                gt_smpl_surreal_joints3d = {}'.format(
            gt_joints2d.shape, gt_joints3d.shape, gt_smpl_surreal_verts.shape, 
            gt_smpl_surreal_joints3d.shape))
        save_fig_gt_surreal = join(args.result_dir, tmp_name + '_fig_gt_surreal.png')
        # gt_joints3d: in shape [3, 14];
        # gt_joints2d: in shape [2, 14], so we use ".T" to make it be [14, 2], required by visualize() func;
        visualize(img_orig, proc_param, gt_joints2d.T, gt_smpl_surreal_verts, gt_cam[0], 
                save_fig_gt_surreal)
        #visualize(img_orig, proc_param, gt_joints2d.T, gt_smpl_surreal_verts, cams[0], save_fig_gt_surreal)
       
        """ save parameters to json files """
        json_fname =  join(args.result_dir, tmp_name + '_params.json')  
        dict_to_save = {}
        dict_to_save['fname'] = tmp_name
        dict_to_save['joints2d_pd_tfSMPL'] = joints[0] # [0] is along batch_size dimension, here we just set batch_size = 1;
        dict_to_save['joints3d_pd_tfSMPL'] = joints3d[0] # [0] is along batch_size dimension, here we just set batch_size = 1;
        dict_to_save['cam_pd_orig'] = cam_for_render
        dict_to_save['cam_pd'] = cams[0]
        
        dict_to_save['joints2d_gt_tfSMPL'] = joints_gt[0] # [0] is along batch_size dimension, here we just set batch_size = 1;
        dict_to_save['joints3d_gt_tfSMPL'] = joints3d_gt[0] # [0] is along batch_size dimension, here we just set batch_size = 1;
        dict_to_save['cam_gt_orig'] = cam_for_render_gt
        dict_to_save['cam_gt'] = gt_cam[0]

        dict_to_save['joints2d_gt_surreal'] = gt_joints2d.T # in shape [14, 2] 
        dict_to_save['joints3d_gt_surreal'] = gt_joints3d.T # in shape [14, 3]
        dict_to_save['joints2d_gt_pySMPL_surreal'] = all_gt_smpl_joints3d_proj[i] # in shape [14, 2]
        dict_to_save['joints3d_gt_pySMPL_surreal'] = gt_smpl_surreal_joints3d # in shape [14, 3]
        print ("dict_to_save['joints3d_gt_pySMPL_surreal'] = {}".format(
            dict_to_save['joints3d_gt_pySMPL_surreal']))

        save_to_json(dict_to_save, json_fname)
        #joints3d_gt_diff = np.sum(joints3d_gt - gt_smpl_surreal_joints3d, )



def save_to_json(dict_to_save, json_fname):
    import json
    new_dict = {}
    for k in dict_to_save:
        new_dict[k] = dict_to_save[k].tolist() if k is not "fname" else dict_to_save[k]
    with open(json_fname, 'w') as fp:
        json.dump(new_dict, fp, indent = 4, sort_keys = True)
    print ("Saved json file : %s ..." % json_fname)



if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    #config.load_path = src.config.PRETRAINED_MODEL
    print ("[****] load_path = %s" % config.load_path)
    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main(args = config, json_path = None)
