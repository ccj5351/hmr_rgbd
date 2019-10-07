"""
Data loader with data augmentation.
Only used for training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
from glob import glob

import tensorflow as tf

from .tf_smpl.batch_lbs import batch_rodrigues
from .util import data_utils

# added by CCJ:
from termcolor import colored
from src.util import image as img_util
import sys

_3D_DATASETS = ['surreal_27k','cad_small', 'cad_60_120',
                'surreal_small', 'surreal_cam', 'surreal', 
                'h36m', 'up', 'mpi_inf_3dhp']


def num_examples(datasets):
    _NUM_TRAIN = {
        'lsp': 1000,
        'lsp_ext': 10000,
        'mpii': 20000,
        'h36m': 312188,
        'coco': 79344,
        #'cad_60_120': 145445, # cad60/120: Totoal = 145445 images;
        #'cad_60_120': 145445 - 13281, # without cad-120/Subject4_rgbd_images;
        'cad_60_120': 27029 - 354, # filtered by MPJPE-PA-thred= 70.0mm, without cad-120/Subject4_rgbd_images;
        'cad_small': 5, # including 5 images from cad-60 and cad-120; 
        'mpi_inf_3dhp': 147221,  # without S8
        # Below is number for MOSH/mocap:
        'H3.6': 1559985,  # without S9 and S11,
        'CMU': 3934267,
        'jointLim': 181968,
        #'surreal_train':  1605030 + 2540380 + 1196680,
        'surreal':  5342090, # i.e., = 1605030 +  2540380 + 1196680,
        'surreal_27k': 26675, # around 27k images saved in cmu/train/mix_27k dir;
        #'surreal_cam':  5342090, # i.e., = 1605030 +  2540380 + 1196680
        'surreal_100':  100, # i.e., = 1605030 +  2540380 + 1196680
        'surreal_small': 5, # i.e., = 1605030 +  2540380 + 1196680
        'surreal_smpl_joints3d_pair':  5342090, # i.e., = 1605030 +  2540380 + 1196680,
        'surreal_smpl_joints3d_pair_small_100': 100, # i.e., = 1605030 +  2540380 + 1196680,
    }

    if not isinstance(datasets, list):
        datasets = [datasets]
    total = 0

    use_dict = _NUM_TRAIN

    for d in datasets:
        total += use_dict[d]
    return total



class DataLoader(object):
    def __init__(self, config, light_weight = False):
        self.config = config
        if not light_weight:
            self.use_3d_label = config.use_3d_label
            # has depth as input
            self.has_depth = config.has_depth
            print ("[**] has_depth = {}".format(self.has_depth))
            print ("[**] use_3d_label = {}".format(self.use_3d_label))

            self.dataset_dir = config.data_dir
            self.datasets = config.datasets
            self.mocap_datasets = config.mocap_datasets
            self.batch_size = config.batch_size
            self.data_format = config.data_format
            self.output_size = config.img_size # e.g., = 224
            # Jitter params:
            self.trans_max = config.trans_max
            self.scale_range = [config.scale_min, config.scale_max]

            self.split = config.split
            self.image_normalizing_fn = data_utils.rescale_image
        else:
            self.dataset_dir = config.data_dir
            self.datasets = config.datasets
            self.batch_size = config.batch_size
            self.split = config.split
    
    
    def loader_smpl_joints3d_pair(self):
        """
        Outputs:
          smpl_batch: batched smpl parameters
          joints3d_batch: batched keypoint 3d labels (x,y,z);
        """
        print ("[**] get_loader_smpl_joints3d_pair(), datasets = {}".format(self.datasets))
        files = data_utils.get_all_files(self.dataset_dir, self.datasets, self.split)

        do_shuffle = True
        filename_queue = tf.train.string_input_producer(
            files, shuffle=do_shuffle, name="pair_data_input")

        with tf.name_scope(None, 'read_data', [filename_queue]):
            reader = tf.TFRecordReader()
            _, example_serialized = reader.read(filename_queue)

            fname, pose, shape, \
            joints3d_from_smpl, gender = data_utils.parse_example_pair_proto(example_serialized)
            

        min_after_dequeue = 5000
        num_threads = 8
        capacity = min_after_dequeue + 3 * self.batch_size

        pack_these = [pose,  shape,   joints3d_from_smpl, gender, fname]
        pack_name = ['pose', 'shape', 'joints3d_from_smpl', 'gender', 'img_name']

        all_batched = tf.train.shuffle_batch(
            pack_these,
            batch_size=self.batch_size,
            num_threads=num_threads,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            enqueue_many=False,
            name='pair_data_input_batch_train')
        batch_dict = {}
        for name, batch in zip(pack_name, all_batched):
            batch_dict[name] = batch

        return batch_dict



    def load(self):
        if self.use_3d_label:
            image_loader = self.get_loader_w3d()
        else:
            image_loader = self.get_loader()

        return image_loader

    
    def get_loader(self):
        """
        Outputs:
          image_batch: batched images as per data_format
          label_batch: batched keypoint labels N x K x 3
        """
        print ("[**] get_loader(), datasets = {}".format(self.datasets))
        files = data_utils.get_all_files(self.dataset_dir, self.datasets, self.split)

        do_shuffle = True
        fqueue = tf.train.string_input_producer(
            files, shuffle=do_shuffle, name="input")
        #image, label = self.read_data(fqueue, has_3d=False)
        image, depth, depth_max, label, fname, cam = self.read_data(fqueue, has_3d=False)
        min_after_dequeue = 5000
        num_threads = 8
        #num_threads = 16
        capacity = min_after_dequeue + 3 * self.batch_size

        #pack_these = [image, label]
        #pack_name = ['image', 'label']
        pack_these = [image, depth, depth_max, label, fname, cam]
        pack_name = ['image', 'depth', 'depth_max', 'label', 'fname', 'cam']

        all_batched = tf.train.shuffle_batch(
            pack_these,
            batch_size=self.batch_size,
            num_threads=num_threads,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            enqueue_many=False,
            name='input_batch_train')
        batch_dict = {}
        for name, batch in zip(pack_name, all_batched):
            batch_dict[name] = batch

        return batch_dict
    
    # get loader with 3d joints grouhd truth;
    def get_loader_w3d(self):
        """
        Similar to get_loader, but outputs are:
          image_batch: batched images as per data_format
          label_batch: batched keypoint labels N x K x 3
          label3d_batch: batched keypoint labels N x (216 + 10 + 42)
                         216=24*3*3 pose, 10 shape, 42=14*3 3D joints
                         (3D datasets only have 14 joints annotated)
          has_gt3d_batch: batched indicator for
                          existence of [3D joints, 3D SMPL] labels N x 2 - bool
                          Note 3D SMPL is only available for H3.6M.


        Problem is that those datasets without pose/shape do not have them
        in the tfrecords. There's no way to check for this in TF,
        so, instead make 2 string_input_producers, one for data without 3d
        and other for data with 3d.
        And send [2 x *] to train.*batch
        """

        #print ("[**] get_loader_w3(), datasets = {}".format(self.datasets))
        #files = data_utils.get_all_files(self.dataset_dir, self.datasets, self.split)
        datasets_no3d = [d for d in self.datasets if d not in _3D_DATASETS]
        datasets_yes3d = [d for d in self.datasets if d in _3D_DATASETS]
         
        print ("[**] get_loader_w3() for datasets_no3d, datasets = {}".format(datasets_no3d))
        files_no3d = data_utils.get_all_files(self.dataset_dir, datasets_no3d, self.split)
        print ("[**] get_loader_w3() for datasets_yes3d, datasets = {}".format(datasets_yes3d))
        files_yes3d = data_utils.get_all_files(self.dataset_dir, datasets_yes3d, self.split)

        # Make sure we have dataset with 3D.
        if len(files_yes3d) == 0:
            print("Dont run this without any datasets with gt 3d")
            import ipdb; ipdb.set_trace()
            exit(1)
        
        do_shuffle = True

        fqueue_yes3d = tf.train.string_input_producer(
            files_yes3d, shuffle=do_shuffle, name="input_w3d")
        image, depth, depth_max, label, label3d, has_smpl3d, fname, cam = self.read_data(
            fqueue_yes3d, has_3d=True)
        #NOTE: debug
        #label = tf.Print(label, [tf.shape(label), tf.shape(image)], "[????] label shape, image shape")
        
        if len(files_no3d) != 0:
            fqueue_no3d = tf.train.string_input_producer(
                files_no3d, shuffle=do_shuffle, name="input_wout3d")
            image_no3d, depth_no3d, depth_max_no3d, label_no3d, cam_no3d = self.read_data(fqueue_no3d, has_3d=False)
            label3d_no3d = tf.zeros_like(label3d)
            image = tf.parallel_stack([image, image_no3d])
            depth = tf.parallel_stack([depth, depth_no3d])
            depth_max = tf.parallel_stack([depth_max, depth_max_no3d])
            label   = tf.parallel_stack([label, label_no3d])
            label3d = tf.parallel_stack([label3d, label3d_no3d])
            cam = tf.parallel_stack([cam, cam_no3d])
            # 3D joint is always available for data with 3d.
            has_3d_joints = tf.constant([True, False], dtype=tf.bool)
            has_3d_smpl = tf.concat([has_smpl3d, [False]], axis=0)
        else:
            # If no "no3d" images, need to make them 1 x *
            image = tf.expand_dims(image, 0)
            depth = tf.expand_dims(depth, 0)
            depth_max = tf.expand_dims(depth_max, 0)
            label = tf.expand_dims(label, 0)
            cam = tf.expand_dims(cam, 0)
            label3d = tf.expand_dims(label3d, 0)
            has_3d_joints = tf.constant([True], dtype=tf.bool)
            has_3d_smpl = has_smpl3d

        # Combine 3D bools.
        # each is 2 x 1, column is [3d_joints, 3d_smpl]
        has_3dgt = tf.stack([has_3d_joints, has_3d_smpl], axis=1)

        min_after_dequeue = 2000
        capacity = min_after_dequeue + 3 * self.batch_size

        image_batch, depth_batch, depth_max_batch, label_batch, label3d_batch, \
        bool_batch, cam_batch = tf.train.shuffle_batch(
            [image, depth, depth_max, label, label3d, has_3dgt, cam],
            batch_size=self.batch_size,
            num_threads=8, # original value in hmr code;
            #num_threads=16,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            enqueue_many=True,
            name='input_batch_train_3d')

        if self.data_format == 'NCHW':
            image_batch = tf.transpose(image_batch, [0, 3, 1, 2])
            depth_batch = tf.transpose(depth_batch, [0, 3, 1, 2])
        elif self.data_format == 'NHWC':
            pass
        else:
            raise Exception("[!] Unkown data_format: {}".format(
                self.data_format))

        batch_dict = {
            'image': image_batch,
            'depth': depth_batch,
            'depth_max': depth_max_batch,
            'label': label_batch,
            'label3d': label3d_batch,
            'has3d': bool_batch,
            'cam' :  cam_batch,
            'fname': fname,
        }

        return batch_dict

    def get_smpl_loader(self):
        """
        Loads dataset in form of queue, loads shape/pose of smpl.
        returns a batch of pose & shape
        """

        data_dirs = [
            join(self.dataset_dir, 'mocap_neutrMosh',
                 'neutrSMPL_%s_*.tfrecord' % dataset)
            for dataset in self.mocap_datasets
        ]
        files = []
        for data_dir in data_dirs:
            files += glob(data_dir)

        if len(files) == 0:
            print('Couldnt find any files!!')
            import ipdb
            ipdb.set_trace()

        return self.get_smpl_loader_from_files(files)

    def get_smpl_loader_from_files(self, files):
        """
        files = list of tf records.
        """
        with tf.name_scope('input_smpl_loader'):
            filename_queue = tf.train.string_input_producer(
                files, shuffle=True)

            mosh_batch_size = self.batch_size * self.config.num_stage

            min_after_dequeue = 1000
            capacity = min_after_dequeue + 3 * mosh_batch_size

            pose, shape = data_utils.read_smpl_data(filename_queue)
            pose_batch, shape_batch = tf.train.batch(
                [pose, shape],
                batch_size=mosh_batch_size,
                num_threads=4,
                capacity=capacity,
                name='input_smpl_batch')

            return pose_batch, shape_batch

    def read_data(self, filename_queue, has_3d=False):
        with tf.name_scope(None, 'read_data', [filename_queue]):
            reader = tf.TFRecordReader()
            _, example_serialized = reader.read(filename_queue)
            if has_3d:
                image, depth, image_size, label, center, fname, \
                pose, shape, gt3d, has_smpl3d, \
                cam, gender = data_utils.parse_example_proto( example_serialized, 
                            has_3d=has_3d, has_depth= self.has_depth)
                

                # Need to send pose bc image can get flipped.
                image, depth, depth_max, label, pose, gt3d = self.image_preprocessing(
                    image, image_size, label, center, pose=pose, gt3d=gt3d, depth = depth)

                # Convert pose to rotation.
                # Do not ignore the global!!
                """ 
                # comments added by CCJ:
                # pose: 1D, (72,), now reshape it to (24, 3)
                # the returned rotations is in (24, 3, 3)
                # here get rotation matrix from 3 angles of the pose;
                """
                rotations = batch_rodrigues(tf.reshape(pose, [-1, 3]))
                gt3d_flat = tf.reshape(gt3d, [-1])
                # Label 3d is:
                #   [rotations, shape-beta, 3Djoints]
                #   [216=24*3*3, 10, 42=14*3]
                
                #label3d = tf.concat([tf.reshape(rotations, [-1]), shape, gt3d_flat], 0)
                #label3d = tf.concat([tf.reshape(rotations, [-1]), shape, gt3d_flat, pose], 0)
                label3d = tf.concat([tf.reshape(rotations, [-1]), shape, gt3d_flat, pose, gender], 0)
            else:
                image, depth, image_size, label, \
                center, fname, cam = data_utils.parse_example_proto(example_serialized, has_3d = False, 
                                       has_depth = self.has_depth)
                
                image, depth, label = self.image_preprocessing(image, image_size, label, center,None,None,depth)

            # label should be K x 3
            label = tf.transpose(label)

            if has_3d:
                #return image, label, label3d, has_smpl3d
                return image, depth, depth_max, label, label3d, has_smpl3d, fname, cam

            else:
                return image, depth, depth_max, label, fname, cam


    def image_preprocessing(self,
                            image,
                            image_size,
                            label,
                            center,
                            pose=None,
                            gt3d=None,
                            depth=None,
                            ):
        margin = tf.to_int32(self.output_size / 2) 
        with tf.name_scope(None, 'image_preprocessing',
                           [image, image_size, label, center]):
            visibility = label[2, :]
            keypoints = label[:2, :]
            # added by CCJ:
            if depth is not None:
                image = tf.concat([image, depth], -1)
                print ("[***] concat ", image.shape, depth.shape) 
                channels = 4 #(rgb + d)
            else:
                channels = 3 # rgb
            # Randomly shift center.
            # NOTE: for debug
            #center = tf.Print(center, [center,], "input center : " )

            center = data_utils.jitter_center(center, self.trans_max) # e.g., trans_max = 20
            
            # randomly scale image.
            image, keypoints, center = data_utils.jitter_scale(
                image, image_size, keypoints, center, self.scale_range)

            # Pad image with safe margin.
            # Extra 50 for safety.
            margin_safe = margin + self.trans_max + 50
            #added by CCJ:
            #margin_safe = tf.Print(margin_safe, [margin, self.trans_max, margin_safe, ], "margin , self.trans_max, margin_safe, " )
            
            image_pad = data_utils.pad_image_edge(image, margin_safe, channels)
            center_pad = center + margin_safe
            keypoints_pad = keypoints + tf.to_float(margin_safe)

            start_pt = center_pad - margin

            # Crop image pad.
            start_pt = tf.squeeze(start_pt)
            bbox_begin = tf.stack([start_pt[1], start_pt[0], 0])
            #bbox_size = tf.stack([self.output_size, self.output_size, 3])
            
            """ note added by ccj: 
                always extract the crop with size of config.img_size, 
                which is predefiend by user 
            #NOTE: after the jitter_scale and jitter_center opration, the image size might change,
            # so have to extract a cropped region with size of self.output_size == config.img_size;
            """
            bbox_size = tf.stack([self.output_size, self.output_size, channels])
            
            #bbox_begin = tf.Print(bbox_begin, [start_pt, bbox_begin, bbox_size, ], "start_pt , bbox_begin, bbox_size " )

            crop = tf.slice(image_pad, bbox_begin, bbox_size)
            x_crop = keypoints_pad[0, :] - tf.to_float(start_pt[0])
            y_crop = keypoints_pad[1, :] - tf.to_float(start_pt[1])

            crop_kp = tf.stack([x_crop, y_crop, visibility])
            
            # Only for surreal dataset right now!!!
            # not used anymore; already modify the pose while tfrecord generation;
            #NOTE:
            #if pose is not None:
            #    pose = data_utils.reflect_pose_v2(pose)
            #    gt3d = data_utils.reflect_joints3d(gt3d)

            if pose is not None:
                crop, crop_kp, new_pose, new_gt3d = data_utils.random_flip(
                    crop, crop_kp, pose, gt3d)
            else:
                crop, crop_kp = data_utils.random_flip(crop, crop_kp)

            # Normalize kp output to [-1, 1]
            final_vis = tf.cast(crop_kp[2, :] > 0, tf.float32)
            final_label = tf.stack([
                2.0 * (crop_kp[0, :] / self.output_size) - 1.0,
                2.0 * (crop_kp[1, :] / self.output_size) - 1.0, final_vis ])
            # Preserving non_vis to be 0.
            final_label = final_vis * final_label
            
            # de-couple rgb and depth
            if depth is not None:
                crop, depth = tf.split(crop, [3,channels-3], -1)
                #NOTE: rescale the depth or not??

            # rescale image from [0, 1] to [-1, 1]
            crop = self.image_normalizing_fn(crop)
            # NOTE: rescale depth from to [-1, 1]
            depth_max = tf.reduce_max(depth)
            depth = 2.0*(depth / depth_max - 0.5)
            if pose is not None:
                return crop, depth, depth_max, final_label, new_pose, new_gt3d
            else:
                return crop, depth, depth_max, final_label
 


