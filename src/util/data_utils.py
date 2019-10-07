"""
Utils for data loading for training.
"""

from os.path import join
from glob import glob

import tensorflow as tf
import numpy as np
#from termcolor import colored
from random import shuffle

def parse_example_pair_proto(example_serialized):
    feature_map = {
        'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'smpl/gender': tf.FixedLenFeature((3,), dtype=tf.float32),
        'smpl/trans' : tf.FixedLenFeature((3,), dtype=tf.float32),
        'mosh/pose': tf.FixedLenFeature((72, ), dtype=tf.float32),
        'mosh/shape': tf.FixedLenFeature((10, ), dtype=tf.float32),
        'mosh/joints3d_from_smpl': tf.FixedLenFeature((24 * 3, ), dtype=tf.float32),
        }

    features = tf.parse_single_example(example_serialized, feature_map)
    fname = tf.cast(features['image/filename'], dtype=tf.string)
    gender = tf.cast(features['smpl/gender'], dtype=tf.float32)
    pose = tf.cast(features['mosh/pose'], dtype=tf.float32)
    shape = tf.cast(features['mosh/shape'], dtype=tf.float32)
    joints3d_from_smpl = tf.reshape(
        tf.cast(features['mosh/joints3d_from_smpl'], dtype=tf.float32), [24, 3])
    return fname, pose, shape, joints3d_from_smpl, gender




def parse_example_proto(example_serialized, has_3d=False, has_depth=False, depthType = 'uint16'):
    """Parses an Example proto.
    It's contents are:

        'image/height'       : _int64_feature(height),
        'image/width'        : _int64_feature(width),
        'image/x'            : _float_feature(label[0,:].astype(np.float)),
        'image/y'            : _float_feature(label[1,:].astype(np.float)),
        'image/visibility'   : _int64_feature(label[2,:].astype(np.int)),
        'image/format'       : _bytes_feature
        'image/filename'     : _bytes_feature
        'image/encoded'      : _bytes_feature
        'image/face_points'  : _float_feature,
         this is the 2D keypoints of the face points in coco 5*3 (x,y,vis) = 15

    if has_3d is on, it also has:
        'mosh/pose'          : float_feature(pose.astype(np.float)),
        'mosh/shape'         : float_feature(shape.astype(np.float)),
        # gt3d is 14x3
        'mosh/gt3d'          : float_feature(shape.astype(np.float)),
    """
    feature_map = {
        'image/encoded':
        tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/height':
        tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
        'image/width':
        tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
        'image/filename':
        tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/center':
        tf.FixedLenFeature((2, 1), dtype=tf.int64),
        'image/visibility':
        tf.FixedLenFeature((1, 14), dtype=tf.int64),
        'image/x':
        tf.FixedLenFeature((1, 14), dtype=tf.float32),
        'image/y':
        tf.FixedLenFeature((1, 14), dtype=tf.float32),
        
        'image/cam':
        tf.FixedLenFeature((1, 3), dtype=tf.float32),

        'image/face_pts':
        tf.FixedLenFeature(
            (1, 15),
            dtype=tf.float32,
            default_value=[
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
            ]),
        
    }
    if has_3d:
        print ("[**] update feature_map due to has_3d = True")
        feature_map.update({
            'smpl/gender':
            tf.FixedLenFeature((3,), dtype=tf.float32),
            'mosh/pose':
            tf.FixedLenFeature((72, ), dtype=tf.float32),
            'mosh/shape':
            tf.FixedLenFeature((10, ), dtype=tf.float32),
            'mosh/gt3d':
            tf.FixedLenFeature((14 * 3, ), dtype=tf.float32),
            # has_3d is for pose and shape: 0 for mpi_inf_3dhp, 1 for h3.6m.
            'meta/has_3d':
            tf.FixedLenFeature((1), dtype=tf.int64, default_value=[0]),
        })

    if has_depth:
        print ("[**] update feature_map due to has_depth=True")
        feature_map.update({
            'depth/raw': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'depth/size': tf.FixedLenFeature([], dtype=tf.string, default_value=''),})

    features = tf.parse_single_example(example_serialized, feature_map)

    height = tf.cast(features['image/height'], dtype=tf.int32)
    width = tf.cast(features['image/width'], dtype=tf.int32)
    center = tf.cast(features['image/center'], dtype=tf.int32)
    fname = tf.cast(features['image/filename'], dtype=tf.string)
    #fname = tf.Print(fname, [fname], message="[!!!] image name: ")

    face_pts = tf.reshape(
        tf.cast(features['image/face_pts'], dtype=tf.float32), [3, 5])
    cam = tf.reshape(
            tf.cast(features['image/cam'], dtype = tf.float32),
            [1,3])
    vis = tf.cast(features['image/visibility'], dtype=tf.float32)
    x = tf.cast(features['image/x'], dtype=tf.float32)
    y = tf.cast(features['image/y'], dtype=tf.float32)

    label = tf.concat([x, y, vis], 0)
    label = tf.concat([label, face_pts], 1)

    image = decode_jpeg(features['image/encoded'])
    image_size = tf.concat([height, width], 0)
    
    depth = tf.expand_dims(tf.zeros(image_size, dtype=np.float32), -1)
    if has_depth:
        #depth = tf.image.decode_png(features['depth/encoded'], channels = 1, dtype=tf.uint16)
        #NOTE:
        size = tf.decode_raw(features['depth/size'], out_type = tf.int32)
        #size = tf.Print(size, [size, image_size, fname], message="[!!!] depth size, image_size,  and file name")
        # if not empty size, return tf.zeros(), since some dataset has not depth infor as input;
        #def f1(): return tf.expand_dims(tf.zeros(image_size, dtype=np.float32), -1)
        #def f2(): return tf.cast(tf.reshape(tf.decode_raw(features['depth/raw'], out_type = tf.uint16), size), tf.float32)
        out_type = tf.uint16
        if depthType == 'float32':
            out_type = tf.float32
        #NOTE: '*0.001' will change the millimeters to meters, 
        depth = tf.cond(tf.equal(tf.size(size), 0), 
                lambda: tf.expand_dims(tf.zeros(image_size, dtype=np.float32), -1),
                lambda: 0.001*tf.cast(tf.reshape(tf.decode_raw(features['depth/raw'], out_type = out_type), size), tf.float32)
                )
        #depth = tf.decode_raw(features['depth/raw'], out_type=tf.float32)
        #depth = tf.reshape(depth, size)
        #tf.print("[***] depth shape: ", depth.shape, output_stream = sys.stdout)
    if has_3d:
        gender = tf.cast(features['smpl/gender'], dtype=tf.float32)
        pose = tf.cast(features['mosh/pose'], dtype=tf.float32)
        shape = tf.cast(features['mosh/shape'], dtype=tf.float32)
        #NOTE: '*0.001' will change the millimeters to meters, 
        gt3d = 0.001*tf.reshape(tf.cast(features['mosh/gt3d'], dtype=tf.float32), [14, 3])
        has_smpl3d = tf.cast(features['meta/has_3d'], dtype=tf.bool)
        return image, depth, image_size, label, center, fname, pose, shape, gt3d, has_smpl3d, cam, gender
    else:
        return image, depth, image_size, label, center, fname, cam


def rescale_image(image):
    """
    Rescales image from [0, 1] to [-1, 1]
    Resnet v2 style preprocessing.
    """
    # convert to [0, 1].
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def get_all_files(dataset_dir, datasets, split=''):

    print ("input datasets = {}".format(datasets))
    # Dataset with different name path
    diff_name = ['cad_small','surreal_small', 'surreal_cam', 'surreal', 'cad_60_120','h36m', 'mpi_inf_3dhp']
    # dataset: cad60 and cad 120
    cad_subjects = [
        'cad-60/Person1', # 19851 imgs;
        'cad-60/Person2', # 18675, 
        'cad-60/Person3', # 19465, 
        'cad-60/Person4', # 22321,
        'cad-120/Subject1_rgbd_images', # 15227, 
        'cad-120/Subject3_rgbd_images', #17277, 
        'cad-120/Subject5_rgbd_images', #19348
        #NOTE: leave this for evaluation;
        #'cad-120/Subject4_rgbd_images', #13281, 
        ]
    surreal_subjects = [
        #'cmu/train/mix_27k', # around 26675 samples
        'cmu/train/run0', # 1605030
        'cmu/train/run1', # 2540380
        'cmu/train/run2', # 1196680

        #NOTE: leave those below for evaluation;
        #'cmu/val/run0', # 15235
        #'cmu/val/run1', # 16176
        #'cmu/val/run2', # 13080
        
        #NOTE: leave those below for evaluation;
        #'cmu/test/run0', # 362214
        #'cmu/test/run1', # 556454
        #'cmu/test/run2', # 275994
        ]

    # small dataset: cad_small and surreal_small
    cad_small_subjects = [
        'cad-60-small-1-img-person1',    
        'cad-60-small-1-img-person2',    
        'cad-120-small-1-img-sub1',
        'cad-120-small-1-img-sub3',
        'cad-120-small-1-img-sub4',
        ]
    surreal_small_subjects = [
        #'small-data/run0',# 1
        #'cmu/debug', # 1
        #'cmu/train_001_img', # 1
        'cmu/train_005_imgs', # 5
        #'cmu/train_100_imgs', # 100
        #'cmu-small/train_100_imgs', # 100
        ]

    data_dirs = [
        join(dataset_dir, dataset, '%s_*.tfrecord' % split)
        for dataset in datasets if dataset not in diff_name
    ]

    if 'h36m' in datasets:
        data_dirs.append(
            join(dataset_dir, 'tf_records_human36m_wjoints', split, '*.tfrecord'))
    if 'mpi_inf_3dhp' in datasets:
        data_dirs.append( join(dataset_dir, 'mpi_inf_3dhp', split, '*.tfrecord'))
    
    if 'cad_60_120' in datasets:
        for sub in cad_subjects:
            print ("[**] add tfrecords from %s" %join(dataset_dir, 'cad_60_120', sub, '%s_*.tfrecord' %split))
            data_dirs.append(join(dataset_dir, 'cad_60_120', sub, '%s_*.tfrecord' % split))
    
    if 'cad_small' in datasets:
        for sub in cad_small_subjects:
            tmp = join(dataset_dir, 'cad_small', sub, '%s_*.tfrecord' % split)
            data_dirs.append(tmp)
            print ("[**] add tfrecords from %s" % tmp)
    
    if 'surreal' in datasets:
        for sub in surreal_subjects:
            tmp = join(dataset_dir, 'surreal', sub, '%s_*.tfrecord' % split)
            data_dirs.append(tmp)
            print ("[***] add tfrecords from %s" %tmp)
    
    if 'surreal_27k' in datasets:
        for sub in surreal_subjects:
            tmp = join(dataset_dir, 'surreal_27k', sub, '%s_*.tfrecord' % split)
            data_dirs.append(tmp)
            print ("[***] add tfrecords from %s" %tmp)
    
    surreal_debugs = ['surreal_cam', 'surreal_small', 'surreal_smpl_joints3d_pair',
        'surreal_smpl_joints3d_pair_small_100',]
    for t in surreal_debugs:
        if  t in datasets:
            for sub in surreal_small_subjects:
                tmp = join(dataset_dir, t, sub, '%s_*.tfrecord' % split)
                data_dirs.append(tmp)
                print ("[****] add tfrecords from %s" %tmp)

    all_files = []
    print ("datasets = {}\ndata_dirs = {}".format(datasets, data_dirs))
    for data_dir in data_dirs:
        all_files += sorted(glob(data_dir))
    print ('Total tfrecords # : %d' % len(all_files))
    shuffle(all_files)
    return all_files


def read_smpl_data(filename_queue):
    """
    Parses a smpl Example proto.
    It's contents are:
        'pose'  : 72-D float
        'shape' : 10-D float
    """
    with tf.name_scope(None, 'read_smpl_data', [filename_queue]):
        reader = tf.TFRecordReader()
        _, example_serialized = reader.read(filename_queue)

        feature_map = {
            'pose': tf.FixedLenFeature((72, ), dtype=tf.float32),
            'shape': tf.FixedLenFeature((10, ), dtype=tf.float32)
        }

        features = tf.parse_single_example(example_serialized, feature_map)
        pose = tf.cast(features['pose'], dtype=tf.float32)
        shape = tf.cast(features['shape'], dtype=tf.float32)

        return pose, shape


def decode_jpeg(image_buffer, name=None):
    """Decode a JPEG string into one 3-D float image Tensor.
      Args:
        image_buffer: scalar string Tensor.
        name: Optional name for name_scope.
      Returns:
        3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope(name, 'decode_jpeg', [image_buffer]):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_jpeg(image_buffer, channels=3)

        # convert to [0, 1].
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def jitter_center(center, trans_max): # e.g., trans_max = 20
    with tf.name_scope(None, 'jitter_center', [center, trans_max]):
        rand_trans = tf.random_uniform(
            [2, 1], minval=-trans_max, maxval=trans_max, dtype=tf.int32)
        return center + rand_trans

def jitter_scale(image, image_size, keypoints, center, scale_range):
    with tf.name_scope(None, 'jitter_scale', [image, image_size, keypoints]):
        scale_factor = tf.random_uniform(
            [1],
            minval=scale_range[0], # e.g, scale_min = 0.8;
            maxval=scale_range[1], # e.g, scale_max = 1.23;
            dtype=tf.float32)
        new_size = tf.to_int32(tf.to_float(image_size) * scale_factor)
        new_image = tf.image.resize_images(image, new_size)

        # This is [height, width] -> [y, x] -> [col, row]
        actual_factor = tf.to_float(
            tf.shape(new_image)[:2]) / tf.to_float(image_size)
        x = keypoints[0, :] * actual_factor[1]
        y = keypoints[1, :] * actual_factor[0]

        cx = tf.cast(center[0], actual_factor.dtype) * actual_factor[1]
        cy = tf.cast(center[1], actual_factor.dtype) * actual_factor[0]

        #NOTE:
        #cam = tf.reshape(cam, [-1])
        #f_scale =  cam[0] * scale_factor
        #tx_scale = cam[1] * actual_factor[1]
        #ty_scale = cam[2] * actual_factor[0]

        return new_image, tf.stack([x, y]), tf.cast(
            tf.stack([cx, cy]), tf.int32)


def pad_image_edge(image, margin, channels = 3):
    """ Pads image in each dimension by margin, in numpy:
    image_pad = np.pad(image,
                       ((margin, margin),
                        (margin, margin), (0, 0)), mode='edge')
    tf doesn't have edge repeat mode,, so doing it with tile
    Assumes image has 3 channels!!
    """

    def repeat_col(col, num_repeat, channels = 3):
        # col is N x 3, ravels
        # i.e. to N*3 and repeats, then put it back to num_repeat x N x 3
        with tf.name_scope(None, 'repeat_col', [col, num_repeat]):
            return tf.reshape(
                tf.tile(tf.reshape(col, [-1]), [num_repeat]),
                [num_repeat, -1, channels])

    with tf.name_scope(None, 'pad_image_edge', [image, margin]):
        top = repeat_col(image[0, :, :], margin, channels)
        bottom = repeat_col(image[-1, :, :], margin, channels)
        image_0 = image
        #print (colored ("[***] top.shape = {}, image.shape = {}, bottom.shape = {}".format(top.shape, image.shape, bottom.shape), 'red'))
        image = tf.concat([top, image, bottom], 0)
        image_1 = image
        # Left requires another permute bc how img[:, 0, :]->(h, 3)
        left = tf.transpose(repeat_col(image[:, 0, :], margin, channels), perm=[1, 0, 2])
        right = tf.transpose(
            repeat_col(image[:, -1, :], margin, channels), perm=[1, 0, 2])
        image = tf.concat([left, image, right], 1)
        #image = tf.Print(image, 
        #        [tf.shape(image_0), tf.shape(top), tf.shape(bottom),
        #         tf.shape(image_1), tf.shape(left), tf.shape(right),
        #         tf.shape(image)],
        #        'shape of image_ori, top, bottom, image, left, right, image')

        return image


def random_flip(image, kp, pose=None, gt3d=None):
    """
    mirrors image L/R and kp, also pose if supplied
    """

    uniform_random = tf.random_uniform([], 0, 1.0)
    
    mirror_cond = tf.less(uniform_random, .5)
    #NOTE: ccj, for debugging just disable the flip;
    #mirror_cond = tf.less(uniform_random, -1.0)

    if pose is not None:
        new_image, new_kp, new_pose, new_gt3d = tf.cond(
            mirror_cond, lambda: flip_image(image, kp, pose, gt3d),
            lambda: (image, kp, pose, gt3d))
        return new_image, new_kp, new_pose, new_gt3d
    else:
        new_image, new_kp = tf.cond(mirror_cond, lambda: flip_image(image, kp),
                                    lambda: (image, kp))
        return new_image, new_kp


def flip_image(image, kp, pose=None, gt3d=None):
    """
    Flipping image and kp.
    kp is 3 x N!
    pose is 72D
    gt3d is 14 x 3
    """
    image = tf.reverse(image, [1])
    new_kp = kp

    new_x = tf.cast(tf.shape(image)[0], dtype=kp.dtype) - kp[0, :] - 1
    new_kp = tf.concat([tf.expand_dims(new_x, 0), kp[1:, :]], 0)
    # Swap left and right limbs by gathering them in the right order
    # For COCO+
    #NOTE: ??
    swap_inds = tf.constant(
        [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 16, 15, 18, 17])
    new_kp = tf.transpose(tf.gather(tf.transpose(new_kp), swap_inds))

    if pose is not None:
        new_pose = reflect_pose(pose)
        new_gt3d = reflect_joints3d(gt3d)
        return image, new_kp, new_pose, new_gt3d
    else:
        return image, new_kp



# HMR original code;
def reflect_pose(pose):
    """
    Input is a 72-Dim vector.
    Global rotation (first 3) is left alone.
    """
    with tf.name_scope("reflect_pose", [pose]):
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
        swap_inds = tf.constant([
            0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13, 14, 18,
            19, 20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33, 34, 35, 30, 31, 32,
            36, 37, 38, 42, 43, 44, 39, 40, 41, 45, 46, 47, 51, 52, 53, 48, 49,
            50, 57, 58, 59, 54, 55, 56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66,
            67, 68
        ], tf.int32)

        # sign_flip = np.tile([1, -1, -1], (24)) (with the first 3 kept)
        sign_flip = tf.constant(
            [
                1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1,
                -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1,
                -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1,
                1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1,
                -1, 1, -1, -1
            ],
            dtype=pose.dtype)

        new_pose = tf.gather(pose, swap_inds) * sign_flip

        return new_pose






def reflect_joints3d(joints):
    """
    Assumes input is 14 x 3 (the LSP skeleton subset of H3.6M)
    """
    swap_inds = tf.constant([5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13])
    with tf.name_scope("reflect_joints3d", [joints]):
        joints_ref = tf.gather(joints, swap_inds)
        flip_mat = tf.constant([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], tf.float32)
        joints_ref = tf.transpose(
            tf.matmul(flip_mat, joints_ref, transpose_b=True))
        # Assumes all joints3d are mean subtracted
        joints_ref = joints_ref - tf.reduce_mean(joints_ref, axis=0)
        return joints_ref
