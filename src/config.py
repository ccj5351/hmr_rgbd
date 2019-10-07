"""
Sets default args

Note all data format is NHWC because slim resnet wants NHWC.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import os.path as osp
from os import makedirs
from glob import glob
from datetime import datetime
import json

import numpy as np

curr_path = osp.dirname(osp.abspath(__file__))
model_dir = osp.join(curr_path, '..', 'models')
if not osp.exists(model_dir):
    print('Fix path to models/')
    import ipdb
    ipdb.set_trace()
#SMPL_MODEL_PATH = osp.join(model_dir, 'neutral_smpl_with_cocoplus_reg.pkl,')
SMPL_MODEL_PATH = osp.join(model_dir, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl,basicModel_m_lbs_10_207_0_v1.0.0.pkl,neutral_smpl_with_cocoplus_reg.pkl')
SMPL_FACE_PATH = osp.join(curr_path, '../src/tf_smpl', 'smpl_faces.npy')

# Default pred-trained model path for the demo.
PRETRAINED_MODEL = osp.join(model_dir, 'model.ckpt-667589')

flags.DEFINE_string('smpl_model_path', SMPL_MODEL_PATH,
                    'path to the neurtral smpl model')
flags.DEFINE_string('smpl_face_path', SMPL_FACE_PATH,
                    'path to smpl mesh faces (for easy rendering)')
flags.DEFINE_string('load_path', None, 'path to trained model')
flags.DEFINE_string('pretrained_model_path', None,
                    'if not None, fine-tunes from this ckpt')
flags.DEFINE_integer('batch_size', 8,
                     'Input image size to the network after preprocessing')

# Don't change if testing:
flags.DEFINE_integer('img_size', 224,
                     'Input image size to the network after preprocessing')
flags.DEFINE_string('data_format', 'NHWC', 'Data format')
flags.DEFINE_integer('num_stage', 3, '# of times to iterate regressor')
flags.DEFINE_string('model_type', 'resnet_fc3_dropout', 'Specifies which network to use')

""" joint types: """
# * lsp = 14 joints;
# cocoplus = 14 lsp joints + 5 face points = 19 joints;
# smpl: = 24 joints;
flags.DEFINE_string(
    'joint_type', 'lsp',
    'could be cocoplus (19 keypoints) or lsp 14 keypoints, or smpl 24 joints, all of them are returned by SMPL layer')

# Training settings:
# TODO! If you want to train, change this to your 'tf_datasets' or 
# specify it with the flag.
#DATA_DIR = '/scratch1/projects/tf_datasets/'
DATA_DIR = 'datasets/tf_datasets/cad60_120/'

flags.DEFINE_string('data_dir', DATA_DIR, 'Where to save training models')
flags.DEFINE_string('log_dir', 'logs', 'Where to save training models')
flags.DEFINE_string('model_dir', None, 'Where model will be saved -- filled automatically')
flags.DEFINE_integer('log_img_step', 100, 'How often to visualize img during training')
flags.DEFINE_integer('epoch', 100, '# of epochs to train')

flags.DEFINE_list('datasets', ['lsp', 'lsp_ext', 'mpii', 'coco'],
                          'datasets to use for training')
flags.DEFINE_list('mocap_datasets', ['CMU', 'H3.6', 'jointLim'],
                  'datasets to use for adversarial prior training')

# Model config
flags.DEFINE_boolean(
    'encoder_only', False,
    'if set, no adversarial prior is trained = monsters')

# has depth as input
flags.DEFINE_boolean(
    'has_depth', True,
    'if set, rgb-d as input')

# use depth loss
flags.DEFINE_boolean(
    'has_depth_loss', True,
    'if set, use depth loss')


flags.DEFINE_boolean(
    'use_3d_label', True,
    'Uses 3D labels if on.')

# Hyper parameters:
flags.DEFINE_float('e_lr', 0.001, 'Encoder learning rate')
flags.DEFINE_float('d_lr', 0.001, 'Adversarial prior learning rate')
flags.DEFINE_float('e_wd', 0.0001, 'Encoder weight decay')
flags.DEFINE_float('d_wd', 0.0001, 'Adversarial prior weight decay')

flags.DEFINE_float('e_loss_weight', 60, 'weight on E_kp losses')
flags.DEFINE_float('d_loss_weight', 1, 'weight on discriminator')


flags.DEFINE_float('e_3d_weight', 1, 'weight on E_3d')
#NOTE:added by CCJ;
flags.DEFINE_float('e_3d_weight_js3d', 1, 'weight on E_3d_js3d_x_y_z')
flags.DEFINE_float('e_3d_weight_smpl', 1, 'weight on E_3d_smpl_beta_theta')
flags.DEFINE_float('e_weight_depth', 1, 'weight on E_depth')

# Data augmentation
flags.DEFINE_integer('trans_max', 20, 'Value to jitter translation')
flags.DEFINE_float('scale_max', 1.23, 'Max value of scale jitter')
flags.DEFINE_float('scale_min', 0.8, 'Min value of scale jitter')

# added to filter tfrecord files;
flags.DEFINE_string('split', 'train', 'train or valid to filter the tfrecord files whose names starting with this splt flag')

#*********************************
# added for pose_perceptron.py
#*********************************
flags.DEFINE_integer('save_models_step', 4000, 'frequency of model saving')
flags.DEFINE_integer('run_validation_step', 200, 'frequency of running validation')
flags.DEFINE_string('method_name', 'joints3d_to_smpl_regressor', 'some unique name to tell from others')
flags.DEFINE_string('smpl_regressor_model_type', '7_hidden_layer_perceptron', 
                    'could be 1_, 3_, or 7_hidden_layer_perceptron')

flags.DEFINE_string('result_dir', './results/tmp', 'Where results will be saved')
# Hyper parameters:
flags.DEFINE_float('lr', 0.001, 'learning rate')
flags.DEFINE_float('wd', 0.0001, 'weight decay')

flags.DEFINE_float('weight_pose', 1.0, 'weight loss of pose')
flags.DEFINE_float('weight_shape', 1.0, 'weight loss of shape')
flags.DEFINE_float('weight_joints3d', 100.0, 'weight loss of joints3d')
# Network Parameters
flags.DEFINE_integer('num_hidden_1', 256*2, '1st layer number of neurons')
flags.DEFINE_integer('num_hidden_2', 256*2, '1st layer number of neurons')
flags.DEFINE_integer('num_input', 24*3, 'data input (24 3d joints shape: 24*3)')
flags.DEFINE_integer('num_output', 23*3 + 10, 
             'output of smpl poses for 23 joints, i.e, excluding the root joint, plus the shape')

# has smpl_layer in this network or not
flags.DEFINE_boolean( 'has_smpl_layer', True, 'if set, network has smpl_layer')
flags.DEFINE_boolean( 'isPoseToRotation', True, 'if set, change pose axis-angle to rotation matrx when calculating loss')
flags.DEFINE_string(
    'task_type', 'train', "could be 'train', 'evaluation', or 'test' ")
flags.DEFINE_string(
    'toy_data_type', 'cad-120-small-1', "including: cad-120-small-1, mosh-hd5-small-1-x, cad-any-1-sample, etc")

flags.DEFINE_string(
    'data_type', 'surreal', "including: cad60, cad120, etc")

# defined for running demo_depth.py file;
flags.DEFINE_string('depth_fname', 'data/cad-60-small/dep-scale-RGB_20.pfm', 'depth image to run')
flags.DEFINE_string('image_fname', 'data/cad-60-small/img-scale-RGB_20.pfm', 'depth image to run')

flags.DEFINE_string('info_fname', 'data/im1963.jpg', 'info file run')
flags.DEFINE_string('gender', 'n', "could be 'f', 'm', or 'n' ")
flags.DEFINE_integer('t_beg', 0, 'frame begin idx')
flags.DEFINE_integer('t_end', 1, 'frame end idx')
flags.DEFINE_string('json_path', None, 'If specified, uses the openpose output to crop the image.')
flags.DEFINE_string('h5_filename', './results/eval-mpjpepa-thred-90mm-5-samples.h5', 'h5 files to load data for mpjpe evaluation')
flags.DEFINE_string('eval_model_type', 'ours', 'models used for evaluation, could be ours, hmr, etc')



def get_config():
    config = flags.FLAGS
    config(sys.argv)

    if 'resnet' in config.model_type:
        setattr(config, 'img_size', 224)
        # Slim resnet wants NHWC..
        setattr(config, 'data_format', 'NHWC')

    return config


# ----- For training ----- #
def prepare_dirs(config, prefix=['HMR']):
    # Continue training from a load_path
    if config.load_path and config.load_path is not '':
        if not osp.exists(config.load_path):
            print("load_path: %s does not exist..!!!" % config.load_path)
            import ipdb
            ipdb.set_trace()
        print('[***] model training continuing from %s!' % config.load_path)

        # Check for changed training parameter:
        # Load prev config param path
        param_path = glob(osp.join(config.load_path, '*.json'))[0]

        with open(param_path, 'r') as fp:
            prev_config = json.load(fp)
        dict_here = config.__dict__
        ignore_keys = ['load_path', 'log_img_step', 'pretrained_model_path']
        diff_keys = [
            k for k in dict_here
            if k not in ignore_keys and k in prev_config.keys()
            and prev_config[k] != dict_here[k]
        ]

        for k in diff_keys:
            if k == 'load_path' or k == 'log_img_step':
                continue
            if prev_config[k] is None and dict_here[k] is not None:
                print("%s is different!! before: None after: %g" %
                      (k, dict_here[k]))
            elif prev_config[k] is not None and dict_here[k] is None:
                print("%s is different!! before: %g after: None" %
                      (k, prev_config[k]))
            else:
                print("%s is different!! before: " % k)
                print(prev_config[k])
                print("now:")
                print(dict_here[k])

        if len(diff_keys) > 0:
            print("really continue??")
            import ipdb
            ipdb.set_trace()

        config.model_dir = config.load_path

    else:
        postfix = []

        # If config.dataset is not the same as default, add that to name.
        default_dataset = [
            'lsp', 'lsp_ext', 'mpii', 'h36m', 'coco', 'mpi_inf_3dhp'
        ]
        
        default_mocap = ['CMU', 'H3.6', 'jointLim']

        if sorted(config.datasets) != sorted(default_dataset):
            has_all_default = np.all(
                [name in config.datasets for name in default_dataset])
            if has_all_default:
                new_names = [
                    name for name in sorted(config.datasets)
                    if name not in default_dataset
                ]
                postfix.append('default+' + '-'.join(sorted(new_names)))
            else:
                postfix.append('-'.join(sorted(config.datasets)))
        if sorted(config.mocap_datasets) != sorted(default_mocap):
            postfix.append('-'.join(config.mocap_datasets))

        postfix.append(config.model_type)

        if config.num_stage != 3: # of times to iterate regressor
            prefix += ["T%d" % config.num_stage]

        postfix.append("Elr%1.e" % config.e_lr)

        if config.e_loss_weight != 1:
            postfix.append("kp-weight%g" % config.e_loss_weight)

        if not config.encoder_only:
            postfix.append("Dlr%1.e" % config.d_lr)
            if config.d_loss_weight != 1:
                postfix.append("d-weight%g" % config.d_loss_weight)

        if config.use_3d_label:
            print('Using 3D labels!!')
            prefix.append("3DSUP")
            if config.e_3d_weight != 1:
                postfix.append("3dsup-weight%g" % config.e_3d_weight)

        # Data:
        # Jitter amount:
        if config.trans_max != 20:
            postfix.append("transmax-%d" % config.trans_max)
        if config.scale_max != 1.23:
            postfix.append("scmax_%.3g" % config.scale_max)
        if config.scale_min != 0.8:
            postfix.append("scmin-%.3g" % config.scale_min)

        prefix = '_'.join(prefix)
        postfix = '_'.join(postfix)
        
        # %b: Month as locale's abbreviated name. e.g, Jan, Feb, ..., Dec;
        # %d: Day of the month as a zero-padded decimal number. e.g, 01, 02, ..., 31;
        # %H: Hour (24-hour clock) as a zero-padded decimal number. e.g, 00, 01, 02, ..., 23;
        # %M: Minute as a zero-padded decimal number. e.g, 00, 01, ..., 59;
        time_str = datetime.now().strftime("%b%d_%H%M")

        save_name = "%s_%s_%s" % (prefix, postfix, time_str)
        config.model_dir = osp.join(config.log_dir, save_name)

    for path in [config.log_dir, config.model_dir]:
        if not osp.exists(path):
            print('making %s' % path)
            makedirs(path)


def save_config(config):
    param_path = osp.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    config_dict = {}
    for k in dir(config):
        config_dict[k] = config.__getattr__(k)

    with open(param_path, 'w') as fp:
        json.dump(config_dict, fp, indent=4, sort_keys=True)
