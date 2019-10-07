""" Evaluates a trained model using placeholders. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from os.path import exists

from .tf_smpl import projection as proj_util
from .tf_smpl.batch_smpl import SMPL
from .models import get_encoder_fn_separate

class RunModelV2(object):
    def __init__(self, config, sess=None, has_smpl_gt = False):
        """
        Args:
          config
        """
        self.config = config
        self.load_path = config.load_path
        self.has_smpl_gt = has_smpl_gt
        
        # Config + path
        if not config.load_path:
            raise Exception(
                "[!] You need to specify `load_path` to load a pretrained model"
            )
        if not exists(config.load_path + '.index'):
            print('%s doesnt exist..' % config.load_path)
            import ipdb
            ipdb.set_trace()


        """ joint types: """
        
        # * lsp = 14 joints;
        # cocoplus = 14 lsp joints + 5 face points = 19 joints;
        # smpl: = 24 joints;
         
        self.joint_type = config.joint_type.lower()
        print("[***] joint_type = {}".format(self.joint_type))
        
        if self.joint_type == 'lsp':
            self.joint_num = 14
        elif self.joint_type == 'cocoplus':
            self.joint_num = 19
        elif self.joint_type == 'smpl':
            self.joint_num = 24
        else:
            print('BAD!! Unknown joint type: %s, it must be either "cocoplus" or "lsp"' % self.joint_type)
            import ipdb
            ipdb.set_trace()
        print("[***] joint_num = {}".format(self.joint_num))        

        # Data
        self.batch_size = config.batch_size
        self.img_size = config.img_size

        self.data_format = config.data_format

        """ extract female, male  and neutral model paths; """
        self.smpl_model_path = config.smpl_model_path
        self.smpl_female_model_path = config.smpl_model_path.split(',')[0]
        posit = config.smpl_model_path.rfind('/')
        self.smpl_male_model_path = config.smpl_model_path[:posit] + '/'+ config.smpl_model_path.split(',')[1]
        self.smpl_neutral_model_path = config.smpl_model_path[:posit] + '/' + config.smpl_model_path.split(',')[2]

        
        input_size = (self.batch_size, self.img_size, self.img_size, 3)
        self.images_pl = tf.placeholder(tf.float32, shape=input_size)
        #NOTE: depth
        depth_input_size = (self.batch_size, self.img_size, self.img_size, 1)
        #depth_input_size = (self.batch_size, self.img_size, self.img_size)
        self.depths_pl = tf.placeholder(tf.float32, shape= depth_input_size)


        # Model Settings
        self.num_stage = config.num_stage
        self.model_type = config.model_type
        self.joint_type = config.joint_type
        # Camera
        self.num_cam = 3
        self.proj_fn = proj_util.batch_orth_proj_idrot

        self.num_theta = 72        
        # Theta size: camera (3) + pose (24*3) + shape (10)
        self.total_params = self.num_cam + self.num_theta + 10 # 85;

        #self.smpl = SMPL(self.smpl_model_path, joint_type=self.joint_type)
        self.smpl = SMPL(
            [self.smpl_female_model_path, self.smpl_male_model_path, self.smpl_neutral_model_path],
            joint_type='lsp')
        
        #added by CCJ:
        if self.has_smpl_gt:
            self.shapes_gt_pl = tf.placeholder(tf.float32, shape = [self.batch_size, 10])
            self.poses_gt_pl = tf.placeholder( tf.float32, shape = [self.batch_size, self.num_theta])
            self.cam_gt_pl = tf.placeholder(tf.float32, shape= [self.batch_size, self.num_cam])
        
        self.genders_gt_pl = tf.placeholder( tf.float32, shape = [self.batch_size, 3])

        
        # self.theta0_pl = tf.placeholder_with_default(
        #     self.load_mean_param(), shape=[self.batch_size, self.total_params], name='theta0')
        # self.theta0_pl = tf.placeholder(tf.float32, shape=[None, self.total_params], name='theta0')

        self.build_test_model_ief()

        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess
        
        # Load data.
        self.saver = tf.train.Saver()
        self.prepare()        

    
    """ Main IEF loop (Iterative Error Feedback) """ 
    def build_test_model_ief(self):
        # Load mean value
        self.mean_var = tf.Variable(tf.zeros((1, self.total_params)), name="mean_param", dtype=tf.float32)

        img_enc_fn, threed_enc_fn = get_encoder_fn_separate(self.model_type, "Encoder_resnet_v2")
        # Extract image features.        
        self.img_dep_feat, self.E_var = img_enc_fn(
                self.images_pl, 
                self.depths_pl,
                is_training=False,
                reuse=False)
        
        # Start loop
        self.all_verts = []
        self.all_kps = []
        self.all_cams = []
        self.all_Js = []
        self.all_J_transformed = [] # added by CCJ;
        self.final_thetas = []
        
        # smpl ground truth related variables
        if self.has_smpl_gt:
            self.all_verts_gt = [] # added by CCJ;
            self.all_kps_gt = [] # added by CCJ;
            self.all_Js_gt = [] # added by CCJ;
            self.all_J_transformed_gt = [] # added by CCJ;

        theta_prev = tf.tile(self.mean_var, [self.batch_size, 1])
        
        # added by CCJ:
        # feeding the gt pose and shape;
        #print ("[***] call tf.smpl 1 ...") 
        #verts_gt, Js_gt, _ = self.smpl(
        #        tf.zeros(self.shapes_gt_pl.shape), 
        #        tf.zeros(self.poses_gt_pl.shape), get_skin = True, trans= None, idx = 0)
        

        #print ("[***] call tf.smpl 2 ...") 
        if self.has_smpl_gt:
            verts_gt, Js_gt, _, J_transformed = self.smpl(self.shapes_gt_pl, self.poses_gt_pl, 
                                     get_skin = True, gender = self.genders_gt_pl)
            self.all_verts_gt.append(verts_gt)
            self.all_Js_gt.append(Js_gt)
            """  N x 24 x 3 joint location after shaping & posing with beta and theta;"""
            self.all_J_transformed_gt.append(J_transformed)
            gt_kp = self.proj_fn(Js_gt, self.cam_gt_pl, name='proj_2d_gt')
            self.all_kps_gt.append(gt_kp)
        
        for i in np.arange(self.num_stage):
            print('Iteration %d' % i)
            # ---- Compute outputs
            state = tf.concat([self.img_dep_feat, theta_prev], 1)

            if i == 0:
                delta_theta, _ = threed_enc_fn(
                    state,
                    num_output=self.total_params,
                    is_training=False,
                    reuse=False)
            else:
                delta_theta, _ = threed_enc_fn(
                    state,
                    num_output=self.total_params,
                    is_training=False,
                    reuse=True)

            # Compute new theta
            theta_here = theta_prev + delta_theta
            # cam = N x 3, pose N x self.num_theta, shape: N x 10
            cams = theta_here[:, :self.num_cam]                
            poses = theta_here[:, self.num_cam:(self.num_cam + self.num_theta)]
            shapes = theta_here[:, (self.num_cam + self.num_theta):]

            verts, Js, _, J_transformed = self.smpl(shapes, poses, get_skin=True, gender = self.genders_gt_pl)
            """  N x 24 x 3 joint location after shaping & posing with beta and theta;"""
            self.all_J_transformed.append(J_transformed)

            # Project to 2D!
            pred_kp = self.proj_fn(Js, cams, name='proj_2d_stage%d' % i)
            self.all_verts.append(verts)
            self.all_kps.append(pred_kp)
            self.all_cams.append(cams)
            self.all_Js.append(Js)
            # save each theta.
            self.final_thetas.append(theta_here)
            # Finally)update to end iteration.
            theta_prev = theta_here
        

    def prepare(self):
        print('Restoring checkpoint %s..' % self.load_path)
        self.saver.restore(self.sess, self.load_path)        
        self.mean_value = self.sess.run(self.mean_var)


    def predict(self, images, depths, poses_gt, shapes_gt, cam_gt, genders_gt, get_theta=False):
        """
        images: num_batch, img_size, img_size, 3
        Preprocessed to range [-1, 1]
        """
        results = self.predict_dict(images, depths, poses_gt, shapes_gt, cam_gt, genders_gt)
        if not self.has_smpl_gt:
            results['joints_gt'] = None 
            results['verts_gt'] = None
            results['joints3d_gt'] = None 
            results['joints3d_gt_24'] = None

        if get_theta:
            return results['joints'], results['verts'], results['cams'], results[
                    'joints3d'], results['joints_gt'], results['verts_gt'], results[
                    'joints3d_gt'], results['joints3d_gt_24'], results['pose'], results['shape'
                    ], results['joints3d_24']
        else:
            return results['joints'], results['verts'], results['cams'], results[
                    'joints3d'], results['joints_gt'], results['verts_gt'], results[
                    'joints3d_gt'], results['joints3d_gt_24'], results['joints3d_24']

    def predict_dict(self, images, depths, poses_gt, shapes_gt, cam_gt, genders_gt):
        """
        images: num_batch, img_size, img_size, 3
        Preprocessed to range [-1, 1]
        Runs the model with images.
        """
        if genders_gt is None:
            genders_gt = np.array([.0, .0, 1.0]).astype(np.float32) # gender vector - neutral
            genders_gt = np.expand_dims(genders_gt, 0)
        feed_dict = {
            self.images_pl: images,
            self.depths_pl: depths, # added by CCJ;
            self.genders_gt_pl : genders_gt
        }

        if self.has_smpl_gt:
            feed_dict.update({
                self.poses_gt_pl: poses_gt, # added by CCJ;
                self.shapes_gt_pl: shapes_gt, # added by CCJ;
                self.cam_gt_pl: cam_gt, # added by CCJ;
        })

        fetch_dict = {
            'joints': self.all_kps[-1],
            'verts': self.all_verts[-1],
            'cams': self.all_cams[-1],
            'joints3d': self.all_Js[-1],
            'pose': self.final_thetas[-1][:,self.num_cam:(self.num_cam + self.num_theta)],
            'shape': self.final_thetas[-1][:,(self.num_cam + self.num_theta):],
            'joints3d_24': self.all_J_transformed[-1], # pred 
        }
        if self.has_smpl_gt:
            fetch_dict.update({
            # GT related variables;
            'verts_gt': self.all_verts_gt[-1],
            'joints_gt': self.all_kps_gt[-1],
            'joints3d_gt': self.all_Js_gt[-1],
            'joints3d_gt_24': self.all_J_transformed_gt[-1],
            })

        results = self.sess.run(fetch_dict, feed_dict)

        # Return joints in original image space.
        joints = results['joints']
        results['joints'] = ((joints + 1) * 0.5) * self.img_size
        if self.has_smpl_gt:
            joints_gt = results['joints_gt']
            results['joints_gt'] = ((joints_gt + 1) * 0.5) * self.img_size

        return results
