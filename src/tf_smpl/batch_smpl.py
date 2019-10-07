""" 
Tensorflow SMPL implementation as batch.
Specify joint types:
'coco': Returns COCO+ 19 joints
'lsp': Returns H3.6M-LSP 14 joints
Note: To get original smpl joints, use self.J_transformed
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import sys
if sys.version_info[0] < 3:
        #raise Exception("Must be using Python 3")
        import cPickle as pickle
else:
    import _pickle as pickle

import tensorflow as tf
from .batch_lbs import batch_rodrigues, batch_global_rigid_transformation

from src.util.surreal_in_extrinc import get_lsp_idx_from_smpl_joints

# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r



class SMPL(object):
    #def __init__(self, pkl_path, joint_type='cocoplus', dtype=tf.float32):
    # added gender for smpl model loading;
    def __init__(self, pkl_paths = ['','',''],
                 joint_type='lsp', #'cocoplus', 'lsp'. 'smpl';
                 dtype = tf.float32):
        """
        pkl_path is the path to a SMPL model
        """
        # -- Load SMPL params -- 
        #pkl_path_first = pkl_path.split(',')[0]
        #pkl_path_second = pkl_path_first[:pkl_path_first.rfind('/')] + '/' + pkl_path.split(',')[-1]
        # in this order: [female, male, nuetral]
        self.dtype = dtype
        self.smpl_model_paths = pkl_paths
        if 0:
            print ("* female model: {}\n* male model: {}\n* neutral model: {}".format(
                self.smpl_model_paths[0], 
                self.smpl_model_paths[1], 
                self.smpl_model_paths[2])) 

        self.joint_type = joint_type
        self.dds = []
        #self.genders = ['female','male', 'nuetral']
        for mfname in self.smpl_model_paths:
            with open(mfname, 'r') as f:
                print ("[**] smpl_model loaded : %s" %( mfname))
                self.dds.append(pickle.load(f))
        
        # Mean template vertices
        # 6890 x 3
        self.v_template_f = tf.Variable( undo_chumpy(self.dds[0]['v_template']),
                      name= 'v_template_f', dtype=dtype, trainable=False)
        self.v_template_m = tf.Variable( undo_chumpy(self.dds[1]['v_template']),
                      name= 'v_template_m', dtype=dtype, trainable=False)
        self.v_template_n = tf.Variable( undo_chumpy(self.dds[2]['v_template']),
                      name= 'v_template_n', dtype=dtype, trainable=False)

        # Size of mesh [Number of vertices, 3]
        self.size = [self.v_template_f.shape[0].value, 3]
        self.num_betas = self.dds[0]['shapedirs'].shape[-1]
        # Shape blend shape basis: 6890 x 3 x 10
        # reshaped to 6890*3 x 10, transposed to 10x6890*3
        shapedirs_f = np.reshape(undo_chumpy(self.dds[0]['shapedirs']), [-1, self.num_betas]).T
        shapedirs_m = np.reshape(undo_chumpy(self.dds[1]['shapedirs']), [-1, self.num_betas]).T
        shapedirs_n = np.reshape(undo_chumpy(self.dds[2]['shapedirs']), [-1, self.num_betas]).T
        #print ("shapedir shape =  ", shapedirs_f.shape)

        # Regressor for joint locations given shape - 6890 x 24
        self.J_regressor_f = tf.Variable(self.dds[0]['J_regressor'].T.todense(), name="J_regressor_f", 
                            dtype=dtype, trainable=False)
        self.J_regressor_m = tf.Variable(self.dds[1]['J_regressor'].T.todense(), name="J_regressor_m", 
                            dtype=dtype, trainable=False)
        self.J_regressor_n = tf.Variable(self.dds[2]['J_regressor'].T.todense(), name="J_regressor_n", 
                            dtype=dtype, trainable=False)
        # 6890 x 24*3
        #self.J_regressor = tf.concat( [self.J_regressor_f, self.J_regressor_m, self.J_regressor_n], axis = 1, name="J_regressor") 
        
        #NOTE:
        # 10x 6890*3
        self.shapedirs_f = tf.Variable(shapedirs_f, name='shapedirs_f', dtype=dtype, trainable=False) 
        self.shapedirs_m = tf.Variable(shapedirs_m, name='shapedirs_m', dtype=dtype, trainable=False) 
        self.shapedirs_n = tf.Variable(shapedirs_n, name='shapedirs_n', dtype=dtype, trainable=False) 
        self.shapedirs = tf.concat([self.shapedirs_f, self.shapedirs_m, self.shapedirs_n], axis = 0) # 10*3 x 6890*3 
            
        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = self.dds[0]['posedirs'].shape[-1] # == 207; 
        # 207 x 6890*3 = 207 x 20670
        posedirs_f = np.reshape( undo_chumpy(self.dds[0]['posedirs']), [-1, num_pose_basis]).T
        self.posedirs_f = tf.Variable( posedirs_f, name='posedirs_f', dtype=dtype, trainable=False)
        posedirs_m = np.reshape( undo_chumpy(self.dds[1]['posedirs']), [-1, num_pose_basis]).T
        self.posedirs_m = tf.Variable( posedirs_m, name='posedirs_m', dtype=dtype, trainable=False)
        posedirs_n = np.reshape( undo_chumpy(self.dds[2]['posedirs']), [-1, num_pose_basis]).T
        self.posedirs_n = tf.Variable( posedirs_n, name='posedirs_n', dtype=dtype, trainable=False)

        # indices of parents for each joints
        self.parents = self.dds[0]['kintree_table'][0].astype(np.int32)
        #print("[***] self.parents = ", self.parents)

        # LBS weights:  size: (6890, 24);
        self.weights_f = tf.Variable(undo_chumpy(self.dds[0]['weights']), name='lbs_weights_f', 
                           dtype=dtype,
                           trainable=False)
        self.weights_m = tf.Variable(undo_chumpy(self.dds[1]['weights']), name='lbs_weights_m', 
                           dtype=dtype,
                           trainable=False)
        self.weights_n = tf.Variable(undo_chumpy(self.dds[2]['weights']), name='lbs_weights_n', 
                           dtype=dtype,
                           trainable=False)
            
        if self.joint_type == 'cocoplus':
            # This returns 19 keypoints: 6890 x 19
            print ("SMPL load cocoplus_regressor!!! for joint type %s" % self.joint_type)
            # joint_regressor, only occurs for 'cocoplus_regressor' joint type;
            self.joint_regressor = tf.Variable( self.dds[2]['cocoplus_regressor'].T.todense(),
                        name="cocoplus_regressor", dtype=dtype, trainable=False)

        #if self.joint_type == 'lsp':  # 14 LSP joints!
        #    self.joint_regressor = self.joint_regressor[:, :14]

        if self.joint_type not in ['cocoplus', 'lsp', 'smpl']:
            print('BAD!! Unknown joint type: %s, it must be either "smpl", cocoplus" or "lsp"' 
                      % self.joint_type)
            import ipdb
            ipdb.set_trace()
        
        self.lsp_idx_from_smpl24_joints = tf.constant(np.array(get_lsp_idx_from_smpl_joints(), dtype= np.int32))


    """ add new argument : 
          * gender 
    """
    def __call__(self, beta, theta, get_skin=False, name=None, gender = None):
        """
        Obtain SMPL with shape (beta) & pose (theta) inputs.
        Theta includes the global rotation.
        Args:
          beta: N x 10
          theta: N x 72 (with 3-D axis-angle rep)

          # added by CCJ:
          gender : N x 3, where 3 corresponds to ['female', 'male', 'neutral'];
                   e.g., gender[some_batch, :] = [0,0,1] means 'neutral', = [1,0,0] means 'female';

        Updates:
        self.J_transformed: N x 24 x 3 joint location after shaping
                 & posing with beta and theta
        Returns:
          - joints: N x 19 or 14 x 3 joint locations depending on joint_type
        If get_skin is True, also returns
          - Verts: N x 6890 x 3
        """
        
        with tf.name_scope(name, "smpl_main", [beta, theta]):
            
            num_batch = beta.shape[0].value

            # 1. Add shape blend shapes
            # before batch_gender: it was (N x 10) x (10 x 6890*3) = N x 6890 x 3
            # now it is : (N x 10*3) x (10*3 x 6890*3) = N x 6890 x 3
            # here '_e' means 'expanded' or 'tiled' ones; 
            # beta : N x 10 --> beta_e : N x 10*3
            beta_e = tf.tile(beta, [1,3]) # N x 10*3
            #gender = tf.Print(gender, [gender[0,:], tf.shape(gender)], '[ ??????] gender shape ...' )
            gender_e = tf.tile(tf.expand_dims(gender, -1), [1, 1, 10]) # N x 3 x 10
            gender_e = tf.cast(tf.reshape(gender_e, [num_batch, 30]), self.dtype) # N x 10*3
            # (N x 3 ) x (3 x 6890*3) = N x 6890*3 ==> N x 6890 x 3
            v_template = tf.reshape(tf.matmul(gender, tf.stack(
                [
                    tf.reshape(self.v_template_f, [-1]), # 6890*3
                    tf.reshape(self.v_template_m, [-1]),
                    tf.reshape(self.v_template_n, [-1]),
                ], axis = 0)), [-1, self.size[0], self.size[1]])
            # N x 6890 x 3
            #self.shapedirs = tf.Print(self.shapedirs, [ tf.shape(self.shapedirs)], message="[????] This shapedirs shape : ")
            v_shaped = tf.reshape(tf.matmul(beta_e*gender_e, # N x 10*3
                                  self.shapedirs,  # 10*3 x 6890*3 
                                  name='shape_bs'),
                                  [-1, self.size[0], self.size[1]]) + v_template
                             
            # 2. Infer shape-dependent joint locations.
            # J_regressor in shape - 6890 x 24
            # (N x 6890) x (6890 x 24) = N x 24
            """ female model """
            # N x 24 multiply N x 1
            gender_f = tf.reshape(gender[:,0], [num_batch, 1]) # N x 1
            Jx = tf.matmul(v_shaped[:, :, 0], self.J_regressor_f)* gender_f # N x 24
            Jy = tf.matmul(v_shaped[:, :, 1], self.J_regressor_f)* gender_f
            Jz = tf.matmul(v_shaped[:, :, 2], self.J_regressor_f) *gender_f

            """ male model """
            gender_m = tf.reshape(gender[:,1], [num_batch, 1]) # N x 1
            Jx += tf.matmul(v_shaped[:, :, 0], self.J_regressor_m) *gender_m # N x 24
            Jy += tf.matmul(v_shaped[:, :, 1], self.J_regressor_m) *gender_m
            Jz += tf.matmul(v_shaped[:, :, 2], self.J_regressor_m) *gender_m
            
            """ neutral model """
            gender_n = tf.reshape(gender[:,2], [num_batch, 1]) # N x 1
            Jx += tf.matmul(v_shaped[:, :, 0], self.J_regressor_n) *gender_n # N x 24
            Jy += tf.matmul(v_shaped[:, :, 1], self.J_regressor_n) *gender_n
            Jz += tf.matmul(v_shaped[:, :, 2], self.J_regressor_n) *gender_n
            #Jx = Jx_f + Jx_m + Jx_n # N x 24
            #Jy = Jy_f + Jy_m + Jy_n # N x 24
            #Jz = Jz_f + Jz_m + Jz_n # N x 24
            # tf.stack: Stacks a list of rank-R tensors into one rank-(R+1) tensor
            J = tf.stack([Jx, Jy, Jz], axis=2) # now in shape N x 24 x 3;

            # 3. Add pose blend shapes
            # N x 24 x 3 x 3
            Rs = tf.reshape(
                batch_rodrigues(tf.reshape(theta, [-1, 3])), [-1, 24, 3, 3])
            with tf.name_scope("lrotmin"):
                #NOTE: Ignore global rotation.
                pose_feature = tf.reshape(Rs[:, 1:, :, :] - tf.eye(3), 
                                  [-1, 207]) # N x 207 = N x 23*3*3;

            # (N x 207) x (207, 20670) -> N x 6890 x 3
            # before batch_gender: it was (N x 209) x (207 x 6890*3) = N x 6890*3 --> reshpe --> N x 6890 x 3;
            # now it is : (N x 207*3) x (207*3 x 6890*3) = N x 6890 x 3

            """ female model """
            # (N x 207) x (207, 6890*3) -> N x 6890*3
            v_posed =  tf.matmul(pose_feature, self.posedirs_f) * gender_f #f
            """ male model """
            v_posed += tf.matmul(pose_feature, self.posedirs_m) * gender_m #f+m
            """ neutral model """
            v_posed += tf.matmul(pose_feature, self.posedirs_n) * gender_n #f+m+n
            # add vertices from shaping;
            v_posed = tf.reshape(v_posed, [-1, self.size[0], self.size[1]]) + v_shaped

            # 4. Get the global joint location
            # with rotate_base=False as defualt;
            self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents) 

            # 5. Do skinning:
            """ female model """
            # W is N x 6890 x 24
            W_tmp = tf.reshape(tf.tile(self.weights_f, [num_batch, 1]), [num_batch, -1, 24])
            # (N x 6890 x 24) x (N x 24 x 16) --> N x 6890 x 16 (i.e., batch matrix multiplication)
            T = tf.reshape(tf.matmul(W_tmp, tf.reshape(A, [num_batch, 24, 16])), [num_batch, -1]) * gender_f # N x 6890*16
            """ male model """
            W_tmp = tf.reshape(tf.tile(self.weights_m, [num_batch, 1]), [num_batch, -1, 24])
            T += tf.reshape(tf.matmul(W_tmp, tf.reshape(A, [num_batch, 24, 16])), [num_batch, -1]) * gender_m # N x 6890*16, f+m
            """ neutral model """
            W_tmp = tf.reshape(tf.tile(self.weights_n, [num_batch, 1]), [num_batch, -1, 24])
            T += tf.reshape(tf.matmul(W_tmp, tf.reshape(A, [num_batch, 24, 16])), [num_batch, -1]) * gender_n # N x 6890*16, f+m+n
            # N x 6890 x 4 x 4
            T = tf.reshape( T, [num_batch, -1, 4, 4])
            
            v_posed_homo = tf.concat(
                [v_posed, tf.ones([num_batch, v_posed.shape[1], 1])], 2)
            
            v_homo = tf.matmul(T, tf.expand_dims(v_posed_homo, -1))
            verts = v_homo[:, :, :3, 0]

            # Get cocoplus or lsp joints:
            if self.joint_type == 'cocoplus':
                joint_x = tf.matmul(verts[:, :, 0], self.joint_regressor)
                joint_y = tf.matmul(verts[:, :, 1], self.joint_regressor)
                joint_z = tf.matmul(verts[:, :, 2], self.joint_regressor)
                joints = tf.stack([joint_x, joint_y, joint_z], axis=2)
            elif self.joint_type == 'lsp':
                # J_transformed : N x 24 x 3
                joints = tf.gather(self.J_transformed, self.lsp_idx_from_smpl24_joints, axis = 1)
            elif self.joint_type == 'smpl':
                joints = self.J_transformed
            # 6. add the trans; [1, 1, 3]
            #trans = tf.reshape(trans, [num_batch, 1, 3])
            if get_skin:
                return verts, joints, Rs, self.J_transformed
                #return verts, joints, Rs
            else:
                return joints, self.J_transformed
                #return joints



    def call__smpl_no_gender(self, beta, theta, get_skin=False, name=None):
        """
        Obtain SMPL with shape (beta) & pose (theta) inputs.
        Theta includes the global rotation.
        Args:
          beta: N x 10
          theta: N x 72 (with 3-D axis-angle rep)

          # added by CCJ:
          gender = could be one of 'female', 'male', or 'neutral'

        Updates:
        self.J_transformed: N x 24 x 3 joint location after shaping
                 & posing with beta and theta
        Returns:
          - joints: N x 19 or 14 x 3 joint locations depending on joint_type
        If get_skin is True, also returns
          - Verts: N x 6890 x 3
        """
        
        with tf.name_scope(name, "smpl_main", [beta, theta]):
            
            num_batch = beta.shape[0].value
             
            # 1. Add shape blend shapes
            # (N x 10) x (10 x 6890*3) = N x 6890 x 3
            v_shaped = tf.reshape(
                tf.matmul(beta, self.shapedirs, name='shape_bs'),
                [-1, self.size[0], self.size[1]]) + self.v_template

            # 2. Infer shape-dependent joint locations.
            # J_regressor in shape - 6890 x 24
            Jx = tf.matmul(v_shaped[:, :, 0], self.J_regressor)
            Jy = tf.matmul(v_shaped[:, :, 1], self.J_regressor)
            Jz = tf.matmul(v_shaped[:, :, 2], self.J_regressor)
            #tf.stack: Stacks a list of rank-R tensors into one rank-(R+1) tensor
            J = tf.stack([Jx, Jy, Jz], axis=2) # now in shape N x 24 x 3;

            # 3. Add pose blend shapes
            # N x 24 x 3 x 3
            Rs = tf.reshape(batch_rodrigues(tf.reshape(theta, [-1, 3])), [-1, 24, 3, 3])
            with tf.name_scope("lrotmin"):
                #NOTE: Ignore global rotation.
                pose_feature = tf.reshape(Rs[:, 1:, :, :] - tf.eye(3), 
                                  [-1, 207]) # 207 = 23*3*3;

            # (N x 207) x (207, 20670) -> N x 6890 x 3
            v_posed = tf.reshape(
                tf.matmul(pose_feature, self.posedirs),
                [-1, self.size[0], self.size[1]]) + v_shaped

            #4. Get the global joint location
            # with rotate_base=False as defualt;
            self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents) 
            
            #NOTE: debugging;
            #idx_tf = tf.constant(idx)
            #self.J_transformed = tf.Print(self.J_transformed, 
            #        [   idx_tf, 
            #            self.J_transformed[0,0,:], 
            #            tf.shape(self.J_transformed), 
            #            self.J_regressor[0,:3],
            #            theta[0, :], beta[0,:] ], 
            #        "[***] tf smpl, idx, J_transformed[0], input pose, input shape: ",
            #        first_n = -1
            #        )

            # 5. Do skinning:
            # W is N x 6890 x 24
            W = tf.reshape(
                tf.tile(self.weights, [num_batch, 1]), [num_batch, -1, 24])
            # (N x 6890 x 24) x (N x 24 x 16)
            T = tf.reshape( tf.matmul(W, tf.reshape(A, [num_batch, 24, 16])),
                [num_batch, -1, 4, 4])
            v_posed_homo = tf.concat(
                [v_posed, tf.ones([num_batch, v_posed.shape[1], 1])], 2)
            
            v_homo = tf.matmul(T, tf.expand_dims(v_posed_homo, -1))
            verts = v_homo[:, :, :3, 0]

            # Get cocoplus or lsp joints:
            if self.joint_type == 'cocoplus':
                joint_x = tf.matmul(verts[:, :, 0], self.joint_regressor)
                joint_y = tf.matmul(verts[:, :, 1], self.joint_regressor)
                joint_z = tf.matmul(verts[:, :, 2], self.joint_regressor)
                joints = tf.stack([joint_x, joint_y, joint_z], axis=2)
            elif self.joint_type == 'lsp':
                # J_transformed : N x 24 x 3
                joints = tf.gather(self.J_transformed,self.lsp_idx_from_smpl24_joints, axis = 1)
            
            elif self.joint_type == 'smpl':
                joints = self.J_transformed
            
            # 6. add the trans; [1, 1, 3]
            #trans = tf.reshape(trans, [num_batch, 1, 3])
            if get_skin:
                return verts, joints, Rs, self.J_transformed
                #return verts, joints, Rs
            else:
                return joints, self.J_transformed
                #return joints


""" The original HMR version """
class SMPL_v1(object):
    def __init__(self, pkl_path, joint_type='cocoplus', dtype=tf.float32):
        """
        pkl_path is the path to a SMPL model
        """
        # -- Load SMPL params --
        with open(pkl_path, 'rb') as f:
            #dd = pickle.load(f, encoding="latin-1") 
            dd = pickle.load(f) 
        # Mean template vertices
        self.v_template = tf.Variable(
            undo_chumpy(dd['v_template']),
            name='v_template',
            dtype=dtype,
            trainable=False)
        # Size of mesh [Number of vertices, 3]
        self.size = [self.v_template.shape[0].value, 3]
        self.num_betas = dd['shapedirs'].shape[-1]
        # Shape blend shape basis: 6890 x 3 x 10
        # reshaped to 6890*30 x 10, transposed to 10x6890*3
        shapedir = np.reshape(
            undo_chumpy(dd['shapedirs']), [-1, self.num_betas]).T
        self.shapedirs = tf.Variable(
            shapedir, name='shapedirs', dtype=dtype, trainable=False)

        # Regressor for joint locations given shape - 6890 x 24
        self.J_regressor = tf.Variable(
            dd['J_regressor'].T.todense(),
            name="J_regressor",
            dtype=dtype,
            trainable=False)

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
        num_pose_basis = dd['posedirs'].shape[-1]
        # 207 x 20670
        posedirs = np.reshape(
            undo_chumpy(dd['posedirs']), [-1, num_pose_basis]).T
        self.posedirs = tf.Variable(
            posedirs, name='posedirs', dtype=dtype, trainable=False)

        # indices of parents for each joints
        self.parents = dd['kintree_table'][0].astype(np.int32)

        # LBS weights
        self.weights = tf.Variable(
            undo_chumpy(dd['weights']),
            name='lbs_weights',
            dtype=dtype,
            trainable=False)

        # This returns 19 keypoints: 6890 x 19
        self.joint_regressor = tf.Variable(
            dd['cocoplus_regressor'].T.todense(),
            name="cocoplus_regressor",
            dtype=dtype,
            trainable=False)
        if joint_type == 'lsp':  # 14 LSP joints!
            self.joint_regressor = self.joint_regressor[:, :14]

        if joint_type not in ['cocoplus', 'lsp']:
            print('BAD!! Unknown joint type: %s, it must be either "cocoplus" or "lsp"' % joint_type)
            import ipdb
            ipdb.set_trace()

    def __call__(self, beta, theta, get_skin=False, name=None):
        """
        Obtain SMPL with shape (beta) & pose (theta) inputs.
        Theta includes the global rotation.
        Args:
          beta: N x 10
          theta: N x 72 (with 3-D axis-angle rep)
        Updates:
        self.J_transformed: N x 24 x 3 joint location after shaping
                 & posing with beta and theta
        Returns:
          - joints: N x 19 or 14 x 3 joint locations depending on joint_type
        If get_skin is True, also returns
          - Verts: N x 6890 x 3
        """

        with tf.name_scope(name, "smpl_main", [beta, theta]):
            num_batch = beta.shape[0].value

            # 1. Add shape blend shapes
            # (N x 10) x (10 x 6890*3) = N x 6890 x 3
            v_shaped = tf.reshape(
                tf.matmul(beta, self.shapedirs, name='shape_bs'),
                [-1, self.size[0], self.size[1]]) + self.v_template

            # 2. Infer shape-dependent joint locations.
            Jx = tf.matmul(v_shaped[:, :, 0], self.J_regressor)
            Jy = tf.matmul(v_shaped[:, :, 1], self.J_regressor)
            Jz = tf.matmul(v_shaped[:, :, 2], self.J_regressor)
            J = tf.stack([Jx, Jy, Jz], axis=2)

            # 3. Add pose blend shapes
            # N x 24 x 3 x 3
            Rs = tf.reshape(
                batch_rodrigues(tf.reshape(theta, [-1, 3])), [-1, 24, 3, 3])
            with tf.name_scope("lrotmin"):
                # Ignore global rotation.
                pose_feature = tf.reshape(Rs[:, 1:, :, :] - tf.eye(3),
                                          [-1, 207])

            # (N x 207) x (207, 20670) -> N x 6890 x 3
            v_posed = tf.reshape(
                tf.matmul(pose_feature, self.posedirs),
                [-1, self.size[0], self.size[1]]) + v_shaped

            #4. Get the global joint location
            self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents)

            # 5. Do skinning:
            # W is N x 6890 x 24
            W = tf.reshape(
                tf.tile(self.weights, [num_batch, 1]), [num_batch, -1, 24])
            # (N x 6890 x 24) x (N x 24 x 16)
            T = tf.reshape(
                tf.matmul(W, tf.reshape(A, [num_batch, 24, 16])),
                [num_batch, -1, 4, 4])
            v_posed_homo = tf.concat(
                [v_posed, tf.ones([num_batch, v_posed.shape[1], 1])], 2)
            v_homo = tf.matmul(T, tf.expand_dims(v_posed_homo, -1))

            verts = v_homo[:, :, :3, 0]

            # Get cocoplus or lsp joints:
            joint_x = tf.matmul(verts[:, :, 0], self.joint_regressor)
            joint_y = tf.matmul(verts[:, :, 1], self.joint_regressor)
            joint_z = tf.matmul(verts[:, :, 2], self.joint_regressor)
            joints = tf.stack([joint_x, joint_y, joint_z], axis=2)

            if get_skin:
                return verts, joints, Rs, self.J_transformed
            else:
                return joints
