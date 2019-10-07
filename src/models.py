"""
Defines networks.

@Encoder_resnet
@Encoder_resnet_v1_101
@Encoder_fc3_dropout

@Discriminator_separable_rotations

Helper:
@get_encoder_fn_separate
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.contrib.layers.python.layers.initializers import variance_scaling_initializer


def Encoder_resnet(x, depth = None, is_training=True, weight_decay=0.001, reuse=False):
    """
    Resnet v2-50
    Assumes input is [batch, height_in, width_in, channels]!!
    Input:
    - x: N x H x W x 3
    - depth : N x H x W x 1
    - weight_decay: float
    - reuse: bool->True if test

    Outputs:
    - cam: N x 3
    - Pose vector: N x 72
    - Shape vector: N x 10
    - variables: tf variables
    """
    from tensorflow.contrib.slim.python.slim.nets import resnet_v2
    with tf.name_scope("Encoder_resnet", [x, depth]):
        with slim.arg_scope(
                resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            """ added by CCJ from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/resnet_v2.py;
                (*) Args: reuse: whether or not the network and its variables should be reused. To be able to reuse 'scope' must be given.
                (*) resnet_v2() Returns:
                'net': A rank-4 tensor of size [batch, height_out, width_out, channels_out].
                      - If 'global_pool' is False, then height_out and width_out are reduced by a factor of output_stride compared to the respective height_in and width_in,
                      - else both height_out and width_out equal one. 
                      - If num_classes is None, then net is the output of the last ResNet block, potentially after global average pooling. 
                      - If num_classes is not None, net contains the pre-softmax activations.
                (*) end_points: A dictionary from components of the network to the corresponding activation.
            """
            net, end_points = resnet_v2.resnet_v2_50(
                inputs = x, # inputs,
                num_classes=None,
                is_training=is_training, 
                #added by CCJ: global_pool=True as default;
                reuse=reuse,
                scope='resnet_v2_50'
                #scope='resnet_v2_50_img'
                )
            net = tf.squeeze(net, axis=[1, 2])
            
            # added by CCJ: for depth encoded by resnet_v2_50
            net_depth = None
            if depth is not None:
                net_depth, end_points_depth = resnet_v2.resnet_v2_50(
                        inputs = depth, # inputs,
                        num_classes=None,
                        is_training=is_training, 
                        #added by CCJ: global_pool=True as default;
                        #reuse= tf.AUTO_REUSE,
                        scope='resnet_v2_50_dep'
                        )
                net_depth = tf.squeeze(net_depth, axis=[1, 2])
    
    variables = tf.contrib.framework.get_variables('resnet_v2_50_img')
    if depth is not None:
        variables = variables + tf.contrib.framework.get_variables('resnet_v2_50_dep')
    #return net, variables
    return net, net_depth, variables

def Encoder_resnet_v2(x, depth = None, is_training=True, weight_decay=0.001, reuse=False, name = 'Encoder_resnet_v2'):
    """
    Resnet v2-50
    Assumes input is [batch, height_in, width_in, channels]!!
    Input:
    - x: N x H x W x 3
    - weight_decay: float
    - reuse: bool->True if test

    Outputs:
    - cam: N x 3
    - Pose vector: N x 72
    - Shape vector: N x 10
    - variables: tf variables
    """
    from tensorflow.contrib.slim.python.slim.nets import resnet_v2
    with tf.variable_scope(name, reuse=reuse) as scope:
    #with tf.name_scope("Encoder_resnet", [x, depth]):
        # Defines the default ResNet arg scope
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=weight_decay)):
            """ added by CCJ from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/resnet_v2.py;
                (*) Args: reuse: whether or not the network and its variables should be reused. To be able to reuse 'scope' must be given.
                (*) resnet_v2() Returns:
                'net': A rank-4 tensor of size [batch, height_out, width_out, channels_out].
                      - If 'global_pool' is False, then height_out and width_out are reduced by a factor of output_stride compared 
                        to the respective height_in and width_in,
                      - else both height_out and width_out equal one. 
                      - If num_classes is None, then net is the output of the last ResNet block, potentially after global average pooling. 
                      - If num_classes is not None, net contains the pre-softmax activations.
                (*) end_points: A dictionary from components of the network to the corresponding activation.
            """
            with tf.variable_scope("resnet_v2_50_img"):
                net_img, end_points_img = resnet_v2.resnet_v2_50(
                        inputs = x, # inputs,
                        num_classes=None,
                        is_training=is_training, 
                        #added by CCJ: global_pool=True as default;
                        #reuse=reuse, 
                        #scope='resnet_v2_50_img'
                        )
                net_img = tf.squeeze(net_img, axis=[1, 2]) # output in N x 2048
            
            # added by CCJ: for depth encoded by resnet_v2_50
            with tf.variable_scope("resnet_v2_50_dep"):
                net_depth = tf.zeros(shape = tf.shape(net_img), dtype=tf.float32)
                if depth is not None:
                    #NOTE: since we want to load the pre-trained resnet_v2_50 model, 
                    # which assumes the input tensor has 3 channels;
                    # So we copy the depth to 3 chanels. Maybe have to find more advanced way to deal with this;
                    net_depth, end_points_depth = resnet_v2.resnet_v2_50(
                            inputs = tf.concat([depth,depth, depth], axis = -1), # inputs,
                            num_classes=None,
                            is_training=is_training, 
                            #added by CCJ: global_pool=True as default;
                            #reuse= tf.AUTO_REUSE,
                            #scope='resnet_v2_50_dep'
                            )
                    net_depth = tf.squeeze(net_depth, axis=[1, 2]) # output in N x 2048
        
        # NOTE: to be updated:!!!     
        # added by CCJ: fully connected layer for encoded_depth and encoded_image
        with tf.variable_scope("encoder_fc"):
            #tmp = tf.concat([net_img, net_depth], 1)
            #net = slim.fully_connected(tmp, 2048*2)
            
            tmp = tf.concat([
                slim.fully_connected(net_img, 2048), 
                slim.fully_connected(net_depth, 2048)], 1)
            net = slim.fully_connected(tmp, 2048)
        

    variables = tf.contrib.framework.get_variables(scope)
    #print ("[**] Encoder_resnet_v2(), returns variables: ", variables)
    return net, variables


def Encoder_fc3_dropout(x,
                        num_output=85,
                        is_training=True,
                        reuse=False,
                        name="3D_module"):
    """
    3D inference module. 3 MLP layers (last is the output)
    With dropout  on first 2.
    Input:
    - x: N x [|img_feat|, |3D_param|]
    - reuse: bool

    Outputs:
    - 3D params: N x num_output
      if orthogonal: 
           either 85: (3 + 24*3 + 10) or 109 (3 + 24*4 + 10) for factored axis-angle representation
      if perspective:
          86: (f, tx, ty, tz) + 24*3 + 10, or 110 for factored axis-angle.
    - variables: tf variables
    """
    if reuse:
        print('Reuse is on!')
    with tf.variable_scope(name, reuse=reuse) as scope:
        net = slim.fully_connected(x, 1024, scope='fc1')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout1')
        net = slim.fully_connected(net, 1024, scope='fc2')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout2')
        small_xavier = variance_scaling_initializer(
            factor=.01, mode='FAN_AVG', uniform=True)
        net = slim.fully_connected(
            net,
            num_output,
            activation_fn=None,
            weights_initializer=small_xavier,
            scope='fc3')

    variables = tf.contrib.framework.get_variables(scope)
    return net, variables


def get_encoder_fn_separate(model_type, 
        # e.g., model_type = 'resnet_fc3_dropout'
        encoder_fn_type = "Encoder_resnet_v2"
        ):
    """
    Retrieves diff encoder fn for image and 3D
    """
    encoder_fn = None
    threed_fn = None
    if 'resnet' in model_type:
        #encoder_fn = Encoder_resnet
        # added by CCJ
        encoder_fn = Encoder_resnet_v2
        if encoder_fn_type is "Encoder_resnet":
            encoder_fn = Encoder_resnet

    else:
        print('Unknown encoder %s!' % model_type)
        exit(1)

    if 'fc3_dropout' in model_type:
        threed_fn = Encoder_fc3_dropout

    if encoder_fn is None or threed_fn is None:
        print('Dont know what encoder to use for %s' % model_type)
        import ipdb
        ipdb.set_trace()

    return encoder_fn, threed_fn


def Discriminator_separable_rotations(
        poses,
        shapes,
        weight_decay,
):
    """
    23 Discriminators on each joint + 1 for all joints + 1 for shape.
    To share the params on rotations, this treats the 23 rotation matrices
    as a "vertical image":
    Do 1x1 conv, then send off to 23 independent classifiers.

    Input:
    - poses: N x 23 x 1 x 9, NHWC ALWAYS!!
    - shapes: N x 10
    - weight_decay: float

    Outputs:
    - prediction: N x (1+23) or N x (1+23+1) if do_joint is on.
    - variables: tf variables
    """
    data_format = "NHWC"
    with tf.name_scope("Discriminator_sep_rotations", [poses, shapes]):
        with tf.variable_scope("D") as scope:
            with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                    weights_regularizer=slim.l2_regularizer(weight_decay)):
                with slim.arg_scope([slim.conv2d], data_format=data_format):
                    poses = slim.conv2d(poses, 32, [1, 1], scope='D_conv1')
                    poses = slim.conv2d(poses, 32, [1, 1], scope='D_conv2')
                    theta_out = []
                    for i in range(0, 23):
                        theta_out.append(
                            slim.fully_connected(
                                poses[:, i, :, :],
                                1,
                                activation_fn=None,
                                scope="pose_out_j%d" % i))
                    theta_out_all = tf.squeeze(tf.stack(theta_out, axis=1))

                    # Do shape on it's own:
                    shapes = slim.stack(
                        shapes,
                        slim.fully_connected, [10, 5],
                        scope="shape_fc1")
                    shape_out = slim.fully_connected(
                        shapes, 1, activation_fn=None, scope="shape_final")
                    """ Compute joint correlation prior!"""
                    nz_feat = 1024
                    poses_all = slim.flatten(poses, scope='vectorize')
                    poses_all = slim.fully_connected(
                        poses_all, nz_feat, scope="D_alljoints_fc1")
                    poses_all = slim.fully_connected(
                        poses_all, nz_feat, scope="D_alljoints_fc2")
                    poses_all_out = slim.fully_connected(
                        poses_all,
                        1,
                        activation_fn=None,
                        scope="D_alljoints_out")
                    out = tf.concat([theta_out_all,
                                     poses_all_out, shape_out], 1)

            variables = tf.contrib.framework.get_variables(scope)
            return out, variables
