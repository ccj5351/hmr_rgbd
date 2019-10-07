# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: smpl_perceptron.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 02-08-2019
# @last modified: Fri 02 Aug 2019 11:00:12 AM EDT

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import json
import numpy as np


def _load_3_hidden_layer_saved_model_json(json_model):
    with open(json_model) as f:
        data = json.load(f)
    saved_model_dict = {}
    # data has keys: 
    #  1) layer1: 'regressor.0.bias', 'regressor.0.weight'
    #  2) layer2: 'regressor.2.bias', 'regressor.2.weight'
    #  3) layer3: 'regressor.4.bias', 'regressor.4.weight'
    #  4) layer4: 'regressor.6.bias', 'regressor.6.weight'
    saved_keys = [
        'regressor.0.weight', 'regressor.0.bias', 
        'regressor.2.weight', 'regressor.2.bias', 
        'regressor.4.weight', 'regressor.4.bias', 
        'regressor.6.weight', 'regressor.6.bias', 
    ]
    want_keys = [
        'weight_h1',  'b1',
        'weight_h2',  'b2',
        'weight_h3',  'b3',
        'weight_out', 'out',
    ]
    for i in range(0, len(want_keys)):
        tmp = np.array(data[saved_keys[i]]).astype(np.float32)
        saved_model_dict[want_keys[i]] = tmp 
        print ("{} has shape {}".format(saved_keys[i], tmp.shape))
    return saved_model_dict


def _load_7_hidden_layer_saved_model_json(json_model):
    with open(json_model) as f:
        data = json.load(f)
    print ('data.keys = {}'.format(data.keys()))
    for i in data.keys():
        print ("{} has shape {}".format(i, np.array(data[i]).shape))

    saved_model_dict = {}
    # data has keys: 
    #  1) layer1: 'regressor.0.bias', 'regressor.0.weight'
    #  2) layer2: 'regressor.2.bias', 'regressor.2.weight'
    #  3) layer3: 'regressor.4.bias', 'regressor.4.weight'
    #  4) layer4: 'regressor.6.bias', 'regressor.6.weight'
    saved_keys = [
        # weights and bias for convolution
        'regressor.0.weight', 'regressor.0.bias', 
        'regressor.2.weight', 'regressor.2.bias', 
        'regressor.4.weight', 'regressor.4.bias', 
        'regressor.6.weight', 'regressor.6.bias', 
        'regressor.8.weight', 'regressor.8.bias', 
        'regressor.10.weight', 'regressor.10.bias', 
        'regressor.12.weight', 'regressor.12.bias', 
        'regressor.14.weight', 'regressor.14.bias', 
        # weights for PReLU
        'regressor.1.weight', 
        'regressor.3.weight', 
        'regressor.5.weight', 
        'regressor.7.weight', 
        'regressor.9.weight', 
        'regressor.11.weight', 
        'regressor.13.weight', 
    ]
    want_keys = [
        'weight_h1',  'b1',
        'weight_h2',  'b2',
        'weight_h3',  'b3',
        'weight_h4',  'b4',
        'weight_h5',  'b5',
        'weight_h6',  'b6',
        'weight_h7',  'b7',
        'weight_out', 'out',
        'a1', 'a2', 'a3', 'a4',
        'a5', 'a6', 'a7',
    ]

    assert (len(want_keys) == len(saved_keys))

    for i in range(0, len(want_keys)):
        tmp = np.array(data[saved_keys[i]]).astype(np.float32)
        saved_model_dict[want_keys[i]] = tmp 
        #print ("{} has shape {}".format(saved_keys[i], tmp.shape))
    return saved_model_dict


""" 1 hidden layer perceptron """
class One_hidden_layer_joints3d_2_SmplRegressor(object):
    def __init__(self, config):
        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([config.num_input, config.num_hidden_1])),
            #'h2': tf.Variable(tf.random_normal([config.num_hidden_1, config.num_hidden_2])),
            #'out': tf.Variable(tf.random_normal([config.num_hidden_2, config.num_output]))
            'out': tf.Variable(tf.random_normal([config.num_hidden_1, config.num_output]))
            }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([config.num_hidden_1])),
            #'b2': tf.Variable(tf.random_normal([config.num_hidden_2])),
            'out': tf.Variable(tf.random_normal([config.num_output]))
        }

    # Create model
    def build_model(self, x):
        layer = tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1'])
        #layer = tf.add(tf.matmul(layer, self.weights['h2']), self.biases['b2'])
        
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer, self.weights['out']) + self.biases['out']
        return out_layer


""" 3 hidden layer perceptron """
class Three_hidden_layer_joints3d2LspSmplRegressor(object):
    def __init__(self, saved_model_json_file = 'models/model_3_layer_perceptron/pytorchSmplRegressor145000.json'):

        
        # Store layers weight & bias
        self.num_input = 14*3 # 42
        self.num_hidden_1 = 64
        self.num_hidden_2 = 128
        self.num_hidden_3 = 128
        self.num_output = 24*3 + 10 # 82
        
        self.saved_model_dict = _load_3_hidden_layer_saved_model_json(saved_model_json_file)
        
        #NOTE:
        """ the pretraiend weights are saved via PyTorch, but their dimension are just transposed 
            w.r.t. the dimension defined here (i.e., Tensorflow). 
            For example: weight_h1 : self.num_input x self.num_hidden_1 = 42 x 64 in here;
                         but the loaded variable is in the transposed format, i.e., 64 x 42;
        """
        keys = ['weight_h1', 'weight_h2', 'weight_h3', 'weight_out']
        #sizes = [self.num_hidden_1, self.num_hidden_2, self.num_hidden_3, self.num_output]
        for i in range(0, len(keys)):
            self.saved_model_dict[keys[i]] = self.saved_model_dict[keys[i]].T # transposed
            #print ('{} transposed'.format(keys[i]))

        self.weights = {
            #'h1': tf.Variable(tf.random_normal([self.num_input, self.num_hidden_1])),
            'h1': tf.Variable(
                np.reshape(
                    self.saved_model_dict['weight_h1'], 
                    [self.num_input, self.num_hidden_1]), 
                    dtype=tf.float32),
            'h2': tf.Variable(
                np.reshape(
                    self.saved_model_dict['weight_h2'], 
                    [self.num_hidden_1, self.num_hidden_2]), 
                    dtype=tf.float32),

            'h3': tf.Variable(
                np.reshape(
                    self.saved_model_dict['weight_h3'], 
                    [self.num_hidden_2, self.num_hidden_3]), 
                    dtype=tf.float32),

            'out': tf.Variable(
                np.reshape(
                    self.saved_model_dict['weight_out'], 
                    [self.num_hidden_3, self.num_output]), 
                    dtype=tf.float32)
            }
        
        self.biases = {
            #'b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'b1':  tf.Variable( np.reshape(self.saved_model_dict['b1'],  [self.num_hidden_1]), dtype=tf.float32),
            'b2':  tf.Variable( np.reshape(self.saved_model_dict['b2'],  [self.num_hidden_2]), dtype=tf.float32),
            'b3':  tf.Variable( np.reshape(self.saved_model_dict['b3'],  [self.num_hidden_3]), dtype=tf.float32),
            'out': tf.Variable( np.reshape(self.saved_model_dict['out'], [self.num_output]  ), dtype=tf.float32),
        }

        #self.layers = {}

    # Create model
    #def build_model(self, x):
    #    self.layers['layer1'] = tf.nn.relu(tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1']))
    #    self.layers['layer2'] = tf.nn.relu(tf.add(tf.matmul(self.layers['layer1'], self.weights['h2']), self.biases['b2']))
    #    self.layers['layer3'] = tf.nn.relu(tf.add(tf.matmul(self.layers['layer2'], self.weights['h3']), self.biases['b3']))
    #    out_layer = tf.matmul(self.layers['layer3'], self.weights['out']) + self.biases['out']
    #    return out_layer
    
    def build_model(self, x):
        layer = tf.nn.relu(tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1']))
        layer = tf.nn.relu(tf.add(tf.matmul(layer, self.weights['h2']), self.biases['b2']))
        layer = tf.nn.relu(tf.add(tf.matmul(layer, self.weights['h3']), self.biases['b3']))
        out_layer = tf.matmul(layer, self.weights['out']) + self.biases['out']
        return out_layer

# > see How to implement PReLU activation in Tensorflow?
# > at https://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow

def parametric_relu(_x, alphas):
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5
  return pos + neg


""" 7 hidden layer perceptron """
class Seven_hidden_layer_joints3d2LspSmplRegressor(object):
    def __init__(self, saved_model_json_file 
                = 'models/model_7_layer_perceptron/pytorchSmplRegressor15000.json'
                ):

        
        # Store layers weight & bias
        self.num_input = 14*3 # 42
        self.num_hidden_1 = 64
        self.num_hidden_2 = 128
        self.num_hidden_3 = 256
        self.num_hidden_4 = 512
        self.num_hidden_5 = 512
        self.num_hidden_6 = 256
        self.num_hidden_7 = 128
        self.num_output = 24*3 + 10 # 82
        
        self.saved_model_dict = _load_7_hidden_layer_saved_model_json(saved_model_json_file)
        
        #NOTE:
        """ the pretraiend weights are saved via PyTorch, but their dimension are just transposed 
            w.r.t. the dimension defined here (i.e., Tensorflow). 
            For example: weight_h1 : self.num_input x self.num_hidden_1 = 42 x 64 in here;
                         but the loaded variable is in the transposed format, i.e., 64 x 42;
        """
        keys = ['weight_h1', 'weight_h2', 'weight_h3', 
                'weight_h4', 'weight_h5', 'weight_h6',
                'weight_h7', 'weight_out']
        for i in range(0, len(keys)):
            self.saved_model_dict[keys[i]] = self.saved_model_dict[keys[i]].T # transposed
            #print ('{} transposed'.format(keys[i]))

        self.weights = {
            #'h1': tf.Variable(tf.random_normal([self.num_input, self.num_hidden_1])),
            'h1': tf.Variable(
                np.reshape(
                    self.saved_model_dict['weight_h1'], 
                    [self.num_input, self.num_hidden_1]), 
                    dtype=tf.float32),
            'h2': tf.Variable(
                np.reshape(
                    self.saved_model_dict['weight_h2'], 
                    [self.num_hidden_1, self.num_hidden_2]), 
                    dtype=tf.float32),

            'h3': tf.Variable(
                np.reshape(
                    self.saved_model_dict['weight_h3'], 
                    [self.num_hidden_2, self.num_hidden_3]), 
                    dtype=tf.float32),

            'h4': tf.Variable(
                np.reshape(
                    self.saved_model_dict['weight_h4'], 
                    [self.num_hidden_3, self.num_hidden_4]), 
                    dtype=tf.float32),

            'h5': tf.Variable(
                np.reshape(
                    self.saved_model_dict['weight_h5'], 
                    [self.num_hidden_4, self.num_hidden_5]), 
                    dtype=tf.float32),

            'h6': tf.Variable(
                np.reshape(
                    self.saved_model_dict['weight_h6'], 
                    [self.num_hidden_5, self.num_hidden_6]), 
                    dtype=tf.float32),

            'h7': tf.Variable(
                np.reshape(
                    self.saved_model_dict['weight_h7'], 
                    [self.num_hidden_6, self.num_hidden_7]), 
                    dtype=tf.float32),
            
            'out': tf.Variable(
                np.reshape(
                    self.saved_model_dict['weight_out'], 
                    [self.num_hidden_7, self.num_output]), 
                    dtype=tf.float32),

        
            # for parametric Rectified Linear Unit (PReLU);
            'a1':  tf.Variable( np.reshape(self.saved_model_dict['a1'],  [1]), dtype=tf.float32),
            'a2':  tf.Variable( np.reshape(self.saved_model_dict['a2'],  [1]), dtype=tf.float32),
            'a3':  tf.Variable( np.reshape(self.saved_model_dict['a3'],  [1]), dtype=tf.float32),
            'a4':  tf.Variable( np.reshape(self.saved_model_dict['a4'],  [1]), dtype=tf.float32),
            'a5':  tf.Variable( np.reshape(self.saved_model_dict['a5'],  [1]), dtype=tf.float32),
            'a6':  tf.Variable( np.reshape(self.saved_model_dict['a6'],  [1]), dtype=tf.float32),
            'a7':  tf.Variable( np.reshape(self.saved_model_dict['a7'],  [1]), dtype=tf.float32),
            }
        
        self.biases = {
            #'b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'b1':  tf.Variable( np.reshape(self.saved_model_dict['b1'],  [self.num_hidden_1]), dtype=tf.float32),
            'b2':  tf.Variable( np.reshape(self.saved_model_dict['b2'],  [self.num_hidden_2]), dtype=tf.float32),
            'b3':  tf.Variable( np.reshape(self.saved_model_dict['b3'],  [self.num_hidden_3]), dtype=tf.float32),
            'b4':  tf.Variable( np.reshape(self.saved_model_dict['b4'],  [self.num_hidden_4]), dtype=tf.float32),
            'b5':  tf.Variable( np.reshape(self.saved_model_dict['b5'],  [self.num_hidden_5]), dtype=tf.float32),
            'b6':  tf.Variable( np.reshape(self.saved_model_dict['b6'],  [self.num_hidden_6]), dtype=tf.float32),
            'b7':  tf.Variable( np.reshape(self.saved_model_dict['b7'],  [self.num_hidden_7]), dtype=tf.float32),
            'out': tf.Variable( np.reshape(self.saved_model_dict['out'], [self.num_output]  ), dtype=tf.float32),
        }
        #self.layers = {}

    # Create model
    def build_model(self, x):
        #layer = tf.nn.relu(tf.add(tf.matmul(x,     self.weights['h1']), self.biases['b1']))
        layer = tf.add(tf.matmul(x,     self.weights['h1']), self.biases['b1'])
        layer = parametric_relu(layer, self.weights['a1'])

        layer = tf.add(tf.matmul(layer, self.weights['h2']), self.biases['b2'])
        layer = parametric_relu(layer, self.weights['a2'])

        layer = tf.add(tf.matmul(layer, self.weights['h3']), self.biases['b3'])
        layer = parametric_relu(layer, self.weights['a3'])
        
        layer = tf.add(tf.matmul(layer, self.weights['h4']), self.biases['b4'])
        layer = parametric_relu(layer, self.weights['a4'])
        
        layer = tf.add(tf.matmul(layer, self.weights['h5']), self.biases['b5'])
        layer = parametric_relu(layer, self.weights['a5'])
        
        layer = tf.add(tf.matmul(layer, self.weights['h6']), self.biases['b6'])
        layer = parametric_relu(layer, self.weights['a6'])
        
        layer = tf.add(tf.matmul(layer, self.weights['h7']), self.biases['b7'])
        layer = parametric_relu(layer, self.weights['a7'])
        
        out_layer = tf.matmul(layer, self.weights['out']) + self.biases['out']
        return out_layer