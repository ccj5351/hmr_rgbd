# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: test_tf_funcs.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 13-08-2019
# @last modified: Tue 13 Aug 2019 05:38:05 PM EDT

import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    

    
    """ test tf.gather_nd """ 
    # data is [[[ 0  1]
    #          [ 2  3]
    #          [ 4  5]]
    #
    #         [[ 6  7]
    #          [ 8  9]
    #          [10 11]]]
    data = np.reshape(np.arange(12), [2, 3, 2])
    x = tf.constant(data)

    idx_1 = [[[0, 0, 0], [0, 1, 1]], [[1, 0, 1], [1, 1, 0]]] # 2 x 2 x 3
    result1 = tf.gather_nd(x, idx_1)
    
    idx_2 = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]] # 4 x 3
    result2 = tf.gather_nd(x, idx_2)
    
    # Construct a 'Session' to execute the graph.
    sess = tf.Session()
    # Execute the graph and store the value that `e` represents in `result`.
    x, res1, res2  = sess.run([x, result1, result2])

    print ('x  = {}'.format(x))
    print ('res1 = {}'.format(res1))
    print ('res2 = {}'.format(res2))
