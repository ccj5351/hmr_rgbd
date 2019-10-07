# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: test_sigmoid.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 14-08-2019
# @last modified: Thu 15 Aug 2019 09:51:02 AM EDT

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

def sigmoid(x, alpha):
    f = (1.0 - np.exp(- alpha* x)) / (1.0 + np.exp(- alpha * x))
    return f

def arc_sigmoid(f, alpha):
    x = (-1.0/alpha)*np.log((1.0-f)/(1.0+f))
    return x

if __name__ == "__main__":
    x_min = -400*1.0e-3
    x_max = 400*1.0e-3

    x_min = -400
    x_max = 400
    c = 1.0e-3
    c = 1.0e-2
    x = np.array(range(x_min, x_max)) * c
    x_min *= c
    x_max *= c
    alpha = 0.2*100
    alpha = 1.0
    f = sigmoid(x, alpha)
    
    f0 = np.array([0.001, 0.01, 0.1, 0.5, 0.99])
    x0 = arc_sigmoid(f0, alpha)
    np.set_printoptions(precision=3, suppress=True)
    print ("x0 = {}".format(x0))
    print ("f0 = {}".format(f0))
    
    x1 = np.array([0.01, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 4.0, 5.0])
    f1 = sigmoid(x1, alpha)
    print ("x1 = {}".format(x1))
    print ("f1 = {}".format(f1))
    
    plt.figure(1)
    plt.clf
    plt.subplot(111)
    plt.title('sigmoid func alpha = %.3f' % alpha)
    plt.plot(x, f)
    colrs = ['r', 'b', 'k', 'y', 'b']

    for i in range(x0.shape[0]):
        # draw horizontal line
        x_ = (x_min, x0[i])
        y_ = (f0[i], f0[i])
        plt.plot(x_, y_, color= colrs[i], linestyle='--')
        # draw vertical line
        x_ = (x0[i], x0[i])
        y_ = (-1, f0[i])
        plt.plot(x_, y_, color= colrs[i], linestyle='--')
    
    plt.savefig('../results/sigmoid_alpha_%.3f.png' % alpha)
    #print('f(200, alpha = {}) = {}'.format(0.2, sigmoid(200.0, 0.2)))
    #print('f(200, alpha = {}) = {}'.format(0.02, sigmoid(200.0, 0.02)))
