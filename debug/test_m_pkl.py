# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: test_m_pkl.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 09-07-2019
# @last modified: Tue 09 Jul 2019 10:17:49 PM EDT

import cPickle as pickle
import numpy as np

pkl_fname = '../models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'

with open(pkl_fname, 'r') as f:
     dd = pickle.load(f)
print (dd.keys())
J_regressor = dd['J_regressor'].T.todense()
print ("J_regressor shape = {}, {}".format(J_regressor.shape, np.sum(J_regressor[:,:])))


 
