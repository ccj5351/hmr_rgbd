# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: hstack_test.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 09-06-2019
# @last modified: Sun 09 Jun 2019 12:26:19 AM EDT

import numpy as np

if __name__ == '__main__':
    batch_size = 4
    num2show = np.minimum(6, batch_size)
    tmp = np.hstack( [np.arange(num2show / 2), batch_size - np.arange(3) - 1])
    print tmp
