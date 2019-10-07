# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: test_dd_h5py.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 18-08-2019
# @last modified: Mon 19 Aug 2019 09:25:04 AM EDT
import numpy as np
import h5py


import deepdish as dd


if __name__ == "__main__":
    
    """ h5py for h5 files write and read """
    # > see: https://www.christopherlovell.co.uk/blog/2016/04/27/h5py-intro.html;
    # > see:  https://stackoverflow.com/questions/28170623/how-to-read-hdf5-files-in-python;
    if 0:
        d1 = np.random.random(size = (1000,20))
        d2 = np.random.random(size = (1000,200))
        print d1.shape, d2.shape
        hf = h5py.File('../results/tmp_data.h5', 'w')

        hf.create_dataset('dataset_1', data=d1)
        hf.create_dataset('dataset_2', data=d2)
        hf.close()
    if 0:
        hf = h5py.File('../results/tmp_data.h5', 'r')
        print("hf.keys = {}".format(hf.keys()))
        n1 = hf.get('dataset_1')
        n2 = hf.get('dataset_2')
        print(n1.shape, n2.shape)
        print(type(n1), type(n2))
        n1 = np.array(n1)
        n2 = np.array(n2)
        # close() should be after all the hdf5 variables read 
        hf.close()
        print(n1.shape, n2.shape)
        print(type(n1), type(n2))
        r1 = np.allclose(d1, n1)
        r2 = np.allclose(d2, n2)
        print (r1, r2)
    
    """ dd for h5 files write and read dictionary """
    # > see: http://deepdish.io/2014/11/11/python-dictionary-to-hdf5/
    if 0:
        x  = np.random.random(size = (1000,20))
        y = np.random.random(size = (1000,200))
        z = "hello world"
        dd.io.save('/home/hmr/results/test_dd.h5', {'x': x, 'y': y, 'z': z}, compression=None)

        mydata = dd.io.load('../results/test_dd.h5')
        x1 = mydata['x']
        y1 = mydata['y']
        z1 = mydata['z']
        r1 = np.allclose(x, x1)
        r2 = np.allclose(y, y1)
        print (z1)
        print (r1, r2)
        print (z == z1)

    if 1:
        mydata = dd.io.load('../results/eval-mpjpepa-thred-90mm-5-samples.h5')
        for i in range(len(mydata)):
            print ("i = {}, key = {}".format(i, mydata[i].keys()))
            for j, k in enumerate(mydata[i].keys()):
                print ("j = {}, subkeys = {}".format(j, mydata[i][k].keys()))




