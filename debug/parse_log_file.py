# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: parse_log_file.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 15-07-2019
# @last modified: Mon 15 Jul 2019 01:29:09 PM EDT


import numpy as np

import matplotlib.pyplot as plt

mylines = []
with open("/home/hmr/results/densepose_resnet18_log_01.txt") as f:
    for line in f:
        if 'json_stats' in line:
            mylines.append(line.rstrip())

print (mylines[:1])

itrs = []
losses = []
accus = []

for i in range(len(mylines)):
#for i in range(5):
    l = mylines[i]
    itr = l[l.find('iter') + 7:].split(',')[0]
    itrs.append(int(itr))
    
    loss = l[l.find('loss') + 7:].split(',')[0]
    losses.append(float(loss))
    
    accu = l[l.find('accuracy_cls') + 15 :].split(',')[0]
    accus.append(float(accu))
    #print accu


plt.figure(1)
plt.clf()
plt.subplot(211)

plt.plot(np.array(itrs), np.array(losses))

plt.subplot(212)
plt.plot(np.array(itrs), np.array(accus))
plt.savefig('./loss_accu.png')
