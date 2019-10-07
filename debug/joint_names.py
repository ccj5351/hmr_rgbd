# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: joint_names.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 29-05-2019
# @last modified: Wed 29 May 2019 10:39:37 AM EDT
joint_names = [
        'R Ankle', 'R Knee', 'R Hip', 'L Hip', 'L Knee', 'L Ankle', 'R Wrist',
        'R Elbow', 'R Shoulder', 'L Shoulder', 'L Elbow', 'L Wrist', 'Neck',
        'Head', 'Nose', 'L Eye', 'R Eye', 'L Ear', 'R Ear' ]

print len(joint_names)
R_ank = joint_names.index('R Ankle')
L_ank = joint_names.index('L Ankle')
print R_ank, L_ank
