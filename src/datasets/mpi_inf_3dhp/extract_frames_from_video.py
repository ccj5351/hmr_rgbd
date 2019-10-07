# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: extract_frames_from_video.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 05-06-2019
# @last modified: Wed 05 Jun 2019 03:05:26 PM EDT

# > see https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames

import cv2
from os.path import join, exists
from os import makedirs

if __name__ == '__main__':
    #base_dir= '/mnt/interns/changjiang/PatientPositioning/Datasets/mpi_inf_3dhp/'
    #NOTE: mount dir with docker according to the following:
    base_dir= '/home/hmr/datasets/mpi_inf_3dhp/'
    sub_ids = range(2, 9) 
    seq_ids = range(1, 3)
    cam_ids = [0, 1, 2, 4, 5, 6, 7, 8]
    for sub_id in sub_ids:
        for seq_id in seq_ids:
            print('collecting S%d, Seq%d' % (sub_id, seq_id))
            data_dir = join(base_dir, 'S%d' % sub_id, 'Seq%d' % seq_id, 'imageSequence')
            for cam_id in cam_ids:
                out_dir = join(data_dir, 'video_%d' % cam_id)
                if not exists(out_dir):
                    makedirs(out_dir)  
                    print ('mkdirs %s' % out_dir)
                src_video = join(data_dir, 'video_%d.avi' % cam_id)
                print('reading video %s' % src_video)

                vidcap = cv2.VideoCapture(src_video)
                success, image = vidcap.read()
                count = 0
                while success:
                    cv2.imwrite(join(out_dir, "frame_%06d.jpg" % count), image) # save frame as JPEG file
                    success, image = vidcap.read()
                    if count % 500 == 0:
                        print('Read a new frame: %6d' %count, success)
                    count += 1
                    if count > 999999:
                        success = False
