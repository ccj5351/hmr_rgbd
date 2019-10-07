# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file: test_call.py
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 09-07-2019
# @last modified: Tue 09 Jul 2019 07:09:07 PM EDT

class Stuff(object):

    def __init__(self, x, y, rge):
        super(Stuff, self).__init__()
        self.x = x
        self.y = y
        self.range = rge

    def __call__(self, x, y):
        self.x = x
        self.y = y
        print '__call__ with (%d,%d)' % (self.x, self.y)

    def __del__(self):
        del self.x
        del self.y
        del self.range
        print ('delete all')


if __name__ == "__main__":

    s = Stuff(1,2,3)
    print (s.x)
    s(7, 8)
    s(14, 10)
