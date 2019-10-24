# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file:
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 23-10-2019
# @last modified: Wed 23 Oct 2019 01:13:22 PM EDT
"""
    file:   ResnetV2.py
    author: Changjiang Cai
    mark:   adopted from:
            1) pytorch source code, and 
            2) and https://github.com/MandyMo/pytorch_HMR.git
            3) and https://github.com/lucasb-eyer/lbtoolbox/blob/master/lbtoolbox/pytorch.py#L61;
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
import torch.optim as optim
import numpy as np
import math
import torchvision


"""Contains definitions for the preactivation form of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer.

"""

########################################
# Kaiming's blocks
########################################
def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, 
                     groups=groups)


def conv1x1(cin, cout, stride=1,bias=False):
    return nn.Conv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)

# bottleneck_v2
# x-->BN --> ReLU-->(conv1, BN, ReLU)-->(conv2, BN, ReLU) --> conv3
# |                                                             |
# |                                                             |
# |                                                             |
# |--------------------------------------------> Addition --> x_new
class Bottleneck_V2(nn.Module):
    expansion = 4
    def __init__(self, cin, cout, stride):
        super(Bottleneck_V2, self).__init__()
        cmid = cout// self.expansion
   
        self.relu = nn.ReLU(inplace=True)
        """ Pre Act """
        self.bn0 = nn.BatchNorm2d(cin)
        
        """ (conv1, BN, ReLU)"""
        self.conv1 = conv1x1(cin, cmid, bias=False) #conv1
        self.bn1 = nn.BatchNorm2d(cmid) #conv1/BatchNorm
        
        """ (conv2, BN, ReLU)"""
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False) #conv2
        self.bn2 = nn.BatchNorm2d(cmid) #conv2/BatchNorm
        """ (conv3 )"""
        self.conv3 = conv1x1(cmid, cout, bias=True) # conv3

        self.stride = stride
        self.maxpool2d= nn.MaxPool2d(kernel_size=1, stride = stride)
        self.shortcut = None
        if cin != cout:
            # conv, 1 x 1
            self.shortcut = conv1x1(cin, cout, stride, bias = True)

    def forward(self, x):
        """ Pre Act """
        preact = self.relu(self.bn0(x))
        if self.shortcut is not None:
            shortcut = self.shortcut(preact) # e.g., stride = 2
        else:
            shortcut = self.maxpool2d(x)
        
        """ (conv1, BN, ReLU)"""
        residual = self.relu(self.bn1(self.conv1(preact)))
        """ (conv2, BN, ReLU)"""
        residual = self.relu(self.bn2(self.conv2(residual)))
        """ (conv3 )"""
        residual = self.conv3(residual)
        output = shortcut + residual
        return output


class ResNet_V2(nn.Module):
    def __init__(self, block, layers, num_classes=None, global_pool = True):
        self.inplanes = 64
        super(ResNet_V2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        # We do not include batch normalization or activation functions in
        # conv1 because the first ResNet unit will perform these. Cf.
        # Appendix of [2].
        #self.bn1 = nn.BatchNorm2d(64)
        #self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block,  64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # output is of size 1 x 1 here;
        self.global_pool = global_pool
        #Note: in HMR project, we set `num_classes=None`;
        if num_classes is not None:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.fc = None
        
        #leave it here FYI:
        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #        m.weight.data.normal_(0, math.sqrt(2. / n))
        #    elif isinstance(m, nn.BatchNorm2d):
        #        m.weight.data.fill_(1)
        #        m.bias.data.zero_()
        
        # the new version is shown below:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    #def __init__(self, cin, cout, stride=1):

    def _make_layer(self, block, planes, numBlocks, stride):
        expansion = block.expansion
        layers = []
        for i in range(0, numBlocks):
            cur_inplanes = planes * expansion if i > 0 else self.inplanes
            tmp_stride = 1 if i < (numBlocks - 1) else stride
            layers.append(block(cur_inplanes, planes*expansion, tmp_stride))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.global_pool:
            x = torch.mean(x, dim=[2,3], keepdim = True)
        if self.fc is not None:
            x = self.fc(torch.flatten(x,1))
        return x


def resnet_v2_50(num_classes=None, global_pool = True):
    model = ResNet_V2(Bottleneck_V2, [3,4,6,3],num_classes, global_pool)
    return model


if __name__ == '__main__':
    encoder = resnet_v2_50()
    print ('resnet_v2_50 state_dict():')
    n = 0
    for k,v in encoder.state_dict().items():
        print (k, v.shape)
        n += 1
    print (n)
    #vx = torch.autograd.Variable(torch.from_numpy(np.array([1, 1, 1])))
    #vy = torch.autograd.Variable(torch.from_numpy(np.array([2, 2, 2])))
    #vz = torch.cat([vx, vy], 0)
    #vz[0] = 100
    #print(vz)
    #print(vx)
