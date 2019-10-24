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
def conv3x3(cin, cout, stride=1, groups=1):
    return nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False, 
                     groups=groups)


def conv1x1(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=False)

# bottleneck_v2
# x-->BN --> ReLU-->(conv1, BN, ReLU)-->(conv2, BN, ReLU) --> conv3
# |                                                             |
# |                                                             |
# |                                                             |
# |--------------------------------------------> Addition --> x_new
class Bottleneck_V2(nn.Module):
    def __init__(self, cin, cout=None, cmid=None, stride=1, downsample=None):
        super(Bottleneck_V2, self).__init__()
        cout = cout or cin
        cmid = cmid or cout//4
   
        self.relu = nn.ReLU(inplace=True)
        """ Pre Act """
        self.preact = nn.BatchNorm2d(cin)
        
        """ (conv1, BN, ReLU)"""
        self.conv1 = conv1x1(cin, cmid) #conv1
        self.bn1 = nn.BatchNorm2d(cmid) #conv1/BatchNorm
        
        """ (conv2, BN, ReLU)"""
        self.conv2 = conv3x3(cmid, cmid, stride) #conv2
        self.bn2 = nn.BatchNorm2d(cmid) #conv2/BatchNorm
        self.conv3 = conv1x1(cmid, cout)

        self.stride = stride
        self.downsample = downsample
        if (stride != 1 or cin != cout) and downsample in (True, None):
            self.downsample = conv1x1(cin, cout, stride)

    def forward(self, x):
        # Conv'ed branch
        out = x
        out = self.conv1(self.relu(self.bn1(out)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))

        # Residual branch
        residual = x
        if self.downsample is not None:
            residual = self.downsample(residual)

        return out + residual

    def reset_parameters(self):
        # Following https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua#L129
        nn.init.kaiming_normal(self.conv1.weight, a=0, mode='fan_out')
        nn.init.kaiming_normal(self.conv2.weight, a=0, mode='fan_out')
        nn.init.kaiming_normal(self.conv3.weight, a=0, mode='fan_out')

        # Not the default =(
        nn.init.constant(self.bn1.weight, 1)
        nn.init.constant(self.bn2.weight, 1)
        nn.init.constant(self.bn3.weight, 1)
        return self


# bottleneck_v2
# x-->BN --> ReLU-->(conv1, BN, ReLU)-->(conv2, BN, ReLU)-->(conv3,BN,ReLU)
# |                                                                |
# |                                                                |
# |                                                                |
# |-----------------------------------------------> Addition --> x_new
class Bottleneck_V2(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck_V2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_V2(nn.Module):
    def __init__(self, block, layers, num_classes=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        # We do not include batch normalization or activation functions in
        # conv1 because the first ResNet unit will perform these. Cf.
        # Appendix of [2].
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,  64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # output is of size 1 x 1 here;
        
        #Note: in HMR project, we set `num_classes=None`;
        if num_classes is not None:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        
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


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
       # x = self.fc(x)

        return x




if __name__ == '__main__':
    encoder = load_Res50Model()
    print ('resnet-50 state_dict():')
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
