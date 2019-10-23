# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file:
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 23-10-2019
# @last modified: Wed 23 Oct 2019 01:13:22 PM EDT

'''
    file:   ResnetV2.py
    author: Changjiang Cai
    mark:   adopted from:
            1) pytorch source code, and 
            2) and https://github.com/MandyMo/pytorch_HMR.git
            3) and https://github.com/lucasb-eyer/lbtoolbox/blob/master/lbtoolbox/pytorch.py#L61;
'''

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
import torch.optim as optim
import numpy as np
import math
import torchvision

class ResNet_V2_50(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
       # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def copy_parameter_from_resnet50(model, res50_dict):
    cur_state_dict = model.state_dict()
    for name, param in list(res50_dict.items())[0:None]:
        if name not in cur_state_dict: 
            print('unexpected ', name, ' !')
            continue 
        if isinstance(param, Parameter): 
            param = param.data
        try:
            cur_state_dict[name].copy_(param)
        except:
            print(name, ' is inconsistent!')
            continue
    print('copy state dict finished!')

def load_Res50Model():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    copy_parameter_from_resnet50(model, torchvision.models.resnet50(pretrained = True).state_dict())
    return model




###############################################################################
# Kaiming blocks


def conv3x3(cin, cout, stride=1, groups=1):
    return nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)


def conv1x1(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=False)


class PreActBlock(nn.Module):
    """
    Follows the implementation of "Identity Mappings in Deep Residual Networks" here:
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
    """
    def __init__(self, cin, cout=None, cmid=None, stride=1, downsample=None, dropout=None):
        super(PreActBlock, self).__init__()
        cout = cout or cin
        cmid = cmid or cout

        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = conv3x3(cin, cmid, stride)
        self.bn2 = nn.BatchNorm2d(cmid)
        self.conv2 = conv3x3(cmid, cout)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride
        self.downsample = downsample
        if (stride != 1 or cin != cout) and downsample in (True, None):
            self.downsample = conv1x1(cin, cout, stride)

        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        # Conv'ed branch
        out = x
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv1(self.relu(self.bn1(out)))
        out = self.conv2(self.relu(self.bn2(out)))

        # Residual branch
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        return out + residual

    def reset_parameters(self):
        # Following https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua#L129
        nn.init.kaiming_normal(self.conv1.weight, a=0, mode='fan_out')
        nn.init.kaiming_normal(self.conv2.weight, a=0, mode='fan_out')

        # Not the default =(
        nn.init.constant(self.bn1.weight, 1)
        nn.init.constant(self.bn2.weight, 1)
        return self


class PreActBottleneck(nn.Module):
    """
    Follows the implementation of "Identity Mappings in Deep Residual Networks" here:
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
    Except it puts the stride on 3x3 conv when available.
    """
    def __init__(self, cin, cout=None, cmid=None, stride=1, downsample=None, dropout=None):
        super(PreActBottleneck, self).__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = conv1x1(cin, cmid)
        self.bn2 = nn.BatchNorm2d(cmid)
        self.conv2 = conv3x3(cmid, cmid, stride)  # Original code has it on conv1!!
        self.bn3 = nn.BatchNorm2d(cmid)
        self.conv3 = conv1x1(cmid, cout)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride
        self.downsample = downsample
        if (stride != 1 or cin != cout) and downsample in (True, None):
            self.downsample = conv1x1(cin, cout, stride)

        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        # Conv'ed branch
        out = x
        if self.dropout is not None:
            out = self.dropout(out)
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


class NeXtBlockC(nn.Module):
    """
    Follows the implementation of "Aggregated Residual Transformations for Deep Neural Networks" here:
    https://github.com/facebookresearch/ResNeXt
    """
    def __init__(self, cin, cout=None, cmid=None, stride=1, downsample=None):
        """
        Now, cmid is (C, D) which means C convolutions on D channels in the bottleneck.
        C == cardinality.
        """
        super(NeXtBlockC, self).__init__()
        cout = cout or cin
        C, D = cmid if isinstance(cmid, tuple) else ((cmid, cout//cmid//2) if cmid is not None else (4, cout//8))

        self.conv1 = conv1x1(cin, C*D)
        self.bn1 = nn.BatchNorm2d(C*D)
        self.conv2 = conv3x3(C*D, C*D, groups=C, stride=stride)
        self.bn2 = nn.BatchNorm2d(C*D)
        self.conv3 = conv1x1(C*D, cout)
        self.bn3 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride
        self.downsample = downsample
        if (stride != 1 or cin != cout) and downsample in (True, None):
            self.downsample = nn.Sequential(OrderedDict([
                ('conv', conv1x1(cin, cout, stride)),
                ('bn', nn.BatchNorm2d(cout))
            ]))
            # TODO: They now optionally use strided, 0-padded identity (abusing avgpool) in the code.

    def forward(self, x):
        # Conv'ed branch
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        # Residual branch
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        return self.relu(out + residual)

    def reset_parameters(self):
        # Following https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua#L208
        nn.init.kaiming_normal(self.conv1.weight, a=0, mode='fan_out')
        nn.init.kaiming_normal(self.conv2.weight, a=0, mode='fan_out')
        nn.init.kaiming_normal(self.conv3.weight, a=0, mode='fan_out')

        # Not the default =(
        nn.init.constant(self.bn1.weight, 1)
        nn.init.constant(self.bn2.weight, 1)
        nn.init.constant(self.bn3.weight, 1)
        return self


class PreActNeXtBlockC(nn.Module):
    """
    My own "pre-activated" version of the ResNeXt block C.
    """
    def __init__(self, cin, cout=None, cmid=None, stride=1, downsample=None):
        """
        Now, cmid is (C, D) which means C convolutions on D channels in the bottleneck.
        C == cardinality.
        """
        super(NeXtBlockC, self).__init__()
        cout = cout or cin
        C, D = cmid if isinstance(cmid, tuple) else ((cmid, cout//cmid//2) if cmid is not None else (4, cout//8))

        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = conv1x1(cin, C*D)
        self.bn2 = nn.BatchNorm2d(C*D)
        self.conv2 = conv3x3(C*D, C*D, groups=C, stride=stride)
        self.bn3 = nn.BatchNorm2d(C*D)
        self.conv3 = conv1x1(C*D, cout)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride
        self.downsample = downsample
        if (stride != 1 or cin != cout) and downsample in (True, None):
            self.downsample = nn.Sequential(OrderedDict([
                ('bn', nn.BatchNorm2d(cin)),
                ('relu', nn.ReLU(inplace=True)),
                ('conv', conv1x1(cin, cout, stride)),
            ]))
            # TODO: They now optionally use strided, 0-padded identity (abusing avgpool) in the code.

    def forward(self, x):
        # Conv'ed branch
        out = x
        out = self.conv1(self.relu(self.bn1(out)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))

        # Residual branch
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        return out + residual

    def reset_parameters(self):
        # Following https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua#L208
        nn.init.kaiming_normal(self.conv1.weight, a=0, mode='fan_out')
        nn.init.kaiming_normal(self.conv2.weight, a=0, mode='fan_out')
        nn.init.kaiming_normal(self.conv3.weight, a=0, mode='fan_out')

        # Not the default =(
        nn.init.constant(self.bn1.weight, 1)
        nn.init.constant(self.bn2.weight, 1)
        nn.init.constant(self.bn3.weight, 1)
        return self




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
