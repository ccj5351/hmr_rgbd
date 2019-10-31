# !/usr/bin/env python3
# -*-coding:utf-8-*-
# @file:
# @brief:
# @author: Changjiang Cai, ccai1@stevens.edu, caicj5351@gmail.com
# @version: 0.0.1
# @creation date: 23-10-2019
# @last modified: Wed 30 Oct 2019 03:17:36 PM EDT
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
import sys
#from dollections import OrderedDict

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
    def __init__(self, block, layers, num_classes=None, global_pool = True, 
            isFetchDictForDebug = False):
        self.isFetchDictForDebug = isFetchDictForDebug
        self.inplanes = 64
        self.expansion = 4
        super(ResNet_V2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        # We do not include batch normalization or activation functions in
        # conv1 because the first ResNet unit will perform these. Cf.
        # Appendix of [2].
        #self.bn1 = nn.BatchNorm2d(64)

        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        #Updated to implement 'same' padding in tensorflow; do manually padding to bottom and right, 
        # then apply the follwoing maxpool with padding = 0 as its argument;
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        # padding size: starting from the last dimension and moving forward;
        self.maxpool_pad = (0,1,0,1)# i.e, (padding_left, padding_right, padding_top, padding_bottom)

        self.layer1 = self._make_layer(block,  64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        
        # This is needed because the pre-activation variant does not have batch
        # normalization or activation functions in the residual unit output. See
        # Appendix of [2].
        self.postnorm = nn.BatchNorm2d(512*self.expansion)
        self.relu = nn.ReLU(inplace=True)
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
        #update self.inplanes = output planes, for next incoming Residual block, with new palnes #;
        self.inplanes = planes * expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """ fetch dict """
        fetch_dict = {}
        
        x = self.conv1(x)
        fetch_dict['x_conv1'] = x

        #Updated to implement 'same' padding in tensorflow; do manually padding to bottom and right, 
        # then apply the follwoing maxpool with padding = 0 as its argument;
        x = F.pad(x, pad = self.maxpool_pad, mode = 'constant', value = 0)
        x = self.maxpool(x)
        fetch_dict['x_maxpool'] = x

        x = self.layer1(x)
        fetch_dict['x_layer1'] = x
        x = self.layer2(x)
        fetch_dict['x_layer2'] = x
        x = self.layer3(x)
        fetch_dict['x_layer3'] = x
        x = self.layer4(x)
        fetch_dict['x_layer4'] = x
        x = self.postnorm(x)
        #Updated on 2019/10/30: missing the relu added!!!
        x = self.relu(x)
        fetch_dict['x_postnorm'] = x
        if self.global_pool:
            x = torch.mean(x, dim=[2,3], keepdim = True)
            fetch_dict['x_global_pool'] = x

        if self.fc is not None:
            x = self.fc(torch.flatten(x,1))
        if self.isFetchDictForDebug:
            return x, fetch_dict
        else:
            return x


def resnet_v2_50(num_classes=None, global_pool = True, isFetchDictForDebug = False):
    model = ResNet_V2(Bottleneck_V2, [3,4,6,3],num_classes, global_pool, isFetchDictForDebug)
    return model

def get_tf2pt_key_map_dict():
    map_dict = {
        '' : '',
        # for root block: conv1 --> pool1
        # that is: input x --> (conv1 --> pool1 )--> (residual-block1,2,3,4) --> postnorm --> global avg-pool --> output
        'conv1/weights' : 'conv1.weight',
        'conv1/biases' : 'conv1.bias',
        # for post norm:
        'postnorm/beta': 'postnorm.bias',
        'postnorm/gamma': 'postnorm.weight',
        'postnorm/moving_mean': 'postnorm.running_mean',
        'postnorm/moving_variance': 'postnorm.running_var',
    }

    """ block 1, has 3 unites """
    """ block 2, has 4 unites """
    """ block 3, has 6 unites """
    """ block 4, has 3 unites """
    # processing tf_key_1
    blks = [(1,3), (2,4), (3,6), (4,3)]
    for t in blks:
        b_idx = t[0]
        for u_idx in range(t[1]):
            key = 'block{}/unit_{}'.format(b_idx, u_idx + 1)
            vaule = 'layer{}.{}'.format(b_idx, u_idx )
            map_dict[key] = vaule
    
    # processing tf_key_2
    #Example: (tf_key, pt_key)
    """ In each bottleneck block: we have the following: """
    bottleneck_tf_pt_tuples = [
            # Note: 'resnet_v2_50/block1/unit_1/bottleneck_v2/preact/beta/Adam':
            # 'Adam' is related to Adam Optimization, so here we do not use it!!!
            # Pre-Act: bn0"""
            # BN: out = gamma * X_norm + beta, so beta is bias, gamma is weight;
            ['preact/gamma','bn0.weight'],
            ['preact/beta', 'bn0.bias'],
            ['preact/moving_mean', 'bn0.running_mean'],
            ['preact/moving_variance', 'bn0.running_var'],
            #conv1 + bn1 + relu1
            ['conv1/weights', 'conv1.weight'],
            ['conv1/BatchNorm/gamma', 'bn1.weight'],
            ['conv1/BatchNorm/beta', 'bn1.bias'],
            ['conv1/BatchNorm/moving_mean', 'bn1.running_mean'],
            ['conv1/BatchNorm/moving_variance', 'bn1.running_var'],
            #conv2 + bn2 + relu2
            ['conv2/weights', 'conv2.weight'],
            ['conv2/BatchNorm/gamma', 'bn2.weight'],
            ['conv2/BatchNorm/beta', 'bn2.bias'],
            ['conv2/BatchNorm/moving_mean', 'bn2.running_mean'],
            ['conv2/BatchNorm/moving_variance', 'bn2.running_var'],
            #conv3
            ['conv3/weights', 'conv3.weight'],
            ['conv3/biases', 'conv3.bias'],

            #shortcut
            ['shortcut/weights', 'shortcut.weight'],
            ['shortcut/biases',  'shortcut.bias'],
        ]
    for cur_tuple in bottleneck_tf_pt_tuples:
        map_dict[cur_tuple[0]] = cur_tuple[1]
    #print (map_dict)
    return map_dict

def map_tf_dictKeys_2PyTorch_dictKeys( map_dict,
    tf_key = 'resnet_v2_50/block1/unit_1/bottleneck_v2/conv1/BatchNorm/beta'):
    # E.g.:
    # tf_key = 'resnet_v2_50/block1/unit_1/bottleneck_v2/conv1/BatchNorm/beta'
    # or tf_key = 'resnet_v2_50/conv1/biases'
    # 1) skip the first part : 'resnet_v2_50'
    tf_key = tf_key[len('resnet_v2_50')+1:]
    # 2) find 'bottleneck_v2' if exists, and pick the part before and after 'bottleneck_v2'
    pos = tf_key.find('bottleneck_v2')
    
    if pos > 0: # if found 'bottleneck_v2'
        tf_key_1, tf_key_2 = tf_key[0:pos-1], tf_key[pos+1+len('bottleneck_v2'):]
    else: # no found 'bottleneck_v2'
        tf_key_1, tf_key_2 = '', tf_key
    
    # processing tf_key_1
    #print (tf_key_1)
    pt_key_1 = map_dict[tf_key_1]
    #print (pt_key_1)
    #print (tf_key_2)
    pt_key_2 = map_dict[tf_key_2]
    #print (pt_key_2)
    if pt_key_1 == '':
        pt_key = pt_key_2
    else:
        pt_key = pt_key_1 + '.' + pt_key_2
    #print ("[***] {} --> {}".format(tf_key, pt_key))
    return pt_key
    


#>see https://stackoverflow.com/questions/51628607/pytorch-passing-numpy-array-for-weight-initialization
def set_resnet_parameter_data(layer, parameter_name, new_torch_data):
    param = getattr(layer, parameter_name)
    param.data = new_torch_data

def pass_np_model_state_to_resnet(src_np_model_state_dict, dst_resnet_model):
    map_dict = get_tf2pt_key_map_dict()
    dst_state_dict = dst_resnet_model.state_dict()
    n_valid = 0
    n_adam = 0
    tf_var_names = list(src_np_model_state_dict['resnet_v2_50_names'])
    N = len(tf_var_names)

    for tf_key in sorted(src_np_model_state_dict.keys()):
        # Note: 'resnet_v2_50/block1/unit_1/bottleneck_v2/preact/beta/Adam':
        # 'Adam' is related to Adam Optimization, so here we do not use it!!!
        param = src_np_model_state_dict[tf_key]
        if 'Adam' in tf_key:
            #print('Adam! {} is only for Adam Optimization, not uesed here!!'.format(tf_key))
            n_adam += 1
            tf_var_names.remove(tf_key)
            continue
        elif 'resnet_v2_50_names' == tf_key:
            continue
        pt_key = map_tf_dictKeys_2PyTorch_dictKeys(map_dict, tf_key)
        if pt_key not in dst_state_dict:
            print('unexpected ', pt_key, ' !')
            continue
        if not isinstance(param, np.ndarray):
            raise ValueError('Expected a np.ndarray')
        else:
            # !!! Note: added by CCJ on 2019/10/24;
            # tensorflow conv2d weight in size of [kernel_size[0], kernel_size[1], in_channels, out_channels], 
            # e.g., weight in size [7,7,3,64] means applying 7x7-kernel-size convolution to input image with 3 channel 
            # and output channel is 64;
            # While, PyTorch will have its weight in shape [out_channels, in_channels/groups, kernel_size[0], kernel_size[1]], 
            # here we assume gropus = 1; 
            if param.ndim == 4:
                param = np.transpose(param, [3,2,0,1])
            param = torch.from_numpy(param).contiguous()
        try:
            dst_state_dict[pt_key].copy_(param)
            n_valid += 1
            tf_var_names.remove(tf_key)
        except:
            print(pt_key, ' is inconsistent!')
            print ('src np.ndarray in shape {}, dst tensor in shape {}'.format(param.shape, 
                    dst_state_dict[pt_key].shape))
            n_valid -= 1
            tf_var_names.append(tf_key)
            continue
    
    
    
    print('%d out of %d variables processed! Wherein:'%(n_valid + n_adam, N))
    print(' [***] Copyed state dict for %d variables and finished!' %n_valid)
    print(' [***] Skip %d adam variables, which are related to Adam optimaization state' %(n_adam))
    print(' [***] {} variables are left unprocessed!'.format(len(tf_var_names)))
    if n_valid + n_adam == N:
        print (" [***] Resnet_V2_50 loading Numpy weights Succeed!!!")
    else:
        print (" [***] Resnet_V2_50 loading Numpy weights Failed !!!")
    #print('[***] Including: ', tf_var_names)


def load_Res50ModelFromNpyFile(npy_file = '/home/ccj/hmr-rgbd/results/saved_weights/hmr_pre_trained_resnet_v2_50.npy'):
    dst_resnet_model = resnet_v2_50()
    assert (npy_file is not None)
    # this npy file is generated by Python2, due to Tensorflow is installed in Python2;
    # load this npy file (generated by Python2) to Python3, due to PyTorch is installed in Python3;
    src_np_model_state_dict = np.load(npy_file, allow_pickle= True, encoding = 'latin1').item()
    #tmp_name = 'resnet_v2_50/block4/unit_3/bottleneck_v2/conv2/weights'
    # check the variable dimensionality
    # print should be : [3, 3, 512, 512];
    #print(src_np_model_state_dict[tmp_name].shape)
    
    pass_np_model_state_to_resnet(src_np_model_state_dict, dst_resnet_model)
    return dst_resnet_model


if __name__ == '__main__':
    
    if 0:
        print ('resnet_v2_50 state_dict():')
        n = 0
        for k,v in resnet_v2_50().state_dict().items():
            print (k, v.shape)
            n += 1
        print (n)
    
    if 0:    
        """ load dictionary """ 
        npy_file = '/home/ccj/hmr-rgbd/results/saved_weights/hmr_pre_trained_resnet_v2_50.npy'
        resnet_dict2 = np.load(npy_file, allow_pickle= True, encoding = 'latin1').item()
        print ('loaded var_names : ', resnet_dict2['resnet_v2_50_names'])
        tmp_name = 'resnet_v2_50/block4/unit_3/bottleneck_v2/conv2/weights'
        # check the variable dimensionality
        # print should be : [3, 3, 512, 512];
        print (resnet_dict2[tmp_name].shape)
    
    """ load numpy dictionary to Pytorch model and save the model""" 
    if 1:
        # this npy file is generated by Python2, due to Tensorflow is installed in Python2;
        npy_file = '/home/ccj/hmr-rgbd/results/saved_weights/hmr_pre_trained_resnet_v2_50.npy'
        # load this npy file (generated by Python2) to Python3, due to PyTorch is installed in Python3;
        dst_resnet_model = load_Res50ModelFromNpyFile(npy_file)
        dst_state_dict = dst_resnet_model.state_dict()

        model_path = '/home/ccj/hmr-rgbd/results/saved_weights/hmr_pre_trained_resnet_v2_50.pt'
        torch.save(dst_state_dict, model_path)
        print ('saved %s' % model_path)
        #n = 0
        #for k,v in dst_state_dict.items():
        #    print (k, v.shape)
        #    n += 1
        #print (n)
    if 1:
        # get a new model
        resnet_v2_50 = resnet_v2_50()
        model_path = '/home/ccj/hmr-rgbd/results/saved_weights/hmr_pre_trained_resnet_v2_50.pt'
        # load the weights
        resnet_v2_50.load_state_dict(torch.load(model_path))
        print ('Loading %s' % model_path)
