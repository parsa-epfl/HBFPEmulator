# Copyright (c) 2021, Parallel Systems Architecture Lab, EPFL
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the Parallel Systems Architecture Laboratory, EPFL,
#    nor the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# -*- coding: utf-8 -*-
import math
import torch.nn as nn
from bfp.bfp_ops import BFPLinear, BFPConv2d, unpack_bfp_args

__all__ = ['resnet']


def conv3x3(in_planes, out_planes, stride=1, bfp_args={}):
    "3x3 convolution with padding."
    return BFPConv2d(
        in_channels=in_planes, out_channels=out_planes, kernel_size=3,
        stride=stride, padding=1, bias=False, **bfp_args)


class BasicBlock(nn.Module):
    """
    [3 * 3, 64]
    [3 * 3, 64]
    """
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None,
                 bfp_args={}):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride, bfp_args)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_planes, out_planes, bfp_args=bfp_args)
        self.bn2 = nn.BatchNorm2d(out_planes)

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


class Bottleneck(nn.Module):
    """
    [1 * 1, x]
    [3 * 3, x]
    [1 * 1, x * 4]
    """
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, downsample=None,
                 bfp_args={}):
        super(Bottleneck, self).__init__()
        self.conv1 = BFPConv2d(
            in_channels=in_planes, out_channels=out_planes,
            kernel_size=1, bias=False, **bfp_args)
        self.bn1 = nn.BatchNorm2d(num_features=out_planes)

        self.conv2 = BFPConv2d(
            in_channels=out_planes, out_channels=out_planes,
            kernel_size=3, stride=stride, padding=1, bias=False,
            **bfp_args)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.conv3 = BFPConv2d(
            in_channels=out_planes, out_channels=out_planes * 4,
            kernel_size=1, bias=False, **bfp_args)
        self.bn3 = nn.BatchNorm2d(num_features=out_planes * 4)
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


class ResNetBase(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bfp_args = unpack_bfp_args(dict(vars(args)))

    def _decide_num_classes(self):
        if self.args.data == 'cifar10' or self.args.data == 'svhn':
            return 10
        elif self.args.data == 'cifar100':
            return 100
        elif self.args.data == 'imagenet':
            return 1000

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, BFPConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(mean=0, std=0.01)
            #     m.bias.data.zero_()

    def _make_block(self, block_fn, planes, block_num, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_fn.expansion:
            downsample = nn.Sequential(
                BFPConv2d(self.inplanes, planes * block_fn.expansion,
                          kernel_size=1, stride=stride, bias=False,
                          **self.bfp_args),
                nn.BatchNorm2d(planes * block_fn.expansion),
            )

        layers = []
        layers.append(block_fn(self.inplanes, planes, stride, downsample,
                               bfp_args=self.bfp_args))
        self.inplanes = planes * block_fn.expansion

        for i in range(1, block_num):
            layers.append(block_fn(self.inplanes, planes,
                                   bfp_args=self.bfp_args))
        return nn.Sequential(*layers)


class ResNet_imagenet(ResNetBase):
    def __init__(self, args, resnet_size):
        super(ResNet_imagenet, self).__init__(args)
        self.args = args

        # define model param.
        model_params = {
            18: {'block': BasicBlock, 'layers': [2, 2, 2, 2]},
            34: {'block': BasicBlock, 'layers': [3, 4, 6, 3]},
            50: {'block': Bottleneck, 'layers': [3, 4, 6, 3]},
            101: {'block': Bottleneck, 'layers': [3, 4, 23, 3]},
            152: {'block': Bottleneck, 'layers': [3, 8, 36, 3]},
        }
        block_fn = model_params[resnet_size]['block']
        block_nums = model_params[resnet_size]['layers']

        # decide the num of classes.
        num_classes = self._decide_num_classes()

        # define layers.
        self.inplanes = 64
        self.conv1 = BFPConv2d(
            in_channels=3, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False,
            **self.bfp_args)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_block(
            block_fn=block_fn, planes=64, block_num=block_nums[0])
        self.layer2 = self._make_block(
            block_fn=block_fn, planes=128, block_num=block_nums[1], stride=2)
        self.layer3 = self._make_block(
            block_fn=block_fn, planes=256, block_num=block_nums[2], stride=2)
        self.layer4 = self._make_block(
            block_fn=block_fn, planes=512, block_num=block_nums[3], stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = BFPLinear(
            in_features=512 * block_fn.expansion, out_features=num_classes,
            **self.bfp_args)

        # weight initialization based on layer type.
        self._weight_initialization()

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
        x = self.fc(x)
        return x


class ResNet_cifar(ResNetBase):
    def __init__(self, args, resnet_size):
        super(ResNet_cifar, self).__init__(args)
        self.args = args

        # define model.
        if resnet_size % 6 != 2:
            raise ValueError('resnet_size must be 6n + 2:', resnet_size)
        block_nums = (resnet_size - 2) // 6
        block_fn = Bottleneck if resnet_size >= 44 else BasicBlock

        # decide the num of classes.
        num_classes = self._decide_num_classes()

        # define layers.
        self.inplanes = 16
        self.conv1 = BFPConv2d(
            in_channels=3, out_channels=16,
            kernel_size=3, stride=1, padding=1, bias=False,
            **self.bfp_args)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_block(
            block_fn=block_fn, planes=16, block_num=block_nums)
        self.layer2 = self._make_block(
            block_fn=block_fn, planes=32, block_num=block_nums, stride=2)
        self.layer3 = self._make_block(
            block_fn=block_fn, planes=64, block_num=block_nums, stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = BFPLinear(
            in_features=64 * block_fn.expansion, out_features=num_classes,
            **self.bfp_args)

        # weight initialization based on layer type.
        self._weight_initialization()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet(args):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    resnet_size = int(args.arch.replace('resnet', ''))
    if 'imagenet' in args.data:
        model = ResNet_imagenet(args, resnet_size)
    elif 'cifar' in args.data or 'svhn' in args.data:
        model = ResNet_cifar(args, resnet_size)
    return model
