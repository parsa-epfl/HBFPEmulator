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
import torch
import torch.nn as nn
import torch.nn.functional as F
from bfp.bfp_ops import BFPLinear, BFPConv2d, unpack_bfp_args


__all__ = ['wideresnet']


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, bfp_args={}):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = BFPConv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False,
                               **bfp_args)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = BFPConv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False,
                               **bfp_args)
        self.droprate = drop_rate
        self.equal_in_out = (in_planes == out_planes)
        self.conv_shortcut = (not self.equal_in_out) and BFPConv2d(
            in_planes, out_planes, kernel_size=1, stride=stride,
            padding=0, bias=False, **bfp_args) or None

    def forward(self, x):
        if not self.equal_in_out:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equal_in_out else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(
            x if self.equal_in_out else self.conv_shortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(
            self, nb_layers, in_planes, out_planes, block, stride,
            drop_rate=0.0, bfp_args={}):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate,
            bfp_args)

    def _make_layer(
            self, block, in_planes, out_planes,
            nb_layers, stride, drop_rate, bfp_args):
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(i == 0 and in_planes or out_planes, out_planes,
                      i == 0 and stride or 1,
                      drop_rate, bfp_args)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, args, net_depth):
        super(WideResNet, self).__init__()
        bfp_args = unpack_bfp_args(dict(vars(args)))
        # define fundamental parameters.
        self.args = args
        widen_factor = self.args.wideresnet_widen_factor
        drop_rate = self.args.drop_rate

        assert((net_depth - 4) % 6 == 0)
        num_channels = [16, 16 * widen_factor,
                        32 * widen_factor, 64 * widen_factor]
        num_blocks = (net_depth - 4) // 6
        block = BasicBlock
        num_classes = self._decide_num_classes()

        # 1st conv before any network block
        self.conv1 = BFPConv2d(3, num_channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False, **bfp_args)
        # 1st block
        self.block1 = NetworkBlock(num_blocks,
                                   num_channels[0], num_channels[1],
                                   block, 1, drop_rate, bfp_args)
        # 2nd block
        self.block2 = NetworkBlock(num_blocks,
                                   num_channels[1], num_channels[2],
                                   block, 2, drop_rate, bfp_args)
        # 3rd block
        self.block3 = NetworkBlock(num_blocks,
                                   num_channels[2], num_channels[3],
                                   block, 2, drop_rate, bfp_args)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(num_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.num_channels = num_channels[3]
        self.fc = BFPLinear(num_channels[3], num_classes, **bfp_args)

        self._weight_initialization()

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
            elif isinstance(m, BFPLinear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_channels)
        return self.fc(out)


def wideresnet(args):
    net_depth = int(args.arch.replace('wideresnet', ''))
    model = WideResNet(args, net_depth)
    return model
