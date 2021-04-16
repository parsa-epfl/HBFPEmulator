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
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from bfp.bfp_ops import BFPLinear, BFPConv2d, unpack_bfp_args

__all__ = ['densenet']


class BasicLayer(nn.Module):
    def __init__(self, num_channels, growth_rate, drop_rate=0.0,
                 bfp_args={}):
        super(BasicLayer, self).__init__()

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = BFPConv2d(
            num_channels, growth_rate, kernel_size=3, padding=1, bias=False,
            **bfp_args)
        self.droprate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        out = torch.cat((x, out), 1)
        return out


class Bottleneck(nn.Module):
    def __init__(self, num_channels, growth_rate, drop_rate=0.0,
                 bfp_args={}):
        super(Bottleneck, self).__init__()

        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = BFPConv2d(
            num_channels, inter_channels, kernel_size=1, bias=False,
            **bfp_args)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = BFPConv2d(
            inter_channels, growth_rate, kernel_size=3, padding=1, bias=False,
            **bfp_args)
        self.droprate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(
                out, p=self.droprate, inplace=False, training=self.training)

        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(
                out, p=self.droprate, inplace=False, training=self.training)

        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, num_channels, num_out_channels, drop_rate=0.0,
                 bfp_args={}):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = BFPConv2d(
            num_channels, num_out_channels, kernel_size=1, bias=False,
            **bfp_args)

        self.droprate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(
                out, p=self.droprate, inplace=False, training=self.training)

        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, args, net_depth):
        super(DenseNet, self).__init__()

        # determine some fundamental configurations.
        self.args = args
        bfp_args = unpack_bfp_args(dict(vars(args)))
        growth_rate = self.args.densenet_growth_rate
        bc_mode = self.args.densenet_bc_mode
        compression = self.args.densenet_compression
        num_classes = self._decide_num_classes()
        drop_rate = self.args.drop_rate
        is_small_inputs = 'imagenet' not in self.args.data
        self.avgpool_size = 8 if is_small_inputs else 7
        assert 0 < compression <= 1, 'compression should be between 0 and 1.'

        # determine block_config for different types of the data.
        if is_small_inputs:
            num_blocks = 3
            num_layers_per_block = (net_depth - (num_blocks + 1)) // num_blocks

            if bc_mode:
                num_layers_per_block = num_layers_per_block // 2
            block_config = [num_layers_per_block] * num_blocks
        else:
            model_params = {
                121: [6, 12, 24, 16],
                169: [6, 12, 32, 32],
                201: [6, 12, 48, 32],
                264: [6, 12, 64, 48]
            }

            assert net_depth not in model_params.keys()
            block_config = model_params[net_depth]

        # init conv.
        num_channels = 2 * growth_rate
        if is_small_inputs:
            self.features = nn.Sequential(
                OrderedDict([
                    ('conv0', BFPConv2d(3, num_channels, kernel_size=3,
                                        stride=1, padding=1, bias=False,
                                        **bfp_args))
                ])
            )
        else:
            self.features = nn.Sequential(
                OrderedDict([
                    ('conv0', BFPConv2d(3, num_channels, kernel_size=7,
                                        stride=2, padding=3, bias=False,
                                        **bfp_args)),
                    ('norm0', nn.BatchNorm2d(num_channels)),
                    ('relu0', nn.ReLU(inplace=True)),
                    ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                           ceil_mode=False))
                ])
            )

        # each denseblock
        for ind, num_layers in enumerate(block_config):
            block = self._make_dense(
                num_channels, growth_rate, num_layers,
                bc_mode, drop_rate,
                bfp_args)
            self.features.add_module('denseblock%d' % (ind + 1), block)

            num_channels += num_layers * growth_rate
            num_out_channels = int(math.floor(num_channels * compression))

            # transition_layer
            if ind != len(block_config) - 1:
                trans = Transition(num_channels, num_out_channels, drop_rate,
                                   bfp_args)
                self.features.add_module('transition%d' % (ind + 1), trans)
                num_channels = num_out_channels

        # final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_channels))

        # Linear layer
        self.classifier = BFPLinear(num_channels, num_classes, bfp_args)

        # init weight.
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

    def _make_dense(
            self, num_channels, growth_rate, num_layers_per_block,
            bc_mode, drop_rate, bfp_args):
        layers = []
        for i in range(int(num_layers_per_block)):
            if bc_mode:
                layers.append(
                    Bottleneck(num_channels, growth_rate, drop_rate,
                               bfp_args))
            else:
                layers.append(
                    BasicLayer(num_channels, growth_rate, drop_rate,
                               bfp_args))
            num_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(
            out, kernel_size=self.avgpool_size).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def densenet(args):
    net_depth = int(args.arch.replace('densenet', ''))
    model = DenseNet(args, net_depth)
    return model
