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
import torch.nn as nn


__all__ = ['AlexNetBN', 'alexnet_bn']


class AlexNetBN(nn.Module):
    """An AlexNet with Batch Normalization.
    It uses the architecture from the original paper.
    """

    def __init__(self, num_classes=1000):
        super(AlexNetBN, self).__init__()
        # define functions.
        self.features = nn.Sequential(
            # conv layer 1.
            nn.Conv2d(
                in_channels=3, out_channels=96, kernel_size=11, stride=4,
                padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # conv layer 2.
            nn.Conv2d(
                in_channels=96, out_channels=256, kernel_size=5,
                padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=192),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # conv layer 3.
            nn.Conv2d(
                in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=384),
            nn.Conv2d(
                in_channels=384, out_channels=384, kernel_size=3,
                padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=384),
            # conv layer 4.
            nn.Conv2d(
                in_channels=384, out_channels=256, kernel_size=3,
                padding=1, groups=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=256)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet_bn(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNetBN(**kwargs)
    return model
