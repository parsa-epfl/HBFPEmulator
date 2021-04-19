# Copyright (c) 2021, Parallel Systems Architecture Laboratory (PARSA), EPFL & 
# Machine Learning and Optimization Laboratory (MLO), EPFL. All rights reserved.
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
# 3. Neither the name of the PARSA, EPFL & MLO, EPFL
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
"""Auxiliary functions that support for system."""
import time
import torch
from datetime import datetime

def str2time(string, pattern):
    """convert the string to the datetime."""
    return datetime.strptime(string, pattern)


def determine_model_info(args):
    if 'resnet' in args.arch:
        return args.arch
    elif 'densenet' in args.arch:
        return '{}{}-{}{}'.format(
            args.arch, args.densenet_growth_rate,
            'BC-' if args.densenet_bc_mode else '',
            args.densenet_compression
        )
    elif 'wideresnet' in args.arch:
        return '{}-{}'.format(
            args.arch, args.wideresnet_widen_factor
        )


def info2path(args):
    info = '{}_{}_'.format(int(time.time()), determine_model_info(args))
    info += 'lr-{}_momentum-{}_epochs-{}_batchsize-{}_droprate-{}_'.format(
        args.lr,
        args.momentum,
        args.num_epochs,
        args.batch_size,
        args.drop_rate)
    info += 'lars-{}-{}_'.format(args.lr_lars_mode, args.lr_lars_eta) \
        if args.lr_lars else ''

    if args.num_format != 'fp32':
        info += '{}-{}_mant_bits-{}_tilesize-{}'.format(
            args.num_format, args.rounding_mode,
            args.mant_bits, args.bfp_tile_size)
        return info
    else:
        return info


def zeros_like(x):
    assert x.__class__.__name__.find('Variable') != -1 or \
        x.__class__.__name__.find('Tensor') != -1, \
        "Object is neither a Tensor nor a Variable"

    y = torch.zeros(x.size())
    y = y.to(x.device)

    if x.__class__.__name__ == 'Variable':
        return torch.autograd.Variable(y, requires_grad=x.requires_grad)
    elif x.__class__.__name__.find('Tensor') != -1:
        return y
