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
from os.path import join, isfile

import torch
import torch.nn as nn

import cnn.models as models
from bfp.bfp_ops import unpack_bfp_args
from cnn.optim.sgd import BFPSGD
from cnn.utils.opfiles import remove_folder


def create_model(args):
    """Create model, criterion and optimizer.
    If args.use_cuda is True, use ps_id as GPU_id.
    """
    print("=> creating model '{}'".format(args.arch))
    if 'wideresnet' in args.arch:
        model = models.__dict__['wideresnet'](args)
    elif 'resnet' in args.arch:
        model = models.__dict__['resnet'](args)
    elif 'densenet' in args.arch:
        model = models.__dict__['densenet'](args)
    else:
        print(f'Warning: {args.arch} is not in the implemented models list. If you added your own implementation, ignore this warning.')
        model = models.__dict__[args.arch](args)

    print('Total params for process {}: {}M'.format(
        args.cur_rank,
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        ))

    # define the criterion.
    criterion = nn.CrossEntropyLoss()

    # define the param to optimize.
    params_dict = dict(model.named_parameters())
    params = [
        {
            'params': [value],
            'name': key,
            'weight_decay': args.weight_decay if 'bn' not in key else 0.0
        }
        for key, value in params_dict.items()
    ]

    # define the optimizer.
    bfp_args = unpack_bfp_args(dict(vars(args)))
    print(bfp_args)
    optimizer = BFPSGD(
        params, lr=args.learning_rate, momentum=args.momentum,
        nesterov=args.use_nesterov, args=args, **bfp_args)

    # place model and criterion.
    device = torch.device(torch.cuda.current_device() if args.device != "cpu" else "cpu")
    model.to(device)
    criterion = criterion.to(device)

    # (optional) reload checkpoint
    resume_previous_status(args, model, optimizer)
    return model, criterion, optimizer


def correct_previous_resume(args, old_args):
    signal = (args.avg_model == old_args.avg_model) and \
        (args.data == old_args.data) and \
        (args.num_epochs >= old_args.num_epochs) and \
        (args.lr == old_args.lr) and \
        (args.momentum == old_args.momentum) and \
        (args.batch_size == old_args.batch_size)
    print('the status of previous resume: {}'.format(signal))
    return signal


def resume_previous_status(args, model, optimizer):
    if args.resume:
        if args.checkpoint_index is not None:
            # reload model from a specific checkpoint index.
            checkpoint_index = '_epoch_' + args.checkpoint_index
        else:
            # reload model from the latest checkpoint.
            checkpoint_index = ''
        checkpoint_path = join(
            args.resume, 'checkpoint{}.pth.tar'.format(checkpoint_index))

        print('try to load previous model from the path:{}'.format(
            checkpoint_path))
        if isfile(checkpoint_path):
            print("=> loading checkpoint {} for {}".format(
                args.resume, args.cur_rank))

            # get checkpoint.
            checkpoint = torch.load(checkpoint_path)

            if not correct_previous_resume(args, checkpoint['arguments']):
                raise RuntimeError('=> the checkpoint is not correct. skip.')
            else:
                # restore some run-time info.
                args.start_epoch = checkpoint['current_epoch'] + 1
                args.local_index = checkpoint['local_index']
                args.best_prec1 = checkpoint['best_prec1']
                args.best_epoch = checkpoint['arguments'].best_epoch

                # reset path for log.
                remove_folder(args.checkpoint_root)
                args.checkpoint_root = args.resume
                args.checkpoint_dir = join(args.resume, str(args.cur_rank))
                # restore model.
                model.load_state_dict(checkpoint['state_dict'])
                # restore optimizer.
                optimizer.load_state_dict(checkpoint['optimizer'])
                # logging.
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['current_epoch']))
                return
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
