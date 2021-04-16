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
from os.path import join
from itertools import groupby
from functools import reduce

import torch
import torch.distributed as dist

from cnn.utils.opfiles import build_dirs

def set_checkpoint(args):
    args.checkpoint_root = join(
        args.checkpoint, args.data, args.arch,
        args.device if args.device is not None else '', args.timestamp)
    args.checkpoint_dir = join(args.checkpoint_root, str(args.cur_rank))
    args.save_some_models = args.save_some_models.split(',')

    # if the directory does not exists, create them.
    build_dirs(args.checkpoint_dir)


def set_lr(args):
    args.lr_change_epochs = [
        int(l) for l in args.lr_decay_epochs.split(',')] \
        if args.lr_decay_epochs is not None \
        else None

    #lr tuning
    args.learning_rate_per_sample = 0.1 / args.batch_size
    args.learning_rate = \
        args.learning_rate_per_sample * args.batch_size * args.world_size \
        if args.lr_scale else args.lr
    args.old_learning_rate = args.learning_rate


def set_conf(args):
    # global conf.
    # configure world.

    torch.manual_seed(args.manual_seed)

    # local conf.
    args.local_index = 0
    args.best_prec1 = 0
    args.best_epoch = []
    args.val_accuracies = []

    args.ranks = list(range(args.world_size))
    args.cur_rank = dist.get_rank()
    if args.device == 'gpu':
        torch.cuda.set_device(args.cur_rank)


    # define checkpoint for logging.
    set_checkpoint(args)

    # define learning rate and learning rate decay scheme.
    set_lr(args)
