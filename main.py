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
import platform

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.distributed as dist
import cnn
import os

from arguments import get_args, log_args, get_cnn_args, get_lstm_args, tutorial_args
from cnn.utils.log import log, configure_log
from cnn.utils.set_conf import set_conf
from cnn.models.create_model import create_model
from cnn.runs.distributed_running import train_and_validate as train_val_op
from lstm.train_and_val import train_lstm
from getting_started.resnet_cifar10 import resnet18_cifar10
from torch.multiprocessing import Process
import pdb

def main(args):
    if args.type == 'getting_started':
        args = tutorial_args()
        resnet18_cifar10(args)
    elif args.type == 'cnn':
        args = get_cnn_args()
        size = args.world_size
        processes = []
        for rank in range(size):
            p = Process(target=init_processes, args=(rank, size, run))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    elif args.type == 'lstm':
        args = get_lstm_args()
        train_lstm(args)

def run(rank, size):
    """ Distributed Synchronous SGD Example """
    args = get_cnn_args()
    set_conf(args)
    print('set_conf...')
    # create model and deploy the model.
    model, criterion, optimizer = create_model(args)
    # config and report.
    configure_log(args)
    print('configure_log...')
    log_args(args)
    print('log_args...')

    device = 'GPU-'+ str(torch.cuda.current_device()) if args.device != "cpu" else "cpu"

    log(
        'Rank {} {}'.format(
            args.cur_rank,
            device
            # args.cur_gpu_device
            )
        )

    train_val_op(args, model, criterion, optimizer)


def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    args = get_args()
    main(args)
