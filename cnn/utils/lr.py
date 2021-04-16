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


def adjust_learning_rate(args, optimizer, init_lr=0.1):
    """Sets the learning rate to the initial LR decayed by # of accessed sample
        We should decay the learning rate based on the number of samples that
        we have accessed.
    """
    # functions.
    def define_lr_decay_by_epoch(args, epoch_index):
        """ decay based on the number of accessed samples per device. """
        for ind, change_epoch in enumerate(args.lr_change_epochs):
            if epoch_index <= change_epoch:
                return args.learning_rate * (0.1 ** ind)
        return args.learning_rate * (0.1 ** 3)

    def define_lr_decay_by_index_poly(args, pow=2):
        """ decay the learning rate polynomially. """
        return args.learning_rate * (
            1 - args.local_index / args.num_batches_total_train) ** 2

    def define_lr_decay_by_auto_detect(args):
        """ decay the learning rate if there is no improvement over epochs. """
        best_epoch = args.best_epoch
        num_best_epoch = len(best_epoch)
        if num_best_epoch < 2:
            return args.lr

        # get best epoch gaps.
        best_epoch_gap = [
            ind for ind in range(1, num_best_epoch)
            if best_epoch[ind] - best_epoch[ind - 1] > args.lr_decay_auto]

        return args.learning_rate * (0.1 ** len(best_epoch_gap))

    # adjust learning rate.
    if args.lr_decay_epochs is not None:
        num_accessed_samples = args.local_index * args.batch_size
        epoch_index = num_accessed_samples // args.num_train_samples_per_device
        lr = define_lr_decay_by_epoch(args, epoch_index)
    elif args.lr_decay_auto is not None:
        lr = define_lr_decay_by_auto_detect(args)
    else:
        lr = define_lr_decay_by_index_poly(args)

    # lr warmup at the first few epochs.
    if args.lr_warmup and args.local_index < args.num_warmup_samples:
        lr = (lr - init_lr) / args.num_warmup_samples * args.local_index + init_lr

    # assign learning rate.
    if args.old_learning_rate != lr:
        args.old_learning_rate = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def adjust_learning_rate_by_lars(args, global_lr, para):
    """Adjust the learning rate via Layer-Wise Adaptive Rate Scaling (LARS)
    """
    lr = global_lr

    if args.lr_lars:
        local_lr = args.lr_lars_eta * para.data.norm() / para.grad.data.norm()
        if args.lr_lars_mode == 'clip':
            lr = min(local_lr, lr)
        elif args.lr_lars_mode == 'scale':
            lr = local_lr * lr
        else:
            raise ValueError('Invalid LARS mode: %s' % args.lr_lars_factor)
    return lr
