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

import os
import time
import logging
import logging.config

import cnn.utils.opfiles as opfile

log_path = None

def configure_log(args=None):
    global log_path

    if args is not None:
        log_path = os.path.join(
            args.checkpoint_dir, 'record' + str(args.cur_rank))
    else:
        log_path = os.path.join(os.getcwd(), "record")


def log(content):
    """print the content while store the information to the path."""
    content = time.strftime("%Y:%m:%d %H:%M:%S") + "\t" + content
    print(content)
    opfile.write_txt(content + "\n", log_path, type="a")


def logging_computing(args, tracker, loss, prec1, prec5, input):
    # measure accuracy and record loss.
    tracker['losses'].update(loss.item(), input.size(0))
    tracker['top1'].update(prec1[0], input.size(0))
    tracker['top5'].update(prec5[0], input.size(0))

    # measure elapsed time.
    tracker['batch_time'].update(time.time() - tracker['end_data_time'])
    tracker['start_sync_time'] = time.time()


def logging_sync(args, tracker):
    # measure elapsed time.
    tracker['sync_time'].update(time.time() - tracker['start_sync_time'])


def logging_load(args, tracker):
    # measure elapsed time.
    tracker['load_time'].update(time.time() - tracker['start_load_time'])


def logging_display(args, tracker):
    log_info = 'Local index: {local_index}. Load: {load:.3f}s | Data: {data:.3f}s | Batch: {batch:.3f}s | Sync: {sync:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
        local_index=args.local_index,
        load=tracker['load_time'].avg,
        data=tracker['data_time'].avg,
        batch=tracker['batch_time'].avg,
        sync=tracker['sync_time'].avg,
        loss=tracker['losses'].avg,
        top1=tracker['top1'].avg,
        top5=tracker['top5'].avg)
    log('Process {}: '.format(args.cur_rank) + log_info)
    tracker['start_load_time'] = time.time()
