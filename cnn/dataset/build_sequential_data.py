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
#
################################################################################
#
# class BinaryILSVRC12 in this file is taken from Sequential Read tutorial in
# tensorpack/docs/tutorial/efficient-dataflow.md which is available under
# Apache License 2.0.


# -*- coding: utf-8 -*-
import argparse
import os
from os.path import join
from tensorpack.dataflow import *

def get_args():
    parser = argparse.ArgumentParser(description='aug data.')

    # define arguments.
    parser.add_argument('--data', default='imagenet',
                        help='a specific dataset name')
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--data_type', default='train', type=str)

    # parse args.
    args = parser.parse_args()

    # check args.
    assert args.data_dir is not None
    assert args.data_type == 'train' or args.data_type == 'val'
    return args


def build_dirs(path):
    try:
        os.makedirs(path)
    except OSError as e:
        print(e)


def sequential_data(root_path):
    # define path.
    lmdb_path = join(root_path, 'lmdb')
    build_dirs(lmdb_path)
    lmdb_file_path = join(lmdb_path, args.data_type + '.lmdb')

    # build lmdb data.
    #from tensorpack.dataflow import dataset, PrefetchDataZMQ, dftools
    import numpy as np

    class BinaryILSVRC12(dataset.ILSVRC12Files):
        def __iter__(self):
            for fname, label in super(BinaryILSVRC12, self).__iter__():
                with open(fname, 'rb') as f:
                    jpeg = f.read()
                jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
                yield [jpeg, label]
    ds0 = BinaryILSVRC12(root_path, args.data_type)
    ds1 = MultiProcessRunnerZMQ(ds0, num_proc=1)
    LMDBSerializer.save(ds1, lmdb_file_path)


def main(args):
    if args.data == 'imagenet':
        root_path = args.data_dir

    sequential_data(root_path)


if __name__ == '__main__':
    args = get_args()
    main(args)
