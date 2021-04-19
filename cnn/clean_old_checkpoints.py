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
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description='aug data.')

    # define arguments.
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--keep', default='', type=str)

    # parse args.
    args = parser.parse_args()

    # check args.
    assert args.data_dir is not None
    return args


def main(args):
    root = args.data_dir
    to_keep_patterns = args.keep.split(',')
    files = []
    valid_files = []

    # get checkpoint files.
    for rank in os.listdir(root):
        rank_path = os.path.join(root, rank)

        for file in os.listdir(rank_path):
            if 'checkpoint_' in file:
                files.append(os.path.join(rank_path, file))

    # filter checkpoint files, and remove valid checkpoint files.
    for file in files:
        for pattern in to_keep_patterns:
            if pattern + '.' not in file:
                valid_files.append(file)
                break

    # remove files.
    for file in valid_files:
        print('remove file from path: {}'.format(file))
        os.remove(file)


if __name__ == '__main__':
    args = get_args()
    main(args)
