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
import random

import numpy as np
import torch
import torch.distributed as dist
import pdb

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, indices):
        self.data = data
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data_idx = self.indices[index]
        return self.data[data_idx]


class Partitioner(object):
    def consistent_indices(self, indices):
        if self.args.cur_rank == 0 and self.args.reshuffle_per_epoch:
            random.shuffle(indices)

        #pdb.set_trace()
        # broadcast.
        indices = torch.IntTensor(indices)
        group = dist.new_group(self.args.ranks)
        dist.broadcast(indices, src=0, group=group)
        return list(indices)


class DataPartitioner(Partitioner):
    """ Partitions a dataset into different chuncks. """
    def __init__(self, args, data, sizes=[0.7, 0.2, 0.1]):
        # prepare info.
        self.args = args
        self.data = data
        self.data_size = len(self.data)
        self.partitions = []

        # get shuffled/unshuffled data.
        indices = [x for x in range(0, self.data_size)]
        indices = self.consistent_indices(indices)

        # partition indices.
        sizes = np.cumsum(sizes)
        from_index = 0
        for ind, frac in enumerate(sizes):
            to_index = int(sizes[ind] * self.data_size)
            self.partitions.append(indices[from_index: to_index])
            from_index = to_index

    def use(self, partition_ind):
        return Partition(self.data, self.partitions[partition_ind])


class IMDBDataPartitioner(Partitioner):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, args, data, sizes=[0.7, 0.2, 0.1]):
        self.args = args
        self.ds = data.ds
        keys = self.ds.keys
        self.data_size = len(keys)
        self.partitions = []
        indices = self.consistent_indices()

        for frac in sizes:
            part_size = int(frac * self.data_size)
            selected_indices = list(np.array(keys)[indices[0: part_size]])
            self.partitions.append(selected_indices)
            indices = indices[part_size:]

    def use(self, partition_ind):
        return self.partitions[partition_ind]
