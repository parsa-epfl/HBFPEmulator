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

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from cnn.utils.log import log
from cnn.dataset.partition import DataPartitioner
from cnn.dataset.imagenet_folder import define_imagenet_folder
from cnn.dataset.svhn_folder import define_svhn_folder


def partition_dataset(args, dataset_type='train'):
    """ Given a dataset, partition it. """
    dataset = get_dataset(args, args.data, args.data_dir, split=dataset_type)
    batch_size = args.batch_size
    world_size = args.world_size

    # partition data.
    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    partition = DataPartitioner(args, dataset, partition_sizes)
    data_to_load = partition.use(args.cur_rank)

    if dataset_type == 'train':
        args.train_dataset_size = len(dataset)
        args.num_train_samples_per_device = len(data_to_load)
        log('  We have {} samples for {}, \
            load {} data for process (rank {}), and partition it'.format(
            len(dataset), dataset_type, len(data_to_load), args.cur_rank))
    else:
        args.val_dataset_size = len(dataset)
        args.num_val_samples_per_device = len(data_to_load)
        log('  We have {} samples for {}, \
            load {} val data for process (rank {}).'.format(
            len(dataset), dataset_type, len(data_to_load), args.cur_rank))

    # use Dataloader.
    data_type_label = (dataset_type == 'train')
    data_loader = torch.utils.data.DataLoader(
        data_to_load, batch_size=batch_size,
        shuffle=data_type_label,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    log('we have {} batches for {} for rank {}.'.format(
        len(data_loader), dataset_type, args.cur_rank))
    return data_loader


def create_dataset(args):
    log('create {} dataset for rank {}'.format(args.data, args.cur_rank))
    train_loader = partition_dataset(args, dataset_type='train')
    val_loader = partition_dataset(args, dataset_type='test')
    return train_loader, val_loader


def get_dataset(
        args, name, datasets_path, split='train', transform=None,
        target_transform=None, download=True):
    train = (split == 'train')
    root = os.path.join(datasets_path, name)
    if not os.path.exists(root):
        os.makedirs(root)

    if name == 'cifar10' or name == 'cifar100':
        # decide normalize parameter.
        if name == 'cifar10':
            dataset_loader = datasets.CIFAR10
            normalize = transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        elif name == 'cifar100':
            dataset_loader = datasets.CIFAR100
            normalize = transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

        # decide data type.
        if train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((32, 32), 4),
                transforms.ToTensor(),
                normalize])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        return dataset_loader(root=root,
                              train=train,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        return define_svhn_folder(root=root,
                                  is_train=train,
                                  transform=transform,
                                  target_transform=target_transform,
                                  download=download)
    elif name == 'imagenet':
        root = os.path.join(datasets_path, 'lmdb') if args.use_lmdb_data \
            else datasets_path
        if train:
            root = os.path.join(root, 'train{}'.format(
                '' if not args.use_lmdb_data else '.lmdb')
            )
        else:
            root = os.path.join(root, 'val{}'.format(
                '' if not args.use_lmdb_data else '.lmdb')
            )
        return define_imagenet_folder(root=root, flag=args.use_lmdb_data)
    else:
        raise NotImplementedError
