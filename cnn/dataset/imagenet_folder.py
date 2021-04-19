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
import os
import sys

import lmdb
import cv2
import numpy as np
from PIL import Image

import torch.utils.data as data
import torchvision.datasets as datasets
import tensorpack.utils.serialize as serialize

from cnn.utils.log import log
from cnn.dataset.preprocess_toolkit import get_transform


if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def define_imagenet_folder(root, flag):
    is_train = 'train' in root
    if flag:
        log('load imagenet from lmdb.')
        transform = pytorch_example_augmentor(is_train)
        return IMDBPT(root, transform=transform)
    else:
        log("load imagenet using pytorch's default dataloader.")
        transform = pytorch_example_augmentor(is_train)
        return datasets.ImageFolder(root=root,
                                    transform=transform,
                                    target_transform=None)


def pytorch_example_augmentor(is_train):
    return get_transform('imagenet', augment=is_train, color_process=False)


def be_ncwh_pt(x):
    return x.permute(0, 3, 1, 2)  # pytorch is (n,c,w,h)


def uint8_to_float(x):
    x = x.permute(0, 3, 1, 2)  # pytorch is (n,c,w,h)
    return x.float()/128. - 1.


class IMDBPT(data.Dataset):
    """
    Args:
        root (string): Either root directory for the database files,
            or a absolute path pointing to the file.
        classes (string or list): One of {'train', 'val', 'test'} or a list of
            categories to load. e,g. ['bedroom_train', 'church_train'].
        transform (callable, optional): A function/transform that
            takes in an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
        target_transform (callable, optional):
            A function/transform that takes in the target and transforms it.
    """

    def __init__(self, root, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.lmdb_files = self._get_valid_lmdb_files()

        # for each class, create an LSUNClassDataset
        self.dbs = []
        for lmdb_file in self.lmdb_files:
            self.dbs.append(LMDBPTClass(
                root=lmdb_file, transform=transform,
                target_transform=target_transform))

        # build up indices.
        self.indices = np.cumsum([len(db) for db in self.dbs])
        self.length = self.indices[-1]
        self._get_index_zones = self._build_indices()

    def _get_valid_lmdb_files(self):
        """get valid lmdb based on given root."""
        if not self.root.endswith('.lmdb'):
            for l in os.listdir(self.root):
                if '_' in l and '-lock' not in l:
                    yield os.path.join(self.root, l)
        else:
            yield self.root

    def _build_indices(self):
        indices = self.indices
        from_to_indices = enumerate(zip(indices[: -1], indices[1:]))

        def f(x):
            if len(list(from_to_indices)) == 0:
                return 0, x

            for ind, (from_index, to_index) in from_to_indices:
                if from_index <= x and x < to_index:
                    return ind, x - from_index
        return f

    def _get_matched_index(self, index):
        return self._get_index_zones(index)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target)
        """
        block_index, item_index = self._get_matched_index(index)
        image, target = self.dbs[block_index][item_index]
        return image, target

    def __len__(self):
        return self.length

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp,
            self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp,
            self.target_transform.__repr__().replace(
                '\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class LMDBPTClass(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        # open lmdb env.
        self.env = self._open_lmdb()

        # get file stats.
        self._get_length()

        # prepare cache_file
        self._prepare_cache()

    def _open_lmdb(self):
        return lmdb.open(
            self.root,
            subdir=os.path.isdir(self.root),
            readonly=True, lock=False, readahead=False,
            map_size=1099511627776 * 2, max_readers=1, meminit=False)

    def _get_length(self):
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

            if txn.get(b'__keys__') is not None:
                self.length -= 1

    def _prepare_cache(self):
        cache_file = self.root + '_cache_'
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key
                             for key, _ in txn.cursor() if key != b'__keys__']
            pickle.dump(self.keys, open(cache_file, "wb"))

    def _image_decode(self, x):
        image = cv2.imdecode(x, cv2.IMREAD_COLOR).astype('uint8')
        return Image.fromarray(image, 'RGB')

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            bin_file = txn.get(self.keys[index])

        image, target = serialize.loads(bin_file)
        image = self._image_decode(image)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.root + ')'
