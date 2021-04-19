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
"""define all global parameters here."""
from os.path import join
import argparse
import time

import cnn.models as models
from cnn.utils.log import log
from cnn.utils.auxiliary import info2path
from bfp.bfp_ops import unpack_bfp_args

def get_args():
    parser = argparse.ArgumentParser(
        description='Training DNNs with HBFP')
    parser.add_argument('--type', type=str, default='getting_started', choices=['getting_started', 'cnn', 'lstm', 'bert'])
    # parse args.
    args, unknown = parser.parse_known_args()
    return args


def tutorial_args():
    parser = argparse.ArgumentParser(
        description='Getting Started')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'])
    parser.add_argument('--num_format', default='fp32', type=str,
                        help='number format for fully connected and convolutional layers')
    parser.add_argument('--rounding_mode', default='stoc', type=str,
                        help='Rounding mode for bfp')
    parser.add_argument('--mant_bits', default=8, type=int,
                        help='Mantissa bits for bfp')
    parser.add_argument('--bfp_tile_size', default=0, type=int,
                        help='Tile size if using tiled bfp. 0 disables tiling')
    parser.add_argument('--weight_mant_bits', default=0, type=int,
                        help='Mantissa bits for weights bfp')

    # parse args.
    args, unknown = parser.parse_known_args()
    if args.weight_mant_bits == 0:
        args.weight_mant_bits = args.mant_bits
    return args


def get_cnn_args():
    ROOT_DIRECTORY = './'
    RAW_DATA_DIRECTORY = join(ROOT_DIRECTORY, 'data/')
    TRAINING_DIRECTORY = join(RAW_DATA_DIRECTORY, 'checkpoint')
    LOG_DIRECTORY = './logging'

    model_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__"))

    # feed them to the parser.
    parser = argparse.ArgumentParser(
        description='PyTorch Training for ConvNet')

    # dataset.
    parser.add_argument('--data', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'svhn'],
                        help='a specific dataset name')
    parser.add_argument('--data_dir', default=RAW_DATA_DIRECTORY,
                        help='path to dataset')
    parser.add_argument('--use_lmdb_data', default=False, type=str2bool,
                        help='use sequential lmdb dataset for better loading.')

    # model
    parser.add_argument('--arch', '-a', default='alexnet',
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: alexnet)')

    # training and learning scheme
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--num_epochs', type=int, default=90)
    parser.add_argument('--avg_model', type=str2bool, default=False)
    parser.add_argument('--reshuffle_per_epoch', default=False, type=str2bool)
    parser.add_argument('--batch_size', '-b', default=256, type=int,
                        help='mini-batch size (default: 256)')

    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate')
    parser.add_argument('--lr_decay', type=str2bool, default=None)
    parser.add_argument('--lr_decay_epochs', type=str, default=None)
    parser.add_argument('--lr_scale', type=str2bool, default=False)
    parser.add_argument('--lr_warmup', type=str2bool, default=False)
    parser.add_argument('--lr_warmup_size', type=int, default=5)
    parser.add_argument('--lr_lars', type=str2bool, default=False)
    parser.add_argument('--lr_lars_eta', type=float, default=0.002)
    parser.add_argument('--lr_lars_mode', type=str, default='clip')
    parser.add_argument('--lr_decay_auto', type=int, default=None)

    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--use_nesterov', default=False, type=str2bool)
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--drop_rate', default=0.0, type=float)

    # models.
    parser.add_argument('--densenet_growth_rate', default=12, type=int)
    parser.add_argument('--densenet_bc_mode', default=False, type=str2bool)
    parser.add_argument('--densenet_compression', default=0.5, type=float)

    parser.add_argument('--wideresnet_widen_factor', default=4, type=int)

    # miscs
    parser.add_argument('--manual_seed', type=int,
                        default=6, help='manual seed')
    parser.add_argument('--evaluate', '-e', dest='evaluate',
                        type=str2bool, default=False,
                        help='evaluate model on validation set')
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--timestamp', default=None, type=str)

    # checkpoint
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--checkpoint', '-c', default=TRAINING_DIRECTORY,
                        type=str,
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--checkpoint_index', type=str, default=None)
    parser.add_argument('--save_all_models', type=str2bool, default=False)
    parser.add_argument('--save_some_models', type=str, default='30,60,80')

    # device
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'])
    parser.add_argument('-j', '--num_workers', default=2, type=int,
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--world_size', default=None, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_backend', default='gloo', type=str,
                        help='distributed backend')
    # bfp
    parser.add_argument('--num_format', default='fp32', type=str,
                        help='number format for fully connected and convolutional layers')
    parser.add_argument('--rounding_mode', default='stoc', type=str,
                        help='Rounding mode for bfp')
    parser.add_argument('--mant_bits', default=8, type=int,
                        help='Mantissa bits for bfp')
    parser.add_argument('--bfp_tile_size', default=0, type=int,
                        help='Tile size if using tiled bfp. 0 disables tiling')
    parser.add_argument('--weight_mant_bits', default=0, type=int,
                        help='Mantissa bits for weights bfp')

    # parse args.
    args, unknown = parser.parse_known_args()
    if args.timestamp is None:
        args.timestamp = info2path(args)
    if args.weight_mant_bits == 0:
        args.weight_mant_bits = args.mant_bits
    return args

def get_lstm_args():
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default='data/penn/',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (LSTM, QRNN, GRU)')
    parser.add_argument('--emsize', type=int, default=400,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=1150,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=30,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=8000,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=70,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouth', type=float, default=0.3,
                        help='dropout for rnn layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.65,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0.1,
                        help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--wdrop', type=float, default=0.5,
                        help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--nonmono', type=int, default=5,
                        help='random seed')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'])
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                        help='path to save the final model')
    parser.add_argument('--alpha', type=float, default=2,
                        help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--resume', type=str,  default='',
                        help='path of model to resume')
    parser.add_argument('--optimizer', type=str,  default='sgd',
                        help='optimizer to use (sgd, adam)')
    parser.add_argument('--when', nargs="+", type=int, default=[-1],
                        help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
    # bfp
    parser.add_argument('--num_format', default='fp32', type=str,
                        help='number format for fully connected and convolutional layers')
    parser.add_argument('--rounding_mode', default='stoc', type=str,
                        help='Rounding mode for bfp')
    parser.add_argument('--mant_bits', default=8, type=int,
                        help='Mantissa bits for bfp')
    parser.add_argument('--bfp_tile_size', default=0, type=int,
                        help='Tile size if using tiled bfp. 0 disables tiling')
    parser.add_argument('--bfp_symmetric', type=str2bool, default=False)
    parser.add_argument('--weight_mant_bits', default=0, type=int,
                        help='Mantissa bits for weights bfp')
    args, unknown = parser.parse_known_args()

    if args.weight_mant_bits == 0:
        args.weight_mant_bits = args.mant_bits
    if args.device == 'gpu':
        args.cuda = True
    else:
        args.cuda = False

    args.bfp_config = unpack_bfp_args(vars(args))
    args.tied = True
    return args

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def print_args(args):
    print('parameters: ')
    for arg in vars(args):
        print(arg, getattr(args, arg))

def log_args(args):
    log('parameters: ')
    for arg in vars(args):
        log(str(arg) + '\t' + str(getattr(args, arg)))

if __name__ == '__main__':
    args = get_args()
