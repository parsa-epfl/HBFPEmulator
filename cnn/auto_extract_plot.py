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
import re
import os
from os import listdir
from os.path import join
import argparse
from functools import reduce

from cnn.utils.opfiles import read_txt, write_pickle
from cnn.utils.auxiliary import str2time


def get_args():
    # feed them to the parser.
    parser = argparse.ArgumentParser(description='Extract results for plots')

    # add arguments.
    parser.add_argument(
        '--in_dir', type=str,
        default='data/checkpoint/cifar10/resnet20/')
    parser.add_argument('--exclude_dir', type=str, default='test')
    parser.add_argument('--out_file', type=str, default='summary.pickle')
    parser.add_argument('--start_with', '-s', type=str, default='')
    parser.add_argument('--paras_to_extract', type=str, default=None)

    # parse args.
    args = parser.parse_args()
    return args


def valid_args(args):
    assert args.in_dir is None


def multiply_two_strs(str1, str2):
    """assume these two strings are in the form of int,
        and return the result as int.
    """
    return str(int(str1) * int(str2))


def get_expected_args(cur_args):
    variables = ['lr', 'batch_size', 'lr_lars_eta',
                 'num_epochs', 'momentum',
                 'weight_decay', 'manual_seed', 'eval_freq', 'summary_freq',
                 'world_size', 'cur_rank', 'cur_gpu_device']
    paras_confs_dict = dict(
        [(k, float(v)) if k in variables else (k, v)
         for k, v in cur_args.items()])
    return paras_confs_dict


def get_roundtime(records):
    round_time_record = [l for l in records if 'Train' in l]
    round_time = round_time_record[-1].split('|')[-4].split(':')[-1][:-2] \
        if len(round_time_record) != 0 else ''
    return round_time


def get_runtime_tracking(records):
    pattern = r'(.*?)\s+Process .: Local index: (.*?)\. Data: (.*?)s \| Batch: (.*?)s \| Loss: (.*?) \| top1: (.*?) \| top5: (.*?)$'

    def helper(line):
        try:
            matched_line = re.findall(pattern, line, re.DOTALL)[0]
            matched_line = [
                float(l) if ind != 3 and ind != 0 else l
                for ind, l in enumerate(matched_line)]
            matched_line[0] = str2time(matched_line[0], '%Y:%m:%d %H:%M:%S')
            names = ['time', 'local_index',
                     'data_time', 'batch_time', 'loss', 'top1', 'top5']
            zip_line = zip(names, matched_line)
            line = dict(zip_line)
            return line
        except Exception as e:
            return None

    runtime_record = [helper(l) for l in records]
    runtime_record = [l for l in runtime_record if l is not None]
    return runtime_record


def get_train_accuracy(runtime_tracking):
    training = [(l['loss'], l['top1'], l['top5']) for l in runtime_tracking]
    return [min(l[0] for l in training), max(l[1] for l in training)]


def get_test_accuracy(records):
    key = ['top1', 'top5']
    pattern = 'Prec@1: (.*?) Prec@5: (.*?)$'

    def helper(line):
        try:
            matched_line = re.findall(pattern, line, re.DOTALL)[0]
            return dict((k, float(v)) for k, v in zip(key, matched_line))
        except Exception as e:
            return None

    test_accuracy_records = [l for l in records if 'Val' in l]
    test_accuracy_records = [helper(l) for l in test_accuracy_records]
    return test_accuracy_records, \
        max([list(l.values())[0] for l in test_accuracy_records])


def extract_from_record(args, path):
    # define negative return.
    negative_return = '', '', ''

    # get record file path.
    pattern = '^record.'
    files = [l for l in listdir(path) if bool(re.search(pattern, l))]

    if len(files) != 1:
        return negative_return
    path = join(path, files[0])

    # load record file and parse args.
    records = read_txt(path)

    # parse the complete record.
    regex = re.compile(r'.\t.*\t')
    cur_args = dict(
        [tuple(l.split('\t')[1:]) for l in records if re.search(regex, l)])
    conf_dict = get_expected_args(cur_args)

    # parse records.
    train_runtime_tracking = get_runtime_tracking(records)
    best_train_accuracy = get_train_accuracy(train_runtime_tracking)
    test_runtime_tracking, best_test_accuracy = get_test_accuracy(records)
    best_accuracy = best_train_accuracy + [best_test_accuracy]
    return conf_dict, best_accuracy, \
        train_runtime_tracking, test_runtime_tracking


def get_summary(all_info):
    # define want to extract.
    if args.paras_to_extract is None:
        paras_to_extract = \
            'lr,lr_scale,lr_decay_ratios,lr_warmup,lr_lars,lr_lars_eta,lr_lars_mode,batch_size'
    interested_results = 'best_train_accuracy,best_train_loss,best_test_accuracy'

    paras_to_extract = paras_to_extract.split(',')
    interested_results = interested_results.split(',')

    header = paras_to_extract + interested_results
    summary = [ str(all_info[h]) for h in header]
    return ' & \t'.join(summary)


def load_and_extract_record(args, path):
    # define path.
    existing_folders = sorted(listdir(path))
    path = join(path, existing_folders[0])
    print(' process the path: {}'.format(path))

    conf_dict, best_accuracy, train_runtime_tracking, test_runtime_tracking = \
        extract_from_record(args, path)

    all = {
        'conf': conf_dict,
        'lr': conf_dict['lr'],
        'lr_scale': conf_dict['lr_scale'],
        'lr_decay_ratios': conf_dict['lr_decay_ratios'],
        'lr_warmup': conf_dict['lr_warmup'],
        'lr_lars': conf_dict['lr_lars'],
        'lr_lars_eta': conf_dict['lr_lars_eta'],
        'lr_lars_mode': conf_dict['lr_lars_mode'],
        'batch_size': conf_dict['batch_size'],
        'num_epochs': conf_dict['num_epochs'],
        'dataset': conf_dict['data'],
        'train_runtime_tracking': train_runtime_tracking,
        'test_runtime_tracking': test_runtime_tracking,
        'best_train_loss': best_accuracy[0],
        'best_train_accuracy': best_accuracy[1],
        'best_test_accuracy': best_accuracy[2],
        'num_gpus': len(existing_folders),
        'world': conf_dict['world']
    }

    summary = get_summary(all)
    return all, summary


def main(args):
    # define path.
    out_path = join(args.in_dir, args.out_file)
    os.system('rm -rf {}'.format(out_path))

    parent_paths = [
        join(args.in_dir, name)
        for name in listdir(args.in_dir)
        if name.startswith(args.start_with)]

    parent_paths = filter(
        lambda name: args.exclude_dir not in name, parent_paths)

    folder_paths = reduce(
        lambda a, b: a + b,
        [[join(parent_path, folder_path)
          for folder_path in listdir(parent_path)]
         for parent_path in parent_paths])

    # parse record.
    results = []
    for folder_path in folder_paths:
        try:
            results.append(load_and_extract_record(args, folder_path))
        except Exception as e:
            pass

    results_all = [l[0] for l in results]
    results_summary = [l[1] for l in results]
    write_pickle(results_all, out_path)

    print('')
    print('\n'.join(results_summary))


if __name__ == '__main__':
    args = get_args()
    main(args)
