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
#
################################################################################
#
# Modified file from Salesforce's LSTM and QRNN Language Model Toolkit
# (https://github.com/salesforce/awd-lstm-lm). See LICENSE for more details.

import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import dill
from torch.autograd import Variable
from bfp.bfp_optim_lstm import BFPSGD, BFPAdam
from bfp.bfp_ops import unpack_bfp_args
import arguments

import dill
import lstm.data as data
import lstm.rnnmodel as rnnmodel

from lstm.utils import batchify, get_batch, repackage_hidden

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

###############################################################################
# Load data
###############################################################################


import os
import hashlib

def train_lstm(args):

    def model_save(fn):
        pass
        #with open(fn, 'wb') as f:
        #    torch.save([model, criterion, optimizer], f, pickle_module=dill)

    def model_load(fn):
        global model, criterion, optimizer
        with open(fn, 'rb') as f:
            model, criterion, optimizer = torch.load(f)
    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
    fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset...')
        corpus = data.Corpus(args.data)
        torch.save(corpus, fn)

    eval_batch_size = 10
    test_batch_size = 1
    train_data = batchify(corpus.train, args.batch_size, args)
    val_data = batchify(corpus.valid, eval_batch_size, args)
    test_data = batchify(corpus.test, test_batch_size, args)

    ###############################################################################
    # Build the model
    ###############################################################################

    from lstm.splitcross import SplitCrossEntropyLoss
    criterion = None

    ntokens = len(corpus.dictionary)
    model = rnnmodel.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth,
                           args.dropouti, args.dropoute, args.wdrop, args.tied, **args.bfp_config)
    ###
    if args.resume:
        print('Resuming model ...')
        model_load(args.resume)
        optimizer.param_groups[0]['lr'] = args.lr
        model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
        if args.wdrop:
            from weight_drop import WeightDrop
            for rnn in model.rnns:
                if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
                elif rnn.zoneout > 0: rnn.zoneout = args.wdrop
    ###
    if not criterion:
        splits = []
        if ntokens > 500000:
            # One Billion
            # This produces fairly even matrix mults for the buckets:
            # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
            splits = [4200, 35000, 180000]
        elif ntokens > 75000:
            # WikiText-103
            splits = [2800, 20000, 76000]
        print('Using', splits)
        criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
    ###
    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    ###
    params = list(model.parameters()) + list(criterion.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
    print('Args:', args)
    print('Model total parameters:', total_params)

    ###############################################################################
    # Training code
    ###############################################################################

    def evaluate(data_source, batch_size=10):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        if args.model == 'QRNN': model.reset()
        total_loss = 0
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(batch_size)
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args, evaluation=True)
            output, hidden = model(data, hidden)
            total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
            hidden = repackage_hidden(hidden)
        return total_loss / len(data_source)


    def train():

        # Turn on training mode which enables dropout.
        if args.model == 'QRNN': model.reset()
        total_loss = 0
        start_time = time.time()
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(args.batch_size)
        batch, i = 0, 0
        while i < train_data.size(0) - 1 - 1:
            bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
            # Prevent excessively small or negative sequence lengths
            seq_len = max(5, int(np.random.normal(bptt, 5)))
            # There's a very small chance that it could select a very long sequence length resulting in OOM
            # seq_len = min(seq_len, args.bptt + 10)

            lr2 = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
            model.train()
            data, targets = get_batch(train_data, i, args, seq_len=seq_len)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = repackage_hidden(hidden)
            optimizer.zero_grad()

            output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
            raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

            loss = raw_loss
            # Activiation Regularization
            if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
            optimizer.step()

            total_loss += raw_loss.data
            optimizer.param_groups[0]['lr'] = lr2
            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                log_str = ('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} |'
                           ' ms/batch {:5.2f} | '
                           'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}\n'.format(
                               epoch, batch, len(train_data) // args.bptt,
                               optimizer.param_groups[0]['lr'],
                               elapsed * 1000 / args.log_interval, cur_loss,
                               math.exp(cur_loss), cur_loss / math.log(2)))
                print(log_str)

                total_loss = 0
                start_time = time.time()
            ###
            batch += 1
            i += seq_len

    # Loop over epochs.
    lr = args.lr
    best_val_loss = []
    stored_loss = 100000000

    log_f = open(args.save + '.log', 'w')

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        optimizer = None
        # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
        if args.optimizer == 'sgd':
            optimizer = BFPSGD(params, lr=args.lr, weight_decay=args.wdecay, **args.bfp_config)
        if args.optimizer == 'adam':
            optimizer = BFPAdam(params, lr=args.lr, weight_decay=args.wdecay, **args.bfp_config)
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train()
            if 't0' in optimizer.param_groups[0]:
                tmp = {}
                for prm in model.parameters():
                    tmp[prm] = prm.data.clone()
                    prm.data = optimizer.state[prm]['ax'].clone()

                val_loss2 = evaluate(val_data)
                print('-' * 89)
                log_str = ('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                  epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
                log_f.write(log_str + "\n")
                print(log_str)
                print('-' * 89)

                if val_loss2 < stored_loss:
                    model_save(args.save)
                    print('Saving Averaged!')
                    stored_loss = val_loss2

                for prm in model.parameters():
                    prm.data = tmp[prm].clone()

            else:
                val_loss = evaluate(val_data, eval_batch_size)
                print('-' * 89)
                log_str = ('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                  epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
                print(log_str)
                log_f.write(log_str + "\n")
                print('-' * 89)

                if val_loss < stored_loss:
                    model_save(args.save)
                    print('Saving model (new best validation)')
                    stored_loss = val_loss

                if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                    print('Switching to ASGD')
                    optimizer = BFPASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

                if epoch in args.when:
                    print('Saving model before learning rate decreased')
                    model_save('{}.e{}'.format(args.save, epoch))
                    print('Dividing learning rate by 10')
                    optimizer.param_groups[0]['lr'] /= 10.

                best_val_loss.append(val_loss)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    model_load(args.save)

    # Run on test data.
    test_loss = evaluate(test_data, test_batch_size)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
        test_loss, math.exp(test_loss), test_loss / math.log(2)))
    print('=' * 89)
