#!/bin/bash

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/parsadata1/Megatron-HBFP/_dataset/my-bert_text_sentence
VOCAB_PATH=/parsadata1/Megatron-HBFP/_dataset/bert-vocab.txt
CHECKPOINT_PATH=debug_hbfp8
NOCHECKPOINT_PATH=_debug_hbfp8


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --tensor-model-parallel-size 1 \
       --num-layers 2 \
       --hidden-size 128 \
       --num-attention-heads 2 \
       --micro-batch-size 4 \
       --global-batch-size 8 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 100 \
       --save $CHECKPOINT_PATH \
       --load $NOCHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_PATH \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 10 \
       --eval-iters 10 \
       --hbfp_num_format bfp \
       --hbfp_mant_bits 7 \
       --hbfp_weight_mant_bits 15
