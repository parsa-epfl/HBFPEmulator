#  ColTraIn HBFP Training Emulator
[HBFP](https://papers.nips.cc/paper/2018/file/6a9aeddfc689c1d0e3b9ccc3ab651bc5-Paper.pdf) is a hybrid Block Floating-Point (BFP) - Floating-Point (FP) number representation for DNN training introduced by [ColTraIn: Co-located DNN Training and Inference](https://parsa.epfl.ch/coltrain/) team of [PARSA](https://parsa.epfl.ch/) and [MLO](https://www.epfl.ch/labs/mlo/) at EPFL. HBFP offers the best of both worlds: the high accuracy of floating-point at the superior hardware density of fixed-point by performing all dot products in BFP and other operations in FP32. For a wide variety of models, HBFP matches floating-point’s accuracy while enabling hardware implementations that deliver up to 8.5x higher throughput. This repository is for ongoing research on training DNNs with HBFP.

More information is available on the ColTraIn [website](https://parsa.epfl.ch/coltrain/).

[![ColTraIn](https://parsa.epfl.ch/coltrain/images/banner.jpg)](https://parsa.epfl.ch/coltrain/)

## GPU Simulation
We train DNNs with the proposed HBFP approach, using BFP in the compute-intensive operations (matrix multiplications, convolutions, and their backward passes) and FP32 in the other operations. We simulate BFP dot products in GPUs by modifying PyTorch’s linear and convolution layers to reproduce the behaviour of BFP matrix multipliers. 
* We redefine PyTorch’s convolution and linear modules using its `autograd.function` feature to create new modules that process the inputs and outputs of both the forward and backward passes to simulate BFP. In the forward pass, we convert the activations to BFP, giving the `x` tensor one exponent per training input. Then we execute the target operation in native floating-point arithmetic. In the backward pass, we perform the same pre-/post-processing of the inputs/outputs of the `x` derivative.
* We handle the weights in the optimizer. We create a shell optimizer that takes the original optimizer, performs its update function in FP32 and converts the weights to two BFP formats: one with wide and another with narrow mantissas. The former is used in future weight updates while the latter is used in forward and backward passes. We also use this same mechanism to simulate different tile sizes for weight matrices. Finally, for convolutional layers, we tile the two outer feature map dimensions of the weight matrices.

## Setup
HBFPEmulator contains several example DNN models including CNNs, LSTMs and BERT (fork of Megatron-LM and included as a submodule). We have tested HBFPEmulator on Python 3.8.5 and PyTorch 1.8. To install the emulator with all the requirements and submodules, run the following commands:
```
git clone --recurse-submodules https://github.com/parsa-epfl/HBFPEmulator.git
cd HBFPEmulator
pip install -r requirements.txt
```
The requirements and installation instructions for the submodule Megatron-LM_HBFP can be found under the project repository.

## Getting Started
HBFP can be deployed easily to any model by using only the files under the [`bfp/`](./bfp) directory. For a quick understanding, please check the usage of the
BFP functions in our ["Hello world!" example](./getting_started/). The folder also contains the sufficient HBFP-related files.

Running the BFP tests:
```
python bfp/bfp_ops.py
```
Training the "Hello world!" example: ResNet18 on CIFAR10:
```
python main.py --type getting_started --num_format bfp --rounding_mode stoc --mant_bits 8 --bfp_tile_size 0 --weight_mant_bits 16 --device {c,g}pu
```

## CNNs
Training one of the CNN models with the selected dataset:
```
python main.py --type cnn --arch resnet50 --data cifar100 \
--lr 0.1 --lr_decay True \
--lr_decay_epochs 150,225 --num_epochs 300 \
--use_nesterov True --momentum 0.9 --weight_decay 1e-4 \
--batch_size 128 --avg_model True  --lr_warmup True \
--num_workers 2 --eval_freq 1  --reshuffle_per_epoch True \
--lr_lars False --lr_scale True \
--world_size 2 --device gpu --save_all_models True \
--num_format bfp --rounding_mode stoc --mant_bits 7 --bfp_tile_size 24  --weight_mant_bits 15
```
We have implemented resnet, wideresnet, densenet, alexnet and alexnet_bn models to be trained on CIFAR10, CIFAR100, ImageNet, and SVHN datasets. For resnet, wideresnet, and densenet models, the model size/depth should be indicated after the model name (e.g., --arch resnet50).

We have provided all the required functions to preprocess the datasets, so passing them as arguments will be enough except ImageNet data. For ImageNet data, after downloading and extracting the dataset, the following lines should be run before the training:
```
python cnn/dataset/build_sequential_data.py --data_dir <path to raw ImageNet data> --data_type train
python cnn/dataset/build_sequential_data.py --data_dir <path to raw ImageNet data> --data_type val
```
We also have provided several scripts for training provided CNN models in [`cnn/scripts`](./cnn/scripts) directory, as well as scripts for HBFP and FP32-only training.

## LSTM
LSTM implementation is adapted from Salesforce's [LSTM and QRNN Language Model Toolkit](https://github.com/salesforce/awd-lstm-lm).

Inside the `lstm/` folder, run `getdata.sh` to acquire the Penn Treebank and WikiText-2 datasets. This step is essential for training.

Training the LSTM model with the Penn Treebank dataset:
```
CUDA_VISIBLE_DEVICES=0 python main.py --type lstm --batch_size 20 \
--data data/penn --dropouti 0.4 --dropouth 0.25 \
--seed 141 --epoch 500 --save bfp8.pt \
--num_format bfp --mant_bits 8 --bfp_tile_size 24 --weight_mant_bits 16
```

We again have provided several scripts for training the model in [`lstm/scripts`](./lstm/scripts) directory. Scripts need to be called from the main hbfp project directory (e.g., `bash lstm/scripts/run_bfp8.sh`).

## BERT
For HBFP training of the BERT model we adapted NVIDIA's [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/) project.

Check the `Megatron-LM_HBFP/examples/` folder for the currently available commands for pretraining a BERT model with HBFP.

## Contributors
Simla Burcu HARMA,
Mario DRUMOND,
Tao LIN.

When using this code, please cite the paper:
```bibtex
@inproceedings{drumond2018training,
  title     = {Training DNNs with hybrid block floating point},
  author    = {Drumond, Mario and Lin, Tao and Jaggi, Martin and Falsafi, Babak},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2018}
}
```
