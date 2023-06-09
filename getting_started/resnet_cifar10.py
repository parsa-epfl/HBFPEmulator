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
# The ResNet model in this file is based on liukuang’s
# (https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py)
# which is available under an MIT license.
#
# Data processing and test implementation in this file is taken from
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# which is available under BSD 3-Clause License.

"""
Training a ResNet model on CIFAR10 with HBFP
--------------------------------------------
This example does the following steps in order:
1. Load and normalizing the CIFAR10 training and test datasets using ``torchvision``
2. Define a ResNet in HBFP
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data

ResNet implementation for cifar10 is taken from
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
Data processing and test implementation is taken from
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from .bfp_ops import BFPLinear, BFPConv2d, unpack_bfp_args
from .bfp_optim import get_bfp_optim
import torch.optim as optim
from tqdm import tqdm, trange

PATH = './cifar_net.pth'

# 1. Load and normalizing the CIFAR10 training and test datasets using ``torchvision``
def prepare_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainset, trainloader, testset, testloader, classes

# 2. Define a ResNet in HBFP
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bfp_args={}):
        super(BasicBlock, self).__init__()
        self.conv1 = BFPConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, **bfp_args)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BFPConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, **bfp_args)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                BFPConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, **bfp_args),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, bfp_args={}):
        super(Bottleneck, self).__init__()
        self.conv1 = BFPConv2d(in_planes, planes, kernel_size=1, bias=False, **bfp_args)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BFPConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, **bfp_args)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = BFPConv2d(planes, self.expansion*planes, kernel_size=1, bias=False, **bfp_args)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                BFPConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, **bfp_args),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args, num_classes=10):
        super(ResNet, self).__init__()
        self.bfp_args = unpack_bfp_args(dict(vars(args)))
        self.in_planes = 64

        self.conv1 = BFPConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, **self.bfp_args)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,
                               bfp_args=self.bfp_args))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(args):
    return ResNet(BasicBlock, [2,2,2,2], args)

def train(net, trainset, trainloader, testset, testloader, classes, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print("Training on", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    BFPSGD = get_bfp_optim(optim.SGD, "SGD")
    optimizer = BFPSGD(
            net.parameters(),
            lr=0.001, momentum=0.9,
        num_format=args.num_format,
        mant_bits=args.mant_bits,
        weight_mant_bits=args.weight_mant_bits,
        device=args.device)

    for epoch in trange(2, desc='epoch'):
        pass
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader, desc='iteration'), 0):
            pass
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 2000 == 1999:    # print every 2000 mini-batches
                tqdm.write('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')


    torch.save(net.module.state_dict(), PATH)

def test_model(net, trainset, trainloader, testset, testloader, classes, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    net.load_state_dict(torch.load(PATH))
    net.to(device)
    outputs = net(images.to(device))
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    # How the network performs on the whole dataset.
    correct = 0
    total = 0
    print('The accuracy on the test dataset is being calculated...')
    with torch.no_grad():
        for data in tqdm(testloader, desc='test iteration'):
            pass
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('The accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    # What are the classes that performed well
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    print('The accuracy of each class is being calculated...')
    with torch.no_grad():
        for data in tqdm(testloader, desc='test iteration'):
            pass
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('The accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

def resnet18_cifar10(args):
    trainset, trainloader, testset, testloader, classes = prepare_data()
    net = ResNet18(args)
    train(net, trainset, trainloader, testset, testloader, classes, args)
    test_model(net, trainset, trainloader, testset, testloader, classes, args)
