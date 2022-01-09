#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
from torch import nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, args):
        super(LeNet5, self).__init__()
        # 1st convolutional layer, output size 28*28*6
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        # 1st pooling layer, output size 14*14*6 (we can also try maxpooling or minpooling while LeNet uses Avgpooling)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 2nd conv layer, out size 10*10*16
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 2nd pooling layer, out size 16*5*5
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 3 fully connected layers
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(torch.tanh(self.conv1(x))) # we can also try leaky relu
        x = self.pool2(torch.tanh(self.conv2(x)))
        # Flatten all dimensions
        x = torch.flatten(x, 1)
        #x = x.view(-1, 16 * 5 * 5)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class CNNCifar(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(CNNCifar, self).__init__()
        # 1st convolutional layer, output size 28*28*6
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.batch1 = nn.BatchNorm2d(num_features=64)
        # 1st pooling layer, output size 14*14*6 (we can also try maxpooling or minpooling while LeNet uses Avgpooling)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, dilation=1)
        # 2nd conv layer, out size 10*10*16

        #basic block 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1)
        self.conv21 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batch21 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1)
        #basic block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batch3 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1)
        self.conv31 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batch31 = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1)

        # basic block 4
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.batch4 = nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1)
        self.conv41 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.batch41 = nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1)

        # basic block 5
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.batch5 = nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1)
        self.conv51 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.batch51 = nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1)

        #final block
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=2)
        self.batch6 = nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1)
        self.conv61 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=1, padding=1)
        self.batch61 = nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1)

        self.avgPool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        # 2nd pooling layer, out size 16*5*5
        """ self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.batch3 = nn.BatchNorm2d(num_features=32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        """

        # fully connected layer
        self.fc1 = nn.Linear(in_features=512, out_features=10)
        #self.fc2 = nn.Linear(in_features=l1, out_features=l2)
        #self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        # Note that relu6 is performing better than relu activation function! (see paper)
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.leaky_relu(x)
        #x = self.drop1(x)
       # x = self.pool1(F.leaky_relu(x)) # we can also try leaky relu
        x = self.conv2(x)
        x = self.batch2(x)
        x = F.leaky_relu(x)
        x = self.pool1(x)
        #x = self.drop1(x)
        x = self.conv3(x)
        x = self.batch3(x)
        x = F.leaky_relu(x)
        x = self.conv31(x)
        x = self.batch31(x)
        x = F.leaky_relu(x)
        #x = self.pool2(F.leaky_relu(x))
        #x = self.drop2(x)
        x = self.conv4(x)
        x = self.batch4(x)
        x = F.leaky_relu(x)
        x = self.conv41(x)
        x = self.batch41(x)
        x = F.leaky_relu(x)
        x = self.conv5(x)
        x = self.batch5(x)
        x = F.leaky_relu(x)
        x = self.conv51(x)
        x = self.batch51(x)
        x = F.leaky_relu(x)
        x = self.conv6(x)
        x = self.batch6(x)
        x = F.leaky_relu(x)
        x = self.conv61(x)
        x = self.batch61(x)

        x = self.avgPool(F.leaky_relu(x))

        # Flatten all dimensions
        x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        #x = F.leaky_relu(self.fc2(x))
        #x = self.fc3(x)

        return F.log_softmax(x, dim=1)

"""
class CNNCifar(nn.Module):
    def __init__(self):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
"""