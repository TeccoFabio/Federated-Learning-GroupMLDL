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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.batch1 = nn.BatchNorm2d(num_features=6)
        self.drop1 = nn.Dropout2d(p=0.5)
        # 1st pooling layer, output size 14*14*6 (we can also try maxpooling or minpooling while LeNet uses Avgpooling)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2nd conv layer, out size 10*10*16
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.batch2 = nn.BatchNorm2d(num_features=16)
        self.drop2 = nn.Dropout2d(p=0.5)
        # 2nd pooling layer, out size 16*5*5
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 3 fully connected layers
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=l1)
        self.fc2 = nn.Linear(in_features=l1, out_features=l2)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        # Note that relu6 is performing better than relu activation function! (see paper)
        x = self.conv1(x)
        x = self.batch1(x)
        #x = self.drop1(x)
        x = self.pool1(F.leaky_relu(x)) # we can also try leaky relu
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.drop2(x)
        x = self.pool2(F.leaky_relu(x))
        # Flatten all dimensions
        x = torch.flatten(x, 1)
        #x = x.view(-1, 16 * 5 * 5)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
