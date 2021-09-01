#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
from torch import nn
import torch.nn.functional as F


#class MLP(nn.Module):
 #   def __init__(self, dim_in, dim_hidden, dim_out):
  #      super(MLP, self).__init__()
   #     self.layer_input = nn.Linear(dim_in, dim_hidden)
    #    self.relu = nn.ReLU()
     #   self.dropout = nn.Dropout()
      #  self.layer_hidden = nn.Linear(dim_hidden, dim_out)
       # self.softmax = nn.Softmax(dim=1)

    #def forward(self, x):
     #   x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
      #  x = self.layer_input(x)
       # x = self.dropout(x)
        #x = self.relu(x)
        #x = self.layer_hidden(x)
        #return self.softmax(x)


# - Da capire cosa succede nella rete!!!
#class CNNMnist(nn.Module):
 #   def __init__(self, args):
  #      super(CNNMnist, self).__init__()
   #     self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
    #    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
     #   self.conv2_drop = nn.Dropout2d()
      #  self.fc1 = nn.Linear(320, 50)
       # self.fc2 = nn.Linear(50, args.num_classes)

    #def forward(self, x):
     #   x = F.relu(F.max_pool2d(self.conv1(x), 2))
      #  x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
       # x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        #return F.log_softmax(x, dim=1)


#class CNNFashion_Mnist(nn.Module):
 #   def __init__(self, args):
  #      super(CNNFashion_Mnist, self).__init__()
   #     self.layer1 = nn.Sequential(
    #        nn.Conv2d(1, 16, kernel_size=5, padding=2),
     #       nn.BatchNorm2d(16),
      #      nn.ReLU(),
       #     nn.MaxPool2d(2))
        #self.layer2 = nn.Sequential(
         #   nn.Conv2d(16, 32, kernel_size=5, padding=2),
          #  nn.BatchNorm2d(32),
           # nn.ReLU(),
            #nn.MaxPool2d(2))
        #self.fc = nn.Linear(7*7*32, 10)

    #def forward(self, x):
     #   out = self.layer1(x)
      #  out = self.layer2(out)
       # out = out.view(out.size(0), -1)
        #out = self.fc(out)
        #return out


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
        x = self.pool1(F.relu(self.conv1(x))) # we can also try leaky relu
        x = self.pool2(F.relu(self.conv2(x)))
        # Flatten all dimensions
        x = torch.flatten(x, 1)
        #x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        # 1st convolutional layer, output size 28*28*6
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.drop1 = nn.Dropout2d(p=0.5)
        # 1st pooling layer, output size 14*14*6 (we can also try maxpooling or minpooling while LeNet uses Avgpooling)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2nd conv layer, out size 10*10*16
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.drop2 = nn.Dropout2d(p=0.5)
        # 2nd pooling layer, out size 16*5*5
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 3 fully connected layers
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Note that relu6 is performing better than relu activation function! (see paper)
        x = self.conv1(x)
        x = self.drop1(x)
        x = self.pool1(F.relu6(x)) # we can also try leaky relu
        x = self.conv2(x)
        #x = self.drop2(x)
        x = self.pool2(F.relu6(x))
        # Flatten all dimensions
        x = torch.flatten(x, 1)
        #x = x.view(-1, 16 * 5 * 5)
        x = F.relu6(self.fc1(x))
        x = F.relu6(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

#class modelC(nn.Module):
#    def __init__(self, input_size, n_classes=10, **kwargs):
 #       super(AllConvNet, self).__init__()
  #      self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
   #     self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
    #    self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
     #   self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
      #  self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
       # self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
        #self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        #self.conv8 = nn.Conv2d(192, 192, 1)

        #self.class_conv = nn.Conv2d(192, n_classes, 1)


    #def forward(self, x):
     #   x_drop = F.dropout(x, .2)
      #  conv1_out = F.relu(self.conv1(x_drop))
       # conv2_out = F.relu(self.conv2(conv1_out))
        #conv3_out = F.relu(self.conv3(conv2_out))
        #conv3_out_drop = F.dropout(conv3_out, .5)
        #conv4_out = F.relu(self.conv4(conv3_out_drop))
        #conv5_out = F.relu(self.conv5(conv4_out))
        #conv6_out = F.relu(self.conv6(conv5_out))
        #conv6_out_drop = F.dropout(conv6_out, .5)
        #conv7_out = F.relu(self.conv7(conv6_out_drop))
        #conv8_out = F.relu(self.conv8(conv7_out))

        #class_out = F.relu(self.class_conv(conv8_out))
        #pool_out = F.adaptive_avg_pool2d(class_out, 1)
        #pool_out.squeeze_(-1)
        #pool_out.squeeze_(-1)
        #return pool_out
