#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# - smart progress bar
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.hub import load
from utils import get_dataset, central_mod_details
from options import args_parser
from update import test_inference, test
from models import CNNCifar, LeNet5


if __name__ == '__main__':
    args = args_parser()
    if args.model == 'lenet':
        global_model = LeNet5(args=args)
    elif args.model == 'cnn':
        global_model = CNNCifar()
    elif args.model == 'resnet':
        global_model = torchvision.models.resnet18(pretrained=False)
        global_model.fc = nn.Sequential(nn.Linear(512, 10))

    if args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        global_model.to(device)
    else:
        device = torch.device("cpu")
    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)
    central_mod_details(args)

    # Training
    # Set optimizer and criterion
    #if args.optimizer == 'sgd':
    #optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    #momentum=0.9)
    #elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr)
                                     #weight_decay=1e-4)

    trainloader = DataLoader(train_dataset,
                             batch_size=64,
                             shuffle=True,
                             num_workers=2)

    # - The negative log likelihood loss. It is useful to train a classification problem with C classes
    if args.model == 'resnet':
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = nn.NLLLoss().to(device)

    epoch_loss = []

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in global_model.state_dict():
        print(param_tensor, "\t", global_model.state_dict()[param_tensor].size())

    for epoch in tqdm(range(args.epochs)):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)

    # testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100 * test_acc))
    print('\n')
    correct, test_loss = test(global_model, test_dataset, args=args)

    #save model and plot
    save_path = "../save/"
    try:
        os.mkdir(save_path)
    except FileExistsError:
        pass
    torch.save(global_model.state_dict(), save_path+'/model_{}_epochs{}'.format(args.model, args.epochs))

    # Plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.title('Train Loss vs. no. of Epochs')
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                 args.epochs))

