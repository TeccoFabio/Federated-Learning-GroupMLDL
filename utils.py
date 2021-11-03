#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import cifar_iid, cifar_noniid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    data_dir = '../data/cifar/'
    transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR10(root=data_dir,
                                     train=True,
                                     transform=transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root=data_dir,
                                    train=False,
                                    transform=transform,
                                    download=True)

    # sample training data amongst users
    if args.iid:
        # Sample IID user data from Mnist
        user_groups = cifar_iid(train_dataset, args.num_users)
    else:
        # Sample Non-IID user data from Mnist
        if args.unequal:
            # Chose uneuqal splits for every user
            # - TODO??
            raise NotImplementedError()

        else:
            # Chose equal splits for every user
            user_groups = cifar_noniid(train_dataset, args.num_users, args.alpha)

    return train_dataset, test_dataset, user_groups

def get_dataset_tune():
    """ Returns train and test datasets that can be used to tune the model
    """
    data_dir = '../data/cifar/'
    transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR10(root=data_dir,
                                     train=True,
                                     transform=transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root=data_dir,
                                    train=False,
                                    transform=transform,
                                    download=True)
    return train_dataset, test_dataset


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
            # Divides each element of the input input by the corresponding element of other.
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def central_mod_details(args):
    print('\nExperimental details:')
    print(f'   Model   : {args.model}')
    print(f'   Optimizer   : {args.optimizer}')
    print(f'   Learning Rate   : {args.lr}')
    print(f'   Global Rounds   : {args.epochs}')
    return