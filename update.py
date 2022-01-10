#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Default criterion set to NLL loss function
        if (self.args.model == 'resnet'):
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        else:
            self.criterion = nn.NLLLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False, drop_last=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False, drop_last=True)
        return trainloader, validloader, testloader

    def fed_avg(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = 'cuda' if args.gpu else 'cpu'
    if args.model == 'resnet':
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = nn.NLLLoss().to(device)

    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        # - Returns a namedtuple (values, indices) where values is the maximum value of each row of the input tensor
        # - in the given dimension dim. And indices is the index location of each maximum value found (argmax).
        _, pred_labels = torch.max(outputs, 1)

        # -  If there is any situation that you don't know how many rows you want but are sure of the number of columns,
        # - then you can specify this with a -1.
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss


def test(model, test_dataset, args):
    """
    Evaluate model performance on test
    """
    # put model in evaluation mode
    model.eval()
    test_loss = 0.0
    correct = 0
    # pass model to gpu if available
    if args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        device = torch.device("cpu")

    # create dataloader
    testloader = DataLoader(test_dataset, batch_size=64,
                            shuffle=False)

    for i, (data, label) in enumerate(testloader):
        data, label = data.to(device), label.to(device)
        probs = model(data)
        # sum up batch loss
        if args.model == 'cnn' or args.model == 'lenet':
            test_loss += nn.functional.nll_loss(probs, label, reduction='sum').item()
        elif args.model =='resnet':
            test_loss += nn.functional.cross_entropy(probs, label, reduction='sum').item()

        y_pred = probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(label.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(testloader.dataset)
    accuracy = 100.00 * correct / len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss,
        correct,
        len(testloader.dataset),
        accuracy))

    return correct, test_loss

class DatasetSplit_1(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUp(object):
    def __init__(self, args, dataset=None, idxs=None, round=None):
        self.round = round
        self.args = args
        self.selected_clients = []
        self.ldataloader_train = DataLoader(DatasetSplit_1(dataset, idxs),
                                            batch_size=self.args.local_bs,
                                            shuffle=True)
        # select loss
        if self.args.model == 'cnn' or self.args.model == 'lenet':
            self.loss_func = nn.NLLLoss()
        elif self.args.model == 'resnet':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            exit('Error: model not defined, impossible setting loss')

        # define device
        if args.gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

    def train(self, model):
        model.train()
        # Train and update locally
        # setting optimizer
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldataloader_train):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                probs = model(images)
                loss = self.loss_func(probs, labels)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('Round: {} Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        self.round,
                        iter,
                        batch_idx * len(images),
                        len(self.ldataloader_train.dataset),
                        100. * batch_idx / len(self.ldataloader_train),
                        loss.item()))
                batch_loss.append(loss.item())
            # append average loss to epoch loss
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        #return model and average epoch loss
        average_ep_loss = sum(epoch_loss)/len(epoch_loss)

        return model.state_dict(), average_ep_loss


def FedAvg_1(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

