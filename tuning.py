#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# - smart progress bar

from torch.utils.data import DataLoader
from utils import get_dataset_tune

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split



class CNNCifar_tune(nn.Module):
    def __init__(self, l1=120, l2=84, k1=5, k2=5, d1=0.5, d2=0.5, d3=0.5, d4=0.5, p1=2, p2=2):
        super(CNNCifar_tune, self).__init__()
        # 1st convolutional layer, output size 28*28*6
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)#k1)
        self.batch1 = nn.BatchNorm2d(num_features=6)
        #self.drop1 = nn.Dropout2d(p=d1)
        # 1st pooling layer, output size 14*14*6 (we can also try maxpooling or minpooling while LeNet uses Avgpooling)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2nd conv layer, out size 10*10*16
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)#k2)
        self.batch2 = nn.BatchNorm2d(num_features=16)
        #self.drop2 = nn.Dropout2d(p=d2)
        # 2nd pooling layer, out size 16*5*5
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 3 fully connected layers
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=l1)
        self.drop3 = nn.Dropout2d(p=d3)
        self.fc2 = nn.Linear(in_features=l1, out_features=l2)
        self.drop4 = nn.Dropout2d(p=d4)
        self.fc3 = nn.Linear(l2, 10)

    def forward(self, x):
        # Note that relu6 is performing better than relu activation function! (see paper)
        x = self.conv1(x)
        x = self.batch1(x)
        #x = self.drop1(x)
        x = self.pool1(F.leaky_relu(x)) # we can also try leaky relu
        x = self.conv2(x)
        x = self.batch2(x)
        #x = self.drop2(x)
        x = self.pool2(F.leaky_relu(x))
        # Flatten all dimensions
        x = torch.flatten(x, 1)
        #x = x.view(-1, 16 * 5 * 5)
        x = F.leaky_relu(self.fc1(x))
        x = self.drop3(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.drop4(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def train_cifar(config, checkpoint_dir=None, data_dir=None):
    net = CNNCifar_tune(config["l1"],
                        config["l2"],
                        #config["k1"],
                        #config["k2"],
                        #config["d1"],
                        #config["d2"],
                        config["d3"],
                        config["d4"],
                        #config["p1"],
                        #config["p2"])
                        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)#config["mnt"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_dataset, test_dataset = get_dataset_tune()

    test_abs = int(len(train_dataset) * 0.8)
    train_subset, val_subset = random_split(
        train_dataset, [test_abs, len(train_dataset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")

def test_accuracy(net, device="cpu"):
    train_dataset, test_dataset = get_dataset_tune()

    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    train_dataset, test_dataset = get_dataset_tune()
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(4, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(4, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        #"mnt": tune.grid_search[(0.5, 0.7, 0.9)],
        "batch_size": tune.choice([2, 4, 8, 16, 32, 64]),
        "d4": tune.grid_search([0.0, 0.5, 0.8, 0.9]),
        "d3": tune.choice([0.0, 0.5, 0.8, 0.9]),
        #"d2": tune.grid_search([0.0, 0.5, 0.8, 0.9]),
        #"d1": tune.grid_search([0.0, 0.5, 0.8, 0.9])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_cifar, data_dir='./data'),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = CNNCifar_tune(best_trial.config["l1"], best_trial.config["l2"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if gpus_per_trial > 1:
        best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)
