import copy

import matplotlib.pyplot as plt

from options import args_parser
from utils import get_dataset, exp_details
import os
import torchvision
from tensorboardX import SummaryWriter
from models import CNNCifar, LeNet5
import torch
from torch import nn
import numpy as np
from update import LocalUp, FedAvg_1, test
from implementation import shared_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchsummary import summary

if __name__ == '__main__':
    #parse args and define paths
    args = args_parser()
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')
    all_clients = True

    args = args_parser()
    exp_details(args)

    # load dataset and split users
    train_dataset, test_dataset, users_group = get_dataset(args=args)

    # build model
    if args.model == 'cnn':
        glob_model = CNNCifar()
    elif args.model == 'lenet':
        glob_model = LeNet5(args=args)
    elif args.model == 'resnet':
        glob_model = torchvision.models.resnet18(pretrained=False)
        glob_model.fc = torch.nn.Sequential(torch.nn.Linear(512, 10))
    else:
        exit('Error: model not defined')

    # send model to gpu if possible
    if args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        glob_model.to(device)
    else:
        device = torch.device("cpu")

    # Show model's details and set in training mode
    summary(glob_model, (3, 32, 32))
    glob_model.train()

    if args.update:
        train_dataset_global, _, users_group = shared_dataset(train_dataset,
                                                                          num_images=1000,
                                                                          num_users=args.num_users,
                                                                          beta=0.1,
                                                                            args=args)
        trainloader_global = DataLoader(train_dataset_global,
                             batch_size=16,
                             shuffle=True,
                             num_workers=2)

        # train global model with selected trainloader_global
        optimizer = torch.optim.Adam(glob_model.parameters(), lr=args.lr)
        if args.model == 'resnet':
            criterion = nn.CrossEntropyLoss().to(device)
        else:
            criterion = nn.NLLLoss().to(device)

        epoch_loss = []

        for epoch in tqdm(range(20)):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(trainloader_global):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = glob_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if batch_idx % 50 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch + 1, batch_idx * len(images), len(trainloader_global.dataset),
                        100. * batch_idx / len(trainloader_global), loss.item()))
                batch_loss.append(loss.item())

            loss_avg = sum(batch_loss) / len(batch_loss)
            print('\nTrain loss:', loss_avg)
            epoch_loss.append(loss_avg)

        # save model
        save_path = "../save/"
        try:
            os.mkdir(save_path)
        except FileExistsError:
            pass
        torch.save(glob_model.state_dict(), save_path + '/model_{}_epochs{}'.format(args.model, args.epochs))

    else:
        pass


    # copy weights
    w_glob = glob_model.state_dict()

    #TRAINING
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    best_net = None
    best_loss = None
    val_acc_list, net_list = [], []

    if all_clients:
        print('Aggregation over all clients')
        w_local = [w_glob for i in range(args.num_users)]

    for iter in range(args.epochs):
        # initialize local loss list
        loss_local = []

        if not all_clients:
            w_local = []

        m = max(int(args.frac*args.num_users), 1)
        # select m random users between all users without repetition
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)


        for idx in idxs_users:
            local = LocalUp(args=args, dataset=train_dataset, idxs=users_group[idx])
            w, loss = local.train(model=copy.deepcopy(glob_model).to(device))

            if all_clients:
                w_local[idx] = copy.deepcopy(w)
            else:
                w_local.append(copy.deepcopy(w))

            loss_local.append(copy.deepcopy(loss))

        # Update global weights (syncronous)
        w_glob = FedAvg_1(w_local)

        # Copy obtained weight to glob_model
        glob_model.load_state_dict(w_glob)

        # Print average  Loss
        loss_avg = sum(loss_local)/len(loss_local)
        loss_train.append(loss_avg)
        print('Round {:3d}, Average Loss {:3f}'.format(iter, loss_avg))

    # Plot loss curve
    save_path = "../save/"
    try:
        os.mkdir(save_path)
    except FileExistsError:
        pass

    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(loss_train)), loss_train, color='b')
    plt.ylabel('Train Loss')
    plt.xlabel('Communication Rounds')
    plt.savefig(save_path+'fed_{}_{}_C{}_iid{}.png'.format(args.model,
                                                               args.epochs,
                                                               args.frac,
                                                               args.iid))


    # TESTING
    glob_model.eval()
    acc_train, loss_train = test(glob_model, train_dataset, args)
    acc_test, loss_test = test(glob_model, test_dataset, args)
    print('Training accuracy: {:.2f}'.format(acc_train))
    print('Test accuracy: {:.2f}'.format(acc_test))







