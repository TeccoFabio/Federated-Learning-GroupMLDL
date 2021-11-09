import torch
from torch.utils.data import Subset
import numpy as np
all_idxs = torch.linspace(0,25000-1,25000)
all_idxs = all_idxs.int()
print(all_idxs.size())
print(all_idxs.shape[0])
print(all_idxs)
selected_idxs = torch.randint(0, all_idxs.shape[0], (5,))
print(selected_idxs)
shuffled_idxs = all_idxs[torch.randperm(all_idxs.size()[0])]
print(shuffled_idxs)
selected_idxs = shuffled_idxs[:3]
rest_idxs = shuffled_idxs[3:]
print(selected_idxs)
print(rest_idxs)

def shared_dataset(dataset, num_images):
    # uniformly random selection of images to use as shared data
    all_idxs = torch.linspace(0, dataset.shape[0]-1, dataset.shape[0]).int()
    shuffled_idxs = all_idxs[torch.randperm(all_idxs.shape()[0])]
    selected_idxs = shuffled_idxs[:num_images]
    users_idxs = shuffled_idxs[num_images:]
    # retrieve selected images
    selected_set = Subset(dataset, selected_idxs)
    users_set = Subset(dataset, users_idxs)
    return selected_set, users_set

#aggiungere al dict di ogni client parte di selected_set (alpha)

def cifar_noniid_implementation(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    # - It returns an array of indices of the same shape as a that index data along the given axis in sorted order.
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users