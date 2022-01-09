import torch
from torch.utils.data import Subset
import numpy as np
from torch.utils.data import Dataset, DataLoader

from utils import get_dataset
from options import args_parser
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

class my_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        image, label = self.dataset[self.idxs[idx]]
        return torch.tensor(image), torch.tensor(label)

    def __len__(self):
        return len(self.indices)


def shared_dataset(dataset, num_images, num_users, beta, args):

    # uniformly random selection of images to use as shared data
    all_idxs = torch.linspace(0, len(dataset)-1, len(dataset)).int()
    print(all_idxs)
    shuffled_idxs = all_idxs[torch.randperm(len(all_idxs))]
    selected_idxs = shuffled_idxs[:num_images]
    users_idxs = shuffled_idxs[num_images:]

    # retrieve selected images
    users_set = Subset(dataset, users_idxs)
    selected_set = Subset(dataset, selected_idxs)

    dict_user = {}

    if args.iid:


        images_per_user = int(len(users_set)/num_users)

        for i in range(num_users):
            el_i = np.random.choice(users_idxs, images_per_user, replace=False)
            users_idxs = list(set(users_idxs) ^ set(el_i))
            el_i_shared = np.random.choice(selected_idxs, int(len(selected_set)*beta), replace=False)
            final_el = np.concatenate((el_i, el_i_shared), axis=0)
            dict_user[i] = set(final_el)
    else:
        #loader_users = DataLoader(users_set, batch_size=len(users_set), shuffle=False)


        idxs = np.random.permutation(np.array(dataset.targets).shape[0])
        min_size = 0
        while min_size < 10:
            # np.random.seed(4)

            proportions = np.random.dirichlet(np.repeat(args.alpha * 100, num_users))
            proportions = proportions / proportions.sum()
            min_size = np.min(proportions * len(idxs))
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs, proportions)

        for i in range(num_users):
            el_i_shared = np.random.choice(selected_idxs, int(len(selected_set) * beta), replace=False)
            final_el = np.concatenate((batch_idxs[i], el_i_shared), axis=0)
            dict_user[i] = set(final_el)

    return selected_set, users_set, dict_user

"""
args = args_parser()
train_dataset, test_dataset, _ = get_dataset(args)

train_dataset_global, train_dataset, users_group = shared_dataset(train_dataset,
                                                                          num_images=1000,
                                                                          num_users=args.num_users,
                                                                          alpha=0.1)
"""



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