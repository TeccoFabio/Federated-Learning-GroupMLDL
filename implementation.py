import torch
from torch.utils.data import Subset
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