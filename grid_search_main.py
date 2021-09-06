# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from models import CNNCifar
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from utils import get_dataset
from options import args_parser
from update import test_inference
from models import CNNCifar

if __name__ == '__main__':
    args = args_parser()
    if args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

# Function to create model, required for KerasClassifier
global_model = CNNCifar(args=args)
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = get_dataset(args)
# split into input (X) and output (Y) variables
all_idxs = torch.linspace(0, dataset.shape[0]-1, dataset.shape[0])
first_half_idxs = all_idxs[:dataset.shape[0]/2]
second_half_idxs = all_idxs[dataset.shape[0]/2:]
X = dataset[first_half_idxs]
Y = dataset[second_half_idxs]
# create model
model = global_model
# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
# add local batch_size
epochs = [10, 50]
# add local_epochs
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# add learning_rate
# add momentum
# add alpha_Dirichlet
# add num_users
# add frac_C
# add activation function (relu, leaky relu, relu6, etc.)
# add random_seed
param_grid = dict(batch_size=batch_size, epochs=epochs, dropout_rate=dropout_rate, weight_constraint=weight_constraint)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))