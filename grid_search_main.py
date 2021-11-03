# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.model_selection import GridSearchCV
from torch.utils.data import Subset
import torch
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
train_set, test_set, _ = get_dataset(args)
# split into input (X) and output (Y) variables
first_half_idxs = torch.linspace(0, int(len(train_set)/2) - 1, int(len(train_set)/2))
first_half_idxs = first_half_idxs.int()
second_half_idxs = torch.linspace(int(len(train_set)/2), len(train_set) - 1, int(len(train_set)/2))
second_half_idxs = second_half_idxs.int()
X = Subset(train_set, first_half_idxs)
Y = Subset(train_set, second_half_idxs)
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