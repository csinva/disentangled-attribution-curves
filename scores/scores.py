from tqdm import tqdm
import pmlb as dsets
import numpy as np
import pickle as pkl
from os.path import join as oj
from copy import deepcopy
import pandas as pd

import sklearn
import eli5


# arg1 - trained rf
# arg2 - type of importance ('mdi', 'mda', our_thing')
# some methods (mda, our thing) require X and Y for calculating importance
    # passing in training data / testing data have different implications
def get_importance_scores(model, score_type='mdi', X=None, Y=None):
    if score_type == 'mdi':
        return model.feature_importances_
    elif score_type == 'mda':
        return eli5.sklearn.PermutationImportance(model).fit(X, Y).feature_importances_
    else:
        raise NotImplementedError(f'{score_type} not implemented')
        
# arg1 -  train_x: N x p
# arg2 - train_y: N x 1
# returns: list of idxs p ranked by how important each of them are - each in the range [0, p)
def rank_features(train_X, train_y):
    return

# arg1 -  train_x: N x p
# arg2 - train_y: N x 1
# returns: list of tuples of idxs ranking how important each of the pairwise interactions are - each in the range [0, p)
def rank_pairwise_interactions(train_X, train_y):
    raise NotImplementedError

# arg1 -  train_x: N x p
# arg2 -  test_X: N x p
# arg3 - train_y: N x 1
# arg4 - test_y: N x 1
# idxs - list of either individual features or pairs of features to add to the logistic regression
# returns: train + test accuracy after adding each of the idxs
def add_features_and_train_logistic(train_X, test_X, train_y, test_y, idxs):
    raise NotImplementedError