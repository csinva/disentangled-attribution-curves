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
# returns: np array of 1 score for each variable
def get_importance_scores(model, score_type='mdi', X=None, Y=None):
    if score_type == 'mdi':
        return model.feature_importances_
    elif score_type == 'mda':
        return eli5.sklearn.PermutationImportance(model, random_state=42, n_iter=4).fit(X, Y).feature_importances_
    else:
        raise NotImplementedError(f'{score_type} not implemented')
