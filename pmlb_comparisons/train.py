import sys
from tqdm import tqdm
import pmlb as dsets
import numpy as np
import pickle as pkl
from os.path import join as oj
from copy import deepcopy
import pandas as pd
from numpy import array as arr

# sklearn models
sys.path.append('../scores')
import scores
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import interactions
from scipy import interpolate

def fit_logit_and_rfs(dset_names, data_dir, out_dir, random_state=42):
    
    logit_test_scores = []
    rf_test_scores = []
    rfs = []

    for dset_name in tqdm(dset_names):
        X, y = dsets.fetch_data(dset_name, return_X_y=True, 
                          local_cache_dir=data_dir)


        train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=random_state)
        logit = LogisticRegression(solver='liblinear', multi_class='auto', random_state=random_state) # liblinear best for small dsets, otherwise lbfgs
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    #     print(dset_name, X.shape)
        logit.fit(train_X, train_y)
        rf.fit(train_X, train_y)

        logit_test_scores.append(logit.score(test_X, test_y))
        rf_test_scores.append(rf.score(test_X, test_y))
        rfs.append(deepcopy(rf))

    # save
    logit_test_scores = np.array(logit_test_scores)
    rf_test_scores = np.array(rf_test_scores)
    classification_results = {'logit_test_score': logit_test_scores,
               'rf_test_score': rf_test_scores,
               'dset_name': dset_names,
               'rf': rfs}
    pkl.dump(classification_results, 
             open(oj(out_dir, 'classification_results_orig_seeded.pkl'), 'wb'))