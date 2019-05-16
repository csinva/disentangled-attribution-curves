import numpy as np
from itertools import combinations
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import sys
sys.path.append('../scores')
from interactions import *
from pdpbox import pdp
import pandas as pd
from scipy.stats import random_correlation
from copy import deepcopy
from os.path import join as oj
from tqdm import tqdm
import pickle as pkl
import seaborn as sns
import os


import pmlb as dsets
import run_sim
from sklearn.model_selection import train_test_split



def get_data(dset_name, data_dir, random_state):
    X, y = dsets.fetch_data(dset_name, return_X_y=True, 
                          local_cache_dir=data_dir)
    feats = ["x" + str(i + 1) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feats)


    # use the smaller dset to train
    X_test, X, y_test, y = train_test_split(X, y, test_size=0.1, random_state=random_state)
    X_cond = X_test
    Y_cond = y_test

    return X, y, df, X_test, y_test, X_cond, Y_cond

if __name__ == '__main__':

    dset_names = deepcopy(dsets.regression_dataset_names)
    data_dir = '/scratch/users/vision/data/pmlb'

    # hyperparams
    seed = 1
    out_dir = '/scratch/users/vision/chandan/rf_sims_real/use_rf_rerun' # sim_results_fix_cov_C=0.25''
    use_rf = True
    C = 1
    dset_num = 0
    random_state = 42 # for each train_test_split


    # dset_num sys argv
    if len(sys.argv) > 1:
        dset_num = int(sys.argv[1])
    print('dset num', dset_num)    
    dset_name = dset_names[dset_num]

    # generate data
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    X, y, df, X_test, y_test, X_cond, y_cond = get_data(dset_name, data_dir, random_state)

    # fit model
    forest, test_mse = run_sim.fit_model(X, y, X_test, y_test)
    pkl.dump({'rf': forest, 'test_mse': test_mse}, open(oj(out_dir, f'model_{dset_num}.pkl'), 'wb'))

    # generate data for conditional
    if use_rf:
        Y_cond = forest.predict(X_cond)

    # calc curves
    run_sim.calc_curves(X, y, df, X_cond, y_cond, forest, out_dir, dset_num, C)
    print('done!')