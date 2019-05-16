import numpy as np
from itertools import combinations
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
import matplotlib.pyplot as plt
import sys
sys.path.append('../../dac')
sys.path.append('../dac')
from dac import *
from pdpbox import pdp
import pandas as pd
from scipy.stats import random_correlation
from copy import deepcopy
from os.path import join as oj
from tqdm import tqdm
import pickle as pkl
import seaborn as sns
import os

# generates y (implicitly requires func_num)
def generate_y(X):
    if func_num == 1:
        y = np.power(np.pi, X[:, 0] * X[:, 1]) * np.sqrt(2 * np.abs(X[:, 2])) - np.arcsin(.5 * X[:, 3])
        y += np.log(np.abs(X[:, 2] + X[:, 4]) + 1)
    elif func_num == 2:
        y = np.power(np.pi, X[:, 0] * X[:, 1]) * np.sqrt(2 * np.abs(X[:, 2])) - np.arcsin(.5 * X[:, 3])
        y += np.log(np.abs(X[:, 2] + X[:, 4]) + 1) - X[:, 1] * X[:, 4]
    elif func_num == 3:
        y = np.exp(np.abs(X[:, 0] - X[:, 2])) + np.abs(X[:, 1] * X[:, 2]) - np.power(np.abs(X[:, 2]), 2 * np.abs(X[:, 3]))
        y += np.log(X[:, 3] ** 2 + X[:, 4] ** 2)
    elif func_num == 4:
        y = np.exp(np.abs(X[:, 0] - X[:, 2])) + np.abs(X[:, 1] * X[:, 2]) - np.power(np.abs(X[:, 2]), 2 * np.abs(X[:, 3]))
        y += np.log(X[:, 3] ** 2 + X[:, 4] ** 2) + (X[:, 0] * X[:, 3]) ** 2
    elif func_num == 5:
        y = 1/(1 + X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2) + np.sqrt(np.exp(X[:, 3] + X[:, 4]))
    elif func_num == 6:
        y = np.exp(np.abs(X[:, 1] * X[:, 2] + 1)) - np.exp(np.abs(X[:, 2] + X[:, 4]) + 1) + np.cos(X[:, 4])
    elif func_num == 7:
        y = (np.arctan(X[:, 0]) + np.arctan(X[:, 1])) ** 2 + np.max(X[:, 2] * X[:, 3], 0)
        y -= 1/(1 + (X[:, 3] * X[:, 4]) ** 2) + np.sum(X, axis=1)        
    elif func_num == 8:
        y = X[:, 0] * X[:, 1] + np.power(2, X[:, 2] + X[:, 4]) + np.power(2, X[:, 2] + X[:, 3] + X[:, 4])    
    elif func_num == 9:
        y = np.arctan(X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3]) * np.sqrt(np.abs(X[:, 4]))
        y += np.exp(X[:, 4] + X[:, 0])        
    elif func_num == 10:
        y = np.sinh(X[:, 0] + X[:, 1]) + np.arccos(np.arctan(X[:, 2] + X[:, 4])) + np.cos(X[:, 3] + X[:, 4])
    return y

# get means and covariances
def get_means_and_cov(num_vars, fix_eigs=False):
    means = np.zeros(num_vars)
    inv_sum = num_vars
    if fix_eigs == 'iid':
        eigs = [1] * num_vars    
    elif fix_eigs == True:
        if num_vars == 5:
            eigs = [2, 2, 1, 0, 0]
        elif num_vars == 10:
            eigs = [4, 3, 2, 1, 0, 0, 0, 0, 0, 0]
            print(np.sum(eigs))
    else:
        eigs = []
        while len(eigs) < num_vars - 1:
            eig = np.random.uniform(0, inv_sum)
            eigs.append(eig)
            inv_sum -= eig
        eigs.append(inv_sum)
    covs = random_correlation.rvs(eigs)
    covs = random_correlation.rvs(eigs)
    return means, covs

# get X and Y using means and covs
def get_X_y(means, covs, num_points=70000, use_rf=False, rf=None):
    X = np.random.multivariate_normal(means, covs, (num_points,))
    no_outliers = np.logical_and(np.all(X <= 2, axis=1), np.all(X >= -2, axis=1))
    # print(np.count_nonzero(no_outliers))
    X = X[no_outliers]
    y = generate_y(X)

    mask = np.isnan(y)
    # print(mask)
    X = X[~mask]
    y = y[~mask]
    
    if use_rf:
        y = rf.predict(X)

    # put X into a dataframe
    feats = ["x" + str(i + 1) for i in range(means.size)]
    df = pd.DataFrame(X, columns=feats)
    
    return X, y, df

# fit an rf
def fit_model(X, y, X_test, y_test, rf_or_boosting):
    print('fitting model...')
    if rf_or_boosting == 'rf':
        forest = RandomForestRegressor(n_estimators=50)
    elif rf_or_boosting == 'boosting':
        forest = AdaBoostRegressor(n_estimators=50)
    forest.fit(X, y)
    preds = forest.predict(X_test)
    test_mse = np.mean((y_test - preds) ** 2)
    return forest, test_mse

# calculate expectation, dac, and pdp curves
def calc_curves(X, y, df, X_cond, y_cond, forest, out_dir, func_num, C, rf_or_boosting):
    num_vars = X.shape[1]
    print('calculating curves...')
    curves = {}
    for i in tqdm(range(num_vars)):
        curves_i = {}
        S = np.zeros(num_vars)
        S[i] = 1
        exp = conditional1D(X_cond, y_cond, S, np.arange(-1, 1, .01), .01)
        curves_i['exp'] = exp
#         exp_train = conditional1D(X, y, S, np.arange(-1, 1, .01), .01)
#         curves_i['exp_train'] = exp_train
        if rf_or_boosting == 'rf':
            curve = make_curve_forest(forest, X, y, S, (-1, 1), .01, C=C, continuous_y = True)
        elif rf_or_boosting == 'boosting':
            curve = ada_boosted_curve_forest(forest, X, y, S, (-1, 1), .01, C=C, continuous_y = True)
            
        curves_i['dac'] = curve
        feats = list(df.keys())
        pdp_xi = pdp.pdp_isolate(model=forest, dataset=df, model_features=feats, feature=feats[i], num_grid_points=200).pdp
        curves_i['pdp'] = pdp_xi
        curves[i] = deepcopy(curves_i)
        pkl.dump(curves, open(oj(out_dir, f'curves_1d_{func_num}.pkl'), 'wb'))
    print("complete!")
    
if __name__ == '__main__':
    
    # hyperparams
    func_num = 1
    seed = 1
    n_train = 70000 # 70000
    num_vars = 5
    fix_eigs = False # False, True, 'iid'
    out_dir = '/scratch/users/vision/chandan/boosting_no_rf/some_corr' # sim_results_fix_cov_C=0.25''
    rf_or_boosting = 'boosting' # 'rf', 'boosting'
    use_rf = False
    C = 1
    
    
    # func_num sys argv
    if len(sys.argv) > 1: # first arg (the func_num)
        func_num = int(sys.argv[1])
    print('func num', func_num)    
    
    # generate data
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    means, covs = get_means_and_cov(num_vars, fix_eigs=True)
    X, y, df = get_X_y(means, covs, num_points=n_train) # 70000
    X_test, y_test, _ = get_X_y(means, covs, num_points=1000000)
    print(y_test.shape)
    
    # fit model
    forest, test_mse = fit_model(X, y, X_test, y_test, rf_or_boosting)
    pkl.dump({'rf': forest, 'test_mse': test_mse}, open(oj(out_dir, f'model_{func_num}.pkl'), 'wb'))
    
    # generate data for conditional
    X_cond, y_cond, _ = get_X_y(means, covs, num_points=15000000, use_rf=use_rf, rf=forest)
    
    # calc curves
    calc_curves(X, y, df, X_cond, y_cond, forest, out_dir, func_num, C, rf_or_boosting)
    print('done!')
    