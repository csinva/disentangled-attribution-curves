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

# generates y (implicitly requires func_num)
def generate_y(X):
    if func_num == 1:
        y = np.power(np.pi, X[:, 0] * X[:, 1]) * np.sqrt(2 * np.abs(X[:, 2])) - np.arcsin(.5 * X[:, 3])
#         y += np.log(np.abs(X[:, 2] + X[:, 4]) + 1) + X[:, 8]/(1 + np.abs(X[:, 9])) * np.sqrt(np.abs(X[:, 6])/(np.abs(X[:, 7]) + 1))
#         y -= X[:, 1] * X[:, 6]
    return y

# get means and covariances
def get_means_and_cov(num_vars):
    means = np.zeros(num_vars)
    inv_sum = num_vars
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
def get_X_y(means, covs, num_points=70000):
    X = np.random.multivariate_normal(means, covs, (num_points,))
    no_outliers = np.logical_and(np.all(X <= 2, axis=1), np.all(X >= -2, axis=1))
    # print(np.count_nonzero(no_outliers))
    X = X[no_outliers]
    y = generate_y(X)

    mask = np.isnan(y)
    # print(mask)
    X = X[~mask]
    y = y[~mask]

    # put X into a dataframe
    feats = ["x" + str(i + 1) for i in range(means.size)]
    df = pd.DataFrame(X, columns=feats)
    
    return X, y, df

# fit an rf
def fit_model(X, y, X_test, y_test, out_dir, func_num):
    print('fitting model...')
    forest = RandomForestRegressor(n_estimators=50)
    forest.fit(X, y)
    preds = forest.predict(X_test)
    test_mse = np.mean((y_test - preds) ** 2)
    return forest, test_mse

# calculate expectation, dac, and pdp curves
def calc_curves(X, y, df, num_vars, X_test, y_test, X_cond, y_cond, out_dir, func_num):
    print('calculating curves...')
    curves = {}
    for i in tqdm(range(num_vars)):
        curves_i = {}
        S = np.zeros(num_vars)
        S[i] = 1
        exp = conditional1D(X_cond, y_cond, S, np.arange(-1, 1, .01), .01)
        curves_i['exp'] = exp
        curve = make_curve_forest(forest, X, y, S, (-1, 1), .01, C = 1, continuous_y = True)
        curves_i['dac'] = curve
        pdp_xi = pdp.pdp_isolate(model=forest, dataset=df, model_features=feats, feature=feats[i], num_grid_points=200).pdp
        curves_i['pdp'] = pdp_xi
        curves[i] = deepcopy(curves_i)
        pkl.dump(curves, open(oj(out_dir, f'curves_1d_{func_num}.pkl'), 'wb'))
    print("complete!")
    
if __name__ == '__main__':
    
    # pick the func num
    func_num = 1
    if len(sys.argv) > 1: # first arg (the func_num)
        func_num = int(sys.argv[1])
    print('func num', func_num)
    
    # generate data
    seed = 1
    np.random.seed(seed)
    num_vars = 5
    out_dir = 'sim_results'
    means, covs = get_means_and_cov(num_vars)
    X, y, df = get_X_y(means, covs, num_points=70000)
    X_test, y_test, _ = get_X_y(means, covs, num_points=1000000)
    X_cond, y_cond, _ = get_X_y(means, covs, num_points=15000000)
    
    # fit model
    forest, test_mse = fit_model(X, y, X_test, y_test, out_dir, func_num)
    pkl.dump({'rf': forest, 'test_mse': test_mse}, open(oj(out_dir, f'model_{func_num}.pkl'), 'wb'))
    
    # calc curves
    calc_curves(X, y, df, num_vars, X_test, y_test, X_cond, y_cond, out_dir, func_num)
    print('done!')
    