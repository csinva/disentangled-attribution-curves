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

def generate_y(X):
    y = np.power(np.pi, X[:, 0] * X[:, 1]) * np.sqrt(2 * np.abs(X[:, 2])) - np.arcsin(.5 * X[:, 3])
    y += np.log(np.abs(X[:, 2] + X[:, 4]) + 1) + X[:, 8]/(1 + np.abs(X[:, 9])) * np.sqrt(np.abs(X[:, 6])/(np.abs(X[:, 7]) + 1))
    y -= X[:, 1] * X[:, 6]
    return y

# get means and covariances
def get_means_and_cov(num_vars):
    means = np.zeros(num_vars)
    inv_sum = 10
    eigs = []
    while(len(eigs) <= 8):
        eig = np.random.uniform(0, inv_sum)
        eigs.append(eig)
        inv_sum -= eig
    eigs.append(10 - np.sum(eigs))
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
    feats = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"]
    df = pd.DataFrame(X, columns=feats)
    
    return X, y, df


def fit_model(X, y, X_test, y_test, out_dir, func_num):
    forest = RandomForestRegressor(n_estimators=50)
    forest.fit(X, y)
    preds = forest.predict(X_test)
    test_mse = np.mean((y_test - preds) ** 2)
    return forest, test_mse

def calc_curves(X, y, df, X_test, y_test, X_cond, y_cond, out_dir, func_num):
    curves = {}
    for i in tqdm(range(10)): #range(3, 10):
        curves_i = {}
        S = np.zeros(10)
        S[i] = 1
        exp = conditional1D(X_cond, y_cond, S, np.arange(-1, 1, .01), .01)
        curves_i['exp'] = exp
        print("conditional expectation complete")
        curve = make_curve_forest(forest, X, y, S, (-1, 1), .01, C = 1, continuous_y = True)
        curves_i['dac'] = curve
        print("DAC curve complete")
        pdp_xi = pdp.pdp_isolate(model=forest, dataset=df, model_features=feats, feature=feats[i], num_grid_points=200).pdp
        curves_i['pdp'] = pdp_xi
        print("PDP curve complete")
        curves[i] = deepcopy(curves_i)
        pkl.dump(curves, open(oj(out_dir, f'curves_1d_{func_num}.pkl'), 'wb'))
    print("complete!")