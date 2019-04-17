import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
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


# right now assumes feat_nums is of length 1
def get_feats(X, feat_nums=[0], add_feature=False):
    feats = None
    if len(feat_nums) == 1:
        feats = X[:, feat_nums[0]].reshape(-1, 1)
    else:
        feats = X[:, feat_num]
    
    return feats
    

def score_logistic_onevar(train_X, train_y, test_X, test_y, feat_nums, f, add_feature):
    logit = LogisticRegression(solver='liblinear', multi_class='auto') # liblinear best for small dsets, otherwise lbfgs

    # full training
#     print('full logit score', row.logit_test_score)

    # get one var at at time
    if not add_feature:
        logit.fit(get_feats(train_X, feat_nums, add_feature), train_y)
        logit_score_orig_onevar = logit.score(get_feats(test_X, feat_nums, add_feature), test_y)


        logit.fit(f(get_feats(train_X, feat_nums, add_feature)), train_y)
        logit_score_altered_onevar = logit.score(f(get_feats(test_X, feat_nums, add_feature)), test_y)
        
    # add new vars to full vars
    elif add_feature:
        logit.fit(get_feats(train_X, feat_nums, add_feature), train_y)
        logit_score_altered_onevar = logit.score(get_feats(test_X, feat_nums, add_feature), test_y)        
        
        logit.fit(np.hstack((train_X, f(get_feats(train_X, feat_nums, add_feature)))), train_y)
        logit_score_orig_onevar = logit.score(np.hstack((test_X, get_feats(test_X, feat_nums, add_feature))), test_y)
    
    return logit_score_orig_onevar, logit_score_altered_onevar    
    
'''
arg1 - trained forest
arg2 - X
arg3 - y
arg4 - S (array of size num_features, 0 to not use this variable otherwise 1)

returns: value of function on a line at regular intervals
'''
def single_var_grid_scores_and_plot(forest, X, y, S, curve_range=None, step=None, plot=True, num_steps=100):
    # deal with params
    if curve_range is None:
        curve_range = (np.min(X), np.max(X))
    step = (curve_range[1] - curve_range[0]) / num_steps
    curve_range = (curve_range[0], curve_range[1] + 10 * step) # do this so we can interpolate properly
    x_axis = np.arange(curve_range[0], curve_range[1], step)
    
    models = forest.estimators_
    length = (curve_range[1] - curve_range[0]) / (1.0 * step)
    line = np.zeros(x_axis.shape[0])
    index = np.nonzero(S)[0][0]
    num_vars = len(S)
    for model in models:
        vals = interactions.traverse_all_paths(model, X, y, S, continuous_y=True)
        line += interactions.make_line(vals, x_axis, step, S)
    line = line / (len(models) * 1.0)
    
    
    if plot:
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6, forward=True)
        plt.plot(x_axis, line, 'k', c='b')
        plt.show()
    return x_axis, line


'''
if add_feature then concatenate the single var
'''
def calc_single_var_scores(results, data_dir, out_dir, random_state, add_feature=False, all_features=False):
    # get dsets where rf outperforms logistic
    idxs_mask = results['rf_test_score'] - results['logit_test_score'] > 0.1 
    # r = results[idxs_mask]
    r = results
    print(results.shape, r.shape)
    print('num idxs after filtering', np.sum(idxs_mask))
    # idxs = np.arange(idxs_mask.size)[idxs_mask] # get actual indexes for this mask

    score_results = {
            'feature_scores_mdi': [],
            'feature_scores_mda': [],
            'logit_score_orig_onevar_list': [],
            'logit_score_altered_onevar_list': [],
        }
    for dset_num in tqdm(range(results.shape[0])): #tqdm(range(2)): #range(r.shape[0]):
        row = r.iloc[dset_num]    

        dset_name = row.dset_name # results['dset_names'][idx_0] #dsets.classification_dataset_names[0]
        X, y = dsets.fetch_data(dset_name, return_X_y=True, 
                          local_cache_dir=data_dir)
        train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=random_state)
        num_features = X.shape[1]
        rf = row.rf
        assert(rf.score(test_X, test_y) == row.rf_test_score) # check that acc matches

        feature_scores_mdi = scores.get_importance_scores(rf, score_type='mdi', X=test_X, Y=test_y)
        feature_scores_mda = scores.get_importance_scores(rf, score_type='mda', X=test_X, Y=test_y)
    #     feature_ranks = 
    #     print(f'feature_scores {feature_scores}\nfeature_ranks {feature_ranks}')

        score_results['feature_scores_mdi'].append(feature_scores_mdi)
        score_results['feature_scores_mda'].append(feature_scores_mda)


        logit_score_orig_onevar_list = []
        logit_score_altered_onevar_list = []
        if all_features:
            feats_to_try = []
        else:
            feats_to_try = [np.argsort(feature_scores_mdi)[-1]]
            
        for i in range(num_features):
            feat_nums = [i] # list of length 1 - longer lists not supported yet
            feat_vals = get_feats(X, feat_nums)
            feat_val_min = np.min(X)
            feat_val_max = np.max(X)
    #         print(f'min {feat_val_min} max {feat_val_max}')

            # appropriate variable to get importance for
            S = np.zeros(num_features)
            S[feat_nums[0]]= 1

            x_axis, scores_on_spaced_line = single_var_grid_scores_and_plot(rf, train_X, train_y, S, (feat_val_min, feat_val_max), plot=False)
            f = interpolate.interp1d(x_axis, scores_on_spaced_line, kind='nearest') # function to interpolate the scores


            logit_score_orig_onevar, logit_score_altered_onevar = score_logistic_onevar(train_X, train_y, test_X, test_y, feat_nums, f, add_feature=add_feature)
            logit_score_orig_onevar_list.append(logit_score_orig_onevar)
            logit_score_altered_onevar_list.append(logit_score_altered_onevar)    

        score_results['logit_score_orig_onevar_list'].append(logit_score_orig_onevar_list)
        score_results['logit_score_altered_onevar_list'].append(logit_score_altered_onevar_list)


        # saving
        scores_df = pd.DataFrame(score_results)
        full_results = pd.concat([results.iloc[list(range(dset_num + 1))], scores_df], axis=1)
        out_str = f'full_results_{dset_num}'
        if add_feature:
            out_str += '_add_feature'
        pkl.dump(full_results, open(oj(out_dir, out_str + '.pkl'), 'wb'))