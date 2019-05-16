import sys
from tqdm import tqdm
import pmlb as dsets
import numpy as np
import pickle as pkl
from os.path import join as oj
import os
from copy import deepcopy
import pandas as pd
from numpy import array as arr

# sklearn models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
import dac
from scipy import interpolate
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

def get_lin(classification_only, random_state):
    if classification_only:
        return LogisticRegression(solver='lbfgs', multi_class='auto', random_state=random_state)
    else:
        return LinearRegression(normalize=True)

def fit_altered(data_dir, out_dir, dset_name, classification_only=True, random_state=42):
    print(dset_name, 'classification:', classification_only)
    continuous_y = False

    score_results = {
            'feature_scores_mdi': [],
    #         'feature_scores_mda': [],
            'logit_score_orig_onevar': [],
            'logit_score_altered_onevar': [],
            'logit_score_altered_append': [],    
            'variances': [],
            'logit_score_altered_interaction_onevar': [],    
            'logit_score_altered_interaction_append': [],
            'logit_test_score': [],
            'rf_test_score': [],
            'rf': [],
            'dset_name': [dset_name],
            'classification_only': [classification_only],
        }

    # fit basic things
    if classification_only:
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    logit = get_lin(classification_only, random_state)

    # load data and rf
    X, y = dsets.fetch_data(dset_name, return_X_y=True, 
                      local_cache_dir=data_dir)
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=random_state)
    logit = logit.fit(train_X, train_y)
    rf = rf.fit(train_X, train_y)
    
    score_results['logit_test_score'].append(logit.score(test_X, test_y))
    score_results['rf_test_score'].append(rf.score(test_X, test_y))    
    
    
    num_features = X.shape[1]
    score_results['rf'].append(rf)


    # feature importances
    feature_scores_mdi = get_importance_scores(rf, score_type='mdi', X=test_X, Y=test_y)
    score_results['feature_scores_mdi'].append(feature_scores_mdi)
    # feature_scores_mda = get_importance_scores(rf, score_type='mda', X=test_X, Y=test_y)
    # score_results['feature_scores_mda'].append(feature_scores_mda)


    # get importance for 1d var
    feat_num = np.argsort(feature_scores_mdi)[-1]
    feat_vals = X[:, feat_num].reshape(-1, 1)
    feat_vals_train = train_X[:, feat_num].reshape(-1, 1)
    feat_vals_test = test_X[:, feat_num].reshape(-1, 1)
    feat_val_min = np.min(feat_vals)
    feat_val_max = np.max(feat_vals)

    # appropriate variable to get importance for
    S = np.zeros(num_features)
    S[feat_num]= 1
    X_alt_train = dac.interactions_forest(forest=rf, input_space_x=train_X, outcome_space_y=train_y, 
                                             assignment=train_X, S=S, continuous_y=continuous_y).reshape(-1, 1)
#     X_alt_train = interactions.interactions_forest(forest=rf, input_space_x=train_X, outcome_space_y=train_y, 
#                                              assignment=train_X, S=S, continuous_y=continuous_y).reshape(-1, 1)
    make_curve_forest(forest, input_space_x, outcome_space_y, S, interval_x, di, C, continuous_y = True):

    
    X_alt_test = dac.interactions_forest(forest=rf, input_space_x=train_X, outcome_space_y=train_y, 
                                             assignment=test_X, S=S, continuous_y=continuous_y).reshape(-1, 1)



    # fit only on one feature orig
    logit = get_lin(classification_only, random_state)
    logit.fit(feat_vals_train, train_y)
    score_results['logit_score_orig_onevar'].append(logit.score(feat_vals_test, test_y))

    # fit only on one feature altered
    logit = get_lin(classification_only, random_state)
    logit.fit(X_alt_train, train_y)
    score_results['logit_score_altered_onevar'].append(logit.score(X_alt_test, test_y))

    # fit with altered feature (appended)
    logit = get_lin(classification_only, random_state)
    logit.fit(np.hstack((train_X, X_alt_train)), train_y)
    score_results['logit_score_altered_append'].append(logit.score(np.hstack((test_X, X_alt_test)), test_y))    

    # fit with 2D interaction (using feat_num and some other variable)
    '''
    variances = np.zeros(num_features) # variance of max feature with other features
    for i in range(num_features):
        if not i == feat_num:
            S = np.zeros(num_features)
            S[feat_num]= 1
            S[i] = 1
        variances[i] = dac.variance2D(forest=rf, X=train_X, y=train_y, S=S, 
                                               intervals='auto', dis='auto', continuous_y=continuous_y)
    score_results['variances'].append(variances)
    '''
    score_results['variances'].append(np.nan)

    # for feat with max interaction with the original max feat (chosen with mdi), refit using the interaction
    # feat_num_2 = np.argmax(variances)
    feat_num_2 = np.argsort(feature_scores_mdi)[-2]
    S = np.zeros(num_features)
    S[feat_num]= 1
    S[feat_num_2] = 1

    X_interaction_train = dac.interactions_forest(forest=rf, input_space_x=train_X, outcome_space_y=train_y, 
                                             assignment=train_X, S=S, continuous_y=continuous_y).reshape(-1, 1)
    X_interaction_test = dac.interactions_forest(forest=rf, input_space_x=train_X, outcome_space_y=train_y, 
                                             assignment=test_X, S=S, continuous_y=continuous_y).reshape(-1, 1)


    # fit only on one interaction altered
    logit = get_lin(classification_only, random_state)
    print('\t', dset_name, 'shapes', X_interaction_train.shape, X_interaction_test.shape, train_y.shape, test_y.shape)
    try:
        logit.fit(X_interaction_train, train_y)
        score_results['logit_score_altered_interaction_onevar'].append(logit.score(X_interaction_test, test_y))
    except:
        print('\terr!', dset_name, 'shapes', X_interaction_train.shape, X_interaction_test.shape, train_y.shape, test_y.shape)
        exit(0)

     # fit with altered interaction (appended)
    logit = get_lin(classification_only, random_state)
    logit.fit(np.hstack((train_X, X_interaction_train)), train_y)
    score_results['logit_score_altered_interaction_append'].append(logit.score(np.hstack((test_X, X_interaction_test)), test_y))    


    # saving
    os.makedirs(out_dir, exist_ok=True)
    out_str = f'{dset_name}_full'
    pkl.dump(score_results, open(oj(out_dir, out_str + '.pkl'), 'wb'))

    
    
    
    
if __name__ == '__main__':    
        
    # dset_name = 'analcatdata_aids'
    dset_name = 'analcatdata_germangss' # check for multiclass
    classification_only = True
    if len(sys.argv) > 1: # first arg (the dset_name)
        dset_name = str(sys.argv[1])
    if len(sys.argv) > 2: # second arg (classification or regression)
        classification_only = str(sys.argv[2]) == 'classification'

    data_dir = '/scratch/users/vision/data/pmlb'
    out_dir = '/scratch/users/vision/chandan/pmlb/classification_3'
    random_state = 42 # for each train_test_split        
        
    fit_altered(data_dir, out_dir, dset_name, classification_only, random_state=42)
    print('success!')