import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import tree
from collections import Counter

# from model_train import *
import sys
sys.path.append('../data')
#from data import *
#from intervals import *
#from piecewise import piecewise_average_1d

"""
PARAMETERS
model: a decision tree trained on some data set
input_space_x: a matrix describing the possible x values in the total outcome space
outcome_space_y: a matrix describing the possible y values in the total outcome space
these two should be ordered such that x outcomes match with y outcomes
assignment: an assignment vector starting at x_0
S: a vector with boolean values indicating variables whose interactions we are attempting to describe,
ordered starting at x_0"""
"""
OUTPUT
a matrix with row vectors in the form [y_i, p(y_i)] for y_i unique y values in outcome_space_y, and
p(y_i) a probability value associated with y_i
"""
def fast_interactions(model, input_space_x, outcome_space_y, assignment, S, continuous_y=True, class_id=1):
    features = model.tree_.feature
    thresholds = model.tree_.threshold
    path = model.decision_path(assignment).indices
    features_used = features[path]
    remove_leaves = features_used != -2
    features_used = features_used[remove_leaves]
    mask = S[features_used] == 1
    features_relevant = features_used[mask]
    if(len(features_relevant) == 0):
        return "never encountered relevant features"
    thresholds_used = thresholds[path]
    thresholds_used = thresholds_used[remove_leaves]
    thresholds_relevant = thresholds_used[mask]
    geq = np.transpose(np.transpose(assignment)[features_relevant]) >= thresholds_relevant
    input_greater = np.transpose(np.transpose(input_space_x)[features_relevant]) >= thresholds_relevant
    output_mask =  np.logical_and.reduce(input_greater == geq, axis = -1)
    output_mask = np.reshape(output_mask, (1, -1))[0]
    masked_y = outcome_space_y[output_mask]
    if(continuous_y):
        return np.mean(masked_y)
    else:
        counts = np.count_nonzero(masked_y == class_id)
        return counts/len(masked_y)

def dac(forest, input_space_x, outcome_space_y, assignment, S, continuous_y=True, class_id=1):
    models = forest.estimators_
    avg = np.zeros(assignment.shape[0])
    for model in models:
        for i in range(assignment.shape[0]):
            point = np.reshape(assignment[i, :], (1, -1))
            val = fast_interactions(model, input_space_x, outcome_space_y, point, S, continuous_y, class_id)
            if val != "never encountered relevant features":
                avg[i] += val
    return avg/len(models)

def fix_shape(distribution, unique_Y):
    if(distribution.shape[0] == len(unique_Y)):
        return distribution
    unique_Y = np.reshape(unique_Y, (-1, 1))
    probs = np.zeros(unique_Y.shape)
    dist = np.concatenate((unique_Y, probs), axis=1)
    for i in range(distribution.shape[0]):
        for j in range(unique_Y.shape[0]):
            if(distribution[i, 0] == dist[j, 0]):
                dist[j, 1] = distribution[i, 1]
    return dist

#Helper function for path traversal/analysis
#X, Y are datasets, where rows in X correspond to features, and Y is 1 column of outcomes
#threshold, feature, and direction describe a decision rule:
#e.g. for x_0>= 1.5 threshold = 1.5, feature = 0, direction = 'geq'
#this function applies a the rule to the X dataset, and then applies that selection to the Y dataset as well
def apply_rule(X, Y, threshold, feature, direction):
    mask = []
    if(direction == "geq"):
        mask = X[:, feature] >= threshold
    else:
        mask = X[:, feature] < threshold
    new_X = X[mask]
    new_Y = Y[mask]
    return (new_X, new_Y)

#The purpose of this function is to traverse each path from root to leaf in a tree
#each leaf node will be associated with an set of intervals, one interval per feature specified in S, a value, and a count
#of training points that fall into said intervals.
def traverse_all_paths(model, input_space_x, outcome_space_y, S, C, continuous_y = False):
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right
    feature = model.tree_.feature
    threshold = model.tree_.threshold
    datasets = {}
    num_feats = len(S)#np.count_nonzero(S)
    datasets[0] = (input_space_x, outcome_space_y)
    intervals = {}
    intervals[0] =[(- float('inf'), float('inf'))] * num_feats
    encounters = {}
    encounters[0] = 0
    leaves = []
    fringe = [0]
    while len(fringe) > 0:
        curr_node = fringe.pop()
        bound = threshold[curr_node]
        feat = feature[curr_node]
        child_left = children_left[curr_node]
        child_right = children_right[curr_node]
        if (child_left != child_right):
            X, Y = datasets[curr_node]
            left_data = (X, Y)
            right_data = (X, Y)
            left_interval = intervals[curr_node]
            right_interval = intervals[curr_node]
            if(S[feat] == 1):
                encounters[curr_node] += 1
                left_data = apply_rule(X, Y, bound, feat, "less")
                right_data = apply_rule(X, Y, bound, feat, "geq")

                left_interval = intervals[curr_node][0:feat] + [(intervals[curr_node][feat][0], bound)] + intervals[curr_node][feat + 1:len(intervals[curr_node])]
                right_interval = intervals[curr_node][0:feat] + [(bound, intervals[curr_node][feat][1])] + intervals[curr_node][feat + 1:len(intervals[curr_node])]
            encounters[child_left] = encounters[curr_node]
            encounters[child_right] = encounters[curr_node]
            datasets[child_left] = left_data
            datasets[child_right] = right_data
            intervals[child_left] = left_interval
            intervals[child_right] = right_interval
            fringe.append(child_left)
            fringe.append(child_right)
        else:
            leaves.append(curr_node)
    values = []
    for leaf in leaves:
        X, Y = datasets[leaf]
        inters = intervals[leaf]
        relevant_X = np.transpose(np.transpose(X)[S == 1])
        x_mu = np.mean(relevant_X, axis=0)
        x_cstd = C * np.std(relevant_X, axis=0)
        lower_mask = np.all(relevant_X >= x_mu - x_cstd, axis=1)
        upper_mask = np.all(relevant_X <= x_mu + x_cstd, axis=1)
        mask = np.logical_and(lower_mask, upper_mask)
        mask = np.reshape(mask, Y.shape)
        Y = Y[mask]
        index = 0
        for i in np.nonzero(S)[0]:
            inters[i] = (x_mu[index] - x_cstd[index], x_mu[index] + x_cstd[index])
            index += 1
        if(continuous_y):
            average = np.average(Y)
            values.append(inters + [average, len(Y)])
        else:
            proportion = np.count_nonzero(Y == 1)/Y.shape[0]
            values.append(inters + [proportion, len(Y)])
    return values


def fill_1d(line, counts, interval, val, weight, rng, di):
    lower_bound = int(np.round((max(interval[0], rng[0]) - rng[0])/di))
    upper_bound = int(np.round((min(interval[1], rng[-1]) - rng[0])/di))
    line[lower_bound:upper_bound + 1] += val * weight
    counts[lower_bound:upper_bound + 1] += weight

def make_line(values, interval_x, di, S, ret_counts=False):
    x_axis = np.arange(interval_x[0], interval_x[1] + di, di)
    line = np.zeros(x_axis.shape[0] - 1)
    counts = np.zeros(x_axis.shape[0] - 1)
    num_vars = len(S)
    ind = np.nonzero(S)[0][0]
    for v in values:
        x_inter = v[0:num_vars][ind]
        val, weight = v[num_vars:]
        fill_1d(line, counts, x_inter, val, weight, x_axis, di)
    for i in range(len(counts)):
        if(counts[i] == 0):
            div = 0
            if(i - 1 >= 0):
                div += 1
                counts[i] += counts[i - 1]
                line[i] += line[i - 1]
            if(i + 1 < len(counts)):
                div += 1
                counts[i] += counts[i + 1]
                line[i] += line[i + 1]
            line[i] = line[i]/div
            counts[i] = counts[i]/div
            if(counts[i] == 0):
                counts[i] = 1
    if(ret_counts):
        return line/counts, counts
    return line/counts

def fill_2d(grid, counts, x_interval, y_interval, val, count, x_rng, y_rng, x_di, y_di):
    x_lower_bound = int(np.round((max(x_interval[0], x_rng[0]) - x_rng[0])/x_di))
    x_upper_bound = int(np.round((min(x_interval[1], x_rng[-1]) - x_rng[0])/x_di))

    y_lower_bound = int(np.round((min(y_interval[0], y_rng[-0]) - y_rng[0])/y_di))
    y_upper_bound = int(np.round((min(y_interval[1], y_rng[-1]) - y_rng[0])/y_di))

    grid[y_lower_bound:y_upper_bound + 1, x_lower_bound:x_upper_bound + 1] += val * count
    counts[y_lower_bound:y_upper_bound + 1, x_lower_bound:x_upper_bound + 1] += count

def make_grid(values, interval_x, interval_y, di_x, di_y, S, ret_counts=False):
    x_rng = np.arange(interval_x[0], interval_x[1] + di_x, di_x)
    y_rng = np.arange(interval_y[0], interval_y[1] + di_y, di_y)
    grid = np.zeros((len(y_rng) - 1, len(x_rng) - 1))
    counts = np.zeros((len(y_rng) - 1, len(x_rng) - 1))
    num_vars = len(S)
    for v in values:
        z = np.nonzero(S)
        x_ind = z[0][0]
        y_ind = z[0][1]
        x_inter = v[0:num_vars][x_ind]
        y_inter = v[0:num_vars][y_ind]
        xval, count = v[num_vars:]
        fill_2d(grid, counts, x_inter, y_inter, xval, count, x_rng, y_rng, di_x, di_y)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if counts[i][j] == 0:
                counts[i][j] = 1
    if(ret_counts):
        return grid/counts, counts
    return grid/counts

def make_curve(model, input_space_x, outcome_space_y, S, interval_x, di, C, continuous_y):
    vals = traverse_all_paths(model, input_space_x, outcome_space_y, S, C, continuous_y)
    line = make_line(vals, interval_x, di, S)
    return line

def make_map(model, input_space_x, outcome_space_y, S, interval_x, interval_y, di_x, di_y, C, continuous_y = True):
    vals = traverse_all_paths(model, input_space_x, outcome_space_y, S, C, continuous_y)
    grid = make_grid(vals, interval_x, interval_y, di_x, di_y, S)
    return grid

def make_curve_forest(forest, input_space_x, outcome_space_y, S, interval_x, di, C, continuous_y = True, weights = None):
    models = forest.estimators_
    final_curve = 0
    i = 0
    if weights is None:
        weights = np.ones(len(models))
    for i in range(len(models)):
        #print("starting model", i, "at", time.ctime())
        model = models[i]
        w = weights[i]
        final_curve += w * make_curve(model, input_space_x, outcome_space_y, S, interval_x, di, C, continuous_y)
        #print("model", i, "of", total, "complete at", time.ctime())
        i += 1
    return final_curve/np.sum(weights)

def make_map_forest(forest, input_space_x, outcome_space_y, S, interval_x, interval_y, di_x, di_y, C, continuous_y = True, weights = None):
    models = forest.estimators_
    final_grid = 0
    if weights is None:
        weights = np.ones(len(models))
    for i in range(len(models)):
        model = models[i]
        w = weights[i]
        final_grid += w * make_map(model, input_space_x, outcome_space_y, S, interval_x, interval_y, di_x, di_y, C, continuous_y)
    return final_grid/np.sum(weights)

def ada_boosted_curve_forest(forest, input_space_x, outcome_space_y, S, interval_x, di, C, continuous_y = True):
    ada_weights = forest.estimator_weights_
    return make_curve_forest(forest, input_space_x, outcome_space_y, S, interval_x, di, C, continuous_y = True, weights = ada_weights)

def ada_boosted_map_forest(forest, input_space_x, outcome_space_y, S, interval_x, interval_y, di_x, di_y, C, continuous_y = True):
    ada_weights = forest.estimator_weights_
    return make_map_forest(forest, input_space_x, outcome_space_y, S, interval_x, interval_y, di_x, di_y, C, continuous_y = True, weights = ada_weights)

def dac_plot(forest, input_space_x, outcome_space_y, S, interval_x=None, interval_y=None, di_x=None, di_y=None, C=1, continuous_y=True, weights=None):
    #new wrapper goes here
    return Null

def variance1D(forest, X, y, S, interval_x, di_x, continuous_y=True):
    data_mean = np.mean(y)
    models = forest.estimators_
    total_var = 0
    for model in models:
        vals = traverse_all_paths(model, X, y, S, continuous_y)
        line, counts = make_line(vals, interval_x, di_x, S, ret_counts=True)
        line -= data_mean
        total_var += np.sum(counts * line ** 2)/np.sum(counts)
    return (total_var/len(models))/np.var(y)

def variance2D(forest, X, y, S, intervals, dis, continuous_y=True):
    data_mean = np.mean(y)
    models = forest.estimators_
    total_var = 0
    S1 = np.zeros(S.shape)
    S2 = np.zeros(S.shape)
    nonzero = np.nonzero(S)[0]
    S1[nonzero[0]] = 1
    S2[nonzero[1]] = 1

    if intervals == 'auto':
        min0 = np.min(X[:, nonzero[0]])
        max0 = np.max(X[:, nonzero[0]])
        min1 = np.min(X[:, nonzero[1]])
        max1 = np.max(X[:, nonzero[1]])
        intervals = [(min0, max0), (min1, max1)]
    if dis == 'auto': # 100 points between the bounds
        dis = [(intervals[0][1] - intervals[0][0]) / 100, (intervals[1][1] - intervals[1][0]) / 100]


    for model in models:
        vals = traverse_all_paths(model, X, y, S, continuous_y)
        vals1 = traverse_all_paths(model, X, y, S1, continuous_y)
        vals2 = traverse_all_paths(model, X, y, S2, continuous_y)
        grid, counts = make_grid(vals, intervals[0], intervals[1], dis[0], dis[1], S, ret_counts=True)
        line1 = make_line(vals1, intervals[0], dis[0], S1)
        line2 = make_line(vals2, intervals[1], dis[1], S2)
        grid = grid - line1
        grid = np.transpose(np.transpose(grid) - line2)
        grid += data_mean
        total_var += np.sum(counts * grid ** 2)/np.sum(counts)
    return (total_var/len(models))/np.var(y)

def aggregate_trees(trees, weights, input_space_x, outcome_space_y, assignment, S):
    vals = np.unique(outcome_space_y)
    weighted_average = np.transpose(np.vstack((vals, np.zeros(vals.shape[0]))))
    for i in range(len(trees)):
        t = trees[i]
        w = weights[i]
        dist = interactions_continuous(t, input_space_x, outcome_space_y, assignment, S)
        probs = dist[:, 1]
        shaped = np.transpose(np.vstack((np.zeros(probs.shape[0]), probs)))
        weighted_average += w * shaped
    return weighted_average

def conditional1D(X, y, S, x_rng, di):
    curve = []
    feature_relevant = np.nonzero(S)[0][0]
    for bucket_start in x_rng:
        bucket_end = bucket_start + di
        mask = np.logical_and(X[:, feature_relevant] >= bucket_start, X[:, feature_relevant] < bucket_end)
        curve.append(np.mean(y[mask]))
    return np.array(curve)

def conditional2D(X, y, S, x1_rng, x2_rng, di1, di2):
    grid = np.zeros((len(x2_rng), len(x1_rng)))
    f1 = np.nonzero(S)[0][0]
    f2 = np.nonzero(S)[0][1]
    for i in range(len(x2_rng)):
        for j in range(len(x1_rng)):
            b1_start = x1_rng[j]
            b1_end = x1_rng[j] + di1
            b2_start = x2_rng[i]
            b2_end = x2_rng[i] + di2
            mask1 = np.logical_and(X[:, f1] >= b1_start, X[:, f1] < b1_end)
            mask2 = np.logical_and(X[:, f2] >= b2_start, X[:, f2] < b2_end)
            mask = np.logical_and(mask1, mask2)
            grid[i, j] = np.mean(y[mask])
    return grid
