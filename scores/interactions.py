import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import tree

# from model_train import *
import sys
sys.path.append('../data')
from data import *
from intervals import *
from piecewise import piecewise_average_1d

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
def interactions(model, input_space_x, outcome_space_y, assignment, S):
    features = model.tree_.feature
    thresholds = model.tree_.threshold
    path = model.decision_path(assignment).indices
    decision_rules = [] #store decisions as tuples: (feature, threshold, 0 if > or 1 if <=)
    for i in path:
        feature_ind = features[i]
        if S[feature_ind] > 0 and feature_ind >= 0:
            threshold = thresholds[i]
            geq = assignment[0][feature_ind] >= threshold
            decision_rules.append((feature_ind, threshold, geq))
    outcomes = []
    for i in range(len(outcome_space_y)):
        complies = True
        for d in decision_rules:
            if(d[2]):
                complies = complies and input_space_x[i][d[0]] >= d[1]
            else:
                complies = complies and input_space_x[i][d[0]] < d[1]
        if complies:
            outcomes.append(outcome_space_y[i])
    unique, counts = np.unique(outcomes, return_counts = True)
    unique = np.reshape(unique, (-1, 1))
    probs = np.reshape(counts/len(outcomes), (-1, 1))
    return np.hstack((unique, probs))

def interactions_continuous(model, input_space_x, outcome_space_y, assignment, S, continuous_y = False):
    features = model.tree_.feature
    thresholds = model.tree_.threshold
    path = model.decision_path(assignment).indices
    decision_rules = {} #store decision rules in dictionary: feature => valid interval
    for i in path:
        feature_ind = features[i]
        if feature_ind >= 0 and S[feature_ind] > 0:
            threshold = thresholds[i]
            leq = assignment[0][feature_ind] < threshold
            bound = np.power(-1, leq) * float('inf')
            interval = (min(threshold, bound), max(threshold, bound))
            if(decision_rules.get(feature_ind) != None):
                decision_rules[feature_ind] = join_intervals(decision_rules.get(feature_ind), interval)
            else:
                decision_rules[feature_ind] = interval
    outcomes = []
    features = list(decision_rules.keys())
    rules = list(decision_rules.values())
    for i in range(len(outcome_space_y)):
        complies = True
        for j in range(len(features)):
            feature = features[j]
            rule = rules[j]
            coord = input_space_x[i][feature]
            complies = complies and point_in_intervals(coord, [rule])
        if complies:
            outcomes.append(outcome_space_y[i])
    if(continuous_y):
        return np.average(outcomes)
    unique, counts = np.unique(outcomes, return_counts = True)
    unique = np.reshape(unique, (-1, 1))
    probs = np.reshape(counts/len(outcomes), (-1, 1))
    unshaped = np.hstack((unique, probs))
    return fix_shape(unshaped, np.unique(outcome_space_y))

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

def interactions_set(model, input_space_x, outcome_space_y, assignment, S, continuous_y=True, class_id=1):
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
    geq_list = np.transpose(np.transpose(assignment)[features_relevant]) >= thresholds_relevant
    input_greater = np.transpose(np.transpose(input_space_x)[features_relevant]) >= thresholds_relevant
    outputs = []
    for geq in geq_list:
        output_mask =  np.logical_and.reduce(input_greater == geq, axis = -1)
        output_mask = np.reshape(output_mask, (1, -1))[0]
        masked_y = outcome_space_y[output_mask]
        if len(masked_y) == 0:
            return 0
        if(continuous_y):
            outputs.append(np.mean(masked_y))
        else:
            counts = np.count_nonzero(masked_y == class_id)
            outputs.append(counts/len(masked_y))
    return outputs

def interactions_forest(forest, input_space_x, outcome_space_y, assignment, S, continuous_y=True, class_id=1):
    models = forest.estimators_
    avg = 0
    for model in models:
        val = interactions_set(model, input_space_x, outcome_space_y, assignment, S, continuous_y=True, class_id=1)
        if val != "never encountered relevant features":
            avg += np.array(val)
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
def traverse_all_paths(model, input_space_x, outcome_space_y, S, continuous_y = False):
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
    encounters[0] = False
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
                encounters[curr_node] = True
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
    if(continuous_y):
        for leaf in leaves:
            X, Y = datasets[leaf]
            inter = intervals[leaf]
            average = np.average(Y)
            if(encounters[leaf]):
                values.append(inter + [average] + [len(Y)])
        return values
    for leaf in leaves:
        X, Y = datasets[leaf]
        inter = intervals[leaf]
        proportion = np.count_nonzero(Y == 1)/Y.shape[0]
        if(encounters[leaf]):
            values.append(inter + [proportion] + [len(Y)])
    return values

def fill_1d(line, counts, interval, val, count, rng, di):
    lower_bound = max(interval[0], rng[0])
    lower_bound = min(lower_bound, rng[-1])
    upper_bound = min(interval[1], rng[-1])
    upper_bound = max(upper_bound, rng[0])
    start_index = np.nonzero(rng - lower_bound >= 0)[0][0]
    end_index = np.nonzero(rng - upper_bound >= 0)[0][0]
    for i in range(start_index, end_index):
        line[i] += count * val
        counts[i] += count
    return

def make_line(values, interval_x, di, S, ret_counts=False):
    x_axis = np.arange(interval_x[0], interval_x[1] + di, di)
    line = np.zeros(x_axis.shape[0] - 1)
    counts = np.zeros(x_axis.shape[0] - 1)
    num_vars = len(S)
    ind = np.nonzero(S)[0][0]
    for v in values:
        x_inter = v[0:num_vars][ind]
        val, count = v[num_vars:]
        fill_1d(line, counts, x_inter, val, count, x_axis, di)
    for i in range(len(counts)):
        if(counts[i] == 0):
            counts[i] = 1
    if(ret_counts):
        return line/counts, counts
    return line/counts

def fill_2d(grid, counts, x_interval, y_interval, val, count, x_rng, y_rng, x_di, y_di):
    x_lower_bound = max(x_interval[0], x_rng[0])
    x_lower_bound = min(x_lower_bound, x_rng[-1])
    x_upper_bound = min(x_interval[1], x_rng[-1])
    x_upper_bound = max(x_upper_bound, x_rng[0])
    x_start_index = np.nonzero(x_rng - x_lower_bound >= 0)[0][0]
    x_end_index = np.nonzero(x_rng - x_upper_bound >= 0)[0][0]

    y_lower_bound = max(y_interval[0], y_rng[0])
    y_lower_bound = min(y_lower_bound, y_rng[-1])
    y_upper_bound = min(y_interval[1], y_rng[-1])
    y_upper_bound = max(y_upper_bound, y_rng[0])
    y_start_index = np.nonzero(y_rng - y_lower_bound >= 0)[0][0]
    y_end_index = np.nonzero(y_rng - y_upper_bound >= 0)[0][0]
    for y in range(y_start_index, y_end_index):
        for x in range(x_start_index, x_end_index):
            grid[y][x] += count * val
            counts[y][x] += count
    return

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

def make_curve(model, input_space_x, outcome_space_y, S, interval_x, di, continuous_y = False):
    vals = traverse_all_paths(model, input_space_x, outcome_space_y, S, continuous_y)
    line = make_line(vals, interval_x, di, S)
    return line

def make_map(model, input_space_x, outcome_space_y, S, interval_x, interval_y, di_x, di_y, continuous_y = False):
    vals = traverse_all_paths(model, input_space_x, outcome_space_y, S, continuous_y)
    grid = make_grid(vals, interval_x, interval_y, di_x, di_y, S)
    return grid

def make_curve_forest(forest, input_space_x, outcome_space_y, S, interval_x, di, continuous_y = False):
    models = forest.estimators_
    final_curve = 0
    for model in models:
        final_curve += make_curve(model, input_space_x, outcome_space_y, S, interval_x, di, continuous_y)
    return final_curve/len(models)

def make_map_forest(forest, input_space_x, outcome_space_y, S, interval_x, interval_y, di_x, di_y, continuous_y = False):
    models = forest.estimators_
    final_grid = 0
    for model in models:
        final_grid += make_map(model, input_space_x, outcome_space_y, S, interval_x, interval_y, di_x, di_y, continuous_y)
    return final_grid/len(models)

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
