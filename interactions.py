import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import tree

from model_train import train, train_cont, train_rf
from data import generate_xor_data, generate_single_variable_boundary, generate_x_y_data
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


def recover_intervals(model, num_features):
    split_feats = model.tree_.feature
    thresholds = model.tree_.threshold
    feat_thresholds = [[] for i in range(num_features)]
    for i in range(len(split_feats)):
        if(split_feats[i] >= 0):
            feat_thresholds[split_feats[i]].append(thresholds[i])
    intervals = []
    for f in range(num_features):
        t = feat_thresholds[f]
        t.sort()
        intervals.append(t)
    return intervals

def generate_all_inputs(intervals, di):
    inputs = []
    for feature in intervals:
        start = feature[0] - di
        feature_inputs = [start]
        end = feature[-1] + di
        for i in range(len(feature) - 1):
            feature_inputs.append((feature[i+1] - feature[i])/2 + feature[i])
        feature_inputs.append(end)
        inputs.append(feature_inputs)
    return inputs

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

def plot_2D_classifier(X_range, distribution):
    #print("X_range", X_range)
    #print("distribution", distribution)
    plt.plot(X_range, distribution)
    plt.ylabel("chance of classifying 1")
    plt.xlabel("input")
    plt.show()

def test_plot_many_trees2(n):
    X, Y = generate_single_variable_boundary(100, (-10, 10), 0)
    rf = train_rf(X, Y, n)
    trees = rf.estimators_
    t = time.clock()
    boundaries = []
    inputs = []
    for model in trees:
        intervals = recover_intervals(model, 1)
        ins = generate_all_inputs(intervals, 1)
        boundaries += intervals[0]
        inputs += ins[0]
    boundaries.sort()
    inputs.sort()
    print("boundaries time", time.clock() - t)
    values = []
    weights = [1/n] * n
    for assignment in inputs:
        v = aggregate_trees(trees, weights, X, Y, [[assignment]], [1])[1, 1]
        values.append(v)
    print("trees time", time.clock() - t)
    x_axis = np.arange(-10, 10, .1)
    distribution = []
    i = 0
    boundaries = [- float('inf')] + boundaries + [float('inf')]
    for x in x_axis:
        if x >= boundaries[i] and x < boundaries[i + 1]:
            distribution.append(values[i])
        else:
            i += 1
            distribution.append(values[i])
    print("total time", time.clock() - t)
    plot_2D_classifier(x_axis, distribution)


def test_nonsense_vars():
    X, Y = generate_xor_data(500, nonsense_vars=1)
    model = train(X, Y)
    X_tests = [[[1, 1, -1]], [[1, -1, -1]], [[1, 1, 1]], [[-1, 1, 1]]]
    for a in X_tests:
        print("Assignment:", a)
        for i in range(5):
            s = np.random.choice([1, 0], (3))
            print("interactions for variables:", s)
            print(interactions(model, X, Y, a, s))

def test_continuous():
    X, Y = generate_xor_data(10000, continuous_x=True)
    model = train(X, Y)
    X_tests = [[[-.5, -.5]], [[.5, .5]], [[-.5, .5]]]
    S = [[1, 0], [0, 1], [0, 0], [1, 1]]
    for a in X_tests:
        print("Assignment:", a)
        for s in S:
            print("interactions for variables:", s)
            print(interactions_continuous(model, X, Y, a, s))

def test_continuous_y():
    X, Y = generate_x_y_data(1000, (-10, 10), lambda a: a[0], features=2)
    model = train_cont(X, Y)
    outcomes = []
    test_X = np.arange(-10, 10, .25)
    for x in test_X:
        y = interactions_continuous(model, X, Y, [[x, np.random.uniform(-10, 10)]], [0, 1], continuous_y = True)
        outcomes.append(y)
    plt.plot(test_X, outcomes)
    plt.ylabel("predicted Y")
    plt.xlabel("input X")
    plt.show()

def heat_map():
    X, Y = generate_xor_data(10000, continuous_x=True)
    model = train(X, Y)
    intervals = recover_intervals(model, 2)
    inputs = generate_all_inputs(intervals)
    heat_values = []
    for x_1 in inputs[0]:
        heat_row = []
        for x_2 in inputs[1]:
            heat_row.append(interactions_continuous(model, X, Y, [[x_1, x_2]], [1, 1])[1, 1])
        heat_values.append(heat_row)
    print(heat_values)
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(heat_values)

    ax.set_xticks(np.arange(len(inputs[0])))
    ax.set_yticks(np.arange(len(inputs[1])))
    intervals_1_labels = [str((round(interval[0], 3), round(interval[1], 3))) for interval in intervals[0]]
    intervals_2_labels = [str((round(interval[0], 3), round(interval[1], 3))) for interval in intervals[1]]
    ax.set_xticklabels(intervals_1_labels)
    ax.set_yticklabels(intervals_2_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    plt.show()

def test_interactions_rf(n):
    X, Y = generate_xor_data(1000, continuous_x=True)
    model = train_rf(X, Y, n)
    estimators = model.estimators_
    input = [[1, -1]]
    s = [1, 0]
    weights = [1/n] * n
    t = time.clock()
    dist = aggregate_trees(estimators, weights, X, Y, input, s)
    print("comp time", time.clock() - t)
    print(dist)

test_plot_many_trees2(100)
#test_interactions_rf(10000)
