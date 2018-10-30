import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import tree

from model_train import train
from data import generate_xor_data, generate_single_variable_boundary
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

"""interactions for continuous X only!"""
def interactions_continuous(model, input_space_x, outcome_space_y, assignment, S):
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
        lower = - float("inf")
        inter = []
        for i in range(len(t)):
            inter.append((lower, t[i]))
            lower = t[i]
        inter.append((t[len(t) - 1], float('inf')))
        intervals.append(inter)
    return intervals

def generate_all_inputs(intervals):
    inputs = []
    for feature in intervals:
        feature_inputs = []
        for interval in feature:
            if interval[0] == - float('inf'):
                feature_inputs.append(interval[1] - 1)
            elif interval[1] == float('inf'):
                feature_inputs.append(interval[0] + 1)
            else:
                feature_inputs.append((interval[1] - interval[0])/2 + interval[0])
        inputs.append(feature_inputs)
    return inputs

def aggregate_trees(trees, weights, input_space_x, outcome_space_y, assignment, S):
    weighted_average = 0
    for i in range(len(trees)):
        t = trees[i]
        w = weights[i]
        dist = interactions_continuous(t, input_space_x, outcome_space_y, assignment, S)[0]
        weighted_average += w * dist
    return weighted_average

def plot_2D_classifier(X_range, distribution):
    #print("X_range", X_range)
    #print("distribution", distribution)
    plt.plot(X_range, distribution)
    plt.ylabel("chance of classifying 1")
    plt.xlabel("input")
    plt.show()

def test_plot_single_variable():
    X_0, Y_0 = generate_single_variable_boundary(100, (-10, 10), 0)
    X_1, Y_1 = generate_single_variable_boundary(100, (-10, 10), 4)
    model_0 = train(X_0, Y_0)
    model_1 = train(X_1, Y_1)
    intervals_0 = recover_intervals(model_0, 1)
    intervals_1 = recover_intervals(model_1, 1)
    inputs_0 = generate_all_inputs(intervals_0)
    inputs_1 = generate_all_inputs(intervals_1)
    values_0 = [interactions_continuous(model_0, X_0, Y_0, [[i]], [1])[1, 1] for i in inputs_0[0]]
    values_1 = [interactions_continuous(model_1, X_1, Y_1, [[i]], [1])[1, 1] for i in inputs_1[0]]
    piece_0 = [(intervals_0[0][i][0], intervals_0[0][i][1], [values_0[i]]) for i in range(len(intervals_0[0]))]
    piece_1 = [(intervals_1[0][i][0], intervals_1[0][i][1], [values_1[i]]) for i in range(len(intervals_1[0]))]
    avg = piecewise_average_1d([piece_0, piece_1])
    x_vals = np.arange(-10, 10, .1)
    i = 0
    distribution = []
    for x in x_vals:
        if x >= avg[i][0] and x < avg[i][1]:
            distribution.append(avg[i][2])
        else:
            i += 1
            distribution.append(avg[i][2])
    plot_2D_classifier(x_vals, distribution)

def test_plot_many_trees(t):
    data_sets = [generate_single_variable_boundary(100, (-10, 10), np.random.uniform(-5, 5)) for i in range(t)]
    trees = [train(data[0], data[1]) for data in data_sets]
    t = time.clock()
    intervals = [recover_intervals(model, 1) for model in trees]
    inputs = [generate_all_inputs(inter) for inter in intervals]
    values = []
    for i in range(len(trees)):
        v_for_tree = []
        for input in inputs[i][0]:
            v = interactions_continuous(trees[i], data_sets[i][0], data_sets[i][1], [[input]], [1])[1, 1]
            v_for_tree.append(v)
        values.append(v_for_tree)
    pieces = []
    for i in range(len(trees)):
        piece_for_tree = []
        for j in range(len(intervals[i][0])):
            piece = (intervals[i][0][j][0], intervals[i][0][j][1], [values[i][j]])
            piece_for_tree.append(piece)
        pieces.append(piece_for_tree)
    avg = piecewise_average_1d(pieces)
    x_vals = np.arange(-10, 10, .1)
    i = 0
    distribution = []
    for x in x_vals:
        if x >= avg[i][0] and x < avg[i][1]:
            distribution.append(avg[i][2])
        else:
            i += 1
            distribution.append(avg[i][2])
    print("comp time", time.clock() - t)
    plot_2D_classifier(x_vals, distribution)


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

#test_continuous()
test_plot_many_trees(5)
