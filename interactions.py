import numpy as np
from sklearn import tree

from model_train import train
from data import generate_xor_data

"""
define an INTERVAL as a tuple, where the entry at index 0 is the lower bound, and the entry at index 1
is the upper bound, both exclusive.
"""
def interval_intersect(a, b):
    radius_a = (a[1] - a[0])/2
    radius_b = (b[1] - b[0])/2
    centroid_a = radius_a + a[0]
    centroid_b = radius_b + b[0]
    return np.abs(centroid_b - centroid_a) < radius_a + radius_b

def join_intervals(a, b):
    if(not interval_intersect(a, b)):
        return -1
    return (max(a[0], b[0]), min(a[1], b[1]))

def test_intervals():
    i1 = (0, 3)
    i2 = (2, 4)
    i3 = (3, 10)
    i4 = (-1, 2)
    print(i1, "intersects with", i2, "?", interval_intersect(i1, i2))
    print(i1, "intersects with", i3, "?", interval_intersect(i1, i3))
    print(i3, "intersects with", i4, "?", interval_intersect(i3, i4))
    print(i3, "intersects with", i2, "?", interval_intersect(i3, i2))
    print(i4, "intersects with", i1, "?", interval_intersect(i4, i1))
    print("joining", i1, "and", i2, ":", join_intervals(i1, i2))
    print("joining", i3, "and", i2, ":", join_intervals(i3, i2))
    print("joining", i1, "and", i4, ":", join_intervals(i1, i4))

"""
PARAMETERS
model: a decision tree trained on some data set
outcome_space_x: a matrix describing the possible x values in the total outcome space
input_space_y: a matrix describing the possible y values in the total outcome space
these two should be ordered such that x outcomes match with y outcomes
assignment: an assignment vector starting at x_0
S: a vector with boolean values indicating variables whose interactions we are attempting to describe,
ordered starting at x_0"""
"""
OUTPUT
a matrix with row vectors in the form [y_i, p(y_i)] for y_i unique y values in input_space_y, and
p(y_i) a probability value associated with y_i
"""
def interactions(model, outcome_space_x, input_space_y, assignment, S):
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
    for i in range(len(input_space_y)):
        complies = True
        for d in decision_rules:
            if(d[2]):
                complies = complies and outcome_space_x[i][d[0]] >= d[1]
            else:
                complies = complies and outcome_space_x[i][d[0]] < d[1]
        if complies:
            outcomes.append(input_space_y[i])
    unique, counts = np.unique(outcomes, return_counts = True)
    unique = np.reshape(unique, (-1, 1))
    probs = np.reshape(counts/len(outcomes), (-1, 1))
    return np.hstack((unique, probs))

def interactions_continuous(model, outcome_space_x, input_space_y, assignment, S):
    features = model.tree_.feature
    thresholds = model.tree_.threshold
    path = model.decision_path(assignment).indices
    decision_rules = {} #store decision rules in dictionary: feature => valid interval
    for i in path:
        feature_ind = features[i]
        if S[feature_ind] > 0 and feature_ind >= 0:
            threshold = thresholds[i]
            leq = assignment[0][feature_ind] < threshold
            bound = (-1 ** leq) * float('inf')
            interval = (min(threshold, bound), max(threshold, bound))
            if(decision_rules.get(feature_ind) != None):
                decision_rules[feature_ind] = interval_intersect(decision_rules.get(feature_ind), interval)
            else:
                decision_rules[feature_ind] = interval
    outcomes = []
    features = list(decision_rules.keys())
    rules = list(decision_rules.values())
    for i in range(len(input_space_y)):
        complies = True
        for j in range(len(features)):
            feature = features[j]
            rule = rules[j]
            coord = outcome_space_x[i][feature]
            complies = complies and interval_contains((coord, coord), rule)
        if complies:
            outcomes.append(input_space_y[i])
    unique, counts = np.unique(outcomes, return_counts = True)
    unique = np.reshape(unique, (-1, 1))
    probs = np.reshape(counts/len(outcomes), (-1, 1))
    return np.hstack((unique, probs))


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
            print(interactions(model, X, Y, a, s))


test_continuous()
