import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

from model_train import train
from data import generate_xor_data, generate_single_variable_boundary

"""
define an INTERVAL as a tuple, where the entry at index 0 is the lower bound, and the entry at index 1
is the upper bound, both exclusive.
"""
def interval_intersect(a, b):
    if(min(a) == -float('inf') and min(b) == -float('inf')):
        return True
    elif(max(a) == float('inf') and max(b) == float('inf')):
        return True
    if(min(a) == -float('inf')):
        return min(b) < max(a)
    elif(min(b) == -float('inf')):
        return min(a) < max(b)
    elif(max(a) == float('inf')):
        return min(a) < max(b)
    elif(max(b) == float('inf')):
        return min(b) < max(a)
    print("general case")
    radius_a = (a[1] - a[0])/2
    radius_b = (b[1] - b[0])/2
    centroid_a = radius_a + a[0]
    centroid_b = radius_b + b[0]
    return np.abs(centroid_b - centroid_a) < radius_a + radius_b

def join_intervals(a, b):
    if(not interval_intersect(a, b)):
        return -1
    return (max(a[0], b[0]), min(a[1], b[1]))

def point_in_intervals(p, intervals):
    for i in intervals:
        if p > i[0] and p < i[1]:
            return 1
    return 0

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
    return (fix_shape(unshaped, np.unique(outcome_space_y)), decision_rules)

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



"""determines the range spanned by all variables in the X and Y dataset"""
def determine_input_output_range(input_space_x, outcome_space_y):
    x_max = input_space_x.max(axis=0)
    x_min = input_space_x.min(axis=0)
    y_max = input_space_y.max()
    y_min = input_space_y.min()
    x_intervals = np.transpose(np.hstack(x_min, x_max))
    y_interval = [y_min, y_max]
    return (x_intervals, y_interval)

def cartesian_product(set1, set2):
    product = []
    for i in set1:
        for j in set2:
            if(type(i) is list):
                product.append(i + [j])
            else:
                product.append([i, j])

def multiple_products(sets):
    if(len(sets) <= 1):
        return sets
    else:
        i = 2
        product = cartesian_product(sets[0], sets[1])
        while(i < len(sets)):
            product = cartesian_product(product, sets[i])
            i += 1
        return product

"""generates all possible combination of X inputs from x_intervals, a matrix describing
the range of each variable in X"""
def generate_all_inputs(input_space_x, outcome_space_y, step):
    x_intervals, y_interval = determine_input_output_range(input_space_x, outcome_space_y)
    x_ranges = [np.arange(x[0], x[1] + step, step) for x in x_intervals]
    y_range = np.arange(y_interval[0], y_interval[1], step)
    products = multiple_products(x_ranges)
    return (products, y_range)


def aggregate_trees(trees, weights, input_space_x, outcome_space_y, assignment, S):
    weighted_average = 0
    for i in range(len(trees)):
        t = trees[i]
        w = weights[i]
        dist = interactions_continuous(t, input_space_x, outcome_space_y, assignment, S)[0]
        weighted_average += w * dist
    return weighted_average

def plot_2D_binary_classifier(X_range, weights, intervals_per_classifier):
    Y_totals = 0
    for i in range(len(intervals_per_classifier)):
        X_valid_intervals = intervals_per_classifier[i]
        Y_totals += weights[i] * np.array([point_in_intervals(i, X_valid_intervals) for i in X_range])
    plt.plot(X_range, Y_totals, 'ro')
    plt.ylabel("classification")
    plt.xlabel("input")
    plt.show()

def plot_2D_classifier(X_range, distribution):
    print("X_range", X_range)
    print("distribution", distribution)
    plt.plot(X_range, distribution)
    plt.ylabel("chance of classifying 1")
    plt.xlabel("input")
    plt.show()

def test_plot_single_variable():
    X_0, Y_0 = generate_single_variable_boundary(100, (-10, 10), 0)
    X_1, Y_1 = generate_single_variable_boundary(100, (-10, 10), 4)
    model_0 = train(X_0, Y_0)
    model_1 = train(X_1, Y_1)
    weights = [.5, .5]
    assignments = [[[i]] for i in np.arange(-10, 10, .5)]
    s = [1]
    distribution = []
    for a in assignments:
        interactions = aggregate_trees([model_0, model_1], weights, np.vstack((X_0, X_1)), np.vstack((Y_0, Y_1)), a, s)
        print("interactions\n", interactions)
        distribution.append(interactions[1, 1])
    plot_2D_classifier(np.arange(-10, 10, .5), distribution)


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
            print(interactions_continuous(model, X, Y, a, s)[0])

#test_continuous()
test_plot_single_variable()
