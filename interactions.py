import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import tree

from model_train import *
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
        #print("curr_node", curr_node)
        bound = threshold[curr_node]
        feat = feature[curr_node]
        #print("bound", bound, "on feature", feat)
        child_left = children_left[curr_node]
        child_right = children_right[curr_node]
        #print("children (left, right)", child_left, child_right)
        if (child_left != child_right):
            #print("non-leaf")
            X, Y = datasets[curr_node]
            #print("current node data set X\n", X)
            #print("current node data set Y\n", Y)
            left_data = (X, Y)
            right_data = (X, Y)
            left_interval = intervals[curr_node]
            right_interval = intervals[curr_node]
            #print("current node interval", intervals[curr_node])
            if(S[feat] == 1):
                encounters[curr_node] = True
                left_data = apply_rule(X, Y, bound, feat, "less")
                #print("left data", left_data)
                right_data = apply_rule(X, Y, bound, feat, "geq")
                #print("right data", right_data)

                left_interval = intervals[curr_node][0:feat] + [(intervals[curr_node][feat][0], bound)] + intervals[curr_node][feat + 1:len(intervals[curr_node])]
                #print("left interval", left_interval)
                right_interval = intervals[curr_node][0:feat] + [(bound, intervals[curr_node][feat][1])] + intervals[curr_node][feat + 1:len(intervals[curr_node])]
                #print("right interval", right_interval)
            encounters[child_left] = encounters[curr_node]
            encounters[child_right] = encounters[curr_node]
            datasets[child_left] = left_data
            datasets[child_right] = right_data
            intervals[child_left] = left_interval
            intervals[child_right] = right_interval
            fringe.append(child_left)
            fringe.append(child_right)
        else:
            #print("leaf")
            leaves.append(curr_node)
    values = []
    #print("leaves", leaves)
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
"""
def make_line(values, interval_x, di, ind, num_vars):
    xdim = (int)((interval_x[1] - interval_x[0])/di)
    x_axis = np.arange(interval_x[0], interval_x[1], di)
    line = np.zeros(x_axis.shape)
    counts = np.zeros(x_axis.shape)
    for i in range(xdim):
        x_coord = x_axis[i]
        for v in values:
            x_inter = v[0:num_vars][ind]
            val, count = v[num_vars:]
            if x_coord>=x_inter[0] and x_coord < x_inter[1]:
                line[i] += count * val
                counts[i] += count

    return line/counts
"""
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

def make_line(values, interval_x, di, S):
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
    return line/counts
"""
def make_grid(values, interval_x, interval_y, di_x, di_y, S, num_vars):
    xdim = (int)((interval_x[1] - interval_x[0])/di_x)
    ydim = (int)((interval_y[1] - interval_y[0])/di_y)
    grid = np.zeros((ydim, xdim))
    counts = np.zeros((ydim, xdim))
    for i in range(ydim):
        for j in range(xdim):
            x_coord = j * di_x + interval_x[0]
            y_coord = i * di_y + interval_y[0]
            for v in values:
                z = np.nonzero(S)
                x_ind = z[0][0]
                y_ind = z[0][1]
                x_inter = v[0:num_vars][x_ind]
                y_inter = v[0:num_vars][y_ind]
                xval, count = v[num_vars:]
                if x_coord>=x_inter[0] and x_coord < x_inter[1] and y_coord >= y_inter[0] and y_coord < y_inter[1]:
                    grid[i, j] += xval * count
                    counts[i, j] += count
    return grid/counts
"""

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

def make_grid(values, interval_x, interval_y, di_x, di_y, S):
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
    return grid/counts

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

def test_plot_many_trees3(n):
    datasets = [generate_single_variable_boundary(100, (-20, 20), np.random.uniform(-10, 10)) for i in range(n)]
    x_vals = np.arange(-20, 20, .1)
    distribution = np.zeros(x_vals.shape[0])
    t_start = time.clock()
    for X, Y in datasets:
        model = train(X, Y)
        boundaries = recover_intervals(model, 1)
        inputs = generate_all_inputs(boundaries, 1)[0]
        values = []
        for input in inputs:
            v = interactions_continuous(model, X, Y, [[input]], [1])[1, 1]
            values.append(v)
        boundaries = [- float('inf')] + boundaries[0] + [float('inf')]
        i = 0
        j = 0
        dist = []
        while(i < len(boundaries) - 1 and j < len(x_vals)):
            if x_vals[j] >= boundaries[i] and x_vals[j] < boundaries[i+1]:
                dist.append(values[i])
                j += 1
            else:
                i += 1
        distribution += np.array(dist)
    distribution = distribution/n
    print("total time", time.clock() - t_start)
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

def heat_map_xor():
    X, Y = generate_xor_data(500, continuous_x=True)
    plt.scatter(x=X[:, 0], y=X[:, 1], c=Y)
    plt.show()
    model = train(X, Y)
    x_axis = np.arange(-1, 1, .1)
    y_axis = np.arange(-1, 1, .1)
    dist = np.zeros((y_axis.shape[0], x_axis.shape[0]))
    for i in range(y_axis.shape[0]):
        for j in range(x_axis.shape[0]):
            input = [[y_axis[i], x_axis[j]]]
            vals = interactions_continuous(model, X, Y, input, [1, 1])
            dist[i, j] = vals[1, 1]
    fig, ax = plt.subplots()
    im = ax.imshow(dist)
    cbar_kw = {}
    cbarlabel = "class +1 proportion"
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.set_xticks(np.arange(20, step=2))
    ax.set_yticks(np.arange(20, step=2))
    ax.set_xticklabels(np.round(np.arange(-1, 1, step = .2), 2))
    ax.set_yticklabels(np.round(np.arange(-1, 1, step = .2), 2))
    plt.show()

def tree_dec(a):
    x1, x2 = a
    if x2 > 0:
        if x1 > -1:
            return 1
        return -1
    else:
        if x1 > 2:
            return 1
        return -1
def heat_map2():
    X, Y = generate_x_y_data(1000, (-5, 5), tree_dec, features=2)
    plt.scatter(x=X[:, 0], y=X[:, 1], c=Y)
    plt.show()
    model = train(X, Y)
    values = traverse_all_paths(model, X, Y, [1, 1])
    grid = make_grid(values, (-5, 5), (-5, 5), .1)
    fig, ax = plt.subplots()
    im = ax.imshow(grid)
    cbar_kw = {}
    cbarlabel = "class +1 proportion"
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.set_xticks(np.arange(100, step=10))
    ax.set_yticks(np.arange(100, step=10))
    ax.set_xticklabels(np.round(np.arange(-5, 5, step = 1), 2))
    ax.set_yticklabels(np.round(np.arange(-5, 5, step = 1), 2))
    plt.show()
    plt.show()

def heat_map_noise():
    X, Y = generate_xor_data(100, continuous_x=True, nonsense_vars=5)
    test_X, test_Y = generate_xor_data(100, continuous_x = True, nonsense_vars=5)
    model = train(X, Y)
    acc = model.score(test_X, test_Y)
    print("accuracy", acc)
    values = traverse_all_paths(model, X, Y, [1, 1, 0, 0, 0, 0, 0])
    grid = make_grid(values, (-1, 1), (-1, 1), .01)
    fig, ax = plt.subplots()
    im = ax.imshow(grid)
    cbar_kw = {}
    cbarlabel = "class +1 proportion"
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.set_xticks(np.arange(200, step=20))
    ax.set_yticks(np.arange(200, step=20))
    ax.set_xticklabels(np.round(np.arange(-1, 1, step = .2), 2))
    ax.set_yticklabels(np.round(np.arange(-1, 1, step = .2), 2))
    plt.show()
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

def test_new_heatmap():
    X, Y = generate_xor_data(100, continuous_x = True)
    model = train(X, Y)
    values = traverse_all_paths(model, X, Y, [1, 1])
    grid = make_grid(values, (-1, 1), (-1, 1), .01)
    fig, ax = plt.subplots()
    im = ax.imshow(grid)
    cbar_kw = {}
    cbarlabel = "class +1 proportion"
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.set_xticks(np.arange(200, step=20))
    ax.set_yticks(np.arange(200, step=20))
    ax.set_xticklabels(np.round(np.arange(-1, 1, step = .2), 2))
    ax.set_yticklabels(np.round(np.arange(-1, 1, step = .2), 2))
    plt.show()

def test_xy_uniform():
    X = np.random.uniform(0, 10, (10000, 2))
    Y = X[:, 0] * X[:, 1]
    model = train_cont(X, Y)
    X_test = np.random.uniform(0, 10, (100, 2))
    Y_test = X_test[:, 0] * X_test[:, 1]
    acc = model.score(X_test, Y_test)
    print("accuracy", acc)
    values = traverse_all_paths(model, X, Y, [1, 0], continuous_y=True)
    x_axis = np.arange(0, 10, .01)
    line = make_line(values, (0, 10), .01, 0)
    expected = 5 * x_axis
    plt.plot(x_axis, line)
    plt.plot(x_axis, expected)
    plt.show()

def test_xy_normal():
    X_0 = np.random.uniform(0, 10, (10000, 1))
    X_1 = np.random.normal(7, 1, (10000, 1))
    X = np.hstack((X_0, X_1))
    Y = X[:, 0] * X[:, 1]
    model = train_cont(X, Y)
    values = traverse_all_paths(model, X, Y, [1, 0], continuous_y=True)
    x_axis = np.arange(0, 10, .01)
    line = make_line(values, (0, 10), .01, 0)
    expected = 7 * x_axis
    plt.plot(x_axis, line)
    plt.plot(x_axis, expected)
    plt.show()

def test_x_squared():
    X_0 = np.random.uniform(0, 10, (10000, 1))
    X = np.hstack((X_0, X_0))
    Y = X[:, 0] * X[:, 1]
    model = train_cont(X, Y)
    X_0_test = np.random.uniform(0, 10, (1000, 1))
    X_test = np.hstack((X_0_test, X_0_test))
    Y_test = X_test[:, 0] * X_test[:, 1]
    acc = model.score(X_test, Y_test)
    preds = model.predict(X_test)
    Y_hat_test = model.predict(X_test)
    print("accuracy", acc)
    values = traverse_all_paths(model, X, Y, [1, 0], continuous_y=True)
    values_2 = traverse_all_paths(model, X, Y, [0, 1], continuous_y=True)
    x_axis = np.arange(0, 10, .01)
    line = make_line(values, (0, 10), .01, 0)
    line_2 = make_line(values_2, (0, 10), .01, 1)
    expected = x_axis * x_axis
    difference1 = np.linalg.norm(expected - line, ord=2)
    difference2 = np.linalg.norm(expected - line_2, ord=2)
    print("MSEs:", difference1, difference2)
    plt.plot(x_axis, line)
    plt.plot(x_axis, line_2)
    plt.plot(x_axis, expected)
    #plt.scatter(X_0, Y)
    plt.scatter(X_test[:, 0], preds)
    plt.show()
"""
def test_x_single():
    X = np.random.uniform(0, 10, (10000, 1))
    X_test = np.random.uniform(0, 10, (50, 1))
    Y = X ** 2
    model = train_cont(X, Y)
    preds = model.predict(X_test)
    values = traverse_all_paths(model, X, Y, [1], continuous_y=True)
    x_axis = np.arange(0, 10, .01)
    line = make_line(values, (0, 10), .01, 0)
    expected = x_axis ** 2
    plt.plot(x_axis, line)
    plt.plot(x_axis, expected)
    plt.scatter(X_test, preds)
    plt.show()

def test_xy_correlated(corr):
    t_start = time.clock()
    X, covs = generate_correlated(10000, [5, 5], [1, 1], corr)
    X_test, _ = generate_correlated(100, [5, 5], [1, 1], corr)
    #plt.scatter(X[:, 0], X[:, 1])
    #plt.show()
    Y = X[:, 0] * X[:, 1]
    Y_test = X_test[:, 0] * X_test[:, 1]
    model = train_cont(X, Y)
    acc = model.score(X_test, Y_test)
    t_train = time.clock()
    values = traverse_all_paths(model, X, Y, [1, 0], continuous_y=True)
    t_trav = time.clock()
    x_axis = np.arange(0, 10, .1)
    line = make_line(values, (0, 10), .1, 0)
    t_line = time.clock()
    plt.scatter(X[:, 0], Y,color='r',s=.0001)
    plt.show()
    plt.plot(x_axis, line)
    plt.show()

def test_xy_uniform_rf(n):
    X = np.random.uniform(0, 10, (10000, 2))
    Y = X[:, 0] * X[:, 1]
    model = train_cont_rf(X, Y, n)
    trees = model.estimators_
    line = np.zeros(100)
    x_axis = np.arange(0, 10, .1)
    for tree in trees:
        values = traverse_all_paths(tree, X, Y, [1, 0], continuous_y=True)
        line += make_line(values, (0, 10), .1, 0)
    line = line/n
    expected = 5 * x_axis
    difference = np.linalg.norm(expected - line, ord=2)
    print("MSE:", difference)
    plt.plot(x_axis, line)
    plt.plot(x_axis, expected)
    plt.show()

def test_x_squared_uniform_rf(n):
    X_0 = np.random.uniform(0, 10, (10000, 1))
    X = np.hstack((X_0, X_0))
    Y = X[:, 0] * X[:, 0]
    model = train_cont_rf(X, Y, n)
    trees = model.estimators_
    line = np.zeros(100)
    x_axis = np.arange(0, 10, .1)
    for tree in trees:
        values = traverse_all_paths(tree, X, Y, [1, 0], continuous_y=True)
        line += make_line(values, (0, 10), .1, 0)
    line = line/n
    expected = x_axis ** 2
    difference = np.linalg.norm(expected - line, ord=2)
    print("MSE:", difference)
    plt.plot(x_axis, line)
    plt.plot(x_axis, expected)
    plt.show()

test_x_single()
"""
#test_x_squared_uniform_rf(100)
