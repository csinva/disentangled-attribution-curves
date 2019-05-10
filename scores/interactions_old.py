import numpy as np
from sklearn import tree

from model_train import train
from data import X, Y


"""
PARAMETERS
model: a decision tree trained on some data set
outcome_space_x: a matrix describing the possible x values in the total outcome space
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
def interactions(model, outcome_space_x, outcome_space_y, assignment, S):
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
                complies = complies and outcome_space_x[i][d[0]] >= d[1]
            else:
                complies = complies and outcome_space_x[i][d[0]] < d[1]
        if complies:
            outcomes.append(outcome_space_y[i])
    unique, counts = np.unique(outcomes, return_counts = True)
    unique = np.reshape(unique, (-1, 1))
    probs = np.reshape(counts/len(unique), (-1, 1))
    return np.hstack((unique, probs))

def main():
    model = train(X, Y)
    X_test = [[1, 1]]
    print(interactions(model, X, Y, X_test, [1, 0]))
    print(interactions(model, X, Y, X_test, [0, 1]))
    print(interactions(model, X, Y, X_test, [1, 1]))
    X_test = [[1, -1]]
    print(interactions(model, X, Y, X_test, [1, 0]))
    print(interactions(model, X, Y, X_test, [0, 1]))
    print(interactions(model, X, Y, X_test, [1, 1]))

main()
