import numpy as np
from sklearn import tree
from sklearn import ensemble

def train(X, Y):
    model = tree.DecisionTreeClassifier()
    model = model.fit(X, Y)
    return model

def train_cont(X, Y):
    model = tree.DecisionTreeRegressor()
    model = model.fit(X, Y)
    return model

def train_rf(X, Y, n):
    model = ensemble.RandomForestClassifier(n_estimators=n)
    model = model.fit(X, Y)
    return model

def train_cont_rf(X, Y, n):
    model = ensemble.RandomForestRegressor(n_estimators=n)
    model = model.fit(X, Y)
    return model

def sanity_check(model):
    print("prediciton for (1, 1):", model.predict([[1, 1]]))
    print("prediction for (1, -1):", model.predict([[1, -1]]))
    print("prediciton for (-1, 1):", model.predict([[-1, 1]]))
    print("prediction for (-1, -1):", model.predict([[-1, -1]]))
