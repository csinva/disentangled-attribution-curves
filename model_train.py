import numpy as np
from sklearn import tree

def train(X, Y):
    model = tree.DecisionTreeClassifier()
    model = model.fit(X, Y)
    return model

def sanity_check(model):
    print("prediciton for (1, 1):", model.predict([[1, 1]]))
    print("prediction for (1, -1):", model.predict([[1, -1]]))
    print("prediciton for (-1, 1):", model.predict([[-1, 1]]))
    print("prediction for (-1, -1):", model.predict([[-1, -1]]))
