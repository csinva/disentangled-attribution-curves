import numpy as np
X = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
Y = [[0], [1], [1], [0]]
X = np.array(X)
Y = np.array(Y)

"""
PARAMETERS
n: number of data points
continuous_x: true if x takes continuous values, false otherwise
nonsense_vars: adds the specified additional features that have no effect on the output
OUTPUT
a tuple containing at index 0 a matrix with rows representing datapoints X,
and at index 1 a column vector of results Y
"""
def generate_xor_data(n, continuous_x=False, nonsense_vars=0):
    if(not continuous_x):
        X = np.random.choice([-1, 1], (n, 2 + nonsense_vars))
    else:
        X = np.random.uniform(-1, 1, (n, 2 + nonsense_vars))
    Y = np.transpose(X[:, 0] * X[:, 1] < 0)
    return (X, Y)
