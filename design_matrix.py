import math
import numpy as np
import random
import scipy as sp

# helper functions:
# 1) for feature scaling (mean 0 and standard deviation 1)
# 2) to create design matrix with column of ones
# 3) split data randomly into 80-20 split for train and test sets

# helper function to add column of ones
def form_design_matrix(training_inputs, degree):
    m = len(training_inputs)
    d = degree
    X = np.ones(shape=(m, d + 1))

    for i in range(m):
        for j in range(1, d + 1):
            X[i][j] = training_inputs[i][j - 1]

    return X

# scaling (mean 0 and standard deviation 1)
def scale(x):
    Xz = sp.stats.zscore(x[:,1:]) # scale *except* for the column of ones
    m = Xz.shape[0]
    ones_vec = np.ones((m,1))
    design_mat_scaled = np.hstack((ones_vec,Xz))
    return design_mat_scaled


# split randomly into 80-20 train and test sets
def split_train_test(x, y):
    xy = list(zip(x, y))
    random.shuffle(xy)
    x, y = zip(*xy)

    split = int(len(x) * 0.8)
    train_x = np.array(x[:split])
    train_y = np.array(y[:split])
    test_x = np.array(x[split:])
    test_y = np.array(y[split:])

    train_y = np.reshape(train_y, (len(train_y), 1))
    test_y = np.reshape(test_y, (len(test_y), 1))

    return train_x, train_y, test_x, test_y