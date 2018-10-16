import numpy as np
import matplotlib.pyplot as plt


def train_test_split(x, y, size=0.2, seed=None):
    if seed:
        np.random.seed(seed)
    shuffle_indexes = np.random.permutation(len(x))
    test_num = int(len(x)*size)
    test_indexes = shuffle_indexes[:test_num]
    train_indexes = shuffle_indexes[test_num:]
    x_train = x[train_indexes]
    y_train = y[train_indexes]
    x_test = x[test_indexes]
    y_test = y[test_indexes]
    return x_train, y_train, x_test, y_test
