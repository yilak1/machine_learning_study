import numpy as np


class LinearRegression:

    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        self._theta = None

    def fit_normal(self, x_train, y_train):
        assert x_train.shape[0] == y_train.shape[0], \
            "the size of x_train must be equal to the size of y_train"

        x_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        self._theta = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y_train)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self, x_predict):
        assert self.interception_ is not None and self.coef_ is not None, \
            "must fit before predict"
        assert x_predict.shape[1] == len(self.coef_), \
            "the feature number of x_predict must be equal to x_train"

        x_b = np.hstack([np.ones([len(x_predict), 1]), x_predict])
        return x_b.dot(self._theta)

    def score(self, x_test, y_test):
        """r2_score"""
        y_predict = self.predict(x_test)
        score = 1 - sum((y_predict - y_test) ** 2) / sum((np.mean(y_test) - y_test) **2)
        return score

    def __repr__(self):
        return "LinearRegression()"
