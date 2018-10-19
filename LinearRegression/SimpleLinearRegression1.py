import numpy as np


class SimpleLinearRegression1:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """根据训练集x_train 和y_train训练回归模型"""
        assert x_train.ndim == 1, "must be single feature training data"
        assert len(x_train) == len(y_train), "must be equal to size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0

        for x_i, y_i in zip(x_train, y_train):
            num += (x_i - x_mean) * (y_i - y_mean)
            d += (x_i - x_mean) ** 2

        self.a_ = num/d
        self.b_ = y_mean - x_mean * self.a_

        return self

    def predict(self, x_predict):
        """x_predict 是个向量"""
        assert x_predict.ndim == 1, "1维度"
        assert self.a_ is not None and self.b_ is not None, ""

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression1()"
