import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from math import sqrt

class KNNClassifier:
    def __init__(self, k):
        """初始化kNN分类器"""
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """根据训练数据集X_train和y_train 训练kNN分类器"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        # 给定带测数据x，返回x的预测结果值
        distance = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distance)
        topK_y = [self._y_train[i] for i in nearest[:self.  k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def score(self, x_test, y_test):
        '''根据meshing数据集新x_test 和 y_test确定模型准确度'''
        y_predict = self.predict(x_test)
        acc = sum(y_predict == y_test)/len(y_test)
        return acc