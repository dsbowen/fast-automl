import numpy as np
from sklearn.linear_model import LinearRegression


class BaselineRegressor(LinearRegression):
    def fit(self, X, y, sample_weight=None):
        return super().fit(np.ones((X.shape[0], 1)), y, sample_weight)
        
    def predict(self, X):
        return super().predict(np.ones((X.shape[0], 1)))