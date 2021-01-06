import numpy as np
from sklearn.linear_model import LinearRegression


class BaselineRegression(LinearRegression):
    def fit(self, X, y):
        self.y_mean = y.mean()
        return self
        
    def predict(self, X):
        return np.full(X.shape[0], self.y_mean)