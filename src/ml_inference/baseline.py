import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils.multiclass import check_classification_targets


class BaselineRegressor(LinearRegression):
    def fit(self, X, y, sample_weight=None):
        return super().fit(np.ones((X.shape[0], 1)), y, sample_weight)
        
    def predict(self, X):
        return super().predict(np.ones((X.shape[0], 1)))


class BaselineClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        check_classification_targets(y)
        if sample_weight is None:
            self.classes_, self.counts_ = np.unique(y, return_counts=True)
        else:
            df = pd.DataFrame({'y': y, 'sample_weight': sample_weight})
            df = df.groupby('y').sum()
            self.classes_ = df.index.values
            self.counts_ = df.sample_weight.values
        self.counts_ = self.counts_ / self.counts_.sum()
        self.dominant_class_ = self.classes_[np.argmax(self.counts_)]
        return self
    
    def predict(self, X):
        return np.array([self.dominant_class_]*X.shape[0])
    
    def predict_proba(self, X):
        return np.array([self.counts_]*X.shape[0])