from sklearn.base import BaseEstimator, TransformerMixin as TransformerMixinBase


class TransformerMixin(TransformerMixinBase, BaseEstimator):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        raise NotImplementedError('Transfomer must implement transform method')


class ColumnSelector(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, y=None):
        X = X[self.columns]
        return X if y is None else (X, y)