import numpy as np
from sklearn.linear_model import LinearRegression, Ridge as RidgeBase


class ConstrainedLinearRegression(LinearRegression):
    def __init__(
            self, constraint=0, normalize=False, copy_X=True, n_jobs=None
        ):
        """
        Parameters
        ----------
        constraint : scalar, default=0
            Sum of the regression coefficients.

        normalize : bool, default=False
        
        copy_X : bool, default=True

        n_jobs : int or None, default=None        
        """
        self.constraint = constraint
        super().__init__(
            fit_intercept=False, 
            normalize=normalize, 
            copy_X=copy_X, 
            n_jobs=n_jobs
        )
        
    def fit(self, X, y, sample_weight=None):
        if X.shape[1] == 1:
            self.coef_ = np.array([self.constraint])
            return self
        if hasattr(X, 'values'):
            X = X.values
        X_0, X_rest = X[:,0], X[:,1:]
        X_rest = (X_rest.T - X_0).T
        y = y - self.constraint * X_0
        super().fit(X_rest, y, sample_weight)
        self.coef_ = np.insert(
            self.coef_, 0, self.constraint - self.coef_.sum()
        )
        return self
    
    def predict(self, X):
        return X @ self.coef_


class Ridge(RidgeBase):
    def __init__(
            self, alpha=1., prior_weight=0, normalize_coef=False, 
            fit_intercept=True, normalize=False, copy_X=True, max_iter=None, 
            tol=.001, solver='auto', random_state=None
        ):
        self.prior_weight = prior_weight
        self.normalize_coef = normalize_coef
        super().__init__(
            alpha, fit_intercept=fit_intercept, normalize=normalize, 
            copy_X=copy_X, max_iter=max_iter, tol=tol, solver=solver, 
            random_state=random_state
        )
        
    def fit(self, X, y, sample_weight=None):
        y = y - (self.prior_weight * X).sum(axis=1)
        super().fit(X, y, sample_weight)
        if self.normalize_coef:
            self.coef_ -= self.coef_.mean()
        return self
    
    def predict(self, X):
        return super().predict(X) + (self.prior_weight * X).sum(axis=1)