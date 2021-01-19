from .linear_model import ConstrainedLinearRegression

import numpy as np
from joblib import Parallel
from scipy.stats import loguniform
from sklearn.base import clone, is_classifier
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble._base import _fit_single_estimator
from sklearn.ensemble._stacking import _BaseStacking
from sklearn.model_selection import (
    check_cv, cross_val_predict, cross_val_score
)
from sklearn.utils.fixes import delayed

from copy import deepcopy

def _predict_single_estimator(estimator, X):
    return estimator.predict(X)


class _BaseStackingCV(_BaseStacking):
    """
    Parameters
    ----------
    shuffle_cv : bool, default=True
        Indicates that cross validator should shuffle observations.
    """
    def __init__(
            self, estimators, cv=None, shuffle_cv=True, n_jobs=None, verbose=0
        ):
        self.shuffle_cv = shuffle_cv
        super().__init__(estimators, cv=cv, n_jobs=n_jobs, verbose=verbose)

    def fit(self, X, y, sample_weight=None):
        raise NotImplementedError()

    def predict(self, X):
        assert hasattr(self, 'best_estimator_'), (
            'Estimator has not been fitted. Call `fit` before `predict`'
        )
        return self.best_estimator_.predict(X)
        
    def transform(self, X):
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_single_estimator)(est, X)
            for est in self.estimators_
        )
        return self._concatenate_predictions(X, predictions)
    
    def _fit_estimators(self, X, y, estimators, sample_weight):
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_estimator)(clone(est), X, y, sample_weight)
            for _, est in estimators
        )
        
    def _check_cv(self, y):
        cv = check_cv(self.cv, y=y, classifier=is_classifier(self))
        if hasattr(cv, 'random_state') and cv.random_state is None:
            cv.random_state = np.random.RandomState()
        if hasattr(cv, 'shuffle') and self.shuffle_cv:
            cv.shuffle = True
        return cv
    
    def _cross_val_predict(self, X, y, estimators, cv, sample_weight):
        fit_params = (
            {} if sample_weight is None else {'sample_weight': sample_weight}
        )
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(cross_val_predict)(
                clone(est), X, y, 
                cv=deepcopy(cv), 
                n_jobs=self.n_jobs, 
                fit_params=fit_params, 
                verbose=self.verbose
            )
            for est in estimators
        )
        return self._concatenate_predictions(X, predictions)
    

class RFEVotingRegressorCV(_BaseStackingCV):
    """
    Selects voting regressors using recursive feature elimination.
    """
    def fit(self, X, y, sample_weight=None):
        def compute_rfe_scores(X_meta, estimators):
            # compute CV scores for each step of recursive feature elimination
            rfe_scores = []
            while estimators:
                linear_reg.fit(X_meta, y)
                # if any weights are negative, estimator has overfit
                # assign -inf as the score
                score = (
                    -np.inf if np.any(linear_reg.coef_ < 0)
                    else cross_val_score(linear_reg, X_meta, y, cv=cv).mean()
                )
                # append score, current estimators, weights
                rfe_scores.append(
                    (score, estimators.copy(), linear_reg.coef_)
                )
                # drop estimator with the least weight
                drop_idx = int(np.argmin(linear_reg.coef_))
                estimators.pop(drop_idx)
                X_meta = np.delete(X_meta, drop_idx, axis=1)
            return max(rfe_scores, key=lambda x: x[0])
        
        names, all_estimators = self._validate_estimators()
        cv = self._check_cv(y)
        X_meta = self._cross_val_predict(
            X, y, all_estimators, cv, sample_weight
        )
        linear_reg = ConstrainedLinearRegression(1)
        self.best_score_, estimators, self.weights_ = compute_rfe_scores(
            X_meta, self.estimators
        )
        self.names_ = [name for name, _ in estimators]
        self._fit_estimators(X, y, estimators, sample_weight)
        self.best_estimator_ = VotingRegressor(
            estimators, weights=self.weights_, 
            n_jobs=self.n_jobs, verbose=self.verbose
        )
        return self


class StepwiseVotingRegressorCV(_BaseStackingCV):
    """
    Selects estimators for the voting regressor using stepwise addition.
    """
    def fit(self, X, y, sample_weight=None):
        def compute_stepwise_scores(X_meta, estimators):
            # compute CV score from stepwise addition of estimators to the 
            # ensemble
            best_score = -np.inf
            # estimators in the ensemble
            in_estimators, X_in = [], None
            # estimators out of the ensemble
            out_estimators, X_out = estimators.copy(), X_meta
            while out_estimators:
                # add estimators from outside the ensemble to the ensemble
                new_scores = []
                for col in X_out.T:
                    # add new estimator CV predictions to the dataframe
                    col = col.reshape(-1, 1)
                    X = (
                        col if X_in is None 
                        else np.concatenate((X_in, col), axis=1)
                    )
                    weights = linear_reg.fit(X, y).coef_
                    # add the CV score for the estimator to new_scores
                    # if any of the weights on the estimators are negative, 
                    # the ensemble has overfit; assign a weight of -inf
                    new_scores.append(
                        -np.inf if np.any(weights < 0)
                        else cross_val_score(linear_reg, X, y, cv=cv).mean()
                    )
                # index of the best new estimator
                idx = np.argmax(new_scores)
                if new_scores[idx] <= best_score:
                    break
                # if the new estimator improved the CV score
                # add it to the ensemble
                best_score = new_scores[idx]
                col = X_out[:, idx].reshape(-1, 1)
                X_in = (
                    col if X_in is None 
                    else np.concatenate((X_in, col), axis=1)
                )
                X_out = np.delete(X_out, idx, axis=1)
                in_estimators.append(out_estimators.pop(idx))
            return best_score, in_estimators, linear_reg.fit(X_in, y).coef_
            
        names, all_estimators = self._validate_estimators()
        cv = self._check_cv(y)
        X_meta = self._cross_val_predict(
            X, y, all_estimators, cv, sample_weight
        )
        linear_reg = ConstrainedLinearRegression(1)
        self.best_score_, estimators, self.weights_ = compute_stepwise_scores(
            X_meta, self.estimators
        )
        self.names_ = [name for name, _ in estimators]
        self._fit_estimators(X, y, estimators, sample_weight)
        self.best_estimator_ = VotingRegressor(
            estimators, weights=self.weights_, 
            n_jobs=self.n_jobs, verbose=self.verbose
        )
        return self