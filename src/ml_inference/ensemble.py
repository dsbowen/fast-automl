from .linear_model import ConstrainedLinearRegression
from .metrics import check_scoring

import numpy as np
from joblib import Parallel
from scipy import optimize
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone, is_classifier
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.ensemble._base import _fit_single_estimator
from sklearn.ensemble._stacking import _BaseStacking
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import (
    check_cv, cross_val_predict, cross_val_score
)
from sklearn.utils.fixes import delayed

from copy import deepcopy

def _predict_single_estimator(estimator, X, method='predict'):
    assert method in ('predict', 'predict_proba')
    return getattr(estimator, method)(X)


class ClassifierWeighter(ClassifierMixin, BaseEstimator):
    def __init__(self, loss=log_loss):
        self.loss = loss
        
    def fit(self, X, y):
        # X is (n_samples, n_classes, n_estimators)
        def loss(coef_):
            return self.loss(y, (X*coef_).sum(axis=-1))
        
        n_estimators = X.shape[-1]
        self.classes_ = np.arange(X.shape[1])
        if n_estimators == 1:
            self.coef_ = np.array([1])
            return self
        constraint = optimize.LinearConstraint(np.array([1]*n_estimators), 1, 1)
        coef0 = np.array([1./n_estimators]*n_estimators)
        self.coef_ = optimize.minimize(loss, x0=coef0, constraints=[constraint]).x
        return self
    
    def predict_proba(self, X):
        return (X*self.coef_).sum(axis=-1)
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class _BaseStackingCV(_BaseStacking):
    """
    Parameters
    ----------
    shuffle_cv : bool, default=True
        Indicates that cross validator should shuffle observations.
    """
    def __init__(
            self, estimators, cv=None, shuffle_cv=True, scoring=None, n_jobs=None, verbose=0
        ):
        self.shuffle_cv = shuffle_cv
        self.scoring = scoring
        super().__init__(estimators, cv=cv, n_jobs=n_jobs, verbose=verbose)

    def fit(self, X, y, sample_weight=None):
        raise NotImplementedError()

    def predict(self, X):
        assert hasattr(self, 'best_estimator_'), (
            'Estimator has not been fitted. Call `fit` before `predict`'
        )
        return self.best_estimator_.predict(X)
    
    def predict_proba(self, X):
        assert hasattr(self, 'best_estimator_'), (
            'Estimator has not been fitted. Call `fit` before `predict`'
        )
        return self.best_estimator_.predict_proba(X)
        
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
        method = 'predict_proba' if is_classifier(self) else 'predict'
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(cross_val_predict)(
                clone(est), X, y, 
                cv=deepcopy(cv), 
                n_jobs=self.n_jobs, 
                fit_params=fit_params, 
                verbose=self.verbose,
                method=method
            )
            for est in estimators
        )
        if is_classifier(self):
            return np.array(predictions).transpose(1, 2, 0)
        return np.array(predictions).T
    
    def _make_best_estimator(self, estimators, weights):
        if is_classifier(self):
            voting_estimator_cls = VotingClassifier
            kwargs = {'voting': 'soft'}
        else:
            voting_estimator_cls = VotingRegressor
            kwargs = {}
        return voting_estimator_cls(
            estimators, weights=weights,
            n_jobs=self.n_jobs, verbose=self.verbose, **kwargs
        )
    

class _RFEVotingEstimatorCV(_BaseStackingCV):
    """
    Selects voting regressors using recursive feature elimination.
    """
    def fit(self, X, y, sample_weight=None):
        def compute_rfe_scores(X_meta):
            # compute CV scores for each step of recursive feature elimination
            estimators = self.estimators.copy()
            rfe_scores = []
            while estimators:
                weight_fitter.fit(X_meta, y)
                # if any weights are negative, estimator has overfit
                # assign -inf as the score
                if np.any(weight_fitter.coef_ < 0):
                    score = -np.inf
                else:
                    score = cross_val_score(
                        weight_fitter, X_meta, y, cv=cv, scoring=scoring
                    ).mean()
                # append score, current estimators, weights
                rfe_scores.append(
                    (score, estimators.copy(), weight_fitter.coef_)
                )
                # drop estimator with the least weight
                drop_idx = int(np.argmin(weight_fitter.coef_))
                estimators.pop(drop_idx)
                X_meta = np.delete(X_meta, drop_idx, axis=-1)
            return max(rfe_scores, key=lambda x: x[0])
        
        if is_classifier(self):
            weight_fitter = ClassifierWeighter()
            best_estimator_cls = VotingClassifier
        else:
            weight_fitter = ConstrainedLinearRegression(1)
            best_estimator_cls = VotingRegressor
        names, all_estimators = self._validate_estimators()
        cv = self._check_cv(y)
        scoring = check_scoring(self.scoring, classifier=is_classifier(self))
        X_meta = self._cross_val_predict(
            X, y, all_estimators, cv, sample_weight
        )
        self.best_score_, estimators, self.weights_ = compute_rfe_scores(X_meta)
        self.names_ = [name for name, _ in estimators]
        self._fit_estimators(X, y, estimators, sample_weight)
        self.best_estimator_ = self._make_best_estimator(estimators, self.weights_)
        return self


class RFEVotingRegressorCV(RegressorMixin, _RFEVotingEstimatorCV):
    pass


class RFEVotingClassifierCV(ClassifierMixin, _RFEVotingEstimatorCV):
    pass


class _StepwiseVotingEstimatorCV(_BaseStackingCV):
    """
    Selects estimators for the voting regressor using stepwise addition.
    """
    def fit(self, X, y, sample_weight=None):
        def compute_stepwise_scores():
            # compute CV score from stepwise addition of estimators to the 
            # ensemble
            best_score = -np.inf
            # estimators in the ensemble
            in_estimators, X_in = [], None
            # estimators out of the ensemble
            out_estimators, X_out = self.estimators.copy(), X_meta
            while out_estimators:
                # add estimators from outside the ensemble to the ensemble
                new_scores = [compute_new_score(X_in, col) for col in X_out.T]
                idx = np.argmax(new_scores)
                if new_scores[idx] <= best_score:
                    break
                # if the new estimator improved the CV score
                # add it to the ensemble
                best_score = new_scores[idx]
                in_estimators.append(out_estimators.pop(idx))
                if is_classifier(self):
                    col = X_out[:, :, idx].reshape(X_meta.shape[0], X_meta.shape[1], 1)
                else:
                    col = X_out[:, idx].reshape(-1, 1)
                X_in = (
                    col if X_in is None 
                    else np.concatenate((X_in, col), axis=-1)
                )
                X_out = np.delete(X_out, idx, axis=-1)
            return best_score, in_estimators, weight_fitter.fit(X_in, y).coef_
        
        def compute_new_score(X_in, col):
            # add new estimator CV predictions to the dataframe
            if is_classifier(self):
                col = col.T.reshape(X_meta.shape[0], X_meta.shape[1], 1)
            else:
                col = col.reshape(-1, 1)
            X = (
                col if X_in is None 
                else np.concatenate((X_in, col), axis=-1)
            )
            weights = weight_fitter.fit(X, y).coef_
            # add the CV score for the estimator to new_scores
            # if any of the weights on the estimators are negative, 
            # the ensemble has overfit; assign a weight of -inf
            if np.any(weights < 0):
                return -np.inf
            return cross_val_score(
                weight_fitter, X, y, cv=cv, scoring=scoring
            ).mean()
            
        if is_classifier(self):
            weight_fitter = ClassifierWeighter()
            best_estimator_cls = VotingClassifier
            best_estimator_kwargs = {'voting': 'soft'}
        else:
            weight_fitter = ConstrainLinearRegression(1)
            best_estimator_cls = VotingRegressor
            best_estimator_kwargs = {}
        names, all_estimators = self._validate_estimators()
        cv = self._check_cv(y)
        scoring = check_scoring(self.scoring, classifier=is_classifier(self))
        X_meta = self._cross_val_predict(
            X, y, all_estimators, cv, sample_weight
        )
        self.best_score_, estimators, self.weights_ = compute_stepwise_scores()
        self.names_ = [name for name, _ in estimators]
        self._fit_estimators(X, y, estimators, sample_weight)
        self.best_estimator_ = self._make_best_estimator(estimators, self.weights_)
        return self
    
    
class StepwiseVotingRegressorCV(RegressorMixin, _StepwiseVotingEstimatorCV):
    pass


class StepwiseVotingClassifierCV(ClassifierMixin, _StepwiseVotingEstimatorCV):
    pass