from .cv_estimators import (
    PCARandomForestRegressorCV, PCALassoLarsCV, PCARidgeCV, PCAElasticNetCV,
    PCAKernelRidgeCV, PCASVRCV, PCAKNeighborsRegressorCV, 
    PCAAdaBoostRegressorCV, PCAXGBRegressorCV,
    RandomForestRegressorCV, LassoLarsCV, RidgeCV, ElasticNetCV,
    KernelRidgeCV, SVRCV, KNeighborsRegressorCV, AdaBoostRegressorCV,
    XGBRegressorCV
)
from .ensemble import RFEVotingRegressorCV, StepwiseVotingRegressorCV

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

def make_default_cv_regressors():
    return [
        RandomForestRegressorCV(),
        PCARandomForestRegressorCV(),
        LassoLarsCV(),
        PCALassoLarsCV(),
        RidgeCV(),
        PCARidgeCV(),
        ElasticNetCV(),
        PCAElasticNetCV(),
        KernelRidgeCV(),
        PCAKernelRidgeCV(),
        SVRCV(),
        PCASVRCV(),
        KNeighborsRegressorCV(),
        PCAKNeighborsRegressorCV(),
        AdaBoostRegressorCV(),
        PCAAdaBoostRegressorCV(),
        XGBRegressorCV(),
        PCAXGBRegressorCV()
    ]

class AutoRegressor(BaseEstimator, RegressorMixin):
    """
    Parameters
    ----------
    cv_estimators : list of CVEstimators, default=[]
        If an empty list, a default list of CVEstimators will be created.

    preprocessors : list, default=[]
        List of preprocessing steps before data is fed to the `cv_estimators`.

    ensemble_method : str, default='rfe'
        If `'rfe'`, the ensemble is created using recursive feature elimination. If `'stepwise'`, the ensemble is created using stepwise addition.

    max_ensemble_size : int, default=100
        The maximum number of estimators to consider adding to the ensemble.

    n_ensembles : int, default=1
        Number of ensembles to create using different CV splits. These ensembles get equal votes in a meta-ensemble.

    n_iter : int, default=10
        Number of iterations to run randomized search for the CVEstimators.

    n_jobs : int or None, default=None
        Number of background jobs to run.

    verbose : bool, default=False

    cv : cv split, default=None
    """
    def __init__(
            self, cv_estimators=[], preprocessors=[], ensemble_method='rfe', max_ensemble_size=100, n_ensembles=1, n_iter=10, n_jobs=None, verbose=False, cv=None
        ):
        self.cv_estimators = cv_estimators or make_default_cv_regressors()
        self.preprocessors = (
            preprocessors if isinstance(preprocessors, list) 
            else [preprocessors]
        )
        assert ensemble_method in ('rfe', 'stepwise')
        self.ensemble_method = ensemble_method
        self.max_ensemble_size = max_ensemble_size
        self.n_ensembles = n_ensembles
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.cv = cv
        
    def fit(self, X, y, sample_weight=None):
        def fit_cv_estimator(i, est):
            print('\nTuning estimator {} of {}: {}'.format(
                i+1, len(self.cv_estimators), est.__class__.__name__
            ))
            est.fit(X, y, n_iter=self.n_iter, n_jobs=self.n_jobs)
            print('Best estimator score: {:.4f}'.format(est.best_score_))

        def make_best_estimators():
            cv_results = [
                (res, est) 
                for est in self.cv_estimators 
                for res in est.cv_results_
            ]
            # create the best estimators
            # x[0][0] is the cv score
            best_params = sorted(
                cv_results, key=lambda x: x[0][0], reverse=True
            )[:self.max_ensemble_size]
            best_estimators = [
                est.make_estimator(**params) 
                for (_, params), est in best_params
            ]
            return [
                ('estimator {}'.format(i), estimator) 
                for i, estimator in enumerate(best_estimators)
            ]

        def build_ensemble(i):
            # build an ensemble of the best estimators
            print(
                '\nBuilding ensemble {} of {}'.format(i+1, self.n_ensembles)
            )
            if self.ensemble_method == 'rfe':
                reg = RFEVotingRegressorCV(
                    best_estimators, n_jobs=self.n_jobs, cv=self.cv
                )
            elif self.ensemble_method == 'stepwise':
                reg = StepwiseVotingRegressorCV(
                    best_estimators, n_jobs=self.n_jobs, cv=self.cv
                )
            reg.fit(X, y, sample_weight=sample_weight)
            print('Best ensemble score: {:.4f}'.format(reg.best_score_))
            return 'ensemble {}'.format(i+1), reg.best_estimator_
                
        for preprocessor in self.preprocessors:
            X = preprocessor.fit_transform(X)
        [fit_cv_estimator(i, est) for i, est in enumerate(self.cv_estimators)]
        best_estimators = make_best_estimators()
        ensembles = [
            build_ensemble(i) for i in range(self.n_ensembles)
        ]
        # if only one ensemble is built, there's no need to create a meta-ensemble
        # if multiple ensembles are built, give them an equal vote in a voting regressor
        meta_ensemble = (
            ensembles[0][-1] if len(ensembles) == 1
            else VotingRegressor(ensembles)
        )
        self.best_estimator_ = make_pipeline(
            *self.preprocessors, meta_ensemble
        )
        return self
    
    def predict(self, X):
        return self.best_estimator_.predict(X)