from .baseline import BaselineRegressor
from .tuners import (
    RandomForestRegressorTuner, LassoLarsTuner, RidgeTuner, ElasticNetTuner, 
    KernelRidgeTuner, SVRTuner, KNeighborsRegressorTuner, 
    AdaBoostRegressorTuner, XGBRegressorTuner
)

import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score

def make_default_tuners():
    return [
        RandomForestRegressorTuner(),
        LassoLarsTuner(),
        RidgeTuner(),
        ElasticNetTuner(),
        KernelRidgeTuner(),
        SVRTuner(),
        KNeighborsRegressorTuner(),
        AdaBoostRegressorTuner(),
        XGBRegressorTuner()
    ]


class AutoRegressor(VotingRegressor):
    def __init__(
            self, tuners=[], preprocess=[], estimators=[], weights=[], 
            n_jobs=None, verbose=False
        ):
        self.tuners = tuners or make_default_tuners()
        self.preprocess = (
            preprocess if isinstance(preprocess, list) else [preprocess]
        )
        super().__init__(estimators, weights=weights, n_jobs=n_jobs, verbose=verbose)
        
    def fit(self, X, y, *args, **kwargs):
        return super().fit(self.preprocess_X(X), y, *args, **kwargs)
    
    def predict(self, X, *args, **kwargs):
        return super().predict(self.preprocess_X(X), *args, **kwargs)
        
    def preprocess_X(self, X, verbose=False):
        for preprocessor in self.preprocess:
            X = preprocessor.fit(X).transform(X)
        return X
        
    def tune(self, X, y, n_iter=10, quantiles=[0, .1, .2]): 
        def tune_estimators():
            for i, tuner in enumerate(self.tuners):
                print('\nRunning tuner {} of {}'.format(i+1, len(self.tuners)))
                tuner.tune(X_preproc, y, n_iter=n_iter, n_jobs=self.n_jobs)
                print('Best estimator score: {:.4f}'.format(tuner.best_params_[0][0]))
                
        def add_estimator(i, best_score):
            print('\nAdding estimator {}'.format(i+1))
            tuner, estimator, weight, score, q = get_best_estimator(
                i, best_score
            )
            print('Best ensemble score: {:.4f}'.format(score))
            if estimator is not None:
                tuner.rm_params(q)
                estimators.append(('estimator_'+str(i), estimator))
                weights.append(weight)
                self.set_params(estimators=estimators, weights=weights)
            return score
            
        def get_best_estimator(i, best_score):
            best_tuner, best_estimator, best_weight, best_quantile = (
                None, None, None, None
            )
            for tuner in self.tuners:
                if tuner.best_params_:
                    for q in quantiles:
                        score, estimator = tuner.make_best_estimator(
                            q, return_score=True
                        )
                        weight = score - baseline_score
                        if weight > 0:
                            weights.append(weight)
                            estimators.append(('estimator_'+str(i),estimator))
                            self.set_params(
                                estimators=estimators, weights=weights
                            )
                            cv_score = cross_val_score(self, X, y).mean()
                            if cv_score > best_score:
                                best_tuner = tuner
                                best_estimator = estimator
                                best_weight = weight
                                best_score = cv_score
                                best_quantile = q
                            estimators.pop(), weights.pop()
            return (
                best_tuner, 
                best_estimator, 
                best_weight, 
                best_score, 
                best_quantile
            )
    
        X_preproc = self.preprocess_X(X)
        tune_estimators()
        baseline_score = cross_val_score(
            BaselineRegressor(), X_preproc, y
        ).mean()
        estimators, weights = [], []
        i, best_score = 0, -np.inf
        while True:
            score = add_estimator(i, best_score)
            if score <= best_score:
                break
            best_score = score
            i += 1
        return self
    
    def get_params(self, **kwargs):
        params = super().get_params(**kwargs)
        params.update({
            'tuners': self.tuners,
            'preprocess': self.preprocess
        })
        return params
    
    def set_params(self, tuners=[], preprocess=[], **params):
        if tuners:
            self.tuners = tuners
        if preprocess:
            self.preprocess = (
                preprocess if isinstance(preprocess, list) else [preprocess]
            )
        return super().set_params(**params)