from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LassoLars, Ridge, ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from scipy.stats import expon, uniform, poisson, randint
from xgboost import XGBRegressor


class Tuner():
    def __init__(self, preprocess=[]):
        self.preprocess = preprocess if isinstance(preprocess, list) else [preprocess]
        
    def tune(self, X, y, n_iter=10, n_jobs=None):
        param_distributions = self.get_param_distributions(X, y)
        est = RandomizedSearchCV(
            self.make_estimator(), 
            param_distributions, 
            n_iter=n_iter,
            n_jobs=n_jobs
        ).fit(X, y)
        res = list(zip(est.cv_results_['mean_test_score'], est.cv_results_['params']))
        self.best_params_ = sorted(res, key=lambda x: -x[0])
        return self
    
    def make_best_estimator(self, idx=0, return_score=False):
        if idx < 1 and idx != 0:
            # interpret idx as a quantile
            idx = round(idx*(len(self.best_params_)-1))
        est = self.make_estimator(**self.best_params_[idx][1])
        return (self.best_params_[idx][0], est) if return_score else est
    
    def rm_params(self, idx=0):
        if idx < 1 and idx != 0:
            # interpret idx as quantile
            idx = round(idx*(len(self.best_params_)-1))
        return self.best_params_.pop(idx)


class RandomForestRegressorTuner(Tuner):    
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocess,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            RandomForestRegressor()
        )
        return est.set_params(**params)
        
    def get_param_distributions(self, X, y):
        return {
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components': list(range(1, X.shape[1])),
            'randomforestregressor__n_estimators': poisson(1, 2**6)
        }
    
    
class LassoLarsTuner(Tuner):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocess,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            LassoLars(normalize=True)
        )
        return est.set_params(**params)
        
    def get_param_distributions(self, X, y):
        return {
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components': list(range(1, X.shape[1])),
            'lassolars__alpha': expon(0, 1)
        }
    
    
class RidgeTuner(Tuner):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocess,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            Ridge(normalize=True)
        )
        return est.set_params(**params)
        
    def get_param_distributions(self, X, y):
        return {
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components': list(range(1, X.shape[1])),
            'ridge__alpha': expon(0, 1)
        }
    
    
class ElasticNetTuner(Tuner):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocess,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            ElasticNet(normalize=True)
        )
        return est.set_params(**params)
    
    def get_param_distributions(self, X, y):
        return {
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components': list(range(1, X.shape[1])),
            'elasticnet__alpha': expon(0, 1),
            'elasticnet__l1_ratio': uniform(0, 1)
        }
    
    
class KernelRidgeTuner(Tuner):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocess,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            StandardScaler(),
            KernelRidge()
        )
        return est.set_params(**params)
    
    def get_param_distributions(self, X, y):
        return {
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components': list(range(1, X.shape[1])),
            'kernelridge__alpha': expon(0, 1),
            'kernelridge__degree': list(range(2, 5)),
            'kernelridge__kernel': ['linear', 'poly', 'rbf', 'laplacian']
        }
    
    
class SVRTuner(Tuner):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocess,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            StandardScaler(),
            SVR()
        )
        return est.set_params(**params)
        
    def get_param_distributions(self, X, y):
        return {
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components': list(range(1, X.shape[1])),
            'svr__C': expon(0, 1),
            'svr__degree': list(range(2, 5)),
            'svr__kernel': ['linear', 'poly', 'rbf'],
        }

    
class KNeighborsRegressorTuner(Tuner):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocess,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            StandardScaler(),
            KNeighborsRegressor()
        )
        return est.set_params(**params)
        
    def get_param_distributions(self, X, y):
        return {
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components': list(range(1, X.shape[1])),
            'kneighborsregressor__n_neighbors': randint(1, .05*X.shape[0]),
            'kneighborsregressor__weights': ['uniform', 'distance']
        }
    
    
class AdaBoostRegressorTuner(Tuner):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocess,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            AdaBoostRegressor()
        )
        return est.set_params(**params)
        
    def get_param_distributions(self, X, y):
        return {
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components': list(range(1, X.shape[1])),
            'adaboostregressor__n_estimators': poisson(1, 2**5)
        }

    
class XGBRegressorTuner(Tuner):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocess,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            XGBRegressor()
        )
        return est.set_params(**params)
    
    def get_param_distributions(self, X, y):
        return {
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components': list(range(1, X.shape[1])),
            'xgbregressor__gamma': expon(0, 1),
            'xgbregressor__max_depth': list(range(10)),
            'xgbregressor__min_child_weight': expon(0, 1),
            'xgbregressor__max_delta_step': expon(0, 1),
            'xgbregressor__lambda': expon(0, 1),
            'xgbregressor__alpha': expon(0, 1),
        }