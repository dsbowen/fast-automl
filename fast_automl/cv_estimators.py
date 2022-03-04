"""# Cross-validation estimators

Examples
--------
```python
from fast_automl.cv_estimators import RandomForestClassifierCV

from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score, train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True)
clf = RandomForestClassifierCV().fit(X_train, y_train, n_jobs=-1)
print('Cross val score: {:.4f}'.format(cross_val_score(clf.best_estimator_, X_train, y_train).mean()))
print('Test score: {:.4f}'.format(clf.score(X_test, y_test)))
```

Out:

```
Cross val score: 0.9696
Test score: 0.9800
```
"""

from .metrics import check_scoring

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, is_classifier
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, LassoLars, LogisticRegression, Ridge
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import expon, geom, uniform, poisson, randint
from xgboost import XGBClassifier, XGBRegressor

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def _get_n_pca_components_distribution(X):
    return list(range(1, min(X.shape)))


class CVBaseEstimator(BaseEstimator):
    """
    Base class for all CV estimators.

    Parameters
    ----------
    preprocessors : list, default=[]
        Preprocessing steps.

    param_distributions : dict, default={}
        Maps names of parameters to distributions. This overrides 
        parameters returned by the `get_param_distributions` method.

    Attributes
    ----------
    best_estimator_ : estimator
        Estimator which attained the best CV score under randomized search.

    best_score_ : scalar
        Best CV score attained by any estimator.

    cv_results_ : list
        List of (mean CV score, parameters) tuples.
    """
    def __init__(self, preprocessors=[], param_distributions={}):
        self.preprocessors = (
            preprocessors if isinstance(preprocessors, list) 
            else [preprocessors]
        )
        self.param_distributions = param_distributions

    def get_param_distributions(self, param_distributions={}):
        """
        Parameters
        ----------
        param_distributions : dict, default={}
            These are overridden by the `param_distributions` parameter passed
            to the constructor.

        Returns
        -------
        param_distributions : dict
            Parameter distributions used for randomized search.
        """
        param_distributions.update(self.param_distributions)
        return param_distributions

    def get_fit_params(self, X, y):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features.

        y : array-like of shape (n_samples, n_targets)
            Targets

        Returns
        -------
        fit_params : dict
            Dictionary of parameters to pass to the `fit` method of 
            `RandomziedSearchCV`.
        """
        return {}

    def make_estimator(self, **params):
        raise NotImplementedError('make_estimator method not implemented')

    @ignore_warnings(category=ConvergenceWarning)
    def fit(
            self, X, y, groups=None, n_iter=10, scoring=None, n_jobs=None, 
            cv=None
        ):
        """
        Fits a CV estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset. 
            Only used in conjunction with `GroupKFold`.

        n_iter : int, default=10
            Number of iterations to use in randomized search.

        scoring : str or callable, default=None
            A str (see model evaluation documentation) or a scorer callable 
            object / function with signature scorer(estimator, X, y) which 
            should return only a single value. If `None`, the estimator's 
            default `score` method is used.

        n_jobs : int or None, default=None
            Number of background jobs to use in randomized search.

        cv : int, cross-validation generator, or iterable, default=None
            Scikit-learn style cv parameter.
        """
        scoring = check_scoring(scoring, classifier=is_classifier(self))
        est = RandomizedSearchCV(
            self.make_estimator(),
            self.get_param_distributions(X, y),
            cv=cv,
            n_iter=n_iter,
            n_jobs=n_jobs,
            scoring=scoring
        ).fit(X, y, groups=groups, **self.get_fit_params(X, y))
        self.cv_results_ = list(zip(
            est.cv_results_['mean_test_score'], est.cv_results_['params']
        ))
        self.best_estimator_ = est.best_estimator_
        self.best_score_ = est.best_score_
        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples, n_targets)
            Predicted values
        """
        assert hasattr(self, 'best_estimator_'), (
            'Estimator has not been fitted. Call `fit` before `predict`'
        )
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """
        Probability estimates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Probability of the sample for each classes on the model, ordered 
            by `self.classes_`.

        Notes
        -----
        Only applicable for classifiers.
        """
        assert hasattr(self, 'best_estimator_'), (
            'Estimator has not been fitted. Call `fit` before `predict`'
        )
        return self.best_estimator_.predict_proba(X)


class RandomForestClassifierCV(ClassifierMixin, CVBaseEstimator):    
    def make_estimator(self, **params):
        return make_pipeline(
            *self.preprocessors,
            RandomForestClassifier()
        ).set_params(**params)
        
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'randomforestclassifier__n_estimators': geom(1./100)
        })

    def get_fit_params(self, X, y):
        weight = compute_sample_weight(class_weight='balanced', y=y)
        return {'randomforestclassifier__sample_weight': weight}


class PCARandomForestClassifierCV(ClassifierMixin, CVBaseEstimator):    
    def make_estimator(self, **params):
        return make_pipeline(
            *self.preprocessors,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            RandomForestClassifier()
        ).set_params(**params)
        
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components' : _get_n_pca_components_distribution(X),
            'randomforestclassifier__n_estimators': geom(1./100)
        })

    def get_fit_params(self, X, y):
        weight = compute_sample_weight(class_weight='balanced', y=y)
        return {'randomforestclassifier__sample_weight': weight}


class RandomForestRegressorCV(RegressorMixin, CVBaseEstimator):    
    def make_estimator(self, **params):
        return make_pipeline(
            *self.preprocessors,
            RandomForestRegressor()
        ).set_params(**params)
        
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'randomforestregressor__n_estimators': geom(1./100)
        })


class PCARandomForestRegressorCV(RegressorMixin, CVBaseEstimator):    
    def make_estimator(self, **params):
        return make_pipeline(
            *self.preprocessors,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            RandomForestRegressor()
        ).set_params(**params)
        
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components' : _get_n_pca_components_distribution(X),
            'randomforestregressor__n_estimators': geom(1./100)
        })


class LogisticLassoCV(ClassifierMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        return make_pipeline(
            *self.preprocessors,
            StandardScaler(),
            LogisticRegression()
        ).set_params(**params)
    
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'logisticregression__C': expon(0, 1),
            'logisticregression__penalty': ['l1'],
            'logisticregression__solver': ['saga']
        })

    def get_fit_params(self, X, y):
        weight = compute_sample_weight(class_weight='balanced', y=y)
        return {'logisticregression__sample_weight': weight}


class PCALogisticLassoCV(ClassifierMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        return make_pipeline(
            *self.preprocessors,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            StandardScaler(),
            LogisticRegression()
        ).set_params(**params)
    
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components' : _get_n_pca_components_distribution(X),
            'logisticregression__C': expon(0, 1),
            'logisticregression__penalty': ['l1'],
            'logisticregression__solver': ['saga']
        })

    def get_fit_params(self, X, y):
        weight = compute_sample_weight(class_weight='balanced', y=y)
        return {'logisticregression__sample_weight': weight}


class LassoLarsCV(RegressorMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        return make_pipeline(
            *self.preprocessors,
            LassoLars(normalize=True)
        ).set_params(**params)
        
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'lassolars__alpha': expon(0, 1)
        })
    
    
class PCALassoLarsCV(RegressorMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        return make_pipeline(
            *self.preprocessors,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            LassoLars(normalize=True)
        ).set_params(**params)
        
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components' : _get_n_pca_components_distribution(X),
            'lassolars__alpha': expon(0, 1)
        })


class LogisticRidgeCV(ClassifierMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        return make_pipeline(
            *self.preprocessors,
            StandardScaler(),
            LogisticRegression()
        ).set_params(**params)
    
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'logisticregression__C': expon(0, 1),
            'logisticregression__penalty': ['l2'],
            'logisticregression__solver': ['lbfgs' if X.shape[0] < 1e5 else 'saga']
        })

    def get_fit_params(self, X, y):
        weight = compute_sample_weight(class_weight='balanced', y=y)
        return {'logisticregression__sample_weight': weight}


class PCALogisticRidgeCV(ClassifierMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        return make_pipeline(
            *self.preprocessors,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            StandardScaler(),
            LogisticRegression()
        ).set_params(**params)
    
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components' : _get_n_pca_components_distribution(X),
            'logisticregression__C': expon(0, 1),
            'logisticregression__penalty': ['l2'],
            'logisticregression__solver': ['lbfgs' if X.shape[0] < 1e5 else 'saga']
        })

    def get_fit_params(self, X, y):
        weight = compute_sample_weight(class_weight='balanced', y=y)
        return {'logisticregression__sample_weight': weight}


class RidgeCV(RegressorMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        return make_pipeline(
            *self.preprocessors,
            Ridge(normalize=True)
        ).set_params(**params)
        
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'ridge__alpha': expon(0, 1)
        })
    
    
class PCARidgeCV(RegressorMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        return make_pipeline(
            *self.preprocessors,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            Ridge(normalize=True)
        ).set_params(**params)
        
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components' : _get_n_pca_components_distribution(X),
            'ridge__alpha': expon(0, 1)
        })


class LogisticElasticNetCV(ClassifierMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        return make_pipeline(
            *self.preprocessors,
            StandardScaler(),
            LogisticRegression()
        ).set_params(**params)
    
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'logisticregression__C': expon(0, 1),
            'logisticregression__penalty': ['elasticnet'],
            'logisticregression__l1_ratio': uniform(0, 1),
            'logisticregression__solver': ['saga']
        })

    def get_fit_params(self, X, y):
        weight = compute_sample_weight(class_weight='balanced', y=y)
        return {'logisticregression__sample_weight': weight}


class PCALogisticElasticNetCV(ClassifierMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        return make_pipeline(
            *self.preprocessors,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            StandardScaler(),
            LogisticRegression()
        ).set_params(**params)
    
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components' : _get_n_pca_components_distribution(X),
            'logisticregression__C': expon(0, 1),
            'logisticregression__penalty': ['elasticnet'],
            'logisticregression__l1_ratio': uniform(0, 1),
            'logisticregression__solver': ['saga']
        })

    def get_fit_params(self, X, y):
        weight = compute_sample_weight(class_weight='balanced', y=y)
        return {'logisticregression__sample_weight': weight}


class ElasticNetCV(RegressorMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        return make_pipeline(
            *self.preprocessors,
            ElasticNet(normalize=True)
        ).set_params(**params)
    
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'elasticnet__alpha': expon(0, 1),
            'elasticnet__l1_ratio': uniform(0, 1)
        })
    
    
class PCAElasticNetCV(RegressorMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        return make_pipeline(
            *self.preprocessors,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            ElasticNet(normalize=True)
        ).set_params(**params)
    
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components' : _get_n_pca_components_distribution(X),
            'elasticnet__alpha': expon(0, 1),
            'elasticnet__l1_ratio': uniform(0, 1)
        })


class KernelRidgeCV(RegressorMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocessors,
            StandardScaler(),
            KernelRidge()
        )
        return est.set_params(**params)
    
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'kernelridge__alpha': expon(0, 1),
            'kernelridge__degree': geom(.5, loc=1),
            'kernelridge__kernel': ['linear', 'poly', 'rbf', 'laplacian']
        })
    
    
class PCAKernelRidgeCV(RegressorMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocessors,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            StandardScaler(),
            KernelRidge()
        )
        return est.set_params(**params)
    
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components' : _get_n_pca_components_distribution(X),
            'kernelridge__alpha': expon(0, 1),
            'kernelridge__degree': geom(.5, loc=1),
            'kernelridge__kernel': ['linear', 'poly', 'rbf', 'laplacian']
        })


class SVCCV(ClassifierMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        return make_pipeline(
            *self.preprocessors,
            StandardScaler(),
            SVC(probability=True, class_weight='balanced')
        ).set_params(**params)
        
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'svc__C': expon(0, 1),
            'svc__degree': geom(.3),
            'svc__kernel': ['linear', 'poly', 'rbf'],
        })

    def get_fit_params(self, X, y):
        weight = compute_sample_weight(class_weight='balanced', y=y)
        return {'svc__sample_weight': weight}
    
    
class PCASVCCV(RegressorMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocessors,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            StandardScaler(),
            SVC(probability=True, class_weight='balanced')
        )
        return est.set_params(**params)
        
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components' : _get_n_pca_components_distribution(X),
            'svc__C': expon(0, 1),
            'svc__degree': geom(.3),
            'svc__kernel': ['linear', 'poly', 'rbf'],
        })

    def get_fit_params(self, X, y):
        weight = compute_sample_weight(class_weight='balanced', y=y)
        return {'svc__sample_weight': weight}


class SVRCV(RegressorMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocessors,
            StandardScaler(),
            SVR()
        )
        return est.set_params(**params)
        
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'svr__C': expon(0, 1),
            'svr__degree': geom(.3),
            'svr__kernel': ['linear', 'poly', 'rbf'],
        })
    
    
class PCASVRCV(RegressorMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocessors,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            StandardScaler(),
            SVR()
        )
        return est.set_params(**params)
        
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components' : _get_n_pca_components_distribution(X),
            'svr__C': expon(0, 1),
            'svr__degree': geom(.3),
            'svr__kernel': ['linear', 'poly', 'rbf'],
        })

    
class KNeighborsClassifierCV(ClassifierMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocessors,
            StandardScaler(),
            KNeighborsClassifier()
        )
        return est.set_params(**params)
        
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'kneighborsclassifier__n_neighbors': geom(1/(.05*X.shape[0])),
            'kneighborsclassifier__weights': ['uniform', 'distance']
        })


class PCAKNeighborsClassifierCV(ClassifierMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocessors,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            StandardScaler(),
            KNeighborsClassifier()
        )
        return est.set_params(**params)
        
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components' : _get_n_pca_components_distribution(X),
            'kneighborsclassifier__n_neighbors': geom(1/(.05*X.shape[0])),
            'kneighborsclassifier__weights': ['uniform', 'distance']
        })


class KNeighborsRegressorCV(RegressorMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocessors,
            StandardScaler(),
            KNeighborsRegressor()
        )
        return est.set_params(**params)
        
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'kneighborsregressor__n_neighbors': geom(1/(.05*X.shape[0])),
            'kneighborsregressor__weights': ['uniform', 'distance']
        })


class PCAKNeighborsRegressorCV(RegressorMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocessors,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            StandardScaler(),
            KNeighborsRegressor()
        )
        return est.set_params(**params)
        
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components' : _get_n_pca_components_distribution(X),
            'kneighborsregressor__n_neighbors': geom(1/(.05*X.shape[0])),
            'kneighborsregressor__weights': ['uniform', 'distance']
        })
    

class AdaBoostClassifierCV(ClassifierMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocessors,
            AdaBoostClassifier()
        )
        return est.set_params(**params)
        
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'adaboostclassifier__n_estimators': geom(1./2**5)
        })

    def get_fit_params(self, X, y):
        weight = compute_sample_weight(class_weight='balanced', y=y)
        return {'adaboostclassifier__sample_weight': weight}


class PCAAdaBoostClassifierCV(ClassifierMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocessors,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            AdaBoostClassifier()
        )
        return est.set_params(**params)
        
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components' : _get_n_pca_components_distribution(X),
            'adaboostclassifier__n_estimators': geom(1./2**5)
        })

    def get_fit_params(self, X, y):
        weight = compute_sample_weight(class_weight='balanced', y=y)
        return {'adaboostclassifier__sample_weight': weight}


class AdaBoostRegressorCV(RegressorMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocessors,
            AdaBoostRegressor()
        )
        return est.set_params(**params)
        
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'adaboostregressor__n_estimators': geom(1./2**5)
        })


class PCAAdaBoostRegressorCV(RegressorMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocessors,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            AdaBoostRegressor()
        )
        return est.set_params(**params)
        
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components' : _get_n_pca_components_distribution(X),
            'adaboostregressor__n_estimators': geom(1./2**5)
        })


class XGBClassifierCV(ClassifierMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocessors,
            XGBClassifier()
        )
        return est.set_params(**params)
    
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'xgbclassifier__gamma': expon(0, 1),
            'xgbclassifier__max_depth': geom(1./6),
            'xgbclassifier__min_child_weight': expon(0, 1),
            'xgbclassifier__max_delta_step': expon(0, 1),
            'xgbclassifier__lambda': expon(0, 1),
            'xgbclassifier__alpha': expon(0, 1),
        })

    def get_fit_params(self, X, y):
        weight = compute_sample_weight(class_weight='balanced', y=y)
        return {'xgbclassifier__sample_weight': weight}


class PCAXGBClassifierCV(ClassifierMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocessors,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            XGBClassifier()
        )
        return est.set_params(**params)
    
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components' : _get_n_pca_components_distribution(X),
            'xgbclassifier__gamma': expon(0, 1),
            'xgbclassifier__max_depth': geom(1./6),
            'xgbclassifier__min_child_weight': expon(0, 1),
            'xgbclassifier__max_delta_step': expon(0, 1),
            'xgbclassifier__lambda': expon(0, 1),
            'xgbclassifier__alpha': expon(0, 1),
        })

    def get_fit_params(self, X, y):
        weight = compute_sample_weight(class_weight='balanced', y=y)
        return {'xgbclassifier__sample_weight': weight}


class XGBRegressorCV(RegressorMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocessors,
            XGBRegressor()
        )
        return est.set_params(**params)
    
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'xgbregressor__gamma': expon(0, 1),
            'xgbregressor__max_depth': geom(1./6),
            'xgbregressor__min_child_weight': expon(0, 1),
            'xgbregressor__max_delta_step': expon(0, 1),
            'xgbregressor__lambda': expon(0, 1),
            'xgbregressor__alpha': expon(0, 1),
        })


class PCAXGBRegressorCV(RegressorMixin, CVBaseEstimator):
    def make_estimator(self, **params):
        est = make_pipeline(
            *self.preprocessors,
            PolynomialFeatures(),
            StandardScaler(),
            PCA(),
            XGBRegressor()
        )
        return est.set_params(**params)
    
    def get_param_distributions(self, X, y):
        return super().get_param_distributions({
            'polynomialfeatures__degree': [1, 2],
            'pca__n_components' : _get_n_pca_components_distribution(X),
            'xgbregressor__gamma': expon(0, 1),
            'xgbregressor__max_depth': geom(1./6),
            'xgbregressor__min_child_weight': expon(0, 1),
            'xgbregressor__max_delta_step': expon(0, 1),
            'xgbregressor__lambda': expon(0, 1),
            'xgbregressor__alpha': expon(0, 1),
        })