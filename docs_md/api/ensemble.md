<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<link rel="stylesheet" href="https://assets.readthedocs.org/static/css/readthedocs-doc-embed.css" type="text/css" />

<style>
    a.src-href {
        float: right;
    }
    p.attr {
        margin-top: 0.5em;
        margin-left: 1em;
    }
    p.func-header {
        background-color: gainsboro;
        border-radius: 0.1em;
        padding: 0.5em;
        padding-left: 1em;
    }
    table.field-table {
        border-radius: 0.1em
    }
</style># Ensemble

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        
    </tbody>
</table>



##fast_automl.ensemble.**ClassifierWeighter**

<p class="func-header">
    <i>class</i> fast_automl.ensemble.<b>ClassifierWeighter</b>(<i>loss=log_loss</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\ensemble.py#L29">[source]</a>
</p>

Trains weights for ensemble of classifiers.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>loss : <i>callable, default=log_loss</i></b>
<p class="attr">
    Loss function to minimize by weight fitting.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Attributes:</b></td>
    <td class="field-body" width="100%"><b>coef_ : <i>array-like of shape (n_estimators,)</i></b>
<p class="attr">
    Weights on the given estimators.
</p></td>
</tr>
    </tbody>
</table>

####Examples

```python
from fast_automl.ensemble import ClassifierWeighter

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from copy import deepcopy

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True)

svc = SVC(probability=True).fit(X_train, y_train)
knn = KNeighborsClassifier().fit(X_train, y_train)

cv = StratifiedKFold(random_state=np.random.RandomState(), shuffle=True)
X_meta = np.array([
    cross_val_predict(clf, X_train, y_train, cv=deepcopy(cv), method='predict_proba')
    for clf in (svc, knn)
]).transpose(1, 2, 0)
weighter = ClassifierWeighter().fit(X_meta, y_train)

X_meta_test = np.array([
    clf.predict_proba(X_test) for clf in (svc, knn)
]).transpose(1, 2, 0)
weighter.score(X_meta_test, y_test)
```

####Methods



<p class="func-header">
    <i></i> <b>fit</b>(<i>self, X, y</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\ensemble.py#L78">[source]</a>
</p>



<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>X : <i>array-like of shape (n_samples, n_classes, n_estimators)</i></b>
<p class="attr">
    X_ice is the probability estimator e puts on sample i being in class c.
</p>
<b>y : <i>array-like of shape (n_samples,)</i></b>
<p class="attr">
    Targets.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>self : <i></i></b>
<p class="attr">
    
</p></td>
</tr>
    </tbody>
</table>





<p class="func-header">
    <i></i> <b>predict</b>(<i>self, X</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\ensemble.py#L107">[source]</a>
</p>

Predict class labels for samples in X.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>X : <i>array-like of shape (n_samples, n_features)</i></b>
<p class="attr">
    Samples.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>C : <i>array of shape (n_samples,)</i></b>
<p class="attr">
    Predicted class label for each sample.
</p></td>
</tr>
    </tbody>
</table>





<p class="func-header">
    <i></i> <b>predict_proba</b>(<i>self, X</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\ensemble.py#L123">[source]</a>
</p>

Probability estimates.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>X : <i>array-like of shape (n_samples, n_features)</i></b>
<p class="attr">
    Samples.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>T : <i>array-like of shape (n_samples, n_classes)</i></b>
<p class="attr">
    Probability of the sample for each classes.
</p></td>
</tr>
    </tbody>
</table>



##fast_automl.ensemble.**BaseStackingCV**

<p class="func-header">
    <i>class</i> fast_automl.ensemble.<b>BaseStackingCV</b>(<i>estimators, cv=None, shuffle_cv=True, scoring=None, n_jobs=None, verbose=0</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\ensemble.py#L140">[source]</a>
</p>



<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>estimators : <i>list</i></b>
<p class="attr">
    Base estimators which will be stacked together. Each element of the list is defined as a tuple of string (i.e. name) and an estimator instance.
</p>
<b>cv : <i>int, cross-validation generator, or iterable, default=None</i></b>
<p class="attr">
    Scikit-learn style cv parameter.
</p>
<b>shuffle_cv : <i>bool, default=True</i></b>
<p class="attr">
    Indicates that cross validator should shuffle observations.
</p>
<b>scoring : <i>str, callable, list, tuple, or dict, default=None</i></b>
<p class="attr">
    Scikit-learn style scoring parameter.
</p>
<b>n_jobs : <i>int, default=None</i></b>
<p class="attr">
    Number of jobs to run in parallel.
</p>
<b>verbose : <i>int, default=0</i></b>
<p class="attr">
    Controls the verbosity.
</p></td>
</tr>
    </tbody>
</table>



####Methods



<p class="func-header">
    <i></i> <b>fit</b>(<i>self, X, y, sample_weight=None</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\ensemble.py#L172">[source]</a>
</p>



<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        
    </tbody>
</table>





<p class="func-header">
    <i></i> <b>predict</b>(<i>self, X</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\ensemble.py#L175">[source]</a>
</p>

Predict class labels for samples in X.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>X : <i>array-like of shape (n_samples, n_features)</i></b>
<p class="attr">
    Samples.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>C : <i>array of shape (n_samples,)</i></b>
<p class="attr">
    Predicted outcome for each sample.
</p></td>
</tr>
    </tbody>
</table>





<p class="func-header">
    <i></i> <b>predict_proba</b>(<i>self, X</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\ensemble.py#L194">[source]</a>
</p>

Probability estimates.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>X : <i>array-like of shape (n_samples, n_features)</i></b>
<p class="attr">
    Samples.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>T : <i>array-like of shape (n_samples, n_classes)</i></b>
<p class="attr">
    Probability of the sample for each classes on the model.
</p></td>
</tr>
    </tbody>
</table>

####Notes

Only applicable to classifiers.



<p class="func-header">
    <i></i> <b>transform</b>(<i>self, X</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\ensemble.py#L217">[source]</a>
</p>

Transforms raw features into a prediction matrix.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>X : <i>array-like of shape (n_samples, n_features)</i></b>
<p class="attr">
    Features.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>X_meta : <i>array-like</i></b>
<p class="attr">
    Prediction matrix of shape (n_samples, n_estimators) for regression and (n_estimators, n_samples, n_classes) for classification.
</p></td>
</tr>
    </tbody>
</table>



##fast_automl.ensemble.**RFEVotingEstimatorCV**



Selects estimators using recursive feature elimination. Inherits from
`BaseStackingCV`.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Attributes:</b></td>
    <td class="field-body" width="100%"><b>best_estimator_ : <i>estimator</i></b>
<p class="attr">
    The voting estimator associated with the highest CV score.
</p>
<b>best_score_ : <i>scalar</i></b>
<p class="attr">
    Highest CV score attained by any voting estimator.
</p>
<b>weights_ : <i>array-like</i></b>
<p class="attr">
    Weights the voting estimator places on each of the estimators in its ensemble.
</p>
<b>names_ : <i>list</i></b>
<p class="attr">
    List of estimator names in the best estimator.
</p></td>
</tr>
    </tbody>
</table>



####Methods



<p class="func-header">
    <i></i> <b>fit</b>(<i>self, X, y, sample_weight=None</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\ensemble.py#L306">[source]</a>
</p>

Fit the model.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>X : <i>array-like of shape (n_samples, n_features)</i></b>
<p class="attr">
    Training data.
</p>
<b>y : <i>array-like of shape (n_samples,)</i></b>
<p class="attr">
    Target values.
</p>
<b>sample_weight, array-like of shape (n_samples,), default=Noone : <i></i></b>
<p class="attr">
    Individual weights for each sample.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>self : <i></i></b>
<p class="attr">
    
</p></td>
</tr>
    </tbody>
</table>



##fast_automl.ensemble.**RFEVotingClassifierCV**



Selects classifiers using recursive feature elimination. Inherits from
`RFEVotingEstimatorCV`.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        
    </tbody>
</table>

####Examples

```python
from fast_automl.ensemble import RFEVotingClassifierCV

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True)

clf = RFEVotingClassifierCV([
    ('rf', RandomForestClassifier()),
    ('knn', KNeighborsClassifier()),
    ('svm', SVC(probability=True))
]).fit(X_train, y_train)
print('CV score: {:.4f}'.format(cross_val_score(clf.best_estimator_, X_train, y_train).mean()))
print('Test score: {:.4f}'.format(clf.score(X_test, y_test)))
```



##fast_automl.ensemble.**RFEVotingRegressorCV**



Selects regressors using recursive feature elimination. Inherits from
`RFEVotingEstimatorCV`.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        
    </tbody>
</table>

####Examples

```python
from fast_automl.ensemble import RFEVotingRegressorCV

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

reg = RFEVotingRegressorCV([
    ('rf', RandomForestRegressor()),
    ('knn', KNeighborsRegressor()),
    ('svm', SVR())
]).fit(X_train, y_train)
print('CV score: {:.4f}'.format(cross_val_score(reg.best_estimator_, X_train, y_train).mean()))
print('Test score: {:.4f}'.format(reg.score(X_test, y_test)))
```



##fast_automl.ensemble.**StepwiseVotingEstimatorCV**



Selects estimators using stepwise addition. Inherits from
`BaseStackingCV`.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Attributes:</b></td>
    <td class="field-body" width="100%"><b>best_estimator_ : <i>estimator</i></b>
<p class="attr">
    The voting estimator associated with the highest CV score.
</p>
<b>best_score_ : <i>scalar</i></b>
<p class="attr">
    Highest CV score attained by any voting estimator.
</p>
<b>weights_ : <i>array-like</i></b>
<p class="attr">
    Weights the voting estimator places on each of the estimators in its ensemble.
</p>
<b>names_ : <i>list</i></b>
<p class="attr">
    List of estimator names in the best estimator.
</p></td>
</tr>
    </tbody>
</table>



####Methods



<p class="func-header">
    <i></i> <b>fit</b>(<i>self, X, y, sample_weight=None</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\ensemble.py#L450">[source]</a>
</p>

Fit the model.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>X : <i>array-like of shape (n_samples, n_features)</i></b>
<p class="attr">
    Training data.
</p>
<b>y : <i>array-like of shape (n_samples,)</i></b>
<p class="attr">
    Target values.
</p>
<b>sample_weight, array-like of shape (n_samples,), default=Noone : <i></i></b>
<p class="attr">
    Individual weights for each sample.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>self : <i></i></b>
<p class="attr">
    
</p></td>
</tr>
    </tbody>
</table>



##fast_automl.ensemble.**StepwiseVotingClassifierCV**



Selects classifiers using stepwise addition. Inherits from
`StepwiseVotingEstimatorCV`.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        
    </tbody>
</table>

####Examples

```python
from fast_automl.ensemble import StepwiseVotingClassifierCV

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True)

clf = StepwiseVotingClassifierCV([
    ('rf', RandomForestClassifier()),
    ('knn', KNeighborsClassifier()),
    ('svm', SVC(probability=True))
]).fit(X_train, y_train)
print('CV score: {:.4f}'.format(cross_val_score(clf.best_estimator_, X_train, y_train).mean()))
print('Test score: {:.4f}'.format(clf.score(X_test, y_test)))
```



##fast_automl.ensemble.**StepwiseVotingRegressorCV**



Selects regressors using stepwise addition. Inherits from
`StepwiseVotingEstimatorCV`.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        
    </tbody>
</table>

####Examples

```python
from fast_automl.ensemble import StepwiseVotingRegressorCV

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

reg = StepwiseVotingRegressorCV([
    ('rf', RandomForestRegressor()),
    ('knn', KNeighborsRegressor()),
    ('svm', SVR())
]).fit(X_train, y_train)
print('CV score: {:.4f}'.format(cross_val_score(reg.best_estimator_, X_train, y_train).mean()))
print('Test score: {:.4f}'.format(reg.score(X_test, y_test)))
```

