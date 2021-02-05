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
</style># Automated machine learning

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        
    </tbody>
</table>



##fast_automl.automl.**make_cv_regressors**

<p class="func-header">
    <i>def</i> fast_automl.automl.<b>make_cv_regressors</b>(<i></i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\automl.py#L18">[source]</a>
</p>



<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>cv_regressors : <i>list</i></b>
<p class="attr">
    List of default CV regresssors.
</p></td>
</tr>
    </tbody>
</table>



##fast_automl.automl.**make_cv_classifiers**

<p class="func-header">
    <i>def</i> fast_automl.automl.<b>make_cv_classifiers</b>(<i></i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\automl.py#L47">[source]</a>
</p>



<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>cv_classifiers : <i>list</i></b>
<p class="attr">
    List of default CV classifiers.
</p></td>
</tr>
    </tbody>
</table>



##fast_automl.automl.**AutoEstimator**

<p class="func-header">
    <i>class</i> fast_automl.automl.<b>AutoEstimator</b>(<i>cv_estimators=[], preprocessors=[], ensemble_method= 'auto', max_ensemble_size=50, n_ensembles=1, n_iter=10, n_jobs=None, verbose=False, cv=None, scoring=None</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\automl.py#L73">[source]</a>
</p>



<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>cv_estimators : <i>list of CVEstimators, default=[]</i></b>
<p class="attr">
    If an empty list, a default list of CVEstimators will be created.
</p>
<b>preprocessors : <i>list, default=[]</i></b>
<p class="attr">
    List of preprocessing steps before data is fed to the <code>cv_estimators</code>.
</p>
<b>ensemble_method : <i>str, default='auto'</i></b>
<p class="attr">
    If <code>'rfe'</code>, the ensemble is created using recursive feature elimination. If <code>'stepwise'</code>, the ensemble is created using stepwise addition. If <code>'auto'</code>, the ensemble is the better of the RFE and stepwise ensemble methods.
</p>
<b>max_ensemble_size : <i>int, default=50</i></b>
<p class="attr">
    The maximum number of estimators to consider adding to the ensemble.
</p>
<b>n_ensembles : <i>int, default=1</i></b>
<p class="attr">
    Number of ensembles to create using different CV splits. These ensembles get equal votes in a meta-ensemble.
</p>
<b>n_iter : <i>int, default=10</i></b>
<p class="attr">
    Number of iterations to run randomized search for the CVEstimators.
</p>
<b>n_jobs : <i>int or None, default=None</i></b>
<p class="attr">
    Number of jobs to run in parallel.
</p>
<b>verbose : <i>bool, default=False</i></b>
<p class="attr">
    Controls the verbosity.
</p>
<b>cv : <i>int, cross-validation generator, or iterable, default=None</i></b>
<p class="attr">
    Scikit-learn style cv parameter.
</p>
<b>scoring : <i>str, callable, list, tuple, or dict, default=None</i></b>
<p class="attr">
    Scikit-learn style scoring parameter. By default, a regressor ensembles maximizes R-squared and a classifier ensemble maximizes ROC AUC.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Attributes:</b></td>
    <td class="field-body" width="100%"><b>best_estimator_ : <i>estimator</i></b>
<p class="attr">
    Ensemble or meta-ensemble associated with the best CV score.
</p></td>
</tr>
    </tbody>
</table>



####Methods



<p class="func-header">
    <i></i> <b>fit</b>(<i>self, X, y, sample_weight=None</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\automl.py#L144">[source]</a>
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





<p class="func-header">
    <i></i> <b>predict</b>(<i>self, X</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\automl.py#L252">[source]</a>
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
    <i></i> <b>predict_proba</b>(<i>self, X</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\automl.py#L268">[source]</a>
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
    Probability of the sample for each classes on the model, ordered by <code>self.classes_</code>.
</p></td>
</tr>
    </tbody>
</table>



##fast_automl.automl.**AutoClassifier**



Automatic classifier. Inherits from `AutoEstimator`.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        
    </tbody>
</table>

####Examples

```python
from fast_automl.automl import AutoClassifier

from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score, train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y)

clf = AutoClassifier(ensemble_method='stepwise', n_jobs=-1, verbose=True).fit(X_train, y_train)
print('CV score: {:.4f}'.format(cross_val_score(clf.best_estimator_, X_train, y_train).mean()))
print('Test score: {:.4f}'.format(clf.score(X_test, y_test)))
```

This runs for about 6-7 minutes and typically achieves a test accuracy of
96-99% and ROC AUC above .999.



##fast_automl.automl.**AutoRegressor**



Automatic regressor. Inherits from `AutoEstimator`.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        
    </tbody>
</table>

####Examples

```python
from fast_automl.automl import AutoRegressor

from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score, train_test_split

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

reg = AutoRegressor(n_jobs=-1, verbose=True).fit(X_train, y_train)
print('CV score: {:.4f}'.format(cross_val_score(reg.best_estimator_, X_train, y_train).mean()))
print('Test score: {:.4f}'.format(reg.score(X_test, y_test)))
```

This runs for about 30 seconds and typically achieves a test R-squared of
.47-.53.

