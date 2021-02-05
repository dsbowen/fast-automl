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
</style># Model comparison tests

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        
    </tbody>
</table>



##fast_automl.test.**corrected_repeated_kfold_cv_test**

<p class="func-header">
    <i>def</i> fast_automl.test.<b>corrected_repeated_kfold_cv_test</b>(<i>estimators, X, y, repetitions=10, cv= 10, scoring=None, n_jobs=None</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\test.py#L68">[source]</a>
</p>

Performs pairwise corrected repeated k-fold cross-validation tests. See [Bouckaert and Frank](https://www.cs.waikato.ac.nz/~eibe/pubs/bouckaert_and_frank.pdf).

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>estimators : <i>list</i></b>
<p class="attr">
    List of (name, estimator) tuples.
</p>
<b>X : <i>array-like of shape (n_samples, n_features)</i></b>
<p class="attr">
    Features.
</p>
<b>y : <i>array-like of shape (n_samples, n_targets)</i></b>
<p class="attr">
    Targets.
</p>
<b>repetitions : <i>int, default=10</i></b>
<p class="attr">
    Number of cross-validation repetitions.
</p>
<b>cv : <i>int, cross-validation generator, or iterable, default=10</i></b>
<p class="attr">
    Scikit-learn style cv parameter.
</p>
<b>scoring : <i>str, callable, list, tuple, or dict, default=None</i></b>
<p class="attr">
    Scikit-learn style scoring parameter.
</p>
<b>n_jobs : <i>int, default=None</i></b>
<p class="attr">
    Number of jobs to run in parallel.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>results_df : <i>pd.DataFrame</i></b>
<p class="attr">
    
</p></td>
</tr>
    </tbody>
</table>

####Examples

```python
from fast_automl.test import corrected_repeated_kfold_cv_test

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

X, y = load_boston(return_X_y=True)
corrected_repeated_kfold_cv_test(
    [
        ('rf', RandomForestRegressor()),
        ('ridge', Ridge()),
        ('svm', SVR())
    ],
    X, y, n_jobs=-1
)
```

Out:

```
Estimator1 Estimator2  PerformanceDifference       Std     t-stat       p-value
        rf      ridge               0.165030  0.030266   5.452600  3.652601e-07
        rf        svm               0.670975  0.045753  14.665154  1.460994e-26
     ridge        svm               0.505945  0.045031  11.235469  2.258586e-19
```

##fast_automl.test.**r_by_k_cv_test**

<p class="func-header">
    <i>def</i> fast_automl.test.<b>r_by_k_cv_test</b>(<i>estimators, X, y, repetitions=5, cv=2, scoring=None, n_jobs=None</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\test.py#L148">[source]</a>
</p>

Performs pariwise RxK (usually 5x2) cross-validation tests. See [here](https://www.kaggle.com/ogrellier/parameter-tuning-5-x-2-fold-cv-statistical-test).

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>estimators : <i>list</i></b>
<p class="attr">
    List of (name, estimator) tuples.
</p>
<b>X : <i>array-like of shape (n_samples, n_features)</i></b>
<p class="attr">
    Features.
</p>
<b>y : <i>array-like of shape (n_samples, n_targets)</i></b>
<p class="attr">
    Targets.
</p>
<b>repetitions : <i>int, default=10</i></b>
<p class="attr">
    Number of cross-validation repetitions.
</p>
<b>cv : <i>int, cross-validation generator, or iterable, default=10</i></b>
<p class="attr">
    Scikit-learn style cv parameter.
</p>
<b>scoring : <i>str, callable, list, tuple, or dict, default=None</i></b>
<p class="attr">
    Scikit-learn style scoring parameter.
</p>
<b>n_jobs : <i>int, default=None</i></b>
<p class="attr">
    Number of jobs to run in parallel.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>results_df : <i>pd.DataFrame</i></b>
<p class="attr">
    
</p></td>
</tr>
    </tbody>
</table>

####Examples

```python
from fast_automl.test import r_by_k_cv_test

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

X, y = load_boston(return_X_y=True)
r_by_k_cv_test(
    [
        ('rf', RandomForestRegressor()),
        ('ridge', Ridge()),
        ('svm', SVR())
    ],
    X, y, n_jobs=-1
)
```

Out:

```
Estimator1 Estimator2  PerformanceDifference       Std     t-stat   p-value
        rf      ridge               0.143314  0.026026   5.506631  0.002701
        rf        svm               0.659547  0.035824  18.410644  0.000009
     ridge        svm               0.516233  0.021601  23.898480  0.000002
```