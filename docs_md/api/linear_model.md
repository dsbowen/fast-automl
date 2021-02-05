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
</style># Linear models

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        
    </tbody>
</table>



##fast_automl.linear_model.**ConstrainedLinearRegression**

<p class="func-header">
    <i>class</i> fast_automl.linear_model.<b>ConstrainedLinearRegression</b>(<i>constraint=0, copy_X=True, n_jobs=None</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\linear_model.py#L7">[source]</a>
</p>

Linear regression where the coefficients are constrained to sum to a given
value.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>constraint : <i>scalar, default=0</i></b>
<p class="attr">
    Sum of the regression coefficients.
</p>
<b>normalize : <i>bool, default=False</i></b>
<p class="attr">
    
</p>
<b>copy_Xbool, default=True : <i></i></b>
<p class="attr">
    If True, X will be copied; else, it may be overwritten.
</p>
<b>n_jobs : <i>int, default=None</i></b>
<p class="attr">
    The number of jobs to use for the computation. This will only provide speedup for n_targets &gt; 1 and sufficient large problems. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Attributes:</b></td>
    <td class="field-body" width="100%"><b>coef_ : <i>array-like of shape (n_features,) or (n_targets, n_features)</i></b>
<p class="attr">
    Estimated coefficients for the linear regression contrained to sum to the given constraint value.
</p></td>
</tr>
    </tbody>
</table>

####Examples

```python
from fast_automl.linear_model import ConstrainedLinearRegression

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
reg = ConstrainedLinearRegression(constraint=.8).fit(X_train, y_train)
print(reg.score(X_test, y_test))
print(reg.coef_.sum())
```

Out:

```
0.6877629260102918
0.8
```

####Methods



<p class="func-header">
    <i></i> <b>fit</b>(<i>self, X, y, sample_weight=None</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\linear_model.py#L60">[source]</a>
</p>



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
<b>y : <i>array-like of shape (n_samples, n_targets)</i></b>
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
    <i></i> <b>predict</b>(<i>self, X</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\linear_model.py#L91">[source]</a>
</p>



<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>X : <i>array-like, shape (n_samples, n_features)</i></b>
<p class="attr">
    Samples.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>C : <i>array, shape (n_samples, n_targets)</i></b>
<p class="attr">
    Predicted values
</p></td>
</tr>
    </tbody>
</table>



##fast_automl.linear_model.**Ridge**

<p class="func-header">
    <i>class</i> fast_automl.linear_model.<b>Ridge</b>(<i>alpha=1.0, prior_weight=0, normalize_coef=False, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol= 0.001, solver='auto', random_state=None</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\linear_model.py#L106">[source]</a>
</p>

Ridge regression with the option for custom prior weights on coefficients.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>prior_weight : <i>array-like of shape (n_features)</i></b>
<p class="attr">
    Prior weight means.
</p>
<b>See [scikit-learn's ridge regression documentation](https : <i>//scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) for additional parameter details.</i></b>
<p class="attr">
    
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Attributes:</b></td>
    <td class="field-body" width="100%"><b>coef_ndarray of shape (n_features,) or (n_targets, n_features) : <i></i></b>
<p class="attr">
    Weight vector(s).
</p>
<b>intercept_float or ndarray of shape (n_targets,) : <i></i></b>
<p class="attr">
    Independent term in decision function. Set to 0.0 if <code>fit_intercept = False</code>.
</p></td>
</tr>
    </tbody>
</table>

####Examples

```python
from fast_automl.linear_model import Ridge

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
reg = Ridge().fit(X_train, y_train)
reg.score(X_test, y_test)
```

####Methods



<p class="func-header">
    <i></i> <b>fit</b>(<i>self, X, y, sample_weight=None</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\linear_model.py#L152">[source]</a>
</p>



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
<b>y : <i>array-like of shape (n_samples, n_targets)</i></b>
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
    <i></i> <b>predict</b>(<i>self, X</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\linear_model.py#L175">[source]</a>
</p>



<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>X : <i>array-like, shape (n_samples, n_features)</i></b>
<p class="attr">
    Samples.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>C : <i>array, shape (n_samples, n_targets)</i></b>
<p class="attr">
    Predicted values
</p></td>
</tr>
    </tbody>
</table>

