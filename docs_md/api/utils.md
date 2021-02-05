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
</style># Utilities

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        
    </tbody>
</table>



##fast_automl.utils.**TransformerMixin**



Version of scikit-learn's [`TransformerMixin`](https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html) which implements a default inert `fit` method.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        
    </tbody>
</table>



####Methods



<p class="func-header">
    <i></i> <b>fit</b>(<i>self, X, y=None</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\utils.py#L15">[source]</a>
</p>

This function doesn't do anything, but is necessary to include the transformer in a `Pipeline`.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>self : <i></i></b>
<p class="attr">
    
</p></td>
</tr>
    </tbody>
</table>





<p class="func-header">
    <i></i> <b>transform</b>(<i>self, X</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\utils.py#L25">[source]</a>
</p>

Must be implemented by the transformer.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        
    </tbody>
</table>



##fast_automl.utils.**ColumnSelector**

<p class="func-header">
    <i>class</i> fast_automl.utils.<b>ColumnSelector</b>(<i>columns</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\utils.py#L32">[source]</a>
</p>

Selects columns from a dataframe.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>columns : <i>list</i></b>
<p class="attr">
    List of columns to select.
</p></td>
</tr>
    </tbody>
</table>

####Examples

```python
from fast_automl.utils import ColumnSelector

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

X = pd.DataFrame({
    'x0': [-1, -2, 1, 2],
    'x1': [-1, -1, 1, 1]
})
y = np.array([1, 1, 2, 2])

reg = make_pipeline(
    ColumnSelector(['x1']),
    LinearRegression()
).fit(X, y)
reg.score(X, y)
```

####Methods



<p class="func-header">
    <i></i> <b>transform</b>(<i>self, X, y=None</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\utils.py#L67">[source]</a>
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
<b>y : <i>optional, array-like of shape (n_samples, n_targets)</i></b>
<p class="attr">
    Target values.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>X or (X, y) : <i>Where X columns have been selected</i></b>
<p class="attr">
    
</p></td>
</tr>
    </tbody>
</table>



##fast_automl.utils.**ColumnRemover**

<p class="func-header">
    <i>class</i> fast_automl.utils.<b>ColumnRemover</b>(<i>columns</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\utils.py#L88">[source]</a>
</p>

Removes columns from a dataframe.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>columns : <i>list</i></b>
<p class="attr">
    List of columns to remove.
</p></td>
</tr>
    </tbody>
</table>

####Examples

```python
from fast_automl.utils import ColumnRemover

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

X = pd.DataFrame({
    'x0': [-1, -2, 1, 2],
    'x1': [-1, -1, 1, 1]
})
y = np.array([1, 1, 2, 2])

reg = make_pipeline(
    ColumnRemover(['x0']),
    LinearRegression()
).fit(X, y)
reg.score(X, y)
```

####Methods



<p class="func-header">
    <i></i> <b>transform</b>(<i>self, X, y=None</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\utils.py#L123">[source]</a>
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
<b>y : <i>optional, array-like of shape (n_samples, n_targets)</i></b>
<p class="attr">
    Target values.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>X or (X, y) : <i>Where X columns have been removed</i></b>
<p class="attr">
    
</p></td>
</tr>
    </tbody>
</table>



##fast_automl.utils.**BoundRegressor**

<p class="func-header">
    <i>class</i> fast_automl.utils.<b>BoundRegressor</b>(<i>estimator</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\utils.py#L144">[source]</a>
</p>

Constrains the predicted target value to be within the range of targets in
the training data.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>estimator : <i>scikit-learn style regressor</i></b>
<p class="attr">
    
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Attributes:</b></td>
    <td class="field-body" width="100%"><b>estimator_ : <i>scikit-learn style regressor</i></b>
<p class="attr">
    Fitted regressor.
</p>
<b>y_max_ : <i>scalar</i></b>
<p class="attr">
    Maximum target value in training data.
</p>
<b>y_min_ : <i>scalar</i></b>
<p class="attr">
    Minimum target value in training data.
</p></td>
</tr>
    </tbody>
</table>

####Examples

```python
from fast_automl.utils import BoundRegressor

import numpy as np
from sklearn.linear_model import LinearRegression

X_train = np.array([
    [1, 2],
    [7, 8]
])
X_test = np.array([
    [3, 4],
    [5, 1000]
])
y_train = np.array([1.5, 7.5])
y_test = np.array([3.5, 5.5])

reg = LinearRegression().fit(X_train, y_train)
reg.predict(X_test)
```

Out:

```
array([3.5, 7.5])
```

####Methods



<p class="func-header">
    <i></i> <b>fit</b>(<i>self, X, y, sample_weight=None</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\utils.py#L196">[source]</a>
</p>



<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        
    </tbody>
</table>





<p class="func-header">
    <i></i> <b>predict</b>(<i>self, X</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\utils.py#L201">[source]</a>
</p>



<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        
    </tbody>
</table>

