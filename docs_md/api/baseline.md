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
</style># Baseline classifier and regressor

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        
    </tbody>
</table>



##fast_automl.baseline.**BaselineClassifier**



Predicts the most frequent class.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Attributes:</b></td>
    <td class="field-body" width="100%"><b>classes_ : <i>array-like of shape (n_classes,)</i></b>
<p class="attr">
    A list of class weights known to the classifier.
</p>
<b>counts_ : <i>array-like of shape (n_classes,)</i></b>
<p class="attr">
    Normalized frequency of each class in the training data.
</p>
<b>dominant_class_ : <i>int</i></b>
<p class="attr">
    Class which appears most frequently in the training data.
</p></td>
</tr>
    </tbody>
</table>

####Examples

```python
from fast_automl.baseline import BaselineClassifier

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
clf = BaselineClassifier().fit(X_train, y_train)
clf.score(X_test, y_test)
```

####Methods



<p class="func-header">
    <i></i> <b>fit</b>(<i>self, X, y, sample_weight=None</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\baseline.py#L39">[source]</a>
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
    <i></i> <b>predict</b>(<i>self, X</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\baseline.py#L72">[source]</a>
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
    <i></i> <b>predict_proba</b>(<i>self, X</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\baseline.py#L88">[source]</a>
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



##fast_automl.baseline.**BaselineRegressor**



Predicts the mean target value.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Attributes:</b></td>
    <td class="field-body" width="100%"><b>y_mean_ : <i>np.array</i></b>
<p class="attr">
    Average target value.
</p></td>
</tr>
    </tbody>
</table>

####Examples

```python
from fast_automl.baseline import BaselineRegressor

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
reg = BaselineRegressor().fit(X_train, y_train)
reg.score(X_test, y_test)
```

####Methods



<p class="func-header">
    <i></i> <b>fit</b>(<i>self, X, y, sample_weight=None</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\baseline.py#L128">[source]</a>
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
    <i></i> <b>predict</b>(<i>self, X</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\baseline.py#L149">[source]</a>
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

