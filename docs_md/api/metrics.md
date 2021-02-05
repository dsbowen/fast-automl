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
</style># Metrics

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        
    </tbody>
</table>



##fast_automl.metrics.**check_scoring**

<p class="func-header">
    <i>def</i> fast_automl.metrics.<b>check_scoring</b>(<i>scoring, classifier</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\metrics.py#L7">[source]</a>
</p>

Creates a default regression or classifier scoring rule. This is R-squared for regressors and ROC AUC for classifiers.

<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>scoring : <i>str or callable</i></b>
<p class="attr">
    A str (see scikit-learn's model evaluation docs) or a scorer callable with signature <code>scorer(estimator, X, y)</code> which returns a single value.
</p>
<b>classifier : <i>bool</i></b>
<p class="attr">
    Indicates that the estimator is a classifier.
</p></td>
</tr>
    </tbody>
</table>



##fast_automl.metrics.**roc_auc_score**

<p class="func-header">
    <i>def</i> fast_automl.metrics.<b>roc_auc_score</b>(<i>y, output</i>) <a class="src-href" target="_blank" href="https://github.com/dsbowen/fast-automl/blob/master/fast_automl\metrics.py#L23">[source]</a>
</p>



<table class="docutils field-list field-table" frame="void" rules="none">
    <col class="field-name" />
    <col class="field-body" />
    <tbody valign="top">
        <tr class="field">
    <th class="field-name"><b>Parameters:</b></td>
    <td class="field-body" width="100%"><b>y : <i>array-like of shape (n_samples,)</i></b>
<p class="attr">
    target values.
</p>
<b>output : <i>array-like of shape (n_samples, n_classes)</i></b>
<p class="attr">
    Predicted probability of each class.
</p></td>
</tr>
<tr class="field">
    <th class="field-name"><b>Returns:</b></td>
    <td class="field-body" width="100%"><b>score : <i>scalar</i></b>
<p class="attr">
    ROC AUC score, default is one-versus-rest for multi-class problems.
</p></td>
</tr>
    </tbody>
</table>

