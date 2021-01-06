import gshap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import cross_val_predict, cross_val_score

from itertools import combinations

def cross_val_plot(estimators, X, y, cross_val_score=cross_val_score):
    """
    Parameters
    ----------
    estimators : list of (estimator name, estimator) tuples
        Estimator is the estimator passed to `cross_val_score`, which usually
        expects an sklearn-like estimator.

    X : pd.DataFrame or np.array

    y : pd.Series or np.array

    cross_val_score : callable, default=sklearn.model_selection.cross_val_score
        Computes the cross validation score.

    Returns
    -------
    cross_val_dataframe, plot : pd.DataFrame, seaborn plot
    """
    mean_xval_score = lambda reg: cross_val_score(reg, X, y).mean()
    df = pd.DataFrame({
        'Model': [name for name, _ in estimators],
        'Score': [mean_xval_score(reg) for _, reg in estimators]
    })
    ax = sns.barplot(x='Model', y='Score', data=df)
    return df, ax

def error_plot(estimators, X, y, error, cov_kwargs={'cov_type': 'HC1'}):
    """
    Parameters
    ----------
    estimators : list of (estimator name, estimator) tuples
        Estimator is the estimator passed to `cross_val_score`, which usually
        expects an sklearn-like estimator.

    X : pd.DataFrame or np.array

    y : pd.Series or np.array

    error : callable
        Takes an estimator and outputs a `y.shape` np.array of errors.

    cov_kwargs : dict, default={'cov_type': 'HC1'}
        Arguments for computing the covariance when comparing models. See the 
        <a href="https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html">statsmodels API</a>.

    Returns
    -------
    error dataframe, comparison dataframe, plot : pd.DataFrame, pd.DataFrame, seaborn plot
    """
    def compare(est0, est1):
        y = est0[1] - est1[1]
        model = sm.OLS(y, np.ones(len(y)))
        res = model.fit().get_robustcov_results(**cov_kwargs)
        return {
            'Model 0': est0[0],
            'Model 1': est1[0],
            'beta': res.params[0],
            'beta_se': res.bse[0],
            't-value': res.tvalues[0],
            'p-value': res.pvalues[0]
        }

    N = len(X)
    error_vectors = [(name, error(est)) for name, est in estimators]
    df = pd.DataFrame({
        'Model': [i for name, _ in estimators for i in N*[name]],
        'Error': [err for _, vector in error_vectors for err in list(vector)]
    })
    ax = sns.boxplot(x='Model', y='Error', data=df, showfliers=False)
    comparison_records = [
        compare(*vectors) for vectors in combinations(error_vectors, r=2)
    ]
    return df, pd.DataFrame(comparison_records), ax

def group_features(df, groups, axis=0):
    """
    Group G-SHAP values by feature.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe comtaning G-SHAP values by features.

    groups : list of (group name, list of group members) tuples
    
    axis : 0 or 1, default=0
        If axis is 0, the columns of the dataframe are the features and the
        rows are the feature importances for each ovservation. If axis is 1,
        the columns of the dataframe are importances and the rows are the
        features.

    Returns
    -------
    df : pd.Dataframe
    """
    assert axis in (0, 1), 'Axis must be 0 or 1'
    
    def create_group(df, group_name, features):
        if axis == 0:
            df[group_name] = df[features].sum(axis=1)
            return df.drop(columns=features)
        # axis == 1
        cols = [col for col in df.columns if col != 'Feature']
        row = {'Feature': group_name}
        sum_df = df[df.Feature.isin(features)].sum()
        for col in cols:
            row[col] = sum_df[col]
        df = df.append(row, ignore_index=True)
        return df[~df.Feature.isin(features)]
    
    for group_name, features in groups:
        df = create_group(df, group_name, features)
    return df

def explain_performance(model, X, y, metric, X_background=None, nsamples=100, groups=[]):
    """
    Parameters
    ----------
    model : callable
        Takes `X` and outputs a np.array of predictions of `y`.

    X : pd.DataFrame or np.array

    y : pd.Series or np.array

    metric : callable
        Performance metric. Takes in `y` and the output of `model(X)` and
        returns a float-like measure of performance.

    X_background : pd.DataFrame or np.array, default=None
        Background data used to compute the G-SHAP values. If `None`, `X` is
        used.

    nsamples : int, default=100
        Number of samples used to approximate G-SHAP values.

    groups : list, default=[]
        Groups feature contributions to model performance. See 
        `group_features`.

    Returns
    -------
    performance dataframe, plot : pd.DataFrame, seaborn plot
        Contributions to model performance organized by feature and
        corresponding plot.
    """
    g = lambda output: metric(y, output)
    explainer = gshap.KernelExplainer(model, X_background or X, g)
    gshap_values = explainer.gshap_values(X, nsamples=nsamples)
    gshap_values /= gshap_values.sum()
    df = pd.DataFrame({'Feature': X.columns, 'G-SHAP': gshap_values})
    df = group_features(df, groups, axis=1)
    df = df.sort_values('G-SHAP', ascending=False).reset_index(drop=True)
    ax = sns.barplot(x='G-SHAP', y='Feature', data=df)
    return df, ax

def explain_predictions(model, X, y, X_background=None, nsamples=100, groups=[]):
    """
    Parameters
    ----------
    model : callable
        Takes `X` and outputs a np.array of predictions of `y`.

    X : pd.DataFrame or np.array

    y : pd.Series or np.array

    metric : callable
        Performance metric. Takes in `y` and the output of `model(X)` and
        returns a float-like measure of performance.

    X_background : pd.DataFrame or np.array, default=None
        Background data used to compute the G-SHAP values. If `None`, `X` is
        used.

    nsamples : int, default=100
        Number of samples used to approximate G-SHAP values.

    groups : list, default=[]
        Groups feature contributions to model performance. See 
        `group_features`.

    Returns
    -------
    prediction dataframe : pd.DataFrame
        Contribution of each feature to the model's prediction by observation.
    """    
    g = lambda output: output
    explainer = gshap.KernelExplainer(model, X_background or X, g)
    gshap_values = explainer.gshap_values(X, nsamples=nsamples)
    df = pd.DataFrame(columns=X.columns, data=gshap_values.T)
    df = group_features(df, groups, axis=0)
    return df.reset_index(drop=True)

def explain_correlations(features, model, X, y, X_background=None, nsamples=100, groups=[]):
    """
    Parameters
    ----------
    features : list of str or (str, array-like) tuples
        The features whose correlation with `y` you want to explain. If the 
        list item is a string, it is assumed to be the name of a column in 
        `X`. If the list item is a (str, array-like) tuple, the string is the
        feature name and the array is the values of that feature for each
        observation in `X`.

    model : callable
        Takes `X` and outputs a np.array of predictions of `y`.

    X : pd.DataFrame or np.array

    y : pd.Series or np.array

    metric : callable
        Performance metric. Takes in `y` and the output of `model(X)` and
        returns a float-like measure of performance.

    X_background : pd.DataFrame or np.array, default=None
        Background data used to compute the G-SHAP values. If `None`, `X` is
        used.

    nsamples : int, default=100
        Number of samples used to approximate G-SHAP values.

    groups : list, default=[]
        Groups feature contributions to model performance. See 
        `group_features`.

    Returns
    -------
    correlation dataframe, plots : pd.DataFrame, list of seaborn plots
        Dataframe where rows are the explaining features (from `X`) and the
        columns are the features whose correlation with `y` you want to 
        explain. The plots correspond to the features whose correlation with
        `y` you want to explain.
    """
    def convert_features(features):
        names, vectors = [], []
        if not isinstance(features, list):
            features = [features]
        for i, feature in enumerate(features):
            name, data = feature, X[feature] if isinstance(feature, str) else feature
            names.append(name)
            vectors.append(data)
        return names, vectors
    
    def correlation_plot(df, col):
        df = df[['Feature', col]].sort_values(col)
        ax = sns.barplot(x=col, y='Feature', data=df)
        plt.show()
        return ax
    
    names, vectors = convert_features(features)
    compute_corr = lambda y: np.array([np.corrcoef(vector, y)[0][1] for vector in vectors])
    g = lambda output: compute_corr(output)
    explainer = gshap.KernelExplainer(model, X_background or X, g)
    gshap_values = explainer.gshap_values(X, nsamples=nsamples)
    df = pd.DataFrame({'Feature': X.columns})
    df[names] = gshap_values
    df = group_features(df, groups, axis=1)
    # normalize to sum to the true correlation
    df[names] *= compute_corr(y) / df[names].sum(axis=0)
    plots = [correlation_plot(df, name) for name in names]
    return df, plots