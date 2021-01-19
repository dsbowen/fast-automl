"""Utilities for hypothesis tests"""

import numpy as np
import pandas as pd
import seaborn as sns
from dask import delayed
from scipy.stats import ttest_1samp
from sklearn.base import clone
from sklearn.model_selection import KFold

import itertools

def _split(X, y, train_idx, test_idx):
    # returns train test split
    if isinstance(X, pd.DataFrame):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    else:
        X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test

def cv_test(estimators, X, y, scorer, repeat=10, cv=10, parallel=True):
    """
    Prepares a dataframe for a cross validation test.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        List of (estimator name, estimator) tuples. The estimator must
        implement `fit` and `predict` methods.

    X : pandas.DataFrame or numpy.array
        Features.

    y : pandas.DataFrame, pandas.Series, or numpy.array
        Targets.

    scorer : callable
        Takes the true and predicted target values and returns a score.

    repeat : int, default=10
        Number of repetitions.

    cv : int, default=10
        Number of folds to cross-validate. TODO: sklearn-style cv parameter.

    parallel: bool, default=True
        Run the CV test using parallel processing. Set to `False` to force a
        single process.

    Returns
    -------
    scores, pairwise_tests : pd.DataFrame, pd.DataFrame
        `scores` is the average cross-validation score for each repetition 
        organized by estimator. `pairwise_tests` is a dataframe of pairwise
        t-tests for each estimator.
    """
    def compute_cv_score(kf):
        # compute CV score for a given split
        scores = {name: [] for name, _ in estimators}
        for train_idx, test_idx in kf.split(X):
            compute_fold_score(scores, train_idx, test_idx)
        return {
            name: np.array(fold_scores).mean() 
            for name, fold_scores in scores.items()
        }
    
    def compute_fold_score(scores, train_idx, test_idx):
        # compute score for a given fold
        f = delayed(compute_estimator_score) if parallel else compute_estimator_score
        for i, (name, _) in enumerate(estimators):
            scores[name].append(f(i, train_idx, test_idx))
    
    def compute_estimator_score(estimator_idx, train_idx, test_idx):
        # compute score for a given estimator and fold
        est = estimators[estimator_idx][1]
        if parallel:
            est = clone(est)
        X_train, X_test, y_train, y_test = _split(X, y, train_idx, test_idx)
        est.fit(X_train, y_train)
        return scorer(y_test, est.predict(X_test))
        
    scores = [compute_cv_score(KFold(cv, shuffle=True)) for _ in range(repeat)]
    score_df = (
        delayed(pd.DataFrame)(scores).compute() if parallel
        else pd.DataFrame(scores)
    )
    return score_df, run_pairwise_tests(score_df)

def run_pairwise_tests(df):
    """
    Run significance tests for pairwise estimator comparisons.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns represent different models, rows represent the average CV
        score for a given CV repetition.

    Returns
    -------
    result_df : pandas.DataFrame
    """
    def run_test(pair):
        delta = df[pair[1]] - df[pair[0]]
        res = ttest_1samp(delta, 0)
        return {
            'Model1': pair[0],
            'Model2': pair[1],
            'MeanDifference': delta.mean(),
            't-stat': res.statistic,
            'p-value': res.pvalue
        }
    
    return pd.DataFrame([
        run_test(pair) 
        for pair in itertools.combinations(df.columns, 2)
    ])

def moderation_test(estimator, X, y, repeat=10, cv=10, parallel=True):
    """
    Runs a test for heterogeneous treatment effects.

    Parameters
    ----------
    estimator : TreatmentRegressor
        Must have `fit` and `score` methods. The `score` method must take a
        parameter `return_constrained` and return a tuple of (score for the 
        unconstrained model, score for the constrained model).

    X : pandas.DataFrame or numpy.array
        Features.

    y : pandas.DataFrame or pandas.Series or numpy.array
        Targets.

    repeat : int, default=10
        Number of repetitions.

    cv : int, default=10
        Number of folds to cross-validate. TODO: sklearn-style cv parameter.

    parallel: bool, default=True
        Run the CV test using parallel processing. Set to `False` to force a
        single process.

    Returns
    -------
    scores, t-stat, p-value : pandas.DataFrame, scalar, scalar
    """
    def compute_cv_score(i, kf):
        f = delayed(compute_fold_score) if parallel else compute_fold_score
        return [
            f(i, train_idx, test_idx) for train_idx, test_idx in kf.split(X)
        ]
        
    def compute_fold_score(i, train_idx, test_idx):
        # compute the score for a given fold
        est = clone(estimator) if parallel else estimator
        X_train, X_test, y_train, y_test = _split(X, y, train_idx, test_idx)
        est.fit(X_train, y_train)
        score_uc, score_c = est.score(X_test, y_test, return_constrained=True)
        return dict(Repetition=i, Constrained=score_c, Unconstrained=score_uc)
    
    def run_test():
        # run a t-test
        res = ttest_1samp(df['Unconstrained'] - df['Constrained'], 0)
        # convert p-value for 1-tailed test
        # hypothesis is that unconstrained is better
        pvalue = (res.pvalue/2) if res.statistic>0 else (1-res.pvalue/2)
        return res.statistic, pvalue
    
    scores = []
    for i in range(repeat):
        kf = KFold(cv, shuffle=True)
        scores.extend(compute_cv_score(i, kf))
    df = (
        delayed(pd.DataFrame)(scores).compute() if parallel
        else pd.DataFrame(scores)
    )
    df = df.groupby('Repetition').mean().reset_index(drop=True)
    return df, *run_test()

def gen_score_plot(df, **boxplot_kwargs):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        Columns represent different models, rows represent the average CV
        score for a given CV repetition. This can be the output of `cv_test`.

    **boxplot_kwargs :
        Keyword arguments for `sns.boxplot`.

    Returns
    -------
    result_plot : seaborn plot
    """
    df = pd.melt(df, var_name='Model', value_name='Score')
    ax = sns.boxplot(x='Model', y='Score', data=df, **boxplot_kwargs)
    ax.set(ylabel='Cross validation score')
    return ax