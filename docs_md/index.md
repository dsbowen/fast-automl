# Fast AutoML

Most autoML packages aim for exceptional performance but need to train for an exceptional amount of time. Fast-autoML aims for reasonable performance in a reasonable amount of time.

Fast-autoML includes additional utilities, such as tools for comparing model performance by repeated cross-validation.

## Installation

```
$ pip install fast-automl
```

## Quickstart

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

## Citation

```
@software{bowen2021fast-automl,
  author = {Dillon Bowen},
  title = {Fast-AutoML},
  url = {https://dsbowen.github.io/fast-automl/},
  date = {2021-02-05},
}
```

## License

Users must cite this package in any publications which use it.

It is licensed with the MIT [License](https://github.com/dsbowen/fast-automl/blob/master/LICENSE).

## Acknowledgments

This package and its documentation draw heavily on [scikit-learn](https://scikit-learn.org/stable/).