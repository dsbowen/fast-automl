import gshap
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


class TreatmentRegression():
    def __init__(self, control_model, treat_model, treat_var, control_val=0, treat_val=1, scoring=r2_score):
        self.control_model = control_model
        self.treat_model = treat_model
        self.treat_var = treat_var
        self.control_val = control_val
        self.treat_val = treat_val
        self.scoring = scoring
        
    def fit(self, X, y, *args, **kwargs):
        X_control, X_treat, y_control, y_treat = self.control_treat_split(X, y)
        self.control_model.fit(X_control, y_control, *args, **kwargs)
        self.treat_model.fit(X_treat, y_treat, *args, **kwargs)
        control_idx, treat_idx = self.control_treat_idx(X)
        y_pred = self.predict(X)
        self.ate_ = y_pred[treat_idx].mean() - y_pred[control_idx].mean()
        return self
        
    def predict(self, X, *args, **kwargs):
        control_idx, treat_idx = self.control_treat_idx(X)
        X = self.transform_X(X)
        return (
            control_idx * self.control_model.predict(X, *args, **kwargs) 
            + treat_idx * self.treat_model.predict(X, *args, **kwargs)
        )
    
    def predict_reverse(self, X, *args, **kwargs):
        control_idx, treat_idx = self.control_treat_idx(X)
        X = self.transform_X(X)
        return (
            control_idx * (self.treat_model.predict(X, *args, **kwargs) - self.ate_)
            + treat_idx * (self.control_model.predict(X, *args, **kwargs) + self.ate_)
        )
    
    def predict_effect(self, X, *args, **kwargs):
        X = self.transform_X(X)
        return self.treat_model.predict(X) - self.control_model.predict(X)
    
    def control_treat_split(self, X, y=None):
        return control_treat_split(self.treat_var, X, y, self.control_val, self.treat_val)
    
    def control_treat_idx(self, X):
        return control_treat_idx(self.treat_var, X, self.control_val, self.treat_val)
        
    def transform_X(self, X):
        return (
            X.drop(columns=self.treat_var) if isinstance(X, pd.DataFrame) 
            else np.delete(X, self.treat_var, axis=1)
        )
    
    def score(self, X, y, *args, **kwargs):
        return self.scoring(y, self.predict(X), *args, **kwargs)
    
    def score_reverse(self, X, y, *args, **kwargs):
        return self.scoring(y, self.predict_reverse(X), *args, **kwargs)
    
    def get_params(self, deep=True):
        params = dict(
            control_model=self.control_model,
            treat_model=self.treat_model,
            treat_var=self.treat_var,
            control_val=self.control_val,
            treat_val=self.treat_val,
            scoring=self.scoring
        )
        if deep:
            control_params = self.control_model.get_params(deep)
            treat_params = self.treat_model.get_params(deep)
            params.update({'control__'+key: val for key, val in control_params.items()})
            params.update({'treat__'+key: val for key, val in treat_params.items()})
        return params
    
    def set_params(self, **params):
        control_params, treat_params = {}, {}
        for key, val in params.items():
            if key.startswith('control__'):
                control_params[key[len('control__'):]] = val
            elif key.startswith('treat__'):
                treat_params[key[len('treat__'):]] = val
        self.control_model = params['control_model']
        self.treat_model = params['treat_model']
        self.treat_var = params['treat_var']
        self.control_val = params['control_val']
        self.treat_val = params['treat_val']
        self.scorer = params['scorer']
        self.control_model.set_params(**control_params)
        self.treat_model.set_params(**treat_params)
        
    
def control_treat_split(treat_var, X, y=None, control_val=0, treat_val=1):
    control_idx, treat_idx = control_treat_idx(treat_var, X, control_val, treat_val)
    if isinstance(X, pd.DataFrame):
        # treat_var is a column name
        X = X.drop(columns=treat_var)
    else:
        # treat_var is a column index
        X = np.delete(X, treat_var, axis=1)
    X_control, X_treat = X[control_idx], X[treat_idx]
    if y is None:
        return X_control, X_treat
    return X_control, X_treat, y[control_idx], y[treat_idx]

def control_treat_idx(treat_var, X, control_val=0, treat_val=1):
    if isinstance(X, pd.DataFrame):
        # treatment variable is a column name
        return X[treat_var] == control_val, X[treat_var] == treat_val
    # treatment variable is a column index
    return X[:,treat_var] == control_val, X[:,treat_var] == treat_val
    
def cross_validate_treatment(estimator, X, y, fit_params={}):
    # TODO scoring, cv
    kf = KFold()
    score, score_reverse = [], []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.values[train_idx], y.values[test_idx]
        reg.fit(X_train, y_train, **fit_params)
        score.append(reg.score(X_test, y_test))
        score_reverse.append(reg.score_reverse(X_test, y_test))
    return np.array(score_reverse), np.array(score)

def explain_effect(reg, X, nsamples=1000, local=True, scoring=r2_score, scoring_params={}):
    if local:
        g = lambda effect_pred: effect_pred
    else:
        effect = reg.predict_effect(X)
        g = lambda effect_pred: scoring(effect, effect_pred, **scoring_params)
    explainer = gshap.KernelExplainer(reg.predict_effect, X, g)
    gshap_values = explainer.gshap_values(X, nsamples=nsamples)
    if local:
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(columns=X.columns, data=gshap_values.T)
        return gshap_values.T
    gshap_values /= gshap_values.sum()
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame({'Feature': X.columns, 'G-SHAP': gshap_values})
    return gshap_values