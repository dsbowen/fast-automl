from ml_inference import TransformerMixin

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

INFILE = '../data/pennycook_et_al_study2_clean.csv'
dummy_cols = ['Party', 'POTUS2016']
df = pd.read_csv(INFILE)
categories = [df[col].unique() for col in dummy_cols]


class Preprocessor(TransformerMixin):
    def __init__(self):
        self.encoder_ = OneHotEncoder(categories=categories, sparse=False).fit(df[dummy_cols])
        self.feature_names_ = list(self.encoder_.get_feature_names(dummy_cols))

    def transform(self, X):
        X = X.copy()
        X[self.feature_names_] = self.encoder_.transform(X[dummy_cols])
        X = X.drop(columns=dummy_cols)
        # indicate correct and intuitive CRT responses
        X['CRT1_1_corr'] = X.CRT1_1 == 4
        X['CRT1_1_intuit'] = X.CRT1_1 == 8
        X['CRT1_2_corr'] = X.CRT1_2 == 10
        X['CRT1_2_intuit'] = X.CRT1_2 == 50
        X['CRT1_3_corr'] = X.CRT1_3 == 39
        X['CRT1_3_intuit'] = X.CRT1_3 == 20
        X['CRT3_1_corr'] = X.CRT3_1 == 2
        X['CRT3_1_intuit'] = X.CRT3_1 == 1
        X['CRT3_2_corr'] = X.CRT3_2 == 8
        X['CRT3_2_intuit'] = X.CRT3_2 == 7
        return X
