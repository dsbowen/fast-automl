from sklearn.metrics import roc_auc_score as roc_auc_score_base, make_scorer

def check_scoring(scoring, classifier):
    if scoring is not None:
        return scoring
    return roc_auc_scorer if classifier else 'r2'

def roc_auc_score(y, output):
    if len(output.shape) == 1:
        # binary classifier
        return roc_auc_score_base(y, output)
    # multi-class
    return roc_auc_score_base(y, output, multi_class='ovr')

roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)