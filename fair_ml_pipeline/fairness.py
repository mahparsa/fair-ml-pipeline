"""Fairness metrics -- works with any binary sensitive group labels."""

import pandas as pd
from sklearn.metrics import confusion_matrix


def compute_fairness(y_true, y_pred, group):
    """Compute fairness metrics as plain floats. Works with any group
    label type -- integers or strings."""
    groups = [g for g in pd.unique(group) if pd.notna(g)]
    fairness = {"groups": list(groups), "positive_rate": {}, "tpr": {}, "fpr": {}, "pp": {}}

    for g in groups:
        subset_idx = (group == g)
        subset_y_true = y_true[subset_idx]
        subset_y_pred = y_pred[subset_idx]
        tn, fp, fn, tp = confusion_matrix(subset_y_true, subset_y_pred, labels=[0, 1]).ravel()
        fairness["positive_rate"][g] = float(subset_y_pred.mean())
        fairness["tpr"][g] = float(tp / (tp + fn)) if (tp + fn) else 0.0
        fairness["fpr"][g] = float(fp / (fp + tn)) if (fp + tn) else 0.0
        fairness["pp"][g] = float(tp / (tp + fp)) if (tp + fp) else 0.0

    if len(groups) >= 2:
        g0, g1 = groups[0], groups[1]
        fairness["DPD"] = float(fairness["positive_rate"][g0] - fairness["positive_rate"][g1])
        fairness["EO"] = float(fairness["tpr"][g0] - fairness["tpr"][g1])
        fairness["FPR_diff"] = float(fairness["fpr"][g0] - fairness["fpr"][g1])
        fairness["PP_diff"] = float(fairness["pp"][g0] - fairness["pp"][g1])
        fairness["DI"] = (float(fairness["positive_rate"][g0] / fairness["positive_rate"][g1])
                           if fairness["positive_rate"][g1] > 0 else float("nan"))
    return fairness


def compute_all_sensitive_fairness(y_true, y_pred, sensitive_test):
    return {col: compute_fairness(y_true, y_pred, sensitive_test[col]) for col in sensitive_test}
