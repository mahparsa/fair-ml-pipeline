"""SHAP-based explainability, plus the interactive teaching helper
`explain_step` that prints an explanation + real source code the moment
a user opts into a pipeline step."""

import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def top_shap_features(shap_array, feature_names, n):
    if shap_array.shape[0] == 0:
        return []
    mean_abs = np.abs(shap_array).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:n]
    return [(feature_names[i], float(mean_abs[i])) for i in order]


def print_ranked_features(label, shap_array, feature_names, n):
    ranked = top_shap_features(shap_array, feature_names, n)
    if not ranked:
        print(f"  {label}: no samples in this group")
        return
    print(f"  {label} -- top {n} features by importance:")
    for rank, (name, val) in enumerate(ranked, 1):
        print(f"    {rank}. {name} ({val:.4f})")


def run_shap_for_fold(fold_idx, best_classifier, X_train, X_test, sensitive_test, n_top_features):
    """Computes and plots SHAP values for one outer fold, both overall
    and stratified by each sensitive attribute. Returns a dict with the
    raw SHAP arrays, or None if SHAP failed for this fold."""
    import shap
    try:
        explainer = shap.Explainer(best_classifier, X_train)
        shap_values = explainer(X_test)
        if shap_values.values.ndim == 3:
            shap_values = shap_values[:, :, -1]
        shap_array = shap_values.values
        feature_names = X_test.columns.tolist()

        fold_shap = {"all": shap_array, "features": feature_names, "by_group": {}}

        print(f"Fold {fold_idx} feature importance ranking (top {n_top_features}):")
        print_ranked_features("All", shap_array, feature_names, n_top_features)

        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(shap_values, max_display=20, show=False)
        plt.title(f"Fold {fold_idx} - SHAP All (Target: 0=Low,1=High)")
        plt.show()

        for col, group_vals in sensitive_test.items():
            fold_shap["by_group"][col] = {}
            unique_groups = [g for g in pd.unique(group_vals) if pd.notna(g)]

            for g in unique_groups:
                mask = (group_vals == g)
                shap_subset = shap_array[mask]
                fold_shap["by_group"][col][g] = shap_subset

                if mask.sum() == 0:
                    continue

                label = f"{col}={g}"
                print_ranked_features(label, shap_subset, feature_names, n_top_features)

                plt.figure(figsize=(10, 6))
                shap.plots.beeswarm(shap_values[mask], max_display=20, show=False)
                plt.title(f"Fold {fold_idx} - SHAP {label} (Target: 0=Low,1=High)")
                plt.show()

        return fold_shap
    except Exception as e:
        print(f"SHAP failed on fold {fold_idx}: {e}")
        return None


# =========================
# Teaching feature: explain_step
# Prints a plain-language explanation + the real function source code,
# right at the moment a student/user opts into that pipeline step.
# =========================
FUNCTION_INFO = {}  # populated lazily in explain_step to avoid heavy imports at module load


def _build_function_info():
    from .cv import normalize_fold, oversample_fold, synthetic_oversample_fold, pick_features
    from .fairness import compute_all_sensitive_fairness

    return {
        "normalization": {
            "func": normalize_fold,
            "description": (
                "Normalization rescales your numeric features so they're on a comparable "
                "scale. This uses RobustScaler, which is fit ONLY on the training fold -- "
                "never the test fold -- to avoid leaking test information into training."
            ),
        },
        "oversampling": {
            "func": oversample_fold,
            "description": (
                "Oversampling balances the number of positive vs. negative rows in the "
                "training data using SMOTENC, which generates realistic synthetic examples "
                "of the minority class rather than just duplicating existing rows. Only "
                "applied to the training fold."
            ),
        },
        "synthetic": {
            "func": synthetic_oversample_fold,
            "description": (
                "This trains a generative model (CTGAN, CopulaGAN, or TVAE) on the training "
                "fold to create brand-new synthetic rows, growing the training set overall "
                "(not just rebalancing classes)."
            ),
        },
        "feature_selection": {
            "func": pick_features,
            "description": (
                "Feature selection tries several different methods for picking the most "
                "useful subset of features, and reports which method (and how many "
                "features) gave the best macro-F1 score."
            ),
        },
        "fairness": {
            "func": compute_all_sensitive_fairness,
            "description": (
                "This checks whether the model's predictions differ systematically between "
                "sensitive groups -- not just how accurate the model is overall, but "
                "whether it's equally accurate for everyone."
            ),
        },
        "shap": {
            "func": run_shap_for_fold,
            "description": (
                "SHAP explains WHY the model made each prediction, by scoring how much each "
                "feature pushed a given prediction up or down. This shows the top features "
                "overall and separately for each sensitive group."
            ),
        },
    }


def explain_step(key):
    """Prints a plain-language explanation and the real source code for a
    pipeline step, triggered right when a user opts into that step."""
    global FUNCTION_INFO
    if not FUNCTION_INFO:
        FUNCTION_INFO = _build_function_info()

    info = FUNCTION_INFO.get(key)
    if not info:
        return
    print("\n" + "=" * 70)
    print(f"STEP EXPLAINED: {key}")
    print("=" * 70)
    print(info["description"])
    print("\n--- Actual function code ---\n")
    print(inspect.getsource(info["func"]))
    print("=" * 70 + "\n")
