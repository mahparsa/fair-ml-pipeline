"""Nested cross-validation: feature selection, normalization, oversampling
(SMOTENC + synthetic generation), the main driver function, and supporting
utilities. Works with any feature matrix and any sensitive attributes."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE, RFECV, SelectKBest, SelectPercentile, SelectFromModel,
    VarianceThreshold, mutual_info_classif,
)
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score,
    recall_score, confusion_matrix, roc_curve, make_scorer,
)


def summarize_metric(outer_results, metric):
    values = [res[metric] for res in outer_results]
    return f"{np.mean(values):.4f} \u00b1 {np.std(values):.4f}"


# =========================
# Feature selection
# =========================
method_by_number = {
    1: "RFECV",
    2: "RFE",
    3: "SelectPercentile",
    4: "SelectKBest",
    5: "VarianceThreshold",
    6: "SelectFromModel",
}
method_options = list(method_by_number.values())

fs_param_grid_options = {
    "small": {
        "n_estimators": [30], "max_depth": [4],
        "min_samples_split": [2], "min_samples_leaf": [2], "max_features": ["sqrt"],
    },
    "large": {
        "n_estimators": [10, 20, 30, 40, 50], "max_depth": [2, 3, 4, 5],
        "min_samples_split": [2, 3, 4, 5, 6], "min_samples_leaf": [2, 3, 4],
        "max_features": ["sqrt", "log2"],
    },
}


def build_selector(method_name, cv):
    """Builds the sklearn selector object for a given method name."""
    if method_name == "RFECV":
        return RFECV(estimator=RandomForestClassifier(random_state=42), step=1, cv=cv,
                      scoring=make_scorer(f1_score, average="macro"), n_jobs=-1)
    elif method_name == "SelectFromModel":
        return SelectFromModel(estimator=RandomForestClassifier(random_state=42))
    elif method_name == "SelectKBest":
        return SelectKBest(mutual_info_classif, k="all")
    elif method_name == "RFE":
        return RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=5)
    elif method_name == "SelectPercentile":
        return SelectPercentile(mutual_info_classif, percentile=50)
    elif method_name == "VarianceThreshold":
        return VarianceThreshold()
    else:
        raise ValueError(f"Unknown method '{method_name}'. Choose from: {method_options}")


def resolve_method(item):
    """Turns a method reference (number or name) into its canonical name."""
    if isinstance(item, int):
        if item not in method_by_number:
            raise ValueError(f"Unknown method number {item}. Valid numbers: {method_by_number}")
        return method_by_number[item]
    elif isinstance(item, str):
        if item not in method_options:
            raise ValueError(f"Unknown method '{item}'. Choose from: {method_options}")
        return item
    else:
        raise TypeError(f"method entries must be int or str, got {type(item)}: {item}")


def pick_features(X, y, method=None, cv_range=range(2, 11), param_grid="small", plot=False, verbose=False):
    """
    Runs one or more feature selection methods, each paired with a
    RandomForest GridSearchCV, and reports which method + CV fold count
    gives the best macro-F1 score.

    Returns
    -------
    scores_table, X_top, summary  (summary has 'method', 'cv', 'feature_names')
    """
    if method is None:
        chosen_methods = method_options
    elif isinstance(method, (int, str)):
        chosen_methods = [resolve_method(method)]
    else:
        chosen_methods = [resolve_method(m) for m in method]

    if param_grid is None:
        chosen_param_grid = fs_param_grid_options["small"]
    elif isinstance(param_grid, str):
        if param_grid not in fs_param_grid_options:
            raise ValueError(f"Unknown param_grid preset '{param_grid}'. Choose from: "
                              f"{list(fs_param_grid_options)}, or pass a custom dict.")
        chosen_param_grid = fs_param_grid_options[param_grid]
    else:
        chosen_param_grid = param_grid

    scorer = make_scorer(f1_score, average="macro")
    col_names = X.columns
    scores_by_method = {}

    for method_name in chosen_methods:
        if verbose:
            print(f"Running {method_name}...")

        top_score = -np.inf
        top_cv = None
        top_n_features = None
        top_features = None
        top_params = None

        for cv_value in cv_range:
            cv = StratifiedKFold(n_splits=cv_value)
            selector = build_selector(method_name, cv)
            selector.fit(X, y)
            X_reduced = selector.transform(X) if hasattr(selector, "transform") else X

            grid_search = GridSearchCV(RandomForestClassifier(random_state=42), chosen_param_grid,
                                        cv=cv, scoring=scorer, n_jobs=-1)
            grid_search.fit(X_reduced, y)
            mean_score = grid_search.best_score_

            if mean_score > top_score:
                top_score = mean_score
                top_cv = cv_value
                mask = selector.get_support() if hasattr(selector, "get_support") else None
                top_n_features = X_reduced.shape[1]
                top_features = col_names[mask] if mask is not None else col_names
                top_params = grid_search.best_params_

        scores_by_method[method_name] = {
            "Best CV Value": top_cv, "Best Number of Features": top_n_features,
            "Best Score": top_score, "Best Features": top_features, "Best RF Params": top_params,
        }

    scores_table = pd.DataFrame(scores_by_method).T
    scores_table = scores_table.sort_values(by="Best Score", ascending=False)

    if plot and len(chosen_methods) > 1:
        plt.figure(figsize=(12, 6))
        colors = sns.color_palette("viridis", len(scores_table))
        bars = plt.bar(scores_table.index, scores_table["Best Score"], color=colors)
        plt.xlabel("Feature Selection Method")
        plt.ylabel("Best F1 Macro Score")
        plt.title("Comparison of Feature Selection Methods")
        plt.xticks(rotation=45, ha="right")
        for bar, cv in zip(bars, scores_table["Best CV Value"]):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"CV={cv}", ha="center", va="bottom", fontsize=10)
        plt.tight_layout()
        plt.show()

    top_method = scores_table.index[0]
    top_method_features = scores_table.loc[top_method, "Best Features"]
    top_method_cv = scores_table.loc[top_method, "Best CV Value"]
    X_top = X[top_method_features] if top_method_features is not None else X

    summary = {
        "method": top_method, "cv": top_method_cv,
        "feature_names": list(top_method_features) if top_method_features is not None else list(col_names),
    }

    if verbose:
        print(f"\nBest Method: {summary['method']}")
        print(f"Best CV Value: {summary['cv']}")
        print(f"Selected Features:\n{summary['feature_names']}")

    return scores_table, X_top, summary


def get_common_features(selected_features_per_fold):
    """Returns the features that were selected in EVERY fold, as a sorted list."""
    common = set(selected_features_per_fold[0])
    for feats in selected_features_per_fold[1:]:
        common &= set(feats)
    return sorted(common)


# =========================
# Normalization
# =========================
def normalize_fold(X_train, X_test, numeric_cols):
    """Fits a RobustScaler on X_train's numeric columns only, then
    applies it to both X_train and X_test."""
    scaler = RobustScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train, X_test


# =========================
# Oversampling (SMOTENC)
# =========================
def oversample_smotenc(data, feature_cols, categorical_cols, id_cols, target_col,
                        sensitive_cols=None, random_state=42, verbose=True):
    """Balances classes in `data` using SMOTENC (SMOTE for mixed
    categorical/numeric data)."""
    from imblearn.over_sampling import SMOTENC

    sensitive_cols = sensitive_cols or []
    extra_sensitive = [c for c in sensitive_cols if c not in feature_cols]
    all_cols = feature_cols + extra_sensitive

    missing = [c for c in all_cols + id_cols + categorical_cols + [target_col] if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    if not categorical_cols:
        raise ValueError(
            "categorical_cols is empty, but SMOTENC requires at least one categorical "
            "feature. If all your features are numeric, use imblearn's plain SMOTE instead."
        )

    bad_cat_names = [c for c in categorical_cols if c not in all_cols]
    if bad_cat_names:
        raise ValueError(f"categorical_cols contains columns not in feature_cols or sensitive_cols: {bad_cat_names}")

    X = data[all_cols].reset_index(drop=True)
    y = data[target_col].reset_index(drop=True)
    metadata = data[id_cols].reset_index(drop=True)

    feature_encoders = {}
    for col in categorical_cols:
        if not pd.api.types.is_numeric_dtype(X[col]):
            enc = LabelEncoder()
            X[col] = enc.fit_transform(X[col])
            feature_encoders[col] = enc

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    if verbose:
        print("Class distribution BEFORE oversampling:")
        print(y.value_counts())

    cat_idx = [all_cols.index(c) for c in categorical_cols]
    smote_nc = SMOTENC(categorical_features=cat_idx, random_state=random_state)
    X_resampled_arr, y_resampled = smote_nc.fit_resample(X, y_encoded)
    X_resampled = pd.DataFrame(X_resampled_arr, columns=all_cols)
    for c in categorical_cols:
        X_resampled[c] = X_resampled[c].astype(X[c].dtype)

    n_original = len(X)
    n_total = len(X_resampled)
    n_synthetic = n_total - n_original
    if verbose:
        print(f"Original samples: {n_original}, synthetic samples added: {n_synthetic}")

    metadata_original = metadata.copy()
    metadata_original["is_synthetic"] = False

    if n_synthetic > 0:
        num_cols = [c for c in all_cols if c not in categorical_cols]
        scaler = StandardScaler()
        X_num_scaled = pd.DataFrame(
            scaler.fit_transform(X[num_cols]) if num_cols else np.empty((len(X), 0)), columns=num_cols)
        X_resampled_num_scaled = pd.DataFrame(
            scaler.transform(X_resampled[num_cols]) if num_cols else np.empty((len(X_resampled), 0)), columns=num_cols)

        if categorical_cols:
            X_cat_all = pd.get_dummies(
                pd.concat([X[categorical_cols], X_resampled[categorical_cols]], axis=0), columns=categorical_cols
            ).reset_index(drop=True)
            X_cat_orig = X_cat_all.iloc[:n_original].reset_index(drop=True)
            X_cat_resampled = X_cat_all.reset_index(drop=True)
        else:
            X_cat_orig = pd.DataFrame(index=range(n_original))
            X_cat_resampled = pd.DataFrame(index=range(n_total))

        X_nn_space_orig = pd.concat([X_num_scaled, X_cat_orig], axis=1).values
        X_nn_space_resampled = pd.concat(
            [X_resampled_num_scaled, X_cat_resampled.iloc[:n_total].reset_index(drop=True)], axis=1).values

        nn = NearestNeighbors(n_neighbors=1).fit(X_nn_space_orig)
        synthetic_space = X_nn_space_resampled[n_original:]
        _, nn_idx = nn.kneighbors(synthetic_space)

        metadata_synthetic = metadata.iloc[nn_idx.flatten()].reset_index(drop=True)
        metadata_synthetic["is_synthetic"] = True
    else:
        metadata_synthetic = pd.DataFrame(columns=list(metadata.columns) + ["is_synthetic"])

    metadata_resampled = pd.concat([metadata_original, metadata_synthetic], axis=0).reset_index(drop=True)
    df_resampled = pd.concat([metadata_resampled, X_resampled], axis=1)
    df_resampled[target_col] = le.inverse_transform(y_resampled)
    df_resampled = df_resampled[id_cols + ["is_synthetic", target_col] + all_cols]

    for col, enc in feature_encoders.items():
        df_resampled[col] = enc.inverse_transform(df_resampled[col].astype(int))

    if verbose:
        print("Balanced dataset size:", len(df_resampled))
        print("Class distribution AFTER oversampling:")
        print(df_resampled[target_col].value_counts())

    return df_resampled, le


def print_oversampling_summary(train_df, df_resampled, target_col, sensitive_col_names):
    """Prints a before/after breakdown of how many rows were added by
    oversampling, split by target class and by each sensitive attribute."""
    n_before = len(train_df)
    n_after = len(df_resampled)
    n_synthetic = n_after - n_before
    synthetic_mask = df_resampled["is_synthetic"]

    print(f"  Oversampling summary: {n_before} -> {n_after} rows (+{n_synthetic} synthetic)")
    print("  By class (target):")
    before_counts = train_df[target_col].value_counts()
    after_counts = df_resampled[target_col].value_counts()
    for cls in sorted(after_counts.index, key=str):
        before_n = int(before_counts.get(cls, 0))
        after_n = int(after_counts.get(cls, 0))
        print(f"    {cls}: {before_n} -> {after_n} (+{after_n - before_n})")

    for col in sensitive_col_names:
        if col not in df_resampled.columns:
            continue
        print(f"  By sensitive attribute '{col}':")
        before_counts = train_df[col].value_counts()
        after_counts = df_resampled[col].value_counts()
        synthetic_counts = df_resampled.loc[synthetic_mask, col].value_counts()
        for group in sorted(after_counts.index, key=str):
            before_n = int(before_counts.get(group, 0))
            after_n = int(after_counts.get(group, 0))
            synth_n = int(synthetic_counts.get(group, 0))
            print(f"    {group}: {before_n} -> {after_n} (+{synth_n} synthetic)")


def oversample_fold(X_train, y_train, categorical_cols, sensitive_train=None, random_state=42, verbose=True):
    """Runs oversample_smotenc on the training fold only (never on X_test)."""
    train_df = X_train.copy()
    train_df["__target__"] = y_train
    train_df["__uid__"] = range(len(train_df))

    sensitive_train = sensitive_train or {}
    sensitive_col_names = list(sensitive_train.keys())
    for col, values in sensitive_train.items():
        train_df[col] = np.asarray(values)

    combined_categorical_cols = list(categorical_cols) + [c for c in sensitive_col_names if c not in categorical_cols]

    df_resampled, _ = oversample_smotenc(
        data=train_df, feature_cols=list(X_train.columns), categorical_cols=combined_categorical_cols,
        id_cols=["__uid__"], target_col="__target__", sensitive_cols=sensitive_col_names,
        random_state=random_state, verbose=False,
    )

    if verbose:
        print_oversampling_summary(train_df, df_resampled, "__target__", sensitive_col_names)

    X_train_new = df_resampled[list(X_train.columns)].reset_index(drop=True)
    y_train_new = df_resampled["__target__"].values
    return X_train_new, y_train_new


def synthetic_oversample_fold(X_train, y_train, categorical_cols, method, multiplier, epochs,
                               sensitive_train=None, random_state=42, verbose=True):
    """
    Trains a CTGAN, CopulaGAN, or TVAE synthesizer (via the SDV library)
    on the training fold only, then generates synthetic rows to grow the
    training set. Unlike SMOTENC, this does NOT rebalance classes.
    """
    n_original = len(X_train)
    n_target_total = n_original * multiplier
    n_synthetic = max(n_target_total - n_original, 0)

    if n_synthetic == 0:
        if verbose:
            print(f"  Synthetic data generation ({method}): multiplier=1, no rows added.")
        return X_train.reset_index(drop=True), y_train, (sensitive_train or {})

    from .models import _install_and_import
    _install_and_import("sdv")
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer, CopulaGANSynthesizer, TVAESynthesizer

    synthesizer_classes = {"CTGAN": CTGANSynthesizer, "CopulaGAN": CopulaGANSynthesizer, "TVAE": TVAESynthesizer}
    if method not in synthesizer_classes:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(synthesizer_classes)}")

    train_df = X_train.copy()
    train_df["__target__"] = y_train
    sensitive_train = sensitive_train or {}
    sensitive_col_names = list(sensitive_train.keys())
    for col, values in sensitive_train.items():
        train_df[col] = np.asarray(values)

    discrete_columns = list(categorical_cols) + [c for c in sensitive_col_names if c not in categorical_cols] + ["__target__"]

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=train_df)
    for col in discrete_columns:
        if col in train_df.columns:
            metadata.update_column(column_name=col, sdtype="categorical")

    synthesizer = synthesizer_classes[method](metadata, epochs=epochs, verbose=False)
    synthesizer.fit(train_df)
    synthetic_df = synthesizer.sample(num_rows=n_synthetic)
    combined = pd.concat([train_df, synthetic_df], ignore_index=True)

    if verbose:
        print(f"  Synthetic data summary ({method}, {multiplier}x): "
              f"{n_original} -> {len(combined)} rows (+{n_synthetic} synthetic)")

    X_train_new = combined[list(X_train.columns)].reset_index(drop=True)
    y_train_new = combined["__target__"].values
    sensitive_train_new = {col: combined[col].values for col in sensitive_col_names if col in combined.columns}
    return X_train_new, y_train_new, sensitive_train_new


def encode_categorical_fold(X_train, X_test, categorical_cols):
    """Label-encodes categorical columns, fitting on this fold's
    train+test categories combined so both sides get matching numeric codes."""
    X_train = X_train.copy()
    X_test = X_test.copy()
    for col in categorical_cols:
        if not pd.api.types.is_numeric_dtype(X_train[col]):
            enc = LabelEncoder()
            enc.fit(pd.concat([X_train[col], X_test[col]], axis=0))
            X_train[col] = enc.transform(X_train[col])
            X_test[col] = enc.transform(X_test[col])
    return X_train, X_test


def evaluate_classifiers_inner_cv(X_train, y_train, X_test, y_test, classifiers, param_grids, inner_cv):
    """Runs grid-search hyperparameter tuning for each candidate classifier
    on the inner CV, scores each tuned model on the outer test fold, and
    returns the best-scoring model."""
    best_classifier = None
    best_params = None
    best_score = 0.0
    best_model_name = None
    inner_models = {}

    for clf, params in zip(classifiers, param_grids):
        model_name = clf.__class__.__name__
        print(f"Evaluating {model_name}...")
        grid_search = GridSearchCV(estimator=clf, param_grid=params, cv=inner_cv, scoring="accuracy", refit=True)
        grid_search.fit(X_train, y_train)
        best_model_inner = grid_search.best_estimator_
        y_pred_inner = best_model_inner.predict(X_test)
        acc = accuracy_score(y_test, y_pred_inner)
        inner_models[model_name] = {"model": best_model_inner, "params": grid_search.best_params_, "accuracy": acc}
        if acc > best_score:
            best_score = acc
            best_classifier = best_model_inner
            best_model_name = model_name
            best_params = grid_search.best_params_

    if best_classifier is None:
        best_classifier = classifiers[0]
        best_model_name = best_classifier.__class__.__name__
        best_params = {}

    return best_classifier, best_model_name, best_params, inner_models


def plot_mean_roc(fold_roc_data):
    """Plots per-fold ROC curves plus the mean ROC curve across all outer folds."""
    mean_fpr = np.linspace(0, 1, 100)
    interpolated_tprs = []
    plt.figure(figsize=(7, 6))
    for fpr, tpr, auc_val, fold_idx in fold_roc_data:
        plt.plot(fpr, tpr, alpha=0.35, label=f"Fold {fold_idx} (AUC={auc_val:.3f})")
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interpolated_tprs.append(interp_tpr)
    mean_tpr = np.mean(interpolated_tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(interpolated_tprs, axis=0)
    mean_auc = np.mean([auc_val for _, _, auc_val, _ in fold_roc_data])
    std_auc = np.std([auc_val for _, _, auc_val, _ in fold_roc_data])
    plt.plot(mean_fpr, mean_tpr, color="navy", lw=2.5, label=f"Mean ROC (AUC={mean_auc:.3f} \u00b1 {std_auc:.3f})")
    plt.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1), color="navy", alpha=0.15)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Across All Nested CV Folds")
    plt.legend(fontsize=8, loc="lower right")
    plt.show()


def print_final_summary(outer_results, run_fairness):
    """Prints the per-fold results table and the overall mean \u00b1 SD
    summary line for each performance metric."""
    print("\n======= Final Summary =======")
    for res in outer_results:
        fold = res["fold"]
        print(f"Fold {fold} - {res['model_name']}: Acc={res['accuracy']:.4f}, F1={res['f1_score']:.4f}, "
              f"AUC={res['auc']:.4f}, BalAcc={res['balanced_accuracy']:.4f}, Recall={res['recall']:.4f}")
        if run_fairness:
            print(f"Fairness Metrics: {res['fairness']}")

    print(f"\nOverall Accuracy: {summarize_metric(outer_results, 'accuracy')}")
    print(f"Overall F1 Score: {summarize_metric(outer_results, 'f1_score')}")
    print(f"Overall AUC: {summarize_metric(outer_results, 'auc')}")
    print(f"Overall Balanced Accuracy: {summarize_metric(outer_results, 'balanced_accuracy')}")
    print(f"Overall Recall: {summarize_metric(outer_results, 'recall')}")


# =========================
# Main nested CV driver
# =========================
def nested_cv_normalized_oversampled_featureselected(
    X, y_array, sensitive_data, classifiers, param_grids,
    categorical_cols=None, numeric_cols=None,
    best_cv=None,
    n_outer_folds=None,
    run_normalization=None,
    run_oversampling=None,
    oversample_sensitive_cols=None,
    run_synthetic_oversampling=None,
    synthetic_method=None,
    synthetic_multiplier=None,
    synthetic_epochs=None,
    treat_sensitive_as_features=None,
    sensitive_feature_cols=None,
    run_feature_selection=None,
    fs_method=None, fs_cv_range=None, fs_param_grid="small",
    run_fairness=True, run_shap=True,
    teach=False,
):
    """
    Nested cross-validation with, per outer training fold, any combination
    of the following steps (each independently toggleable): SMOTENC
    oversampling, categorical encoding, RobustScaler normalization, and
    feature selection.

    sensitive_data : dict, or a list of (name, array) tuples, e.g.
        [('gender', gender_array), ('ethnicity', ethnicity_array)]
    teach : bool, default False. If True, calls explain_step(...) right
        after the user opts into normalization/oversampling/synthetic
        data/feature selection/fairness/SHAP, printing a plain-language
        explanation and the real function source code for that step.
    """
    from .prompts import (
        ask_int_in_range, ask_yes_no, ask_synthetic_method,
        ask_feature_selection_methods, ask_n_top_features,
    )
    from .fairness import compute_all_sensitive_fairness

    categorical_cols = categorical_cols or []
    numeric_cols = numeric_cols or []
    sensitive_data = dict(sensitive_data)  # accepts a list of (name, array) tuples OR a dict
    sensitive_data = {col: np.asarray(vals) for col, vals in sensitive_data.items()}

    def _teach(key):
        if teach:
            from .explain import explain_step
            explain_step(key)

    if best_cv is None:
        best_cv = ask_int_in_range("How many inner CV folds (hyperparameter tuning)?", 3, 10)
    if n_outer_folds is None:
        n_outer_folds = ask_int_in_range("How many outer CV folds?", 2, 10)
    if run_normalization is None:
        run_normalization = ask_yes_no("Apply normalization (RobustScaler)?") if numeric_cols else False
    if run_normalization:
        _teach("normalization")
    if run_oversampling is None:
        run_oversampling = ask_yes_no("Apply oversampling (SMOTENC)?") if (categorical_cols or sensitive_data) else False
    if run_oversampling:
        _teach("oversampling")
    if run_synthetic_oversampling is None:
        run_synthetic_oversampling = ask_yes_no(
            "Apply synthetic data generation (CTGAN / CopulaGAN / TVAE) inside each fold?"
        )
    if run_synthetic_oversampling:
        _teach("synthetic")
    if run_synthetic_oversampling and synthetic_method is None:
        synthetic_method = ask_synthetic_method()
    if run_synthetic_oversampling and synthetic_multiplier is None:
        synthetic_multiplier = ask_int_in_range(
            "How many times larger should the training set become? "
            "(1 = same size / no growth, 2 = double, 3 = triple, ...)", 1, 5,
        )
    if run_synthetic_oversampling and synthetic_epochs is None:
        synthetic_epochs = ask_int_in_range(
            "How many training epochs for the synthesizer? (higher = better quality, slower)", 1, 1000,
        )
    if run_feature_selection is None:
        run_feature_selection = ask_yes_no("Apply feature selection?")
    if run_feature_selection:
        _teach("feature_selection")
    if run_feature_selection and fs_method is None:
        fs_method = ask_feature_selection_methods()
    if treat_sensitive_as_features is None:
        treat_sensitive_as_features = ask_yes_no(
            "Include sensitive attributes as candidate features, letting feature "
            "selection decide whether to keep them?"
        ) if sensitive_data else False
    if treat_sensitive_as_features and sensitive_feature_cols is None:
        sensitive_feature_cols = list(sensitive_data.keys())

    if fs_cv_range is None:
        fs_cv_range = range(3, best_cv + 1)

    if oversample_sensitive_cols is None:
        already_features = set(sensitive_feature_cols or [])
        oversample_sensitive_cols = [
            c for c in sensitive_data.keys() if c not in already_features
        ] if (run_oversampling or run_synthetic_oversampling) else []

    print(f"\nPipeline configuration -- Inner CV folds: {best_cv}, Outer folds: {n_outer_folds}, "
          f"Normalization: {run_normalization}, Oversampling (SMOTENC): {run_oversampling}"
          + (f" (using sensitive cols: {oversample_sensitive_cols})" if run_oversampling and oversample_sensitive_cols else "")
          + f", Synthetic generation: {run_synthetic_oversampling}"
          + (f" ({synthetic_method}, {synthetic_multiplier}x, {synthetic_epochs} epochs)" if run_synthetic_oversampling else "")
          + f", Feature selection: {run_feature_selection}"
          + (f" (methods: {fs_method or 'all'}, CV sweep {fs_cv_range.start}-{fs_cv_range.stop - 1})" if run_feature_selection else "")
          + f", Sensitive attrs as candidate features: {treat_sensitive_as_features}"
          + (f" ({sensitive_feature_cols})" if treat_sensitive_as_features else "")
          + "\n")

    if run_fairness:
        _teach("fairness")
    if run_shap:
        _teach("shap")

    n_top_features = ask_n_top_features(X.shape[1]) if run_shap else None

    outer_cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=42)
    outer_results = []
    best_models_per_fold = {}
    inner_best_models = {}
    fold_roc_data = []
    selected_features_per_fold = []
    fs_method_per_fold = []

    for fold_idx, (train_index, test_index) in enumerate(outer_cv.split(X, y_array), 1):
        print(f"\n========= Fold {fold_idx} =========")
        X_train, X_test = X.iloc[train_index].reset_index(drop=True), X.iloc[test_index].reset_index(drop=True)
        y_train, y_test = y_array[train_index], y_array[test_index]
        sensitive_test = {col: vals[test_index] for col, vals in sensitive_data.items()}
        sensitive_train = {col: vals[train_index] for col, vals in sensitive_data.items()}

        fold_categorical_cols = list(categorical_cols)
        if treat_sensitive_as_features:
            for col in sensitive_feature_cols:
                if col in sensitive_train:
                    X_train[col] = sensitive_train[col]
                    X_test[col] = sensitive_test[col]
                    if col not in fold_categorical_cols:
                        fold_categorical_cols.append(col)

        n_start = len(X_train)

        if run_synthetic_oversampling:
            fold_sensitive_train = {col: sensitive_train[col] for col in oversample_sensitive_cols if col in sensitive_train}
            X_train, y_train, fold_sensitive_train = synthetic_oversample_fold(
                X_train, y_train, fold_categorical_cols,
                method=synthetic_method, multiplier=synthetic_multiplier, epochs=synthetic_epochs,
                sensitive_train=fold_sensitive_train,
            )
        else:
            fold_sensitive_train = {col: sensitive_train[col] for col in oversample_sensitive_cols if col in sensitive_train}
        n_after_synthetic = len(X_train)

        if run_oversampling:
            X_train, y_train = oversample_fold(X_train, y_train, fold_categorical_cols, sensitive_train=fold_sensitive_train)
        n_after_oversample = len(X_train)

        n_synthetic_rows = n_after_synthetic - n_start
        n_oversampled_rows = n_after_oversample - n_after_synthetic

        if fold_categorical_cols:
            X_train, X_test = encode_categorical_fold(X_train, X_test, fold_categorical_cols)

        if run_normalization and numeric_cols:
            X_train, X_test = normalize_fold(X_train, X_test, numeric_cols)

        if run_feature_selection:
            _, _, fs_summary = pick_features(X_train, y_train, method=fs_method, cv_range=fs_cv_range,
                                              param_grid=fs_param_grid, plot=False, verbose=False)
            fold_selected_features = fs_summary["feature_names"]
            fold_fs_method = fs_summary["method"]
            X_train = X_train[fold_selected_features]
            X_test = X_test[fold_selected_features]
        else:
            fold_selected_features = list(X_train.columns)
            fold_fs_method = None

        selected_features_per_fold.append(fold_selected_features)
        fs_method_per_fold.append(fold_fs_method)
        print(f"Fold {fold_idx} selected features ({len(fold_selected_features)}): {fold_selected_features}")
        if fold_fs_method is not None:
            print(f"Fold {fold_idx} winning feature selection method: {fold_fs_method}")

        inner_cv = StratifiedKFold(n_splits=best_cv, shuffle=True, random_state=42)
        best_classifier, best_model_name, best_params, inner_models = evaluate_classifiers_inner_cv(
            X_train, y_train, X_test, y_test, classifiers, param_grids, inner_cv
        )
        for model_name, info in inner_models.items():
            inner_best_models[(fold_idx, model_name)] = info

        print(f"Best Hyperparameters from Inner CV for Fold {fold_idx}: {best_model_name} with {best_params}")

        y_pred_outer = best_classifier.predict(X_test)
        if hasattr(best_classifier, "predict_proba"):
            y_score_outer = best_classifier.predict_proba(X_test)[:, 1]
        else:
            y_score_outer = y_pred_outer.astype(float)

        acc = accuracy_score(y_test, y_pred_outer)
        f1 = f1_score(y_test, y_pred_outer)
        bal_acc = balanced_accuracy_score(y_test, y_pred_outer)
        rec = recall_score(y_test, y_pred_outer)
        auc_score = roc_auc_score(y_test, y_score_outer)
        cm = confusion_matrix(y_test, y_pred_outer)

        fpr, tpr, _ = roc_curve(y_test, y_score_outer)
        fold_roc_data.append((fpr, tpr, auc_score, fold_idx))

        if run_fairness:
            fairness_metrics = compute_all_sensitive_fairness(y_test, y_pred_outer, sensitive_test)
        else:
            fairness_metrics = None
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}, Balanced Acc: {bal_acc:.4f}, Recall: {rec:.4f}")
        if run_fairness:
            print(f"Fairness Metrics for Fold {fold_idx}: {fairness_metrics}")

        outer_results.append({"fold": fold_idx, "model_name": best_model_name, "accuracy": acc, "f1_score": f1,
            "auc": auc_score, "balanced_accuracy": bal_acc, "recall": rec, "confusion_matrix": cm, "fairness": fairness_metrics,
            "selected_features": fold_selected_features, "fs_method": fold_fs_method,
            "n_synthetic_rows": n_synthetic_rows, "n_oversampled_rows": n_oversampled_rows})
        best_models_per_fold[fold_idx] = {"model": best_classifier, "model_name": best_model_name, "params": best_params, "metrics": outer_results[-1]}

        if run_shap:
            from .explain import run_shap_for_fold as _run_shap_for_fold
            fold_shap = _run_shap_for_fold(fold_idx, best_classifier, X_train, X_test, sensitive_test, n_top_features)
            if fold_shap is not None:
                best_models_per_fold[fold_idx]["shap"] = fold_shap

    common_features = get_common_features(selected_features_per_fold)
    print(f"\n======= Common Features Across All {n_outer_folds} Folds =======")
    print(f"{len(common_features)} feature(s) selected in every fold: {common_features}")

    if run_feature_selection:
        print(f"\n======= Winning Feature Selection Method Per Fold =======")
        for i, m in enumerate(fs_method_per_fold, 1):
            print(f"Fold {i}: {m}")
        method_counts = pd.Series(fs_method_per_fold).value_counts()
        print(f"\nMost frequent winning method overall: {method_counts.index[0]} "
              f"(won {method_counts.iloc[0]} of {len(fs_method_per_fold)} folds)")
        print(f"Full breakdown:\n{method_counts.to_string()}")

    plot_mean_roc(fold_roc_data)
    print_final_summary(outer_results, run_fairness)

    return outer_results, best_models_per_fold, common_features
