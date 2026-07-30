"""
fair_ml_pipeline
================

A fair, explainable nested-cross-validation pipeline for tabular ML,
built to work with *any* feature set and *any* sensitive attributes --
not tied to a specific domain or dataset.

Typical usage
-------------
    import pandas as pd
    from fair_ml_pipeline import (
        load_and_merge_data, handle_nan,
        configure_pipeline_interactively, build_X_y_from_config,
        build_classifiers_and_grids,
        nested_cv_normalized_oversampled_featureselected,
        save_nested_cv_results, interactive_results_explorer,
    )

    data = load_and_merge_data("features.csv", "stats.csv", id_col="uid")
    data = handle_nan(data, strategy="drop")

    config = configure_pipeline_interactively(data)
    X, y, le, sensitive_df, sensitive_cols = build_X_y_from_config(data, config)

    classifiers, param_grids = build_classifiers_and_grids()

    outer_results, best_models_per_fold, common_features = nested_cv_normalized_oversampled_featureselected(
        X, y, list(zip(sensitive_cols, [sensitive_df[c].values for c in sensitive_cols])),
        classifiers, param_grids,
        numeric_cols=list(X.columns),
        run_feature_selection=True, run_fairness=True, run_shap=True,
    )

    save_nested_cv_results(outer_results, best_models_per_fold, common_features)
    interactive_results_explorer(outer_results, best_models_per_fold)
"""

from .data import load_and_merge_data, sanity_check
from .cleaning import describe_data, plot_numeric_distributions, plot_categorical_distributions, handle_nan, explore_and_clean
from .config import (
    ask_text, ask_choice,
    configure_label_interactively,
    configure_sensitive_features_interactively,
    configure_feature_columns_interactively,
    configure_pipeline_interactively,
    build_label_from_config,
    build_X_y_from_config,
    prepare_sensitive_columns,
    prepare_sensitive_features,
)
from .outliers import (
    plot_before_outlier_detection,
    plot_after_outlier_detection,
    remove_outliers_lof,
)
from .models import (
    classifier_by_number,
    build_classifier,
    param_grid_presets,
    build_classifiers_and_grids,
)
from .prompts import (
    ask_int_in_range,
    ask_yes_no,
    ask_synthetic_method,
    ask_feature_selection_methods,
    ask_n_top_features,
)
from .cv import (
    summarize_metric,
    method_by_number,
    pick_features,
    get_common_features,
    normalize_fold,
    oversample_smotenc,
    oversample_fold,
    synthetic_oversample_fold,
    encode_categorical_fold,
    evaluate_classifiers_inner_cv,
    plot_mean_roc,
    print_final_summary,
    nested_cv_normalized_oversampled_featureselected,
)
from .fairness import compute_fairness, compute_all_sensitive_fairness
from .explain import (
    top_shap_features,
    print_ranked_features,
    run_shap_for_fold,
    explain_step,
    FUNCTION_INFO,
)
from .results import (
    save_nested_cv_results,
    load_nested_cv_results,
    ask_about_results,
    interactive_results_explorer,
)

__version__ = "0.1.0"

__all__ = [
    "load_and_merge_data", "sanity_check", "handle_nan",
    "ask_text", "ask_choice",
    "configure_label_interactively", "configure_sensitive_features_interactively",
    "configure_feature_columns_interactively", "configure_pipeline_interactively",
    "build_label_from_config", "build_X_y_from_config",
    "prepare_sensitive_columns", "prepare_sensitive_features",
    "plot_before_outlier_detection", "plot_after_outlier_detection", "remove_outliers_lof",
    "classifier_by_number", "build_classifier", "param_grid_presets", "build_classifiers_and_grids",
    "ask_int_in_range", "ask_yes_no", "ask_synthetic_method", "ask_feature_selection_methods", "ask_n_top_features",
    "summarize_metric", "method_by_number", "pick_features", "get_common_features",
    "normalize_fold", "oversample_smotenc", "oversample_fold", "synthetic_oversample_fold",
    "encode_categorical_fold", "evaluate_classifiers_inner_cv", "plot_mean_roc", "print_final_summary",
    "nested_cv_normalized_oversampled_featureselected",
    "compute_fairness", "compute_all_sensitive_fairness",
    "top_shap_features", "print_ranked_features", "run_shap_for_fold", "explain_step", "FUNCTION_INFO",
    "save_nested_cv_results", "load_nested_cv_results", "ask_about_results", "interactive_results_explorer",
]
