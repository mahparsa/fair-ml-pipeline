# fair-ml-pipeline

A fair, explainable nested-cross-validation pipeline for tabular ML.

Unlike most tutorials, nothing here is hardcoded to a specific dataset or domain:

- **Any features** -- point it at whatever columns you want to use as model inputs.
- **Any label** -- a category column (e.g. `"sick"`/`"healthy"`) or a continuous score with a cutoff (e.g. a symptom scale where `>= 10` means "positive").
- **Any sensitive attributes** -- gender, age, ethnicity, education, financial status, or anything else in your data -- added interactively, as many as you want.

It runs nested cross-validation with optional normalization, SMOTENC oversampling, synthetic data generation (CTGAN/CopulaGAN/TVAE), feature selection, fairness metrics (Demographic Parity Difference, Equalized Odds, etc.), and SHAP explainability -- all computed correctly within each fold (no data leakage).

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/YOUR_USERNAME/fair-ml-pipeline.git
```

Or clone and install locally (useful if you're editing it):

```bash
git clone https://github.com/YOUR_USERNAME/fair-ml-pipeline.git
cd fair-ml-pipeline
pip install -e .
```

Optional extras (only needed if you use these specific features):

```bash
pip install "fair-ml-pipeline[boosting] @ git+https://github.com/YOUR_USERNAME/fair-ml-pipeline.git"   # XGBoost, CatBoost
pip install "fair-ml-pipeline[synthetic] @ git+https://github.com/YOUR_USERNAME/fair-ml-pipeline.git"  # CTGAN/CopulaGAN/TVAE (SDV)
pip install "fair-ml-pipeline[all] @ git+https://github.com/YOUR_USERNAME/fair-ml-pipeline.git"        # everything
```

`xgboost`/`catboost`/`sdv` are also installed automatically on the fly, the first time you actually pick one of them at runtime -- so the extras above are optional convenience, not required.

## Quick start

Say you have your own two CSV files:

- `features.csv` -- one row per subject, with an ID column plus whatever feature columns you want to model
- `stats.csv` -- one row per subject, with the same ID column plus sensitive attributes (e.g. `age`, `gender`, `ethnicity`, `education`, `financial_status`) and/or your label

```python
from fair_ml_pipeline import (
    load_and_merge_data, handle_nan,
    configure_pipeline_interactively, build_X_y_from_config,
    build_classifiers_and_grids,
    nested_cv_normalized_oversampled_featureselected,
    save_nested_cv_results, interactive_results_explorer,
)

# 1. Load and merge your own data -- any column names, any ID column
data = load_and_merge_data(
    features_path="features.csv",
    stats_path="stats.csv",
    id_col="uid",              # column name shared by both files
)
data = handle_nan(data, strategy="drop")

# 2. Configure interactively -- you'll be asked:
#    - which column is your label, and whether it's categorical or a
#      score that needs a cutoff
#    - which columns are your sensitive attributes (as many as you want --
#      e.g. gender, ethnicity, education, financial_status), each tagged
#      as categorical or numeric (with bins if numeric)
#    - which columns are your actual model features
config = configure_pipeline_interactively(data)
X, y, le, sensitive_df, sensitive_cols = build_X_y_from_config(data, config)

# 3. Pick your classifier(s) interactively
classifiers, param_grids = build_classifiers_and_grids()

# 4. Run the nested CV pipeline -- also interactive for the remaining
#    choices (normalization, oversampling, synthetic data, feature
#    selection), unless you pass them explicitly to skip the prompts
sensitive_data = list(zip(sensitive_cols, [sensitive_df[c].values for c in sensitive_cols]))

outer_results, best_models_per_fold, common_features = nested_cv_normalized_oversampled_featureselected(
    X, y, sensitive_data, classifiers, param_grids,
    numeric_cols=list(X.columns),
    run_feature_selection=True,
    run_fairness=True,
    run_shap=True,
)

# 5. Save your results, then explore them interactively
save_nested_cv_results(outer_results, best_models_per_fold, common_features)
interactive_results_explorer(outer_results, best_models_per_fold)
```

## Non-interactive usage (scripts, notebooks you want to re-run without prompts)

Every interactive choice can be passed explicitly to skip the prompt -- useful for scripted runs:

```python
config = {
    "label_col": "diagnosis",
    "label_type": "score",
    "cutoff": 10,                      # anything >= 10 is the positive class
    "cutoff_mode": "inclusive",
    "sensitive_specs": {
        "gender": {"type": "categorical"},
        "ethnicity": {"type": "categorical"},
        "financial_status": {
            "type": "numeric",
            "bins": [0, 30000, 70000, 250000],
            "labels": ["Low", "Middle", "High"],
            "new_col": "financial_status_group",
        },
    },
    "feature_cols": ["feature_1", "feature_2", "feature_3"],  # or None to use everything else
}

X, y, le, sensitive_df, sensitive_cols = build_X_y_from_config(data, config)

outer_results, best_models_per_fold, common_features = nested_cv_normalized_oversampled_featureselected(
    X, y, list(zip(sensitive_cols, [sensitive_df[c].values for c in sensitive_cols])),
    classifiers, param_grids,
    numeric_cols=list(X.columns),
    best_cv=5, n_outer_folds=10,
    run_normalization=True, run_oversampling=False,
    run_synthetic_oversampling=False,
    run_feature_selection=True, fs_method="RFECV",
    run_fairness=True, run_shap=True,
)
```

## Teaching mode

If you're using this to teach ML, pass `teach=True` to the main pipeline function. Every time you (or a student) opts into a step -- normalization, oversampling, synthetic data, feature selection, fairness, SHAP -- the notebook will print a plain-language explanation *and* the real source code for that step, right when it happens:

```python
outer_results, best_models_per_fold, common_features = nested_cv_normalized_oversampled_featureselected(
    X, y, sensitive_data, classifiers, param_grids,
    numeric_cols=list(X.columns),
    run_feature_selection=True, run_fairness=True, run_shap=True,
    teach=True,
)
```

## What's included

| Module | Contents |
|---|---|
| `data.py` | Generic CSV loading/merging, NaN handling |
| `config.py` | Interactive label/sensitive-attribute/feature configuration |
| `outliers.py` | LOF-based outlier detection + before/after PCA visualization |
| `models.py` | Classifier construction, hyperparameter grid presets |
| `prompts.py` | Reusable interactive prompt helpers |
| `cv.py` | Feature selection, normalization, SMOTENC + synthetic oversampling, the main nested-CV driver |
| `fairness.py` | Demographic Parity Difference, Equalized Odds, FPR/PPV gaps, Disparate Impact |
| `explain.py` | SHAP explainability + the `explain_step` teaching helper |
| `results.py` | Save/load results, interactive results explorer (Q&A over performance, hyperparameters, feature selection, SHAP, fairness) |

## License

MIT
