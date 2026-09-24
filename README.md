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

- `features.csv` -- one row per participant, with whatever feature columns you want to model
- `stats.csv` (optional) -- one row per participant, with demographic attributes and/or your label

Nothing about the column names is assumed. The participant ID column can be called anything
(and be named differently in the two files), or there may be no ID at all -- you are asked.

```python
from fair_ml_pipeline import (
    load_and_merge_data, handle_nan,
    configure_pipeline_interactively, build_X_y_from_config,
    build_classifiers_and_grids,
    nested_cv_normalized_oversampled_featureselected,
    save_nested_cv_results, interactive_results_explorer,
)

# 1. Load and merge your own data. You'll be asked which column is the
#    participant ID in each file (names may differ), or to match the
#    files row by row if there is no ID column.
data = load_and_merge_data(features_path="features.csv", stats_path="stats.csv")
data = handle_nan(data, strategy="drop")

# 2. Configure interactively, picking columns from numbered lists:
#    - which column (if any) identifies participants -- it is then kept
#      out of the label, demographic, and feature choices
#    - which column is your label, and whether it's categorical or a
#      score that needs a cutoff
#    - which demographic attributes to analyse for fairness (any columns,
#      any number, or none), each categorical or numeric (numeric ones are
#      split into groups automatically or at boundaries you type)
#    - which columns are your model features
config = configure_pipeline_interactively(data)
X, y, le, sensitive_df, sensitive_cols = build_X_y_from_config(data, config)
# (add return_ids=True to also get the participant IDs aligned with X)

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
data = load_and_merge_data("features.csv", "stats.csv",
                           id_col="participant_code", stats_id_col="subject",  # or id_col=False: match rows
                           interactive=False)

config = {
    "id_col": "participant_code",      # or None if there is no ID column
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

## Tune each model separately, then pick one (`nested_cv_per_model`)

The original driver tunes every classifier inside each outer fold and keeps
whichever wins that fold, so fold 1 might end up with XGBoost and fold 2 with
RandomForest. If you'd rather get **one model type with its own best
hyperparameters**, use `nested_cv_per_model`. It takes the same arguments and:

1. preprocesses each outer fold once, so every model sees identical folds;
2. for each model in turn (XGBoost, then RandomForest, ...) tunes it on the
   inner CV, scores it on the outer folds, and prints its full report:
   per-fold best params, metrics, fairness, ROC;
3. ranks the models by their mean outer-fold `selection_metric`;
4. refits each model (or only the winner) with GridSearchCV on all the data,
   so whichever model is selected comes with its own final tuned parameters.

```python
from fair_ml_pipeline import (
    nested_cv_per_model, save_per_model_results, interactive_per_model_explorer,
    save_nested_cv_results,
)

results = nested_cv_per_model(
    X, y, sensitive_data, classifiers, param_grids,
    numeric_cols=list(X.columns),
    run_feature_selection=True, run_fairness=True, run_shap=True,
    selection_metric="balanced_accuracy",  # accuracy | f1_score | auc | balanced_accuracy | recall
    refit_final="all",                     # "all", "best", or None
    shap_for="best",                       # SHAP only for the selected model (or "all")
)

print(results["comparison"])        # one row per model, best first
print(results["best_model_name"])   # e.g. "XGBClassifier"
print(results["best_params"])       # that model's own tuned hyperparameters
final_model = results["best_model"] # fitted on all data, using results["final_features"]

xgb = results["models"]["XGBClassifier"]           # any model's full results
print(xgb["final_params"], xgb["summary"]["auc"])

save_per_model_results(results)
interactive_per_model_explorer(results)             # pick a model, then explore it

# or plug one model into the existing tools:
winner = results["models"][results["best_model_name"]]
save_nested_cv_results(winner["outer_results"], winner["best_models_per_fold"], results["common_features"])
```

Note: `final_model` expects the same preprocessing as the final refit
(the columns in `results["final_features"]`, normalized/encoded the same way).
The honest performance estimate is the nested-CV mean ± SD, not the CV score
printed for the final refit.

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
