"""
Smoke test: runs the full nested CV pipeline end-to-end, non-interactively
(every choice passed explicitly), using fake data with Education,
Ethnicity, and Financial_status as sensitive attributes -- to prove the
library works outside the schizophrenia/speech domain it was originally
built for.
"""
import numpy as np
import pandas as pd

from fair_ml_pipeline import (
    prepare_sensitive_columns,
    build_label_from_config,
    build_classifier,
    nested_cv_normalized_oversampled_featureselected,
)


def make_fake_data(n=100, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "feature_1": rng.normal(0, 1, n),
        "feature_2": rng.normal(5, 2, n),
        "feature_3": rng.uniform(0, 10, n),
        "feature_4": rng.normal(-2, 1, n),
        "outcome_score": rng.normal(50, 15, n),
        "Education": rng.choice(["HighSchool", "Bachelors", "Graduate"], n),
        "Ethnicity": rng.choice(["GroupA", "GroupB"], n),
        "Financial_status": rng.uniform(10000, 200000, n),
    })
    return df


def test_full_pipeline_runs_end_to_end():
    df = make_fake_data()

    specs = {
        "Education": {"type": "categorical"},
        "Ethnicity": {"type": "categorical"},
        "Financial_status": {
            "type": "numeric",
            "bins": [0, 30000, 100000, 250000],
            "labels": ["Low", "Middle", "High"],
            "new_col": "Financial_status_group",
        },
    }
    df, sensitive_cols = prepare_sensitive_columns(df, specs)

    label_config = {"label_col": "outcome_score", "label_type": "score",
                     "cutoff": "median", "cutoff_mode": "inclusive"}
    df, y, le = build_label_from_config(df, label_config)

    feature_cols = ["feature_1", "feature_2", "feature_3", "feature_4"]
    X = df[feature_cols]
    sensitive_df = df[sensitive_cols]
    sensitive_data = [(c, sensitive_df[c].values) for c in sensitive_cols]

    classifiers = [build_classifier("RandomForest")]
    param_grids = [{"n_estimators": [20], "max_depth": [3]}]

    outer_results, best_models_per_fold, common_features = nested_cv_normalized_oversampled_featureselected(
        X, y, sensitive_data, classifiers, param_grids,
        categorical_cols=[],
        numeric_cols=feature_cols,
        best_cv=2,
        n_outer_folds=2,
        run_normalization=True,
        run_oversampling=False,
        run_synthetic_oversampling=False,
        treat_sensitive_as_features=False,
        run_feature_selection=False,
        run_fairness=True,
        run_shap=False,
    )

    assert len(outer_results) == 2
    for res in outer_results:
        assert 0.0 <= res["accuracy"] <= 1.0
        assert "fairness" in res
        assert "Education" in res["fairness"]
        assert "Ethnicity" in res["fairness"]
        assert "Financial_status_group" in res["fairness"]
    assert len(best_models_per_fold) == 2

    print("Full pipeline smoke test passed.")
    print("Fold accuracies:", [r["accuracy"] for r in outer_results])
    print("Fairness attributes checked:", list(outer_results[0]["fairness"].keys()))


if __name__ == "__main__":
    test_full_pipeline_runs_end_to_end()
