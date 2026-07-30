"""
Non-interactive tests for the generic config logic, using fake data with
sensitive attributes like Education, Ethnicity, and Financial status --
demonstrating the library works outside the schizophrenia/speech domain.
"""
import numpy as np
import pandas as pd

from fair_ml_pipeline import (
    prepare_sensitive_columns,
    build_label_from_config,
    build_X_y_from_config,
)


def make_fake_data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "id": range(n),
        "feature_1": rng.normal(0, 1, n),
        "feature_2": rng.normal(5, 2, n),
        "feature_3": rng.uniform(0, 10, n),
        "outcome_score": rng.normal(50, 15, n),  # a continuous score label
        "Education": rng.choice(["HighSchool", "Bachelors", "Graduate"], n),
        "Ethnicity": rng.choice(["GroupA", "GroupB", "GroupC"], n),
        "Financial_status": rng.uniform(10000, 200000, n),  # numeric sensitive attribute
    })
    return df


def test_prepare_sensitive_columns_categorical_and_numeric():
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
    df_prepared, sensitive_cols = prepare_sensitive_columns(df, specs)

    assert "Education" in sensitive_cols
    assert "Ethnicity" in sensitive_cols
    assert "Financial_status_group" in sensitive_cols
    assert "Financial_status_group" in df_prepared.columns
    # every row should have been binned into one of the labels (no NaN,
    # since bins cover the full range 0-250000)
    assert df_prepared["Financial_status_group"].isna().sum() == 0


def test_build_label_from_config_categorical():
    df = pd.DataFrame({"diagnosis": ["sick", "healthy", "sick", "healthy"]})
    config = {"label_col": "diagnosis", "label_type": "categorical"}
    df_out, y, le = build_label_from_config(df, config)
    assert set(y) == {0, 1}
    assert le is not None
    assert len(y) == len(df)


def test_build_label_from_config_score_cutoff():
    df = pd.DataFrame({"score": [1, 5, 10, 15, 20]})
    config = {"label_col": "score", "label_type": "score", "cutoff": 10, "cutoff_mode": "inclusive"}
    df_out, y, le = build_label_from_config(df, config)
    assert list(y) == [0, 0, 1, 1, 1]
    assert le is None


def test_build_label_from_config_score_median():
    df = pd.DataFrame({"score": [1, 2, 3, 4, 5]})
    config = {"label_col": "score", "label_type": "score", "cutoff": "median", "cutoff_mode": "inclusive"}
    df_out, y, le = build_label_from_config(df, config)
    # median is 3, so >=3 is positive
    assert list(y) == [0, 0, 1, 1, 1]


def test_build_label_from_config_excluded_band():
    df = pd.DataFrame({"score": [1, 4, 5, 6, 10]})
    config = {"label_col": "score", "label_type": "score", "cutoff_mode": "band", "exclude_band": (4, 6)}
    df_out, y, le = build_label_from_config(df, config)
    # rows with score 4,5,6 should be dropped
    assert len(df_out) == 2
    assert list(df_out["score"]) == [1, 10]
    assert list(y) == [0, 1]


def test_build_X_y_from_config_end_to_end():
    df = make_fake_data()
    config = {
        "label_col": "outcome_score",
        "label_type": "score",
        "cutoff": "median",
        "cutoff_mode": "inclusive",
        "sensitive_specs": {
            "Education": {"type": "categorical"},
            "Ethnicity": {"type": "categorical"},
            "Financial_status": {
                "type": "numeric",
                "bins": [0, 30000, 100000, 250000],
                "labels": ["Low", "Middle", "High"],
                "new_col": "Financial_status_group",
            },
        },
        "feature_cols": ["feature_1", "feature_2", "feature_3"],
    }

    X, y, le, sensitive_df, sensitive_cols = build_X_y_from_config(df, config)

    assert list(X.columns) == ["feature_1", "feature_2", "feature_3"]
    assert len(y) == len(X)
    assert set(sensitive_cols) == {"Education", "Ethnicity", "Financial_status_group"}
    assert sensitive_df.shape[0] == X.shape[0]
    assert sensitive_df.shape[1] == 3


if __name__ == "__main__":
    test_prepare_sensitive_columns_categorical_and_numeric()
    test_build_label_from_config_categorical()
    test_build_label_from_config_score_cutoff()
    test_build_label_from_config_score_median()
    test_build_label_from_config_excluded_band()
    test_build_X_y_from_config_end_to_end()
    print("All tests passed.")
