"""Interactive configuration for label handling, sensitive attributes, and
feature selection -- works with any dataframe, any domain."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def ask_text(question, valid_options=None):
    """Interactively asks for free-text input, optionally validating
    against a list of valid options (case-sensitive)."""
    while True:
        raw = input(f"{question}: ").strip()
        if valid_options is None or raw in valid_options or raw == "":
            return raw
        print(f"'{raw}' not recognized. Valid options: {valid_options}")


def ask_choice(question, options):
    """Interactively asks the user to pick one option from a numbered list."""
    print(question)
    for i, opt in enumerate(options, 1):
        print(f"  {i}: {opt}")
    while True:
        raw = input(f"Choice [1-{len(options)}]: ").strip()
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        except ValueError:
            pass
        print(f"Please enter a number between 1 and {len(options)}.")


def prepare_sensitive_columns(df, specs):
    """
    Turns raw columns into ready-to-use sensitive group columns.
    Numeric columns get binned into groups; categorical columns are
    used as-is (just checked for missing values).

    specs : dict of {column_name: {'type': 'categorical'}} or
        {'type': 'numeric', 'bins': [...], 'labels': [...], 'new_col': str}
    """
    df_prepared = df.copy()
    sensitive_cols = []

    for col, cfg in specs.items():
        col_type = cfg.get("type", "categorical")

        if col_type == "numeric":
            if col not in df_prepared.columns:
                raise ValueError(f"Numeric column '{col}' not found in dataframe.")
            bins = cfg["bins"]
            labels = cfg["labels"]
            if len(labels) != len(bins) - 1:
                raise ValueError(
                    f"'{col}': labels must have exactly len(bins)-1 entries "
                    f"({len(bins) - 1}), got {len(labels)}."
                )
            new_col = cfg.get("new_col", f"{col}_group")
            df_prepared[new_col] = pd.cut(df_prepared[col], bins=bins, labels=labels, right=False)
            n_missing = df_prepared[new_col].isna().sum()
            if n_missing:
                print(f"Warning: {n_missing} rows fell outside the bins for '{col}' -> '{new_col}' (NaN).")
            sensitive_cols.append(new_col)

        elif col_type == "categorical":
            if col not in df_prepared.columns:
                raise ValueError(f"Categorical column '{col}' not found in dataframe.")
            n_missing = df_prepared[col].isna().sum()
            if n_missing:
                print(f"Warning: '{col}' has {n_missing} missing values -- consider filling or "
                      f"dropping these rows before running fairness analysis.")
            sensitive_cols.append(col)
        else:
            raise ValueError(f"Unknown type '{col_type}' for '{col}'. Use 'numeric' or 'categorical'.")

    return df_prepared, sensitive_cols


def prepare_sensitive_features(df, sensitive_cols=None, group_labels=None):
    """Validates sensitive-feature config and builds the data dict used
    downstream for fairness/explainability analysis."""
    if sensitive_cols is None:
        sensitive_cols = []
    if group_labels is None:
        group_labels = {}
    missing = [c for c in sensitive_cols if c not in df.columns]
    if missing:
        raise ValueError(f"sensitive_cols not found in dataframe: {missing}")
    sensitive_data = {col: np.array(df[col].values) for col in sensitive_cols}
    return sensitive_cols, group_labels, sensitive_data


def configure_label_interactively(df):
    """
    Asks the user which column is the label, and whether it should be
    used as-is (categorical) or converted from a continuous score using
    a cutoff.

    Returns
    -------
    dict with keys: 'label_col', 'label_type' ('categorical' or 'score'),
    and (if 'score') 'cutoff'/'cutoff_mode' or 'exclude_band'.
    """
    print("\n--- Label configuration ---")
    label_col = ask_text(
        f"Which column is your label/outcome? Available columns: {list(df.columns)}",
        valid_options=list(df.columns),
    )

    label_type = ask_choice(
        f"Is '{label_col}' already a category (e.g. 'sick'/'healthy'), or a continuous "
        f"score that needs a cutoff to become a category (e.g. a symptom severity score)?",
        ["categorical", "score"],
    )

    config = {"label_col": label_col, "label_type": label_type}

    if label_type == "score":
        print(f"\n'{label_col}' summary statistics:")
        print(df[label_col].describe())

        cutoff_choice = ask_choice(
            "How should the cutoff be set?",
            ["Use the median as the cutoff", "Enter a specific cutoff value",
             "Exclude a middle band (drop borderline scores)"],
        )

        if cutoff_choice == "Use the median as the cutoff":
            config["cutoff"] = "median"
            config["cutoff_mode"] = "inclusive"
        elif cutoff_choice == "Enter a specific cutoff value":
            while True:
                raw = input("Cutoff value (scores >= this are the positive class): ").strip()
                try:
                    config["cutoff"] = float(raw)
                    break
                except ValueError:
                    print("Please enter a number.")
            config["cutoff_mode"] = "inclusive"
        else:
            while True:
                raw_low = input("Lower bound of the excluded middle band: ").strip()
                raw_high = input("Upper bound of the excluded middle band: ").strip()
                try:
                    config["exclude_band"] = (float(raw_low), float(raw_high))
                    break
                except ValueError:
                    print("Please enter numbers.")
            config["cutoff_mode"] = "band"

    return config


def configure_sensitive_features_interactively(df, exclude_cols=None):
    """
    Lets the user add as many sensitive attributes as they want, one at a
    time, each tagged as categorical or numeric (with bins if numeric).

    Returns
    -------
    dict of {column_name: spec_dict}, in the format expected by
    prepare_sensitive_columns's `specs` argument.
    """
    exclude_cols = exclude_cols or []
    print("\n--- Sensitive attributes ---")
    print("Add the attributes you want to check fairness for (e.g. gender, age, "
          "ethnicity, education, financial status, or anything else in your data).")

    available = [c for c in df.columns if c not in exclude_cols]
    specs = {}

    while True:
        col = ask_text(
            f"Sensitive attribute column name (available: {available}), or leave blank to finish"
        )
        if not col:
            break
        if col not in df.columns:
            print(f"'{col}' not found in the dataframe. Available columns: {available}")
            continue

        col_type = ask_choice(
            f"Is '{col}' categorical (e.g. 'Male'/'Female', ethnicity groups) "
            f"or numeric (e.g. income, a continuous score)?",
            ["categorical", "numeric"],
        )

        if col_type == "categorical":
            specs[col] = {"type": "categorical"}
            print(f"'{col}' values found: {sorted(df[col].dropna().unique().tolist())}")
        else:
            print(f"\n'{col}' summary statistics:")
            print(df[col].describe())
            from .prompts import ask_int_in_range
            n_bins = ask_int_in_range(f"How many groups do you want to bin '{col}' into?", 2, 8)
            bins = []
            print(f"Enter {n_bins + 1} bin edges, from lowest to highest "
                  f"(e.g. for income in 3 groups: 0, 30000, 70000, 200000)")
            for i in range(n_bins + 1):
                while True:
                    raw = input(f"  Edge {i + 1}: ").strip()
                    try:
                        bins.append(float(raw))
                        break
                    except ValueError:
                        print("Please enter a number.")
            labels = []
            for i in range(n_bins):
                label = ask_text(f"  Label for group {i + 1} ({bins[i]} to {bins[i+1]})")
                labels.append(label)
            new_col = f"{col}_group"
            specs[col] = {"type": "numeric", "bins": bins, "labels": labels, "new_col": new_col}

        from .prompts import ask_yes_no
        again = ask_yes_no("Add another sensitive attribute?")
        if not again:
            break

    return specs


def configure_feature_columns_interactively(df, exclude_cols):
    """Lets the user pick which columns are actual model features."""
    print("\n--- Feature columns ---")
    available = [c for c in df.columns if c not in exclude_cols]
    print(f"Columns available to use as features (label/sensitive columns already excluded): {available}")

    choice = ask_choice("Which features do you want to use?",
                         ["Use all available columns", "Type a specific list"])

    if choice == "Use all available columns":
        return available

    raw = input("Enter feature column names, comma-separated: ").strip()
    chosen = [c.strip() for c in raw.split(",") if c.strip()]
    invalid = [c for c in chosen if c not in available]
    if invalid:
        print(f"Warning: these columns weren't found and will be skipped: {invalid}")
    return [c for c in chosen if c in available]


def configure_pipeline_interactively(df):
    """
    Runs the full interactive configuration: label, sensitive attributes,
    and feature columns. Works for any dataframe/domain.

    Returns
    -------
    config : dict with keys 'label_col', 'label_type', 'cutoff'/'exclude_band'
        (if applicable), 'sensitive_specs', 'feature_cols'
    """
    label_config = configure_label_interactively(df)

    exclude_for_sensitive = [label_config["label_col"]]
    sensitive_specs = configure_sensitive_features_interactively(df, exclude_cols=exclude_for_sensitive)

    exclude_for_features = [label_config["label_col"]] + list(sensitive_specs.keys())
    feature_cols = configure_feature_columns_interactively(df, exclude_cols=exclude_for_features)

    config = {**label_config, "sensitive_specs": sensitive_specs, "feature_cols": feature_cols}

    print("\n" + "=" * 60)
    print("Configuration summary:")
    print(f"  Label column: {config['label_col']} ({config['label_type']})")
    print(f"  Sensitive attributes: {list(sensitive_specs.keys())}")
    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")
    print("=" * 60)

    return config


def build_label_from_config(df, config):
    """Builds the binary label array y based on the label configuration."""
    label_col = config["label_col"]
    label_type = config["label_type"]

    if label_type == "categorical":
        le = LabelEncoder()
        y = le.fit_transform(df[label_col].values)
        print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
        return df, y, le

    # label_type == "score"
    if config["cutoff_mode"] == "band":
        low, high = config["exclude_band"]
        n_before = len(df)
        df = df[(df[label_col] < low) | (df[label_col] > high)].reset_index(drop=True)
        print(f"Dropped {n_before - len(df)} rows with '{label_col}' in the excluded band [{low}, {high}]")
        cutoff = (low + high) / 2
    elif config["cutoff"] == "median":
        cutoff = df[label_col].median()
        print(f"Using median as cutoff: {cutoff}")
    else:
        cutoff = config["cutoff"]

    y = (df[label_col] >= cutoff).astype(int).values
    print(f"Label distribution after cutoff ({cutoff}): {dict(zip(*np.unique(y, return_counts=True)))}")
    return df, y, None


def build_X_y_from_config(df, config):
    """
    Fully generic feature/label builder -- uses the interactively (or
    programmatically) configured label, sensitive attributes, and feature
    columns instead of any hardcoded column names.

    Returns
    -------
    X, y, le, sensitive_df, sensitive_cols
    """
    df, sensitive_cols = prepare_sensitive_columns(df, config["sensitive_specs"])
    df, y, le = build_label_from_config(df, config)

    X = df[config["feature_cols"]]
    sensitive_df = df[sensitive_cols]

    print(f"\nFinal X shape: {X.shape}, y shape: {y.shape}")
    return X, y, le, sensitive_df, sensitive_cols
