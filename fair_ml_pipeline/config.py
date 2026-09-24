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


def configure_id_column_interactively(df):
    """
    Asks which column (if any) identifies each participant. Nothing is
    assumed: the column can have any name, and "no ID column" is a valid
    answer. The chosen column is only used to keep track of participants;
    it is never used as a feature, label, or demographic attribute.

    Returns the column name, or None.
    """
    from .data import id_column_hints
    from .prompts import ask_one_column, ask_yes_no

    print("\n--- Participant ID ---")
    chosen_at_load = df.attrs.get("id_col")
    if chosen_at_load in df.columns and ask_yes_no(
            f"When loading the data you chose '{chosen_at_load}' as the participant ID. Keep it?"):
        return chosen_at_load
    return ask_one_column(
        "Which column identifies each participant?",
        list(df.columns), allow_none=True,
        none_label="there is no ID column in this data",
        hints=id_column_hints(df),
    )


def configure_label_interactively(df, exclude_cols=None):
    """
    Asks the user which column is the label, and whether it should be
    used as-is (categorical) or converted from a continuous score using
    a cutoff.

    exclude_cols : columns not offered as the label (e.g. the ID column).

    Returns
    -------
    dict with keys: 'label_col', 'label_type' ('categorical' or 'score'),
    and (if 'score') 'cutoff'/'cutoff_mode' or 'exclude_band'.
    """
    from .prompts import ask_one_column

    exclude_cols = [c for c in (exclude_cols or []) if c is not None]
    available = [c for c in df.columns if c not in exclude_cols]
    hints = {c: f"{df[c].nunique()} distinct values" for c in available}

    print("\n--- Label configuration ---")
    label_col = ask_one_column("Which column is your label/outcome (what the model should predict)?",
                               available, hints=hints)

    n_unique = df[label_col].nunique()
    if n_unique == 2:
        print(f"'{label_col}' has 2 distinct values: {sorted(df[label_col].dropna().unique().tolist(), key=str)}")
    elif not pd.api.types.is_numeric_dtype(df[label_col]):
        print(f"Warning: '{label_col}' is not numeric and has {n_unique} distinct values; "
              f"this pipeline expects a binary (two-class) label.")

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


def _ask_numeric_binning(df, col):
    """Asks how to turn a numeric demographic column (e.g. age, income)
    into groups. Returns (bins, labels)."""
    from .prompts import ask_int_in_range

    print(f"\n'{col}' summary statistics:")
    print(df[col].describe())
    n_bins = ask_int_in_range(f"How many groups do you want to split '{col}' into?", 2, 8)
    how = ask_choice(f"How should the {n_bins} groups be defined?",
                     ["Automatically, with about the same number of participants in each group",
                      "I'll type the group boundaries myself"])

    if how.startswith("Automatically"):
        edges = np.unique(np.quantile(df[col].dropna(), np.linspace(0, 1, n_bins + 1)))
        if len(edges) - 1 < n_bins:
            print(f"Only {len(edges) - 1} distinct groups are possible for '{col}' (many repeated values).")
        edges = [float(e) for e in edges]
        edges[-1] = np.nextafter(edges[-1], np.inf)  # include the maximum (bins are [low, high))
        bins = edges
        labels = [f"{bins[i]:g} to <{bins[i + 1]:g}" if i < len(bins) - 2 else f"{bins[i]:g} to {df[col].max():g}"
                  for i in range(len(bins) - 1)]
        print(f"Groups for '{col}': {labels}")
        if ask_choice("Use these group names?", ["Yes", "No, let me rename them"]) == "Yes":
            return bins, labels
    else:
        bins = []
        print(f"Enter {n_bins + 1} boundaries, from lowest to highest "
              f"(e.g. for 3 age groups: 18, 30, 50, 90). Each group includes its lower boundary.")
        for i in range(n_bins + 1):
            while True:
                raw = input(f"  Boundary {i + 1}: ").strip()
                try:
                    value = float(raw)
                    if bins and value <= bins[-1]:
                        print("Each boundary must be larger than the previous one.")
                        continue
                    bins.append(value)
                    break
                except ValueError:
                    print("Please enter a number.")

    labels = []
    for i in range(len(bins) - 1):
        label = ask_text(f"  Name for group {i + 1} ({bins[i]:g} to {bins[i + 1]:g})") or f"group_{i + 1}"
        labels.append(label)
    return bins, labels


def configure_sensitive_features_interactively(df, exclude_cols=None):
    """
    Lets the user choose which demographic / sensitive attributes to
    analyse (any columns, any number -- including none), then asks for
    each whether it is categorical or numeric (numeric ones are split
    into groups).

    Returns
    -------
    dict of {column_name: spec_dict}, in the format expected by
    prepare_sensitive_columns's `specs` argument.
    """
    from .prompts import ask_columns

    exclude_cols = [c for c in (exclude_cols or []) if c is not None]
    available = [c for c in df.columns if c not in exclude_cols]
    hints = {}
    for c in available:
        n = df[c].nunique()
        kind = "numeric" if pd.api.types.is_numeric_dtype(df[c]) else "text"
        hints[c] = f"{kind}, {n} distinct values"

    print("\n--- Demographic / sensitive attributes ---")
    print("Choose the columns you want to check fairness for. Any column can be used, "
          "and you can choose none.")
    chosen = ask_columns("Which demographic attributes do you want to analyse?",
                         available, allow_empty=True, allow_all=False, hints=hints)

    specs = {}
    for col in chosen:
        values = df[col].dropna().unique()
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if is_numeric and len(values) > 10:
            suggestion = "numeric"
        else:
            suggestion = "categorical"
        options = ["categorical", "numeric"] if suggestion == "categorical" else ["numeric", "categorical"]
        col_type = ask_choice(
            f"\nIs '{col}' categorical (a fixed set of groups) or numeric (a quantity to be split into groups)? "
            f"It has {len(values)} distinct values; suggested: {suggestion}.",
            options,
        )
        if col_type == "categorical":
            specs[col] = {"type": "categorical"}
            if len(values) > 20:
                print(f"Warning: '{col}' has {len(values)} groups; fairness metrics will be noisy with that many.")
            else:
                print(f"'{col}' groups: {sorted(values.tolist(), key=str)}")
        else:
            if not is_numeric:
                print(f"'{col}' is not numeric, so it will be used as categorical.")
                specs[col] = {"type": "categorical"}
                continue
            bins, labels = _ask_numeric_binning(df, col)
            specs[col] = {"type": "numeric", "bins": bins, "labels": labels, "new_col": f"{col}_group"}

    return specs


def configure_feature_columns_interactively(df, exclude_cols):
    """Lets the user pick which columns are model inputs. Columns already
    used as the ID, the label, or a demographic attribute are excluded."""
    from .prompts import ask_columns

    exclude_cols = [c for c in (exclude_cols or []) if c is not None]
    available = [c for c in df.columns if c not in exclude_cols]
    numeric = [c for c in available if pd.api.types.is_numeric_dtype(df[c])]
    non_numeric = [c for c in available if c not in numeric]

    print("\n--- Feature columns ---")
    print(f"Excluded (ID / label / demographic attributes): {exclude_cols}")
    if not available:
        raise ValueError("No columns are left to use as features.")

    options = [f"All remaining columns ({len(available)})"]
    if non_numeric and numeric:
        options.append(f"All remaining numeric columns ({len(numeric)})")
    options.append("Let me choose")
    choice = ask_choice("Which columns should the model use as features?", options)

    if choice.startswith("All remaining numeric"):
        chosen = numeric
    elif choice.startswith("All remaining columns"):
        chosen = available
    else:
        hints = {c: "not numeric" for c in non_numeric}
        chosen = ask_columns("Pick the feature columns:", available,
                             allow_empty=False, allow_all=True, hints=hints)

    text_cols = [c for c in chosen if c in non_numeric]
    if text_cols:
        print(f"Note: these features are not numeric and must be passed as categorical_cols "
              f"to the pipeline: {text_cols}")
    return chosen


def configure_pipeline_interactively(df, id_col="ask"):
    """
    Runs the full interactive configuration, in this order:
      1. which column (if any) is the participant ID,
      2. which column is the label,
      3. which demographic / sensitive attributes to analyse,
      4. which columns are features.
    Every column used in one step is left out of the later steps, so the
    ID is never used as a feature. Works for any dataframe/domain.

    id_col : "ask" (default) to ask, or a column name / None to skip the
        question.

    Returns
    -------
    config : dict with keys 'id_col', 'label_col', 'label_type',
        'cutoff'/'exclude_band' (if applicable), 'sensitive_specs',
        'feature_cols'
    """
    if id_col == "ask":
        id_col = configure_id_column_interactively(df)
    elif id_col is not None and id_col not in df.columns:
        raise ValueError(f"id_col '{id_col}' not found in dataframe.")

    label_config = configure_label_interactively(df, exclude_cols=[id_col])

    sensitive_specs = configure_sensitive_features_interactively(
        df, exclude_cols=[id_col, label_config["label_col"]])

    exclude_for_features = [id_col, label_config["label_col"]] + list(sensitive_specs.keys())
    feature_cols = configure_feature_columns_interactively(df, exclude_cols=exclude_for_features)

    config = {"id_col": id_col, **label_config, "sensitive_specs": sensitive_specs, "feature_cols": feature_cols}

    print("\n" + "=" * 60)
    print("Configuration summary:")
    print(f"  Participant ID column: {id_col if id_col else '(none)'}")
    print(f"  Label column: {config['label_col']} ({config['label_type']})")
    print(f"  Demographic attributes: {list(sensitive_specs.keys()) or '(none)'}")
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


def build_X_y_from_config(df, config, return_ids=False):
    """
    Fully generic feature/label builder -- uses the interactively (or
    programmatically) configured label, sensitive attributes, and feature
    columns instead of any hardcoded column names.

    config may contain 'id_col' (a column name, or None/absent if there is
    no participant ID). The ID column is never allowed into X.

    Returns
    -------
    X, y, le, sensitive_df, sensitive_cols
        plus `ids` (a Series aligned with X, or None if there is no ID
        column) as a sixth value when return_ids=True.
    """
    id_col = config.get("id_col")
    feature_cols = list(config["feature_cols"])
    if id_col is not None:
        if id_col not in df.columns:
            raise ValueError(f"id_col '{id_col}' not found in dataframe.")
        if id_col in feature_cols:
            print(f"Warning: removing the participant ID column '{id_col}' from the features.")
            feature_cols.remove(id_col)
        if id_col == config["label_col"] or id_col in config["sensitive_specs"]:
            raise ValueError(f"The participant ID column '{id_col}' cannot also be the label "
                             f"or a demographic attribute.")

    df, sensitive_cols = prepare_sensitive_columns(df, config["sensitive_specs"])
    df, y, le = build_label_from_config(df, config)
    df = df.reset_index(drop=True)

    X = df[feature_cols]
    sensitive_df = df[sensitive_cols]

    print(f"\nFinal X shape: {X.shape}, y shape: {y.shape}")
    if return_ids:
        ids = df[id_col] if id_col is not None else None
        return X, y, le, sensitive_df, sensitive_cols, ids
    return X, y, le, sensitive_df, sensitive_cols
