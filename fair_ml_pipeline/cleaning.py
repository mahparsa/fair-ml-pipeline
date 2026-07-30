"""Data cleaning and exploration -- descriptive stats, generic
visualizations, and missing-value handling. Not tied to any specific
dataset; works with whatever numeric/categorical columns you give it."""

import matplotlib.pyplot as plt
import seaborn as sns


def describe_data(df):
    """Prints summary statistics about the dataframe, if the user wants them."""
    from .prompts import ask_yes_no

    want_stats = ask_yes_no("Would you like to see summary statistics and general "
                             "description of your data?")
    if not want_stats:
        return

    print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumn types:")
    print(df.dtypes)

    n_missing = df.isna().sum()
    n_missing = n_missing[n_missing > 0]
    if len(n_missing) > 0:
        print("\nMissing values per column:")
        print(n_missing)
    else:
        print("\nNo missing values found.")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        print("\nNumeric column summary:")
        print(df[numeric_cols].describe())

    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    if categorical_cols:
        print("\nCategorical column value counts (top 5 each):")
        for col in categorical_cols:
            print(f"\n  {col}:")
            print(df[col].value_counts().head(5))


def plot_numeric_distributions(df, numeric_cols, max_cols_per_row=4):
    """Histograms for any list of numeric columns."""
    n = len(numeric_cols)
    if n == 0:
        return
    ncols = min(max_cols_per_row, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = axes.flatten() if n > 1 else [axes]
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(df, categorical_cols, max_cols_per_row=3):
    """Bar charts of value counts for any list of categorical columns
    (e.g. sensitive attributes like gender, ethnicity, education)."""
    n = len(categorical_cols)
    if n == 0:
        return
    ncols = min(max_cols_per_row, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    axes = axes.flatten() if n > 1 else [axes]
    for i, col in enumerate(categorical_cols):
        counts = df[col].value_counts()
        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=axes[i])
        axes[i].set_title(col)
        axes[i].tick_params(axis="x", rotation=30)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.show()


def handle_nan(df, strategy=None):
    """
    Handle missing values in df.

    strategy : str, optional
        One of: 'drop', 'fill_mean', 'fill_median', 'fill_zero', 'keep'.
        If not given, shows how many rows/columns are affected and asks.
    """
    n_missing_rows = df.isna().any(axis=1).sum()
    print(f"Rows with at least one missing value: {n_missing_rows} / {len(df)}")

    if strategy is None:
        if n_missing_rows == 0:
            print("No missing values found -- nothing to do.")
            return df
        cols_with_na = df.columns[df.isna().any()].tolist()
        print(f"Columns with missing values: {cols_with_na}")
        print("\nHow do you want to handle missing values?")
        print("  1: Drop rows with any missing values")
        print("  2: Fill numeric columns with their mean")
        print("  3: Fill numeric columns with their median")
        print("  4: Fill with 0")
        print("  5: Leave as-is (do nothing)")
        choice = input("Choice [1-5]: ").strip()
        strategy = {"1": "drop", "2": "fill_mean", "3": "fill_median",
                    "4": "fill_zero", "5": "keep"}.get(choice, "drop")

    print(f"\nUsing strategy: '{strategy}'")

    if strategy == "drop":
        dropped_rows = df[df.isna().any(axis=1)]
        if len(dropped_rows) > 0:
            print(f"\n{len(dropped_rows)} rows will be dropped due to missing values:")
            print(dropped_rows)
        df = df.dropna()
    elif strategy == "fill_mean":
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == "fill_median":
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == "fill_zero":
        df = df.fillna(0)
    elif strategy == "keep":
        pass
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from: "
                          f"'drop', 'fill_mean', 'fill_median', 'fill_zero', 'keep'.")

    print("\nRows after handling NaN:", len(df))
    return df


def explore_and_clean(df, numeric_cols=None, categorical_cols=None):
    """
    One interactive step combining: descriptive stats, visualization,
    and missing-value handling. Asks at each stage whether you want it.
    """
    from .prompts import ask_yes_no

    describe_data(df)

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    want_viz = ask_yes_no("\nWould you like to see visualizations of your data "
                           "(distributions of numeric and categorical columns)?")
    if want_viz:
        if numeric_cols:
            print("\nNumeric feature distributions:")
            plot_numeric_distributions(df, numeric_cols)
        if categorical_cols:
            print("\nCategorical feature distributions:")
            plot_categorical_distributions(df, categorical_cols)

    df = handle_nan(df)
    return df
