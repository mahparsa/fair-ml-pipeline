"""Generic data loading and cleaning -- works with any two CSVs (or one),
merged on a shared ID column. Not tied to any specific dataset or domain."""

import pandas as pd


def load_and_merge_data(features_path, stats_path=None, id_col="uid", stats_id_col=None,
                         stats_cols=None):
    """
    Loads a features CSV (and optionally a second CSV with additional
    columns, e.g. demographic/sensitive attributes), and merges them on
    a shared ID column.

    Parameters
    ----------
    features_path : str
        Path or URL to the main features CSV. Must contain `id_col`.
    stats_path : str, optional
        Path or URL to a second CSV (e.g. sensitive attributes, labels).
        If None, only the features CSV is loaded and returned as-is.
    id_col : str, default "uid"
        Name of the ID column in the features CSV.
    stats_id_col : str, optional
        Name of the ID column in the stats CSV, if different from `id_col`.
        If None, assumes it's also named `id_col`.
    stats_cols : list of str, optional
        Which columns to keep from the stats CSV (besides the ID column).
        If None, keeps every column.

    Returns
    -------
    pd.DataFrame
        The merged dataframe (left join on `id_col`, keeping all rows/columns
        from the features CSV).
    """
    features = pd.read_csv(features_path)

    if stats_path is None:
        return features

    stats = pd.read_csv(stats_path)
    stats_id_col = stats_id_col or id_col

    if stats_cols is not None:
        keep = [stats_id_col] + [c for c in stats_cols if c != stats_id_col]
        stats = stats[keep]

    if stats_id_col != id_col:
        stats = stats.rename(columns={stats_id_col: id_col})

    merged = features.merge(stats, on=id_col, how="left")
    return merged


def sanity_check(features_df, stats_df, merged_df):
    """Optional: quick checks to confirm a merge worked as expected."""
    print("features:", features_df.shape)
    print("stats:", stats_df.shape)
    print("merged:", merged_df.shape)
    print(merged_df.head())


def handle_nan(df, strategy="drop"):
    """
    Handle missing values in df.

    strategy options:
        'drop' -- remove rows with any NaN values
        (extend this function with more strategies as needed, e.g.
        'fill_mean', 'fill_median', 'fill_zero')
    """
    print("Rows before handling NaN:", len(df))
    if strategy == "drop":
        dropped_rows = df[df.isna().any(axis=1)]
        if len(dropped_rows) > 0:
            print(f"\n{len(dropped_rows)} rows will be dropped due to missing values:")
            print(dropped_rows)
        df = df.dropna()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    print("\nRows after handling NaN:", len(df))
    return df
