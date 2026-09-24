"""Generic data loading and cleaning -- works with any one or two CSVs.
The participant ID column (if there is one) is chosen by the user, never assumed."""

import pandas as pd


def id_column_hints(df):
    """Marks columns that *look* like participant IDs (every value unique,
    or 'id' in the name). Only used as hints next to the choices -- the
    user always decides which column, if any, is the ID."""
    hints = {}
    for col in df.columns:
        reasons = []
        is_float = pd.api.types.is_float_dtype(df[col])
        if not is_float and df[col].notna().all() and df[col].is_unique:
            reasons.append("every value is unique")
        name = str(col).lower()
        if name == "id" or name.endswith("id") or name.startswith("id") or "_id" in name or "id_" in name:
            reasons.append("name looks like an ID")
        if reasons:
            hints[col] = "possible ID (" + ", ".join(reasons) + ")"
    return hints


def load_and_merge_data(features_path, stats_path=None, id_col=None, stats_id_col=None,
                         stats_cols=None, interactive=True):
    """
    Loads a features CSV (and optionally a second CSV with additional
    columns, e.g. demographic attributes or the label) and merges them.

    Nothing about the ID column is assumed: if two files need to be merged
    and `id_col` is not given, you are asked which column identifies each
    participant in each file (the names may differ), or whether the files
    should simply be matched row by row because there is no ID column.

    Parameters
    ----------
    features_path : str
        Path or URL to the main features CSV.
    stats_path : str, optional
        Path or URL to a second CSV. If None, only the features CSV is
        loaded and returned as-is (no ID column needed).
    id_col : str, optional
        Name of the participant ID column in the features CSV. If None and
        stats_path is given, you are asked (when interactive=True).
    stats_id_col : str, optional
        Name of the ID column in the second CSV, if different from id_col.
        If None, you are asked (when interactive=True); non-interactively
        it is assumed to be named the same as id_col.
    stats_cols : list of str, optional
        Which columns to keep from the second CSV (besides its ID column).
        If None, keeps every column.
    interactive : bool, default True
        If False, never prompts. Then, to merge two files, pass id_col, or
        pass id_col=False to match the files row by row.

    Returns
    -------
    pd.DataFrame
        The merged dataframe (left join on the ID column, keeping all rows
        from the features CSV), or the two files side by side if matched
        row by row. The chosen ID column name, if any, is stored in
        `df.attrs["id_col"]`.
    """
    from .prompts import ask_one_column, ask_yes_no

    features = pd.read_csv(features_path)
    print(f"Loaded features file: {features.shape[0]} rows, {features.shape[1]} columns")

    if stats_path is None:
        features.attrs["id_col"] = id_col or None
        return features

    stats = pd.read_csv(stats_path)
    print(f"Loaded second file: {stats.shape[0]} rows, {stats.shape[1]} columns")

    # ---- which column identifies participants in the features file? ----
    if id_col is None:
        if not interactive:
            raise ValueError("To merge two files, pass id_col (the participant ID column), "
                             "or id_col=False to match the files row by row.")
        id_col = ask_one_column(
            "\nWhich column in the FEATURES file identifies each participant?",
            list(features.columns), allow_none=True,
            none_label="there is no ID column -- match the two files row by row",
            hints=id_column_hints(features),
        )
    if id_col is False:
        id_col = None

    # ---- no ID: match row by row ----
    if id_col is None:
        if len(features) != len(stats):
            raise ValueError(f"Cannot match row by row: the files have different numbers of rows "
                             f"({len(features)} vs {len(stats)}). An ID column is needed to merge them.")
        if stats_cols is not None:
            stats = stats[list(stats_cols)]
        duplicated = [c for c in stats.columns if c in features.columns]
        if duplicated:
            print(f"Note: columns in both files, kept from the features file only: {duplicated}")
            stats = stats.drop(columns=duplicated)
        merged = pd.concat([features.reset_index(drop=True), stats.reset_index(drop=True)], axis=1)
        print(f"Matched row by row: {merged.shape[0]} rows, {merged.shape[1]} columns")
        merged.attrs["id_col"] = None
        return merged

    if id_col not in features.columns:
        raise ValueError(f"ID column '{id_col}' not found in the features file. Columns: {list(features.columns)}")

    # ---- which column is the ID in the second file? ----
    if stats_id_col is None:
        if interactive:
            if id_col in stats.columns and ask_yes_no(
                    f"The second file also has a column called '{id_col}'. Is that its participant ID?"):
                stats_id_col = id_col
            else:
                stats_id_col = ask_one_column(
                    f"Which column in the SECOND file holds the same participant IDs as '{id_col}'?",
                    list(stats.columns), hints=id_column_hints(stats),
                )
        else:
            stats_id_col = id_col
    if stats_id_col not in stats.columns:
        raise ValueError(f"ID column '{stats_id_col}' not found in the second file. Columns: {list(stats.columns)}")

    if stats_cols is not None:
        keep = [stats_id_col] + [c for c in stats_cols if c != stats_id_col]
        stats = stats[keep]
    if stats_id_col != id_col:
        stats = stats.rename(columns={stats_id_col: id_col})

    for name, frame in (("features", features), ("second", stats)):
        n_dup = frame[id_col].duplicated().sum()
        if n_dup:
            print(f"Warning: {n_dup} duplicated ID value(s) in the {name} file -- the merge may repeat rows.")

    duplicated = [c for c in stats.columns if c in features.columns and c != id_col]
    if duplicated:
        print(f"Note: columns in both files, kept from the features file only: {duplicated}")
        stats = stats.drop(columns=duplicated)

    merged = features.merge(stats, on=id_col, how="left")
    n_unmatched = merged[[c for c in stats.columns if c != id_col]].isna().all(axis=1).sum() if stats.shape[1] > 1 else 0
    print(f"Merged on '{id_col}': {merged.shape[0]} rows, {merged.shape[1]} columns"
          + (f" ({n_unmatched} participant(s) had no match in the second file)" if n_unmatched else ""))
    merged.attrs["id_col"] = id_col
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
