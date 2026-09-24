"""Small reusable interactive prompt helpers, used throughout the pipeline."""


def ask_int_in_range(question, min_val, max_val):
    """Interactively prompts for an integer within [min_val, max_val],
    validating the input."""
    while True:
        raw = input(f"{question} ({min_val}-{max_val}): ").strip()
        try:
            candidate = int(raw)
            if min_val <= candidate <= max_val:
                return candidate
            print(f"Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print(f"Please enter a whole number between {min_val} and {max_val}.")


def ask_yes_no(question):
    """Interactively prompts a yes/no question, validating the input.
    Returns True for yes, False for no."""
    while True:
        raw = input(f"{question} [y/n]: ").strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")


def ask_synthetic_method():
    """Interactively asks which synthetic data generation method to use
    (CTGAN, CopulaGAN, or TVAE, all via the SDV library)."""
    methods = {1: "CTGAN", 2: "CopulaGAN", 3: "TVAE"}
    print("Choose synthetic data generation method:")
    for number, name in methods.items():
        print(f"  {number}: {name}")
    while True:
        raw = input("Method [1/2/3]: ").strip()
        try:
            number = int(raw)
            if number in methods:
                return methods[number]
        except ValueError:
            pass
        print("Please enter 1, 2, or 3.")


def ask_feature_selection_methods():
    """Interactively asks which feature selection method(s) to compete
    against each other per fold. Returns None (meaning 'all methods') or
    a list of method names."""
    from .cv import method_by_number
    print("Choose feature selection method(s):")
    for number, name in method_by_number.items():
        print(f"  {number}: {name}")
    print("Enter numbers separated by commas (e.g. 1,3,4), or 'all' for every method.")
    raw = input("Method(s): ").strip()

    if not raw or raw.lower() == "all":
        return None

    chosen = []
    for piece in raw.split(","):
        piece = piece.strip()
        try:
            number = int(piece)
        except ValueError:
            print(f"Skipping unrecognized entry '{piece}'.")
            continue
        if number not in method_by_number:
            print(f"Skipping unknown method number {number}.")
            continue
        chosen.append(method_by_number[number])

    return chosen if chosen else None


def ask_n_top_features(n_features):
    """Interactively prompts for how many top SHAP features to rank."""
    n_top_features = None
    while n_top_features is None:
        raw = input(f"How many top features to rank (1-{n_features})? ").strip()
        try:
            candidate = int(raw)
            if 1 <= candidate <= n_features:
                n_top_features = candidate
            else:
                print(f"Please enter a number between 1 and {n_features}.")
        except ValueError:
            print(f"Please enter a whole number between 1 and {n_features}.")
    return n_top_features


def _print_numbered_columns(columns, hints=None):
    hints = hints or {}
    for i, col in enumerate(columns, 1):
        hint = f"   <- {hints[col]}" if col in hints else ""
        print(f"  {i}: {col}{hint}")


def _parse_column_selection(raw, columns):
    """Parses '1,3,5-7' and/or column names into a list of column names.
    Returns (selected, errors)."""
    selected, errors = [], []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if piece in columns:
            candidates = [piece]
        elif "-" in piece and all(p.strip().isdigit() for p in piece.split("-", 1)):
            start, end = (int(p) for p in piece.split("-", 1))
            if not (1 <= start <= end <= len(columns)):
                errors.append(piece)
                continue
            candidates = columns[start - 1:end]
        elif piece.isdigit() and 1 <= int(piece) <= len(columns):
            candidates = [columns[int(piece) - 1]]
        else:
            errors.append(piece)
            continue
        for c in candidates:
            if c not in selected:
                selected.append(c)
    return selected, errors


def ask_one_column(question, columns, allow_none=False, none_label="none", hints=None):
    """Asks the user to pick exactly one column from a numbered list, by
    number or by name. If allow_none, entering 0 (or leaving it blank)
    returns None."""
    columns = list(columns)
    print(question)
    if allow_none:
        print(f"  0: {none_label}")
    _print_numbered_columns(columns, hints)
    while True:
        raw = input("Choice (number or column name): ").strip()
        if allow_none and raw in ("", "0"):
            return None
        selected, errors = _parse_column_selection(raw, columns)
        if len(selected) == 1 and not errors:
            return selected[0]
        print("Please pick exactly one column from the list" + (" (or 0 for none)." if allow_none else "."))


def ask_columns(question, columns, allow_empty=True, allow_all=True, hints=None):
    """Asks the user to pick any number of columns from a numbered list.

    Accepts numbers ('1,4'), ranges ('2-6'), column names, a mix of these,
    'all' (if allow_all), or a blank line for none (if allow_empty).
    """
    columns = list(columns)
    print(question)
    _print_numbered_columns(columns, hints)
    tips = ["numbers like 1,3,5", "ranges like 2-6", "column names"]
    if allow_all:
        tips.append("'all'")
    if allow_empty:
        tips.append("blank for none")
    print("Enter " + ", ".join(tips) + ".")
    while True:
        raw = input("Columns: ").strip()
        if not raw:
            if allow_empty:
                return []
            print("Please choose at least one column.")
            continue
        if allow_all and raw.lower() == "all":
            return columns
        selected, errors = _parse_column_selection(raw, columns)
        if errors:
            print(f"Not recognized: {errors}. Please try again.")
            continue
        if not selected and not allow_empty:
            print("Please choose at least one column.")
            continue
        return selected
