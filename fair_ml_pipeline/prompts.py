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
