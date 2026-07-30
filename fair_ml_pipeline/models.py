"""Classifier construction, hyperparameter grid presets, and interactive
model selection."""

import subprocess
import sys

from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, BaggingClassifier,
)
from sklearn.tree import DecisionTreeClassifier


def _install_and_import(package_name, import_name=None):
    """Pip-installs a package on the fly and returns the imported module."""
    import_name = import_name or package_name
    try:
        return __import__(import_name)
    except ImportError:
        print(f"'{package_name}' not found -- installing it now...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])
        except subprocess.CalledProcessError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                                    "--break-system-packages", package_name])
        return __import__(import_name)


classifier_by_number = {
    1: "RandomForest",
    2: "ExtraTrees",
    3: "AdaBoost",
    4: "GradientBoosting",
    5: "Bagging",
    6: "XGBoost",
    7: "CatBoost",
    8: "DecisionTree",
}


def build_classifier(name):
    if name == "RandomForest":
        return RandomForestClassifier(random_state=42)
    elif name == "ExtraTrees":
        return ExtraTreesClassifier(random_state=42)
    elif name == "AdaBoost":
        return AdaBoostClassifier(random_state=42)
    elif name == "GradientBoosting":
        return GradientBoostingClassifier(random_state=42)
    elif name == "Bagging":
        return BaggingClassifier(random_state=42)
    elif name == "DecisionTree":
        return DecisionTreeClassifier(random_state=42)
    elif name == "XGBoost":
        xgboost = _install_and_import("xgboost")
        return xgboost.XGBClassifier(random_state=42, eval_metric="logloss")
    elif name == "CatBoost":
        catboost = _install_and_import("catboost")
        return catboost.CatBoostClassifier(verbose=0, random_state=42)
    else:
        raise ValueError(f"Unknown classifier: {name!r}. "
                          f"Expected one of: {list(classifier_by_number.values())}")


param_grid_presets = {
    "small": {
        "RandomForest":     {"n_estimators": [30, 60], "max_depth": [3, 5]},
        "ExtraTrees":       {"n_estimators": [30, 60], "max_depth": [3, 5]},
        "AdaBoost":         {"n_estimators": [30, 60], "learning_rate": [0.05, 0.1]},
        "GradientBoosting": {"n_estimators": [30, 60], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
        "Bagging":          {"n_estimators": [10, 30], "max_samples": [0.5, 1.0]},
        "XGBoost":          {"n_estimators": [30, 60], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
        "CatBoost":         {"iterations": [30, 60], "learning_rate": [0.05, 0.1], "depth": [2, 3]},
        "DecisionTree":     {"max_depth": [3, 5], "min_samples_split": [2, 4]},
    },
    "large": {
        "RandomForest": {
            "n_estimators": [10, 30, 50, 70, 100], "max_depth": [2, 3, 4, 5, None],
            "min_samples_split": [2, 3, 4, 5], "min_samples_leaf": [1, 2, 3],
            "max_features": ["sqrt", "log2"],
        },
        "ExtraTrees": {
            "n_estimators": [10, 30, 50, 70, 100], "max_depth": [2, 3, 4, 5, None],
            "min_samples_split": [2, 3, 4, 5], "min_samples_leaf": [1, 2, 3],
            "max_features": ["sqrt", "log2"],
        },
        "AdaBoost": {
            "n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        },
        "GradientBoosting": {
            "n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            "learning_rate": [0.01, 0.05, 0.1], "max_depth": [1, 2, 3, 4, 5],
        },
        "Bagging": {
            "n_estimators": [10, 20, 30, 50, 70, 100],
            "max_samples": [0.5, 0.7, 1.0], "max_features": [0.5, 0.7, 1.0],
        },
        "XGBoost": {
            "n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            "learning_rate": [0.01, 0.05, 0.1], "max_depth": [1, 2, 3, 4, 5],
        },
        "CatBoost": {
            "iterations": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            "learning_rate": [0.01, 0.05, 0.1], "depth": [1, 2, 3, 4, 5],
        },
        "DecisionTree": {
            "max_depth": [2, 3, 4, 5, 6, None],
            "min_samples_split": [2, 3, 4, 5, 6], "min_samples_leaf": [1, 2, 3, 4],
        },
    },
}


def _parse_value_list(raw):
    values = []
    for piece in raw.split(","):
        piece = piece.strip()
        if piece.lower() == "none":
            values.append(None)
            continue
        try:
            values.append(int(piece))
            continue
        except ValueError:
            pass
        try:
            values.append(float(piece))
            continue
        except ValueError:
            pass
        values.append(piece)
    return values


def build_classifiers_and_grids():
    """
    Interactively asks for classifier name(s) and, for each, whether to
    use a 'small'/'large' preset param grid or type in your own values.

    Returns
    -------
    classifiers : list of unfitted estimator instances
    param_grids : list of dicts, same order as classifiers
    """
    print("Choose classifier(s):")
    for number, name in classifier_by_number.items():
        print(f"  {number}: {name}")
    print("Enter numbers separated by commas (e.g. 1,4,6), or 'all' for every classifier.")
    raw = input("Classifier(s): ").strip()

    if not raw or raw.lower() == "all":
        chosen_names = list(classifier_by_number.values())
    else:
        chosen_names = [classifier_by_number[int(n.strip())] for n in raw.split(",")]

    classifiers = []
    param_grids = []

    for name in chosen_names:
        print(f"\n--- {name} ---")
        if name == "AdaBoost":
            print("Note: SHAP will be slower for AdaBoost than for tree-based models.")
        print("Param grid: 1 = small (fast)   2 = large (thorough)   3 = type my own values")
        mode = input("Choice [1/2/3] (default 1): ").strip()

        if mode == "3":
            template = param_grid_presets["large"][name]
            print(f"Enter values for {name}'s hyperparameters (comma-separated). Leave blank to skip a param.")
            grid = {}
            for param_name, example_values in template.items():
                raw_val = input(f"  {param_name} (example: {example_values}): ").strip()
                if raw_val:
                    grid[param_name] = _parse_value_list(raw_val)
            if not grid:
                print(f"No values entered for {name}, falling back to 'small' preset.")
                grid = param_grid_presets["small"][name]
        elif mode == "2":
            grid = param_grid_presets["large"][name]
        else:
            grid = param_grid_presets["small"][name]

        classifiers.append(build_classifier(name))
        param_grids.append(grid)

    return classifiers, param_grids
