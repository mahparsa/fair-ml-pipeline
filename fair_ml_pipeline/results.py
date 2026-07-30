"""Save/load nested CV results, plus an interactive Q&A explorer for
digging into performance, hyperparameters, feature selection, SHAP, and
fairness -- all generic, no domain-specific assumptions."""

import pickle
from collections import Counter, defaultdict

import numpy as np


def save_nested_cv_results(outer_results, best_models_per_fold, common_features,
                            filepath="nested_cv_results.pkl"):
    with open(filepath, "wb") as f:
        pickle.dump({
            "outer_results": outer_results,
            "best_models_per_fold": best_models_per_fold,
            "common_features": common_features,
        }, f)
    print(f"Results saved to {filepath}")


def load_nested_cv_results(filepath="nested_cv_results.pkl"):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    print(f"Results loaded from {filepath}")
    return data["outer_results"], data["best_models_per_fold"], data["common_features"]


def _top_n_features_local(shap_array, feature_names, n):
    if shap_array.shape[0] == 0:
        return []
    mean_abs = np.abs(shap_array).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:n]
    return [(feature_names[i], float(mean_abs[i])) for i in order]


def _answer_stability(outer_results):
    metrics = ["accuracy", "f1_score", "auc", "balanced_accuracy", "recall"]
    print("\nPerformance stability across folds:")
    for m in metrics:
        values = [res[m] for res in outer_results]
        print(f"  {m}: {np.mean(values):.4f} \u00b1 {np.std(values):.4f}  (range: {min(values):.4f}-{max(values):.4f})")


def _answer_best_worst_fold(outer_results, metric="accuracy"):
    best = max(outer_results, key=lambda r: r[metric])
    worst = min(outer_results, key=lambda r: r[metric])
    print(f"\nRanking folds by {metric}:")
    for res in sorted(outer_results, key=lambda r: r[metric], reverse=True):
        print(f"  Fold {res['fold']}: {metric} = {res[metric]:.4f}")
    print(f"\nBest fold: Fold {best['fold']} ({metric} = {best[metric]:.4f})")
    print(f"Worst fold: Fold {worst['fold']} ({metric} = {worst[metric]:.4f})")


def _answer_precision_recall(outer_results):
    print("\nPrecision vs recall per fold (from confusion matrix, class 1 = positive):")
    for res in outer_results:
        cm = res.get("confusion_matrix")
        if cm is None:
            continue
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"  Fold {res['fold']}: precision = {precision:.4f}, recall = {recall:.4f}")


def _answer_synthetic_vs_performance(outer_results):
    has_data = all("n_synthetic_rows" in res for res in outer_results)
    if not has_data:
        print("\nNot tracked in this run.")
        return
    print("\nSynthetic/oversampled rows added, per fold, vs. that fold's accuracy:")
    rows = []
    for res in outer_results:
        total_added = res["n_synthetic_rows"] + res["n_oversampled_rows"]
        rows.append((res["fold"], res["n_synthetic_rows"], res["n_oversampled_rows"], total_added, res["accuracy"]))
        print(f"  Fold {res['fold']}: +{res['n_synthetic_rows']} synthetic, +{res['n_oversampled_rows']} SMOTENC, "
              f"{total_added} total added -- accuracy = {res['accuracy']:.4f}")
    max_added_fold = max(rows, key=lambda r: r[3])
    min_added_fold = min(rows, key=lambda r: r[3])
    print(f"\nMost rows added: Fold {max_added_fold[0]} (+{max_added_fold[3]}), accuracy = {max_added_fold[4]:.4f}")
    print(f"Fewest rows added: Fold {min_added_fold[0]} (+{min_added_fold[3]}), accuracy = {min_added_fold[4]:.4f}")
    if len(rows) >= 3:
        totals = [r[3] for r in rows]
        accs = [r[4] for r in rows]
        corr = float(np.corrcoef(totals, accs)[0, 1])
        print(f"\nCorrelation between total rows added and accuracy across folds: {corr:+.3f}")


def _answer_outlier_fold(outer_results, metric="accuracy"):
    values = np.array([res[metric] for res in outer_results])
    mean, std = values.mean(), values.std()
    print(f"\nChecking for outlier folds on {metric} (mean={mean:.4f}, std={std:.4f}):")
    if std == 0:
        print("  All folds identical on this metric -- no outliers possible.")
        return
    for res in outer_results:
        z = (res[metric] - mean) / std
        flag = "  <-- outlier (|z| > 1)" if abs(z) > 1 else ""
        print(f"  Fold {res['fold']}: {metric} = {res[metric]:.4f}, z = {z:+.2f}{flag}")


def _answer_overall_verdict(outer_results):
    _answer_stability(outer_results)
    acc_std = np.std([res["accuracy"] for res in outer_results])
    n_folds = len(outer_results)
    print(f"\nBased on {n_folds} outer folds:")
    if acc_std > 0.08:
        print("  Accuracy varies substantially across folds -- treat results as preliminary.")
    else:
        print("  Accuracy is relatively consistent across folds.")


def _answer_best_hyperparams(best_models_per_fold, outer_results=None):
    print("\nBest model and hyperparameters per fold:")
    model_names = []
    fold_accuracies = {}
    if outer_results:
        fold_accuracies = {res["fold"]: res["accuracy"] for res in outer_results}
    param_values = defaultdict(list)
    for fold_idx, info in best_models_per_fold.items():
        print(f"  Fold {fold_idx}: {info['model_name']} -- {info['params']}")
        model_names.append(info["model_name"])
        acc = fold_accuracies.get(fold_idx)
        for param_name, value in info["params"].items():
            param_values[param_name].append((value, acc))
    most_common_model, count = Counter(model_names).most_common(1)[0]
    print(f"\nMost frequently winning model: {most_common_model} ({count}/{len(model_names)} folds)")
    if not outer_results:
        return
    print("\nHyperparameter breakdown:")
    for param_name, values in param_values.items():
        value_counts = Counter(v for v, _ in values)
        parts = [f"{val!r} ({cnt}x)" for val, cnt in value_counts.most_common()]
        print(f"  {param_name}: " + ", ".join(parts))
        numeric_values = [(v, acc) for v, acc in values if isinstance(v, (int, float)) and acc is not None]
        if numeric_values and len(numeric_values) == len(values):
            weights = np.array([acc for _, acc in numeric_values])
            vals = np.array([v for v, _ in numeric_values])
            weighted_avg = float(np.sum(vals * weights) / weights.sum()) if weights.sum() > 0 else float(np.mean(vals))
            print(f"    accuracy-weighted average: {weighted_avg:.3g}")


def _answer_best_fs_method(outer_results):
    methods = [res["fs_method"] for res in outer_results if res.get("fs_method")]
    if not methods:
        print("\nFeature selection wasn't used in this run.")
        return
    counts = Counter(methods)
    print("\nFeature selection method wins per fold:")
    for method, count in counts.most_common():
        print(f"  {method}: won {count}/{len(methods)} folds")
    print(f"\nMost attractive method overall: {counts.most_common(1)[0][0]}")


def _answer_n_features_selected(outer_results):
    counts = [len(res["selected_features"]) for res in outer_results if res.get("selected_features")]
    if not counts:
        print("\nFeature selection wasn't used in this run.")
        return
    print("\nNumber of features selected per fold:")
    for res in outer_results:
        if res.get("selected_features"):
            print(f"  Fold {res['fold']}: {len(res['selected_features'])} features")
    print(f"\nMean: {np.mean(counts):.1f}, std: {np.std(counts):.1f}, range: {min(counts)}-{max(counts)}")


def _answer_features_selected_every_fold(outer_results):
    fold_feature_sets = [set(res["selected_features"]) for res in outer_results if res.get("selected_features")]
    if not fold_feature_sets:
        print("\nFeature selection wasn't used in this run.")
        return
    common = set.intersection(*fold_feature_sets)
    print(f"\nFeatures selected in EVERY fold ({len(common)}): {sorted(common)}")


def _answer_features_selected_sometimes(outer_results):
    fold_feature_sets = {res["fold"]: set(res["selected_features"]) for res in outer_results if res.get("selected_features")}
    if not fold_feature_sets:
        print("\nFeature selection wasn't used in this run.")
        return
    all_features = set.union(*fold_feature_sets.values())
    common = set.intersection(*fold_feature_sets.values())
    sometimes = all_features - common
    print(f"\nFeatures selected in SOME folds but not all ({len(sometimes)}):")
    for feat in sorted(sometimes):
        folds_with = [fold for fold, feats in fold_feature_sets.items() if feat in feats]
        print(f"  {feat}: folds {folds_with}")


def _answer_stable_features(best_models_per_fold, top_n=4):
    fold_top_sets = {}
    feature_appearances = Counter()
    for fold_idx, info in best_models_per_fold.items():
        shap_info = info.get("shap")
        if shap_info is None:
            continue
        top_feats = _top_n_features_local(shap_info["all"], shap_info["features"], top_n)
        top_names = [f for f, _ in top_feats]
        fold_top_sets[fold_idx] = set(top_names)
        feature_appearances.update(top_names)
    n_folds = len(fold_top_sets)
    print(f"\nHow often each feature appeared in the top {top_n} (by mean |SHAP|), across {n_folds} folds:")
    for feat, count in feature_appearances.most_common():
        print(f"  {feat}: top-{top_n} in {count}/{n_folds} folds")
    always_top = set.intersection(*fold_top_sets.values()) if fold_top_sets else set()
    print(f"\nFeatures in the top {top_n} in EVERY fold: {sorted(always_top) if always_top else 'none'}")
    near_consistent = [f for f, c in feature_appearances.items() if c == n_folds - 1 and f not in always_top]
    if near_consistent:
        print(f"Features top-{top_n} in all but ONE fold: {sorted(near_consistent)}")


def _answer_group_features(best_models_per_fold, top_n=4):
    group_feature_counts = {}
    for fold_idx, info in best_models_per_fold.items():
        shap_info = info.get("shap")
        if shap_info is None:
            continue
        for col, groups in shap_info["by_group"].items():
            group_feature_counts.setdefault(col, {})
            for g, shap_subset in groups.items():
                if shap_subset.shape[0] == 0:
                    continue
                top_feats = _top_n_features_local(shap_subset, shap_info["features"], top_n)
                group_feature_counts[col].setdefault(g, Counter())
                group_feature_counts[col][g].update(f for f, _ in top_feats)
    for col, group_counts in group_feature_counts.items():
        print(f"\nTop {top_n} features by sensitive attribute '{col}' (counted across folds):")
        for g, counter in group_counts.items():
            top_for_group = ", ".join(f"{feat} ({count}x)" for feat, count in counter.most_common(top_n))
            print(f"  {g}: {top_for_group}")


def _answer_unstable_features_between_groups(best_models_per_fold, top_n=4):
    for fold_idx, info in best_models_per_fold.items():
        shap_info = info.get("shap")
        if shap_info is None:
            continue
        print(f"\n--- Fold {fold_idx} ---")
        for col, groups in shap_info["by_group"].items():
            group_top_sets = {}
            for g, shap_subset in groups.items():
                if shap_subset.shape[0] == 0:
                    continue
                top_feats = _top_n_features_local(shap_subset, shap_info["features"], top_n)
                group_top_sets[g] = set(f for f, _ in top_feats)
            if len(group_top_sets) < 2:
                continue
            group_names = list(group_top_sets.keys())
            print(f"\n  Sensitive attribute '{col}':")
            for i in range(len(group_names)):
                for j in range(i + 1, len(group_names)):
                    g1, g2 = group_names[i], group_names[j]
                    only_g1 = group_top_sets[g1] - group_top_sets[g2]
                    only_g2 = group_top_sets[g2] - group_top_sets[g1]
                    shared = group_top_sets[g1] & group_top_sets[g2]
                    print(f"    {g1} vs {g2}:")
                    print(f"      shared top features: {sorted(shared) if shared else 'none'}")
                    print(f"      top for '{g1}' only: {sorted(only_g1) if only_g1 else 'none'}")
                    print(f"      top for '{g2}' only: {sorted(only_g2) if only_g2 else 'none'}")


def _answer_shuffled_features_ranking(best_models_per_fold, top_n=4):
    print(f"\nChecking whether any 'shuffled_*'-style control features rank in the top {top_n} per fold:")
    for fold_idx, info in best_models_per_fold.items():
        shap_info = info.get("shap")
        if shap_info is None:
            continue
        top_feats = _top_n_features_local(shap_info["all"], shap_info["features"], top_n)
        shuffled_in_top = [f for f, _ in top_feats if f.startswith("shuffled_")]
        print(f"  Fold {fold_idx}: top-{top_n} = {[f for f, _ in top_feats]}")
        if shuffled_in_top:
            print(f"    Shuffled control features in top-{top_n}: {shuffled_in_top}")


def _get_top_feature_and_direction(shap_info, n=1):
    feature_names = shap_info["features"]
    shap_array = shap_info["all"]
    mean_abs = np.abs(shap_array).mean(axis=0)
    mean_signed = shap_array.mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:n]
    return [(feature_names[i], float(mean_abs[i]), float(mean_signed[i])) for i in order]


def _answer_shap_direction(best_models_per_fold, label_meaning="1 = positive class, 0 = negative class"):
    print(f"\nSHAP direction check (assuming label encoding: {label_meaning})\n")
    top_feature_directions = defaultdict(list)
    for fold_idx, info in best_models_per_fold.items():
        shap_info = info.get("shap")
        if shap_info is None:
            continue
        top = _get_top_feature_and_direction(shap_info, n=1)
        if not top:
            continue
        feat, mean_abs, mean_signed = top[0]
        direction = "toward class 1" if mean_signed > 0 else "toward class 0"
        print(f"  Fold {fold_idx}: top feature = '{feat}', mean SHAP = {mean_signed:+.4f} -> higher values push {direction}")
        top_feature_directions[feat].append((fold_idx, mean_signed))
    print("\nConsistency check -- does each feature's direction flip between folds?")
    for feat, values in top_feature_directions.items():
        signs = set(1 if v > 0 else -1 for _, v in values)
        if len(signs) == 1:
            direction = "toward class 1" if list(signs)[0] > 0 else "toward class 0"
            print(f"  '{feat}': consistent direction across all its appearances ({direction})")
        else:
            print(f"  '{feat}': DIRECTION FLIPS across folds")


def _answer_bias(outer_results, threshold=0.1):
    print(f"\nChecking fairness metrics for |DPD|, |EO|, |FPR_diff|, |PP_diff| >= {threshold}:")
    any_flag = False
    group_favored_counts = Counter()
    for res in outer_results:
        fold_idx = res["fold"]
        fairness = res.get("fairness")
        if not fairness:
            continue
        for col, metrics in fairness.items():
            groups = metrics["groups"]
            for metric in ["DPD", "EO", "FPR_diff", "PP_diff"]:
                if metric in metrics and abs(metrics[metric]) >= threshold:
                    any_flag = True
                    g0, g1 = groups[0], groups[1]
                    favored = g0 if metrics[metric] > 0 else g1
                    print(f"  Fold {fold_idx}, {col}: {metric} = {metrics[metric]:+.3f} (favors '{favored}')")
                    group_favored_counts[(col, favored)] += 1
    if not any_flag:
        print("  No flags at this threshold.")
    else:
        print("\nSummary -- how often each group was favored across all flagged folds/metrics:")
        for (col, group), count in group_favored_counts.most_common():
            print(f"  {col} = '{group}': favored in {count} flagged metric(s)")


def _answer_most_concerning_attribute(outer_results, threshold=0.1):
    attr_flag_counts = Counter()
    for res in outer_results:
        fairness = res.get("fairness")
        if not fairness:
            continue
        for col, metrics in fairness.items():
            for metric in ["DPD", "EO", "FPR_diff", "PP_diff"]:
                if metric in metrics and abs(metrics[metric]) >= threshold:
                    attr_flag_counts[col] += 1
    if not attr_flag_counts:
        print(f"\nNo sensitive attribute crossed the |{threshold}| threshold on any metric.")
        return
    print("\nFlag counts per sensitive attribute:")
    for col, count in attr_flag_counts.most_common():
        print(f"  {col}: {count} flagged metric-instance(s)")
    most_concerning, _ = attr_flag_counts.most_common(1)[0]
    print(f"\nMost fairness-concerning attribute: '{most_concerning}'")


def _answer_favored_disadvantaged_groups(outer_results, threshold=0.1):
    group_favored_counts = Counter()
    for res in outer_results:
        fairness = res.get("fairness")
        if not fairness:
            continue
        for col, metrics in fairness.items():
            groups = metrics["groups"]
            for metric in ["DPD", "EO", "FPR_diff", "PP_diff"]:
                if metric in metrics and abs(metrics[metric]) >= threshold:
                    g0, g1 = groups[0], groups[1]
                    favored = g0 if metrics[metric] > 0 else g1
                    disadvantaged = g1 if metrics[metric] > 0 else g0
                    group_favored_counts[(col, favored)] += 1
                    group_favored_counts[(col, disadvantaged)] += 0
    if not group_favored_counts:
        print(f"\nNo flags at threshold {threshold}.")
        return
    print("\nHow often each group was favored, across all flagged folds/metrics:")
    for (col, group), count in group_favored_counts.most_common():
        print(f"  {col} = '{group}': favored {count} time(s)")


def _answer_fpr_vs_tpr_gap(outer_results):
    print("\nComparing False Positive Rate gap vs. True Positive Rate (EO) gap, per fold/attribute:")
    for res in outer_results:
        fairness = res.get("fairness")
        if not fairness:
            continue
        for col, metrics in fairness.items():
            if "FPR_diff" in metrics and "EO" in metrics:
                larger = "FPR gap" if abs(metrics["FPR_diff"]) > abs(metrics["EO"]) else "TPR (EO) gap"
                print(f"  Fold {res['fold']}, {col}: FPR_diff = {metrics['FPR_diff']:+.3f}, EO = {metrics['EO']:+.3f} -> larger: {larger}")


def _answer_ppv_gap(outer_results, threshold=0.1):
    print(f"\nPositive Predictive Value (precision) gaps by group (|PP_diff| >= {threshold}):")
    any_flag = False
    for res in outer_results:
        fairness = res.get("fairness")
        if not fairness:
            continue
        for col, metrics in fairness.items():
            if "PP_diff" in metrics and abs(metrics["PP_diff"]) >= threshold:
                any_flag = True
                print(f"  Fold {res['fold']}, {col}: PP_diff = {metrics['PP_diff']:+.3f}")
    if not any_flag:
        print("  No PPV gaps at or above threshold.")


def _answer_fairness_volatility(outer_results):
    metric_values = defaultdict(lambda: defaultdict(list))
    for res in outer_results:
        fairness = res.get("fairness")
        if not fairness:
            continue
        for col, metrics in fairness.items():
            for metric in ["DPD", "EO", "FPR_diff", "PP_diff"]:
                if metric in metrics:
                    metric_values[col][metric].append(metrics[metric])
    print("\nFairness metric volatility across folds:")
    attribute_avg_std = {}
    for col, metrics in metric_values.items():
        print(f"\n  '{col}':")
        stds = []
        for metric, values in metrics.items():
            std = float(np.std(values))
            stds.append(std)
            print(f"    {metric}: values={['%.3f' % v for v in values]}, std={std:.3f}")
        attribute_avg_std[col] = float(np.mean(stds))
    if attribute_avg_std:
        most_volatile = max(attribute_avg_std, key=attribute_avg_std.get)
        print(f"\nMost volatile sensitive attribute overall: '{most_volatile}' (average std = {attribute_avg_std[most_volatile]:.3f})")


def _answer_bias_real_or_artifact(outer_results, threshold=0.1):
    _answer_bias(outer_results, threshold)
    print()
    _answer_fairness_volatility(outer_results)
    print("\nA consistent favored direction across folds points toward a real pattern; "
          "high volatility with flipping direction points toward small-sample noise instead.")


def _answer_synthetic_rows_per_fold(outer_results):
    if not all("n_synthetic_rows" in res for res in outer_results):
        print("\nNot tracked in this run.")
        return
    print("\nSynthetic rows added per fold:")
    for res in outer_results:
        print(f"  Fold {res['fold']}: +{res['n_synthetic_rows']} synthetic rows")


def _answer_smotenc_rows_per_fold(outer_results):
    if not all("n_oversampled_rows" in res for res in outer_results):
        print("\nNot tracked in this run.")
        return
    print("\nSMOTENC oversampled rows added per fold:")
    for res in outer_results:
        print(f"  Fold {res['fold']}: +{res['n_oversampled_rows']} SMOTENC rows")


def _answer_not_tracked(*args, **kwargs):
    print("\nNot tracked in this run -- this question needs data the current pipeline doesn't record.")


_TOPIC_HANDLERS = {
    "stability": lambda o, b: _answer_stability(o),
    "best_worst_fold": lambda o, b: _answer_best_worst_fold(o),
    "precision_recall": lambda o, b: _answer_precision_recall(o),
    "synthetic_vs_perf": lambda o, b: _answer_synthetic_vs_performance(o),
    "outlier_fold": lambda o, b: _answer_outlier_fold(o),
    "overall_verdict": lambda o, b: _answer_overall_verdict(o),
    "hyperparams": lambda o, b: _answer_best_hyperparams(b, o),
    "fs_method": lambda o, b: _answer_best_fs_method(o),
    "n_features_selected": lambda o, b: _answer_n_features_selected(o),
    "features_every_fold": lambda o, b: _answer_features_selected_every_fold(o),
    "features_sometimes": lambda o, b: _answer_features_selected_sometimes(o),
    "stable_features": lambda o, b: _answer_stable_features(b),
    "group_features": lambda o, b: _answer_group_features(b),
    "unstable_between_groups": lambda o, b: _answer_unstable_features_between_groups(b),
    "shuffled_ranking": lambda o, b: _answer_shuffled_features_ranking(b),
    "shap_direction": lambda o, b: _answer_shap_direction(b),
    "bias": lambda o, b: _answer_bias(o),
    "most_concerning_attribute": lambda o, b: _answer_most_concerning_attribute(o),
    "favored_disadvantaged": lambda o, b: _answer_favored_disadvantaged_groups(o),
    "fpr_vs_tpr": lambda o, b: _answer_fpr_vs_tpr_gap(o),
    "ppv_gap": lambda o, b: _answer_ppv_gap(o),
    "fairness_volatility": lambda o, b: _answer_fairness_volatility(o),
    "bias_real_or_artifact": lambda o, b: _answer_bias_real_or_artifact(o),
    "synthetic_rows": lambda o, b: _answer_synthetic_rows_per_fold(o),
    "smotenc_rows": lambda o, b: _answer_smotenc_rows_per_fold(o),
    "not_tracked": _answer_not_tracked,
}

QUESTION_MENU = {
    "Performance & stability": [
        ("How stable is performance across folds?", "stability"),
        ("Which fold performed best / worst?", "best_worst_fold"),
        ("What are precision and recall per fold?", "precision_recall"),
        ("Does synthetic/oversampled data correlate with performance?", "synthetic_vs_perf"),
        ("Is there an outlier fold?", "outlier_fold"),
        ("What's the overall performance verdict?", "overall_verdict"),
    ],
    "Hyperparameters & model choice": [
        ("What hyperparameters worked best?", "hyperparams"),
    ],
    "Feature selection": [
        ("Which feature selection method won?", "fs_method"),
        ("How many features were typically selected?", "n_features_selected"),
        ("Which features were selected in every fold?", "features_every_fold"),
        ("Which features were selected in some folds but not others?", "features_sometimes"),
    ],
    "SHAP / feature importance": [
        ("Which features are most stable / top-ranked across folds?", "stable_features"),
        ("Which features are top-ranked per sensitive group?", "group_features"),
        ("Which features are NOT stable between groups?", "unstable_between_groups"),
        ("Are any 'shuffled'-style control features ranking suspiciously high?", "shuffled_ranking"),
        ("How does the top feature affect the outcome (direction)?", "shap_direction"),
    ],
    "Fairness / bias": [
        ("Is there bias toward any group?", "bias"),
        ("Which sensitive attribute shows the most fairness concern?", "most_concerning_attribute"),
        ("Which group is favored / disadvantaged?", "favored_disadvantaged"),
        ("Is the False Positive Rate gap larger than the TPR gap?", "fpr_vs_tpr"),
        ("Does Positive Predictive Value differ between groups?", "ppv_gap"),
        ("How volatile are fairness metrics across folds?", "fairness_volatility"),
        ("Is the observed bias real, or an artifact of small samples?", "bias_real_or_artifact"),
    ],
    "Data / pipeline process": [
        ("How much synthetic data was generated per fold?", "synthetic_rows"),
        ("How many rows were added by SMOTENC per fold?", "smotenc_rows"),
    ],
}


def ask_about_results(outer_results, best_models_per_fold):
    """Free-form question interface -- matches keywords in your question
    to the closest handler."""
    keyword_map = [
        (["not stable", "differ between", "differ across", "unstable"], "unstable_between_groups"),
        (["fairness volatil", "how noisy", "fairness vary"], "fairness_volatility"),
        (["direction", "push", "affect outcome", "higher or lower"], "shap_direction"),
        (["stable", "consistent feature", "top feature", "important feature"], "stable_features"),
        (["hyperparameter", "parameter", "best model", "which model"], "hyperparams"),
        (["feature selection", "selection method"], "fs_method"),
        (["bias", "fair", "fairness", "favor", "discriminat"], "bias"),
        (["group feature", "per group", "by group", "subgroup"], "group_features"),
        (["performance", "how stable"], "stability"),
    ]
    print("Ask a question about your results. Type 'exit' to stop.\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        q = question.lower()
        topic = None
        for keywords, t in keyword_map:
            if any(kw in q for kw in keywords):
                topic = t
                break
        handler = _TOPIC_HANDLERS.get(topic)
        if handler is None:
            print("I couldn't match that to a known topic.")
        else:
            handler(outer_results, best_models_per_fold)
        print()


def interactive_results_explorer(outer_results, best_models_per_fold):
    """Browse questions about your nested CV results by category."""
    from .prompts import ask_yes_no

    categories = list(QUESTION_MENU.keys())
    while True:
        print("\nWhat is your question about?")
        for i, cat in enumerate(categories, 1):
            print(f"  {i}: {cat}")
        print("  exit: stop")
        cat_choice = input("Choose a category: ").strip().lower()
        if cat_choice == "exit":
            break
        try:
            category = categories[int(cat_choice) - 1]
        except (ValueError, IndexError):
            print("Please enter a valid category number.")
            continue
        questions = QUESTION_MENU[category]
        print(f"\n{category} -- pick a question:")
        for i, (q_text, _) in enumerate(questions, 1):
            print(f"  {i}: {q_text}")
        print("  back: choose a different category")
        q_choice = input("Choose a question: ").strip().lower()
        if q_choice == "back":
            continue
        try:
            _, topic = questions[int(q_choice) - 1]
        except (ValueError, IndexError):
            print("Please enter a valid question number.")
            continue
        handler = _TOPIC_HANDLERS.get(topic)
        if handler is None:
            print("\n(Not built yet.)")
        else:
            handler(outer_results, best_models_per_fold)
        again = ask_yes_no("\nAsk another question?")
        if not again:
            break
