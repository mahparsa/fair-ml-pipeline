"""Outlier detection (LOF-based) and before/after visualization -- works
with any list of numeric feature columns."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor


def plot_before_outlier_detection(data, feature_cols):
    """PCA scatter + per-feature boxplots, before outlier removal."""
    X = data[feature_cols].fillna(data[feature_cols].median(numeric_only=True)).values

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(7, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c="steelblue", alpha=0.6, s=25)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.title("Data before outlier detection")
    plt.tight_layout()
    plt.show()

    n_feats = len(feature_cols)
    fig, axes = plt.subplots(1, n_feats, figsize=(4 * n_feats, 4))
    if n_feats == 1:
        axes = [axes]
    for i, col in enumerate(feature_cols):
        axes[i].boxplot(X[:, i], vert=True)
        axes[i].set_title(col)
    fig.suptitle("Feature distributions before outlier detection")
    plt.tight_layout()
    plt.show()


def plot_after_outlier_detection(data, feature_cols, outlier_positions):
    """PCA scatter + per-feature boxplots, after outlier removal (outliers highlighted)."""
    X = data[feature_cols].fillna(data[feature_cols].median(numeric_only=True)).values

    is_outlier = np.zeros(len(X), dtype=bool)
    is_outlier[outlier_positions] = True

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(7, 6))
    plt.scatter(X_pca[~is_outlier, 0], X_pca[~is_outlier, 1], c="steelblue", alpha=0.5, label="Inliers", s=25)
    plt.scatter(X_pca[is_outlier, 0], X_pca[is_outlier, 1], c="crimson", alpha=0.9, label="Outliers", s=60, edgecolor="black")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.title("Data after outlier detection (outliers highlighted)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    n_feats = len(feature_cols)
    fig, axes = plt.subplots(1, n_feats, figsize=(4 * n_feats, 4))
    if n_feats == 1:
        axes = [axes]
    for i, col in enumerate(feature_cols):
        axes[i].boxplot(X[:, i], vert=True)
        jitter = np.random.normal(1, 0.03, size=is_outlier.sum())
        axes[i].scatter(jitter, X[is_outlier, i], color="crimson", s=25, zorder=3, label="Outlier")
        axes[i].set_title(col)
    axes[0].legend()
    fig.suptitle("Feature distributions after outlier detection (outliers highlighted)")
    plt.tight_layout()
    plt.show()


def remove_outliers_lof(data, feature_cols, id_cols, neighbors_list, contamination=0.05, verbose=True):
    """
    Detects outliers using Local Outlier Factor (LOF), run multiple times
    with different neighbor settings, and only removes rows that get
    flagged as outliers consistently across all of them.
    """
    X = data[feature_cols].fillna(data[feature_cols].median(numeric_only=True)).values

    outliers_sets = []
    for n_neighbors in neighbors_list:
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        preds = lof.fit_predict(X)
        outliers_idx = np.where(preds == -1)[0]
        outliers_sets.append(set(outliers_idx))
        if verbose:
            print(f"n_neighbors={n_neighbors}: Detected {len(outliers_idx)} outliers")

    consistent_outliers = set.intersection(*outliers_sets) if outliers_sets else set()
    if verbose:
        print(f"Consistently identified outliers: {len(consistent_outliers)}")

    outlier_positions = sorted(consistent_outliers)
    removed_rows = data.iloc[outlier_positions][id_cols]
    if verbose:
        print("Removed rows (outliers):")
        print(removed_rows)

    mask = np.ones(len(X), dtype=bool)
    mask[outlier_positions] = False
    df_cleaned = data[mask].reset_index(drop=True)

    if verbose:
        print("Shape after outlier removal:", df_cleaned.shape)

    return df_cleaned, removed_rows
