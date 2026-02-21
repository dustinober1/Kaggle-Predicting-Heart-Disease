"""
visualization.py — Reusable plotting helpers. All figures are saved to results/figures/.
"""
from __future__ import annotations

import pathlib
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

FIGURES_DIR = pathlib.Path(__file__).resolve().parent.parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Project-wide style
plt.rcParams.update({
    "figure.dpi": 120,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})
PALETTE = "Set2"


def save_fig(name: str, fig: Optional[plt.Figure] = None, dpi: int = 150) -> pathlib.Path:
    """Save figure to results/figures/<name>.png and return the path."""
    fig = fig or plt.gcf()
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_target_distribution(y: pd.Series, title: str = "Target Distribution") -> plt.Figure:
    counts = y.value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.index.astype(str), counts.values, color=sns.color_palette(PALETTE)[:2])
    for bar, cnt in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{cnt:,}\n({cnt/len(y)*100:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    return fig


def plot_roc_curves(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str = "ROC Curves",
) -> plt.Figure:
    """Plot multiple ROC curves.

    Parameters
    ----------
    results : dict mapping model_name → (fpr_array, tpr_array, auc_score)
    """
    from sklearn.metrics import auc
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = sns.color_palette(PALETTE, n_colors=len(results))
    for (name, (fpr, tpr, auc_score)), color in zip(results.items(), colors):
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})", color=color, lw=1.8)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[str] = ("Absence", "Presence"),
    title: str = "Confusion Matrix",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    ax.set_title(title)
    return fig


def plot_feature_importance(
    importances: pd.Series,
    title: str = "Feature Importances",
    top_n: int = 20,
) -> plt.Figure:
    importances = importances.nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    bars = ax.barh(importances.index, importances.values,
                   color=sns.color_palette(PALETTE)[0])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    return fig


def plot_learning_curves(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    train_sizes: np.ndarray,
    metric: str = "Score",
    title: str = "Learning Curves",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train", color="steelblue")
    ax.fill_between(train_sizes,
                    train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1),
                    alpha=0.15, color="steelblue")
    ax.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validation", color="darkorange")
    ax.fill_between(train_sizes,
                    val_scores.mean(axis=1) - val_scores.std(axis=1),
                    val_scores.mean(axis=1) + val_scores.std(axis=1),
                    alpha=0.15, color="darkorange")
    ax.set_xlabel("Training Examples")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend()
    return fig
