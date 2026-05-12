from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def evaluate_fold(y_val: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "auc_pr": average_precision_score(y_val, y_prob),
    }


def aggregate_results(
    fold_results: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    rows = []
    for model_name, folds in fold_results.items():
        fold_metrics = [
            evaluate_fold(f["y_val"], f["y_prob"]) for f in folds
        ]
        mean_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        std_metrics = {f"{k}_std": np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        rows.append({"model": model_name, **mean_metrics, **std_metrics})

    df = pd.DataFrame(rows).set_index("model")
    return df.sort_values("f1", ascending=False)


def plot_pr_curves(
    fold_results: dict[str, list[dict[str, Any]]],
    output_path: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_name, folds in fold_results.items():
        all_y_val = np.concatenate([f["y_val"].values for f in folds])
        all_y_prob = np.concatenate([f["y_prob"] for f in folds])
        precision, recall, _ = precision_recall_curve(all_y_val, all_y_prob)
        auc_pr = average_precision_score(all_y_val, all_y_prob)
        ax.plot(recall, precision, label=f"{model_name} (AUC-PR={auc_pr:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)
