#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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

from data_gen.generate import generate_dataset
from pipeline.features import build_features
from pipeline.dataset import prepare_splits, FEATURE_COLS, LABEL_COL
from models.train import build_models, _scale_pos_weight

SEEDS = list(range(1, 51))
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def events_to_df(events: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(events)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["username"] = df["user"].apply(lambda u: u["username"])
    df["status_code"] = df["responseStatus"].apply(lambda r: r["code"])
    df = df.drop(columns=["user", "responseStatus"])
    return df.sort_values("timestamp").reset_index(drop=True)


def evaluate(y_val, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "auc_pr": average_precision_score(y_val, y_prob),
    }


def main() -> None:
    t0 = time.perf_counter()

    per_fold_metrics: dict[str, list[dict[str, float]]] = {}
    pooled_pred: dict[str, dict[str, list]] = {}
    all_windows: list[pd.DataFrame] = []
    spw_values: list[float] = []
    window_counts: list[int] = []
    anomalous_counts: list[int] = []
    val_positive_counts: list[int] = []
    importances: dict[str, list[np.ndarray]] = {}

    for seed in SEEDS:
        events = generate_dataset(normal_count=10_000, anomaly_count=1_000, seed=seed)
        df = events_to_df(events)
        feats = build_features(df)

        n_win = len(feats)
        n_anom = int(feats[LABEL_COL].sum())
        window_counts.append(n_win)
        anomalous_counts.append(n_anom)
        all_windows.append(feats[FEATURE_COLS + [LABEL_COL]].copy())
        print(f"seed={seed}: {n_win} windows, {n_anom} anomalous "
              f"({100 * n_anom / n_win:.2f}%)")

        splits = prepare_splits(feats, n_splits=5)
        for X_train, y_train, X_val, y_val in splits:
            spw = _scale_pos_weight(y_train)
            spw_values.append(spw)
            val_positive_counts.append(int((y_val == 1).sum()))
            models = build_models(spw)
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_prob = (
                    model.predict_proba(X_val)[:, 1]
                    if hasattr(model, "predict_proba")
                    else model.predict(X_val).astype(float)
                )
                per_fold_metrics.setdefault(name, []).append(evaluate(y_val, y_prob))
                p = pooled_pred.setdefault(name, {"y": [], "p": []})
                p["y"].append(np.asarray(y_val))
                p["p"].append(np.asarray(y_prob))
                if hasattr(model, "feature_importances_"):
                    imp = np.asarray(model.feature_importances_, dtype=float)
                    s = imp.sum()
                    if s > 0:
                        imp = imp / s
                    importances.setdefault(name, []).append(imp)

    # ---- aggregated metrics table (mean & std over 5 seeds x 5 folds) ----
    rows = []
    for name, folds in per_fold_metrics.items():
        agg = {"model": name}
        for k in ("precision", "recall", "f1", "auc_pr"):
            vals = np.array([f[k] for f in folds])
            agg[k] = vals.mean()
            agg[f"{k}_std"] = vals.std()
        rows.append(agg)
    metrics = pd.DataFrame(rows).set_index("model").sort_values("f1", ascending=False)
    metrics.reset_index().to_csv(RESULTS_DIR / "metrics.csv", index=False)

    print("\n===== AGGREGATED METRICS (mean +/- std over "
          f"{len(SEEDS)} seeds x 5 folds = {len(SEEDS) * 5} measurements) =====")
    for name, r in metrics.iterrows():
        print(f"{name:20s} P={r['precision']:.3f}+/-{r['precision_std']:.3f}  "
              f"R={r['recall']:.3f}+/-{r['recall_std']:.3f}  "
              f"F1={r['f1']:.3f}+/-{r['f1_std']:.3f}  "
              f"AUC-PR={r['auc_pr']:.3f}+/-{r['auc_pr_std']:.3f}")

    # ---- dataset-level stats averaged across seeds ----
    print("\n===== DATASET STATS (averaged across seeds) =====")
    print(f"windows:       mean={np.mean(window_counts):.1f}  "
          f"min={min(window_counts)}  max={max(window_counts)}  per-seed={window_counts}")
    print(f"anomalous:     mean={np.mean(anomalous_counts):.1f}  "
          f"per-seed={anomalous_counts}")
    fracs = [a / w for a, w in zip(anomalous_counts, window_counts)]
    print(f"anom fraction: mean={100 * np.mean(fracs):.2f}%  "
          f"(=> AUC-PR baseline {np.mean(fracs):.3f})")
    print(f"scale_pos_weight: mean={np.mean(spw_values):.2f}  "
          f"range=[{min(spw_values):.2f}, {max(spw_values):.2f}]")
    print(f"val positives per fold: min={min(val_positive_counts)}  "
          f"max={max(val_positive_counts)}  mean={np.mean(val_positive_counts):.1f}")

    # ---- feature distribution table (pooled windows, by label) ----
    pooled_win = pd.concat(all_windows, ignore_index=True)
    print("\n===== FEATURE DISTRIBUTION (pooled windows, by label) =====")
    order = [
        "secrets_read_count", "failed_request_count", "rolebinding_create_count",
        "unique_users", "entropy_verbs", "entropy_users", "secrets_read_rate_change",
    ]
    norm = pooled_win[pooled_win[LABEL_COL] == 0]
    anom = pooled_win[pooled_win[LABEL_COL] == 1]
    print(f"{'feature':32s} {'norm_mean':>10s} {'norm_std':>9s} "
          f"{'anom_mean':>10s} {'anom_std':>9s}")
    for f in order:
        print(f"{f:32s} {norm[f].mean():10.2f} {norm[f].std():9.2f} "
              f"{anom[f].mean():10.2f} {anom[f].std():9.2f}")

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, p in pooled_pred.items():
        y_all = np.concatenate(p["y"])
        p_all = np.concatenate(p["p"])
        prec, rec, _ = precision_recall_curve(y_all, p_all)
        auc_pr = average_precision_score(y_all, p_all)
        ax.plot(rec, prec, label=f"{name} (AUC-PR={auc_pr:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves (pooled over {len(SEEDS)} seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pr_curves.png", dpi=150)
    plt.close(fig)

    # feature importance: mean over all seeds x folds, tree models
    fig, ax = plt.subplots(figsize=(9, 6))
    width = 0.8 / max(1, len(importances))
    x = np.arange(len(FEATURE_COLS))
    for i, (name, imps) in enumerate(importances.items()):
        mean_imp = np.mean(np.vstack(imps), axis=0)
        ax.bar(x + i * width, mean_imp, width, label=name)
    ax.set_xticks(x + width * (len(importances) - 1) / 2)
    ax.set_xticklabels(FEATURE_COLS, rotation=30, ha="right")
    ax.set_ylabel("Normalized importance")
    ax.set_title(f"Feature importance (mean over {len(SEEDS)} seeds x 5 folds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)

    # feature distributions: normal vs anomalous mean +/- std
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(order))
    nm = [norm[f].mean() for f in order]
    ns = [norm[f].std() for f in order]
    am = [anom[f].mean() for f in order]
    as_ = [anom[f].std() for f in order]
    ax.bar(x - 0.2, nm, 0.4, yerr=ns, capsize=3, label="Normal")
    ax.bar(x + 0.2, am, 0.4, yerr=as_, capsize=3, label="Anomalous")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("Value")
    ax.set_title(f"Feature distribution by label (pooled over {len(SEEDS)} seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_distributions.png", dpi=150)
    plt.close(fig)

    print(f"\nTotal time: {time.perf_counter() - t0:.1f}s")
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
