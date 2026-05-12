#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from pipeline.ingest import load_jsonl as load_logs
from pipeline.features import build_features
from models.train import train_all
from models.evaluate import aggregate_results, plot_pr_curves

DATA_PATH = Path("data/audit_logs.jsonl")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    t0 = time.perf_counter()

    print("Loading logs...")
    t = time.perf_counter()
    df = load_logs(str(DATA_PATH))
    print(f"  {len(df):,} events loaded in {time.perf_counter() - t:.2f}s")

    print("Building features...")
    t = time.perf_counter()
    features = build_features(df)
    n_anomalous = features["label"].sum()
    print(f"  {len(features)} windows ({n_anomalous} anomalous) in {time.perf_counter() - t:.2f}s")

    print("Training models (5-fold TimeSeriesSplit)...")
    fold_times: dict[str, list[float]] = {}
    fold_results_all: dict[str, list] = {}

    from pipeline.dataset import prepare_splits, FEATURE_COLS
    from models.train import build_models, _scale_pos_weight

    splits = prepare_splits(features, n_splits=5)

    for fold_idx, (X_train, y_train, X_val, y_val) in enumerate(splits):
        spw = _scale_pos_weight(y_train)
        models = build_models(spw)
        for name, model in models.items():
            t = time.perf_counter()
            model.fit(X_train, y_train)
            elapsed = time.perf_counter() - t
            fold_times.setdefault(name, []).append(elapsed)

            y_prob = (
                model.predict_proba(X_val)[:, 1]
                if hasattr(model, "predict_proba")
                else model.predict(X_val).astype(float)
            )
            fold_results_all.setdefault(name, []).append(
                {"fold": fold_idx, "model": model, "y_val": y_val, "y_prob": y_prob, "X_val": X_val}
            )

    for name, times in fold_times.items():
        total = sum(times)
        avg = total / len(times)
        print(f"  {name}: {total:.2f}s total, {avg:.2f}s/fold")

    print("\nEvaluating...")
    metrics = aggregate_results(fold_results_all)
    metrics_out = metrics.reset_index()
    metrics_out.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print(metrics_out.to_string(index=False))

    print("\nPlotting PR curves...")
    plot_pr_curves(fold_results_all, output_path=str(RESULTS_DIR / "pr_curves.png"))

    print(f"\nTotal time: {time.perf_counter() - t0:.2f}s")
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
