from __future__ import annotations

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

FEATURE_COLS = [
    "secrets_read_count",
    "failed_request_count",
    "rolebinding_create_count",
    "unique_users",
    "entropy_verbs",
    "entropy_users",
    "secrets_read_rate_change",
]
LABEL_COL = "label"


def prepare_splits(
    features: pd.DataFrame,
    n_splits: int = 5,
) -> list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    X = features[FEATURE_COLS]
    y = features[LABEL_COL]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_idx, val_idx in tscv.split(X):
        splits.append((X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx], y.iloc[val_idx]))
    return splits
