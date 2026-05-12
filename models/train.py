from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from pipeline.dataset import prepare_splits

RANDOM_STATE = 42


def _scale_pos_weight(y: pd.Series) -> float:
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    return neg / pos if pos > 0 else 1.0


def build_models(scale_pos_weight: float) -> dict[str, Any]:
    return {
        "LightGBM": lgb.LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_estimators=200,
            verbose=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_estimators=200,
            eval_metric="logloss",
        ),
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=1000,
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_estimators=200,
        ),
    }


def train_all(
    features: pd.DataFrame,
    n_splits: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    splits = prepare_splits(features, n_splits=n_splits)
    results: dict[str, list] = {}

    for fold_idx, (X_train, y_train, X_val, y_val) in enumerate(splits):
        spw = _scale_pos_weight(y_train)
        models = build_models(spw)

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_prob = (
                model.predict_proba(X_val)[:, 1]
                if hasattr(model, "predict_proba")
                else model.predict(X_val).astype(float)
            )
            if name not in results:
                results[name] = []
            results[name].append(
                {
                    "fold": fold_idx,
                    "model": model,
                    "y_val": y_val,
                    "y_prob": y_prob,
                    "X_val": X_val,
                }
            )

    return results
