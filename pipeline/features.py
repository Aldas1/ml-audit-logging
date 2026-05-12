from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import pandas as pd


_SECRET_READ_VERBS = {"get", "list", "watch"}
_SECRET_WRITE_VERBS = {"create", "patch", "update", "delete"}
_RBAC_RESOURCES = {"rolebindings", "clusterrolebindings"}


def _shannon_entropy(values: Iterable[str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def extract_window_features(window: pd.DataFrame) -> dict[str, float]:
    secrets_mask = window["resource"] == "secrets"
    secrets_read = (secrets_mask & window["verb"].isin(_SECRET_READ_VERBS)).sum()
    rbac_create = (
        window["resource"].isin(_RBAC_RESOURCES) & window["verb"].isin({"create"})
    ).sum()
    unique_users = window["username"].nunique()
    entropy_verbs = _shannon_entropy(window["verb"].tolist())
    entropy_users = _shannon_entropy(window["username"].tolist())
    failed_requests = (window["status_code"] >= 400).sum()
    label = int(window["label"].any()) if "label" in window.columns else 0

    return {
        "secrets_read_count": float(secrets_read),
        "failed_request_count": float(failed_requests),
        "rolebinding_create_count": float(rbac_create),
        "unique_users": float(unique_users),
        "entropy_verbs": entropy_verbs,
        "entropy_users": entropy_users,
        "label": label,
    }


def build_features(
    df: pd.DataFrame,
    window_seconds: int = 60,
    stride_seconds: int = 60,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("timestamp").reset_index(drop=True)
    start = df["timestamp"].iloc[0].floor(f"{stride_seconds}s")
    end = df["timestamp"].iloc[-1]

    rows: list[dict] = []
    prev_read_count: float | None = None
    t = start

    while t <= end:
        t_end = t + pd.Timedelta(seconds=window_seconds)
        mask = (df["timestamp"] >= t) & (df["timestamp"] < t_end)
        window = df[mask]

        if not window.empty:
            feats = extract_window_features(window)
            feats["window_start"] = t
            rate_change = (
                feats["secrets_read_count"] - prev_read_count
                if prev_read_count is not None
                else 0.0
            )
            feats["secrets_read_rate_change"] = rate_change
            prev_read_count = feats["secrets_read_count"]
            rows.append(feats)

        t += pd.Timedelta(seconds=stride_seconds)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.set_index("window_start")
    return result
