from __future__ import annotations

import pandas as pd
import pytest

from pipeline.features import _shannon_entropy, build_features, extract_window_features


def _make_events(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def test_shannon_entropy_uniform():
    assert _shannon_entropy(["a", "b", "c", "d"]) == pytest.approx(2.0, abs=1e-9)


def test_shannon_entropy_single():
    assert _shannon_entropy(["a", "a", "a"]) == pytest.approx(0.0, abs=1e-9)


def test_shannon_entropy_empty():
    assert _shannon_entropy([]) == 0.0


def test_extract_window_basic():
    events = _make_events([
        {"timestamp": "2026-01-01T00:00:01Z", "username": "sa1", "verb": "get", "resource": "secrets", "namespace": "default", "status_code": 200, "label": 0},
        {"timestamp": "2026-01-01T00:00:10Z", "username": "sa1", "verb": "list", "resource": "secrets", "namespace": "default", "status_code": 200, "label": 0},
        {"timestamp": "2026-01-01T00:00:20Z", "username": "sa2", "verb": "create", "resource": "rolebindings", "namespace": "default", "status_code": 403, "label": 1},
    ])
    feats = extract_window_features(events)
    assert feats["secrets_read_count"] == 2.0
    assert feats["failed_request_count"] == 1.0
    assert feats["rolebinding_create_count"] == 1.0
    assert feats["unique_users"] == 2.0
    assert feats["label"] == 1


def test_extract_window_no_label_column():
    events = _make_events([
        {"timestamp": "2026-01-01T00:00:01Z", "username": "sa1", "verb": "get", "resource": "secrets", "namespace": "default", "status_code": 200},
    ])
    feats = extract_window_features(events)
    assert feats["label"] == 0


def test_build_features_window_count():
    events = _make_events([
        {"timestamp": "2026-01-01T00:00:01Z", "username": "sa1", "verb": "get", "resource": "secrets", "namespace": "default", "status_code": 200, "label": 0},
        {"timestamp": "2026-01-01T00:01:01Z", "username": "sa1", "verb": "get", "resource": "secrets", "namespace": "default", "status_code": 200, "label": 0},
        {"timestamp": "2026-01-01T00:02:01Z", "username": "sa1", "verb": "get", "resource": "secrets", "namespace": "default", "status_code": 200, "label": 0},
    ])
    result = build_features(events, window_seconds=60, stride_seconds=60)
    assert len(result) == 3


def test_build_features_rate_change():
    events = _make_events([
        {"timestamp": "2026-01-01T00:00:01Z", "username": "sa1", "verb": "get", "resource": "secrets", "namespace": "default", "status_code": 200, "label": 0},
        {"timestamp": "2026-01-01T00:00:10Z", "username": "sa1", "verb": "get", "resource": "secrets", "namespace": "default", "status_code": 200, "label": 0},
        {"timestamp": "2026-01-01T00:01:01Z", "username": "sa1", "verb": "get", "resource": "secrets", "namespace": "default", "status_code": 200, "label": 0},
        {"timestamp": "2026-01-01T00:01:05Z", "username": "sa1", "verb": "get", "resource": "secrets", "namespace": "default", "status_code": 200, "label": 0},
        {"timestamp": "2026-01-01T00:01:10Z", "username": "sa1", "verb": "get", "resource": "secrets", "namespace": "default", "status_code": 200, "label": 0},
    ])
    result = build_features(events, window_seconds=60, stride_seconds=60)
    assert result["secrets_read_rate_change"].iloc[0] == pytest.approx(0.0)
    assert result["secrets_read_rate_change"].iloc[1] == pytest.approx(1.0)


def test_build_features_empty():
    result = build_features(pd.DataFrame(), window_seconds=60, stride_seconds=60)
    assert result.empty
