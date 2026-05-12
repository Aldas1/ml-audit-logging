from __future__ import annotations

import io
import json
import textwrap

import pandas as pd
import pytest

from data_gen.generate import generate_dataset
from pipeline.dataset import FEATURE_COLS, prepare_splits
from pipeline.features import build_features
from pipeline.ingest import load_jsonl


# ── data generation ──────────────────────────────────────────────────────────

def test_generate_dataset_counts():
    # normal_count is exact; anomaly_count is a budget distributed via floor-division
    # across burst slots (min 1 event per slot), so actual count may differ slightly.
    events = generate_dataset(normal_count=200, anomaly_count=100, seed=42)
    normal = sum(1 for e in events if e["label"] == 0)
    anomalous = sum(1 for e in events if e["label"] == 1)
    assert normal == 200
    assert 0 < anomalous <= 100


def test_generate_dataset_sorted():
    events = generate_dataset(normal_count=200, anomaly_count=20, seed=42)
    timestamps = [e["timestamp"] for e in events]
    assert timestamps == sorted(timestamps)


def test_generate_dataset_schema():
    events = generate_dataset(normal_count=50, anomaly_count=10, seed=42)
    required = {"timestamp", "user", "verb", "resource", "namespace", "responseStatus", "label"}
    for ev in events:
        assert required.issubset(ev.keys())
        assert ev["label"] in (0, 1)


def test_generate_dataset_reproducible():
    a = generate_dataset(normal_count=100, anomaly_count=10, seed=7)
    b = generate_dataset(normal_count=100, anomaly_count=10, seed=7)
    assert a == b


# ── ingest ────────────────────────────────────────────────────────────────────

def _jsonl_file(events: list[dict]) -> str:
    """Write events to a temp file and return the path."""
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    return path


def test_ingest_columns():
    events = generate_dataset(normal_count=50, anomaly_count=10, seed=42)
    path = _jsonl_file(events)
    df = load_jsonl(path)
    assert "timestamp" in df.columns
    assert "username" in df.columns
    assert "status_code" in df.columns
    assert df.dtypes["timestamp"].tz is not None


def test_ingest_sorted():
    events = generate_dataset(normal_count=50, anomaly_count=10, seed=42)
    path = _jsonl_file(events)
    df = load_jsonl(path)
    assert df["timestamp"].is_monotonic_increasing


# ── features ─────────────────────────────────────────────────────────────────

def _df_from_events(events: list[dict]) -> pd.DataFrame:
    path = _jsonl_file(events)
    return load_jsonl(path)


def test_features_all_columns_present():
    events = generate_dataset(normal_count=200, anomaly_count=20, seed=42)
    df = _df_from_events(events)
    feats = build_features(df)
    assert set(FEATURE_COLS).issubset(feats.columns)
    assert "label" in feats.columns


def test_features_first_rate_change_zero():
    events = generate_dataset(normal_count=200, anomaly_count=20, seed=42)
    df = _df_from_events(events)
    feats = build_features(df)
    assert feats["secrets_read_rate_change"].iloc[0] == pytest.approx(0.0)


def test_features_non_negative_counts():
    non_delta = [c for c in FEATURE_COLS if "rate_change" not in c]
    events = generate_dataset(normal_count=200, anomaly_count=20, seed=42)
    df = _df_from_events(events)
    feats = build_features(df)
    assert (feats[non_delta] >= 0).all().all()


def test_features_index_sorted():
    events = generate_dataset(normal_count=200, anomaly_count=20, seed=42)
    df = _df_from_events(events)
    feats = build_features(df)
    assert feats.index.is_monotonic_increasing


# ── splits ────────────────────────────────────────────────────────────────────

def test_splits_no_future_leakage():
    events = generate_dataset(normal_count=1000, anomaly_count=100,
                               num_rbac_bursts=10, num_spike_bursts=10, seed=42)
    df = _df_from_events(events)
    feats = build_features(df)
    for i, (Xtr, ytr, Xval, yval) in enumerate(prepare_splits(feats)):
        assert Xtr.index[-1] < Xval.index[0], f"fold {i}: future leakage"


def test_splits_five_folds():
    events = generate_dataset(normal_count=1000, anomaly_count=100,
                               num_rbac_bursts=10, num_spike_bursts=10, seed=42)
    df = _df_from_events(events)
    feats = build_features(df)
    splits = prepare_splits(feats)
    assert len(splits) == 5


def test_splits_all_folds_have_positives():
    events = generate_dataset(normal_count=1000, anomaly_count=100,
                               num_rbac_bursts=10, num_spike_bursts=10, seed=42)
    df = _df_from_events(events)
    feats = build_features(df)
    for i, (_, _, _, yval) in enumerate(prepare_splits(feats)):
        assert yval.sum() > 0, f"fold {i} has no positive validation examples"
