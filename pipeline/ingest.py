from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_jsonl(path: str | Path) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["username"] = df["user"].apply(lambda u: u["username"])
    df["status_code"] = df["responseStatus"].apply(lambda r: r["code"])
    df = df.drop(columns=["user", "responseStatus"])
    return df.sort_values("timestamp").reset_index(drop=True)
