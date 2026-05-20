from __future__ import annotations

import json
import math
import os
import random
from datetime import datetime, timedelta, timezone
from typing import Any

NORMAL_SERVICE_ACCOUNTS = [
    "system:serviceaccount:default:app-sa",
    "system:serviceaccount:default:backend-sa",
    "system:serviceaccount:monitoring:prometheus-sa",
    "system:serviceaccount:kube-system:coredns",
    "system:serviceaccount:production:api-sa",
    "system:serviceaccount:production:worker-sa",
]

_COMPROMISED_POOL = [
    f"system:serviceaccount:default:compromised-sa-{i}" for i in range(1, 16)
]

_OTHER_RESOURCES = ["configmaps", "pods", "deployments", "services", "endpoints"]
NAMESPACES = ["default", "kube-system", "monitoring", "production"]

_SIMULATION_HOURS = 24


def _event(
    timestamp: datetime,
    username: str,
    verb: str,
    resource: str,
    namespace: str,
    status_code: int,
    label: int,
) -> dict[str, Any]:
    return {
        "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "user": {"username": username},
        "verb": verb,
        "resource": resource,
        "namespace": namespace,
        "responseStatus": {"code": status_code},
        "label": label,
    }


def _burst_start_times(
    num_bursts: int,
    sim_start: datetime,
    sim_seconds: int,
    min_gap_seconds: int,
    rng: random.Random,
) -> list[datetime]:
    slots = sorted(rng.sample(range(0, sim_seconds, min_gap_seconds), num_bursts))
    return [sim_start + timedelta(seconds=s) for s in slots]


def generate_normal(
    count: int,
    start: datetime,
    sim_seconds: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    timestamps: list[float] = []
    while len(timestamps) < count:
        s = rng.uniform(0, sim_seconds)
        hour = (s % 86400) / 3600
        weight = 0.625 + 0.375 * math.sin(math.pi * (hour - 3) / 12)
        if rng.random() < weight:
            timestamps.append(s)
    timestamps.sort()

    events: list[dict[str, Any]] = []
    for s in timestamps:
        t = start + timedelta(seconds=s)
        roll = rng.random()
        status_code = 403 if rng.random() < 0.015 else 200
        username = rng.choice(NORMAL_SERVICE_ACCOUNTS)

        if roll < 0.78:
            verb = rng.choice(["get", "list", "watch"])
            resource = "secrets"
        elif roll < 0.90:
            verb = rng.choice(["get", "list", "watch"])
            resource = rng.choice(_OTHER_RESOURCES)
        elif roll < 0.98:
            verb = rng.choice(["create", "patch", "update", "delete"])
            resource = "secrets"
        else:
            verb = rng.choice(["get", "list"])
            resource = rng.choice(["rolebindings", "clusterrolebindings"])

        events.append(
            _event(
                timestamp=t,
                username=username,
                verb=verb,
                resource=resource,
                namespace=rng.choice(NAMESPACES),
                status_code=status_code,
                label=0,
            )
        )
    return events


def generate_rbac_burst(
    count: int,
    start: datetime,
    rng: random.Random,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    t = start
    attacker = rng.choice(_COMPROMISED_POOL)
    for _ in range(count):
        t += timedelta(seconds=rng.uniform(5.0, 15.0))
        events.append(
            _event(
                timestamp=t,
                username=attacker,
                verb=rng.choice(["create", "patch", "update"]),
                resource=rng.choice(["rolebindings", "clusterrolebindings"]),
                namespace=rng.choice(NAMESPACES),
                status_code=rng.choice([200, 201, 403]),
                label=1,
            )
        )
    return events


def generate_spike_burst(
    count: int,
    start: datetime,
    rng: random.Random,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    t = start
    attacker = rng.choice(_COMPROMISED_POOL)
    spike_count = max(1, int(count * 0.80))
    recovery_count = count - spike_count

    for _ in range(spike_count):
        t += timedelta(seconds=rng.uniform(1.0, 3.0))
        status_code = 403 if rng.random() < 0.10 else 200
        events.append(
            _event(
                timestamp=t,
                username=attacker,
                verb=rng.choice(["get", "list", "watch"]),
                resource="secrets",
                namespace=rng.choice(NAMESPACES),
                status_code=status_code,
                label=1,
            )
        )
    for _ in range(recovery_count):
        t += timedelta(seconds=rng.uniform(8.0, 20.0))
        events.append(
            _event(
                timestamp=t,
                username=attacker,
                verb="get",
                resource="secrets",
                namespace=rng.choice(NAMESPACES),
                status_code=200,
                label=1,
            )
        )
    return events


def generate_slow_recon(
    count: int,
    start: datetime,
    rng: random.Random,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    t = start
    attackers = rng.sample(_COMPROMISED_POOL, 3)
    for _ in range(count):
        t += timedelta(seconds=rng.uniform(3.0, 7.0))
        status_code = 403 if rng.random() < 0.06 else 200
        events.append(
            _event(
                timestamp=t,
                username=rng.choice(attackers),
                verb=rng.choice(["get", "list", "watch"]),
                resource="secrets",
                namespace=rng.choice(NAMESPACES),
                status_code=status_code,
                label=1,
            )
        )
    return events


def generate_dataset(
    normal_count: int = 10_000,
    anomaly_count: int = 1_000,
    num_rbac_bursts: int = 12,
    num_spike_bursts: int = 8,
    num_recon_bursts: int = 5,
    seed: int = 42,
    output_path: str | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sim_start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    sim_seconds = _SIMULATION_HOURS * 3600

    events: list[dict[str, Any]] = []
    events.extend(generate_normal(normal_count, sim_start, sim_seconds, rng))

    total_bursts = num_rbac_bursts + num_spike_bursts + num_recon_bursts
    min_gap = sim_seconds // (total_bursts * 2)
    burst_times = _burst_start_times(total_bursts, sim_start, sim_seconds, min_gap, rng)
    rng.shuffle(burst_times)

    rbac_times = burst_times[:num_rbac_bursts]
    spike_times = burst_times[num_rbac_bursts : num_rbac_bursts + num_spike_bursts]
    recon_times = burst_times[num_rbac_bursts + num_spike_bursts :]

    rbac_budget = int(anomaly_count * 0.40)
    spike_budget = int(anomaly_count * 0.35)
    recon_budget = anomaly_count - rbac_budget - spike_budget

    events_per_rbac = max(1, rbac_budget // num_rbac_bursts)
    events_per_spike = max(1, spike_budget // num_spike_bursts)
    events_per_recon = max(1, recon_budget // num_recon_bursts)

    for i, t in enumerate(rbac_times):
        extra = rbac_budget - events_per_rbac * num_rbac_bursts if i == len(rbac_times) - 1 else 0
        events.extend(generate_rbac_burst(events_per_rbac + extra, t, rng))
    for i, t in enumerate(spike_times):
        extra = spike_budget - events_per_spike * num_spike_bursts if i == len(spike_times) - 1 else 0
        events.extend(generate_spike_burst(events_per_spike + extra, t, rng))
    for i, t in enumerate(recon_times):
        extra = recon_budget - events_per_recon * num_recon_bursts if i == len(recon_times) - 1 else 0
        events.extend(generate_slow_recon(events_per_recon + extra, t, rng))

    events.sort(key=lambda e: e["timestamp"])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

    return events


if __name__ == "__main__":
    events = generate_dataset(output_path="data/audit_logs.jsonl")
    normal = sum(1 for e in events if e["label"] == 0)
    anomalous = sum(1 for e in events if e["label"] == 1)
    print(f"Generated {len(events)} events: {normal} normal, {anomalous} anomalous")
