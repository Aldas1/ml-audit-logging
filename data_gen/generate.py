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
    """Normal traffic with realistic variety.

    Mix:
      78% secrets reads  — primary k8s usage pattern
      12% other resource reads (configmaps, pods, …)
       8% secrets writes  — legitimate rotation / sync
       2% RBAC list/get   — admin read-only checks

    1.5% of events produce a 403 (transient permission failures).
    Traffic follows a business-hour sine curve so some windows have
    many events and some have very few — preventing trivial separation
    on window size alone.
    """
    # Pre-assign event timestamps via business-hour density: peak 10h, trough 3h.
    # Draw seconds-from-start with rejection sampling against a sine-shaped weight.
    timestamps: list[float] = []
    while len(timestamps) < count:
        s = rng.uniform(0, sim_seconds)
        hour = (s % 86400) / 3600
        # weight in [0.25, 1.0]: high around 10h, low around 3h
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
    """RBAC escalation: a single compromised SA creates rolebindings over ~3-4 minutes.

    Events are spread wide enough (5–15 s gaps) to spill across 3–4 windows so
    the anomaly signal per window is subtle (≈5–7 creates, vs. normal ≈0).
    """
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
    """Secrets exfiltration spike: rapid reads by one account over 2–3 minutes.

    10% failure rate (recon against namespaces with restricted access).
    The last 20% of events form a slow recovery tail (wider gaps) simulating
    the attacker decelerating before stopping. Total events stays at `count`.
    """
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
    """Slow reconnaissance: 3 rotating compromised accounts probing secrets over ~8 minutes.

    Designed to challenge simpler models:
    - per-window reads are only ~30-50% above the normal baseline (not a spike)
    - 3 rotating accounts keep entropy_users ≈ 1.5 bits (normal ≈ 2.3 bits but varies)
    - 6% failure rate (vs. normal 1.5%)
    - spans 7–9 windows — the sustained, gradual nature is what distinguishes it
    """
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
    """Generate a labelled synthetic Kubernetes audit log dataset.

    Three anomaly types keep ~100+ anomalous windows spread across all folds:
      - RBAC escalation  (easy,   ~3-4 windows/burst)
      - Secrets spike    (medium, ~2-3 windows/burst)
      - Slow recon       (hard,   ~7-9 windows/burst)
    """
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

    # Split anomaly budget: 40% RBAC, 35% spike, 25% recon
    rbac_budget = int(anomaly_count * 0.40)
    spike_budget = int(anomaly_count * 0.35)
    recon_budget = anomaly_count - rbac_budget - spike_budget

    events_per_rbac = max(1, rbac_budget // num_rbac_bursts)
    events_per_spike = max(1, spike_budget // num_spike_bursts)
    events_per_recon = max(1, recon_budget // num_recon_bursts)

    for t in rbac_times:
        events.extend(generate_rbac_burst(events_per_rbac, t, rng))
    for t in spike_times:
        events.extend(generate_spike_burst(events_per_spike, t, rng))
    for t in recon_times:
        events.extend(generate_slow_recon(events_per_recon, t, rng))

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
