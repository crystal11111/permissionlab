"""Task-clustered bootstrap. Repetitions of one task are not independent, so we resample tasks."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Callable, Iterable

Row = dict[str, Any]


def _percentile(sorted_values: list[float], q: float) -> float:
    pos = q * (len(sorted_values) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def _cluster_sums(rows: Iterable[Row], value: Callable[[Row], float | None], key: str) -> dict[str, list[float]]:
    sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for row in rows:
        v = value(row)
        if v is None:
            continue
        bucket = sums[row[key]]
        bucket[0] += float(v)
        bucket[1] += 1
    return dict(sums)


def _empty() -> dict[str, Any]:
    return {"point": None, "lo": None, "hi": None, "n": 0, "n_clusters": 0}


def cluster_bootstrap(rows: list[Row], value: Callable[[Row], float | None], *, cluster_key: str = "task_id",
                      n_boot: int = 5000, seed: int = 0, ci: float = 0.95) -> dict[str, Any]:
    sums = _cluster_sums(rows, value, cluster_key)
    ids = sorted(sums)
    n = sum(c for _, c in sums.values())
    if not n:
        return _empty()
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        s = c = 0.0
        for _ in ids:
            cs, cc = sums[ids[rng.randrange(len(ids))]]
            s += cs
            c += cc
        stats.append(s / c)
    stats.sort()
    alpha = (1 - ci) / 2
    return {"point": sum(s for s, _ in sums.values()) / n, "lo": _percentile(stats, alpha),
            "hi": _percentile(stats, 1 - alpha), "n": int(n), "n_clusters": len(ids)}


def paired_cluster_bootstrap(rows_a: list[Row], rows_b: list[Row], value: Callable[[Row], float | None], *,
                             cluster_key: str = "task_id", n_boot: int = 5000, seed: int = 0,
                             ci: float = 0.95) -> dict[str, Any]:
    """CI for mean(a) - mean(b), resampling the same tasks for both conditions."""
    sa = _cluster_sums(rows_a, value, cluster_key)
    sb = _cluster_sums(rows_b, value, cluster_key)
    ids = sorted(set(sa) & set(sb))
    if not ids:
        return _empty()

    def diff(sample: list[str]) -> float:
        return (sum(sa[i][0] for i in sample) / sum(sa[i][1] for i in sample)
                - sum(sb[i][0] for i in sample) / sum(sb[i][1] for i in sample))

    rng = random.Random(seed)
    stats = sorted(diff([ids[rng.randrange(len(ids))] for _ in ids]) for _ in range(n_boot))
    alpha = (1 - ci) / 2
    return {"point": diff(ids), "lo": _percentile(stats, alpha), "hi": _percentile(stats, 1 - alpha),
            "n": len(ids), "n_clusters": len(ids)}
