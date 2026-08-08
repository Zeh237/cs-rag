"""Retrieval evaluation metrics with binary relevance judgments.

These functions are pure: they operate on an ordered list of chunk ids (the
retrieved ranking) and a set of relevant chunk ids. The evaluation scripts reuse
these functions so the numbers in the thesis are auditable.
"""

from __future__ import annotations

import math
import random
from statistics import mean

Ranking = list[str]  # ordered retrieved chunk ids
Relevant = set[str]  # chunk ids judged relevant


def precision_at_k(ranks: Ranking, relevant: Relevant, k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved = set(ranks[:k])
    return len(retrieved & relevant) / k


def recall_at_k(ranks: Ranking, relevant: Relevant, k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    retrieved = set(ranks[:k])
    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(ranks: Ranking, relevant: Relevant) -> float:
    relevant = set(relevant)
    for i, chunk_id in enumerate(ranks, start=1):
        if chunk_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranks: Ranking, relevant: Relevant, k: int) -> float:
    relevant = set(relevant)
    top = ranks[:k]
    dcg = sum(1.0 / math.log2(i + 1) for i, chunk_id in enumerate(top, start=1) if chunk_id in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def average_precision(ranks: Ranking, relevant: Relevant, k: int | None = None) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    limit = ranks if k is None else ranks[:k]
    hits = 0
    sum_precisions = 0.0
    for i, chunk_id in enumerate(limit, start=1):
        if chunk_id in relevant:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / len(relevant)


def mean_average_precision(rankings: list[Ranking], relevance: list[Relevant], k: int | None = None) -> float:
    if not rankings:
        return 0.0
    return mean(average_precision(ranks, rel, k) for ranks, rel in zip(rankings, relevance))


def _resampled_means(values: list[float], n_resamples: int, rng: random.Random) -> list[float]:
    n = len(values)
    return [mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(n_resamples)]


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (sample mean, lower bound, upper bound) of a bootstrap CI."""
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    resampled = sorted(_resampled_means(values, n_resamples, rng))
    lo = resampled[int((alpha / 2) * len(resampled))]
    hi = resampled[int((1 - alpha / 2) * len(resampled))]
    return (mean(values), lo, hi)


def paired_bootstrap_diff(
    a: list[float],
    b: list[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean difference, lo, hi) for a pair (a - b) of aligned metrics.

    If the interval does not contain zero, the difference is significant at the
    given alpha level.
    """
    if len(a) != len(b) or not a:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_resamples):
        baseline = [a[rng.randrange(n)] - b[rng.randrange(n)] for _ in range(n)]
        diffs.append(mean(baseline))
    diffs.sort()
    lo = diffs[int((alpha / 2) * len(diffs))]
    hi = diffs[int((1 - alpha / 2) * len(diffs))]
    return (mean([x - y for x, y in zip(a, b)]), lo, hi)