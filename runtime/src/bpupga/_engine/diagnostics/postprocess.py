from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


Pair = Tuple[int, int]


def _as_cost(matrix: np.ndarray, larger_is_better: bool) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Postprocessing matrix must be two-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("Postprocessing matrix contains NaN or infinite values")
    return -values if larger_is_better else values


def full_hungarian_pairs(
    matrix: np.ndarray,
    larger_is_better: bool,
) -> List[Pair]:
    """Return min(m, n) one-to-one pairs."""
    cost = _as_cost(matrix, larger_is_better)
    rows, cols = linear_sum_assignment(cost)
    return [(int(u), int(v)) for u, v in zip(rows, cols)]


def oracle_cardinality_pairs(
    matrix: np.ndarray,
    cardinality: int,
    larger_is_better: bool,
) -> List[Pair]:
    """Return exactly ``cardinality`` one-to-one pairs.

    This augments the rectangular assignment with dummy rows and columns.
    Dummy-to-dummy assignments receive a prohibitive cost, forcing exactly k
    real-to-real assignments. Because k comes from ground truth, this function
    is evaluation-only and must never be called inside a baseline algorithm.
    """
    cost = _as_cost(matrix, larger_is_better)
    m, n = cost.shape
    k = int(cardinality)
    if not 0 <= k <= min(m, n):
        raise ValueError(f"cardinality must be in [0, {min(m, n)}], got {k}")
    if k == 0:
        return []
    if k == min(m, n):
        return full_hungarian_pairs(cost, larger_is_better=False)

    dummy_source = n - k
    dummy_target = m - k
    size = m + n - k

    finite_min = float(cost.min())
    finite_max = float(cost.max())
    span = max(1.0, finite_max - finite_min)
    # Shifting preserves every fixed-cardinality solution and makes the dummy
    # penalty valid even when a score matrix becomes a negative cost matrix.
    shifted_cost = cost - finite_min
    penalty = span * (size + 1)

    augmented = np.zeros((size, size), dtype=np.float64)
    augmented[:m, :n] = shifted_cost
    if dummy_source and dummy_target:
        augmented[m:, n:] = penalty

    rows, cols = linear_sum_assignment(augmented)
    pairs = [
        (int(u), int(v))
        for u, v in zip(rows, cols)
        if u < m and v < n
    ]
    if len(pairs) != k:
        raise RuntimeError(f"Oracle assignment produced {len(pairs)} real pairs, expected {k}")
    return pairs
