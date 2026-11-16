from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import pairwise_distances


Pair = Tuple[int, int]


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _prf(predicted: set, truth: set) -> Dict[str, float | int]:
    true_positive = len(predicted & truth)
    precision = _safe_div(true_positive, len(predicted))
    recall = _safe_div(true_positive, len(truth))
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "predicted": len(predicted),
        "ground_truth": len(truth),
    }


def validate_predicted_pairs(
    predicted_pairs: Iterable[Pair],
    num_source_nodes: int,
    num_target_nodes: int,
) -> Dict[str, object]:
    pairs = [(int(u), int(v)) for u, v in predicted_pairs]
    source_counts = Counter(u for u, _ in pairs)
    target_counts = Counter(v for _, v in pairs)
    pair_counts = Counter(pairs)

    out_of_range = [
        [u, v]
        for u, v in pairs
        if not (0 <= u < num_source_nodes and 0 <= v < num_target_nodes)
    ]
    duplicate_sources = sorted(u for u, count in source_counts.items() if count > 1)
    duplicate_targets = sorted(v for v, count in target_counts.items() if count > 1)
    duplicate_pairs = sorted([list(pair) for pair, count in pair_counts.items() if count > 1])

    return {
        "num_raw_pairs": len(pairs),
        "num_unique_pairs": len(pair_counts),
        "out_of_range_count": len(out_of_range),
        "out_of_range_examples": out_of_range[:10],
        "duplicate_pair_count": len(duplicate_pairs),
        "duplicate_pair_examples": duplicate_pairs[:10],
        "duplicate_source_count": len(duplicate_sources),
        "duplicate_source_examples": duplicate_sources[:10],
        "duplicate_target_count": len(duplicate_targets),
        "duplicate_target_examples": duplicate_targets[:10],
        "is_partial_function": not out_of_range and not duplicate_sources,
        "is_one_to_one": not out_of_range and not duplicate_sources and not duplicate_targets,
    }


def _valid_pair_set(
    predicted_pairs: Iterable[Pair],
    num_source_nodes: int,
    num_target_nodes: int,
) -> set[Pair]:
    return {
        (int(u), int(v))
        for u, v in predicted_pairs
        if 0 <= int(u) < num_source_nodes and 0 <= int(v) < num_target_nodes
    }


def evaluate_pair_set(
    predicted_pairs: Iterable[Pair],
    ground_truth: Mapping[int, int],
    num_source_nodes: int,
    num_target_nodes: int,
) -> Dict[str, float | int]:
    predicted = _valid_pair_set(predicted_pairs, num_source_nodes, num_target_nodes)
    truth = {(int(u), int(v)) for u, v in ground_truth.items()}
    return _prf(predicted, truth)


def evaluate_matchability(
    predicted_pairs: Iterable[Pair],
    ground_truth: Mapping[int, int],
    num_source_nodes: int,
    num_target_nodes: int,
) -> Dict[str, Dict[str, float | int]]:
    predicted = _valid_pair_set(predicted_pairs, num_source_nodes, num_target_nodes)
    predicted_sources = {u for u, _ in predicted}
    predicted_targets = {v for _, v in predicted}
    true_sources = {int(u) for u in ground_truth.keys()}
    true_targets = {int(v) for v in ground_truth.values()}
    return {
        "source": _prf(predicted_sources, true_sources),
        "target": _prf(predicted_targets, true_targets),
    }


def evaluate_overlap_estimation(
    predicted_pairs: Iterable[Pair],
    ground_truth: Mapping[int, int],
    num_source_nodes: int,
    num_target_nodes: int,
) -> Dict[str, float | int]:
    """Measure how well a native partial matching estimates overlap size.

    Pair cardinality and the two directional node coverages are kept separate
    so invalid many-to-one outputs cannot hide behind a single count.
    """

    predicted = _valid_pair_set(
        predicted_pairs, num_source_nodes, num_target_nodes
    )
    predicted_sources = {source for source, _ in predicted}
    predicted_targets = {target for _, target in predicted}
    ground_truth_pairs = len(ground_truth)
    predicted_pair_count = len(predicted)
    absolute_error = abs(predicted_pair_count - ground_truth_pairs)
    true_source_ratio = _safe_div(ground_truth_pairs, num_source_nodes)
    true_target_ratio = _safe_div(ground_truth_pairs, num_target_nodes)
    predicted_source_ratio = _safe_div(len(predicted_sources), num_source_nodes)
    predicted_target_ratio = _safe_div(len(predicted_targets), num_target_nodes)
    return {
        "ground_truth_pairs": ground_truth_pairs,
        "predicted_pairs": predicted_pair_count,
        "absolute_pair_count_error": absolute_error,
        "relative_pair_count_error": _safe_div(absolute_error, ground_truth_pairs),
        "true_source_overlap_ratio": true_source_ratio,
        "predicted_source_overlap_ratio": predicted_source_ratio,
        "source_overlap_absolute_error": abs(
            predicted_source_ratio - true_source_ratio
        ),
        "true_target_overlap_ratio": true_target_ratio,
        "predicted_target_overlap_ratio": predicted_target_ratio,
        "target_overlap_absolute_error": abs(
            predicted_target_ratio - true_target_ratio
        ),
    }


def _metrics_from_ranks(ranks: Sequence[int], ks: Sequence[int]) -> Dict[str, float | int]:
    if not ranks:
        output: Dict[str, float | int] = {f"hits@{k}": 0.0 for k in ks}
        output.update({"mrr": 0.0, "evaluated_pairs": 0})
        return output

    rank_array = np.asarray(ranks, dtype=np.int64)
    output = {f"hits@{k}": float(np.mean(rank_array <= k)) for k in ks}
    output["mrr"] = float(np.mean(1.0 / rank_array))
    output["evaluated_pairs"] = int(len(rank_array))
    return output


def evaluate_ranking_from_matrix(
    matrix: np.ndarray,
    ground_truth: Mapping[int, int],
    ks: Sequence[int] = (1, 5, 10),
    larger_is_better: bool = True,
) -> Dict[str, float | int]:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("Ranking matrix must be two-dimensional")

    ranks: list[int] = []
    for u, v_true in ground_truth.items():
        u, v_true = int(u), int(v_true)
        if not (0 <= u < matrix.shape[0] and 0 <= v_true < matrix.shape[1]):
            continue
        row = matrix[u]
        if not np.isfinite(row).all():
            raise ValueError("Ranking row contains NaN/Inf")
        value = row[v_true]
        better = row > value if larger_is_better else row < value
        # Exactly the stable-sort rank, including the lower-node-ID tie break,
        # without sorting every full 11k-node candidate list.
        ranks.append(1 + int(np.count_nonzero(better))
                     + int(np.count_nonzero(row[:v_true] == value)))
    return _metrics_from_ranks(ranks, ks)


def evaluate_ranking_from_candidates(
    candidates: Iterable[tuple[int, int, float]],
    ground_truth: Mapping[int, int],
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, float | int | str]:
    """Evaluate a sparse, scored retrieval list without inventing missing scores.

    A ground-truth pair absent from the method's candidate set contributes zero
    to Hits@k and MRR.  Within each query, candidates are sorted by descending
    score and then target ID, matching NumPy's stable dense-ranking convention.
    """

    by_source: dict[int, dict[int, float]] = {}
    unique_pairs = 0
    for raw_source, raw_target, raw_score in candidates:
        source, target, score = int(raw_source), int(raw_target), float(raw_score)
        if not np.isfinite(score):
            raise ValueError("Sparse ranking scores must be finite")
        row = by_source.setdefault(source, {})
        if target not in row:
            unique_pairs += 1
            row[target] = score
        else:
            row[target] = max(row[target], score)

    reciprocal_ranks: list[float] = []
    ranks: list[int | None] = []
    retrieved_truth = 0
    for raw_source, raw_target in ground_truth.items():
        source, true_target = int(raw_source), int(raw_target)
        row = by_source.get(source, {})
        if true_target not in row:
            ranks.append(None)
            reciprocal_ranks.append(0.0)
            continue
        ordered = sorted(row, key=lambda target: (-row[target], target))
        rank = ordered.index(true_target) + 1
        ranks.append(rank)
        reciprocal_ranks.append(1.0 / rank)
        retrieved_truth += 1

    denominator = len(ranks)
    output: Dict[str, float | int | str] = {
        f"hits@{k}": (
            float(sum(rank is not None and rank <= k for rank in ranks) / denominator)
            if denominator
            else 0.0
        )
        for k in ks
    }
    output["mrr"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    output["evaluated_pairs"] = denominator
    output["retrieved_ground_truth_pairs"] = retrieved_truth
    output["candidate_pairs"] = unique_pairs
    output["missing_ground_truth_policy"] = "zero reciprocal rank"
    output["tie_break"] = "descending score then target ID"
    return output


def evaluate_ranking_from_embeddings(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    ground_truth: Mapping[int, int],
    ks: Sequence[int] = (1, 5, 10),
    metric: str = "euclidean",
    batch_size: int = 512,
) -> Dict[str, float | int]:
    source = np.asarray(source_embeddings, dtype=np.float32)
    target = np.asarray(target_embeddings, dtype=np.float32)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("Embeddings must be two-dimensional")
    if source.shape[1] != target.shape[1]:
        raise ValueError("Source and target embedding dimensions differ")

    valid_items = [
        (int(u), int(v))
        for u, v in ground_truth.items()
        if 0 <= int(u) < source.shape[0] and 0 <= int(v) < target.shape[0]
    ]
    ranks: list[int] = []
    for start in range(0, len(valid_items), batch_size):
        batch = valid_items[start : start + batch_size]
        source_rows = [u for u, _ in batch]
        distances = pairwise_distances(source[source_rows], target, metric=metric)
        for row_index, (_, true_target) in enumerate(batch):
            order = np.argsort(distances[row_index], kind="stable")
            location = np.flatnonzero(order == true_target)
            if location.size:
                ranks.append(int(location[0]) + 1)
    return _metrics_from_ranks(ranks, ks)
