from __future__ import annotations

"""Post-inference failure diagnostics for BPUPGA methods.

The functions in this module are evaluation-only.  They receive an already
computed :class:`AlignmentResult` and may use ground truth to perform oracle
interventions, but ground truth is never exposed to a baseline's ``fit`` call.
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, pairwise_distances, roc_auc_score

from common.dataset import AlignmentDataset
from common.result import AlignmentResult

from .metrics import evaluate_matchability, evaluate_pair_set
from .postprocess import full_hungarian_pairs, oracle_cardinality_pairs


Pair = tuple[int, int]
DIAGNOSTIC_SCHEMA_VERSION = 1


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _prf(true_positive: int, predicted: int, ground_truth: int) -> dict[str, float | int]:
    precision = _safe_div(true_positive, predicted)
    recall = _safe_div(true_positive, ground_truth)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": int(true_positive),
        "predicted": int(predicted),
        "ground_truth": int(ground_truth),
    }


def _rank_metrics(ranks: Sequence[int], ks: Sequence[int]) -> dict[str, float | int]:
    if not ranks:
        output: dict[str, float | int] = {f"hits@{k}": 0.0 for k in ks}
        output.update({"mrr": 0.0, "mean_rank": 0.0, "evaluated_pairs": 0})
        return output
    values = np.asarray(ranks, dtype=np.int64)
    output = {f"hits@{k}": float(np.mean(values <= k)) for k in ks}
    output.update(
        {
            "mrr": float(np.mean(1.0 / values)),
            "mean_rank": float(np.mean(values)),
            "evaluated_pairs": int(values.size),
        }
    )
    return output


def _finite_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
        }
    quantiles = np.quantile(array, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "max": float(np.max(array)),
    }


def _comparison_summary(values: Iterable[float], tolerance: float = 1e-12) -> dict[str, object]:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    positive = int(np.sum(finite > tolerance))
    tied = int(np.sum(np.abs(finite) <= tolerance))
    negative = int(np.sum(finite < -tolerance))
    total = int(finite.size)
    return {
        "distribution": _finite_summary(finite),
        "positive_count": positive,
        "tie_count": tied,
        "negative_count": negative,
        "positive_rate": _safe_div(positive, total),
        "tie_rate": _safe_div(tied, total),
        "negative_rate": _safe_div(negative, total),
    }


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_diagnostics_report(path: str | Path, report: Mapping[str, object]) -> Path:
    """Atomically save a diagnostics report."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    temporary.replace(destination)
    return destination


def _pair_record(dataset: AlignmentDataset, pair: Pair) -> dict[str, list[int]]:
    u, v = pair
    return {
        "reindexed": [int(u), int(v)],
        "original": [
            int(dataset.g1_new_to_old[int(u)]),
            int(dataset.g2_new_to_old[int(v)]),
        ],
    }


def _valid_unique_pairs(
    pairs: Iterable[Pair],
    num_source_nodes: int,
    num_target_nodes: int,
) -> tuple[set[Pair], list[Pair]]:
    valid: set[Pair] = set()
    invalid: list[Pair] = []
    for raw_u, raw_v in pairs:
        u, v = int(raw_u), int(raw_v)
        if 0 <= u < num_source_nodes and 0 <= v < num_target_nodes:
            valid.add((u, v))
        else:
            invalid.append((u, v))
    return valid, invalid


def _native_error_decomposition(
    result: AlignmentResult,
    dataset: AlignmentDataset,
    example_limit: int,
) -> dict[str, object]:
    n1 = dataset.graph1.number_of_nodes()
    n2 = dataset.graph2.number_of_nodes()
    truth = {(int(u), int(v)) for u, v in dataset.ground_truth.items()}
    source_overlap = set(dataset.ground_truth)
    target_overlap = set(dataset.ground_truth.values())
    valid_pairs, invalid_pairs = _valid_unique_pairs(result.predicted_pairs, n1, n2)

    categories: dict[str, list[Pair]] = {
        "correct_pair": [],
        "wrong_overlap_pair": [],
        "source_overlap_target_private": [],
        "source_private_target_overlap": [],
        "both_private": [],
    }
    for pair in sorted(valid_pairs):
        u, v = pair
        if pair in truth:
            category = "correct_pair"
        elif u in source_overlap and v in target_overlap:
            category = "wrong_overlap_pair"
        elif u in source_overlap:
            category = "source_overlap_target_private"
        elif v in target_overlap:
            category = "source_private_target_overlap"
        else:
            category = "both_private"
        categories[category].append(pair)

    denominator = len(valid_pairs)
    category_report = {
        name: {
            "count": len(items),
            "rate_among_valid_unique_predictions": _safe_div(len(items), denominator),
            "examples": [_pair_record(dataset, pair) for pair in items[:example_limit]],
        }
        for name, items in categories.items()
    }

    predictions_by_source: dict[int, set[int]] = defaultdict(set)
    for u, v in valid_pairs:
        predictions_by_source[u].add(v)
    source_outcomes = Counter()
    source_examples: dict[str, list[int]] = defaultdict(list)
    for u, true_v in sorted(dataset.ground_truth.items()):
        targets = predictions_by_source.get(int(u), set())
        if int(true_v) in targets:
            outcome = "correct_pair_present"
        elif not targets:
            outcome = "no_prediction"
        else:
            has_overlap = any(v in target_overlap for v in targets)
            has_private = any(v not in target_overlap for v in targets)
            if has_overlap and has_private:
                outcome = "mixed_wrong_targets"
            elif has_overlap:
                outcome = "wrong_overlap_target_only"
            else:
                outcome = "private_target_only"
        source_outcomes[outcome] += 1
        if len(source_examples[outcome]) < example_limit:
            source_examples[outcome].append(int(dataset.g1_new_to_old[int(u)]))

    overlap_filtered = {
        pair for pair in valid_pairs if pair[0] in source_overlap and pair[1] in target_overlap
    }
    source_counts = Counter(int(u) for u, _ in result.predicted_pairs)
    target_counts = Counter(int(v) for _, v in result.predicted_pairs)

    predicted_sources = {u for u, _ in valid_pairs}
    predicted_targets = {v for _, v in valid_pairs}
    full_source_baseline = _prf(len(source_overlap), n1, len(source_overlap))
    full_target_baseline = _prf(len(target_overlap), n2, len(target_overlap))

    return {
        "pair": evaluate_pair_set(valid_pairs, dataset.ground_truth, n1, n2),
        "matchability": evaluate_matchability(valid_pairs, dataset.ground_truth, n1, n2),
        "valid_unique_prediction_count": len(valid_pairs),
        "invalid_raw_prediction_count": len(invalid_pairs),
        "invalid_raw_prediction_examples": [list(pair) for pair in invalid_pairs[:example_limit]],
        "raw_duplicate_source_count": sum(count > 1 for count in source_counts.values()),
        "raw_duplicate_target_count": sum(count > 1 for count in target_counts.values()),
        "pair_categories": category_report,
        "ground_truth_source_outcomes": {
            name: {
                "count": int(count),
                "rate": _safe_div(int(count), len(dataset.ground_truth)),
                "source_original_id_examples": source_examples[name],
            }
            for name, count in sorted(source_outcomes.items())
        },
        "oracle_filtered_native_overlap": {
            "pair": evaluate_pair_set(overlap_filtered, dataset.ground_truth, n1, n2),
            "predicted_overlap_to_overlap_pairs": len(overlap_filtered),
            "warning": (
                "Ground-truth overlap masks were used only after inference; this is a "
                "diagnostic result, not an operational BPUPGA score."
            ),
        },
        "coverage": {
            "source": {
                "covered_nodes": len(predicted_sources),
                "total_nodes": n1,
                "coverage_ratio": _safe_div(len(predicted_sources), n1),
                "observed_unmatched_nodes": n1 - len(predicted_sources),
            },
            "target": {
                "covered_nodes": len(predicted_targets),
                "total_nodes": n2,
                "coverage_ratio": _safe_div(len(predicted_targets), n2),
                "observed_unmatched_nodes": n2 - len(predicted_targets),
            },
        },
        "predict_every_node_matchability_baseline": {
            "source": full_source_baseline,
            "target": full_target_baseline,
            "interpretation": (
                "This is the score obtained by labeling every node as matchable; "
                "it contains no learned matchability information."
            ),
        },
    }


def _quality_matrix(
    result: AlignmentResult,
    max_dense_entries: int,
) -> tuple[np.ndarray | None, str | None, str | None]:
    if result.score_matrix is not None:
        raw = np.asarray(result.score_matrix)
        source = "score_matrix"
        quality = raw
    elif result.cost_matrix is not None:
        raw = np.asarray(result.cost_matrix)
        source = "cost_matrix"
        quality = -raw
    elif result.source_embeddings is not None and result.target_embeddings is not None:
        source_embeddings = np.asarray(result.source_embeddings, dtype=np.float32)
        target_embeddings = np.asarray(result.target_embeddings, dtype=np.float32)
        entries = source_embeddings.shape[0] * target_embeddings.shape[0]
        if entries > max_dense_entries:
            return (
                None,
                "embedding_distance",
                f"Dense embedding distances require {entries} entries, above "
                f"max_dense_entries={max_dense_entries}.",
            )
        if source_embeddings.ndim != 2 or target_embeddings.ndim != 2:
            return None, "embedding_distance", "Embeddings must be two-dimensional."
        if source_embeddings.shape[1] != target_embeddings.shape[1]:
            return None, "embedding_distance", "Source and target embedding dimensions differ."
        distances = pairwise_distances(
            source_embeddings,
            target_embeddings,
            metric="euclidean",
        ).astype(np.float32, copy=False)
        source = "embedding_distance"
        quality = -distances
    else:
        return None, None, "No score matrix, cost matrix, or paired embeddings are available."

    if quality.ndim != 2:
        return None, source, "The score representation is not two-dimensional."
    entries = int(quality.shape[0] * quality.shape[1])
    if entries > max_dense_entries:
        return (
            None,
            source,
            f"Score representation contains {entries} entries, above "
            f"max_dense_entries={max_dense_entries}.",
        )
    non_finite = int(quality.size - np.count_nonzero(np.isfinite(quality)))
    if non_finite:
        return None, source, f"Score representation contains {non_finite} NaN/Inf entries."
    return np.asarray(quality), source, None


def _stable_descending_rank(row: np.ndarray, true_index: int, candidates: np.ndarray) -> int:
    values = row[candidates]
    true_value = row[true_index]
    # This exactly matches a stable descending sort over ascending candidate ids.
    better = int(np.sum(values > true_value))
    tied_before = int(np.sum((values == true_value) & (candidates < true_index)))
    return better + tied_before + 1


def _direction_score_diagnostics(
    quality: np.ndarray,
    ground_truth: Mapping[int, int],
    overlap_candidates: set[int],
    ks: Sequence[int],
    example_limit: int,
) -> dict[str, object]:
    all_candidates = np.arange(quality.shape[1], dtype=np.int64)
    overlap_array = np.asarray(sorted(overlap_candidates), dtype=np.int64)
    private_array = np.asarray(
        sorted(set(range(quality.shape[1])) - overlap_candidates), dtype=np.int64
    )

    all_ranks: list[int] = []
    overlap_ranks: list[int] = []
    correspondence_margins: list[float] = []
    private_margins: list[float] = []
    rank_improvements: list[int] = []
    top1_categories = Counter()
    joint_margin_categories = Counter()
    recoverable_private_intrusions: list[dict[str, object]] = []
    overlap_confusions: list[dict[str, object]] = []

    for raw_query, raw_true in sorted(ground_truth.items()):
        query, true_target = int(raw_query), int(raw_true)
        row = quality[query]
        all_rank = _stable_descending_rank(row, true_target, all_candidates)
        overlap_rank = _stable_descending_rank(row, true_target, overlap_array)
        all_ranks.append(all_rank)
        overlap_ranks.append(overlap_rank)
        rank_improvements.append(all_rank - overlap_rank)

        best_all = int(np.argmax(row))
        if best_all == true_target:
            top1_categories["correct_pair"] += 1
        elif best_all in overlap_candidates:
            top1_categories["wrong_overlap_candidate"] += 1
        else:
            top1_categories["private_candidate"] += 1

        overlap_competitors = overlap_array[overlap_array != true_target]
        corr_margin = math.inf
        if overlap_competitors.size:
            corr_margin = float(row[true_target] - np.max(row[overlap_competitors]))
            correspondence_margins.append(corr_margin)

        private_margin = math.inf
        if private_array.size:
            private_margin = float(row[true_target] - np.max(row[private_array]))
            private_margins.append(private_margin)

        corr_fails = corr_margin < -1e-12
        private_fails = private_margin < -1e-12
        if corr_fails and private_fails:
            joint_margin_categories["both_overlap_and_private_beat_truth"] += 1
        elif corr_fails:
            joint_margin_categories["overlap_only_beats_truth"] += 1
        elif private_fails:
            joint_margin_categories["private_only_beats_truth"] += 1
        else:
            joint_margin_categories["truth_not_strictly_beaten"] += 1

        if overlap_rank == 1 and best_all in set(private_array.tolist()):
            if len(recoverable_private_intrusions) < example_limit:
                recoverable_private_intrusions.append(
                    {
                        "query": query,
                        "true_candidate": true_target,
                        "private_winner": best_all,
                        "true_quality": float(row[true_target]),
                        "private_winner_quality": float(row[best_all]),
                        "all_candidate_rank": all_rank,
                    }
                )
        if overlap_rank > 1 and len(overlap_confusions) < example_limit:
            best_overlap = int(overlap_array[np.argmax(row[overlap_array])])
            overlap_confusions.append(
                {
                    "query": query,
                    "true_candidate": true_target,
                    "wrong_overlap_winner": best_overlap,
                    "true_quality": float(row[true_target]),
                    "winner_quality": float(row[best_overlap]),
                    "overlap_only_rank": overlap_rank,
                }
            )

    evaluated = len(all_ranks)
    recoverable_count = sum(
        1
        for query, true_target in ground_truth.items()
        if _stable_descending_rank(
            quality[int(query)], int(true_target), overlap_array
        )
        == 1
        and int(np.argmax(quality[int(query)])) not in overlap_candidates
    )
    return {
        "all_candidates_ranking": _rank_metrics(all_ranks, ks),
        "overlap_only_ranking": _rank_metrics(overlap_ranks, ks),
        "rank_improvement_after_private_removal": _finite_summary(rank_improvements),
        "top1_candidate_partition": {
            name: {
                "count": int(top1_categories.get(name, 0)),
                "rate": _safe_div(int(top1_categories.get(name, 0)), evaluated),
            }
            for name in ("correct_pair", "wrong_overlap_candidate", "private_candidate")
        },
        "correspondence_margin": {
            **_comparison_summary(correspondence_margins),
            "definition": "quality(true) - best quality(wrong overlap candidate)",
        },
        "private_margin": {
            **_comparison_summary(private_margins),
            "definition": "quality(true) - best quality(private candidate)",
        },
        "joint_margin_failure": {
            name: {
                "count": int(joint_margin_categories.get(name, 0)),
                "rate": _safe_div(int(joint_margin_categories.get(name, 0)), evaluated),
            }
            for name in (
                "truth_not_strictly_beaten",
                "overlap_only_beats_truth",
                "private_only_beats_truth",
                "both_overlap_and_private_beat_truth",
            )
        },
        "recoverable_private_intrusion": {
            "count": recoverable_count,
            "rate_over_ground_truth_pairs": _safe_div(recoverable_count, evaluated),
            "definition": (
                "The true pair ranks first within overlap candidates, but a private "
                "candidate ranks first when all candidates are present."
            ),
            "examples_reindexed": recoverable_private_intrusions,
        },
        "overlap_confusion_examples_reindexed": overlap_confusions,
    }


def _best_threshold_f1(scores: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int64)
    positives = int(np.sum(sorted_labels))
    best: dict[str, object] = {
        "threshold": None,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "selected": 0,
        "ground_truth": positives,
        "rule": "predict matchable when score >= threshold",
    }
    true_positive = 0
    index = 0
    while index < len(sorted_scores):
        end = index + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[index]:
            end += 1
        true_positive += int(np.sum(sorted_labels[index:end]))
        selected = end
        precision = _safe_div(true_positive, selected)
        recall = _safe_div(true_positive, positives)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        if f1 > float(best["f1"]) + 1e-15:
            best = {
                "threshold": float(sorted_scores[index]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "selected": selected,
                "ground_truth": positives,
                "rule": "predict matchable when score >= threshold",
            }
        index = end
    return best


def _binary_signal_report(scores: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=bool)
    report: dict[str, object] = {
        "overlap_score_distribution": _finite_summary(values[truth]),
        "private_score_distribution": _finite_summary(values[~truth]),
        "oracle_best_threshold": _best_threshold_f1(values, truth),
    }
    if np.unique(truth).size < 2:
        report.update(
            {
                "auroc": None,
                "average_precision": None,
                "warning": "AUROC/AUPRC require both overlap and private nodes.",
            }
        )
    else:
        report.update(
            {
                "auroc": float(roc_auc_score(truth, values)),
                "average_precision": float(average_precision_score(truth, values)),
            }
        )
    return report


def _top1_gap(values: np.ndarray, axis: int) -> np.ndarray:
    if values.shape[axis] < 2:
        return np.zeros(values.shape[1 - axis], dtype=np.float64)
    partitioned = np.partition(values, kth=values.shape[axis] - 2, axis=axis)
    if axis == 1:
        return partitioned[:, -1] - partitioned[:, -2]
    return partitioned[-1, :] - partitioned[-2, :]


def _matchability_signal_diagnostics(
    quality: np.ndarray,
    source_overlap: set[int],
    target_overlap: set[int],
) -> dict[str, object]:
    source_labels = np.asarray(
        [node in source_overlap for node in range(quality.shape[0])], dtype=bool
    )
    target_labels = np.asarray(
        [node in target_overlap for node in range(quality.shape[1])], dtype=bool
    )
    return {
        "source": {
            "row_max_quality": _binary_signal_report(np.max(quality, axis=1), source_labels),
            "row_top1_gap": _binary_signal_report(_top1_gap(quality, axis=1), source_labels),
        },
        "target": {
            "column_max_quality": _binary_signal_report(np.max(quality, axis=0), target_labels),
            "column_top1_gap": _binary_signal_report(_top1_gap(quality, axis=0), target_labels),
        },
        "warning": (
            "Oracle thresholds use ground-truth labels after inference and measure "
            "whether a simple confidence signal exists; they are not deployable thresholds."
        ),
    }


def _mapped_hungarian_pairs(
    quality: np.ndarray,
    source_nodes: Sequence[int],
    target_nodes: Sequence[int],
) -> list[Pair]:
    source_array = np.asarray(source_nodes, dtype=np.int64)
    target_array = np.asarray(target_nodes, dtype=np.int64)
    local_pairs = full_hungarian_pairs(
        quality[np.ix_(source_array, target_array)],
        larger_is_better=True,
    )
    return [
        (int(source_array[local_u]), int(target_array[local_v]))
        for local_u, local_v in local_pairs
    ]


def _oracle_entry(
    pairs: Sequence[Pair],
    dataset: AlignmentDataset,
    intervention: str,
) -> dict[str, object]:
    n1 = dataset.graph1.number_of_nodes()
    n2 = dataset.graph2.number_of_nodes()
    return {
        "intervention": intervention,
        "pair": evaluate_pair_set(pairs, dataset.ground_truth, n1, n2),
        "matchability": evaluate_matchability(pairs, dataset.ground_truth, n1, n2),
        "predicted_pairs": len(set(pairs)),
        "warning": "Ground truth was used after inference; diagnostic only.",
    }


def _oracle_decoding_ladder(
    quality: np.ndarray,
    result: AlignmentResult,
    dataset: AlignmentDataset,
    include_oracle_k: bool,
) -> dict[str, object]:
    n1, n2 = quality.shape
    source_overlap = sorted(int(node) for node in dataset.ground_truth)
    target_overlap = sorted(int(node) for node in dataset.ground_truth.values())
    all_sources = list(range(n1))
    all_targets = list(range(n2))

    output: dict[str, object] = {
        "native": _oracle_entry(
            result.predicted_pairs,
            dataset,
            "No oracle intervention; the baseline's original output.",
        ),
        "oracle_source_overlap_mask": _oracle_entry(
            _mapped_hungarian_pairs(quality, source_overlap, all_targets),
            dataset,
            "Keep only true source-overlap rows; retain all target candidates.",
        ),
        "oracle_target_overlap_mask": _oracle_entry(
            _mapped_hungarian_pairs(quality, all_sources, target_overlap),
            dataset,
            "Keep all source candidates; retain only true target-overlap columns.",
        ),
        "oracle_bilateral_overlap_mask": _oracle_entry(
            _mapped_hungarian_pairs(quality, source_overlap, target_overlap),
            dataset,
            "Restrict both sides to their true overlap sets before Hungarian decoding.",
        ),
    }
    if include_oracle_k:
        try:
            pairs = oracle_cardinality_pairs(
                quality,
                cardinality=len(dataset.ground_truth),
                larger_is_better=True,
            )
            output["oracle_cardinality"] = _oracle_entry(
                pairs,
                dataset,
                "Use the true matching cardinality, but do not reveal either overlap set.",
            )
        except (ValueError, RuntimeError) as error:
            output["oracle_cardinality"] = {"skipped": str(error)}
    else:
        output["oracle_cardinality"] = {
            "skipped": (
                "Disabled by default because fixed-cardinality augmented Hungarian "
                "can be substantially more expensive; enable include_oracle_k explicitly."
            )
        }
    return output


def _decoder_loss_diagnostics(
    quality: np.ndarray,
    result: AlignmentResult,
    dataset: AlignmentDataset,
) -> dict[str, object]:
    predicted, _ = _valid_unique_pairs(
        result.predicted_pairs,
        dataset.graph1.number_of_nodes(),
        dataset.graph2.number_of_nodes(),
    )
    target_overlap = np.asarray(sorted(dataset.ground_truth.values()), dtype=np.int64)
    all_top1_correct = 0
    all_top1_lost = 0
    overlap_top1_correct = 0
    overlap_top1_lost = 0
    for u, true_v in dataset.ground_truth.items():
        pair = (int(u), int(true_v))
        row = quality[int(u)]
        if int(np.argmax(row)) == int(true_v):
            all_top1_correct += 1
            if pair not in predicted:
                all_top1_lost += 1
        best_overlap = int(target_overlap[np.argmax(row[target_overlap])])
        if best_overlap == int(true_v):
            overlap_top1_correct += 1
            if pair not in predicted:
                overlap_top1_lost += 1
    total = len(dataset.ground_truth)
    return {
        "all_candidates": {
            "score_top1_correct": all_top1_correct,
            "correct_top1_missing_from_native_output": all_top1_lost,
            "loss_rate_over_all_ground_truth": _safe_div(all_top1_lost, total),
            "loss_rate_conditioned_on_correct_top1": _safe_div(
                all_top1_lost, all_top1_correct
            ),
        },
        "overlap_candidates": {
            "score_top1_correct": overlap_top1_correct,
            "correct_top1_missing_from_native_output": overlap_top1_lost,
            "loss_rate_over_all_ground_truth": _safe_div(overlap_top1_lost, total),
            "loss_rate_conditioned_on_correct_top1": _safe_div(
                overlap_top1_lost, overlap_top1_correct
            ),
        },
    }


def _compact_signature(report: Mapping[str, object]) -> dict[str, float | int | None]:
    matrix = report.get("score_space")
    if not isinstance(matrix, Mapping) or not matrix.get("available"):
        return {
            "native_pair_f1": float(report["native_output"]["pair"]["f1"])
            if report.get("native_output") else None,
            "forward_all_hits@1": None,
            "forward_overlap_hits@1": None,
            "reverse_all_hits@1": None,
            "reverse_overlap_hits@1": None,
            "bilateral_oracle_pair_f1": None,
            "recoverable_private_intrusion_rate_forward": None,
        }
    forward = matrix["directional_correspondence"]["source_to_target"]
    reverse = matrix["directional_correspondence"]["target_to_source"]
    ladder = matrix["oracle_decoding_ladder"]
    native_f1 = ladder["native"]["pair"]["f1"]
    forward_all = float(forward["all_candidates_ranking"]["hits@1"])
    forward_overlap = float(forward["overlap_only_ranking"]["hits@1"])
    reverse_all = float(reverse["all_candidates_ranking"]["hits@1"])
    reverse_overlap = float(reverse["overlap_only_ranking"]["hits@1"])
    bilateral_f1 = float(ladder["oracle_bilateral_overlap_mask"]["pair"]["f1"])
    source_mask_f1 = float(ladder["oracle_source_overlap_mask"]["pair"]["f1"])
    target_mask_f1 = float(ladder["oracle_target_overlap_mask"]["pair"]["f1"])
    return {
        "native_pair_f1": float(native_f1),
        "forward_all_hits@1": forward_all,
        "forward_overlap_hits@1": forward_overlap,
        "forward_private_removal_gain_hits@1": forward_overlap - forward_all,
        "reverse_all_hits@1": reverse_all,
        "reverse_overlap_hits@1": reverse_overlap,
        "reverse_private_removal_gain_hits@1": reverse_overlap - reverse_all,
        "bilateral_oracle_pair_f1": bilateral_f1,
        "bilateral_oracle_gain_over_native_f1": bilateral_f1 - float(native_f1),
        "source_mask_oracle_pair_f1": source_mask_f1,
        "source_mask_gain_over_native_f1": source_mask_f1 - float(native_f1),
        "target_mask_oracle_pair_f1": target_mask_f1,
        "target_mask_gain_over_native_f1": target_mask_f1 - float(native_f1),
        "recoverable_private_intrusion_rate_forward": float(
            forward["recoverable_private_intrusion"]["rate_over_ground_truth_pairs"]
        ),
        "recoverable_private_intrusion_rate_reverse": float(
            reverse["recoverable_private_intrusion"]["rate_over_ground_truth_pairs"]
        ),
        "source_row_max_matchability_auroc": matrix["matchability_signal"]["source"][
            "row_max_quality"
        ]["auroc"],
        "target_column_max_matchability_auroc": matrix["matchability_signal"]["target"][
            "column_max_quality"
        ]["auroc"],
        "source_oracle_threshold_matchability_f1": float(
            matrix["matchability_signal"]["source"]["row_max_quality"]
            ["oracle_best_threshold"]["f1"]
        ),
        "target_oracle_threshold_matchability_f1": float(
            matrix["matchability_signal"]["target"]["column_max_quality"]
            ["oracle_best_threshold"]["f1"]
        ),
        "native_source_matchability_f1": float(
            report["native_output"]["matchability"]["source"]["f1"]
        ),
        "native_target_matchability_f1": float(
            report["native_output"]["matchability"]["target"]["f1"]
        ),
        "predict_all_source_matchability_f1": float(
            report["native_output"]["predict_every_node_matchability_baseline"]
            ["source"]["f1"]
        ),
        "predict_all_target_matchability_f1": float(
            report["native_output"]["predict_every_node_matchability_baseline"]
            ["target"]["f1"]
        ),
        "decoder_loss_rate_given_correct_all_top1": float(
            matrix["decoder_loss"]["all_candidates"]
            ["loss_rate_conditioned_on_correct_top1"]
        ),
        "decoder_loss_rate_given_correct_overlap_top1": float(
            matrix["decoder_loss"]["overlap_candidates"]
            ["loss_rate_conditioned_on_correct_top1"]
        ),
    }


def diagnose_alignment_result(
    result: AlignmentResult,
    dataset: AlignmentDataset,
    *,
    ks: Sequence[int] = (1, 5, 10),
    max_dense_entries: int = 25_000_000,
    include_oracle_k: bool = False,
    example_limit: int = 10,
) -> dict[str, object]:
    """Run common BPUPGA failure diagnostics after baseline inference.

    The report separates native-output errors, correspondence quality within
    the known overlap, private-node intrusion, matchability signal, and decoder
    loss.  Every oracle intervention is explicitly labeled.
    """

    n1 = dataset.graph1.number_of_nodes()
    n2 = dataset.graph2.number_of_nodes()
    result.validate_shapes(n1, n2)
    targets = list(int(v) for v in dataset.ground_truth.values())
    gt_is_injective = len(targets) == len(set(targets))
    if not gt_is_injective:
        raise ValueError(
            "Failure diagnostics require an injective ground-truth alignment; "
            "ordinary Hungarian diagnostics are invalid for multi-valued targets."
        )

    report: dict[str, object] = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "method": result.method,
        "dataset": {
            "name": dataset.name,
            "num_source_nodes": n1,
            "num_target_nodes": n2,
            "ground_truth_pairs": len(dataset.ground_truth),
            "source_overlap_nodes": len(dataset.ground_truth),
            "target_overlap_nodes": len(set(targets)),
            "source_private_nodes": n1 - len(dataset.ground_truth),
            "target_private_nodes": n2 - len(set(targets)),
            "source_overlap_ratio": dataset.source_overlap_ratio,
            "target_overlap_ratio": dataset.target_overlap_ratio,
            "ground_truth_is_function": True,
            "ground_truth_is_injective": gt_is_injective,
        },
        "native_output": _native_error_decomposition(result, dataset, example_limit),
        "leakage_statement": (
            "Ground truth is used only in this post-inference report.  No oracle "
            "mask, cardinality, label, or correspondence is passed to the baseline."
        ),
    }

    quality, matrix_source, skip_reason = _quality_matrix(result, max_dense_entries)
    if quality is None:
        report["score_space"] = {
            "available": False,
            "source": matrix_source,
            "skipped_reason": skip_reason,
        }
        report["compact_signature"] = _compact_signature(report)
        return report

    reverse_ground_truth = {
        int(target): int(source) for source, target in dataset.ground_truth.items()
    }
    source_overlap = set(int(node) for node in dataset.ground_truth)
    target_overlap = set(int(node) for node in dataset.ground_truth.values())
    score_space: dict[str, object] = {
        "available": True,
        "source": matrix_source,
        "shape": [int(quality.shape[0]), int(quality.shape[1])],
        "quality_convention": "larger_is_better",
        "directional_correspondence": {
            "source_to_target": _direction_score_diagnostics(
                quality,
                dataset.ground_truth,
                target_overlap,
                ks,
                example_limit,
            ),
            "target_to_source": _direction_score_diagnostics(
                quality.T,
                reverse_ground_truth,
                source_overlap,
                ks,
                example_limit,
            ),
        },
        "matchability_signal": _matchability_signal_diagnostics(
            quality, source_overlap, target_overlap
        ),
        "decoder_loss": _decoder_loss_diagnostics(quality, result, dataset),
        "oracle_decoding_ladder": _oracle_decoding_ladder(
            quality, result, dataset, include_oracle_k
        ),
    }
    report["score_space"] = score_space
    report["compact_signature"] = _compact_signature(report)
    return report


__all__ = [
    "DIAGNOSTIC_SCHEMA_VERSION",
    "diagnose_alignment_result",
    "save_diagnostics_report",
]
